"""Data wrangling: transform and aggregate data into the shape a plot needs."""

from __future__ import annotations

import gc
from typing import List, MutableMapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

import salk_toolkit.utils as utils
from salk_toolkit.io import list_aliases
from salk_toolkit.utils import merge_pydantic_models
from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate

from .common import PlotInput, _get_cat_num_vals, _question_meta_clone
from .filters import _discretize_continuous, _ensure_ldf_categories, _pp_filter_data_lz
from .meta import _update_data_meta_with_pp_desc
from .registry import get_plot_meta
from .transforms import _transform_cont


def pp_transform_data(
    full_df: pl.LazyFrame | pd.DataFrame,
    data_meta: DataMeta,
    pp_desc: PlotDescriptor,
    columns: Sequence[str] | None = None,
) -> PlotInput:
    """Get all data required for a given graph.

    Only returns columns and rows that are needed, aggregated to the format plot requires.
    Internally works with polars LazyDataFrame for large data set performance.
    """

    pl.enable_string_cache()  # So we can work on categorical columns

    plot_meta = get_plot_meta(pp_desc.plot)
    assert plot_meta is not None, f"Plot '{pp_desc.plot}' not found in registry"
    c_meta, gc_dict = _update_data_meta_with_pp_desc(data_meta, pp_desc)

    # Setup lazy frame if not already:
    if not isinstance(full_df, pl.LazyFrame):
        full_df = pl.DataFrame(full_df).lazy()

    schema = full_df.collect_schema()
    all_col_names = schema.names()

    # Figure out which columns we actually need
    weight_col = data_meta.weight_col or "row_weights"
    factor_cols = list(pp_desc.factor_cols).copy()

    # Ensure weight column is present (fill with 1.0 if not)
    if weight_col not in all_col_names:
        full_df = full_df.with_columns(pl.lit(1.0).alias(weight_col))
        all_col_names += [weight_col]
    else:
        full_df = full_df.with_columns(pl.col(weight_col).fill_null(1.0))

    # For transforming purposest, res_col is not a factor.
    # It will be made one for categorical plots for plotting part, but for pp_transform_data, remove it
    if pp_desc.res_col in factor_cols:
        factor_cols.remove(pp_desc.res_col)
    base_cols = list(columns) if columns is not None else []
    extra_cols = base_cols + ([weight_col] + (["draw"] if plot_meta.draws else []))
    cols = [pp_desc.res_col] + factor_cols + list(pp_desc.filter.keys() if pp_desc.filter else [])
    cols += [c for c in extra_cols if c in all_col_names and c not in cols]

    # If any aliases are used, cconvert them to column names according to the data_meta
    cols = [c for c in np.unique(list_aliases(cols, gc_dict)) if c in all_col_names]

    # Remove draws_data if calcualted_draws is disabled
    draws_data = data_meta.draws_data or {}
    if not pp_desc.calculated_draws:
        draws_data = {}

    # Add row id-s and find count - both need to happen before filtering
    full_df = full_df.with_row_index("id")
    total_n = full_df.select(pl.len()).collect().item()

    # For more customized filtering in dashboards
    # Has to be done before downselecting to only needed columns
    if pp_desc.pl_filter:
        full_df = full_df.filter(eval(pp_desc.pl_filter, {"pl": pl}))

    cols += ["id"]
    df = full_df.select(cols)  # Select only the columns we need

    # Filter the data with given filters
    if pp_desc.filter:
        filtered_df, cols = _pp_filter_data_lz(df, pp_desc.filter, c_meta, gc_dict)
    else:
        filtered_df = df

    # Discretize factor columns that are numeric
    for c in factor_cols:
        if c in cols and schema[c].is_numeric():
            current_meta = c_meta.get(c)
            filtered_df, labels = _discretize_continuous(filtered_df, c, current_meta)
            merge_payload = soft_validate(
                {"categories": list(labels), "ordered": True, "continuous": False}, GroupOrColumnMeta
            )
            c_meta[c] = merge_pydantic_models(c_meta.get(c, GroupOrColumnMeta()), merge_payload)

    # Sample from filtered data. The sampled ordered-population experiment must
    # cap rows here, before group observations are melted into long form.
    sample_n = pp_desc.sample
    if sample_n is None and pp_desc.plot == "ordered_population_sampled":
        sample_n = int((pp_desc.plot_args or {}).get("sample_size", 1000))
        if sample_n <= 0:
            raise ValueError("sample_size must be positive")
    if sample_n:
        ids = filtered_df.select("id").collect().get_column("id")
        if len(ids) > sample_n:
            sampled_ids = utils.stable_rng(42).choice(ids.to_numpy(), size=int(sample_n), replace=False)
            filtered_df = filtered_df.filter(pl.col("id").is_in(sampled_ids.tolist()))

    original_question_meta = c_meta.get(pp_desc.res_col, GroupOrColumnMeta()).model_copy(deep=True)
    original_question_colors = original_question_meta.question_colors

    # Convert ordered categorical to continuous if we can
    rcl = gc_dict.get(pp_desc.res_col, [pp_desc.res_col])
    rcl = [c for c in rcl if c in cols]
    for rc in rcl:
        res_meta = c_meta[rc]
        if pp_desc.convert_res == "continuous":
            res_meta = _ensure_ldf_categories(c_meta, rc, filtered_df)
            nvals = _get_cat_num_vals(res_meta, pp_desc)

            # Conversion only makes sense for ordered (or binary) data
            if nvals is not None and len(nvals) > 2 and not res_meta.ordered:
                raise Exception(
                    f"Cannot convert {rc} to continuous because it has more than 2 values and is not ordered"
                )

            categories = res_meta.categories or []
            nvals = nvals or []
            cmap = dict(zip(categories, nvals))
            filtered_df = filtered_df.with_columns(
                pl.col(rc).cast(pl.String).replace(cmap).cast(pl.Float32).fill_nan(None)
            )
            nvals = np.array(nvals, dtype="float")  # To handle null as nan
            val_range = (np.nanmin(nvals), np.nanmax(nvals)) if len(nvals) > 0 else (0.0, 1.0)
            update_model = soft_validate(
                {
                    "continuous": True,
                    "categories": None,
                    "ordered": False,
                    "groups": {},
                    "colors": {},
                    "num_values": None,
                    "likert": False,
                    "neutral_middle": None,
                    "val_range": val_range,
                },
                GroupOrColumnMeta,
            )
            c_meta[rc] = merge_pydantic_models(c_meta.get(rc, GroupOrColumnMeta()), update_model)
            c_meta[pp_desc.res_col] = merge_pydantic_models(
                c_meta.get(pp_desc.res_col, GroupOrColumnMeta()), update_model
            )

    # Apply continuous transformation - needs to happen when data still in table form
    if c_meta[rcl[0]].continuous:
        val_format = c_meta[rcl[0]].val_format or ".1f"
        val_range = c_meta[rcl[0]].val_range
        transform_fn = plot_meta.transform_fn
        if transform_fn:
            pp_desc = pp_desc.model_copy(update={"cont_transform": transform_fn})
        if pp_desc.cont_transform:
            filtered_df, val_format, val_range = _transform_cont(
                filtered_df,
                rcl,
                transform=pp_desc.cont_transform,
                val_format=val_format,
                val_range=val_range,
            )
    else:
        val_format, val_range = ".1%", None  # Categoricals report %
    val_format = pp_desc.val_format or val_format  # Plot can override the default
    val_range = pp_desc.val_range or val_range

    # Compute draws if needed - Nb: also applies if the draws are shared for the group of questions
    if "draw" in cols and pp_desc.res_col in draws_data:
        uid, ndraws = draws_data[pp_desc.res_col]
        draws = utils.stable_draws(total_n, ndraws, uid)
        draw_df = pl.DataFrame({"draw": draws, "id": np.arange(0, total_n)})
        filtered_df = filtered_df.drop("draw").join(draw_df.lazy(), on=["id"], how="left")

    # If res_col is a group of questions, melt i.e. unpivot the questions and handle draws if needed
    if pp_desc.res_col in gc_dict:
        value_vars = [c for c in gc_dict[pp_desc.res_col] if c in cols]
        n_questions = len(value_vars)  # Only cols that exist in the data
        id_vars = [c for c in cols if (c not in value_vars or c in factor_cols)]
        prefix = original_question_meta.col_prefix or ""
        categories = [v.removeprefix(prefix) for v in value_vars]
        question_meta = _question_meta_clone(original_question_meta, categories)
        if original_question_colors:
            question_meta.colors = original_question_colors
        c_meta["question"] = question_meta

        if "draw" in cols and draws_data:
            draw_dfs, ddf_cache = [], {}
            for c in value_vars:
                if c in draws_data:
                    uid, ndraws = draws_data[c]
                    if (uid, ndraws) not in ddf_cache:
                        draws = utils.stable_draws(total_n, ndraws, uid)
                        ddf_cache[(uid, ndraws)] = pl.DataFrame(
                            {"draw": draws, "question": c, "id": np.arange(0, total_n)}
                        )
                    ddf = ddf_cache[(uid, ndraws)]
                    draw_dfs.append(ddf)

            # Check if they all have the same draws. If yes (very common), perform a single merge
            # This is a lot more memory efficient than merging one by one post-unpivot
            if len(ddf_cache) == 1 and len(draw_dfs) == len(value_vars):
                filtered_df = filtered_df.drop("draw").join(draw_dfs[0].drop("question").lazy(), on=["id"], how="left")
                draw_dfs = []  # To avoid adding draws again below

        # Melt i.e. unpivot the questions
        filtered_df = filtered_df.unpivot(
            variable_name="question",
            value_name=pp_desc.res_col,
            index=id_vars,
            on=value_vars,
        )

        # Handle draws for each question
        if "draw" in cols and draws_data and len(draw_dfs) > 0:
            filtered_df = (
                filtered_df.rename({"draw": "old_draw"})
                .join(pl.concat(draw_dfs).lazy(), on=["id", "question"], how="left")
                .with_columns(pl.col("draw").fill_null(pl.col("old_draw")))
                .drop("old_draw")
            )

        # Convert question to categorical with correct order
        filtered_df = filtered_df.with_columns(pl.col("question").cast(pl.Enum(value_vars)))
    else:
        n_questions = 1
        if "question" in factor_cols:
            filtered_df = filtered_df.with_columns(pl.lit(pp_desc.res_col).alias("question").cast(pl.Categorical))
            question_meta = _question_meta_clone(original_question_meta, categories=[pp_desc.res_col])
            if original_question_colors:
                question_meta.colors = original_question_colors
            c_meta["question"] = question_meta

    # Aggregate the data into right shape
    pi = _wrangle_data(filtered_df, c_meta, factor_cols, weight_col, pp_desc, n_questions)

    pi.val_format = val_format
    pi.val_range = val_range  # Currently not used

    # Remove prefix from question names in plots
    res_col_meta = c_meta[pp_desc.res_col]
    if res_col_meta.col_prefix and "question" in pi.data.columns:
        prefix = res_col_meta.col_prefix
        question_dtype = pi.data["question"].dtype
        question_categories = utils.get_categories(question_dtype)
        cmap = {c: c.replace(prefix, "") for c in question_categories}
        pi.data["question"] = pi.data["question"].cat.rename_categories(cmap)

    return pi


def _wrangle_data(
    raw_df: pl.LazyFrame,
    col_meta: MutableMapping[str, GroupOrColumnMeta],
    factor_cols: List[str],
    weight_col: str,
    pp_desc: PlotDescriptor,
    n_questions: int,
) -> PlotInput:
    """Aggregate filtered data into a structured ``PlotInput`` model for create_plot."""

    plot_meta = get_plot_meta(pp_desc.plot)
    assert plot_meta is not None, f"Plot '{pp_desc.plot}' not found in registry"
    schema = raw_df.collect_schema()
    res_col = pp_desc.res_col
    assert res_col is not None, "res_col is required"

    draws = plot_meta.draws
    data_format = plot_meta.data_format

    # if pp_desc['res_col'] in factor_cols: factor_cols.remove(pp_desc['res_col']) # Res cannot also be a factor

    # Determine the groupby dimensions
    gb_dims = factor_cols + (["draw"] if draws else []) + (["id"] if plot_meta.data_format == "raw" else [])

    # If we have no groupby dimensions, add a dummy one so we don't have to handle the empty case
    if len(gb_dims) == 0:
        raw_df = raw_df.with_columns(pl.lit("dummy").alias("dummy_col"))
        gb_dims = ["dummy_col"]

    # if draws and 'draw' in schema.names() and 'augment_to' in pp_desc:
    #     # Should we try to bootstrap the data to always have augment_to points. Note this is relatively slow
    #     raw_df = _augment_draws(raw_df,gb_dims[1:],threshold=pp_desc['augment_to'])

    value_col = "value"
    cat_col: str | None = None

    if data_format == "raw":
        value_col = res_col
        if plot_meta.sample:
            selected = raw_df.select(gb_dims + [res_col])
            grouped = getattr(selected, "groupby", lambda *args: selected)(gb_dims)
            sample_method = getattr(grouped, "sample", None)
            if sample_method is not None:
                data = sample_method(n=plot_meta.sample, with_replacement=True)
            else:
                data = grouped
        else:
            data = raw_df.select(gb_dims + [res_col])

    elif data_format == "longform":
        agg_fn = pp_desc.agg_fn or "mean"
        agg_fn = plot_meta.agg_fn or agg_fn
        # Check if categorical by looking at schema
        is_categorical = isinstance(schema[res_col], (pl.Categorical, pl.Enum, pl.String))
        # Check if _highest_lowest_ranked is called before _wrangle_data

        if is_categorical:
            cat_col = res_col
            value_col = "percent"

            # Aggregate the data
            data = raw_df.group_by(gb_dims + [res_col]).agg(pl.col(weight_col).sum().alias("percent"))

            # Add weight_col to the data
            totals = raw_df.group_by(gb_dims).agg(pl.col(weight_col).sum())
            data = data.join(totals, on=gb_dims)

            if agg_fn == "mean":
                data = data.with_columns(pl.col("percent") / pl.col(weight_col))
            elif agg_fn == "posneg_mean":
                raise Exception("Use maxdiff plot only on ordinal data")
            elif agg_fn != "sum":
                raise Exception(f"Unknown agg_fn: {agg_fn}")

        else:  # Continuous
            if agg_fn in [
                "mean",
                "sum",
            ]:  # Use weighted sum to compute both sum and mean
                data = (
                    raw_df.with_columns((pl.col(res_col) * pl.col(weight_col)).alias(res_col))
                    .group_by(gb_dims)
                    .agg(pl.col([res_col, weight_col]).sum())
                )
                if agg_fn == "mean":
                    data = data.with_columns(pl.col(res_col) / pl.col(weight_col).alias(res_col))
            elif agg_fn == "posneg_mean":
                # Needs prefix to avoid name conflict while aggregating
                data = (
                    raw_df.with_columns(((pl.col(res_col) == -1) * pl.col(weight_col)).alias("reverse_" + res_col))
                    .with_columns(((pl.col(res_col) == 1) * pl.col(weight_col)).alias(res_col))
                    .group_by(gb_dims)
                    .agg(
                        pl.col([res_col, weight_col]).sum(),
                        pl.col(["reverse_" + res_col, weight_col]).sum().name.prefix("reverse_"),
                    )
                    .select(pl.exclude("reverse_N"))
                    .rename({"reverse_reverse_" + res_col: "reverse_" + res_col})
                    .with_columns(pl.col("reverse_" + res_col) / pl.col(weight_col).alias("reverse_" + res_col))
                    .with_columns(pl.col(res_col) / pl.col(weight_col).alias(res_col))
                    .with_columns((pl.col(res_col) + pl.col("reverse_" + res_col)).alias("ordering_value"))
                )
            else:  # median, min, max, etc. - ignore weight_col
                data = raw_df.group_by(gb_dims).agg(
                    [
                        getattr(pl.col(res_col), agg_fn)().alias(res_col),
                        pl.col(weight_col).sum(),
                    ]
                )

            value_col = res_col

        if plot_meta.group_sizes:
            data = data.rename({weight_col: "group_size"})
        else:
            data = data.drop(weight_col)
    else:
        raise Exception("Unknown data_format")

    # Remove dummy column after aggregation
    if gb_dims == ["dummy_col"]:
        data = data.drop("dummy_col")

    # Streaming collect keeps memory down on large lazy pipelines; `streaming=` kwarg
    # was renamed to `engine="streaming"` in polars 1.25+.
    data = data.collect(engine="streaming").to_pandas()
    # Force immediate garbage collection
    gc.collect()  # Does not help much, but unlikely to hurt either

    # How many datapoints the plot is based on. This is useful metainfo to display sometimes
    filtered_size = raw_df.select(pl.col(weight_col).sum()).collect().item() / n_questions

    # Ensure derived columns have placeholder metadata so later lookups succeed
    for key in [value_col, cat_col]:
        if key and key not in col_meta:
            col_meta[key] = GroupOrColumnMeta()

    # Fix categorical types that polars does not read properly from parquet
    # Also filter out unused categories so plots are cleaner
    for c in data.columns:
        meta = col_meta.get(c)
        col_dtype = data[c].dtype
        if meta and meta.categories and isinstance(col_dtype, pd.CategoricalDtype):
            m_cats = meta.categories if meta.categories != "infer" else sorted(list(data[c].unique()))
            dtype_cats = utils.get_categories(col_dtype)
            if dtype_cats and len(set(dtype_cats) - set(m_cats)) > 0:
                m_cats = dtype_cats

            # Get the categories that are in use
            if c != pp_desc.res_col or not meta.likert:
                u_cats = [cv for cv in m_cats if cv in data[c].unique()]
            else:
                u_cats = m_cats

            data[c] = pd.Categorical(data[c], u_cats, ordered=meta.ordered)

    return PlotInput(
        data=data,
        col_meta=dict(col_meta),  # As this has been adjusted for discretization etc
        value_col=value_col,
        cat_col=cat_col,
        filtered_size=filtered_size,
    )
