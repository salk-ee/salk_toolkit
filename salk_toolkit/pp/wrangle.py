"""Data wrangling: transform and aggregate data into the shape a plot needs."""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Sequence

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

PRECISION = 10**6  # Hash-sampling granularity for the raw-format row cap


def _eval_expr(expr: str) -> pl.Expr:
    """A descriptor-supplied row expression - the ``pl_filter`` trust model."""
    return eval(expr, {"pl": pl})  # noqa: S307


def _resolve_weight_col(
    weights: bool | str, declared: str | None, all_col_names: Sequence[str]
) -> tuple[str, pl.Expr | None]:
    """The column to weigh by, per the descriptor's ``weights``.

    ``True`` (the default) = the annotation's declared column, which must exist;
    an annotation declaring none is unweighted. ``False`` = a synthesized unit
    weight. A string referencing ``pl.`` is a per-row expression, any other
    string a column name; both are required.
    """
    if weights is False:
        return "__unit_weight__", None  # never a data column; synthesized as 1.0 by the caller
    if weights is True:
        if declared is None:
            return "row_weights", None  # the conventional name; absent means unweighted
        if declared not in all_col_names:
            raise ValueError(
                f"declared weight column {declared!r} is not in the data (columns: {sorted(all_col_names)[:20]}). "
                "Fix data_meta.weight_col, or pass weights=False for deliberately unweighted numbers."
            )
        return declared, None
    # An expression must reference pl, so a column name that is not a Python
    # identifier ("w.2024") still resolves as a column - and a *missing* one still
    # fails loudly rather than being eval'd.
    if "pl." in weights:
        return "__expr_weight__", _eval_expr(weights)
    if weights not in all_col_names:
        raise ValueError(
            f"weight column {weights!r} is not in the data (columns: {sorted(all_col_names)[:20]}). "
            "Pass weights=False for unweighted numbers, or name a column that exists."
        )
    return weights, None


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
    weight_col, weight_expr = _resolve_weight_col(pp_desc.weights, data_meta.weight_col, all_col_names)
    facet_dims = list(pp_desc.facet_dims)

    # Materialize an expression weight into the plan before the column downselect. Its nulls are
    # left alone: a computed design weight has no reason to sit near 1.0, so filling would distort.
    if weight_expr is not None:
        full_df = full_df.with_columns(weight_expr.cast(pl.Float64).alias(weight_col))
        all_col_names += [weight_col]
    elif weight_col not in all_col_names:  # Unweighted: synthesize a unit weight
        full_df = full_df.with_columns(pl.lit(1.0).alias(weight_col))
        all_col_names += [weight_col]
    else:
        full_df = full_df.with_columns(pl.col(weight_col).fill_null(1.0))

    # res_col is not a facet here; create_plot re-adds it for categorical plots
    if pp_desc.res_col in facet_dims:
        facet_dims.remove(pp_desc.res_col)
    base_cols = list(columns) if columns is not None else []
    # Stat expressions name their columns directly; keep those through the projection
    base_cols += [c for s in pp_desc.stats or [] for c in _eval_expr(s.expr).meta.root_names()]
    extra_cols = base_cols + ([weight_col] + (["draw"] if plot_meta.draws else []))
    cols = [pp_desc.res_col] + facet_dims + list(pp_desc.filter.keys() if pp_desc.filter else [])
    cols += [c for c in extra_cols if c in all_col_names and c not in cols]

    # If any aliases are used, convert them to column names according to the data_meta
    cols = [c for c in np.unique(list_aliases(cols, gc_dict)) if c in all_col_names]

    # Remove draws_data if calculated_draws is disabled
    draws_data = data_meta.draws_data or {}
    if not pp_desc.calculated_draws:
        draws_data = {}

    # Resolve the plot-declared pre-melt row cap early - the id-based sampling below needs the row index
    sample_n = pp_desc.sample
    if sample_n is None and plot_meta.sample:
        sample_n = int((pp_desc.plot_args or {}).get("sample_size", plot_meta.sample))
        if sample_n <= 0:
            raise ValueError("sample_size must be positive")

    # Raw-format row cap, honouring the plot's own full_data opt-out (as `sample` honours sample_size)
    max_rows = None if (pp_desc.plot_args or {}).get("full_data") else plot_meta.max_rows

    # Count and population total both read off the pre-filter frame; the weight-sum scan counts for
    # free. Captured before the row index below, which would cost a full scan instead of metadata.
    # The annotation's declared population total describes the *declared* weighting, so any
    # weights override (unweighted, another column, an expression) recomputes the total instead.
    counting_df, counted_n = full_df, None
    total_weight = data_meta.total_size if pp_desc.weights is True else None
    if total_weight is None:
        if pp_desc.weights is False:
            # The weight is a synthesized 1.0, so the total weight *is* the row count.
            # Summing the literal instead forces the whole pre-filter frame to be
            # produced; pl.len() alone comes off the scan's metadata.
            counted_n = int(counting_df.select(pl.len()).collect().item())
            total_weight = float(counted_n)
        else:
            counts = counting_df.select(pl.len().alias("n"), pl.col(weight_col).sum().alias("w")).collect()
            counted_n, total_weight = int(counts["n"].item()), counts["w"].item()

    def get_total_n() -> int:
        """Pre-filter row count, needed only to build draws - so scan for it lazily and only once."""
        nonlocal counted_n
        if counted_n is None:
            counted_n = int(counting_df.select(pl.len()).collect().item())
        return counted_n

    # The row index blocks predicate pushdown (numbering is pre-filter), so add it only when actually consumed
    need_id = plot_meta.data_format == "raw" or ("draw" in cols and bool(draws_data)) or bool(sample_n)
    if need_id:
        full_df = full_df.with_row_index("id")
        cols += ["id"]

    # Custom dashboard filtering - must run before downselecting to the needed columns
    if pp_desc.pl_filter:
        full_df = full_df.filter(_eval_expr(pp_desc.pl_filter))

    df = full_df.select(cols)  # Select only the columns we need

    res_cols = gc_dict.get(pp_desc.res_col, [pp_desc.res_col])

    # Filter the data with given filters
    if pp_desc.filter:
        filtered_df, cols = _pp_filter_data_lz(df, pp_desc.filter, c_meta, gc_dict)

        # Drop filter-only columns so the unpivot doesn't multiply them by n_questions
        needed = set(res_cols) | set(facet_dims) | set(base_cols) | {weight_col, "draw", "id"}
        keep = [c for c in cols if c in needed]
        if keep != cols:
            filtered_df = filtered_df.select(keep)
            cols = keep
    else:
        filtered_df = df

    # Discretize facet dimensions that are numeric
    for c in facet_dims:
        if c in cols and schema[c].is_numeric():
            current_meta = c_meta.get(c)
            filtered_df, labels = _discretize_continuous(filtered_df, c, current_meta)
            merge_payload = soft_validate(
                {"categories": list(labels), "ordered": True, "continuous": False}, GroupOrColumnMeta
            )
            c_meta[c] = merge_pydantic_models(c_meta.get(c, GroupOrColumnMeta()), merge_payload)

    # Sample from filtered data (sample_n resolved above, before the row index was added)
    if sample_n:
        ids = filtered_df.select("id").collect().get_column("id")
        if len(ids) > sample_n:
            sampled_ids = utils.stable_rng(42).choice(ids.to_numpy(), size=int(sample_n), replace=False)
            filtered_df = filtered_df.filter(pl.col("id").is_in(sampled_ids.tolist()))

    original_question_meta = c_meta.get(pp_desc.res_col, GroupOrColumnMeta()).model_copy(deep=True)
    original_question_colors = original_question_meta.question_colors

    # Convert ordered categorical to continuous if we can
    rcl = [c for c in res_cols if c in cols]
    if pp_desc.convert_res == "categorical" and len(rcl) > 1:
        # Each column would get its own quantiles and its own labels, and only the last one's
        # would survive on the shared axis - an incoherent distribution rather than an error.
        raise ValueError(
            f"convert_res='categorical' needs a single-column res_col; {pp_desc.res_col!r} is a block of "
            f"{len(rcl)} columns, which would be binned separately. Bin one column, or declare the "
            "categories in the annotation."
        )
    for rc in rcl:
        res_meta = c_meta[rc]
        if pp_desc.convert_res == "categorical":
            # Bucket edges come from bin_breaks/bin_labels in the column meta, so a descriptor's
            # col_meta picks them per plot. Gate on dtype: an override may not carry `continuous`.
            if not schema[rc].is_numeric():
                raise ValueError(
                    f"convert_res='categorical' needs a numeric column; {rc!r} is {schema[rc]}. "
                    "Use convert_res='continuous' to number an ordinal first."
                )
            filtered_df, labels = _discretize_continuous(filtered_df, rc, res_meta)
            update_payload: Dict[str, Any] = {
                "continuous": False,
                "categories": list(labels),
                "ordered": True,
                "num_values": None,
                "val_range": None,
            }
            update_model = soft_validate(update_payload, GroupOrColumnMeta)
            c_meta[rc] = merge_pydantic_models(c_meta.get(rc, GroupOrColumnMeta()), update_model)
            c_meta[pp_desc.res_col] = merge_pydantic_models(
                c_meta.get(pp_desc.res_col, GroupOrColumnMeta()), update_model
            )
        elif pp_desc.convert_res == "continuous":
            nvals: Sequence[float | int] = []
            if res_meta.is_categorical:
                res_meta = _ensure_ldf_categories(c_meta, rc, filtered_df)
                nvals = _get_cat_num_vals(res_meta, pp_desc)

                # Conversion only makes sense for ordered (or binary) data
                if len(nvals) > 2 and not res_meta.ordered:
                    raise Exception(
                        f"Cannot convert {rc} to continuous because it has more than 2 values and is not ordered"
                    )

                cmap = dict(zip(res_meta.categories or [], nvals))
                # Categories without a numeric value (e.g. nonresponse) become null, not a failed cast
                converted = pl.col(rc).cast(pl.String).replace_strict(cmap, default=None, return_dtype=pl.Float32)
            elif res_meta.datetime:
                raise Exception(f"Cannot convert datetime column {rc} to continuous")
            elif res_meta.continuous or schema[rc].is_numeric():
                # Already numbers: just parse them (unparseable -> null). Only non-numeric dtypes need the
                # String hop - polars refuses to cast a categorical straight to a number, and on a numeric
                # column the detour stringifies every value for nothing
                col = pl.col(rc) if schema[rc].is_numeric() else pl.col(rc).cast(pl.String)
                converted = col.cast(pl.Float64, strict=False)
            else:
                raise Exception(f"Cannot convert {rc} to continuous: it is neither categorical nor continuous")
            filtered_df = filtered_df.with_columns(converted.fill_nan(None))
            anvals = np.array(nvals, dtype="float")  # To handle null as nan
            val_range = (np.nanmin(anvals), np.nanmax(anvals)) if len(nvals) > 0 else res_meta.val_range
            update_payload: Dict[str, Any] = {
                "continuous": True,
                "categories": None,
                "datetime": False,  # The converted column holds numbers, whatever it was parsed from
                "ordered": False,
                "groups": {},
                "colors": {},
                "num_values": None,
                "likert": False,
                "neutral_middle": None,
                "val_range": val_range,
            }
            update_model = soft_validate(update_payload, GroupOrColumnMeta)
            c_meta[rc] = merge_pydantic_models(c_meta.get(rc, GroupOrColumnMeta()), update_model)
            c_meta[pp_desc.res_col] = merge_pydantic_models(
                c_meta.get(pp_desc.res_col, GroupOrColumnMeta()), update_model
            )

    # Apply continuous transformation - needs to happen when data still in table form
    if c_meta[rcl[0]].continuous:
        val_format = c_meta[rcl[0]].val_format or ".1f"
        val_range = c_meta[rcl[0]].val_range
        # The plot's registered transform_fn is only a default; the descriptor wins when set
        transform_fn = pp_desc.cont_transform or plot_meta.transform_fn
        if transform_fn:
            pp_desc = pp_desc.model_copy(update={"cont_transform": transform_fn})
        if pp_desc.cont_transform:
            filtered_df, val_format, val_range = _transform_cont(
                filtered_df,
                rcl,
                transform=pp_desc.cont_transform,
                val_format=val_format,
                val_range=val_range,
                agg_fn=pp_desc.agg_fn or plot_meta.agg_fn,
            )
    else:
        val_format, val_range = ".1%", None  # Categoricals report %
    val_format = pp_desc.val_format or val_format  # Plot can override the default
    val_range = pp_desc.val_range or val_range

    # A raw plot's row cap can be met before the unpivot: drop whole respondents by hash (no extra
    # scan, pushes into the plan) so the melt lands near the cap instead of n_rows x n_questions
    if max_rows and len(rcl) > 1 and "id" in cols:
        est = filtered_df.select(pl.len()).collect().item() * len(rcl)
        if est > max_rows:
            share = int(max_rows / est * PRECISION)
            filtered_df = filtered_df.filter(pl.col("id").hash(seed=42) % PRECISION < share)

    # Compute draws if needed - Nb: also applies if the draws are shared for the group of questions
    if "draw" in cols and pp_desc.res_col in draws_data:
        uid, ndraws = draws_data[pp_desc.res_col]
        total_n = get_total_n()
        draws = utils.stable_draws(total_n, ndraws, uid)
        draw_df = pl.DataFrame({"draw": draws, "id": np.arange(0, total_n)})
        filtered_df = filtered_df.drop("draw").join(draw_df.lazy(), on=["id"], how="left")

    # If res_col is a group of questions, melt i.e. unpivot the questions and handle draws if needed
    if pp_desc.res_col in gc_dict:
        value_vars = [c for c in gc_dict[pp_desc.res_col] if c in cols]
        n_questions = len(value_vars)  # Only cols that exist in the data
        id_vars = [c for c in cols if (c not in value_vars or c in facet_dims)]
        prefix = original_question_meta.col_prefix or ""
        categories = [v.removeprefix(prefix) for v in value_vars]
        c_meta["question"] = _question_meta_clone(original_question_meta, categories, original_question_colors)

        draw_dfs: List[pl.DataFrame] = []
        if "draw" in cols and draws_data:
            ddf_cache = {}
            for c in value_vars:
                if c in draws_data:
                    uid, ndraws = draws_data[c]
                    if (uid, ndraws) not in ddf_cache:
                        total_n = get_total_n()
                        draws = utils.stable_draws(total_n, ndraws, uid)
                        ddf_cache[(uid, ndraws)] = pl.DataFrame(
                            {"draw": draws, "question": c, "id": np.arange(0, total_n)}
                        )
                    ddf = ddf_cache[(uid, ndraws)]
                    draw_dfs.append(ddf)

            # If all draws are identical (very common), one pre-melt merge beats merging per question
            if len(ddf_cache) == 1 and len(draw_dfs) == len(value_vars):
                filtered_df = filtered_df.drop("draw").join(draw_dfs[0].drop("question").lazy(), on=["id"], how="left")
                draw_dfs = []  # To avoid adding draws again below

        # Wide-aggregate the group when draws are shared or absent (mean/sum only for categorical) - skips the big melt
        fschema = filtered_df.collect_schema()
        agg_fn_resolved = pp_desc.agg_fn or plot_meta.agg_fn or "mean"
        cat_flags = [isinstance(fschema[c], (pl.Categorical, pl.Enum, pl.String)) for c in value_vars]
        if (
            plot_meta.data_format == "longform"
            and "question" in facet_dims
            and not draw_dfs
            and agg_fn_resolved != "posneg_mean"
            and (not any(cat_flags) or (all(cat_flags) and agg_fn_resolved in ("mean", "sum")))
        ):
            wide_value_vars = value_vars
        else:
            wide_value_vars = None

            # Melt i.e. unpivot the questions
            filtered_df = filtered_df.unpivot(
                variable_name="question",
                value_name=pp_desc.res_col,
                index=id_vars,
                on=value_vars,
            )

            # Handle draws for each question
            if len(draw_dfs) > 0:
                filtered_df = (
                    filtered_df.rename({"draw": "old_draw"})
                    .join(pl.concat(draw_dfs).lazy(), on=["id", "question"], how="left")
                    .with_columns(pl.col("draw").fill_null(pl.col("old_draw")))
                    .drop("old_draw")
                )

            # Convert question to categorical with correct order
            filtered_df = filtered_df.with_columns(pl.col("question").cast(pl.Enum(value_vars)))
    else:
        wide_value_vars = None
        n_questions = 1
        if "question" in facet_dims:
            filtered_df = filtered_df.with_columns(pl.lit(pp_desc.res_col).alias("question").cast(pl.Categorical))
            c_meta["question"] = _question_meta_clone(
                original_question_meta, [pp_desc.res_col], original_question_colors
            )

    # Aggregate the data into right shape
    pi = _wrangle_data(
        filtered_df, c_meta, facet_dims, weight_col, pp_desc, n_questions, float(total_weight), wide_value_vars
    )

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
    facet_dims: List[str],
    weight_col: str,
    pp_desc: PlotDescriptor,
    n_questions: int,
    total_size: float = 0.0,
    wide_value_vars: List[str] | None = None,
) -> PlotInput:
    """Aggregate filtered data into a structured ``PlotInput`` model for create_plot.

    If ``wide_value_vars`` is given, ``raw_df`` is still in wide form: the question columns
    are aggregated per group first and only the (small) aggregated frame is unpivoted.
    """

    plot_meta = get_plot_meta(pp_desc.plot)
    assert plot_meta is not None, f"Plot '{pp_desc.plot}' not found in registry"
    schema = raw_df.collect_schema()
    res_col = pp_desc.res_col
    assert res_col is not None, "res_col is required"

    draws = plot_meta.draws
    data_format = plot_meta.data_format

    # Determine the groupby dimensions
    gb_dims = facet_dims + (["draw"] if draws else []) + (["id"] if plot_meta.data_format == "raw" else [])

    # If we have no groupby dimensions, add a dummy one so we don't have to handle the empty case
    if len(gb_dims) == 0:
        raw_df = raw_df.with_columns(pl.lit("dummy").alias("dummy_col"))
        gb_dims = ["dummy_col"]

    value_col = "value"
    cat_col: str | None = None
    scope_df = raw_df  # filtered_size counts the scope, so it is measured before any res_col null filter

    if pp_desc.stats:  # Checked before the dispatch below, which would silently drop stats on a raw plot
        if data_format != "longform":
            raise ValueError(f"stats needs a longform plot; {pp_desc.plot!r} is {data_format!r}. Use plot='columns'.")
        if n_questions > 1:
            raise ValueError(
                f"stats cannot take a block res_col ({res_col!r}, {n_questions} columns): the block aggregation "
                "drops the columns the expressions name. Use a single-column res_col and name the block's "
                "columns in the expressions."
            )
        if clash := [s.name for s in pp_desc.stats if s.name in gb_dims]:
            raise ValueError(f"stat name(s) {clash} collide with the group-by dimensions {gb_dims}; rename them")

    if data_format == "raw":
        value_col = res_col
        data = raw_df.select(gb_dims + [res_col])

    elif pp_desc.stats:
        # Every named statistic is one aggregate in a single group_by, so cells over *different row
        # sets* ride one scan instead of one descriptor per cell. Each expression is materialized as
        # a column first, so the weighted mean's denominator reuses it instead of rebuilding it.
        names = [f"__stat_{i}__" for i in range(len(pp_desc.stats))]
        raw_df = raw_df.with_columns(
            [_eval_expr(s.expr).cast(pl.Float64).alias(n) for n, s in zip(names, pp_desc.stats)]
        )
        aggs = []
        for name, spec in zip(names, pp_desc.stats):
            value, weight = pl.col(name), pl.col(weight_col)
            weighted = (value * weight).sum()
            if spec.agg_fn == "mean":  # Weighted mean over the rows where the expression is non-null
                weighted = weighted / pl.when(value.is_not_null()).then(weight).sum()
            aggs.append(weighted.alias(spec.name))
        data = raw_df.group_by(gb_dims).agg(aggs)
        value_col = pp_desc.stats[0].name

    elif data_format == "longform":
        # Descriptor-level agg_fn overrides the plot's registered default
        agg_fn = pp_desc.agg_fn or plot_meta.agg_fn or "mean"

        if wide_value_vars is not None:  # Question group, still in wide form
            gb = [d for d in gb_dims if d != "question"]
            if not gb:
                raw_df = raw_df.with_columns(pl.lit("dummy").alias("dummy_col"))
                gb = ["dummy_col"]

            if isinstance(schema[wide_value_vars[0]], (pl.Categorical, pl.Enum, pl.String)):
                cat_col = res_col
                value_col = "percent"

                # Per-question group_bys share the filtered scan (comm_subplan_elim), like the melt path would
                parts = [
                    raw_df.group_by(gb + [pl.col(q).cast(pl.Categorical).alias(res_col)])
                    .agg(pl.col(weight_col).sum().alias("percent"))
                    .with_columns(pl.lit(q).alias("question"))
                    for q in wide_value_vars
                ]
                data = pl.concat(parts)
                data = data.with_columns(pl.col("percent").sum().over(gb + ["question"]).alias(weight_col))

            else:  # Continuous: aggregate each question column per group
                value_col = res_col
                if agg_fn in ["mean", "sum"]:  # Use weighted sum to compute both sum and mean
                    aggs = [(pl.col(q) * pl.col(weight_col)).sum().alias(q) for q in wide_value_vars]
                else:  # median, min, max, etc. - ignore weight_col
                    aggs = [getattr(pl.col(q), agg_fn)().alias(q) for q in wide_value_vars]
                # Weight has to be summed per question over that question's non-null rows only, the
                # way the melt path's null filter does - a shared group total biases means by missingness
                wcols = [f"_w_{q}" for q in wide_value_vars]
                wsums = [
                    pl.when(pl.col(q).is_not_null()).then(pl.col(weight_col)).sum().alias(w)
                    for q, w in zip(wide_value_vars, wcols)
                ]
                agg_df = raw_df.group_by(gb).agg(aggs + wsums)
                data = agg_df.unpivot(variable_name="question", value_name=res_col, index=gb, on=wide_value_vars).join(
                    agg_df.unpivot(variable_name="question", value_name=weight_col, index=gb, on=wcols).with_columns(
                        pl.col("question").str.strip_prefix("_w_")
                    ),
                    on=gb + ["question"],
                    how="left",
                )

            # Wide-path guard restricts categorical groups to mean/sum, so this covers both branches
            if agg_fn == "mean":
                data = data.with_columns(pl.col(value_col) / pl.col(weight_col))
            data = data.with_columns(pl.col("question").cast(pl.Enum(wide_value_vars)))
            if gb == ["dummy_col"]:
                data = data.drop("dummy_col")

        elif isinstance(schema[res_col], (pl.Categorical, pl.Enum, pl.String)):  # Categorical
            cat_col = res_col
            value_col = "percent"

            # Group totals as a window sum over the small aggregate - avoids a second full group_by + join
            data = raw_df.group_by(gb_dims + [res_col]).agg(pl.col(weight_col).sum().alias("percent"))
            data = data.with_columns(pl.col("percent").sum().over(gb_dims).alias(weight_col))

            if agg_fn == "mean":
                data = data.with_columns(pl.col("percent") / pl.col(weight_col))
            elif agg_fn == "posneg_mean":
                raise Exception("Use maxdiff plot only on ordinal data")
            elif agg_fn != "sum":
                raise Exception(f"Unknown agg_fn: {agg_fn}")

        else:  # Continuous
            # Null values (e.g. nonresponse converted to continuous) don't count toward any aggregate
            raw_df = raw_df.filter(pl.col(res_col).is_not_null())
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
                    data = data.with_columns(pl.col(res_col) / pl.col(weight_col))
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
                    .select(pl.exclude("reverse_" + weight_col))
                    .rename({"reverse_reverse_" + res_col: "reverse_" + res_col})
                    .with_columns(pl.col("reverse_" + res_col) / pl.col(weight_col))
                    .with_columns(pl.col(res_col) / pl.col(weight_col))
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

    # Both branches share the scope_df subplan, so collect_all (comm_subplan_elim) scans the data only once
    data, fsize = pl.collect_all(
        [data, scope_df.select(pl.col(weight_col).sum())],
        engine="streaming",
    )
    # Raw plots reduce to a fixed number of points anyway, so drop the excess here rather than
    # convert, re-categorize and sort rows the plot is only going to throw away. Only safe on
    # unmelted rows: a melted group was already cut by whole respondents, which plots that
    # re-pivot on id (corr_matrix) depend on
    row_cap = None if (pp_desc.plot_args or {}).get("full_data") else plot_meta.max_rows
    if row_cap and n_questions == 1 and len(data) > row_cap:
        data = data.sample(row_cap, seed=42)
    data = data.to_pandas()
    # In wide form each row covers all questions at once, so no division is needed
    filtered_size = fsize.item() / (1 if wide_value_vars is not None else n_questions)

    # Ensure derived columns have placeholder metadata so later lookups succeed
    for key in [value_col, cat_col]:
        if key and key not in col_meta:
            col_meta[key] = GroupOrColumnMeta()

    # Fix categoricals polars misreads from parquet, and drop unused categories for cleaner plots
    for c in data.columns:
        meta = col_meta.get(c)
        col_dtype = data[c].dtype
        if meta and meta.categories and isinstance(col_dtype, pd.CategoricalDtype):
            uniques = data[c].unique()  # Hoisted: this loop was quadratic when done per category
            present = set(uniques)
            m_cats = meta.categories if meta.categories != "infer" else sorted(list(uniques))
            dtype_cats = utils.get_categories(col_dtype)
            if dtype_cats and len(set(dtype_cats) - set(m_cats)) > 0:
                m_cats = dtype_cats

            # Get the categories that are in use
            if c != pp_desc.res_col or not meta.likert:
                u_cats = [cv for cv in m_cats if cv in present]
            else:
                u_cats = m_cats

            data[c] = pd.Categorical(data[c], u_cats, ordered=meta.ordered)

    # group_by returns hash order; sort (after the categorical fix, so meta order wins) for reproducibility
    sort_cols = [c for c in gb_dims + [res_col] if c in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return PlotInput(
        data=data,
        col_meta=dict(col_meta),  # As this has been adjusted for discretization etc
        value_col=value_col,
        cat_col=cat_col,
        filtered_size=filtered_size,
        total_size=total_size,
    )
