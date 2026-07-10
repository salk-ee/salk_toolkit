"""Plot creation: turn wrangled data into Altair charts."""

from __future__ import annotations

import itertools as it
import json
from copy import copy as shallow_copy
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Tuple, cast

import altair as alt
import pandas as pd
import polars as pl

import salk_toolkit.utils as utils
from salk_toolkit.io import read_parquet_with_metadata
from salk_toolkit.utils import batch, clean_kwargs
from salk_toolkit.validation import ColumnMeta, DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate

from .common import (
    AltairChart,
    FacetMeta,
    PlotInput,
    _get_cat_num_vals,
    _meta_to_plain,
    _normalize_color_dict,
    special_columns,
)
from .matching import _inner_outer_factors, impute_factor_cols, matching_plots
from .meta import _extract_column_meta_cached
from .registry import PlotMeta, _get_plot_fn, _stk_deregister, get_plot_meta, stk_plot
from .wrangle import pp_transform_data


def _get_neutral_cats(cmeta: ColumnMeta) -> List[str]:
    """Return the list of neutral Likert categories defined in metadata."""

    neutrals = list(cmeta.nonordered).copy()
    neutral_middle = cmeta.neutral_middle
    if neutral_middle:
        neutrals.append(neutral_middle)
    return neutrals


def _meta_color_scale(
    cmeta: ColumnMeta,
    column: pd.Series | pd.Categorical | None = None,
    translate: Callable[[str], str] | None = None,
) -> Dict[str, object] | object | alt.UndefinedType:  # type: ignore[return-value]
    """Create a color scale by converting metadata colors (or defaults) into an Altair scale definition."""

    scale, neutrals = _normalize_color_dict(cmeta.colors), _get_neutral_cats(cmeta)
    cats = utils.get_categories(column.dtype) if column is not None and column.dtype.name == "category" else None
    if scale is None and column is not None and column.dtype.name == "category" and utils.get_ordered(column.dtype):
        # Split the values into negative, neutral, positive
        neg, neut, pos = utils.split_to_neg_neutral_pos(cats or [], neutrals)

        # Create a color scale for each category and combine them
        bidir_mid = len(utils.default_bidirectional_gradient) // 2
        reds = utils.gradient_to_discrete_color_scale(
            utils.default_bidirectional_gradient[: bidir_mid + 1],
            len(neg) + 1,
        )
        greys = utils.gradient_to_discrete_color_scale(utils.greyscale_gradient, len(neut) + 2)
        blues = utils.gradient_to_discrete_color_scale(
            utils.default_bidirectional_gradient[bidir_mid:],
            len(pos) + 1,
        )
        scale = {
            **dict(zip(neg, reds[:-1])),
            **dict(zip(neut, greys[1:-1])),
            **dict(zip(pos, blues[1:])),
        }

    if translate and cats is not None:
        remap = dict(zip(cats, [translate(c) for c in cats]))
        scale = {(remap[k] if k in remap else k): v for k, v in scale.items()} if scale else scale
        cats = [remap[c] for c in cats]
    return utils.to_alt_scale(scale, cats)


def _translate_df(df: pd.DataFrame, translate: Callable[[str], str]) -> pd.DataFrame:
    """Translate column names and categorical levels for display."""

    df.columns = [(translate(c) if c not in special_columns and not c.endswith("_label") else c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype.name == "category":
            cats = utils.get_categories(df[c].dtype)
            remap = dict(zip(cats, [translate(c) for c in cats]))
            df[c] = df[c].cat.rename_categories(remap)
    return df


def _create_tooltip(
    pi: PlotInput,
    data: pd.DataFrame,
    c_meta: Mapping[str, GroupOrColumnMeta],
    tfn: Callable[[str], str],
) -> tuple[list[alt.Tooltip], pd.DataFrame]:
    """Build tooltip metadata (labels + formatted values) for plots and return with modified data."""

    # Find labels mappings for regular columns
    label_dict = {}
    for cn in data.columns:
        meta = c_meta.get(cn)
        if meta and meta.labels:
            label_dict[cn] = dict(meta.labels)

    # Find a mapping for multi-column questions
    if "question" in data.columns and "question" in c_meta:
        prefix = c_meta["question"].col_prefix or ""
        qvals = [prefix + c for c in data["question"].unique()]
        q_labels = {}
        for c in qvals:
            meta = c_meta.get(c)
            if meta and meta.label:
                q_labels[c.removeprefix(prefix)] = meta.label
        if q_labels:
            label_dict["question"] = q_labels

    # Determine the columns we need tooltips for:
    tcols = [f.col for f in pi.facets if f.col in data.columns]
    for c in pi.outer_factors:  # Add outer factors with detailed labels
        if c in label_dict and c in data.columns and c not in tcols:
            tcols.append(c)

    # Create the tooltips
    tooltips = [
        alt.Tooltip(
            field=tfn(pi.value_col),
            type="quantitative",
            format=pi.val_format,
        )
    ]
    for cn in tcols:
        if label_dict.get(cn):
            label_col = f"{cn}_label"
            data[label_col] = data[cn].astype("object").replace({k: v for k, v in label_dict[cn].items()})
            t = alt.Tooltip(field=label_col, type="nominal", title=tfn(cn))
        else:
            t = alt.Tooltip(field=tfn(cn), type="nominal")
        tooltips.append(t)

    return tooltips, data


# --------------------------------------------------------
#          PLOTTING PART OF THE PIPELINE
# --------------------------------------------------------


def create_plot(
    pi: PlotInput,
    pp_desc: PlotDescriptor,
    alt_properties: Mapping[str, Any] | None = None,
    alt_wrapper: Callable[[AltairChart], AltairChart] | None = None,
    dry_run: bool = False,
    width: int = 200,
    height: int | None = None,
    return_matrix_of_plots: bool = False,
    translate: Callable[[str], str] | None = None,
    publish_mode: bool = False,
) -> AltairChart | List[List[AltairChart]] | PlotInput:
    """Produce an Altair plot (or matrix of plots) from prepared parameters.

    Handles all of the data wrangling and parameter formatting.
    """

    # Make a shallow copy so we don't mess with the original object. Important for caching
    pi = shallow_copy(pi)
    pi.facets = list(pi.facets or [])
    pi.tooltip = list(pi.tooltip or [])
    pi.outer_factors = list(pi.outer_factors or [])
    alt_properties = dict(alt_properties or {})

    # Get most commonly needed things in usable forms
    data, col_meta = pi.data.copy(), pi.col_meta
    plot_meta = get_plot_meta(pp_desc.plot)
    assert plot_meta is not None, f"Plot '{pp_desc.plot}' not found in registry"

    if "question" in data.columns:
        if "question" not in col_meta:
            col_meta["question"] = GroupOrColumnMeta()

    # `pp_desc.plot_args` are always forwarded to the concrete plot function.
    # PlotInput itself should not be mutated by ad-hoc keys.
    plot_args = {**dict(pi.plot_args), **dict(pp_desc.plot_args or {})}
    pi.plot_args = plot_args

    # Get list of factor columns (adding question and category if needed)
    factor_cols_input = list(pp_desc.factor_cols or [])
    factor_cols, n_inner = _inner_outer_factors(factor_cols_input, pp_desc, plot_meta)

    # Reorder categories if required
    if pp_desc.sort:
        sort_spec = pp_desc.sort if isinstance(pp_desc.sort, dict) else {}
        for cn in pp_desc.sort.keys() if isinstance(pp_desc.sort, dict) else pp_desc.sort:
            ascending = sort_spec.get(cn, False) if isinstance(sort_spec, dict) else False
            if cn not in data.columns or cn == pi.value_col:
                raise Exception(f"Sort column {cn} not found")

            # Some plots (like likert_bars) need a more complex sort
            # This converts the categorical into numeric values and then sorts by the mean of the value
            # If the sort column IS the first facet, the numeric-scale sort is incoherent -
            # fall through to the plain mean-of-value_col sort below.
            if plot_meta.sort_numeric_first_facet and cn != factor_cols[0]:
                f0 = factor_cols[0]
                nvals = _get_cat_num_vals(col_meta[f0], pp_desc)
                cats = col_meta[f0].categories or []
                cmap = dict(zip(cats, nvals))
                sdf = data[[cn, f0, pi.value_col]]
                sdf["sort_val"] = sdf[pi.value_col] * sdf[f0].astype("object").replace(cmap)
                ordervals = sdf.groupby(cn, observed=True)["sort_val"].mean()

            # Otherwise, do not sort ordered categories as categories.
            # Only creates confusion if left on by accident
            elif cn in col_meta and col_meta[cn].ordered:
                continue

            # Otherwise, we are good to sort, simply by value_col
            else:
                ordervals = data.groupby(cn, observed=True)[pi.value_col].mean()
            order = ordervals.sort_values(ascending=ascending).index
            data[cn] = pd.Categorical(data[cn], list(order))
    elif "ordering_value" in data.columns and pp_desc.plot == "maxdiff":
        if pp_desc.factor_cols:
            q = pp_desc.factor_cols[0]
            order = data.groupby(q, observed=True)["ordering_value"].mean().sort_values(ascending=False).index
            data[q] = pd.Categorical(data[q], list(order))

    # Handle internal facets (and translate as needed)
    pi.facets = []

    if n_inner > 0:
        for cn in factor_cols[:n_inner]:
            base_meta = col_meta.get(cn, GroupOrColumnMeta())
            fd = FacetMeta(
                col=cn,
                ocol=cn,
                order=utils.get_categories(data[cn].dtype),
                colors=_meta_color_scale(base_meta, data[cn]),
                neutrals=_get_neutral_cats(base_meta),
                meta=soft_validate(base_meta.model_dump(mode="python"), ColumnMeta),
            )

            pi.facets.append(fd)

        # Pass on data from facet column meta if specified by plot
        for i, d in enumerate(plot_meta.requires):
            for k, v in d.items():
                if v == "pass":
                    facet_meta = col_meta.get(pi.facets[i].ocol)
                    if facet_meta:
                        plot_args[k] = _meta_to_plain(facet_meta).get(k)

        factor_cols = factor_cols[n_inner:]  # Leave rest for external faceting
    pi.outer_factors = factor_cols

    if plot_meta.no_faceting and len(pi.outer_factors) > 0:
        return_matrix_of_plots = True

    pi.value_range = tuple(data[pi.value_col].agg(["min", "max"]))

    pi.outer_colors = (
        _normalize_color_dict(col_meta.get(pi.outer_factors[0], GroupOrColumnMeta()).colors or {}) or {}
        if pi.outer_factors
        else {}
    )

    # Rename res_col if label provided (or remove prefix if present)
    value_meta = col_meta.get(pi.value_col)
    if value_meta and (value_meta.label or value_meta.col_prefix):
        label = value_meta.label
        if not label:
            prefix = value_meta.col_prefix or ""
            label = pi.value_col
            if label.startswith(prefix) and label != prefix:
                label = pi.value_col[len(prefix) :]
        data = data.rename(columns={pi.value_col: label})
        pi.value_col = label

    # Handle translation funcion
    if translate is None:
        translate = lambda s: s

    # Add escaping as Vega Lite goes crazy for symbols like ".[]"
    # It would be enough to do it just for column names, but it's easier to do it for all
    def _tfunc(s: str) -> str:
        return utils.escape_vega_label(translate(s))

    pi.translate = _tfunc

    # Handle tooltips (handles translation inside)
    pi.tooltip, data = _create_tooltip(pi, data, col_meta, _tfunc)

    # Translate everything
    for f in pi.facets:
        f.col = _tfunc(f.col)
        f.order = [_tfunc(c) for c in f.order]
        updated_colors = _meta_color_scale(f.meta, data[f.ocol], translate=_tfunc)
        if updated_colors is not alt.Undefined:
            f.colors = updated_colors
        f.neutrals = [_tfunc(c) for c in f.neutrals]

    data = _translate_df(data, _tfunc)
    pi.value_col = _tfunc(pi.value_col)
    pi.outer_factors = [_tfunc(c) for c in pi.outer_factors]

    # If we still have more than 1 factor left, merge the rest into one so we have a 2d facet
    if len(pi.outer_factors) > 1:
        n_facet_cols = len(utils.get_categories(data[pi.outer_factors[-1]].dtype))
        if not return_matrix_of_plots and len(pi.outer_factors) > 2:
            # Preserve ordering of categories we combine
            cat_lists = [utils.get_categories(data[c].dtype) for c in pi.outer_factors[1:]]
            nf_order = [", ".join(str(x) for x in t) for t in it.product(*cat_lists)]
            factor_col = ", ".join(pi.outer_factors[1:])
            jfs = data[pi.outer_factors[1:]].agg(", ".join, axis=1)
            data.loc[:, factor_col] = pd.Categorical(jfs, nf_order)
            pi.outer_factors = [pi.outer_factors[0], factor_col]

        if len(pi.outer_factors) >= 2:
            pi.outer_factors = list(reversed(pi.outer_factors))
            n_facet_cols = len(utils.get_categories(data[pi.outer_factors[1]].dtype))
    else:
        n_facet_cols = plot_meta.factor_columns or 1

    # Allow value col name to be changed. This can be useful in distinguishing different
    # aggregation options for a column
    if pp_desc.val_name:
        data = data.rename(columns={pi.value_col: pp_desc.val_name})
        pi.value_col = pp_desc.val_name

    # We consistenly used (and mutated) data and now put it back into pi
    pi.data = data

    # Do width/height calculations
    if pi.outer_factors:
        n_facet_cols = pp_desc.n_facet_cols or n_facet_cols  # Allow pp_desc to override col nr
    dims = {"width": width // n_facet_cols if pi.outer_factors else width}

    if height is not None:
        dims["height"] = int(height)
    elif plot_meta.aspect_ratio:
        dims["height"] = int(dims["width"] / plot_meta.aspect_ratio)

    # Make plot properties available to plot function (mostly useful for as_is plots)
    pi.width = width
    pi.alt_properties = alt_properties

    # Create the plot using it's function
    if dry_run:
        return pi

    plot_fn = _get_plot_fn(pp_desc.plot)
    if alt_wrapper is None:
        alt_wrapper = lambda p: p

    if publish_mode:
        old_alt_wrapper = alt_wrapper
        alt_wrapper = lambda p: old_alt_wrapper(_apply_publish_mode(p))

    plot_arg_payload = clean_kwargs(plot_fn, plot_args)

    def _call_plot_fn(
        data_override: pd.DataFrame | None = None,
    ) -> AltairChart:
        payload = pi if data_override is None else pi.model_copy(update={"data": data_override})
        return plot_fn(payload, **plot_arg_payload)

    if plot_meta.as_is:  # if as_is set, just return the plot as-is
        return _call_plot_fn()
    elif pi.outer_factors:
        if return_matrix_of_plots:  # return a 2d list of plots which can be rendeed one plot at a time
            combs = it.product(*[utils.get_categories(data[fc].dtype) for fc in pi.outer_factors])
            return [
                list(batch_item)
                for batch_item in batch(
                    [
                        alt_wrapper(
                            _call_plot_fn(data[(data[pi.outer_factors] == c).all(axis=1)])
                            .properties(title="-".join(map(str, c)), **dims, **alt_properties)
                            .configure_view(discreteHeight={"step": 20})
                        )
                        for c in combs
                    ],
                    n_facet_cols,
                )
            ]
        else:  # Use faceting
            if n_facet_cols == 1:
                plot = alt_wrapper(
                    _call_plot_fn()
                    .properties(**dims, **alt_properties)
                    .facet(
                        row=alt.Row(
                            field=pi.outer_factors[0],
                            type="ordinal",
                            sort=utils.get_categories(data[pi.outer_factors[0]].dtype),
                            header=alt.Header(labelOrient="top"),
                        )
                    )
                )
            elif len(pi.outer_factors) > 1:
                plot = alt_wrapper(
                    _call_plot_fn()
                    .properties(**dims, **alt_properties)
                    .facet(
                        column=alt.Column(
                            field=pi.outer_factors[1],
                            type="ordinal",
                            sort=utils.get_categories(data[pi.outer_factors[1]].dtype),
                        ),
                        row=alt.Row(
                            field=pi.outer_factors[0],
                            type="ordinal",
                            sort=utils.get_categories(data[pi.outer_factors[0]].dtype),
                            header=alt.Header(labelOrient="top"),
                        ),
                    )
                )
            else:  # n_facet_cols!=1 but just one facet
                plot = alt_wrapper(
                    _call_plot_fn()
                    .properties(**dims, **alt_properties)
                    .facet(
                        alt.Facet(
                            field=pi.outer_factors[0],
                            type="ordinal",
                            sort=utils.get_categories(data[pi.outer_factors[0]].dtype),
                        ),
                        columns=n_facet_cols,
                    )
                )
            plot = plot.configure_view(discreteHeight={"step": 20})
    else:
        plot = alt_wrapper(
            _call_plot_fn().properties(**dims, **alt_properties).configure_view(discreteHeight={"step": 20})
        )

        if return_matrix_of_plots:
            plot = [[plot]]

    return plot


# Extra configuration for plots in publish mode
publish_spec = {"config": {"legend": {"labelLimit": 0}, "axis": {"labelLimit": 0}}}


def _apply_publish_mode(plot: AltairChart) -> AltairChart:
    """Apply publish mode to the plot."""
    spec = cast(Dict[str, Any], utils.recursive_dict_merge(plot.to_dict(), publish_spec))
    return type(plot).from_dict(spec)


def e2e_plot(
    pp_desc: Dict[str, Any] | PlotDescriptor,
    data_file: str | None = None,
    full_df: pl.LazyFrame | pd.DataFrame | None = None,
    data_meta: DataMeta | None = None,
    width: int = 800,
    height: int | None = None,
    check_match: bool = True,
    impute: bool = True,
    plot_cache: MutableMapping[str, PlotInput] | None = None,
    return_data: bool = False,
    **kwargs: object,
) -> AltairChart | List[List[AltairChart]] | pd.DataFrame:
    """A convenience function to draw a plot straight from a dataset.

    High-level helper that loads data, transforms, and renders a plot.
    """

    if data_file is None and full_df is None:
        raise Exception("Data must be provided either as data_file or full_df")
    if data_file is None and data_meta is None:
        raise Exception("If data provided as full_df then data_meta must also be given")

    # Validate the plot descriptor and convert to Pydantic object
    pp_desc = soft_validate(pp_desc, PlotDescriptor)

    if full_df is None:
        data_file_str = cast(str, data_file)
        full_df, full_meta = read_parquet_with_metadata(data_file_str, lazy=True)
        if full_meta is None:
            raise ValueError(f"Parquet file {data_file} has no metadata")
        if data_meta is None:
            data_meta = full_meta.data

    if data_meta is None:
        raise ValueError(f"Parquet file {data_file} has no data metadata")

    if impute:
        factor_cols = impute_factor_cols(pp_desc, _extract_column_meta_cached(data_meta), get_plot_meta(pp_desc.plot))
        pp_desc = pp_desc.model_copy(update={"factor_cols": factor_cols})

    if check_match:
        # If we imputed above, the descriptor already has final factor_cols - skip re-imputing
        matches = matching_plots(pp_desc, full_df, data_meta, details=True, list_hidden=True, impute=not impute)
        if isinstance(matches, list) or pp_desc.plot not in matches:
            raise Exception(f"Plot not registered: {pp_desc.plot}")

        matches_dict = cast(dict[str, tuple[int, List[str]]], matches)
        fit, imp = matches_dict[pp_desc.plot]
        if fit < 0:
            raise Exception(f"Plot {pp_desc.plot} not applicable in this situation because of flags {imp}")

    if plot_cache is not None:
        key = json.dumps(pp_desc.model_dump(mode="python"), sort_keys=True)
        if key not in plot_cache:
            plot_cache[key] = pp_transform_data(full_df, data_meta, pp_desc)
        cached = plot_cache[key]
        # Shallow copies protect the cache: create_plot copies pi.data before mutating and
        # col_meta values are replaced (never mutated in place), so no deepcopy is needed
        pi = cached.model_copy(update={"data": cached.data.copy(), "col_meta": dict(cached.col_meta)})
    else:  # No caching
        pi = pp_transform_data(full_df, data_meta, pp_desc)

    if return_data:
        return pi.data
    # dry_run=True can cause create_plot to return Dict[str, Any], but we don't support that here
    assert not kwargs.get("dry_run", False), "dry_run not supported in e2e_plot"
    return create_plot(pi, pp_desc, width=width, height=height, **kwargs)  # type: ignore[return-value]


# Another convenience function to simplify testing new plots


# --------------------------------------------------------
#          TESTING
# --------------------------------------------------------


def test_new_plot(
    fn: Callable[..., AltairChart],
    pp_desc: PlotDescriptor | Dict[str, Any],
    *args: object,
    plot_meta: Mapping[str, Any] | PlotMeta | None = None,
    **kwargs: object,
) -> AltairChart | List[List[AltairChart]] | pd.DataFrame:
    """Temporarily register a plot for interactive testing."""

    # Ensure pp_desc is a PlotDescriptor object
    pp_desc = soft_validate(pp_desc, PlotDescriptor)

    if isinstance(plot_meta, PlotMeta):
        meta_payload = plot_meta.model_dump(mode="python")
    else:
        meta_payload = dict(plot_meta or {})
    meta_payload.pop("name", None)

    stk_plot(**{**meta_payload, "plot_name": "test"})(fn)  # Register the plot under name 'test'
    try:
        pp_desc = pp_desc.model_copy(update={"plot": "test"})
        return e2e_plot(pp_desc, *cast(Tuple[Any, ...], args), **cast(Dict[str, Any], kwargs))
    finally:
        _stk_deregister("test")  # And de-register it again
