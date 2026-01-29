"""Plot Pipeline
----------------

This is the end-to-end plotting stack that used to live in `02_pp.ipynb`.
It maps annotated survey data to Altair charts by:

- discovering eligible plots via the registry metadata in `salk_toolkit.plots`
- lazily transforming data with Polars, including filters, melts, draws, and
  aggregation helpers such as `pp_transform_data` / `_wrangle_data`
- enriching metadata (`update_data_meta_with_pp_desc`, `impute_factor_cols`,
  `matching_plots`) so dashboards and CLI tools can reason about aliases,
  ordering, draws, and group scales
- building final Altair specs with `create_plot`, colour utilities, tooltips,
  and translation-ready labels
"""

from __future__ import annotations

# These are the only functions that should be exposed to the public
__all__ = ["e2e_plot", "matching_plots", "test_new_plot", "cont_transform_options", "create_plot", "get_plot_fn"]

import gc
import inspect
import itertools as it
import json
from math import ceil
from copy import copy as shallow_copy, deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
    overload,
)

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

from salk_toolkit import utils as utils
from salk_toolkit.validation import DF
from salk_toolkit.io import (
    extract_column_meta,
    group_columns_dict,
    list_aliases,
    read_parquet_with_metadata,
)
from salk_toolkit.utils import batch, clean_kwargs, merge_pydantic_models
from pydantic_extra_types.color import Color

from salk_toolkit.validation import (
    BlockScaleMeta,
    ColumnBlockMeta,
    ColumnMeta,
    DataMeta,
    GroupOrColumnMeta,
    PlotDescriptor,
    PBase,
    soft_validate,
)


def _meta_to_plain(meta: ColumnMeta) -> Dict[str, Any]:
    """Return a plain dict copy of column metadata."""

    return meta.model_dump(mode="python")


def _question_meta_clone(base_meta: GroupOrColumnMeta, categories: Sequence[str] | None = None) -> GroupOrColumnMeta:
    """Produce a categorical copy of ``base_meta`` for the synthetic ``question`` column."""

    clone = base_meta.model_copy(deep=True)
    clone.continuous = False
    if categories is not None:
        clone.categories = list(categories)
    return clone


@dataclass
class FacetMeta:
    """Facet definition consumed by the plotting pipeline."""

    col: str  # Column name used for faceting within the processed dataframe
    ocol: str  # Original column (before translations or label tweaks)
    order: List[str] = field(default_factory=list)  # Ordered categories for the facet column
    colors: object | None = None  # Altair-ready color definition (often `alt.Scale`, `alt.Undefined`, or a dict)
    neutrals: List[str] = field(default_factory=list)  # Likert neutral categories to mute in gradients
    meta: ColumnMeta = field(default_factory=ColumnMeta)  # Full metadata reference for the facet column


@dataclass
class PlotInput:
    """Structured container passed to individual plot functions."""

    data: pd.DataFrame
    col_meta: Dict[str, GroupOrColumnMeta]
    value_col: str
    cat_col: Optional[str] = None
    val_format: str = "%"
    val_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    filtered_size: float = 0.0
    facets: List[FacetMeta] = field(default_factory=list)
    translate: Optional[Callable[[str], str]] = None
    tooltip: List[Any] = field(default_factory=list)
    value_range: Optional[Tuple[float, float]] = None
    outer_colors: Dict[str, Any] = field(default_factory=dict)
    width: int = 800
    alt_properties: Dict[str, Any] = field(default_factory=dict)
    outer_factors: List[str] = field(default_factory=list)
    plot_args: Dict[str, Any] = field(default_factory=dict)

    def model_copy(self, *, deep: bool = False, update: dict[str, Any] | None = None) -> "PlotInput":
        """Backwards-compatible copy helper (mirrors the old Pydantic API used internally)."""

        out = deepcopy(self) if deep else shallow_copy(self)
        if update:
            for k, v in update.items():
                setattr(out, k, v)
        return out


def _normalize_color_dict(scale: Dict[str, Color | str] | None) -> Dict[str, str] | None:
    """Convert Color objects to hex strings so Altair accepts the scale."""

    if not scale:
        return None
    normalized: Dict[str, str] = {}
    for key, value in scale.items():
        if isinstance(value, Color):
            original = value.original()
            normalized[key] = original if isinstance(original, str) else value.as_hex().upper()
        else:
            normalized[key] = value
    return normalized


# Type alias for all Altair chart types that plot functions may return
AltairChart = alt.Chart | alt.LayerChart | alt.FacetChart | alt.VConcatChart | alt.HConcatChart | alt.ConcatChart


# --------------------------------------------------------
#          SHARED UTILITY FUNCTIONS
# --------------------------------------------------------


def _augment_draws(
    data: pd.DataFrame,
    factors: Sequence[str] | None = None,
    n_draws: int | None = None,
    threshold: int = 50,
) -> pd.DataFrame:
    """Augment each draw with bootstrap data from across whole population.

    Ensures at least ``threshold`` samples per bucket.
    """

    if n_draws is None:
        n_draws = data.draw.max() + 1

    assert n_draws is not None, "n_draws must be set"

    if factors:  # Run recursively on each factor separately and concatenate results
        factors_list = list(factors)
        if data[["draw"] + factors_list].value_counts().min() >= threshold:
            return data  # This takes care of large datasets fast
        return (
            data.groupby(factors_list, observed=False)
            .apply(_augment_draws, n_draws=n_draws, threshold=threshold)
            .reset_index(drop=True)
        )  # Slow-ish, but only needed on small data now

    # Get count of values for each draw
    draw_counts = data["draw"].value_counts()  # Get value counts of existing draws
    if len(draw_counts) < n_draws:  # Fill in completely missing draws
        draw_counts = (draw_counts + pd.Series(0, index=range(n_draws))).fillna(0).astype(int)

    # If no new draws needed, just return original
    if draw_counts.min() >= threshold:
        return data

    # Generate an index for new draws
    new_draws = [d for d, c in draw_counts[draw_counts < threshold].items() for _ in range(threshold - c)]

    # Generate new draws
    new_rows = data.iloc[np.random.choice(len(data), len(new_draws)), :].copy()
    new_rows = new_rows.assign(draw=new_draws)

    return pd.concat([data, new_rows])


def _get_cat_num_vals(
    res_meta: GroupOrColumnMeta,
    pp_desc: PlotDescriptor,
) -> Sequence[float | int]:
    """Get the numerical values to map categories to for ordered plots."""

    categories = res_meta.categories
    if not categories:
        return []
    try:
        nvals = [float(x) for x in cast(Sequence[Any], categories)]
    except ValueError:
        # For string categories, create numeric mapping
        nvals = res_meta.num_values
        if nvals is None:
            nvals = list(range(len(categories)))
    num_values = pp_desc.num_values
    if num_values is not None:
        nvals = num_values  # type: ignore[assignment]
    return nvals  # type: ignore[return-value]


# --------------------------------------------------------
#          PLOT REGISTRY FUNCTIONS
# --------------------------------------------------------


class PlotMeta(PBase):
    """Metadata registered for each plot function via ``@stk_plot``."""

    name: str
    data_format: Literal["longform", "raw"] = "longform"
    draws: bool = False
    continuous: bool = False
    n_facets: Optional[Tuple[int, int]] = None
    requires: List[Dict[str, Any]] = DF(list)
    no_question_facet: bool = False
    agg_fn: Optional[str] = None
    sample: Optional[int] = None
    group_sizes: bool = False
    sort_numeric_first_facet: bool = False
    no_faceting: bool = False
    factor_columns: int = 1
    aspect_ratio: Optional[float] = None
    as_is: bool = False
    priority: int = 0
    args: Dict[str, Any] = DF(dict)
    hidden: bool = False
    transform_fn: Optional[str] = None
    nonnegative: bool = False


special_columns: List[str] = [
    "id",
    "weight",
    "draw",
    "original_inds",
    "__index_level_0__",
    "group_size",
    "ordering_value",
]

registry: Dict[str, Callable[..., Any]] = {}
registry_meta: Dict[str, PlotMeta] = {}
_registry_bootstrapped = False

stk_plot_defaults = {"data_format": "longform"}


def _ensure_plot_registry_loaded() -> None:
    """Import the plots module lazily to populate the registry."""
    global _registry_bootstrapped
    if _registry_bootstrapped:
        return
    try:
        import salk_toolkit.plots  # noqa: F401
    except Exception as exc:  # pragma: no cover
        utils.warn(f"Plot registry bootstrap failed: {exc}")
    else:
        _registry_bootstrapped = True


def _ensure_plot_args_sync(func: Callable[..., Any], decorator_kwargs: Dict[str, Any]) -> None:
    """Verify that declared args (including pass-through requirements) match the function signature.

    Used as part of the decorator for registering a plot type with metadata.
    """

    args_dict = decorator_kwargs.get("args")
    declared_args = set(cast(dict[str, object], args_dict if args_dict is not None else {}).keys())
    requires = cast(list[dict[str, object]], decorator_kwargs.get("requires") or [])
    pass_args = {k for req in requires for k, v in cast(dict[str, object], req).items() if v == "pass"}

    expected = declared_args | pass_args
    if not expected:
        return

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    extra_params = [p for p in params[1:] if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    seen = {p.name for p in extra_params}
    if seen != expected:
        raise ValueError(
            f"Plot '{func.__name__}' signature args {sorted(seen)} do not match declared args {sorted(expected)}"
        )


def stk_plot(plot_name: str, **r_kwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register a plotting function inside the global plot registry."""

    def _decorator(gfunc: Callable[..., object]) -> Callable[..., object]:
        _ensure_plot_args_sync(gfunc, r_kwargs)
        registry[plot_name] = gfunc
        meta_payload = {"name": plot_name, **stk_plot_defaults, **r_kwargs}
        registry_meta[plot_name] = PlotMeta.model_validate(meta_payload)

        return gfunc

    return _decorator


def _stk_deregister(plot_name: str) -> None:
    """Remove a plot from the registry (used in tests)."""

    del registry[plot_name]
    del registry_meta[plot_name]


def _get_plot_fn(plot_name: str) -> Callable[..., Any]:
    """Retrieve a registered plot function by name."""

    _ensure_plot_registry_loaded()
    return registry[plot_name]


# External legacy plot builder callable for Streamlit tools.
def get_plot_fn(plot_name: str) -> Callable[..., AltairChart]:
    """Return a legacy plot builder callable for Streamlit tools.

    Historically, `salk_internal_package` tools called `stk` plots directly via:
    `get_plot_fn("matrix")(**pparams)`, where `pparams` was a dict containing
    keys like `data`, `facets`, and `value_col`.

    The plot pipeline was refactored to pass a `PlotInput` object into plot
    functions instead. This helper restores the old calling convention by
    wrapping the registered plot function.

    Args:
        plot_name: Registry name of the plot to retrieve (e.g. "matrix", "density").

    Returns:
        Callable that accepts legacy `pparams` keyword arguments and returns an Altair chart.
    """

    plot_fn = _get_plot_fn(plot_name)
    sig = inspect.signature(plot_fn)
    params = list(sig.parameters.keys())
    first_param = params[0] if params else None
    plot_param_names = {p for p in params if p != first_param}

    def _facet(raw: object) -> FacetMeta:
        if isinstance(raw, FacetMeta):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("Facet definitions must be dicts or FacetMeta instances")
        col = raw.get("col")
        if not isinstance(col, str) or not col:
            raise ValueError("Facet dict must contain non-empty 'col'")
        meta_raw = raw.get("meta")
        return FacetMeta(
            col=col,
            ocol=cast(str, raw.get("ocol", col)),
            order=[str(x) for x in cast(Sequence[Any], raw.get("order", []))],
            colors=raw.get("colors"),
            neutrals=[str(x) for x in cast(Sequence[Any], raw.get("neutrals", []))],
            meta=soft_validate(meta_raw, ColumnMeta) if meta_raw is not None else ColumnMeta(),
        )

    def _legacy_callable(**pparams: object) -> AltairChart:
        # Split PlotInput-ish keys vs plot-function kwargs.
        plot_kwargs = {k: v for k, v in dict(pparams).items() if k in plot_param_names}

        data = pparams.get("data")
        if data is None:
            raise ValueError("Legacy plot call requires 'data'")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)  # type: ignore[arg-type]

        facets_raw = pparams.get("facets", [])
        facets_list = [_facet(f) for f in cast(Sequence[Any], facets_raw)] if facets_raw else []

        value_col = pparams.get("value_col")
        if not isinstance(value_col, str) or not value_col:
            raise ValueError("Legacy plot call requires non-empty 'value_col'")

        # pyright: ignore[reportCallIssue] - legacy dict-based API intentionally coerces loosely typed inputs.
        pi = PlotInput(  # type: ignore[call-arg]
            data=data,
            col_meta=cast(Dict[str, GroupOrColumnMeta], pparams.get("col_meta") or {}),
            value_col=value_col,
            cat_col=cast(Optional[str], pparams.get("cat_col")),
            val_format=cast(str, pparams.get("val_format") or "%"),
            val_range=cast(Optional[Tuple[Optional[float], Optional[float]]], pparams.get("val_range")),
            filtered_size=float(cast(object, pparams.get("filtered_size") or 0.0)),
            facets=facets_list,
            tooltip=cast(List[Any], pparams.get("tooltip") or []),
            value_range=cast(Optional[Tuple[float, float]], pparams.get("value_range")),
            outer_colors=cast(Dict[str, Any], pparams.get("outer_colors") or {}),
            width=int(cast(object, pparams.get("width") or 800)),
            alt_properties=cast(Dict[str, Any], pparams.get("alt_properties") or {}),
            outer_factors=cast(List[str], pparams.get("outer_factors") or []),
            plot_args=cast(Dict[str, Any], pparams.get("plot_args") or {}),
        )

        return cast(AltairChart, plot_fn(pi, **plot_kwargs))

    return _legacy_callable


def get_plot_meta(plot_name: str) -> PlotMeta | None:
    """Return the registry metadata entry for ``plot_name``."""

    _ensure_plot_registry_loaded()
    if plot_name not in registry_meta:
        return None
    return registry_meta[plot_name].model_copy(deep=True)


def _get_all_plots() -> List[str]:
    """List registered plot names in alphabetical order."""

    _ensure_plot_registry_loaded()
    return sorted(list(registry.keys()))


# First is weight if not matching, second if match
# This is very much a placeholder right now
n_a = -1000000
priority_weights = {
    "draws": [n_a, 0],
    "nonnegative": [n_a, 40],
    "hidden": [n_a, 0],
    "ordered": [n_a, 10],
    "likert": [n_a, 200],
    "required_meta": [n_a, 500],
}

# More agressive old version:
# priority_weights = {
#     'draws': [n_a, 50],
#     'nonnegative': [n_a, 50],
#     'hidden': [n_a, 0],

#     'ordered': [n_a, 100],
#     'likert': [n_a, 200],
#     'required_meta': [n_a, 500],
# }

# Method for choosing a sensible default plot based on the data and plot metadata


def _calculate_priority(plot_meta: PlotMeta, match: Mapping[str, Any]) -> tuple[int, List[str]]:
    """Score how well a plot definition matches the requested descriptor."""
    priority, reasons = int(plot_meta.priority or 0), []

    facet_metas = match["facet_metas"]
    if plot_meta.no_question_facet:
        facet_metas = [f for f in facet_metas if f["name"] not in ["question", match["res_col"]]]

    # Plots with raw data assume numerical values so remove them as options
    if match["categorical"] and plot_meta.data_format == "raw":
        return n_a, ["raw_data"]

    # Plots marked as continuous-only should not match categorical data
    if match["categorical"] and plot_meta.continuous:
        return n_a, ["continuous_only"]

    n_min_facets, n_rec_facets = plot_meta.n_facets or (0, 0)
    if len(facet_metas) < n_min_facets:
        return n_a, ["n_facets"]  # Not enough factors
    else:  # Prioritize plots that have the right number of factors
        priority += 10 * abs(len(facet_metas) - n_rec_facets)

    # Check plot requirements
    if plot_meta.draws:
        val = priority_weights["draws"][1 if match.get("draws") else 0]
        if val < 0:
            reasons.append("draws")
        priority += val

    if plot_meta.nonnegative:
        val = priority_weights["nonnegative"][1 if match.get("nonnegative") else 0]
        if val < 0:
            reasons.append("nonnegative")
        priority += val

    if plot_meta.hidden:
        val = priority_weights["hidden"][1 if match.get("hidden") else 0]
        if val < 0:
            reasons.append("hidden")
        priority += val
    for i, d in enumerate(plot_meta.requires):
        md = facet_metas[i]
        for k, v in d.items():
            if v != "pass":
                val = priority_weights[k][1 if md.get(k) == v else 0]
            else:
                val = priority_weights["required_meta"][
                    1 if md.get(k) is not None else 0
                ]  # Use these weights for things plots require from metadata

            if k == "ordered" and md.get("continuous"):
                val = priority_weights[k][1]  # Continuous is turned into ordered categoricals for facets
            if val < 0:
                reasons.append(k)
            priority += val

    return priority, reasons


# --------------------------------------------------------
#          DATA TRANSFORMATIONS
# --------------------------------------------------------


def _update_data_meta_with_pp_desc(
    data_meta: DataMeta,
    pp_desc: PlotDescriptor,
) -> tuple[Dict[str, GroupOrColumnMeta], Dict[str, List[str]]]:
    """Allow pp_desc to modify data meta by merging plot-descriptor overrides into the canonical data metadata."""

    # Ensure data_meta is a DataMeta object (handle legacy dict inputs from tests)
    if not isinstance(data_meta, DataMeta):
        meta_obj = soft_validate(data_meta, DataMeta)
    else:
        meta_obj = data_meta
    desc_obj = pp_desc

    working_meta = meta_obj
    structure = dict(meta_obj.structure or {})

    res_meta_raw = desc_obj.res_meta
    if res_meta_raw:
        if isinstance(res_meta_raw, dict):
            res_meta = soft_validate(res_meta_raw, ColumnBlockMeta)
        else:
            # Already a ColumnBlockMeta object
            res_meta = res_meta_raw

        if res_meta.scale is None and res_meta.columns:
            base_col_name = next(iter(res_meta.columns.keys()), None)
            base_col_meta = extract_column_meta(data_meta).get(base_col_name)
            if base_col_meta is not None:
                scale_payload = base_col_meta.model_dump(mode="python")
                scale_payload["col_prefix"] = ""
                res_meta = res_meta.model_copy(update={"scale": soft_validate(scale_payload, BlockScaleMeta)})

        structure[res_meta.name] = res_meta
        working_meta = working_meta.model_copy(update={"structure": structure})

    col_meta = extract_column_meta(working_meta)
    gc_dict = group_columns_dict(working_meta)

    col_meta_override = desc_obj.col_meta or {}
    if col_meta_override:
        assert isinstance(col_meta_override, dict), "col_meta must be a dict"
        for key, update in col_meta_override.items():
            if key in col_meta:
                # Ensure update is a dict (convert ColumnMeta to dict if needed)
                if isinstance(update, BaseModel):
                    update = update.model_dump(mode="python", exclude_unset=True)
                elif not isinstance(update, dict):
                    update = dict(update)
                col_meta[key] = col_meta[key].model_copy(update=update)

    return col_meta, gc_dict


@overload
def matching_plots(
    pp_desc: PlotDescriptor | Dict[str, Any],
    df: pl.LazyFrame | pd.DataFrame,
    data_meta: DataMeta,
    details: Literal[False] = False,
    list_hidden: bool = ...,
    impute: bool = ...,
) -> List[str]: ...


@overload
def matching_plots(
    pp_desc: PlotDescriptor | Dict[str, Any],
    df: pl.LazyFrame | pd.DataFrame,
    data_meta: DataMeta,
    details: Literal[True],
    list_hidden: bool = ...,
    impute: bool = ...,
) -> Dict[str, tuple[int, List[str]]]: ...


def matching_plots(
    pp_desc: PlotDescriptor | Dict[str, Any],
    df: pl.LazyFrame | pd.DataFrame,
    data_meta: DataMeta,
    details: bool = False,
    list_hidden: bool = False,
    impute: bool = True,
) -> Dict[str, tuple[int, List[str]]] | List[str]:
    """Get a list of plot types matching required spec, sorted by suitability."""

    # This is meant to find suitable plot types, so we forgive plot being missing
    if isinstance(pp_desc, dict) and "plot" not in pp_desc:
        pp_desc["plot"] = "default"

    # Ensure pp_desc is a PlotDescriptor object
    pp_desc = soft_validate(pp_desc, PlotDescriptor)

    if impute:
        factor_cols = impute_factor_cols(pp_desc, extract_column_meta(data_meta), get_plot_meta(pp_desc.plot))
        pp_desc = pp_desc.model_copy(update={"factor_cols": factor_cols})

    col_meta, _ = _update_data_meta_with_pp_desc(data_meta, pp_desc)

    rc = pp_desc.res_col
    rcm = col_meta[rc]

    lazy = isinstance(df, pl.LazyFrame)
    if lazy:
        df_cols = df.collect_schema().names()
    else:
        df_cols = df.columns

    # Determine if values are non-negative
    ocols = list(rcm.columns) if rcm.columns else [rc]
    cols = [c for c in ocols if c in df_cols]
    if not cols:
        raise ValueError(f"Columns {ocols} not found in data")

    if rcm.categories is not None:
        nonneg = True
    else:
        if not lazy:
            min_val = df[cols].min(axis=None)
        else:
            # Casting with strict=False makes this robust even if some columns are non-numeric.
            min_val = df.select(pl.min_horizontal(pl.col(cols).cast(pl.Float64, strict=False).min())).collect().item()
        nonneg = cast(float, min_val) >= 0

    convert_res = pp_desc.convert_res
    if convert_res == "continuous" and (rcm.categories is not None):
        cat_vals_seq = _get_cat_num_vals(rcm, pp_desc)
        cat_vals = [v for v in (cat_vals_seq or []) if v is not None]
        if cat_vals:
            nonneg = min(cat_vals) >= 0

    factor_cols = pp_desc.factor_cols
    facet_metas = []
    for cn in factor_cols:
        meta = col_meta.get(cn, GroupOrColumnMeta())
        facet_metas.append({"name": cn, **meta.model_dump(mode="python")})
    # Determine if data is categorical
    # Data is categorical if it has categories AND is not being converted to continuous
    # AND is not explicitly marked as continuous
    is_categorical = (rcm.categories is not None) and not rcm.continuous and convert_res != "continuous"
    match = {
        "draws": ("draw" in df_cols),
        "nonnegative": nonneg,
        "hidden": list_hidden,
        "res_col": rc,
        "categorical": is_categorical,
        "facet_metas": facet_metas,
    }

    res = [(pn, *_calculate_priority(get_plot_meta(pn), match)) for pn in registry.keys()]
    if details:
        return {n: (p, i) for (n, p, i) in res}  # Return dict with priorities and failure reasons
    else:
        return [
            n for (n, p, i) in sorted(res, key=lambda t: t[1], reverse=True) if p >= 0
        ]  # Return list of possibilities in decreasing order of fit


# Mechanics to allow row-wise numpy transformations here
# They are noticeably slower, so only use them if polars expression is infeasible
custom_row_transforms: Dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {}


def _apply_npf_on_pl_df(
    df: pl.DataFrame,
    cols: Sequence[str],
    npf: Callable[[np.ndarray], np.ndarray],
) -> pl.DataFrame:
    """Apply a NumPy-only transformation to selected columns."""

    df[cols] = npf(df[cols].to_numpy())
    return df


# Polars is annoyingly verbose for these but it is fast enough to be worth it


def _transform_cont(
    data: pl.LazyFrame,
    cols: Sequence[str],
    transform: str | None,
    val_format: str = ".1f",
    val_range: tuple[float, float] | None = None,
) -> tuple[pl.LazyFrame, str, tuple[float, float] | None]:
    """Apply standardized continuous transforms (center, z-score, etc.)."""

    if not transform:
        return data, val_format, val_range
    elif transform == "center":
        return data.with_columns(pl.col(cols) - pl.col(cols).mean()), val_format, None
    elif transform == "zscore":
        return (
            data.with_columns((pl.col(cols) - pl.col(cols).mean()) / pl.col(cols).std(0)),
            ".2f",
            None,
        )
    elif transform == "01range":
        return (
            data.with_columns((pl.col(cols) - pl.col(cols).min()) / (pl.col(cols).max() - pl.col(cols).min())),
            ".2f",
            None,
        )
    elif transform == "proportion":
        return (
            data.with_columns(pl.col(cols) / pl.sum_horizontal(pl.col(cols).abs())),
            ".1%",
            (0.0, 1.0),
        )
    elif transform in ["softmax", "softmax-ratio"]:
        mult, val_format = (
            (len(cols), ".1f") if transform == "softmax-ratio" else (1.0, ".1%")
        )  # Ratio is just a multiplier
        return (
            data.with_columns(pl.col(cols).exp() * mult / pl.sum_horizontal(pl.col(cols).exp())),
            val_format,
            (0.0, 1.0 * mult),
        )
    elif transform in custom_row_transforms:
        _tfunc, fmt = custom_row_transforms[transform]
        data = data.map_batches(
            lambda bdf: _apply_npf_on_pl_df(bdf, cols, _tfunc),
            streamable=True,
            validate_output_schema=False,
        )  # NB! Set validate to true if debugging this
        return data, fmt, None

    else:
        raise Exception(f"Unknown transform '{transform}'")


def _softmax_expected_ranks(p: np.ndarray) -> np.ndarray:
    """Compute expected rank given Plackett-Luce (softmax) log-odds.

    Relies on the fact that sum of probs of pairwise comparisons is average rank.
    """

    # Convert from log-odds to proportions, but reverse probabilities
    p = np.exp(-p)

    # Create a matrix where element [i,j] is p[j]/(p[i] + p[j])
    sum_matrix = p[..., :, None] + p[..., None, :] + 1e-10  # Shape (..., n, n)
    m = p[..., None, :] / sum_matrix

    # Sum over columns
    sums = m.sum(axis=-1)

    # Subtract diagonal term (0.5) and add 1
    expected_ranks = 1 + (sums - 0.5)
    return expected_ranks


custom_row_transforms["softmax-avgrank"] = _softmax_expected_ranks, ".1f"


def _avg_rank(ovs: np.ndarray) -> np.ndarray:
    """Return 1-indexed average ranks for each row (average rank order)."""

    return 1 + np.argsort(np.argsort(ovs, axis=1), axis=1)
    # Rankdata is insanely slow for some reason
    # return sps.rankdata(ovs, axis=1, method='average')


def _highest_ranked(ovs: np.ndarray) -> np.ndarray:
    """Indicator matrix for the maximum value per row."""

    return (ovs == np.max(ovs, axis=1)[:, None]).astype("int")


def _lowest_ranked(ovs: np.ndarray) -> np.ndarray:
    """Indicator matrix for the minimum value per row."""

    return (ovs == np.min(ovs, axis=1)[:, None]).astype("int")


def _highest_lowest_ranked(ovs: np.ndarray) -> np.ndarray:
    """Encode top choice as +1 and bottom choice as -1."""

    return _highest_ranked(ovs) - _lowest_ranked(ovs)


def _topk_ranked(ovs: np.ndarray, k: int = 3) -> np.ndarray:
    """Return a mask marking the top-k ranked options per row."""

    return np.argsort(np.argsort(ovs, axis=1), axis=1) >= ovs.shape[1] - k


def _win_against_random_field(ovs: np.ndarray, opponents: int = 12) -> np.ndarray:
    """Estimate win probability vs. a random opponent pool."""

    p = np.argsort(np.argsort(ovs, axis=1), axis=1) / ovs.shape[1]
    return np.power(p, opponents)


custom_row_transforms["ordered-avgrank"] = _avg_rank, ".1f"
custom_row_transforms["ordered-warf"] = _win_against_random_field, ".1%"
custom_row_transforms["ordered-top1"] = _highest_ranked, ".1%"
custom_row_transforms["ordered-bot1"] = _lowest_ranked, ".1%"
custom_row_transforms["ordered-topbot1"] = _highest_lowest_ranked, ".1%"
custom_row_transforms["ordered-top2"] = lambda ovs: _topk_ranked(ovs, 2), ".1%"
custom_row_transforms["ordered-top3"] = lambda ovs: _topk_ranked(ovs, 3), ".1%"

cont_transform_options = [
    "center",
    "zscore",
    "01range",
    "proportion",
    "softmax",
    "softmax-ratio",
] + list(custom_row_transforms.keys())


def _ensure_ldf_categories(
    col_meta: MutableMapping[str, GroupOrColumnMeta],
    col: str,
    ldf: pl.LazyFrame,
) -> GroupOrColumnMeta:
    """Get categories from a lazy frame, resolving (and caching) category ordering for ``col`` inside ``col_meta``."""

    meta = col_meta[col]
    cats = meta.categories
    if cats == "infer":
        # Cast to ndarray explicitly for np.sort overload matching
        values = ldf.select(pl.col(col).unique()).collect().to_pandas()[col].values
        resolved = np.sort(np.asarray(values))  # type: ignore[call-overload]
        meta = meta.model_copy(update={"categories": list(resolved)})
        col_meta[col] = meta
    return meta


def _pp_filter_data_lz(
    df: pl.LazyFrame,
    filter_dict: Mapping[str, Any],
    c_meta: MutableMapping[str, GroupOrColumnMeta],
    gc_dict: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pl.LazyFrame, List[str]]:
    """Apply pp-style filters on a Polars ``LazyFrame``."""

    gc_dict = dict(gc_dict or {})
    schema = df.collect_schema()
    colnames = schema.names()

    inds = True

    for k, v in filter_dict.items():
        # Filter on question i.e. filter a subset of res_cols
        if k in gc_dict:
            # Map short names to full column names w prefix
            cnmap = {}
            for c in colnames:
                meta = c_meta.get(c)
                prefix = meta.col_prefix if meta and meta.col_prefix else ""
                short = c.removeprefix(prefix) if prefix else c
                cnmap[short] = c
            # Find columns to remove
            remove_cols = set(gc_dict[k]) - {cnmap[c] for c in v}

            # Remove from both the list and the selection in df
            colnames = [c for c in colnames if c not in remove_cols]
            df = df.select(colnames)

            continue

        # Range filters have form [None,start,end]
        is_range = isinstance(v, (list, tuple)) and v[0] is None and len(v) == 3

        # Handle continuous variables separately
        if is_range and (
            not isinstance(v[1], str)
            or c_meta.get(k, GroupOrColumnMeta()).continuous
            or c_meta.get(k, GroupOrColumnMeta()).datetime
        ):  # Only special case where we actually need a range
            if v[1] is not None:
                inds = (pl.col(k) >= v[1]) & inds
            if v[2] is not None:
                inds = (pl.col(k) <= v[2]) & inds
            # NB! this approach does not work for ordered categoricals with polars LazyDataFrame,
            # hence handling that separately below
            continue

        # Handle categoricals
        if is_range:  # Range of values over ordered categorical
            cats = _ensure_ldf_categories(c_meta, k, df).categories or []
            if set(v[1:]) & set(cats) != set(v[1:]):
                utils.warn(f"Column {k} values {v} not found in {cats}, not filtering")
                flst = cats
            else:
                bi, ei = cats.index(v[1]), cats.index(v[2])
                flst = cats[bi : ei + 1]  #
        elif isinstance(v, (list, tuple)):
            flst = list(v)  # Iterable indicates a set of values
        else:
            groups = c_meta.get(k, GroupOrColumnMeta()).groups
            if groups and v in groups:
                flst = groups[v]
            else:
                flst = [v]  # Just filter on single value

        col_expr = pl.col(k)
        dtype = schema.get(k) if k in schema else None
        if dtype == pl.Categorical or dtype == pl.Enum:
            col_expr = col_expr.cast(pl.Utf8)
        values = flst if isinstance(flst, (list, tuple)) else [flst]
        value_expr = None
        for val in values:
            cond = col_expr == val
            value_expr = cond if value_expr is None else (value_expr | cond)
        if value_expr is None:
            continue
        inds &= value_expr & ~col_expr.is_null()

    filtered_df = df.filter(inds)

    return filtered_df, colnames


def _pp_filter_data(
    df: pd.DataFrame,
    filter_dict: Mapping[str, Any],
    c_meta: MutableMapping[str, GroupOrColumnMeta],
    gc_dict: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Wrapper that allows the filter to work on pandas DataFrames."""

    ldf, _ = _pp_filter_data_lz(pl.DataFrame(df).lazy(), filter_dict, c_meta, gc_dict)
    return ldf.collect().to_pandas()


def _pl_quantiles(ldf: pl.LazyFrame, cname: str, qs: Sequence[float]) -> np.ndarray:
    """Efficient way to calculate multiple quantiles at once with polars in one pass."""

    return ldf.select([pl.col(cname).quantile(q).alias(str(q)) for q in qs]).collect().to_numpy()[0]


# While polars-ized, it is still slow because of the collects.
# This can likely be improved by batching all of the required collects into a single select (over all columns)
# In practice, this is probably not worth it because this is not used very often


def _discretize_continuous(
    ldf: pl.LazyFrame,
    col: str,
    col_meta: GroupOrColumnMeta | None = None,
) -> tuple[pl.LazyFrame, Sequence[str]]:
    """Bucket a continuous column into categorical bins with nice labels."""

    breaks = col_meta.bin_breaks if col_meta and col_meta.bin_breaks is not None else 5
    labels = list(col_meta.bin_labels) if col_meta and col_meta.bin_labels is not None else None
    fmt = col_meta.val_format or ".1f" if col_meta else ".1f"
    schema = ldf.collect_schema()
    isint = schema[col].is_integer()

    if isinstance(breaks, int):  # Quantiles
        if isint:
            ldf = ldf.with_columns(
                pl.col(col).map_batches(
                    lambda x: x + np.random.uniform(-0.5, 0.5, len(x)),
                    is_elementwise=True,
                )
            )
        bpoints = np.linspace(0, 1, breaks + 1)
        breaks = list(_pl_quantiles(ldf, col, bpoints))
        span = breaks[-1] - breaks[0]
        if isint and ceil(span) < len(breaks) - 1:  # Fewer categories than breaks - just show the categories
            breaks = list(np.linspace(breaks[0], breaks[-1], int(ceil(span)) + 1))
            if not labels:
                labels = [f"{round(b + 0.5)}" for b in breaks[:-1]]
        elif not labels:
            labels = [f"{bpoints[i]:.0%} - {bpoints[i + 1]:.0%}" for i in range(len(bpoints) - 1)]
            labels[0], labels[-1] = f"Bottom {bpoints[1]:.0%}", f"Top {bpoints[1]:.0%}"
    else:  # Given breaks
        mi, ma = tuple(_pl_quantiles(ldf, col, [0, 1]))  # Determine range of values
        breaks, labs = utils.cut_nice_labels(breaks, mi, ma, isint, fmt)  # Adds mi/ma to breaks if not in range
        if labels is None:
            labels = labs

    ldf = ldf.with_columns(pl.col(col).cut(breaks[1:-1], labels=labels, left_closed=True).cast(pl.Categorical))

    return ldf, labels


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

    # Sample from filtered data
    if pp_desc.sample:
        sample_method = getattr(filtered_df, "sample", None)
        if sample_method is not None:
            filtered_df = sample_method(n=pp_desc.sample, with_replacement=True)

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

    # For old streaming, the query does not generally seem to stream
    # For new_stream, polars 1.23 considers categoricals to still be broken
    # TODO: Check back here when they fix unpivot in streaming!
    # print("final\n",data.explain(streaming=True))
    # data = data.collect(engine='streaming').to_pandas() # New streaming - does not stream unpivot and is slow
    # Polars collect with streaming parameter - type stubs may not include this
    data = data.collect(streaming=True).to_pandas()  # type: ignore[call-overload]
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
        neg, neut, pos = utils.split_to_neg_neutral_pos(cats, neutrals)

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

    label_dict = {}

    # Determine the columns we need tooltips for:
    tcols = [f.col for f in pi.facets if f.col in data.columns]

    # Find labels mappings for regular columns
    for cn in tcols:
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


def _remove_from_internal_fcols(cname: str, factor_cols: List[str], n_inner: int) -> int:
    """Shift ``cname`` out of the inner facet slice while preserving order."""

    if cname not in factor_cols[:n_inner]:
        return n_inner
    factor_cols.remove(cname)
    if n_inner > len(factor_cols):
        n_inner -= 1
    factor_cols.insert(n_inner, cname)
    return n_inner


def _inner_outer_factors(
    factor_cols: List[str],
    pp_desc: PlotDescriptor,
    plot_meta: PlotMeta,
) -> tuple[List[str], int]:
    """Return `(factor_cols, n_inner)` after respecting descriptor overrides."""

    # Determine how many factors to use as inner facets
    in_f = pp_desc.internal_facet if pp_desc.internal_facet is not None else False
    res_col = pp_desc.res_col
    n_min_f, n_rec_f = plot_meta.n_facets or (0, 0)
    n_inner: int = (n_rec_f if in_f else n_min_f) if isinstance(in_f, bool) else int(in_f)  # type: ignore[arg-type]
    if n_inner > len(factor_cols):
        n_inner = len(factor_cols)

    # If question facet as inner facet for a no_question_facet plot, just move it out
    if plot_meta.no_question_facet:
        n_inner = _remove_from_internal_fcols("question", factor_cols, n_inner)
        n_inner = _remove_from_internal_fcols(res_col, factor_cols, n_inner)

    return factor_cols, n_inner


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
            if plot_meta.sort_numeric_first_facet:
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

    if plot_meta.no_faceting and len(factor_cols) > 0:
        return_matrix_of_plots = True

    pi.value_range = tuple(data[pi.value_col].agg(["min", "max"]))

    pi.outer_colors = (
        _normalize_color_dict(col_meta.get(factor_cols[0], GroupOrColumnMeta()).colors or {}) if factor_cols else {}
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
    factor_cols = [_tfunc(c) for c in factor_cols]

    # If we still have more than 1 factor left, merge the rest into one so we have a 2d facet
    if len(factor_cols) > 1:
        n_facet_cols = len(utils.get_categories(data[factor_cols[-1]].dtype))
        if not return_matrix_of_plots and len(factor_cols) > 2:
            # Preserve ordering of categories we combine
            cat_lists = [utils.get_categories(data[c].dtype) for c in factor_cols[1:]]
            nf_order = [", ".join(str(x) for x in t) for t in it.product(*cat_lists)]
            factor_col = ", ".join(factor_cols[1:])
            jfs = data[factor_cols[1:]].agg(", ".join, axis=1)
            data.loc[:, factor_col] = pd.Categorical(jfs, nf_order)
            factor_cols = [factor_cols[0], factor_col]

        if len(factor_cols) >= 2:
            factor_cols = list(reversed(factor_cols))
            n_facet_cols = len(utils.get_categories(data[factor_cols[1]].dtype))
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
    if factor_cols:
        n_facet_cols = pp_desc.n_facet_cols or n_facet_cols  # Allow pp_desc to override col nr
    dims = {"width": width // n_facet_cols if factor_cols else width}

    if height is not None:
        dims["height"] = int(height)
    elif plot_meta.aspect_ratio:
        dims["height"] = int(dims["width"] / plot_meta.aspect_ratio)

    # Make plot properties available to plot function (mostly useful for as_is plots)
    pi.width = width
    pi.alt_properties = alt_properties
    pi.outer_factors = factor_cols

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
    ) -> AltairChart | List[List[AltairChart]] | PlotInput:
        payload = pi if data_override is None else deepcopy(pi)
        if data_override is not None:
            payload.data = data_override
        return plot_fn(payload, **plot_arg_payload)

    if plot_meta.as_is:  # if as_is set, just return the plot as-is
        return _call_plot_fn()
    elif factor_cols:
        if return_matrix_of_plots:  # return a 2d list of plots which can be rendeed one plot at a time
            combs = it.product(*[utils.get_categories(data[fc].dtype) for fc in factor_cols])
            return [
                list(batch_item)
                for batch_item in batch(
                    [
                        alt_wrapper(
                            _call_plot_fn(data[(data[factor_cols] == c).all(axis=1)])
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
                            field=factor_cols[0],
                            type="ordinal",
                            sort=utils.get_categories(data[factor_cols[0]].dtype),
                            header=alt.Header(labelOrient="top"),
                        )
                    )
                )
            elif len(factor_cols) > 1:
                plot = alt_wrapper(
                    _call_plot_fn()
                    .properties(**dims, **alt_properties)
                    .facet(
                        column=alt.Column(
                            field=factor_cols[1],
                            type="ordinal",
                            sort=utils.get_categories(data[factor_cols[1]].dtype),
                        ),
                        row=alt.Row(
                            field=factor_cols[0],
                            type="ordinal",
                            sort=utils.get_categories(data[factor_cols[0]].dtype),
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
                            field=factor_cols[0],
                            type="ordinal",
                            sort=utils.get_categories(data[factor_cols[0]].dtype),
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
    spec = utils.recursive_dict_merge(plot.to_dict(), publish_spec)
    return type(plot).from_dict(spec)


def impute_factor_cols(
    pp_desc: PlotDescriptor | Dict[str, Any],
    col_meta: Mapping[str, GroupOrColumnMeta],
    plot_meta: PlotMeta | None = None,
) -> List[str]:
    """Compute the full factor_cols list, including question and res_col as needed.

    Ensures descriptor has sensible defaults for `factor_cols`.
    """

    if isinstance(pp_desc, dict) and "plot" not in pp_desc:
        pp_desc["plot"] = "default"

    # Ensure pp_desc is a PlotDescriptor object
    pp_desc = soft_validate(pp_desc, PlotDescriptor)

    factor_cols = list(pp_desc.factor_cols or [])

    # Determine if res is categorical
    res_col = pp_desc.res_col
    convert_res = pp_desc.convert_res
    res_col_meta = col_meta[res_col]
    has_categories = res_col_meta.categories is not None
    has_q = res_col_meta.columns is not None
    cat_res = has_categories and convert_res != "continuous"

    # Add res_col if we are working with a categorical input (and not converting it to continuous)
    if cat_res and res_col not in factor_cols:
        factor_cols.insert(0, res_col)
    if len(factor_cols) < 1 and not has_q:
        # Create 'question' as a dummy dimension so we have at least one factor
        # (generally required for plotting)
        has_q = True

    # If we need to, add question as a factor to list
    if has_q and "question" not in factor_cols:
        if cat_res:
            factor_cols.append("question")  # Put it last for categorical values
        else:
            factor_cols.insert(
                0, "question"
            )  # And first for continuous values, as it then often represents the "category"

    # Pass the factor_cols through the same changes done inside plot pipeline to make more explicit what happens
    if plot_meta:
        factor_cols, _ = _inner_outer_factors(factor_cols, pp_desc, plot_meta)

    return factor_cols


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
        factor_cols = impute_factor_cols(pp_desc, extract_column_meta(data_meta), get_plot_meta(pp_desc.plot))
        pp_desc = pp_desc.model_copy(update={"factor_cols": factor_cols})

    if check_match:
        matches = matching_plots(pp_desc, full_df, data_meta, details=True, list_hidden=True)
        if isinstance(matches, list) or pp_desc.plot not in matches:
            raise Exception(f"Plot not registered: {pp_desc.plot}")

        matches_dict = cast(dict[str, tuple[int, List[str]]], matches)
        fit, imp = matches_dict[pp_desc.plot]
        if fit < 0:
            raise Exception(f"Plot {pp_desc.plot} not applicable in this situation because of flags {imp}")

    if plot_cache is not None:
        key = json.dumps(pp_desc.model_dump(mode="python"), sort_keys=True)
        if key in plot_cache:
            pi = deepcopy(plot_cache[key])
        else:
            pi = pp_transform_data(full_df, data_meta, pp_desc)
            plot_cache[key] = deepcopy(pi)
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
        return e2e_plot(pp_desc, *args, **kwargs)
    finally:
        _stk_deregister("test")  # And de-register it again
