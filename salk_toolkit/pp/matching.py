"""Plot matching: score plot types against data and impute factor columns."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, overload

import pandas as pd
import polars as pl

from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate

from .common import _get_cat_num_vals
from .meta import _update_data_meta_with_pp_desc
from .registry import PlotMeta, _ensure_plot_registry_loaded, get_plot_meta, registry, registry_meta


# [weight if not matching, weight if match] - very much a placeholder right now
n_a = -1000000
priority_weights = {
    "draws": [n_a, 0],
    "nonnegative": [n_a, 40],
    "hidden": [n_a, 0],
    "ordered": [n_a, 10],
    "likert": [n_a, 200],
    "required_meta": [n_a, 500],
}


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
        pp_desc, col_meta = _impute_facet_dims(pp_desc, data_meta)
    else:
        col_meta, _ = _update_data_meta_with_pp_desc(data_meta, pp_desc)

    rc = pp_desc.res_col
    rcm = col_meta[rc]

    if isinstance(df, pl.LazyFrame):
        df_cols = df.collect_schema().names()
    else:
        df_cols = df.columns

    # Determine if values are non-negative
    ocols = list(rcm.columns) if rcm.columns else [rc]
    cols = [c for c in ocols if c in df_cols]
    if not cols:
        raise ValueError(f"Columns {ocols} not found in data")

    if rcm.is_categorical:
        nonneg = True
    else:
        # Metadata-only: a data scan would cost a full pass per render; unknown counts as not non-negative
        val_range = rcm.val_range
        nonneg = val_range is not None and val_range[0] is not None and val_range[0] >= 0

    convert_res = pp_desc.convert_res
    if convert_res == "continuous" and rcm.is_categorical:
        cat_vals_seq = _get_cat_num_vals(rcm, pp_desc)
        cat_vals = [v for v in (cat_vals_seq or []) if v is not None]
        if cat_vals:
            nonneg = min(cat_vals) >= 0

    facet_dims = pp_desc.facet_dims
    facet_metas = []
    for cn in facet_dims:
        meta = col_meta.get(cn, GroupOrColumnMeta())
        facet_metas.append({"name": cn, **meta.model_dump(mode="python")})
    # convert_res decides it outright ('categorical' bins a number, 'continuous' numbers an ordinal)
    is_categorical = (convert_res == "categorical") if convert_res else rcm.is_categorical
    match = {
        "draws": ("draw" in df_cols),
        "nonnegative": nonneg,
        "hidden": list_hidden,
        "res_col": rc,
        "categorical": is_categorical,
        "facet_metas": facet_metas,
    }

    # _calculate_priority only reads the meta, so skip get_plot_meta's defensive deep copy per plot
    _ensure_plot_registry_loaded()
    res = [(pn, *_calculate_priority(registry_meta[pn], match)) for pn in registry.keys()]
    if details:
        return {n: (p, i) for (n, p, i) in res}  # Return dict with priorities and failure reasons
    else:
        return [
            n for (n, p, i) in sorted(res, key=lambda t: t[1], reverse=True) if p >= 0
        ]  # Return list of possibilities in decreasing order of fit


def _remove_from_inner_facets(cname: str, facet_dims: List[str], n_inner: int) -> int:
    """Shift ``cname`` out of the inner facet slice while preserving order."""

    if cname not in facet_dims[:n_inner]:
        return n_inner
    facet_dims.remove(cname)
    if n_inner > len(facet_dims):
        n_inner -= 1
    facet_dims.insert(n_inner, cname)
    return n_inner


def _inner_outer_facets(
    facet_dims: List[str],
    pp_desc: PlotDescriptor,
    plot_meta: PlotMeta,
) -> tuple[List[str], int]:
    """Return `(facet_dims, n_inner)` after respecting descriptor overrides."""

    # Determine how many factors to use as inner facets
    in_f = pp_desc.internal_facet if pp_desc.internal_facet is not None else False
    res_col = pp_desc.res_col
    n_min_f, n_rec_f = plot_meta.n_facets or (0, 0)
    n_inner: int = (n_rec_f if in_f else n_min_f) if isinstance(in_f, bool) else int(in_f)  # type: ignore[arg-type]
    if n_inner > len(facet_dims):
        n_inner = len(facet_dims)

    # If question facet as inner facet for a no_question_facet plot, just move it out
    if plot_meta.no_question_facet:
        n_inner = _remove_from_inner_facets("question", facet_dims, n_inner)
        n_inner = _remove_from_inner_facets(res_col, facet_dims, n_inner)

    return facet_dims, n_inner


def impute_facet_dims(
    pp_desc: PlotDescriptor | Dict[str, Any],
    col_meta: Mapping[str, GroupOrColumnMeta],
    plot_meta: PlotMeta | None = None,
) -> List[str]:
    """Compute the full facet_dims list, including question and res_col as needed."""

    if isinstance(pp_desc, dict) and "plot" not in pp_desc:
        pp_desc["plot"] = "default"

    # Ensure pp_desc is a PlotDescriptor object
    pp_desc = soft_validate(pp_desc, PlotDescriptor)

    facet_dims = list(pp_desc.facet_dims or [])

    # Determine if res is categorical
    res_col = pp_desc.res_col
    convert_res = pp_desc.convert_res
    res_col_meta = col_meta[res_col]
    has_q = res_col_meta.columns is not None
    # A number bound for binning ends up categorical too, so it wants res_col as a facet
    cat_res = (res_col_meta.is_categorical or convert_res == "categorical") and convert_res != "continuous"

    # Add res_col if we are working with a categorical input (and not converting it to continuous)
    if cat_res and res_col not in facet_dims:
        facet_dims.insert(0, res_col)
    if len(facet_dims) < 1 and not has_q:
        # Create 'question' as a dummy dimension so we have at least one factor (usually required)
        has_q = True

    # If we need to, add question as a factor to list
    if has_q and "question" not in facet_dims:
        if cat_res:
            facet_dims.append("question")  # Put it last for categorical values
        else:
            facet_dims.insert(
                0, "question"
            )  # And first for continuous values, as it then often represents the "category"

    # Pass the facet_dims through the same changes done inside plot pipeline to make more explicit what happens
    if plot_meta:
        facet_dims, _ = _inner_outer_facets(facet_dims, pp_desc, plot_meta)

    return facet_dims


def _impute_facet_dims(
    pp_desc: PlotDescriptor, data_meta: DataMeta
) -> tuple[PlotDescriptor, Dict[str, GroupOrColumnMeta]]:
    """Fill in facet_dims, reading the meta through the descriptor's overrides so res_meta blocks are visible."""

    col_meta, _ = _update_data_meta_with_pp_desc(data_meta, pp_desc)
    facet_dims = impute_facet_dims(pp_desc, col_meta, get_plot_meta(pp_desc.plot))
    return pp_desc.model_copy(update={"facet_dims": facet_dims}), col_meta


# Legacy name kept for external callers
impute_factor_cols = impute_facet_dims
