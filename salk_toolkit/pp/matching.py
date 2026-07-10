"""Plot matching: score plot types against data and impute factor columns."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, cast, overload

import pandas as pd
import polars as pl

from salk_toolkit.io import extract_column_meta
from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate

from .common import _get_cat_num_vals
from .meta import _update_data_meta_with_pp_desc
from .registry import PlotMeta, get_plot_meta, registry


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
