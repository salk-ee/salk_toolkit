"""PlotPayload v1: serialize a plot's prepared data + descriptor for non-Vega renderers (e.g. ECharts)."""

__all__ = ["create_plot_payload", "UnsupportedPayloadError"]

import itertools as it
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import altair as alt
import numpy as np
import pandas as pd

import salk_toolkit.utils as utils
from salk_toolkit.pp import FacetMeta, PlotInput, _get_plot_fn, _normalize_color_dict, create_plot, get_plot_meta
from salk_toolkit.utils import clean_kwargs
from salk_toolkit.validation import PlotDescriptor


class UnsupportedPayloadError(Exception):
    """Raised for plots without payload support (`payload=False` in their `@stk_plot` meta)."""

    def __init__(self, plot: str) -> None:
        """Store the unsupported plot name on `.plot` for callers to inspect."""
        super().__init__(f"No payload support registered for plot '{plot}'")
        self.plot = plot


def _to_json_scalar(v: object) -> object:
    """Normalize a value for JSON: NaN/NA -> None, numpy scalars -> python scalars."""
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    try:
        if pd.isna(cast(Any, v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _serialize_column(series: pd.Series) -> List[Any]:
    """Serialize one dataframe column: categoricals as strings, scalars via `_to_json_scalar`."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [None if pd.isna(v) else str(v) for v in series]
    return [_to_json_scalar(v) for v in series.tolist()]


def _plain_colors(facet: FacetMeta) -> Optional[Dict[str, str]]:
    """Reduce a facet's Altair-ready `colors` (Scale / dict / Undefined) to a plain hex map or None."""
    colors = facet.colors
    if isinstance(colors, alt.Scale):
        domain, rng = colors.domain, colors.range
        if domain is alt.Undefined or rng is alt.Undefined:
            return None
        return {str(k): v for k, v in zip(cast(Sequence[Any], domain), cast(Sequence[Any], rng))}
    if isinstance(colors, dict):
        return _normalize_color_dict(cast(Dict[str, Any], colors))
    return None


def _facet_colors(facet: FacetMeta) -> Optional[Dict[str, str]]:
    """Facet colors as plain hex; falls back to salk's default palette (what Vega would render)."""
    plain = _plain_colors(facet)
    if plain is not None or not facet.order:
        return plain
    palette = utils.altair_default_config["range"]["category"]
    return {str(c): palette[i % len(palette)] for i, c in enumerate(facet.order)}


def _cell(cell_pi: PlotInput, title: str, keys: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize one prepared `PlotInput` into a `cells[i][j]` entry."""
    data = cell_pi.data
    columns = list(data.columns)
    return {
        "title": title,
        "keys": keys,
        "columns": columns,
        "data": {c: _serialize_column(data[c]) for c in columns},
    }


def create_plot_payload(
    pi: PlotInput,
    pp_desc: PlotDescriptor,
    translate: Callable[[str], str] | None = None,
) -> Dict[str, Any]:
    """Serialize a `PlotInput` + `PlotDescriptor` into a PlotPayload v1 dict.

    Runs the `create_plot` dry-run pipeline, then calls the plot function per grid cell with
    `return_df=True` to get its prepared frame. Raises `UnsupportedPayloadError` for plots
    without `payload=True` in their meta.
    """
    plot_meta = get_plot_meta(pp_desc.plot)
    if plot_meta is None or not plot_meta.payload:
        raise UnsupportedPayloadError(pp_desc.plot)

    dry_pi = create_plot(pi, pp_desc, dry_run=True, escape_labels=False, translate=translate)
    assert isinstance(dry_pi, PlotInput)

    plot_fn = _get_plot_fn(pp_desc.plot)
    plot_kwargs = clean_kwargs(plot_fn, dry_pi.plot_args)

    data = dry_pi.data
    outer_factors = list(dry_pi.outer_factors or [])
    n_facet_cols = dry_pi.n_facet_cols or 1

    # Sourced from dry_pi before the per-cell loop so cell mutations can't corrupt it
    facets_payload = [
        {"col": f.col, "order": list(f.order), "colors": _facet_colors(f), "neutrals": list(f.neutrals)}
        for f in dry_pi.facets
    ]

    # Each cell gets fresh container copies; objects inside them (e.g. FacetMeta) stay shared
    # across cells and must be replaced, not mutated, by the plot fn.
    combs = list(it.product(*[utils.get_categories(data[fc].dtype) for fc in outer_factors]))
    extras: Dict[str, Any] = {}
    cell_list: List[Dict[str, Any]] = []
    for comb in combs:  # `it.product()` of no factors yields one empty comb = one cell
        cell_pi = dry_pi.model_copy(
            update={
                "data": data[(data[outer_factors] == comb).all(axis=1)] if comb else data,
                "extras": dict(dry_pi.extras),
                "facets": list(dry_pi.facets),
                "plot_args": dict(dry_pi.plot_args),
                "return_df": True,
            }
        )
        cell_pi = plot_fn(cell_pi, **plot_kwargs)
        if not isinstance(cell_pi, PlotInput):
            raise UnsupportedPayloadError(pp_desc.plot)
        extras.update(cell_pi.extras or {})
        keys = {oc: _to_json_scalar(v) for oc, v in zip(outer_factors, comb)}
        cell_list.append(_cell(cell_pi, "-".join(map(str, comb)), keys))

    cells = [list(row) for row in utils.batch(cell_list, n_facet_cols)]
    value_range = [_to_json_scalar(v) for v in dry_pi.value_range] if dry_pi.value_range is not None else None

    return {
        "payload_version": 1,
        "plot": pp_desc.plot,
        "value_col": dry_pi.value_col,
        "cat_col": dry_pi.cat_col,
        "val_format": dry_pi.val_format,
        "filtered_size": dry_pi.filtered_size,
        "value_range": value_range,
        "facets": facets_payload,
        "outer_factors": outer_factors,
        "grid": {
            "rows": len(cells),
            "cols": n_facet_cols if outer_factors else 1,
            "requested_cols": pp_desc.n_facet_cols,
        },
        "cells": cells,
        "scale": extras.get("scale"),
        "geo": extras.get("geo"),
        "plot_args": {k: v for k, v in dry_pi.plot_args.items() if not str(k).startswith("_")},
    }
