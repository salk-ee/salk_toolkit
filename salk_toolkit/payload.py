"""PlotPayload v1: serialize a plot's data + descriptor for non-Vega renderers (e.g. ECharts).

Two extraction paths, picked per plot:
- `payload=True` plots early-return their prepared `PlotInput` on `return_df` -- the authoritative
  path (shares the plot's shaping code, decoupled from Altair's chart internals).
- every other chart-producing plot falls back to building its Altair chart and reading the frame /
  color-scale / geo back off it, so payload coverage is universal without per-plot annotation.
"""

__all__ = ["create_plot_payload", "UnsupportedPayloadError"]

import itertools as it
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, cast

import altair as alt
import numpy as np
import pandas as pd

import salk_toolkit.utils as utils
from salk_toolkit.pp import FacetMeta, PlotInput, _get_plot_fn, _normalize_color_dict, create_plot, get_plot_meta
from salk_toolkit.utils import clean_kwargs
from salk_toolkit.validation import PlotDescriptor

# ColorBrewer stops for Vega's "yellowgreen"/"redyellowgreen" schemes -- Vega resolves scheme names
# to hex only at render, so the resolved ramp is never on the chart object. Used by the fallback
# path; the `payload=True` matrix/geoplot supply their own resolved stops via `extras`.
_SCHEME_STOPS: Dict[str, List[str]] = {
    "yellowgreen": ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#006837", "#004529"],
    "redyellowgreen": [
        "#a50026",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#d9ef8b",
        "#a6d96a",
        "#66bd63",
        "#1a9850",
        "#006837",
    ],  # noqa: E501
}


class UnsupportedPayloadError(Exception):
    """Raised when a plot yields no extractable data frame (e.g. a streamlit-only widget, no `return_df`)."""

    def __init__(self, plot: str) -> None:
        """Store the unsupported plot name on `.plot` for callers to inspect."""
        super().__init__(f"Plot '{plot}' produced no payload-extractable data")
        self.plot = plot


# ---- chart introspection (fallback path) ---------------------------------------------------
# These read data / scale / geo off a built Altair chart. They are coupled to Altair's object
# model: attribute reads return `_PropertySetter` proxies and transform `from` lives under the
# schema key `"from"` (not `.from_`), so we read via `.to_dict()` / `[...]` where needed.


def _seq(x: object) -> list:
    """Altair-aware list coercion: None / Undefined -> []."""
    return [] if x is None or x is alt.Undefined else list(cast(Sequence[Any], x))


def _walk(chart: object) -> Iterator[object]:
    """Yield a chart and every nested sub-chart (layer / concat / facet spec)."""
    yield chart
    for attr in ("layer", "hconcat", "vconcat", "concat"):
        for sub in _seq(getattr(chart, attr, None)):
            yield from _walk(sub)
    spec = getattr(chart, "spec", alt.Undefined)
    if spec is not alt.Undefined and spec is not None:
        yield from _walk(spec)


def _sk(obj: object, key: str) -> object:
    """Schema-key read on an altair object (`obj["key"]`), tolerating attribute-proxy shadowing."""
    try:
        return obj[key]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None


def _lookup_from(t: object) -> object:
    """The `from` of a transform_lookup -- stored under schema-key `"from"`, not the `.from_` attr."""
    return _sk(t, "from") or getattr(t, "_kwds", {}).get("from_")


def _chart_frame(chart: object) -> Optional[pd.DataFrame]:
    """The DataFrame a chart draws from: a transform_lookup's survey table, else `.data`."""
    for c in _walk(chart):
        for t in _seq(getattr(c, "transform", None)):
            fdata = getattr(_lookup_from(t), "data", None)
            if isinstance(fdata, pd.DataFrame):
                return fdata
        cdata = getattr(c, "data", None)
        if isinstance(cdata, pd.DataFrame):
            return cdata
    return None


def _chart_scale(chart: object) -> Optional[Dict[str, Any]]:
    """A continuous color scale as {stops: [hex], domain: [...]} -- None for categorical color."""
    for c in _walk(chart):
        color = getattr(getattr(c, "encoding", None), "color", None)
        if color in (None, alt.Undefined):
            continue
        try:
            d = color.to_dict()
        except Exception:  # noqa: BLE001 -- alt.condition() colors etc.
            continue
        sc = d.get("scale") if isinstance(d, dict) and d.get("type") == "quantitative" else None
        if not isinstance(sc, dict):
            continue
        if isinstance(sc.get("range"), (list, tuple)):  # explicit hex ramp
            return {"stops": list(sc["range"]), "domain": sc.get("domain")}
        if sc.get("scheme") in _SCHEME_STOPS:  # named scheme
            domain = [sc.get("domainMin"), sc.get("domainMid", 0), sc.get("domainMax")]
            return {"stops": _SCHEME_STOPS[sc["scheme"]], "domain": domain}
    return None


def _chart_geo(chart: object) -> Optional[Dict[str, Any]]:
    """Geo join spec off a geoshape mark: url + object + region_key + name_property."""
    for c in _walk(chart):
        mark = getattr(c, "mark", None)
        if (mark if isinstance(mark, str) else getattr(mark, "type", None)) != "geoshape":
            continue
        region_key = name_property = None
        for t in _seq(getattr(c, "transform", None)):
            lookup = _sk(t, "lookup")
            region_key = _sk(_lookup_from(t), "key")
            if isinstance(lookup, str) and lookup.startswith("properties."):
                name_property = lookup[len("properties.") :]
        cdata = getattr(c, "data", None)
        src = cdata.to_dict() if cdata is not None and hasattr(cdata, "to_dict") else {}
        feature = src.get("format", {}).get("feature") if isinstance(src, dict) else None
        geo: Dict[str, Any] = {
            "url": src.get("url") if isinstance(src, dict) else None,
            "format": "topojson" if feature is not None else "geojson",
            "region_key": region_key,
            "name_property": name_property,
        }
        if feature is not None:
            geo["object"] = feature
        return geo
    return None


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


def _cell(frame: pd.DataFrame, title: str, keys: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize one cell's DataFrame into a `cells[i][j]` entry."""
    columns = list(frame.columns)
    return {
        "title": title,
        "keys": keys,
        "columns": columns,
        "data": {c: _serialize_column(frame[c]) for c in columns},
    }


def create_plot_payload(
    pi: PlotInput,
    pp_desc: PlotDescriptor,
    translate: Callable[[str], str] | None = None,
) -> Dict[str, Any]:
    """Serialize a `PlotInput` + `PlotDescriptor` into a PlotPayload v1 dict.

    Dry-runs `create_plot`, then per grid cell either early-returns the plot's prepared frame
    (`payload=True` plots, via `return_df`) or builds its Altair chart and reads the frame /
    scale / geo back off it. Raises `UnsupportedPayloadError` only when a cell yields no frame.
    """
    plot_meta = get_plot_meta(pp_desc.plot)
    uses_return_df = plot_meta is not None and plot_meta.payload

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

    combs = list(it.product(*[utils.get_categories(data[fc].dtype) for fc in outer_factors]))
    scale = geo = None
    cell_list: List[Dict[str, Any]] = []
    for comb in combs:  # `it.product()` of no factors yields one empty comb = one cell
        # deepcopy facets: plots may reorder FacetMeta in place across the shared cells
        cell_pi = dry_pi.model_copy(
            update={
                "data": data[(data[outer_factors] == comb).all(axis=1)] if comb else data,
                "facets": deepcopy(dry_pi.facets),
                "plot_args": dict(dry_pi.plot_args),
                "return_df": uses_return_df,
            }
        )
        result = plot_fn(cell_pi, **plot_kwargs)
        if uses_return_df and isinstance(result, PlotInput):  # authoritative path (return_df)
            frame = result.data
        else:  # fallback: read frame / scale / geo back off the built chart
            frame = _chart_frame(result)
            if frame is None:
                raise UnsupportedPayloadError(pp_desc.plot)
            scale = scale or _chart_scale(result)
            geo = geo or _chart_geo(result)
        keys = {oc: _to_json_scalar(v) for oc, v in zip(outer_factors, comb)}
        cell_list.append(_cell(frame, "-".join(map(str, comb)), keys))

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
        "scale": scale,
        "geo": geo,
        "plot_args": {k: v for k, v in dry_pi.plot_args.items() if not str(k).startswith("_")},
    }
