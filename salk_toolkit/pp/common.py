"""Shared types and small helpers used across the plot pipeline."""

from __future__ import annotations

from copy import copy as shallow_copy, deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import altair as alt
import numpy as np
import pandas as pd
from pydantic_extra_types.color import Color

from salk_toolkit.validation import ColumnMeta, GroupOrColumnMeta, PlotDescriptor


def _meta_to_plain(meta: ColumnMeta) -> Dict[str, Any]:
    """Return a plain dict copy of column metadata."""

    return meta.model_dump(mode="python")


def _question_meta_clone(base_meta: GroupOrColumnMeta, categories: Sequence[str] | None = None) -> GroupOrColumnMeta:
    """Produce a categorical copy of ``base_meta`` for the synthetic ``question`` column.

    ``continuous`` and ``ordered`` are cleared so question identity is nominal, not inherited from the group.
    """

    clone = base_meta.model_copy(deep=True)
    clone.continuous = False
    clone.ordered = False
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


special_columns: List[str] = [
    "id",
    "weight",
    "draw",
    "original_inds",
    "__index_level_0__",
    "group_size",
    "ordering_value",
]
