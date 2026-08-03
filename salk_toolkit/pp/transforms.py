"""Continuous transforms, including numpy-based row transforms for ranked data."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import polars as pl


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
        # Probe the transform with a 1-row dummy to derive an explicit map_batches schema,
        # so the streaming engine can initialise array builders for downstream group_by/agg
        # without hitting the OPAQUE_PYTHON boundary. The probe must use the same numpy
        # dtype `df[cols].to_numpy()` will yield at runtime: dtype-preserving transforms
        # (softmax-avgrank) would otherwise declare Float64 over a Float32 batch and panic
        # in the reducer (`values.dtype() == &self.in_dtype`). validate_output_schema stays
        # False because dtype-changing transforms (e.g. Float32 → Int64 for topbot1) are OK.
        input_schema = data.collect_schema()
        set_cols = set(cols)
        in_np_dtype = np.result_type(*(pl.Series([], dtype=input_schema[c]).to_numpy().dtype for c in cols))
        _probe = _tfunc(np.zeros((1, len(cols)), dtype=in_np_dtype))
        col_dtype = pl.Series(_probe[0]).dtype
        output_schema = pl.Schema({c: (col_dtype if c in set_cols else input_schema[c]) for c in input_schema})
        data = data.map_batches(
            lambda bdf: _apply_npf_on_pl_df(bdf, cols, _tfunc),
            streamable=True,
            validate_output_schema=False,
            schema=output_schema,
            projection_pushdown=False,  # Keeps batch columns consistent with declared schema
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
