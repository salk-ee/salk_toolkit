"""Continuous transforms, including numpy-based row transforms for ranked data."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import polars as pl


# Row-wise numpy transforms: noticeably slower, so only use where a polars expression is infeasible
custom_row_transforms: Dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {}


def _apply_npf_on_pl_df(
    df: pl.DataFrame,
    cols: Sequence[str],
    npf: Callable[[np.ndarray], np.ndarray],
) -> pl.DataFrame:
    """Apply a NumPy-only transformation to selected columns."""

    df[cols] = npf(df[cols].to_numpy())
    return df


# Counting pairwise comparisons is 3-4x faster than concat_list + list.eval, and nulls fall out for free
def _count_ahead(col: str, cols: Sequence[str], ahead: Callable[[str, str], pl.Expr]) -> pl.Expr:
    """How many of ``cols`` come before ``col`` in the row, counting only non-null rivals."""

    counts = [ahead(x, col).fill_null(False).cast(pl.Int32) for x in cols if x != col]
    return pl.sum_horizontal(counts) if counts else pl.lit(0, pl.Int32)


def _ordinal_rank(col: str, cols: Sequence[str], *, descending: bool = False) -> pl.Expr:
    """Rank of ``col`` among the row's non-null values, 1..m, ties broken by column order."""

    i = cols.index(col)

    def ahead(x: str, c: str) -> pl.Expr:
        better = pl.col(x) > pl.col(c) if descending else pl.col(x) < pl.col(c)
        return better | ((pl.col(x) == pl.col(c)) & pl.lit(cols.index(x) < i))

    return pl.when(pl.col(col).is_null()).then(None).otherwise(1 + _count_ahead(col, cols, ahead))


def _rank_transform(
    data: pl.LazyFrame,
    cols: Sequence[str],
    fn: Callable[[pl.Expr], pl.Expr],
    *,
    descending: bool = False,
) -> pl.LazyFrame:
    """Rewrite each column as ``fn`` of its row-wise rank, without leaving polars."""

    return data.with_columns([fn(_ordinal_rank(c, cols, descending=descending)).alias(c) for c in cols])


# Row-wise transforms expressible natively; the rest fall through to custom_row_transforms
ordered_expr_transforms: Dict[str, tuple[Callable[[pl.LazyFrame, Sequence[str]], pl.LazyFrame], str]] = {
    "ordered-avgrank": (lambda d, c: _rank_transform(d, c, lambda r: r), ".1f"),
    "ordered-warf": (lambda d, c: _rank_transform(d, c, lambda r: ((r - 1) / len(c)) ** 12), ".1%"),
    "ordered-top1": (
        lambda d, c: d.with_columns([(pl.col(x) == pl.max_horizontal(c)).cast(pl.Int64).alias(x) for x in c]),
        ".1%",
    ),
    "ordered-bot1": (
        lambda d, c: d.with_columns([(pl.col(x) == pl.min_horizontal(c)).cast(pl.Int64).alias(x) for x in c]),
        ".1%",
    ),
    "ordered-topbot1": (
        lambda d, c: d.with_columns(
            [
                (
                    (pl.col(x) == pl.max_horizontal(c)).cast(pl.Int64)
                    - (pl.col(x) == pl.min_horizontal(c)).cast(pl.Int64)
                ).alias(x)
                for x in c
            ]
        ),
        ".1%",
    ),
    # Fixed-k conveniences; ordered-top-ties:<k> covers any k, tie-inclusively
    "ordered-top2": (lambda d, c: _rank_transform(d, c, lambda r: r <= 2, descending=True), ".1%"),
    "ordered-top3": (lambda d, c: _rank_transform(d, c, lambda r: r <= 3, descending=True), ".1%"),
}


def _ordered_topk(transform: str) -> int | None:
    """Parse ``ordered-top-ties:<k>`` — every column reaching the row's k-th best value."""

    op, _, value = transform.partition(":")
    if op != "ordered-top-ties":
        return None
    try:
        k = int(value)
    except ValueError:
        raise ValueError(f"{transform!r}: {value!r} is not an integer, expected ordered-top-ties:<k>") from None
    if k < 1:
        raise ValueError(f"{transform!r}: k must be at least 1")
    return k


def _ties_topk_transform(data: pl.LazyFrame, cols: Sequence[str], k: int) -> pl.LazyFrame:
    """Mark columns with fewer than k strictly better values in the row, so ties all count."""

    def keep(c: str) -> pl.Expr:
        beaten = _count_ahead(c, cols, lambda x, cc: pl.col(x) > pl.col(cc))
        return pl.when(pl.col(c).is_null()).then(None).otherwise(beaten < k)

    return data.with_columns([keep(c).alias(c) for c in cols])


# Polars is annoyingly verbose for these but it is fast enough to be worth it


def _threshold_cutoff(transform: str) -> float | None:
    """Parse ``ge:<x>`` — "at least x". The threshold belongs to the transform, so it travels as one string."""

    op, _, value = transform.partition(":")
    if op != "ge":
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{transform!r}: {value!r} is not a number, expected ge:<x>") from None


def _transform_cont(
    data: pl.LazyFrame,
    cols: Sequence[str],
    transform: str | None,
    val_format: str = ".1f",
    val_range: tuple[float, float] | None = None,
    agg_fn: str | None = None,
) -> tuple[pl.LazyFrame, str, tuple[float, float] | None]:
    """Apply standardized continuous transforms (center, z-score, etc.); ``agg_fn`` only picks the value format."""

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
    elif (cutoff := _threshold_cutoff(transform)) is not None:
        # An indicator column: `mean` is the share past the threshold, `sum` the weighted count
        data = data.with_columns((pl.col(cols) >= cutoff).cast(pl.Float32))
        return (data, ".0f", None) if agg_fn == "sum" else (data, ".1%", (0.0, 1.0))
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
    elif (k := _ordered_topk(transform)) is not None:
        return _ties_topk_transform(data, cols, k), ".1%", None

    elif transform in ordered_expr_transforms:
        build, fmt = ordered_expr_transforms[transform]
        return build(data, cols), fmt, None

    elif transform in custom_row_transforms:
        _tfunc, fmt = custom_row_transforms[transform]
        # Probe a 1-row dummy at the runtime numpy dtype to declare a map_batches schema (else it panics)
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


# Inline scale transforms, i.e. the ones _transform_cont handles without consulting a registry
SCALE_TRANSFORMS = ("center", "zscore", "01range", "proportion", "softmax", "softmax-ratio")

cont_transform_options = (
    list(SCALE_TRANSFORMS) + list(ordered_expr_transforms.keys()) + list(custom_row_transforms.keys())
)
