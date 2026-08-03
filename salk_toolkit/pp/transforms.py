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


# Row-wise ordinal rank across cols, 1..n, matching numpy's argsort-of-argsort on distinct values
def _ordinal_ranks(cols: Sequence[str]) -> pl.Expr:
    return pl.concat_list(cols).list.eval(pl.element().rank(method="ordinal"))


def _rank_transform(data: pl.LazyFrame, cols: Sequence[str], fn: Callable[[pl.Expr], pl.Expr]) -> pl.LazyFrame:
    """Rewrite each column as ``fn`` of its row-wise rank, without leaving polars."""

    ranks = "__ranks__"
    return (
        data.with_columns(_ordinal_ranks(cols).alias(ranks))
        .with_columns([fn(pl.col(ranks).list.get(i)).alias(c) for i, c in enumerate(cols)])
        .drop(ranks)
    )


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
    "ordered-top2": (lambda d, c: _rank_transform(d, c, lambda r: r >= len(c) - 1), ".1%"),
    "ordered-top3": (lambda d, c: _rank_transform(d, c, lambda r: r >= len(c) - 2), ".1%"),
}


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


cont_transform_options = (
    [
        "center",
        "zscore",
        "01range",
        "proportion",
        "softmax",
        "softmax-ratio",
    ]
    + list(ordered_expr_transforms.keys())
    + list(custom_row_transforms.keys())
)
