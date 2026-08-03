"""Filtering and discretization of lazy frames for the plot pipeline."""

from __future__ import annotations

from math import ceil
from typing import Any, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

import salk_toolkit.utils as utils
from salk_toolkit.validation import GroupOrColumnMeta


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
