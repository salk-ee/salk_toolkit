"""Core helpers shared across the io package: shared series converters and the processed-data alias."""

import warnings
from typing import TypeAlias

import numpy as np
import pandas as pd

from salk_toolkit.utils import (
    is_date_str_series,
    is_numeric_str_series,
)
from salk_toolkit.validation import (
    DataMeta,
)


warnings.filterwarnings("ignore", "DataFrame is highly fragmented.*", pd.errors.PerformanceWarning)


def _str_from_list(val: list[str] | object) -> str:
    """Convert a list to a newline-separated string, or return string representation."""
    if isinstance(val, list):
        return "\n".join(val)
    return str(val)


ProcessedDataReturn: TypeAlias = tuple[pd.DataFrame, DataMeta | None]


# --------------------------------------------------------
#          STABLE ROW IDENTITY
# --------------------------------------------------------
#
# Every row carries a stable string id in the reserved ``ROW_ID`` column, assigned at load
# time (before any concat/filter can move rows) and joined across nesting levels as a
# ``{file_code}::...::{leaf}`` path. It becomes the frame index at every io return boundary,
# giving callers a unique, deterministic, provenance-bearing index. Ids are opaque: code only
# prepends prefixes and does exact-string matching, never splits them.

ROW_ID = "row_id"

# Per-file provenance columns injected into every row (paired, must stay 1-to-1).
PROVENANCE_COLUMNS = ("file_code", "file_name")


def mint_positional_row_id(df: pd.DataFrame, file_code: str = "F0") -> pd.DataFrame:
    """Assign a fresh positional ``{file_code}::{i}`` row id in place (leaf files, inline data)."""
    df[ROW_ID] = file_code + "::" + pd.RangeIndex(len(df)).astype(str)
    return df


def assert_row_id_intact(df: pd.DataFrame, context: str) -> None:
    """Fail loudly if user code dropped, nulled, or duplicated the stable row id.

    Filtering/reordering rows is fine; rows added without an id (null) or id collisions are not.
    """
    if ROW_ID not in df.columns:
        raise ValueError(f"{context} removed the '{ROW_ID}' column - stable row ids must be preserved")
    col = df[ROW_ID]
    if col.isna().any():
        raise ValueError(f"{context} produced null row ids - were rows added programmatically without ids?")
    if col.duplicated().any():
        dups = list(col[col.duplicated()].unique()[:5])
        raise ValueError(f"{context} produced duplicate row ids, e.g. {dups} - were rows added programmatically?")


def restore_or_assert_row_id(df: pd.DataFrame, context: str, file_code: str = "PP") -> pd.DataFrame:
    """Re-mint row ids when a hook legitimately rebuilt the frame; otherwise assert them intact.

    Aggregating postprocessing (e.g. a census ``groupby(...).sum()``) cannot preserve source-row
    identity - the output rows get fresh positional ``{file_code}::{i}`` ids with a warning.
    """
    if ROW_ID not in df.columns:
        warnings.warn(
            f"{context} rebuilt the frame without '{ROW_ID}' (e.g. via groupby); minting fresh positional row ids",
            UserWarning,
            stacklevel=2,
        )
        return mint_positional_row_id(df, file_code)
    assert_row_id_intact(df, context)
    return df


def finalize_row_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set ``ROW_ID`` as the frame index at an io return boundary, asserting uniqueness.

    Idempotent (returns unchanged if already row_id-indexed). Frames with no id at all
    (legacy parquet, inline data) get a fresh positional one.
    """
    if df.index.name == ROW_ID:
        return df
    if ROW_ID not in df.columns:
        mint_positional_row_id(df)
    assert_row_id_intact(df, "finalize")
    return df.set_index(ROW_ID)


def _convert_number_series_to_categorical(s: pd.Series) -> pd.Series:
    """Convert number series to categorical, avoiding long and unwieldy fractions like 24.666666666667.

    This is a practical judgement call right now - round to two digits after comma and remove .00 from integers.
    """
    return s.astype("float").map("{:.2f}".format).str.replace(".00", "").replace({"nan": None})


def _convert_datetime_series_to_categorical(s: pd.Series) -> pd.Series:
    """Convert datetime series to categorical strings like "01 Dec 25".

    Note: Month names are locale-dependent; this is intentional per user preference.
    """
    dt = pd.to_datetime(s, errors="coerce")
    # out = dt.dt.strftime("%Y-%m-%d") # ISO YYYY-MM-DD
    out = dt.dt.strftime("%d %b %y")
    return out.where(dt.notna(), None)


def _deterministic_categories_and_values(s: pd.Series) -> tuple[pd.Series, list[object]]:
    """Return (possibly coerced) values + deterministic category list.

    Sorts numeric and datetime values (even if strings) in expected order.
    Also converts values to strings if needed.
    """
    s_nonnull = s.dropna()
    if len(s_nonnull) == 0:
        return s, []

    # Merge numeric dtype and numeric-like string paths to share code
    is_true_numeric = pd.api.types.is_numeric_dtype(s_nonnull)  # Case 1
    is_numeric_like_str = is_numeric_str_series(s)  # Case 2

    # Same for datetime
    is_true_datetime = pd.api.types.is_datetime64_any_dtype(s_nonnull)
    is_datetime_like_str = is_date_str_series(s)

    if is_true_numeric:
        s_str = _convert_number_series_to_categorical(s)
    elif is_true_datetime:
        s_str = _convert_datetime_series_to_categorical(s)
    else:
        s_str = s.copy()

    # Ensure everything is a string
    s_str.loc[~s_str.isna()] = s_str[~s_str.isna()].astype(str)

    # true datetime types are numeric so have to be excluded
    is_numeric = (is_true_numeric or is_numeric_like_str) and not is_true_datetime
    conv_f = pd.to_numeric if is_numeric else pd.to_datetime

    if is_true_numeric or is_numeric_like_str or is_true_datetime or is_datetime_like_str:
        unique_vals = pd.unique(s_str.dropna())
        cats = sorted(unique_vals, key=conv_f)
        return s_str, cats

    # Case 3: general categorical-like values -> deterministic lexicographic ordering
    uniq = list(sorted(pd.unique(s_str.dropna())))
    return s_str, uniq


def _is_series_of_lists(s: pd.Series) -> bool:
    """Check if a pandas Series contains lists or arrays as values."""
    dropped = s.dropna()
    if len(dropped) == 0:
        return False
    s_rep = dropped.iloc[0]  # Find a non-na element
    return isinstance(s_rep, list) or isinstance(s_rep, np.ndarray)
