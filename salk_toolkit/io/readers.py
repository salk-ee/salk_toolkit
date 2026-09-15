"""File tracking, path remapping for reproducible packaging, and raw tabular format readers."""

import warnings
from typing import Any

import pandas as pd
import pyreadstat  # type: ignore[import-untyped]


# Tracks which files a model needs, so they can be packaged together

# NB! only map values when loading actual files, not when calling other functions here

#  a global list of files that have been loaded
stk_loaded_files_set = set()


def get_loaded_files() -> list[str]:
    """Get list of all files that have been loaded during this session.

    Returns:
        List of file paths that have been loaded.
    """
    global stk_loaded_files_set
    return list(stk_loaded_files_set)


def reset_file_tracking() -> None:
    """Clear the set of tracked loaded files."""
    global stk_loaded_files_set
    stk_loaded_files_set.clear()


# a global map that allows remapping file paths/names to different paths
stk_file_map = {}


def get_file_map() -> dict[str, str]:
    """Get the current file path mapping dictionary.

    Returns:
        Copy of the file map dictionary.
    """
    global stk_file_map
    return stk_file_map.copy()


def set_file_map(file_map: dict[str, str]) -> None:
    """Set the file path mapping dictionary.

    Args:
        file_map: Dictionary mapping original paths to new paths.
    """
    global stk_file_map
    stk_file_map = file_map.copy()


_TABULAR_EXTENSIONS = ["csv", "gz", "sav", "dta", "xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]


def _apply_value_labels(df: pd.DataFrame, value_labels: dict[str, dict]) -> pd.DataFrame:
    """What pyreadstat's ``set_value_labels`` does on pandas - labelled values recoded, unlabelled
    kept, the column cast to an unordered category - built the same way (one value Series inferred
    from labels + observed values, taken by position), one column at a time."""
    labelled = {}
    for c, labels in value_labels.items():
        if c not in df.columns:
            continue
        s = df[c]
        ext = dict(labels)
        for v in s.unique():
            if v not in ext:
                ext[v] = v
        pos = pd.Index(list(ext)).get_indexer(s)
        out = pd.Series(list(ext.values())).iloc[pos]
        labelled[c] = pd.Series(out.to_numpy(), index=s.index, name=c, dtype=out.dtype).astype("category")
    return df.assign(**labelled) if labelled else df


def _read_tabular(
    mapped_file: str, extension: str, read_opts: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read a raw tabular file (csv/sav/excel family), returning the frame and any reader metadata."""
    if extension in ["csv", "gz"]:
        csv_defaults: dict[str, Any] = {"low_memory": False}
        if read_opts.get("engine") == "python":
            csv_defaults.pop("low_memory")  # python engine doesn't support low_memory
        return pd.read_csv(mapped_file, **{**csv_defaults, **read_opts}), {}  # type: ignore[call-overload]
    if extension in ["sav", "dta"]:
        read_fn = getattr(pyreadstat, "read_" + mapped_file[-3:].lower())
        # pyreadstat's own label pass copies the whole frame once per labelled column (quadratic on
        # wide files); apply the labels here unless the caller asks for pyreadstat's category options.
        own_labels = read_opts.get("apply_value_formats", True) and not (
            {"formats_as_category", "formats_as_ordered_category"} & set(read_opts)
        )
        with warnings.catch_warnings():  # While pyreadstat has not been updated to pandas 2.2 standards
            warnings.simplefilter("ignore")
            raw_data, fmeta = read_fn(
                mapped_file,
                **{
                    "dates_as_pandas_datetime": True,
                    **read_opts,
                    "apply_value_formats": read_opts.get("apply_value_formats", True) and not own_labels,
                },
            )
        if own_labels:
            raw_data = _apply_value_labels(raw_data, fmeta.variable_value_labels)
        # fmeta fields can be used in hooks just like self-defined constants
        return raw_data, dict(fmeta.__dict__)
    return pd.read_excel(mapped_file, **read_opts), {}  # type: ignore[call-overload]
