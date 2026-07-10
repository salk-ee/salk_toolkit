"""File tracking, path remapping for reproducible packaging, and raw tabular format readers."""

import warnings
from typing import Any

import pandas as pd
import pyreadstat  # type: ignore[import-untyped]


# This is here so we can easily track which files would be needed for a model
# so we can package them together if needed

# NB! for unpacking to work, the processing needs to not be changed w.r.t. paths
# For this, we only map values when loading actual files, not when calling other functions here

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
        with warnings.catch_warnings():  # While pyreadstat has not been updated to pandas 2.2 standards
            warnings.simplefilter("ignore")
            raw_data, fmeta = read_fn(
                mapped_file,
                **{"apply_value_formats": True, "dates_as_pandas_datetime": True},
                **read_opts,
            )
        # fmeta fields can be used in hooks just like self-defined constants
        return raw_data, dict(fmeta.__dict__)
    return pd.read_excel(mapped_file, **read_opts), {}  # type: ignore[call-overload]
