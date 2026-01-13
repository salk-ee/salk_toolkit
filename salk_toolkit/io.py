"""Data Annotation I/O
---------------------

This module now contains the functionality that used to be split across
`01_io.ipynb`.  It covers:

- tracking loaded files and remapping paths for reproducible packaging
- loaders for JSON/YAML annotations, Parquet files with embedded metadata,
  CSV/Excel/SPSS datasets, and helper utilities for top-k/maxdiff transforms
- helpers for explicating annotation structures (`extract_column_meta`,
  `group_columns_dict`, `fix_meta_categories`, etc.)
- orchestration helpers such as `read_annotated_data` and
  `read_and_process_data` that execute preprocessing hooks, merges, and transformations end-to-end
"""

__all__ = [
    # Public IO helpers are limited to what salk_internal_package imports.
    "extract_column_meta",
    "get_file_map",
    "get_loaded_files",
    "group_columns_dict",
    "list_aliases",
    "read_and_process_data",
    "read_annotated_data",
    "read_parquet_with_metadata",
    "reset_file_tracking",
    "set_file_map",
    "write_parquet_with_metadata",
    "replace_data_meta_in_parquet",
    "read_parquet_metadata",
    "infer_meta",
    "update_meta_with_model_fields",
]

import json
import os
import warnings
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any, Callable, Literal, TypeAlias, cast, overload

import numpy as np
import pandas as pd
import scipy as sp
import polars as pl


import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pyreadstat  # type: ignore[import-untyped]

import salk_toolkit as stk
from salk_toolkit import utils
from salk_toolkit.utils import (
    replace_constants,
    is_date_str_series,
    is_datetime,
    is_numeric_str_series,
    warn,
    cached_fn,
    read_yaml,
    read_json,
)
from salk_toolkit.validation import (
    DataMeta,
    DataDescription,
    ParquetMeta,
    soft_validate,
    ColumnBlockMeta,
    ColumnMeta,
    FileDesc,
    GroupOrColumnMeta,
    SingleMergeSpec,
    TopKBlock,
)

# Ignore fragmentation warnings
warnings.filterwarnings("ignore", "DataFrame is highly fragmented.*", pd.errors.PerformanceWarning)


def _str_from_list(val: list[str] | object) -> str:
    """Convert a list to a newline-separated string, or return string representation."""
    if isinstance(val, list):
        return "\n".join(val)
    return str(val)


# This is here so we can easily track which files would be needed for a model
# so we can package them together if needed

# NB! for unpacking to work, the processing needs to not be changed w.r.t. paths
# For this, we only map values when loading actual files, not when calling other functions here

#  a global list of files that have been loaded
stk_loaded_files_set = set()

ProcessedDataReturn: TypeAlias = tuple[pd.DataFrame, DataMeta | None]


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


def _reconcile_categories(
    raw_data_dict: dict[str, pd.DataFrame],
    initial_cat_dtypes: dict[str, pd.CategoricalDtype | None] | None = None,
) -> dict[str, pd.CategoricalDtype]:
    """Detect and reconcile categorical dtypes across multiple files, preserving all categories.

    Args:
        raw_data_dict: Dictionary of dataframes keyed by file code.
        initial_cat_dtypes: Initial categorical dtypes (e.g., from file_code, extra fields).

    Returns:
        Dictionary mapping column names to reconciled categorical dtypes.
    """
    if initial_cat_dtypes is None:
        initial_cat_dtypes = {}

    cat_dtypes: dict[str, pd.CategoricalDtype | None] = dict(initial_cat_dtypes)
    multiple_files = len(raw_data_dict) > 1

    # Handle categorical types for more complex situations
    # Detect categorical columns and strip categories when multiple files involved
    for file_code, raw_data in raw_data_dict.items():
        for c in raw_data.columns:
            dropped = raw_data[c].dropna()
            if raw_data[c].dtype.name == "object" and len(dropped) > 0 and not isinstance(dropped.iloc[0], list):
                cat_dtypes[c] = cat_dtypes.get(c, None)  # Infer a categorical type unless already given
            elif (
                raw_data[c].dtype.name == "category" and multiple_files
            ):  # Strip categories when multiple files involved
                existing_dtype = cat_dtypes.get(c)
                if (
                    c not in cat_dtypes
                    or existing_dtype is None
                    or len(utils.get_categories(existing_dtype)) <= len(utils.get_categories(raw_data[c].dtype))
                ):
                    cat_dtypes[c] = raw_data[c].dtype
                raw_data[c] = raw_data[c].astype("object")

    # Now reconcile across all files
    reconciled: dict[str, pd.CategoricalDtype] = {}

    # Concatenate all dataframes to get all unique values
    all_dfs = list(raw_data_dict.values())
    if not all_dfs:
        return reconciled

    fdf = pd.concat(all_dfs)

    # Reconcile each categorical column
    for c, dtype in cat_dtypes.items():
        if c not in fdf.columns:
            continue

        if dtype is None:  # Added as an extra field, infer categories
            s = fdf[c].dropna()
            if (
                s.dtype.name == "object"
                and len(s) > 0
                and not isinstance(s.iloc[0], str)  # Check for string as string is also iterable
                and isinstance(s.iloc[0], Iterable)
            ):
                continue  # Skip if it's a list or tuple or ndarray
            _, cats = _deterministic_categories_and_values(s)
            reconciled[c] = pd.Categorical([], cats).dtype
        elif not set(fdf[c].dropna().unique()) <= set(
            utils.get_categories(dtype)
        ):  # If the categories are not the same, create a new dtype
            _, cats = _deterministic_categories_and_values(fdf[c].dropna())
            reconciled[c] = pd.Categorical([], cats).dtype
            n_cats = len(utils.get_categories(reconciled[c]))
            warn(f"Categories for {c} are different between files - merging to total {n_cats} cats")
        else:
            reconciled[c] = dtype

    return reconciled


def _load_data_files(
    data_files: list[FileDesc],
    path: str | None,
    read_opts: dict[str, Any] | None = None,
    ignore_exclusions: bool = False,
    only_fix_categories: bool = False,
    add_original_inds: bool = False,
) -> tuple[dict[str, pd.DataFrame], DataMeta | None, dict[str, object]]:
    """Internal helper to load files defined in metadata or descriptions.

    Returns per-file dataframes keyed by file code, keeping them separate for per-file processing.
    """

    global stk_loaded_files_set, stk_file_map

    raw_data_dict: dict[str, pd.DataFrame] = {}
    metas, einfo = [], {}
    if read_opts is None:
        read_opts = {}
    for fi, fd in enumerate(data_files):
        data_file = fd.file
        opts = fd.opts or read_opts
        file_code = fd.code if fd.code is not None else f"F{fi}"  # Default to F0, F1, F2, etc.
        if path:
            # path is guaranteed to be str here due to the if check
            path_str = cast(str, path)
            data_file = os.path.join(os.path.dirname(path_str), cast(str, data_file))
        mapped_file = stk_file_map.get(cast(str, data_file), cast(str, data_file))

        extension = os.path.splitext(cast(str, data_file))[1][1:].lower()
        if extension in [
            "json",
            "parquet",
            "yaml",
        ]:  # Allow loading metafiles or annotated data
            if extension == "json":
                warn(f"Processing {data_file}")  # Print this to separate warnings for input jsons from main
            # Pass in orig_data_file here as it might loop back to this function here and we need to preserve paths
            raw_data, result_meta = read_annotated_data(
                cast(str, data_file),
                infer=False,
                return_meta=True,
                ignore_exclusions=ignore_exclusions,
                only_fix_categories=only_fix_categories,
                add_original_inds=add_original_inds,
            )
            if result_meta is not None:
                metas.append(soft_validate(result_meta, DataMeta, warnings=True))
        elif extension in ["csv", "gz"]:
            read_opts = cast(dict[str, Any], opts) if opts else {}
            raw_data = pd.read_csv(cast(str, mapped_file), low_memory=False, **read_opts)  # type: ignore[call-overload]
        elif extension in ["sav", "dta"]:
            read_fn = getattr(pyreadstat, "read_" + (mapped_file[-3:]).lower())
            with warnings.catch_warnings():  # While pyreadstat has not been updated to pandas 2.2 standards
                warnings.simplefilter("ignore")
                read_opts = cast(dict[str, Any], opts) if opts else {}
                raw_data, fmeta = read_fn(
                    cast(str, mapped_file),
                    **{"apply_value_formats": True, "dates_as_pandas_datetime": True},
                    **read_opts,
                )
                einfo.update(fmeta.__dict__)  # Allow the fields in meta to be used just like self-defined constants
        elif extension in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
            read_opts = cast(dict[str, Any], opts) if opts else {}
            raw_data = pd.read_excel(cast(str, mapped_file), **read_opts)  # type: ignore[call-overload]
        else:
            raise Exception(f"Not a known file format for {data_file}: {extension}")

        stk_loaded_files_set.add(mapped_file)

        # If data is multi-indexed, flatten the index
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = [" | ".join(tpl) for tpl in raw_data.columns]

        # Add extra columns to raw data that contain info about the file
        # Always includes 'file_ind' with index and 'file_code' with the code identifier
        # Can be used to add survey_date or other useful metainfo
        if len(data_files) > 1:
            raw_data["file_ind"] = fi
            raw_data["file_code"] = file_code
        # Add extra fields from FileDesc (any fields beyond file, opts, code)
        # In Pydantic v2, extra fields are stored in __pydantic_extra__ when extra="allow"
        pydantic_extra = getattr(fd, "__pydantic_extra__", None) or {}
        for k, v in pydantic_extra.items():
            if len(data_files) <= 1 and k == "file":
                continue
            raw_data[k] = v

        raw_data_dict[file_code] = raw_data

    if metas:  # Do we have any metainfo?
        meta = metas[-1]

        # This will fix categories inside meta too - use concatenated view for this
        fdf = pd.concat(raw_data_dict.values())
        meta = _fix_meta_categories(meta, fdf, warnings=False)
        # TODO: one should also merge the structures in case the columns don't match
        return raw_data_dict, meta, einfo
    else:
        return raw_data_dict, None, einfo


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


def _throw_vals_left(df: pd.DataFrame) -> None:
    """Move all NaN values to the right in each row (in-place)."""
    # Helper fun to move inplace all nan values to right.
    df.iloc[:, :] = df.apply(lambda row: sorted(row, key=pd.isna), axis=1).to_list()


def _create_topk_metas_and_dfs(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame."""
    if block_with_create.create is None:
        raise ValueError("ColumnBlockMeta must have a create block")
    create = block_with_create.create
    if not isinstance(create, TopKBlock):
        raise ValueError("Expecting TopKBlock create block")
    name = f"{block_with_create.name}_{create.type}"
    from_columns = create.from_columns
    if from_columns is None:
        raise ValueError("TopKBlock must have from_columns specified")

    has_regex = isinstance(from_columns, str)
    has_list = isinstance(from_columns, list)

    if has_regex:
        return _create_topk_metas_and_dfs_regex(df, block_with_create, create, name)
    elif has_list:
        return _create_topk_metas_and_dfs_list(df, block_with_create, create, name)
    else:
        raise ValueError(f"from_columns must be either str (regex) or list, got {type(from_columns)}")


def _create_topk_metas_and_dfs_regex(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
    create: TopKBlock,
    name: str,
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame using regex pattern matching."""
    from_columns = create.from_columns
    assert isinstance(from_columns, str), "from_columns must be str for regex mode"

    regex_from = re.compile(from_columns)
    from_cols = list(filter(lambda s: regex_from.match(s), df.columns))
    # agg_ind is index of the regex group that we want to aggregate over.
    # All other indeces are unique identifiers for each subgroup
    # e.g. ['A'] and ['B'] (that are also added to meta names)
    # Recall that regex group 0 is the whole match.
    # Note that re.Match.groups() does not include the whole match.
    # This means that we need to subtract 1 from agg_ind.
    agg_ind = create.agg_index
    agg_ind = agg_ind - 1 if agg_ind > 0 else agg_ind
    first_match = regex_from.match(from_cols[0])
    assert first_match is not None, f"Column {from_cols[0]} should match regex {regex_from.pattern}"
    n_groups = len(first_match.groups())
    has_subgroups = n_groups >= 2  # Multiple aggregations needed?
    regex_to = create.res_columns if isinstance(create.res_columns, str) else None
    kmax = create.k
    na_vals = create.na_vals if create.na_vals is not None else []

    if has_subgroups:
        # collect all subgroups, later aggregate each subgroup separately
        def _get_subgroup_id(column: str) -> tuple[str, ...]:
            """Discard the aggregation index and collect the rest as identifier."""
            match = regex_from.match(column)
            assert match is not None, f"Column {column} should match regex {regex_from.pattern}"
            subgroup_id = list(match.groups())
            subgroup_id.pop(agg_ind)
            return tuple(subgroup_id)

        subgroups_ids = dict.fromkeys(map(_get_subgroup_id, from_cols))
        subgroups = [
            [col for col in from_cols if _get_subgroup_id(col) == subgroup_id] for subgroup_id in subgroups_ids
        ]
    else:
        subgroups = [from_cols]
    topk_dfs, subgroup_metas = [], []

    # select group at agg_ind in col name to allow translate if spec-d in scale
    # e.g. {A_11: selected} |-> {11: selected}, later by using mask |-> {11: 11}
    # this fun is def-d in current fun, so agg_ind acts as global var
    def _get_regex_group_at_agg_ind(s: str) -> str:
        """Get regex group at aggregation index."""
        match = regex_from.match(s)
        assert match is not None, f"Column {s} should match regex {regex_from.pattern}"
        return match.groups()[agg_ind]

    for subgroup in subgroups:
        sdf = df[subgroup].astype("object").replace(na_vals, None)

        def _expand_col(col: str) -> str:
            match = regex_from.match(col)
            assert match is not None, f"Column {col} should match regex {regex_from.pattern}"
            return match.expand(regex_to)  # type: ignore[call-overload]

        newcols = [
            # from_cols names map to res_cols names
            # note regex groups stay the same: e.g A_11 |-> A_R11
            _expand_col(col)
            for col in sdf.columns
        ]

        # Convert one-hot encoded columns into a list-of-selected format
        if "|" in from_columns:
            raise ValueError(sdf.columns)
        sdf.columns = sdf.columns.map(_get_regex_group_at_agg_ind)

        sdf = sdf.mask(
            ~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1
        )  # replace cell with column name where not NA
        _throw_vals_left(sdf)  # changes df in place, Nones go to rightmost side
        sdf.columns = newcols  # set column names per the regex_to template

        if "|" in from_columns:
            raise NotImplementedError("Regex symbol `|` is not supported yet.")

        sdf = sdf.dropna(axis=1, how="all")  # drop rightmost cols that are all NA
        # Handle kmax: if it's "max" or None, don't limit; otherwise limit to that number
        if kmax and kmax != "max" and isinstance(kmax, int):
            sdf = sdf.iloc[:, :kmax]  # up to kmax columns if spec-d
        sname = name
        if has_subgroups:
            sname += "_" + "_".join(map(str, _get_subgroup_id(subgroup[0])))
        meta_subgroup = {
            "name": sname,
            "scale": deepcopy(block_with_create.scale.model_dump(mode="python") if block_with_create.scale else {}),
            "columns": sdf.columns.tolist(),
        }
        if create.translate_after is not None:
            sdf = sdf.replace(create.translate_after)
        meta_subgroup = soft_validate(meta_subgroup, ColumnBlockMeta)
        topk_dfs.append(sdf)
        subgroup_metas.append(meta_subgroup)
    return topk_dfs, subgroup_metas  # note: each df has one meta for zip later


def _create_topk_metas_and_dfs_list(
    df: pd.DataFrame,
    block_with_create: ColumnBlockMeta,
    create: TopKBlock,
    name: str,
) -> tuple[list[pd.DataFrame], list[ColumnBlockMeta]]:
    """Create top K aggregations from DataFrame using explicit column lists."""
    from_columns = create.from_columns
    assert isinstance(from_columns, list), "from_columns must be list for list mode"

    if not isinstance(create.res_columns, list):
        raise ValueError("res_columns must be a list when from_columns is a list")

    if len(from_columns) != len(create.res_columns):
        raise ValueError(
            f"from_columns ({len(from_columns)}) and res_columns ({len(create.res_columns)}) must have the same length"
        )

    from_cols = from_columns
    res_cols = create.res_columns
    kmax = create.k
    na_vals = create.na_vals if create.na_vals is not None else []
    if create.from_prefix:
        without_prefix = {col: col.removeprefix(create.from_prefix) for col in from_cols}
        from_cols = [without_prefix.get(col, col) for col in from_cols]
        df.columns = [without_prefix.get(col, col) for col in df.columns]

    sdf = df[from_cols].replace(na_vals, None)
    sdf = sdf.mask(
        ~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1
    )  # replace cell with column name where not NA

    if create.translate_after is not None:
        sdf = sdf.replace(create.translate_after)
    sdf.columns = res_cols

    _throw_vals_left(sdf)  # changes df in place, Nones go to rightmost side
    sdf = sdf.dropna(axis=1, how="all")  # drop rightmost cols that are all NA

    # Handle kmax: if it's "max" or None, don't limit; otherwise limit to that number
    if kmax and kmax != "max" and isinstance(kmax, int):
        sdf = sdf.iloc[:, :kmax]  # up to kmax columns if spec-d

    meta_subgroup = {
        "name": name,
        "scale": deepcopy(block_with_create.scale.model_dump(mode="python") if block_with_create.scale else {}),
        "columns": sdf.columns.tolist(),
    }
    meta_subgroup = soft_validate(meta_subgroup, ColumnBlockMeta)

    return [sdf], [meta_subgroup]


create_block_type_to_create_fn = {
    "topk": _create_topk_metas_and_dfs,
    "maxdiff": NotImplementedError("Maxdiff not implemented yet"),
}


def _create_new_columns_and_metas(
    df: pd.DataFrame, group: ColumnBlockMeta
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Create new columns and metadata from a group definition."""
    if group.create is None:
        raise ValueError("Group must have a create block")

    create_type = group.create.type
    if create_type not in create_block_type_to_create_fn:
        raise NotImplementedError(f"Create block {create_type} not supported")
    dfs, metas = create_block_type_to_create_fn[create_type](df, group)
    return zip(dfs, metas)


# Default usage with mature metafile: _process_annotated_data(<metafile name>)
# When figuring out the metafile, it can also be run as: _process_annotated_data(meta=<dict>, data_file=<>)


# Type annotations to know the return type for return_meta=True/False
@overload
def _process_annotated_data(
    meta_fname: str | None = ...,
    meta: DataMeta | dict[str, object] | None = ...,
    data_file: str | None = ...,
    raw_data: pd.DataFrame | None = ...,
    *,
    return_meta: Literal[True],
    ignore_exclusions: bool = ...,
    only_fix_categories: bool = ...,
    return_raw: bool = ...,
    add_original_inds: bool = ...,
) -> ProcessedDataReturn: ...


@overload
def _process_annotated_data(
    meta_fname: str | None = ...,
    meta: DataMeta | dict[str, object] | None = ...,
    data_file: str | None = ...,
    raw_data: pd.DataFrame | None = ...,
    return_meta: Literal[False] = False,
    ignore_exclusions: bool = ...,
    only_fix_categories: bool = ...,
    return_raw: bool = ...,
    add_original_inds: bool = ...,
) -> pd.DataFrame: ...


def _process_annotated_data(
    meta_fname: str | None = None,
    meta: DataMeta | dict[str, object] | None = None,
    data_file: str | None = None,
    raw_data: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    return_meta: bool = False,
    ignore_exclusions: bool = False,
    only_fix_categories: bool = False,
    return_raw: bool = False,
    add_original_inds: bool = False,
) -> pd.DataFrame | ProcessedDataReturn:
    """Process annotated data according to metadata specifications."""
    # Read metafile
    metafile = cast(dict[str, str], stk_file_map).get(meta_fname, meta_fname)  # type: ignore[call-overload]
    meta_input: DataMeta | dict[str, object] | None = meta
    if meta_fname is not None:
        ext = os.path.splitext(metafile)[1]
        if ext == ".yaml":
            meta_raw = read_yaml(metafile)
        elif ext == ".json":
            meta_raw = read_json(metafile)
        else:
            raise Exception(f"Unknown meta file format {ext} for file: {meta_fname}")
        assert isinstance(meta_raw, dict), "Meta file must contain a dict"
        meta_input = dict(meta_raw)  # Cast to ensure object values

    # Soft-validate and work with Pydantic DataMeta object throughout
    if meta_input is None:
        raise ValueError("Metadata cannot be None")
    meta_obj = soft_validate(meta_input, DataMeta, warnings=True)
    constants: dict[str, object] = dict(meta_obj.constants)
    # Now meta is guaranteed to be a DataMeta object, not None

    # Read datafile(s) - now returns dict[str, pd.DataFrame] for per-file processing
    # Handle data_file override or ensure files list is populated
    raw_data_dict: dict[str, pd.DataFrame] | None = None
    if raw_data is None:
        if data_file is not None:
            # data_file override: use it directly
            files_list = [FileDesc(file=data_file, opts=meta_obj.read_opts)]
        elif meta_obj.files is not None:
            files_list = meta_obj.files
        else:
            raise ValueError("No files provided in metadata")

        raw_data_dict, inp_meta, einfo = _load_data_files(
            files_list,
            path=meta_fname if meta_fname is not None else (data_file if data_file is not None else None),
            read_opts=meta_obj.read_opts,
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )
        if inp_meta is not None:
            warn("Processing main meta file")  # Print this to separate warnings for input jsons from main
    elif isinstance(raw_data, dict):
        raw_data_dict = raw_data
        einfo = {}
    else:
        # Backward compatibility: single DataFrame -> treat as single-file dict
        raw_data_dict = {"F0": raw_data}
        einfo = {}

    if return_raw:
        # Return concatenated for backward compatibility
        raw_data_concat = pd.concat(raw_data_dict.values()) if raw_data_dict else pd.DataFrame()
        if return_meta:
            return (raw_data_concat, meta_obj)
        return raw_data_concat

    # Run preprocessing per file
    if meta_obj.preprocessing is not None and not only_fix_categories:
        for file_code, df in raw_data_dict.items():
            globs = {
                "pd": pd,
                "np": np,
                "sp": sp,
                "stk": stk,
                "df": df,
                "file_code": file_code,
                **einfo,
                **constants,
            }
            exec(_str_from_list(meta_obj.preprocessing), globs)
            raw_data_dict[file_code] = globs["df"]

    # Initialize concatenated DataFrame - start empty, will be built column by column
    ndf_df = pd.DataFrame()

    # Compute and store row index ranges for each file_code
    file_index_ranges: dict[str, slice] = {}
    current_idx = 0
    for file_code, file_raw in raw_data_dict.items():
        file_len = len(file_raw)
        file_index_ranges[file_code] = slice(current_idx, current_idx + file_len)
        current_idx += file_len

    # Add file_code column if multiple files
    if len(raw_data_dict) > 1:
        file_code_series_list = []
        for file_code, file_raw in raw_data_dict.items():
            file_code_series_list.append(pd.Series(file_code, index=file_raw.index, name="file_code"))
        ndf_df = ndf_df.assign(file_code=pd.concat(file_code_series_list).reset_index(drop=True))

    all_cns = dict()
    # Build new structure dict as we process groups (Pydantic models are immutable)
    new_structure: dict[str, ColumnBlockMeta] = {}
    for group_name, group in meta_obj.structure.items():
        if group.name in all_cns:
            raise Exception(f"Group name {group.name} duplicates a column name in group {all_cns[group.name]}")
        all_cns[group.name] = group.name
        g_cols = []

        # Process columns in this group
        # group.columns is Dict[str, ColumnMeta]
        # Note: scale metadata is already merged with column metadata
        # by ColumnBlockMeta.merge_scale_with_columns validator
        # Keep for forward-compat; metadata categories are reconciled later by _fix_meta_categories.
        # (We do not remove missing columns from meta, we only skip materializing them into df.)
        updated_columns: dict[str, ColumnMeta] = {}
        for orig_cn, col_meta in group.columns.items():
            # Work with Pydantic objects directly
            # Get the merged column metadata (scale already merged in during validation)
            mcm = col_meta
            # Extract source from ColumnMeta - can be str or dict[str, str]
            source_spec = col_meta.source if col_meta.source is not None else orig_cn
            scale_meta = group.scale if group.scale is not None else None

            # Col prefix is used to avoid name clashes when different groups naturally share same column names
            # col_prefix is only in BlockScaleMeta, not ColumnMeta, so get it from scale_meta directly
            col_prefix = scale_meta.col_prefix if scale_meta is not None and scale_meta.col_prefix is not None else None
            cn = orig_cn
            if col_prefix is not None:
                cn = col_prefix + orig_cn

            # Detect duplicate columns in meta - including among those missing or generated
            # Only flag if they are duplicates even after prefix
            if cn in all_cns:
                raise Exception(f"Duplicate column name found: '{cn}' in {all_cns[cn]} and {group.name}")
            all_cns[cn] = group.name

            if only_fix_categories:
                source_spec = cn

            # Extract columns from all files and concatenate immediately
            per_file_series: list[pd.Series] = []
            for file_code, file_raw_data in raw_data_dict.items():
                # Determine source column name for this file
                if isinstance(source_spec, dict):
                    # Dict mapping: use file code, fallback to 'default', or use orig_cn
                    sn = source_spec.get(file_code, source_spec.get("default", orig_cn))
                else:
                    # String: use for all files
                    sn = source_spec

                if sn not in file_raw_data:
                    if not group.generated:  # bypass warning for columns marked as being generated later
                        warn(f"Column {sn} not found in file {file_code}")
                    # Create empty series with same index as file
                    s = pd.Series(index=file_raw_data.index, dtype=object, name=cn)
                    per_file_series.append(s)
                    continue

                if file_raw_data[sn].isna().all():
                    warn(f"Column {sn} is empty in file {file_code} and thus ignored")
                    # Create empty series with same index as file
                    s = pd.Series(index=file_raw_data.index, dtype=object, name=cn)
                    per_file_series.append(s)
                    continue

                # Extract raw column value - no processing yet
                s = file_raw_data[sn].copy()
                s.name = cn  # Set name early
                per_file_series.append(s)

            # Concatenate all per-file series immediately - now process on concatenated data
            if not per_file_series:
                # No valid series found, skip this column
                continue

            # Concatenated series ready for processing
            s: pd.Series = pd.concat(per_file_series).reset_index(drop=True)

            # Declared in meta but missing/empty in all source files -> drop from df and meta.
            if s.isna().all():
                continue

            # Store original dtype info if categorical (before any conversions)
            original_dtype = s.dtype
            if not only_fix_categories and not _is_series_of_lists(s):
                if original_dtype.name == "category":
                    s = s.astype("object")  # This makes it easier to use common ops like replace and fillna
                if mcm.translate:
                    s = s.astype("str").replace(mcm.translate).replace("nan", None).replace("None", None)
                if mcm.transform is not None and isinstance(mcm.transform, str):
                    # For transform, process per-file using index ranges
                    transformed_parts: list[pd.Series] = []
                    for file_code, file_raw_data in raw_data_dict.items():
                        idx_range = file_index_ranges[file_code]
                        s_local = s.iloc[idx_range]
                        ndf_local = ndf_df.iloc[idx_range]
                        transformed = eval(
                            mcm.transform,
                            {
                                "s": s_local,
                                "df": file_raw_data,  # Per-file raw data
                                "ndf": ndf_local,  # Per-file processed data view
                                "pd": pd,
                                "np": np,
                                "stk": stk,
                                **constants,
                            },
                        )

                        s_out = pd.Series(transformed, index=s_local.index, name=cn)
                        transformed_parts.append(s_out)
                    s = pd.concat(transformed_parts, ignore_index=True)
                if mcm.translate_after:
                    s = (
                        pd.Series(s)
                        .astype("str")
                        .replace(mcm.translate_after)
                        .replace("nan", None)
                        .replace("None", None)
                    )

                if mcm.datetime:
                    s = pd.to_datetime(s, errors="coerce")
                elif mcm.continuous:
                    s = pd.to_numeric(s, errors="coerce")

            s.name = cn  # Ensure name is set

            # Category inference on concatenated data
            if mcm.categories and not _is_series_of_lists(s):
                if mcm.categories == "infer":
                    # Check for ordered warning before we modify s
                    should_warn_ordered = mcm.ordered and not pd.api.types.is_numeric_dtype(s)
                    # Infer categories from the transformed data (s has already gone through
                    # translate -> transform -> translate_after pipeline)
                    # Always use s as the source of truth - it has the final transformed values
                    if mcm.translate and not mcm.transform and set(mcm.translate.values()) >= set(s.dropna().unique()):
                        # Infer order from translation dict
                        translate_values = list(mcm.translate.values())
                        cats = [str(c) for c in translate_values if c in s.dropna().unique()]
                        # As mapping can be many-to-one, we need to use unique and preserve order
                        inferred_cats = list(dict.fromkeys(cats))
                        # Update metadata with inferred categories (preserves translation dict order)
                        mcm = mcm.model_copy(update={"categories": inferred_cats})
                    else:
                        # Deterministic ordering:
                        # - numeric dtype -> numeric sort
                        # - numeric-like strings -> numeric sort
                        # - otherwise -> lexicographic sort
                        if should_warn_ordered:
                            warn(
                                f"Ordered category {cn} had category: infer. "
                                "This only works correctly if you want lexicographic ordering!"
                            )
                        s, inferred_cats = _deterministic_categories_and_values(s)
                        # Update metadata with inferred categories
                        mcm = mcm.model_copy(update={"categories": inferred_cats})
                    cats = inferred_cats
                elif pd.api.types.is_numeric_dtype(s):
                    # Numeric datatype being coerced into categorical - map to nearest category value
                    cats = mcm.categories
                    if not isinstance(cats, list):
                        raise ValueError(f"Categories for {cn} must be a list when series is numeric")

                    try:
                        fcats = np.array(cats).astype(float)
                        s_values_arr = np.asarray(s.values)
                        if s_values_arr.ndim == 0:
                            s_values_arr = s_values_arr.reshape(-1)
                        cats_array = np.array(cats)
                        s_values_reshaped = s_values_arr.reshape(-1, 1)
                        fcats_reshaped = fcats.reshape(1, -1)
                        distances = np.abs(s_values_reshaped - fcats_reshaped)
                        indices = distances.argmin(axis=1)
                        s = pd.Series(
                            cats_array[indices],
                            index=s.index,
                            name=s.name,
                            dtype=pd.CategoricalDtype(
                                categories=cats, ordered=mcm.ordered if mcm.ordered is not None else False
                            ),
                        )
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Categories for {cn} are not numeric: {cats}") from e
                else:
                    cats = mcm.categories
                    if not isinstance(cats, list):
                        raise ValueError(f"Categories for {cn} must be a list")

                if isinstance(cats, list):
                    # Use updated ordered flag if we modified it during inference
                    final_ordered = mcm.ordered if mcm.ordered is not None else False
                    s = pd.Series(
                        pd.Categorical(s, categories=cats, ordered=final_ordered),
                        name=cn,
                    )

            # Add column directly to concatenated DataFrame
            ndf_df[cn] = s
            g_cols.append(cn)

            # Store updated column metadata (with inferred categories if applicable)
            updated_columns[orig_cn] = mcm

        if group.subgroup_transform is not None:
            # subgroups is not in ColumnBlockMeta, so we'll use g_cols as default
            subgroups = [g_cols]  # TODO: Add subgroups field to ColumnBlockMeta if needed
            for sg in subgroups:
                # Apply per-file using index ranges
                subgroup_transformed_parts: list[pd.DataFrame] = []
                for file_code in raw_data_dict.keys():
                    idx_range = file_index_ranges[file_code]
                    gdf_local = ndf_df[sg].iloc[idx_range].copy()
                    ndf_local = ndf_df.iloc[idx_range].copy()
                    transformed_gdf = eval(
                        group.subgroup_transform,
                        {
                            "gdf": gdf_local,
                            "ndf": ndf_local,  # Per-file processed data view
                            "df": file_raw_data,  # Per-file raw data
                            "pd": pd,
                            "np": np,
                            "stk": stk,
                            **constants,
                        },
                    )
                    subgroup_transformed_parts.append(transformed_gdf)
                # Concatenate and assign back
                transformed_combined = pd.concat(subgroup_transformed_parts).reset_index(drop=True)
                ndf_df[sg] = transformed_combined[sg].values
        # Handle create blocks - may add new groups to structure
        if group.create is not None:
            for newdf, newmeta_dict in _create_new_columns_and_metas(ndf_df, group):
                ndf_df = ndf_df.combine_first(newdf)
                new_group_meta = soft_validate(newmeta_dict, ColumnBlockMeta)
                new_structure[new_group_meta.name] = new_group_meta
            # Create a copy of group without the create field
            group = group.model_copy(update={"create": None})

        # Add processed group to structure
        new_structure[group.name] = group

    if meta_obj.postprocessing is not None and not only_fix_categories:
        globs = {
            "pd": pd,
            "np": np,
            "sp": sp,
            "stk": stk,
            "df": ndf_df,
            **einfo,
            **constants,
        }
        exec(_str_from_list(meta_obj.postprocessing), globs)
        ndf_df = globs["df"]

    # Update meta with new structure (including any groups created during processing)
    meta_obj = meta_obj.model_copy(update={"structure": new_structure})

    # Fix categories after postprocessing
    # Also replaces infer with the actual categories
    meta_obj = _fix_meta_categories(meta_obj, ndf_df, warnings=True)
    # Ensure we have a DataMeta object
    if not isinstance(meta_obj, DataMeta):
        meta_obj = soft_validate(meta_obj, DataMeta)

    ndf_df["original_inds"] = np.arange(len(ndf_df))
    if meta_obj.excluded and not ignore_exclusions:
        excl_inds = [i for i, _ in meta_obj.excluded]
        ndf_df = ndf_df[~ndf_df["original_inds"].isin(excl_inds)]
    if not add_original_inds:
        ndf_df.drop(columns=["original_inds"], inplace=True)

    # Return with meta as dict if requested (for backward compatibility)
    if return_meta:
        return (ndf_df, meta_obj)
    return ndf_df


@overload
def read_annotated_data(
    fname: str, infer: bool = ..., return_raw: bool = ..., *, return_meta: Literal[True], **kwargs: object
) -> ProcessedDataReturn: ...


@overload
def read_annotated_data(
    fname: str, infer: bool = ..., return_raw: bool = ..., *, return_meta: Literal[False] = False, **kwargs: object
) -> pd.DataFrame: ...


def read_annotated_data(
    fname: str, infer: bool = True, return_raw: bool = False, return_meta: bool = False, **kwargs: object
) -> pd.DataFrame | ProcessedDataReturn:
    """Read either a json annotation and process the data, or a processed parquet with the annotation attached.

    Args:
        fname: Path to data file (JSON, YAML, or Parquet).
        infer: Whether to infer metadata if not found (default: True).
        return_raw: Whether to return raw unprocessed data (for debugging).
            Return_raw is here for easier debugging of metafiles and is not meant to be used in production.
        return_meta: Whether to return metadata along with data.
        **kwargs: Additional arguments passed to processing functions.

    Returns:
        DataFrame, or tuple of (DataFrame, metadata) if return_meta=True.
    """
    _, ext = os.path.splitext(fname)
    data: pd.DataFrame | None = None
    meta_obj: DataMeta | None = None
    if ext in {".json", ".yaml"}:
        # Extract parameters from kwargs that are valid for _process_annotated_data
        ignore_exclusions = bool(kwargs.get("ignore_exclusions", False))
        only_fix_categories = bool(kwargs.get("only_fix_categories", False))
        add_original_inds = bool(kwargs.get("add_original_inds", False))
        # Pass all parameters explicitly to match overloads - return_meta=True means ProcessedDataReturn
        data, meta_obj = _process_annotated_data(
            meta_fname=fname,
            return_meta=True,
            return_raw=return_raw,
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )
    elif ext == ".parquet":
        data, full_meta = read_parquet_with_metadata(fname)
        if full_meta is not None:
            meta_obj = full_meta.data

    if meta_obj is not None or not infer:
        assert isinstance(data, pd.DataFrame), "Expected data to be DataFrame"
        if return_meta:
            return (data, meta_obj)
        return data

    warn(f"Warning: using inferred meta for {fname}")
    inferred_meta = infer_meta(fname, meta_file=False)
    # Extract parameters from kwargs that are valid for _process_annotated_data
    ignore_exclusions = bool(kwargs.get("ignore_exclusions", False))
    only_fix_categories = bool(kwargs.get("only_fix_categories", False))
    add_original_inds = bool(kwargs.get("add_original_inds", False))
    # Use conditional to match overloads based on return_meta value
    # Pass return_meta first (after *) to help Pyright match overloads
    if return_meta:
        return _process_annotated_data(
            data_file=fname,
            meta=inferred_meta,
            return_meta=True,
            return_raw=return_raw,
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )
    else:
        return _process_annotated_data(
            data_file=fname,
            meta=inferred_meta,
            return_meta=False,
            return_raw=return_raw,
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )


def fix_df_with_meta(df: pd.DataFrame, dmeta: DataMeta, exclude_cols: set[str] | None = None) -> pd.DataFrame:
    """Fix df dtypes etc using meta - needed after a lazy load.

    Args:
        df: DataFrame to fix.
        dmeta: Data metadata dictionary.
        exclude_cols: Optional set of column names to skip (useful when fixing with fallback metadata).

    Returns:
        DataFrame with corrected dtypes and categories.
    """
    if exclude_cols is None:
        exclude_cols = set()
    cmeta = extract_column_meta(dmeta)
    for c in df.columns:
        if c not in cmeta or c in exclude_cols:
            continue
        cd = cmeta[c]
        if cd.categories:
            # Ensure categorical values behave well after lazy loads:
            if cd.categories == "infer":
                s_fixed, cats = _deterministic_categories_and_values(df[c])
            else:
                s_fixed, cats = df[c].astype("str"), [str(v) for v in cd.categories]

            df[c] = pd.Series(
                pd.Categorical(s_fixed, categories=cats, ordered=bool(cd.ordered)),
                name=c,
            )
    return df


def fix_df_with_meta_fallback(
    df: pd.DataFrame,
    primary_meta: DataMeta | None,
    fallback_metas: DataMeta | Iterable[DataMeta | None] | None = None,
) -> pd.DataFrame:
    """Fix df dtypes using primary metadata, with fallback metadata for missing columns.

    For columns present in multiple metadata sources, categories are merged:
    - Primary metadata categories come first (preserving order)
    - Fallback metadata categories not already present are appended in order

    This is useful when:
    - primary_meta is from a population file (e.g., cdf_meta)
    - fallback_metas contains model/survey data metadata (e.g., data_meta)
    - Some columns exist in multiple sources but may have different category sets

    Args:
        df: DataFrame to fix.
        primary_meta: Primary metadata (typically from the population file), can be None.
        fallback_metas: Optional fallback metadata source(s) for columns not in primary.
                        Can be a single DataMeta, an iterable of DataMeta objects, or None.
                        Categories are merged in order: primary first, then each fallback.

    Returns:
        DataFrame with corrected dtypes and merged categories.
    """
    # Build list of column metadata dicts in priority order
    all_cmetas: list[dict[str, GroupOrColumnMeta]] = []

    if primary_meta:
        all_cmetas.append(extract_column_meta(primary_meta))

    if fallback_metas is not None:
        # Handle single DataMeta or iterable of DataMeta
        if isinstance(fallback_metas, DataMeta):
            all_cmetas.append(extract_column_meta(fallback_metas))
        else:
            for meta in fallback_metas:
                if meta is not None:
                    all_cmetas.append(extract_column_meta(meta))

    for c in df.columns:
        # Find the first metadata that has info for this column
        primary_cd = None
        for cmeta in all_cmetas:
            if c in cmeta:
                primary_cd = cmeta[c]
                break

        if not primary_cd or not primary_cd.categories:
            continue

        # Get base categories from the first source that has this column
        if primary_cd.categories == "infer":
            s_fixed, cats = _deterministic_categories_and_values(df[c])
        else:
            s_fixed = df[c].astype("str")
            cats = [str(v) for v in primary_cd.categories]

            # Merge categories from all fallback sources
            for cmeta in all_cmetas:
                fallback_cd = cmeta.get(c)
                if fallback_cd and fallback_cd.categories and fallback_cd.categories != "infer":
                    for cat in [str(v) for v in fallback_cd.categories]:
                        if cat not in cats:
                            cats.append(cat)

        ordered = bool(primary_cd.ordered)
        df[c] = pd.Series(pd.Categorical(s_fixed, categories=cats, ordered=ordered), name=c)

    return df


# Helper functions designed to be used with the annotations


def extract_column_meta(data_meta: DataMeta) -> dict[str, GroupOrColumnMeta]:
    """Convert data_meta into a dict where each group and column maps to their metadata dict."""

    if data_meta.structure is None:
        return {}

    res: dict[str, GroupOrColumnMeta] = {}
    for block in data_meta.structure.values():
        scale_meta = block.scale
        col_prefix = scale_meta.col_prefix if scale_meta and scale_meta.col_prefix else ""

        # Create group metadata
        group_columns = [f"{col_prefix}{cn}" for cn in block.columns.keys()]
        if scale_meta is not None:
            # Convert BlockScaleMeta to GroupOrColumnMeta by dumping and adding columns field
            group_dict = scale_meta.model_dump(mode="python")
            group_dict["columns"] = group_columns
            res[block.name] = soft_validate(group_dict, GroupOrColumnMeta)
        else:
            # No scale, create empty GroupOrColumnMeta with just columns
            res[block.name] = GroupOrColumnMeta(columns=group_columns)

        for cn, col_meta in block.columns.items():
            # Note: scale metadata is already merged with column metadata by
            # ColumnBlockMeta.merge_scale_with_columns validator
            col_name = f"{col_prefix}{cn}"
            # Convert ColumnMeta to GroupOrColumnMeta
            # If scale exists and col_meta.label is None, ensure label is None
            update_dict: dict[str, object] = {}
            if scale_meta is not None and col_meta.label is None:
                update_dict["label"] = None
            col_dict = col_meta.model_dump(mode="python")
            col_dict.update(update_dict)
            res[col_name] = soft_validate(col_dict, GroupOrColumnMeta)
    return res


# Convert data_meta into a dict of group_name -> [column names]
# TODO: deprecate - info available in extract_column_meta


def group_columns_dict(data_meta: DataMeta) -> dict[str, list[str]]:
    """Get dictionary mapping group names to their column lists."""

    return {name: list(meta.columns) for name, meta in extract_column_meta(data_meta).items() if meta.columns}

    # return { g['name'] : [(t[0] if type(t)!=str else t) for t in g['columns']] for g in data_meta['structure'] }


def list_aliases(lst: list[str], da: dict[str, list[str]]) -> list[str]:
    """Take a list and a dict and replace all dict keys in list with their corresponding lists in-place.

    Expand aliases in a list using a dictionary mapping.

    Args:
        lst: List of strings that may contain aliases.
        da: Dictionary mapping aliases to lists of expanded values.

    Returns:
        List with aliases expanded to their corresponding lists.
    """
    return [fv for v in lst for fv in (da[v] if isinstance(v, str) and v in da else [v])]


def _get_original_column_names(dmeta: DataMeta) -> dict[str, str]:
    """Get mapping of original column names (before any transformations)."""
    if dmeta.structure is None:
        return {}

    res: dict[str, str] = {}
    for block in dmeta.structure.values():
        col_prefix = block.scale.col_prefix if block.scale and block.scale.col_prefix else ""
        for cn, col_meta in block.columns.items():
            sn = col_meta.source if col_meta.source is not None else cn
            col_name = f"{col_prefix}{cn}"
            res[col_name] = sn
    return res


def _change_mapping(ot: dict[str, str], nt: dict[str, str], only_matches: bool = False) -> dict[str, str]:
    """Map ot backwards and nt forwards to move from one to the other.

    Create mapping from old translation to new translation.
    """
    # Todo: warn about non-bijective mappings
    matches = {v: nt[k] for k, v in ot.items() if k in nt and v != nt[k]}  # change those that are shared
    if only_matches:
        return matches
    else:
        return {
            **{v: k for k, v in ot.items() if k not in nt},  # undo those in ot not in nt
            **{k: v for k, v in nt.items() if k not in ot},  # do those in nt not in ot
            **matches,
        }


def _change_df_to_meta(
    df: pd.DataFrame,
    old_dmeta: DataMeta,
    new_dmeta: DataMeta,
) -> pd.DataFrame:
    """Change an existing dataset to correspond better to a new meta_data.

    This is intended to allow making small improvements in the meta even after a model has been run.
    It is by no means perfect, but is nevertheless a useful tool to avoid re-running long pymc models
    for simple column/translation changes.
    """
    warn("This tool handles only simple cases of column name, translation and category order changes.")

    # Rename columns
    ocn = _get_original_column_names(old_dmeta)
    ncn = _get_original_column_names(new_dmeta)
    shared_sources = set(ocn.values()) & set(ncn.values())
    old_source_map = {source: name for name, source in ocn.items()}
    new_source_map = {source: name for name, source in ncn.items()}
    name_changes = {
        old_source_map[source]: new_source_map[source]
        for source in shared_sources
        if old_source_map[source] != new_source_map[source]
    }
    if name_changes != {}:
        print(f"Renaming columns: {name_changes}")
    df.rename(columns=name_changes, inplace=True)

    rev_name_changes = {v: k for k, v in name_changes.items()}

    # Get metadata for each column
    ocm = extract_column_meta(old_dmeta)
    ncm = extract_column_meta(new_dmeta)

    for c, ncd in ncm.items():
        if ncd.columns is not None:
            continue  # skip group entries
        if c not in df.columns:
            continue  # probably group
        lookup_key = rev_name_changes[c] if c in rev_name_changes else c
        if lookup_key not in ocm:
            continue  # new column
        ocd = ocm[lookup_key]
        if ocd.columns is not None:
            continue

        # Warn about transformations and don't touch columns where those change
        if ocd.transform != ncd.transform:
            warn(f"Column {c} has a different transformation. Leaving it unchanged")
            continue

        # Handle translation changes
        ot = ocd.translate or {}
        nt = ncd.translate or {}
        remap = _change_mapping(ot, nt)
        if remap != {}:
            # Validate that mapping keys exist in current categories
            cat_categories = utils.get_categories(df[c].dtype)
            invalid_keys = set(remap.keys()) - {str(c) for c in cat_categories}
            if invalid_keys:
                raise ValueError(
                    f"Translation mapping keys {invalid_keys} not found in current categories "
                    f"{cat_categories} for column {c}"
                )
            print(f"Remapping {c} with {remap}")
            df[c] = df[c].cat.rename_categories(remap)

        # Reorder categories and/or change ordered status
        ncd_categories = ncd.categories
        dtype = df[c].dtype
        dtype_categories = utils.get_categories(dtype)
        dtype_ordered = utils.get_ordered(dtype)
        if (
            ncd_categories not in (None, "infer") and dtype_categories and list(dtype_categories) != ncd_categories
        ) or dtype_ordered != ncd.ordered:
            cats = ncd_categories if ncd_categories not in (None, "infer") else dtype_categories
            if isinstance(cats, list):
                print(f"Changing {c} to Cat({cats},ordered={ncd.ordered})")
                df[c] = pd.Categorical(df[c], categories=cats, ordered=ncd.ordered)

    # Column order changes
    gcdict = group_columns_dict(new_dmeta)

    cols = ["draw", "obs_idx", "training_subsample"]
    if new_dmeta.structure:
        for block in new_dmeta.structure.values():
            cols.extend(gcdict.get(block.name, []))
    cols.append(new_dmeta.weight_col if new_dmeta.weight_col else "N")
    cols = [c for c in cols if c in df.columns]

    if len(set(df.columns) - set(cols)) > 0:
        print("Dropping columns:", set(df.columns) - set(cols))

    return df[cols]


def replace_data_meta_in_parquet(parquet_name: str, metafile_name: str, advanced: bool = True) -> pd.DataFrame:
    """Replace metadata in a Parquet file with metadata from a JSON file.

    Args:
        parquet_name: Path to Parquet file.
        metafile_name: Path to JSON metadata file.
        advanced: Whether to use advanced metadata replacement.

    Returns:
        Tuple of (DataFrame, updated metadata).
    """
    df, meta = read_parquet_with_metadata(parquet_name)
    assert meta is not None, "Expected metadata to be present"

    ometa = meta.data
    nmeta_dict = replace_constants(read_json(metafile_name))
    nmeta = soft_validate(nmeta_dict, DataMeta)
    nmeta = update_meta_with_model_fields(nmeta, ometa)

    # Perform the column name changes and category translations
    # Do this before inferring meta as categories might change in this step
    if advanced:
        df = _change_df_to_meta(df, ometa, nmeta)

    nmeta = _fix_meta_categories(nmeta, df)  # replace infer with values

    # Set original_data extra field if not already present
    if meta.model_extra is None or "original_data" not in meta.model_extra:
        if meta.model_extra is None:
            meta.model_extra = {}
        meta.model_extra["original_data"] = ometa.model_dump(mode="json")
    meta.data = nmeta

    write_parquet_with_metadata(df, meta, parquet_name)

    return df


def update_meta_with_model_fields(meta: DataMeta, donor: DataMeta) -> DataMeta:
    """Local helper to update metadata with fields from a donor metadata.

    This is a local copy of the function that lives in salk_internal_package.sampling.
    We keep it here to avoid circular dependencies since io.py is used by salk_internal_package.
    """
    structure = dict(meta.structure or {})
    if donor.structure:
        for name, block in donor.structure.items():
            if block.generated and name not in structure:
                structure[name] = block

    updates: dict[str, object] = {"structure": structure}
    updates["draws_data"] = donor.draws_data or meta.draws_data or {}
    if donor.total_size is not None:
        updates["total_size"] = donor.total_size
    if donor.weight_col is not None:
        updates["weight_col"] = donor.weight_col

    return meta.model_copy(update=updates)


def _fix_meta_categories(
    data_meta: DataMeta,
    df: pd.DataFrame,
    infers_only: bool = False,
    warnings: bool = True,
) -> DataMeta:
    """Infer categories (and validate the ones already present) in metadata based on DataFrame, working in-place."""
    if data_meta.structure is None:
        return data_meta

    # Work with Pydantic structure (dict of ColumnBlockMeta objects)
    structure = data_meta.structure
    # Build new structure dict with updated blocks
    new_structure: dict[str, ColumnBlockMeta] = {}

    for g in structure.values():
        all_cats = set()
        # Work with Pydantic object directly
        scale_meta = g.scale
        prefix = scale_meta.col_prefix if scale_meta is not None and scale_meta.col_prefix is not None else ""
        columns_raw = g.columns

        # Ensure prefix is a string
        prefix = prefix if prefix is not None else ""

        # Update column metadata in Pydantic objects
        updated_columns: dict[str, ColumnMeta] = {}
        for cn, col_meta in columns_raw.items():
            updated_col_meta = col_meta
            full_col_name = prefix + cn
            if full_col_name in df.columns and df[full_col_name].dtype.name == "category":
                dtype = df[full_col_name].dtype
                cats = utils.get_categories(dtype)
                if col_meta.categories == "infer":
                    # Categories were already inferred in lexicographic/numeric order during processing
                    # Use the order from the DataFrame's categorical dtype (which should be deterministic)
                    # This preserves the lexicographic/numeric order that was set during inference
                    updated_col_meta = col_meta.model_copy(update={"categories": cats})
                elif (
                    (not infers_only)
                    and col_meta.categories is not None
                    and isinstance(col_meta.categories, list)
                    and not set(col_meta.categories) >= set(cats)
                ):
                    # Missing categories - add them while preserving the original order
                    diff = set(cats) - set(col_meta.categories)
                    if warnings:
                        warn(f"Fixing missing categories for {cn}: {diff}")
                    # Preserve original order and append missing categories at the end
                    existing_cats = list(col_meta.categories)
                    # Preserve the observed dtype category order (do NOT sort).
                    missing_cats = [c for c in cats if c not in existing_cats]
                    updated_col_meta = col_meta.model_copy(update={"categories": existing_cats + missing_cats})
                all_cats |= set(cats)
            updated_columns[cn] = updated_col_meta

        # Get scale for category inference - work with Pydantic object directly
        scale_meta = g.scale
        updated_scale = scale_meta
        if scale_meta is not None and scale_meta.categories == "infer":
            # IF they all share same categories, keep the category order
            tr = scale_meta.translate or {}
            if all_cats == set(tr.values()):  # First prefer translate order
                scats = list(dict.fromkeys(tr.values()))
            elif all_cats == set(cats):  # Then prefer order from single col
                scats = list(cats)
            else:  # Otherwise, sort categories
                scats = sorted(list(all_cats))
            # Update scale categories using model_copy
            updated_scale = scale_meta.model_copy(update={"categories": scats})
        elif (
            (not infers_only)
            and scale_meta is not None
            and scale_meta.categories is not None
            and isinstance(scale_meta.categories, list)
            and not set(scale_meta.categories) >= all_cats
        ):
            diff = all_cats - set(scale_meta.categories)
            if warnings:
                warn(f"Fixing missing categories for group {g.name}: {diff}")
            updated_scale = scale_meta.model_copy(update={"categories": list(all_cats)})

        # Update the block with new scale and columns if they changed
        update_dict: dict[str, object] = {}
        if updated_scale != scale_meta:
            update_dict["scale"] = updated_scale
        if updated_columns != columns_raw:
            update_dict["columns"] = updated_columns
        if update_dict:
            g = g.model_copy(update=update_dict)

        new_structure[g.name] = g

    # Return updated DataMeta with new structure
    return data_meta.model_copy(update={"structure": new_structure})


def _fix_parquet_categories(parquet_name: str) -> None:
    """Fix categories in a Parquet file by reading, fixing, and rewriting."""
    df, meta = read_parquet_with_metadata(parquet_name)
    if meta is None:
        raise ValueError(f"Parquet file {parquet_name} has no metadata")
    meta.data = _fix_meta_categories(meta.data, df, infers_only=False)
    write_parquet_with_metadata(df, meta, parquet_name)


def _is_categorical(col: pd.Series) -> bool:
    """Check if a pandas Series is categorical."""
    return col.dtype.name in ["object", "str", "category"] and not is_datetime(col)


max_cats = 50


def infer_meta(
    data_file: str | None = None,
    meta_file: bool | str = True,
    read_opts: dict[str, object] | None = None,
    df: pd.DataFrame | None = None,
    translate_fn: Callable[[str], str] | None = None,
    translation_blacklist: list[str] | None = None,
) -> dict[str, object]:
    """Create a very basic metafile for a dataset based on its contents.

    This is not meant to be directly used, rather to speed up the annotation process.
    Creates a basic metadata file based on data contents to speed up annotation.

    Args:
        data_file: Path to data file to analyze.
        meta_file: Whether to write metadata file (True/False) or path to write to.
        read_opts: Options passed to file reader.
        df: Pre-loaded DataFrame (alternative to data_file).
        translate_fn: Function to translate column/category names.
        translation_blacklist: List of column names to skip translation.

    Returns:
        Inferred metadata dictionary.
    """
    if read_opts is None:
        read_opts = {}
    if translation_blacklist is None:
        translation_blacklist = []
    meta = {"constants": {}, "read_opts": read_opts}

    if translate_fn is not None:
        otfn = translate_fn
        translate_fn = cached_fn(lambda x: otfn(str(x)) if x else "")
    else:
        translate_fn = str

    # Read datafile
    col_labels = {}
    if data_file is not None:
        path, fname = os.path.split(data_file)
        ext = os.path.splitext(fname)[1].lower()[1:]
        meta["files"] = [{"file": fname, "opts": read_opts, "code": "F0"}]
        if ext in ["csv", "gz"]:
            df = pd.read_csv(data_file, low_memory=False, **read_opts)  # type: ignore[call-overload]
        elif ext in ["sav", "dta"]:
            read_fn = getattr(pyreadstat, "read_" + ext)
            df, sav_meta = read_fn(
                data_file,
                **{"apply_value_formats": True, "dates_as_pandas_datetime": True},
                **read_opts,
            )
            col_labels = dict(
                zip(sav_meta.column_names, sav_meta.column_labels)
            )  # Make this data easy to access by putting it in meta as constant
            if translate_fn:
                col_labels = {k: translate_fn(v) for k, v in col_labels.items()}
        elif ext == "parquet":
            df = pd.read_parquet(data_file, **read_opts)
        elif ext in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
            df = pd.read_excel(data_file, **read_opts)  # type: ignore[call-overload]
        else:
            raise Exception(f"Not a known file format {data_file}")

    if df is None:
        raise ValueError("Either data_file or df must be provided")

    # If data is multi-indexed, flatten the index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(tpl) for tpl in df.columns]

    cats, grps = {}, defaultdict(lambda: list())

    main_grp = {"name": "main", "columns": []}
    meta["structure"] = [main_grp]

    # Remove empty columns
    cols = [c for c in df.columns if df[c].notna().any()]

    # Determine category lists for all categories
    for cn in cols:
        if not _is_categorical(df[cn]):
            continue
        dtype = df[cn].dtype
        dtype_categories = utils.get_categories(dtype)
        cats[cn] = sorted(list(df[cn].dropna().unique())) if df[cn].dtype.name != "category" else dtype_categories

        for cs in grps:
            # if cn.startswith('Q2_'): print(len(set(cats[cn]) & cs)/len(cs),set(cats[cn]),cs)
            if len(set(cats[cn]) & cs) / len(cs) > 0.75:  # match to group if most of the values match
                key = frozenset(cs | set(cats[cn]))
                if key in grps:
                    cs = key  # Check if we already have this exact key (can happen)
                lst = grps[cs]
                del grps[cs]
                grps[key] = lst + [cn]
                break
        else:
            grps[frozenset(cats[cn])].append(cn)

    # Fn to create the meta for a categorical column
    def _cat_meta(cn: str) -> dict[str, object]:
        """Create metadata for a categorical column."""
        m: dict[str, object] = {"categories": cats[cn] if len(cats[cn]) <= max_cats else "infer"}
        if cn in df.columns and df[cn].dtype.name == "category" and df[cn].dtype.ordered:
            m["ordered"] = True
        if translate_fn is not None and cn not in translation_blacklist and len(cats[cn]) <= max_cats:
            categories = cast(list[str], m["categories"])
            tdict = {c: translate_fn(c) for c in categories}
            m["categories"] = "infer"  # [ tdict[c] for c in m['categories'] ]
            m["translate"] = tdict
        return m

    # Create groups from values that share a category
    handled_cols = set()
    for k, g_cols in grps.items():
        if len(g_cols) < 2:
            continue

        # Set up the columns part
        m_cols = []
        for cn in g_cols:
            ce = [cn, {"label": col_labels[cn]}] if cn in col_labels else [cn]
            if translate_fn is not None:
                ce = [translate_fn(cn)] + ce
            if len(ce) == 1:
                ce = ce[0]
            m_cols.append(ce)

        kl = [str(c) for c in k]
        cats[str(kl)] = kl  # so cat_meta would use the full list

        grp = {"name": ";".join(kl), "scale": _cat_meta(str(kl)), "columns": m_cols}

        meta["structure"].append(grp)
        handled_cols.update(g_cols)

    # Put the rest of variables into main category
    main_cols = [c for c in cols if c not in handled_cols]
    for cn in main_cols:
        if cn in cats:
            cdesc = _cat_meta(cn)
        else:
            if is_datetime(df[cn]):
                cdesc = {"datetime": True}
            else:
                cdesc = {"continuous": True}
        if cn in col_labels:
            cdesc["label"] = col_labels[cn]
        main_grp["columns"].append([cn, cdesc] if translate_fn is None else [translate_fn(cn), cn, cdesc])

    # print(json.dumps(meta,indent=2,ensure_ascii=False))

    # Write file to disk
    if data_file is not None and meta_file:
        if meta_file is True:
            meta_file = os.path.join(path, os.path.splitext(fname)[0] + "_meta.json")
        if not os.path.exists(meta_file):
            print(f"Writing {meta_file} to disk")
            with open(meta_file, "w", encoding="utf8") as jf:
                json.dump(meta, jf, indent=2, ensure_ascii=False)
        else:
            print(f"{meta_file} already exists, skipping write")

    return meta


def _data_with_inferred_meta(data_file: str, **kwargs: object) -> tuple[pd.DataFrame, DataMeta]:
    """Read data file and infer metadata if not present."""
    meta = infer_meta(data_file, meta_file=False, **kwargs)
    df, meta_result = _process_annotated_data(meta=meta, data_file=data_file, return_meta=True)  # type: ignore[call-overload]
    assert meta_result is not None, "Expected metadata to be present"
    return df, meta_result


def _perform_merges(
    df: pd.DataFrame,
    merges: SingleMergeSpec | list[SingleMergeSpec],
    constants: dict[str, object] | None = None,
    data_meta: DataMeta | None = None,
) -> pd.DataFrame:
    """Perform merge operations on a DataFrame.

    Args:
        df: DataFrame to merge into.
        merges: Merge specification(s).
        constants: Constants for path substitution.
        data_meta: Optional metadata to apply categorical conversions to merged columns.

    Returns:
        Merged DataFrame with categorical conversions applied based on metadata.
    """
    if constants is None:
        constants = {}
    if not isinstance(merges, list):
        merges = [merges]

    for ms in merges:
        # ms is always a SingleMergeSpec Pydantic model
        file = ms.file
        on = ms.on if isinstance(ms.on, list) else [ms.on]
        add = ms.add
        how = ms.how

        ndf = read_and_process_data(file, constants=constants)
        if add:
            ms_on = on
            ms_add = list(add) if isinstance(add, list) else [add]
            ndf = ndf[ms_on + ms_add]
        overlap = (set(df.columns) & set(ndf.columns)) - set(on)
        if overlap:
            cols = ", ".join(sorted(overlap))
            raise ValueError(
                f"Merge would suffix overlapping columns ({cols}). "
                "Fix by dropping/renaming columns or using merge.add to select only non-overlapping columns."
            )
        # print(df.columns,ndf.columns,ms.on)
        if data_meta is not None:
            ndf = fix_df_with_meta(ndf, data_meta)
        mdf = pd.merge(df, ndf, on=on, how=how)

        for c in on:
            mdf[c] = mdf[c].astype(df[c].dtype)
        if len(df) != len(mdf):
            missing = set(list(df[on].drop_duplicates().itertuples(index=False, name=None))) - set(
                list(ndf[on].drop_duplicates().itertuples(index=False, name=None))
            )
            file_str = file if isinstance(file, str) else str(file)
            warn(f"Merge with {file_str} removes {1 - len(mdf) / len(df):.1%} rows with missing merges on: {missing}")

        df = mdf
    return df


@overload
def read_and_process_data(
    desc: str | dict[str, Any] | DataDescription,
    return_meta: Literal[False] = False,
    constants: dict[str, object] | None = ...,
    skip_postprocessing: bool = ...,
    **kwargs: object,
) -> pd.DataFrame: ...


@overload
def read_and_process_data(
    desc: str | dict[str, Any] | DataDescription,
    return_meta: Literal[True],
    constants: dict[str, object] | None = ...,
    skip_postprocessing: bool = ...,
    **kwargs: object,
) -> tuple[pd.DataFrame, DataMeta]: ...


def read_and_process_data(
    desc: str | dict[str, Any] | DataDescription,
    return_meta: bool = False,
    constants: dict[str, object] | None = None,
    skip_postprocessing: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, DataMeta]:
    """Read and process data according to a description object.

    Args:
        desc: Data description (string file path, dict, or DataDescription object).
        return_meta: Whether to return metadata along with data.
        constants: Dictionary of constants for preprocessing/postprocessing.
        skip_postprocessing: Whether to skip postprocessing step.
        **kwargs: Additional arguments passed to file readers.

    Returns:
        DataFrame, or tuple of (DataFrame, metadata) if return_meta=True.
    """
    if constants is None:
        constants = {}
    if isinstance(desc, str):
        desc_obj = DataDescription(
            files=[{"file": desc, "opts": {}, "code": "F0"}]
        )  # Allow easy shorthand for simple cases
    elif isinstance(desc, dict):
        desc_obj = soft_validate(desc, DataDescription, warnings=True)  # Convert dict to DataDescription
    else:
        desc_obj = desc

    meta_obj: DataMeta | None
    # Check if desc_obj is DataDescription with inline data
    if isinstance(desc_obj, DataDescription) and desc_obj.data is not None:
        df, meta_obj, einfo = pd.DataFrame(data=desc_obj.data), None, {}
    else:
        # Extract parameters from kwargs
        ignore_exclusions = kwargs.get("ignore_exclusions", False)
        only_fix_categories = kwargs.get("only_fix_categories", False)
        add_original_inds = kwargs.get("add_original_inds", False)

        # Get files list from description
        if isinstance(desc_obj, DataDescription) and desc_obj.files:
            files_list = desc_obj.files
        else:
            raise ValueError("No files provided in DataDescription")

        # Load files directly
        raw_data_dict, meta_raw, einfo = _load_data_files(
            files_list,
            path=None,
            read_opts={},
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )

        # Reconcile categories across files (only for read_and_process_data)
        reconciled_dtypes = _reconcile_categories(raw_data_dict, None)

        # Apply reconciled dtypes to each file's dataframe
        for file_code, df in raw_data_dict.items():
            for c, dtype in reconciled_dtypes.items():
                if c in df.columns:
                    df[c] = pd.Categorical(df[c], dtype=dtype)

        # Concatenate for backward compatibility
        df = pd.concat(raw_data_dict.values())
        if meta_raw is None:
            meta_obj = None
        elif isinstance(meta_raw, DataMeta):
            meta_obj = meta_raw
        else:
            meta_obj = soft_validate(meta_raw, DataMeta, warnings=True)

        # Ensure returned metadata categories reflect reconciled categoricals (including injected extra fields).
        if meta_obj is not None:
            meta_obj = _fix_meta_categories(meta_obj, df, warnings=False)

    if meta_obj is None and return_meta:
        raise Exception("No meta found on any of the files")

    # Perform transformation and filtering (only for DataDescription)
    if isinstance(desc_obj, DataDescription):
        globs = {"pd": pd, "np": np, "sp": sp, "stk": stk, "df": df, **einfo, **constants}
        if desc_obj.preprocessing:
            exec(_str_from_list(desc_obj.preprocessing), globs)

        if desc_obj.filter:
            globs["df"] = globs["df"][eval(desc_obj.filter, globs)]
        if desc_obj.merge:
            globs["df"] = _perform_merges(globs["df"], desc_obj.merge, constants, meta_obj)
        if desc_obj.postprocessing and not skip_postprocessing:
            exec(_str_from_list(desc_obj.postprocessing), globs)
        df = globs["df"]

    if return_meta:
        assert meta_obj is not None, "Meta should not be None when return_meta=True"
        return df, meta_obj
    return df


def _find_type_in_dict(d: object, dtype: type, path: str = "") -> None:
    """Find values of a specific type in a nested dictionary to debug non-serializable JSONs."""
    print(d, path)
    if isinstance(d, dict):
        for k, v in d.items():
            _find_type_in_dict(v, dtype, path + f"{k}:")
    if isinstance(d, list):
        for i, v in enumerate(d):
            _find_type_in_dict(v, dtype, path + f"[{i}]")
    elif isinstance(d, dtype):
        raise Exception(f"Value {d} of type {dtype} found at {path}")


# These two very helpful functions are borrowed from https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e

custom_meta_key = "salk-toolkit-meta"


def write_parquet_with_metadata(df: pd.DataFrame, meta: dict[str, object] | ParquetMeta, file_name: str) -> None:
    """Write DataFrame to Parquet file with embedded metadata.

    Args:
        df: DataFrame to write.
        meta: Metadata dictionary to embed.
        file_name: Path to output Parquet file.
    """
    table = pa.Table.from_pandas(df)

    # Convert meta to dict and ensure DataMeta objects are serialized
    if isinstance(meta, ParquetMeta):
        meta_payload = deepcopy(meta.model_dump(mode="json"))
    else:
        meta_payload = deepcopy(meta)
    data_meta = meta_payload.get("data")
    if isinstance(data_meta, DataMeta):
        meta_payload["data"] = data_meta.model_dump(mode="json")

    custom_meta_json = json.dumps(meta_payload)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)

    pq.write_table(table, file_name, compression="ZSTD")


def read_parquet_metadata(file_name: str) -> ParquetMeta | None:
    """Just load the metadata from the parquet file.

    Args:
        file_name: Path to Parquet file.

    Returns:
        ParquetMeta bundle, or None if no metadata found.
    """
    schema = pq.read_schema(file_name)
    if custom_meta_key.encode() in schema.metadata:
        restored_meta_json = schema.metadata[custom_meta_key.encode()]
        restored_meta = cast(dict[str, object], json.loads(restored_meta_json))
        return soft_validate(restored_meta, ParquetMeta)
    return None


@overload
def read_parquet_with_metadata(
    file_name: str, lazy: Literal[True], **kwargs: object
) -> tuple[pl.LazyFrame, ParquetMeta | None]: ...


@overload
def read_parquet_with_metadata(
    file_name: str, lazy: Literal[False] = False, **kwargs: object
) -> tuple[pd.DataFrame, ParquetMeta | None]: ...


def read_parquet_with_metadata(
    file_name: str, lazy: bool = False, **kwargs: object
) -> tuple[pd.DataFrame | pl.LazyFrame, ParquetMeta | None]:
    """Load parquet with metadata.

    Args:
        file_name: Path to Parquet file.
        lazy: Whether to return Polars LazyFrame instead of pandas DataFrame.
        **kwargs: Additional arguments passed to Parquet reader.

    Returns:
        Tuple of (DataFrame/LazyFrame, ParquetMeta bundle).
    """
    if lazy:  # Load it as a polars lazy dataframe
        meta = read_parquet_metadata(file_name)
        ldf = pl.scan_parquet(file_name, **kwargs)
        return ldf, meta

    # Read it as a normal pandas dataframe
    restored_table = pq.read_table(file_name, **kwargs)
    restored_df = restored_table.to_pandas()
    if custom_meta_key.encode() in restored_table.schema.metadata:
        restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
        restored_meta_payload = cast(dict[str, object], json.loads(restored_meta_json))
        restored_meta = soft_validate(restored_meta_payload, ParquetMeta)
    else:
        restored_meta = None

    return restored_df, restored_meta
