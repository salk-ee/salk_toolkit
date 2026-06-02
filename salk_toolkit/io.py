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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, Literal, TypeAlias, TypeVar, cast, overload

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
    JSONValue,
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
    MaxDiffBlock,
    OneHotBlock,
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
    warnings: bool = True,
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
            if warnings:
                warn(f"Categories for {c} are different between files - merging to total {n_cats} cats")
        else:
            reconciled[c] = dtype

    return reconciled


def _merge_categories(a: list | str | None, b: list | str | None) -> list | str | None:
    """Union two category specs. 'infer' on either side defers to data-driven
    resolution downstream, so propagate 'infer'. Otherwise union preserving a's
    order then appending b's new categories."""
    if a == "infer" or b == "infer":
        return "infer"
    if a is None:
        return b
    if b is None:
        return a
    out = list(a)
    for c in b:
        if c not in out:
            out.append(c)
    return out


_MERGE_SCALAR_FIELDS = (
    "continuous",
    "datetime",
    "ordered",
    "likert",
    "neutral_middle",
    "num_values",
    "label",
    "colors",
    "nonordered",
)


def _merge_column_meta(a: ColumnMeta, b: ColumnMeta, ctx: str) -> ColumnMeta:
    """Merge two ColumnMeta (or scale metas) for the same logical column across files.
    Categories union; listed scalar fields last-file-wins with a warn on genuine
    conflict. Raises on scale-kind mismatch and num_values/category length mismatch."""
    update: dict[str, Any] = {}
    cats = _merge_categories(a.categories, b.categories)
    if cats is not None:
        update["categories"] = cats
    for f in _MERGE_SCALAR_FIELDS:
        av, bv = getattr(a, f, None), getattr(b, f, None)
        if av is not None and bv is not None and av != bv:
            warn(f"{ctx}: field {f!r} differs across files ({av!r} vs {bv!r}); using last-file value")
        if bv is not None:
            update[f] = bv
    a_cat = a.categories is not None
    b_cat = b.categories is not None
    if (a.continuous and b_cat) or (b.continuous and a_cat):
        raise ValueError(f"{ctx}: scale-kind mismatch (categorical vs continuous) across files")
    merged = a.model_copy(update=update)
    final_cats = update.get("categories", a.categories)
    if isinstance(final_cats, list) and merged.num_values is not None and len(merged.num_values) != len(final_cats):
        raise ValueError(f"{ctx}: num_values length {len(merged.num_values)} != categories length {len(final_cats)}")
    return merged


def _merge_blocks(a: ColumnBlockMeta, b: ColumnBlockMeta) -> ColumnBlockMeta:
    """Merge two same-named blocks across files. Columns union (first-seen order);
    scale + per-column categories union; scalars last-file-wins. Raises on block-type
    mismatch."""
    if a.type != b.type:
        raise ValueError(f"Block {a.name!r}: type mismatch across files ({a.type!r} vs {b.type!r})")
    cols = dict(a.columns)
    for cn, cm in b.columns.items():
        cols[cn] = _merge_column_meta(cols[cn], cm, ctx=f"Block {a.name!r} column {cn!r}") if cn in cols else cm
    update: dict[str, Any] = {"columns": cols}
    if a.scale is not None and b.scale is not None:
        update["scale"] = _merge_column_meta(a.scale, b.scale, ctx=f"Block {a.name!r} scale")
    elif b.scale is not None:
        update["scale"] = b.scale
    return a.model_copy(update=update)


def _merge_data_metas(metas: list[DataMeta]) -> DataMeta:
    """Build the combined DataMeta for a multi-file load by unioning block structure
    across all file metas (in file order). Top-level fields come from the last file;
    only `structure` is merged. Single meta -> returned unchanged (no-op)."""
    if not metas:
        raise ValueError("_merge_data_metas called with no metas")
    if len(metas) == 1:
        return metas[0]
    merged = {}
    for meta in metas:
        if meta.structure is None:
            continue
        for name, block in meta.structure.items():
            merged[name] = _merge_blocks(merged[name], block) if name in merged else block
    return metas[-1].model_copy(update={"structure": merged})


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

    # Pre-scan extra FileDesc fields across all files so we can inject them as Categoricals
    # with a consistent category list (in file order, deduped) right at injection time.
    extra_field_categories: dict[str, list] = {}
    for fd in data_files:
        pydantic_extra = getattr(fd, "__pydantic_extra__", None) or {}
        for k, v in pydantic_extra.items():
            if k not in extra_field_categories:
                extra_field_categories[k] = []
            if v not in extra_field_categories[k]:
                extra_field_categories[k].append(v)

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
            csv_defaults: dict[str, Any] = {"low_memory": False}
            if read_opts.get("engine") == "python":
                csv_defaults.pop("low_memory")  # python engine doesn't support low_memory
            raw_data = pd.read_csv(cast(str, mapped_file), **{**csv_defaults, **read_opts})  # type: ignore[call-overload]
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

        # Add extra columns to raw data that contain info about the file.
        #
        # Explicit expectation: these are reserved system columns and are overwritten here for raw
        # file loads so downstream code can rely on them being consistent with `data_files`.
        #
        # However, when we load an already-annotated multi-file dataset (e.g. a JSON/YAML meta file
        # that expands to multiple source files, or a processed parquet with embedded meta),
        # we must NOT overwrite its internal per-file provenance columns (file_code/file_name),
        # otherwise we'd collapse them into a single file.
        preserve_annotated_multi = (
            extension in ["json", "yaml", "parquet"]
            and "result_meta" in locals()
            and result_meta is not None
            and getattr(result_meta, "files", None) is not None
            and len(cast(list[object], getattr(result_meta, "files"))) > 1
        )
        if not preserve_annotated_multi:
            raw_data["file_code"] = file_code
            raw_data["file_name"] = os.path.basename(data_file)

        # Add extra fields from FileDesc (any fields beyond file, opts, code)
        # In Pydantic v2, extra fields are stored in __pydantic_extra__ when extra="allow"
        pydantic_extra = getattr(fd, "__pydantic_extra__", None) or {}
        for k, v in pydantic_extra.items():
            if len(data_files) <= 1 and k == "file":
                continue
            cats = extra_field_categories[k]
            raw_data[k] = pd.Categorical([v] * len(raw_data), categories=cats)

        raw_data_dict[file_code] = raw_data

    if metas:  # Do we have any metainfo?
        meta = _merge_data_metas(metas)

        # Reconcile categoricals across files: pd.concat collapses categoricals with
        # different category lists to object dtype, so we need to re-unify them first.
        if len(raw_data_dict) > 1:
            reconciled_dtypes = _reconcile_categories(raw_data_dict, warnings=False)
            for fc, rdf in raw_data_dict.items():
                for col, dtype in reconciled_dtypes.items():
                    if col in rdf.columns:
                        raw_data_dict[fc][col] = rdf[col].astype("object").astype(dtype)

        # This will fix categories inside meta too - use concatenated view for this
        fdf = pd.concat(raw_data_dict.values())
        meta = _fix_meta_categories(meta, fdf, warnings=False)
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


def _file_meta_map(dfs: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return mapping of file_code -> file_name, requiring a 1-to-1 relationship."""
    fm = (
        pd.concat((df[["file_code", "file_name"]] for df in dfs.values()), ignore_index=True)
        .assign(file_code=lambda d: d.file_code.astype(str), file_name=lambda d: d.file_name.astype(str))
        .drop_duplicates()
    )
    if fm.duplicated(["file_code"]).any() or fm.duplicated(["file_name"]).any():
        raise ValueError("file_code/file_name must be 1-to-1")
    return dict(zip(fm["file_code"], fm["file_name"]))


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


def _check_topk_na_vals_after_replace(sdf: pd.DataFrame, *, block_name: str) -> None:
    """Require na_vals to match somewhere overall; warn on columns with no NA after replace."""
    if not sdf.isna().to_numpy().any():
        raise ValueError(f"No na_vals found in topk block {block_name!r}")

    cols_no_na = [c for c in sdf.columns if not sdf[c].isna().any()]
    if cols_no_na:
        listed = ", ".join(map(str, cols_no_na))
        warn(
            f"Topk block {block_name!r}: no NA in column(s) after na_vals replace ({listed})",
            UserWarning,
        )


def _apply_pre_transform_translate(block: ColumnBlockMeta, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if block.scale is None or not block.scale.translate:
        return df
    # TODO: consider str-coercing cells before .replace() — CSV round-trips turn
    # integer-string index cells into int64, so .replace({"1": "Economy"}) silently
    # no-ops. See follow-up from 2026-04-23 block-processing refactor.
    translate = cast("dict[object, object]", dict(block.scale.translate))
    df = df.copy()

    def _map_list(lst: object) -> object:
        if lst is None:
            return None
        if isinstance(lst, float) and pd.isna(lst):
            return None
        return [translate.get(x, x) for x in cast(Iterable[object], lst)]

    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if isinstance(block, MaxDiffBlock) and _is_series_of_lists(s):
            df[c] = s.map(_map_list)
        else:
            df[c] = s.astype("object").replace(translate)
    return df


def _apply_post_transform_translate(
    block: ColumnBlockMeta,
    sdf: pd.DataFrame,
    meta: ColumnBlockMeta,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Universal stage 5. If `scale.translate_after` is set on the *output* meta:
    - scalar cells: element-wise `.replace` via translate_after dict per column.
    - list-valued cells (future MaxDiff set-column support): map inside the list.
    - Categorical columns: rebuild with translated categories in the same order.
    - Rewrite `meta.scale.categories` to the translated values.
    Blocks with no scale or no translate_after pass through unchanged.
    MaxDiff blocks reject translate_after at pydantic validation (Task 3's
    `_reject_translate_after` model validator), so this branch is a defensive
    safety net — unreachable in practice."""
    scale = meta.scale
    if scale is None or not scale.translate_after:
        return sdf, meta
    if isinstance(block, MaxDiffBlock):
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate_after is not supported on maxdiff; "
            f"use scale.translate (pre-transform) instead."
        )
    t = dict(scale.translate_after)

    def _map_scalar(v: object) -> object:
        if isinstance(v, str):
            return t.get(v, v)
        return v

    def _map_list(lst: object, _t: dict[str, str] = t) -> object:
        if lst is None:
            return None
        if isinstance(lst, float) and pd.isna(lst):
            return None
        return [_t.get(x, x) if isinstance(x, str) else x for x in cast(Iterable[object], lst)]

    for col in sdf.columns:
        s = sdf[col]
        if _is_series_of_lists(s):
            sdf[col] = s.map(_map_list)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            new_cats = [t.get(c, c) if isinstance(c, str) else c for c in s.cat.categories]
            sdf[col] = s.cat.rename_categories(new_cats)
        else:
            sdf[col] = s.map(_map_scalar)

    scale_dict = scale.model_dump(mode="python")
    if not scale_dict.get("categories") or scale_dict.get("categories") == "infer":
        new_categories = list(dict.fromkeys(t.values()))
    else:
        new_categories = [t.get(c, c) if isinstance(c, str) else c for c in scale_dict["categories"]]
    scale_dict["categories"] = new_categories
    new_scale = type(scale).model_validate(scale_dict)
    # Propagate the post-translate categories onto each column too, mirroring what
    # `merge_scale_with_columns` does at block-validation time. Columns whose
    # categories were inherited from the pre-translate scale (`"infer"` or the
    # pre-translate list) get re-synced to the new categories.
    updated_columns: dict[str, ColumnMeta] = {}
    pre_cats = scale.categories
    for cn, col_meta in meta.columns.items():
        col_update: dict[str, object] = {}
        if col_meta.categories == "infer" or col_meta.categories == pre_cats:
            col_update["categories"] = new_categories
        if col_meta.translate_after == scale.translate_after:
            col_update["translate_after"] = new_scale.translate_after
        updated_columns[cn] = col_meta.model_copy(update=col_update) if col_update else col_meta
    meta_out = meta.model_copy(update={"scale": new_scale, "columns": updated_columns})
    return sdf, meta_out


def _process_block(
    block: ColumnBlockMeta, df: pd.DataFrame, **kwargs: object
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Universal driver for all block types (plain, topk, maxdiff, onehot).

    Executes the 5-stage processing pipeline:
    1. Match: Identify candidate columns from the dataframe.
    2. Explode: Fan out regex-matched columns into subgroup siblings.
    3. Pre-translate: Map raw cell values using scale.translate.
    4. Transform: Dispatch to type-specific transformation and output building.
    5. Post-translate: Map output cells and categories using scale.translate_after.
    """
    siblings: list[ColumnBlockMeta]
    if isinstance(block, MaxDiffBlock) and getattr(block, "from_columns", None) is None:
        siblings = [_apply_role_resolution(block, block, df)]
    elif isinstance(block, OneHotBlock):
        siblings = [block]
    elif block.type == "plain":
        siblings = [block]
    else:
        siblings = _subgroup_explode(block, df)

    for sib in siblings:
        cols = sib.input_df_columns(df)
        # STAGE 3: Pre-translate
        df_t = _apply_pre_transform_translate(sib, df, cols)

        # Pass choice_sets for MaxDiff from sibling context
        if isinstance(sib, MaxDiffBlock):
            cs = (
                _get_subgroup_config(block.choice_sets, sib.name, block.name) if hasattr(block, "choice_sets") else None
            )
            kwargs["choice_sets"] = cs

        # STAGE 4: Transform (MUST use df_t which has translated values)
        sdf, meta = _apply_transform(sib, df_t, source_block=block, **kwargs)

        # STAGE 5: Post-translate
        sdf, meta = _apply_post_transform_translate(sib, sdf, meta)
        yield sdf, meta


def _apply_transform(
    block: ColumnBlockMeta,
    df: pd.DataFrame,
    *,
    source_block: ColumnBlockMeta,
    **kwargs: object,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Dispatch to the per-type transform."""
    if isinstance(block, TopKBlock):
        assert isinstance(source_block, TopKBlock)
        source_pattern = source_block.from_columns if isinstance(source_block.from_columns, str) else None
        return _topk_apply_transform(block, df, source_pattern=source_pattern, source_block=source_block)
    if isinstance(block, MaxDiffBlock):
        assert isinstance(source_block, MaxDiffBlock)
        cs = kwargs.get("choice_sets")
        return _maxdiff_apply_transform(block, df, cs, source_block=source_block)
    if isinstance(block, OneHotBlock):
        assert isinstance(source_block, OneHotBlock)
        return _onehot_apply_transform(block, df, source_block.choices)
    if block.type == "plain":
        return _plain_apply_transform(block, df, source_block=source_block, **kwargs)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported block type for _apply_transform: {type(block)}")


def _plain_apply_transform(
    block: ColumnBlockMeta,
    df: pd.DataFrame,
    *,
    source_block: ColumnBlockMeta,
    raw_data_dict: dict[str, pd.DataFrame] | None = None,
    file_index_ranges: dict[str, slice] | None = None,
    constants: dict[str, Any] | None = None,
    einfo: dict[str, Any] | None = None,
    only_fix_categories: bool = False,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Universal transform for plain blocks.

    Processes columns one-by-one, applying the translate -> transform ->
    translate_after pipeline.
    """
    ndf_df = pd.DataFrame(index=df.index)
    updated_columns: dict[str, ColumnMeta] = {}

    for orig_cn, col_meta in block.columns.items():
        mcm = col_meta
        scale_meta = block.scale
        col_prefix = scale_meta.col_prefix if scale_meta is not None else None
        cn = col_prefix + orig_cn if col_prefix is not None else orig_cn

        if cn not in df.columns:
            # Column defined in meta but missing from source.
            # MATCH STABLE BEHAVIOR: skip creation in ndf_df, but keep it in metadata structure.
            updated_columns[orig_cn] = mcm
            continue

        s = df[cn].copy()

        if not only_fix_categories and not _is_series_of_lists(s):
            if s.dtype.name == "category":
                s = s.astype("object")

            if mcm.translate:
                s = s.astype("str").replace(mcm.translate).replace("nan", None).replace("None", None)

            if mcm.transform is not None and isinstance(mcm.transform, str):
                if raw_data_dict is None or file_index_ranges is None:
                    warn(f"Column {cn}: transform skipped because context is missing")
                else:
                    transformed_parts: list[pd.Series] = []
                    for file_code, file_raw_data in raw_data_dict.items():
                        idx_range = file_index_ranges[file_code]
                        s_local = s.iloc[idx_range]
                        # ndf_local = ndf_df.iloc[idx_range] # ndf view might be too complex for simple eval
                        transformed = eval(
                            mcm.transform,
                            {
                                "s": s_local,
                                "df": file_raw_data,
                                "pd": pd,
                                "np": np,
                                "stk": stk,
                                **(constants or {}),
                            },
                        )
                        transformed_parts.append(pd.Series(transformed, index=s_local.index, name=cn))
                    s = pd.concat(transformed_parts, ignore_index=True)

            if mcm.translate_after:
                s = pd.Series(s).astype("str").replace(mcm.translate_after).replace("nan", None).replace("None", None)

            if mcm.datetime:
                s = pd.to_datetime(s, errors="coerce")
            elif mcm.continuous:
                s = pd.to_numeric(s, errors="coerce")

        s.name = cn

        # Category mapping / inference logic
        if mcm.categories and not _is_series_of_lists(s):
            if mcm.categories == "infer":
                should_warn_ordered = mcm.ordered and not pd.api.types.is_numeric_dtype(s)
                if mcm.translate and not mcm.transform and set(mcm.translate.values()) >= set(s.dropna().unique()):
                    translate_values = list(mcm.translate.values())
                    cats = [str(c) for c in translate_values if c in s.dropna().unique()]
                    inferred_cats = list(dict.fromkeys(cats))
                    mcm = mcm.model_copy(update={"categories": inferred_cats})
                else:
                    if should_warn_ordered:
                        warn(f"Ordered category {cn} had category: infer. This only works for lexicographic ordering!")
                    s, inferred_cats = _deterministic_categories_and_values(s)
                    mcm = mcm.model_copy(update={"categories": inferred_cats})
                cats = inferred_cats
            elif pd.api.types.is_numeric_dtype(s):
                cats = mcm.categories
                if not isinstance(cats, list):
                    raise ValueError(f"Categories for {cn} must be a list")
                try:
                    fcats = np.array(cats).astype(float)
                    s_values_arr = np.asarray(s.values).reshape(-1)
                    distances = np.abs(s_values_arr.reshape(-1, 1) - fcats.reshape(1, -1))
                    indices = distances.argmin(axis=1)
                    s = pd.Series(
                        np.array(cats)[indices],
                        index=s.index,
                        name=s.name,
                        dtype=pd.CategoricalDtype(categories=cats, ordered=mcm.ordered or False),
                    )
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Categories for {cn} are not numeric: {cats}") from e
            else:
                cats = mcm.categories
                if not isinstance(cats, list):
                    raise ValueError(f"Categories for {cn} must be a list")

            if isinstance(cats, list):
                dropped = set(s.dropna().unique()) - set(cats)
                if dropped:
                    warn(f"Values for {cn} not in categories and will be dropped: {dropped}")
                s = pd.Series(pd.Categorical(s, categories=cats, ordered=mcm.ordered or False), name=cn)

        ndf_df[cn] = s
        updated_columns[orig_cn] = mcm

    if block.subgroup_transform is not None:
        if raw_data_dict is None or file_index_ranges is None:
            warn(f"Block {block.name!r}: subgroup_transform skipped because context is missing")
        else:
            subgroup_transformed_parts: list[pd.DataFrame] = []
            for file_code, file_raw_data in raw_data_dict.items():
                idx_range = file_index_ranges[file_code]
                gdf_local = ndf_df.iloc[idx_range].copy()
                transformed_gdf = eval(
                    block.subgroup_transform,
                    {
                        "gdf": gdf_local,
                        "df": file_raw_data,
                        "pd": pd,
                        "np": np,
                        "stk": stk,
                        **(constants or {}),
                    },
                )
                subgroup_transformed_parts.append(transformed_gdf)
            ndf_df = pd.concat(subgroup_transformed_parts).reset_index(drop=True)

    out_meta = block.model_copy(update={"columns": updated_columns})
    return ndf_df, out_meta


def _get_subgroup_config(value: object, sibling_name: str, source_name: str) -> object:
    """Extract a sibling-specific configuration from a parent field.

    A parent field (like `choice_sets` or `choice_mapping`) can be either:
    1. A 'flat' structure (list or dict) that is shared by all siblings.
    2. A 'keyed' dictionary where keys correspond to subgroup labels (e.g.,
       'economics', 'politics') and values are the sibling-specific configs.

    This helper extracts the correct config based on the sibling's name suffix.

    Args:
        value: The configuration value from the parent block.
        sibling_name: The name of the narrowed sibling block.
        source_name: The name of the original parent block.

    Returns:
        The configuration (flat or picked from the keyed dict) for this sibling.
    """
    if value is None:
        return None

    sibling_label = sibling_name.removeprefix(source_name).lstrip("_")
    is_keyed = isinstance(value, dict) and len(value) > 0 and all(isinstance(v, (list, dict)) for v in value.values())

    if not sibling_label:
        if is_keyed:
            raise ValueError(f"Block {source_name!r}: single sibling but field is keyed; expected flat")
        return value

    if not is_keyed:
        raise ValueError(f"Block {source_name!r}: multiple siblings but field is flat; expected dict keyed by label")
    keyed = cast(dict[str, object], value)
    if sibling_label not in keyed:
        raise ValueError(
            f"Block {source_name!r}: sibling {sibling_label!r} missing from field keys {list(keyed.keys())}"
        )
    return keyed[sibling_label]


def _match_columns(block: ColumnBlockMeta, df: pd.DataFrame) -> list[str]:
    pattern = block.from_columns
    if pattern is None:
        cols = [c for c in block.columns.keys() if c in df.columns]
    elif isinstance(pattern, list):
        cols = list(pattern)
    elif isinstance(pattern, str):
        regex = re.compile(pattern)
        cols = [c for c in df.columns if regex.match(c)]
    else:
        raise TypeError(f"from_columns must be str, list, or None; got {type(pattern)}")
    if not cols:
        raise ValueError(f"No columns matched for block {block.name!r} (from_columns={pattern!r})")
    return cols


def _block_scale_dict(block: ColumnBlockMeta) -> dict[str, Any]:
    """A deep-copied plain dict of the block's scale (empty dict when no scale),
    safe to mutate while building an output block."""
    return deepcopy(block.scale.model_dump(mode="python") if block.scale else {})


def _resolved_from_cols(block: ColumnBlockMeta, df: pd.DataFrame) -> list[str]:
    """`from_columns` as a concrete list: identity for an explicit list, else
    regex-matched against `df`."""
    return list(block.from_columns) if isinstance(block.from_columns, list) else _match_columns(block, df)


def _narrow_sibling(block: ColumnBlockMeta, cols: list[str], *, label_suffix: str) -> ColumnBlockMeta:
    new_name = block.name if not label_suffix else f"{block.name}_{label_suffix}"
    return block.model_copy(
        update={
            "name": new_name,
            "from_columns": cols,
            "subgroup_labels": None,
        }
    )


def _subgroup_explode(block: ColumnBlockMeta, df: pd.DataFrame) -> list[ColumnBlockMeta]:
    """Fan out regex-matched columns into subgroup siblings.

    If from_columns is a regex with capture groups, this function identifies every
    unique combination of captured values and returns a list of 'narrowed' siblings.
    """
    matched_cols = _match_columns(block, df)
    pattern = block.from_columns
    if not isinstance(pattern, str):
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    regex = re.compile(pattern)
    first = regex.match(matched_cols[0])
    assert first is not None
    n_groups = len(first.groups())

    # TOPK-specific: skip one group for sibling identity if aggregating.
    # Otherwise, use all groups.
    agg_pos = None
    if isinstance(block, TopKBlock):
        agg_idx = block.agg_index
        agg_pos = agg_idx - 1 if agg_idx > 0 else agg_idx
        if agg_pos < 0:
            agg_pos = n_groups + agg_pos

        if not (0 <= agg_pos < n_groups):
            raise ValueError(f"Block {block.name!r}: agg_index={agg_idx} out of range for {n_groups} capture group(s)")

    non_agg_positions = [i for i in range(n_groups) if i != agg_pos] if agg_pos is not None else list(range(n_groups))

    if not non_agg_positions:
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    def _key(col: str) -> tuple[str, ...]:
        m = regex.match(col)
        assert m is not None
        g = m.groups()
        return tuple(g[i] for i in non_agg_positions)

    sibling_cols: dict[tuple[str, ...], list[str]] = {}
    for c in matched_cols:
        sibling_cols.setdefault(_key(c), []).append(c)

    labels = block.subgroup_labels or {}

    def _label(key: tuple[str, ...]) -> str:
        parts = []
        for val, pos in zip(key, non_agg_positions, strict=True):
            parts.append(str(labels.get(str(pos + 1), {}).get(val, val)))
        return "_".join(parts)

    return [
        _apply_role_resolution(_narrow_sibling(block, cols, label_suffix=_label(key)), block, df)
        for key, cols in sibling_cols.items()
    ]


def _apply_role_resolution(sib: ColumnBlockMeta, source: ColumnBlockMeta, df: pd.DataFrame) -> ColumnBlockMeta:
    """Apply per-type role-column resolution. Label is derived from sibling vs source name."""
    sib_label = sib.name.removeprefix(source.name).lstrip("_")
    updates = sib.resolve_role_columns(df, sib_label)
    return sib.model_copy(update=updates) if updates else sib


def _topk_apply_transform(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """Dispatch TopK transformation based on input_format."""
    fmt = block.input_format
    if fmt in ("leftpacked", "ranked_leftpack"):
        # No transform needed, just extract columns
        return _topk_transform_passthrough(block, df, source_block=source_block)
    if fmt == "onehot":
        return _topk_transform_onehot(block, df, source_pattern=source_pattern, source_block=source_block)
    if fmt == "ranked_onehot":
        raise NotImplementedError("ranked_onehot transform not yet implemented")
    raise ValueError(f"unknown TopK input_format: {fmt!r}")


def _topk_transform_passthrough(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK passthrough: input is already in TopK format (one row per response)."""
    from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
    if not from_cols:
        raise ValueError(f"TopK {block.name!r}: missing explicit from_columns for passthrough")

    # Check for res_columns mismatch (enforce for skip transforms)
    res_cols = list(source_block.res_columns) if isinstance(source_block.res_columns, list) else []
    if res_cols and from_cols != res_cols:
        raise ValueError(
            f"TopK block {block.name!r}: input_format={block.input_format!r} requires "
            f"res_columns to match from_columns; got res_columns={res_cols!r} "
            f"vs from_columns={from_cols!r}"
        )

    sdf = df[from_cols].copy()
    meta_out = _build_topk_output_block(
        name=block.name,
        columns=from_cols,
        from_cols=from_cols,
        res_cols=from_cols,
        block=block,
    )
    return sdf, meta_out


def _topk_transform_onehot(
    block: TopKBlock,
    df: pd.DataFrame,
    *,
    source_pattern: str | None,
    source_block: TopKBlock,
) -> tuple[pd.DataFrame, TopKBlock]:
    """TopK onehot transform: pivot multiple columns (mentions) into ranked TopK columns."""
    from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
    na_vals = list(block.na_vals or [])

    # 1. Standardize input (mentions -> values)
    sdf = df[from_cols].astype("object").replace(na_vals, None)
    _check_topk_na_vals_after_replace(sdf, block_name=block.name)

    # 2. Pivot to values
    if source_pattern:
        regex = re.compile(source_pattern)
        agg_pos = block.agg_index
        agg_pos = agg_pos - 1 if agg_pos > 0 else agg_pos
        # Use capture group from regex as the value for each cell
        sdf.columns = [regex.match(c).groups()[agg_pos] for c in sdf.columns]  # type: ignore[union-attr]
        sdf = sdf.mask(~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1)
    else:
        # If no pattern, use column names (stripping prefix if present)
        if source_block.from_prefix:
            sdf.columns = [c.removeprefix(source_block.from_prefix) for c in sdf.columns]
        sdf = sdf.mask(~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1)

    # 3. Collapse left
    _throw_vals_left(sdf)

    # 4. Map to result column names
    res_cols = _resolve_topk_res_cols(block, source_block, source_pattern)
    sdf.columns = res_cols
    sdf = sdf.dropna(axis=1, how="all")

    # 5. Truncate to K
    kmax = block.k
    if isinstance(kmax, int) and sdf.shape[1] > kmax:
        if sdf.iloc[:, kmax:].notna().any().any():
            raise ValueError(
                f"TopK block {block.name!r}: truncation to k={kmax} would drop "
                f"non-NA values in columns {list(sdf.columns[kmax:])}"
            )
        sdf = sdf.iloc[:, :kmax]

    meta_out = _build_topk_output_block(
        name=block.name,
        columns=sdf.columns.tolist(),
        from_cols=from_cols,
        res_cols=res_cols,
        block=block,
    )
    return sdf, meta_out


def _resolve_topk_res_cols(block: TopKBlock, source: TopKBlock, pattern: str | None) -> list[str]:
    """Determine final result column names for a TopK sibling."""
    if isinstance(source.res_columns, list):
        return list(source.res_columns)

    if isinstance(source.res_columns, str) and pattern:
        regex = re.compile(pattern)
        from_cols = list(block.from_columns) if isinstance(block.from_columns, list) else []
        return [regex.match(c).expand(source.res_columns) for c in from_cols]  # type: ignore[union-attr]

    raise ValueError(f"TopK {block.name!r}: cannot resolve res_columns")


def _build_topk_output_block(
    *,
    name: str,
    columns: list[str],
    from_cols: list[str],
    res_cols: list[str],
    block: TopKBlock,
) -> TopKBlock:
    scale_dict = _block_scale_dict(block)
    return soft_validate(
        {
            "type": "topk",
            "name": name,
            "scale": scale_dict,
            "columns": list(columns),
            "from_columns": list(from_cols),
            "res_columns": list(res_cols),
            "agg_index": block.agg_index,
            "na_vals": list(block.na_vals or []),
            "k": block.k,
            "from_prefix": block.from_prefix,
            "input_format": block.input_format,
        },
        TopKBlock,
    )


def _maxdiff_apply_transform(
    block: MaxDiffBlock,
    df: pd.DataFrame,
    choice_sets: object,
    *,
    source_block: MaxDiffBlock,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    if block.input_format == "choice_sets":
        return _maxdiff_transform_choice_sets(block, df, choice_sets, source_block=source_block)
    if block.input_format == "resolved":
        return _maxdiff_transform_resolved(block, df)
    raise ValueError(f"unknown MaxDiff input_format: {block.input_format!r}")


def _resolve_maxdiff_role(spec: object, df: pd.DataFrame) -> list[str]:
    if spec is None:
        raise ValueError("maxdiff role column missing")
    if isinstance(spec, list):
        return list(spec)
    r = re.compile(cast(str, spec))
    return sorted(c for c in df.columns if r.match(c))


def _align_resolved_roles(
    block: MaxDiffBlock, best: list[str], worst: list[str], sets: list[str]
) -> tuple[list[str], list[str], list[str]]:
    all_regex = all(isinstance(v, str) for v in (block.best_columns, block.worst_columns, block.set_columns))
    if all_regex:
        bp = re.compile(cast(str, block.best_columns))
        wp = re.compile(cast(str, block.worst_columns))
        sp = re.compile(cast(str, block.set_columns))
        by_key: dict[str, list[str | None]] = {}
        for c in best:
            m = bp.match(c)
            assert m is not None
            by_key.setdefault(m.group(1), [None, None, None])[0] = c
        for c in worst:
            m = wp.match(c)
            assert m is not None
            by_key.setdefault(m.group(1), [None, None, None])[1] = c
        for c in sets:
            m = sp.match(c)
            assert m is not None
            by_key.setdefault(m.group(1), [None, None, None])[2] = c
        missing = [(k, t) for k, t in by_key.items() if None in t]
        if missing:
            raise ValueError(f"MaxDiff resolved: incomplete alignment: {missing}")
        keys = sorted(by_key, key=lambda s: int(s) if s.isdigit() else s)
        triples = [by_key[k] for k in keys]
        return [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]  # type: ignore[return-value]

    if not (len(best) == len(worst) == len(sets)):
        raise ValueError(
            f"MaxDiff resolved lists must have equal length; got best={len(best)}, worst={len(worst)}, sets={len(sets)}"
        )
    return best, worst, sets


def _maxdiff_transform_resolved(
    block: MaxDiffBlock,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    best = _resolve_maxdiff_role(block.best_columns, df)
    worst = _resolve_maxdiff_role(block.worst_columns, df)
    sets = _resolve_maxdiff_role(block.set_columns, df)
    best, worst, sets = _align_resolved_roles(block, best, worst, sets)
    cols = sorted(set(best) | set(worst) | set(sets))
    sdf = df[cols].copy()
    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "maxdiff",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in cols},
            "best_columns": best,
            "worst_columns": worst,
            "set_columns": sets,
            "input_format": "resolved",
        },
        MaxDiffBlock,
    )
    return sdf, out


def _maxdiff_transform_choice_sets(
    block: MaxDiffBlock,
    df: pd.DataFrame,
    choice_sets: object,
    *,
    source_block: MaxDiffBlock,
) -> tuple[pd.DataFrame, MaxDiffBlock]:
    df = df.copy(deep=True)
    if not (
        isinstance(block.best_columns, list)
        and isinstance(block.worst_columns, list)
        and isinstance(block.set_columns, list)
    ):
        raise TypeError(
            f"_maxdiff_transform_choice_sets expects resolved role columns; got "
            f"best={type(block.best_columns).__name__}, "
            f"worst={type(block.worst_columns).__name__}, "
            f"set={type(block.set_columns).__name__}"
        )
    translate: dict[str, str] = (
        {str(k): str(v) for k, v in block.scale.translate.items()} if (block.scale and block.scale.translate) else {}
    )
    if not translate:
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate is required (maps 1-based "
            f"index strings to display names). Got empty translate."
        )
    topics: list[str] = [translate[k] for k in sorted(translate.keys(), key=int)]
    sets = choice_sets
    best_cols: Sequence[str] | str = block.best_columns
    worst_cols: Sequence[str] | str = block.worst_columns
    set_cols: Sequence[str] | str | None = block.set_columns
    # Parse setindex_column: can be None, str, or [str] or [str, dict]
    setindex_col_name: str | None = None
    setindex_col_meta: ColumnMeta | None = None
    if isinstance(block.setindex_column, str):
        setindex_col_name = block.setindex_column
    elif isinstance(block.setindex_column, list):
        _setindex_parts = list(block.setindex_column)
        setindex_col_name = str(_setindex_parts[0]) if _setindex_parts else None
        if _setindex_parts and len(_setindex_parts) > 1 and isinstance(_setindex_parts[1], dict):
            setindex_col_meta = soft_validate(_setindex_parts[1], ColumnMeta)
    if set_cols is None:
        raise ValueError("Maxdiff create blocks must define 'set_columns'.")
    best_cols = list(best_cols)
    worst_cols = list(worst_cols)
    set_cols = list(set_cols)

    def _maybe_json_load(value: str | None) -> list[object] | None:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else None
        except (TypeError, json.JSONDecodeError):
            return None

    def _is_int_like(token: object) -> bool:
        if isinstance(token, (int, np.integer)):
            return True
        if isinstance(token, str):
            stripped = token.strip()
            if not stripped:
                return False
            try:
                int(stripped)
                return True
            except ValueError:
                return False
        return False

    def _tokens_from_value(value: object) -> list[str] | float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            parsed = _maybe_json_load(stripped) if stripped.startswith("[") and stripped.endswith("]") else None
            if parsed is not None:
                return [str(item) for item in parsed] if isinstance(parsed, list) else None
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raise ValueError(f"Unsupported maxdiff set specification value: {value}")

    def _convert_tokens_to_topics(tokens: list[str] | float | None) -> list[str] | None | float:
        if tokens is None or (isinstance(tokens, float) and pd.isna(tokens)):
            return tokens
        if not isinstance(tokens, list):
            tokens = [str(tokens)]
        if not tokens:
            return []
        if all(isinstance(token, str) for token in tokens) and not all(_is_int_like(token) for token in tokens):
            stripped = [token.strip() for token in tokens]
            return [translate.get(t, t) for t in stripped] if translate else stripped  # type: ignore[return-value]
        if all(_is_int_like(token) for token in tokens):
            converted = []
            for token in tokens:
                idx = int(token) if not isinstance(token, (int, np.integer)) else int(token)
                if idx < 1 or idx > len(topics):
                    raise ValueError(f"Maxdiff set index {idx} is out of bounds for topics list of size {len(topics)}.")
                converted.append(topics[idx - 1])
            return converted
        if all(isinstance(token, str) for token in tokens):
            return [token.strip() for token in tokens]
        raise ValueError(f"Unsupported token types in maxdiff set definition: {tokens}")

    ordered_cols = best_cols + worst_cols

    if setindex_col_name:
        df = df[ordered_cols + [setindex_col_name]]
        if sets is None:
            raise ValueError("Maxdiff definitions using 'setindex_column' must also define 'sets'.")
        topics_arr = np.array(["", *topics], dtype=object)
        sets_arr = np.asarray(sets, dtype=int)
        lsets = topics_arr[sets_arr]

        setindex = df[setindex_col_name].astype(np.int64).to_numpy() - 1
        selected_sets = lsets[setindex]
        df_setcols = pd.DataFrame(selected_sets.tolist(), columns=set_cols, index=df.index)
        df[set_cols] = df_setcols
    else:
        df = df[ordered_cols + set_cols]
        for col in set_cols:
            converted_values: list[list[str] | None] = []
            for value in df[col].tolist():
                tokens = _tokens_from_value(value)
                converted = _convert_tokens_to_topics(tokens)
                if converted is None or (isinstance(converted, float) and pd.isna(converted)):
                    converted_values.append(None)
                elif isinstance(converted, list):
                    converted_values.append(converted)
                else:
                    converted_values.append(None)
            df[col] = converted_values  # type: ignore[assignment]

    # Pre-translate already mapped index strings to topic names before the transform ran,
    # so df[col] already contains translated values — just cast to categorical.
    for col in best_cols + worst_cols:
        s = df[col]
        s = pd.Categorical(s, categories=topics)
        df[col] = s

    df = df.sort_index(axis=1)

    base_columns = sorted(best_cols + worst_cols + cast(list[str], set_cols))
    best_worst_col_meta = ColumnMeta(categories=topics)
    columns_spec: dict[str, ColumnMeta] = {col: best_worst_col_meta for col in base_columns}
    if setindex_col_name is not None:
        if setindex_col_meta is None:
            setindex_col_meta = ColumnMeta()
        if setindex_col_meta.categories is None:
            setindex_col_meta = setindex_col_meta.model_copy(update={"categories": topics})
        if setindex_col_meta.continuous is False:
            setindex_col_meta = setindex_col_meta.model_copy(update={"continuous": True})
        columns_spec = {setindex_col_name: setindex_col_meta} | columns_spec

    scale_dict = _block_scale_dict(block)
    scale_dict["categories"] = topics

    output_block = soft_validate(
        {
            "type": "maxdiff",
            "name": block.name,
            "scale": scale_dict,
            "columns": columns_spec,
            "best_columns": list(best_cols),
            "worst_columns": list(worst_cols),
            "set_columns": list(cast(list[str], set_cols)),
            "setindex_column": source_block.setindex_column,
        },
        MaxDiffBlock,
    )
    return df, output_block


def _onehot_apply_transform(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    if block.input_format == "leftpacked":
        return _onehot_transform_leftpacked(block, df, choices)
    if block.input_format == "wide":
        return _onehot_transform_wide(block, df, choices)
    raise ValueError(f"unknown OneHot input_format: {block.input_format!r}")


def _onehot_transform_leftpacked(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    from_cols = _resolved_from_cols(block, df)
    src = df[from_cols].astype("object")
    if block.na_vals:
        src = src.replace(block.na_vals, None)

    observed = [
        v for v in pd.unique(src.values.ravel("K")) if v is not None and not (isinstance(v, float) and pd.isna(v))
    ]

    if choices is not None:
        unknown = set(observed) - set(choices)
        if unknown:
            raise ValueError(f"OneHot block {block.name!r}: values {sorted(unknown)} not in choices")
        final_choices = list(choices)
    else:
        final_choices = sorted(observed)

    prefix = block.res_prefix or f"{block.name}_"
    out_cols = [f"{prefix}{c}" for c in final_choices]
    out_df = pd.DataFrame(
        {f"{prefix}{c}": src.eq(c).any(axis=1) for c in final_choices},
        index=df.index,
    )

    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "onehot",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in out_cols},
            "from_columns": from_cols,
            "input_format": "leftpacked",
            "choices": final_choices,
            "res_prefix": block.res_prefix,
        },
        OneHotBlock,
    )
    return out_df, out


def _onehot_transform_wide(
    block: OneHotBlock,
    df: pd.DataFrame,
    choices: list[str] | None,
) -> tuple[pd.DataFrame, OneHotBlock]:
    from_cols = _resolved_from_cols(block, df)
    sdf = df[from_cols].copy()
    prefix = block.res_prefix or ""
    final_choices: list[str] | None
    if choices is not None:
        final_choices = list(choices)
    elif prefix:
        final_choices = [c.removeprefix(prefix) for c in from_cols]
    else:
        final_choices = None

    scale_dict = _block_scale_dict(block)
    out = soft_validate(
        {
            "type": "onehot",
            "name": block.name,
            "scale": scale_dict,
            "columns": {c: {} for c in from_cols},
            "from_columns": from_cols,
            "input_format": "wide",
            "choices": final_choices,
            "res_prefix": block.res_prefix,
        },
        OneHotBlock,
    )
    return sdf, out


def _demote_to_plain(block: ColumnBlockMeta) -> ColumnBlockMeta:
    """Demote a specialized block (TopKBlock / MaxDiffBlock) to a plain ColumnBlockMeta,
    preserving every field declared on ``ColumnBlockMeta`` and dropping subclass-specific
    ones. Using ``model_fields`` instead of a hand-enumerated list means new fields added
    to ``ColumnBlockMeta`` are carried over automatically. Input-only directives
    (from_columns, subgroup_labels) are cleared."""
    kwargs = {k: getattr(block, k) for k in ColumnBlockMeta.model_fields if k != "type"}
    # Clear input-only directives that are not part of the demoted plain block
    kwargs["from_columns"] = None
    kwargs["subgroup_labels"] = None
    return ColumnBlockMeta(**kwargs)


# Default usage with mature metafile: _process_annotated_data(<metafile name>)
# When figuring out the metafile, it can also be run as: _process_annotated_data(meta=<dict>, data_file=<>)


def _combine_first_preserving_order(*frames: pd.DataFrame) -> pd.DataFrame:
    """Like ``DataFrame.combine_first(other1).combine_first(other2)...`` but
    keeps the natural source-data column order instead of lex-sorting.

    ``DataFrame.combine_first`` lex-sorts the union of column names in its
    result. That ordering is wrong for our TopK/MaxDiff/OneHot block pipeline
    when ``from_columns`` is a regex with a single numeric capture group
    (e.g. ``vA10_M_(\\d+)`` matching ``vA10_M_1, ..., vA10_M_14, vA10_M_99``):
    after the onehot pivot + leftpack, the rename step assumes the columns are
    still in source order, so a lex-sorted ``vA10_M_1, vA10_M_10, vA10_M_11,
    ..., vA10_M_2, ...`` produces ``issue_top_1, issue_top_10, issue_top_11``
    instead of ``issue_top_1, issue_top_2, issue_top_3``.

    Restore the natural order: first frame's columns first (in their order),
    then any new columns from each subsequent frame (in their order), deduped.
    The underlying ``combine_first`` semantics (left-priority value coalescing
    plus index union) are preserved — only the column ordering changes.
    """
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for other in frames[1:]:
        result = result.combine_first(other)
    desired: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for c in frame.columns:
            if c not in seen:
                desired.append(c)
                seen.add(c)
    # Tolerate any columns combine_first invented that we didn't account for
    desired += [c for c in result.columns if c not in seen]
    return result[desired]


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

    assert raw_data_dict is not None, "Expected raw_data_dict to be initialized before processing"

    file_meta_map = _file_meta_map(raw_data_dict)
    file_codes_in_order = list(raw_data_dict.keys())
    raw_data_dict = {fc: raw_data_dict[fc] for fc in file_codes_in_order}

    # Run preprocessing per file
    if meta_obj.preprocessing is not None and not only_fix_categories:
        for file_code, df in raw_data_dict.items():
            file_name = file_meta_map[file_code]
            globs = {
                "pd": pd,
                "np": np,
                "sp": sp,
                "stk": stk,
                "df": df,
                "file_code": file_code,
                "file_name": file_name,
                **einfo,
                **constants,
            }
            exec(_str_from_list(meta_obj.preprocessing), globs)
            raw_data_dict[file_code] = globs["df"]

    # Ensure file metadata columns always survive end-to-end (also if preprocessing dropped/mutated them).
    file_names_in_order: list[str] = []
    for file_code in file_codes_in_order:
        df = raw_data_dict[file_code]
        file_name_val = file_meta_map[file_code]
        # Overwrite to guarantee correctness even if preprocessing mutated/dropped these columns.
        df["file_code"] = str(file_code)
        df["file_name"] = file_name_val
        raw_data_dict[file_code] = df
        file_names_in_order.append(file_name_val)

    # Inject implicit metadata for system file columns so they can be used downstream (e.g. plotting/pipeline).
    # Provide explicit ordered category order (no "infer") for determinism.
    sys_block_name = "files"
    sys_block_hidden = len(raw_data_dict) <= 1
    sys_block_dict: dict[str, object] = {
        "name": sys_block_name,
        "generated": True,
        "hidden": sys_block_hidden,
        "columns": {
            "file_code": {"categories": [str(fc) for fc in file_codes_in_order], "ordered": True},
            "file_name": {"categories": file_names_in_order, "ordered": True},
        },
    }
    sys_block = (
        soft_validate(sys_block_name, ColumnBlockMeta)
        if isinstance(sys_block_name, ColumnBlockMeta)
        else soft_validate(sys_block_dict, ColumnBlockMeta)
    )
    structure2 = dict(meta_obj.structure)
    if sys_block_name in structure2:
        existing = structure2[sys_block_name]
        merged_cols = dict(existing.columns)
        for k, v in sys_block.columns.items():
            merged_cols.setdefault(k, v)
        structure2[sys_block_name] = existing.model_copy(
            update={"columns": merged_cols, "hidden": sys_block_hidden, "generated": True}
        )
    else:
        structure2[sys_block_name] = sys_block
    meta_obj = meta_obj.model_copy(update={"structure": structure2})

    raw_data_concat = pd.concat(raw_data_dict.values()).reset_index(drop=True) if raw_data_dict else pd.DataFrame()
    # Initialize concatenated DataFrame - start empty, will be built column by column
    ndf_df = pd.DataFrame()

    # Compute and store row index ranges for each file_code
    file_index_ranges: dict[str, slice] = {}
    current_idx = 0
    for file_code, file_raw in raw_data_dict.items():
        file_len = len(file_raw)
        file_index_ranges[file_code] = slice(current_idx, current_idx + file_len)
        current_idx += file_len

    all_cns: dict[str, str] = {}
    # Build new structure dict as we process groups (Pydantic models are immutable)
    new_structure: dict[str, ColumnBlockMeta] = {}
    for group_name, group in meta_obj.structure.items():
        if group.name in all_cns:
            raise Exception(f"Group name {group.name} duplicates a column name in group {all_cns[group.name]}")
        all_cns[group.name] = group.name

        # 1. Build source DF for this group by concatenating raw columns from all files.
        group_source_cols: dict[str, pd.Series] = {}
        has_any_source = False
        for orig_cn, col_meta in group.columns.items():
            scale_meta = group.scale
            col_prefix = scale_meta.col_prefix if scale_meta is not None else None
            cn = (col_prefix + orig_cn) if col_prefix is not None else orig_cn

            # In only_fix_categories mode, source is the processed column name itself
            source_spec = cn if only_fix_categories else (col_meta.source if col_meta.source is not None else orig_cn)

            if cn in all_cns and cn != group.name:
                raise Exception(f"Duplicate column name found: '{cn}' in {all_cns[cn]} and {group.name}")
            all_cns[cn] = group.name

            per_file_series: list[pd.Series] = []
            for file_code, file_raw_data in raw_data_dict.items():
                if isinstance(source_spec, dict):
                    sn = source_spec.get(file_code, source_spec.get("default", orig_cn))
                else:
                    sn = source_spec

                # If we've already processed this column in a previous block, take it from ndf_df
                if only_fix_categories and sn in ndf_df.columns:
                    # In only_fix_categories, ndf_df is already populated with processed data.
                    idx_range = file_index_ranges[file_code]
                    s = ndf_df[sn].iloc[idx_range].copy()
                    s.name = cn
                    per_file_series.append(s)
                    has_any_source = True
                    continue

                if sn not in file_raw_data:
                    # Don't pre-initialize missing columns in group_df to avoid merge overlap issues later.
                    continue

                s = file_raw_data[sn].copy()
                s.name = cn
                per_file_series.append(s)
                has_any_source = True

            if per_file_series:
                s_concat = pd.concat(per_file_series).reset_index(drop=True)
                group_source_cols[cn] = s_concat

        # If it's a plain block and NO source columns were found at all across any files,
        # we only skip it if it has no declared columns and isn't generated.
        if not has_any_source and not group.generated and not group.columns and group.type == "plain":
            continue

        group_df = pd.DataFrame(group_source_cols)
        if group_df.empty:
            group_df = pd.DataFrame(index=raw_data_concat.index)

        # 2. Process the block (Stage 1-5) using the universal driver.
        # `DataFrame.combine_first` lex-sorts column names in the result. That
        # broke TopK on inputs like `vA10_M_(\d+)` where lex order interleaves
        # `vA10_M_10/_11/_12/_13/_14` between `_M_1` and `_M_2`, causing the
        # post-leftpack positional rename to tag rank-2 slots as `issue_top_10`
        # instead of `issue_top_2`. Restore the natural source order: columns
        # produced by this group's own meta first, then carry-overs from prior
        # blocks, then the raw concat — each in their original order, deduped.
        create_source_df = _combine_first_preserving_order(group_df, ndf_df, raw_data_concat)

        if isinstance(group, (TopKBlock, MaxDiffBlock, OneHotBlock)):
            # Specialized blocks: handle explosion and transformation

            # Pass 1: Demote the input specialized block to a plain ColumnBlockMeta: its raw columns
            # are preserved under the original name, but processing directives are dropped.
            demoted_parent = _demote_to_plain(group)
            new_structure[demoted_parent.name] = demoted_parent

            # We must also process the raw columns of the specialized block as plain
            # if they were explicitly declared in its 'columns' field.
            if demoted_parent.columns:
                p_cols = demoted_parent.input_df_columns(create_source_df)
                df_pt = _apply_pre_transform_translate(demoted_parent, create_source_df, p_cols)
                sdf_p, _ = _plain_apply_transform(
                    demoted_parent,
                    df_pt,
                    source_block=group,
                    raw_data_dict=raw_data_dict,
                    file_index_ranges=file_index_ranges,
                    constants=constants,
                    einfo=einfo,
                    only_fix_categories=only_fix_categories,
                )
                for c in sdf_p.columns:
                    ndf_df[c] = sdf_p[c]
            else:
                # If no columns, still ensure they are in ndf_df from group_df
                for c in group_df.columns:
                    ndf_df[c] = group_df[c]

            # Pass 2: specialized explosion / transformation
            # Refresh source df with Pass 1 results. Same column-order
            # restoration as Pass 1 (see comment above).
            create_source_df_refreshed = _combine_first_preserving_order(ndf_df, raw_data_concat)

            processed_siblings: list[tuple[pd.DataFrame, ColumnBlockMeta]] = list(
                _process_block(
                    group,
                    create_source_df_refreshed,
                    raw_data_dict=raw_data_dict,
                    file_index_ranges=file_index_ranges,
                    constants=constants,
                    einfo=einfo,
                    only_fix_categories=only_fix_categories,
                )
            )

            for sdf, smeta in processed_siblings:
                for c in sdf.columns:
                    ndf_df[c] = sdf[c]
                new_structure[smeta.name] = smeta
        else:
            # Plain blocks: process directly
            processed_plain: list[tuple[pd.DataFrame, ColumnBlockMeta]] = list(
                _process_block(
                    group,
                    create_source_df,
                    raw_data_dict=raw_data_dict,
                    file_index_ranges=file_index_ranges,
                    constants=constants,
                    einfo=einfo,
                    only_fix_categories=only_fix_categories,
                )
            )

            for sdf, smeta in processed_plain:
                for c in sdf.columns:
                    ndf_df[c] = sdf[c]

                # MATCH OLD BEHAVIOR: Drop columns that are entirely NA from the DATAFRAME
                # but keep them in the METADATA if they were explicitly declared.
                final_cols_map = {}
                prefix = smeta.scale.col_prefix if smeta.scale and smeta.scale.col_prefix else ""
                for orig_k, v in smeta.columns.items():
                    cn = prefix + orig_k
                    if cn in ndf_df.columns and ndf_df[cn].isna().all() and not smeta.generated:
                        del ndf_df[cn]
                    final_cols_map[orig_k] = v

                new_structure[smeta.name] = smeta.model_copy(update={"columns": final_cols_map})

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

    if not only_fix_categories:
        # Pass 3: Lightweight category fix-up pass for columns that appeared after postprocessing
        # (e.g. merged columns). We process only blocks that are explicitly plain and exist in df.
        for group in meta_obj.structure.values():
            if group.type != "plain":
                continue

            # Skip if none of its columns are in ndf_df (avoid creating NaNs)
            prefix = group.scale.col_prefix if group.scale and group.scale.col_prefix else ""
            if not any((prefix + k) in ndf_df.columns for k in group.columns.keys()):
                continue

            processed_fixup: list[tuple[pd.DataFrame, ColumnBlockMeta]] = list(
                _process_block(
                    group,
                    ndf_df,
                    only_fix_categories=True,
                )
            )
            for sdf, smeta in processed_fixup:
                for c in sdf.columns:
                    ndf_df[c] = sdf[c]

    # Final fix categories after all processing passes
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


def fix_df_with_meta(df: pd.DataFrame, dmeta: DataMeta) -> pd.DataFrame:
    """Fix df dtypes etc using meta - needed after a lazy load.

    Args:
        df: DataFrame to fix.
        dmeta: Data metadata dictionary.

    Returns:
        DataFrame with corrected dtypes and categories.
    """
    cmeta = extract_column_meta(dmeta)
    for c in df.columns:
        if c not in cmeta:
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

    if data_meta.structure is None:
        return {}

    res: dict[str, list[str]] = {}
    for block in data_meta.structure.values():
        if block.hidden:
            continue
        scale_meta = block.scale
        prefix = scale_meta.col_prefix if scale_meta is not None and scale_meta.col_prefix is not None else ""
        res[block.name] = [f"{prefix}{cn}" for cn in block.columns.keys()]
    return res

    # return { g['name'] : [(t[0] if type(t)!=str else t) for t in g['columns']] for g in data_meta['structure'] }


T_list_alias = TypeVar("T_list_alias")


def list_aliases(lst: Sequence[T_list_alias], da: dict[str, list[str]]) -> list[T_list_alias | str]:
    """Take a list and a dict and replace all dict keys in list with their corresponding lists in-place.

    Expand aliases in a list using a dictionary mapping.

    Args:
        lst: List of strings (or other res-col tokens) that may contain aliases.
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


def _make_deepl_translate_fn(deepl_key: str, source_lang: str) -> Callable[[str], str]:
    """Build a cached DeepL translation function (source_lang → EN)."""
    import deepl

    translator = deepl.Translator(deepl_key)

    def _translate(txt: str) -> str:
        if not txt:
            return ""
        result = translator.translate_text(txt, source_lang=source_lang, target_lang="EN-US")
        if isinstance(result, list):
            return result[0].text
        return result.text

    return _translate


def infer_meta(
    data_file: str | None = None,
    meta_file: bool | str = True,
    read_opts: dict[str, object] | None = None,
    df: pd.DataFrame | None = None,
    translate_fn: Callable[[str], str] | None = None,
    translation_blacklist: list[str] | None = None,
    deepl_key: str | None = None,
    source_lang: str | None = None,
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
        deepl_key: DeepL API key. When provided, builds a translate_fn automatically.
            Requires source_lang to be set explicitly.
        source_lang: Source language code for DeepL (e.g. 'LT', 'ET', 'RO').
            Required when deepl_key is provided.

    Returns:
        Inferred metadata dictionary.
    """
    if deepl_key is not None and source_lang is None:
        raise ValueError("source_lang is required when deepl_key is provided (e.g. 'LT', 'ET', 'RO')")
    if source_lang is not None and deepl_key is None:
        raise ValueError("deepl_key is required when source_lang is provided")
    if deepl_key is not None and translate_fn is not None:
        raise ValueError("Cannot specify both deepl_key and translate_fn")

    if deepl_key is not None:
        assert source_lang is not None  # guaranteed by check above
        translate_fn = _make_deepl_translate_fn(deepl_key, source_lang)

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
            csv_defaults: dict[str, Any] = {"low_memory": False}
            if read_opts.get("engine") == "python":
                csv_defaults.pop("low_memory")  # python engine doesn't support low_memory
            df = pd.read_csv(data_file, **{**csv_defaults, **read_opts})  # type: ignore[call-overload]
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
    constants: Mapping[str, JSONValue] | None = None,
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
        # Drop system provenance columns from the merge-side to avoid suffix collisions.
        # Keep them on the left df (the main dataset) only.
        ndf = ndf.drop(
            columns=[c for c in ["file_code", "file_name"] if c in ndf.columns and c not in on],
            errors="ignore",
        )
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
    constants: Mapping[str, JSONValue] | None = ...,
    skip_postprocessing: bool = ...,
    **kwargs: object,
) -> pd.DataFrame: ...


@overload
def read_and_process_data(
    desc: str | dict[str, Any] | DataDescription,
    return_meta: Literal[True],
    constants: Mapping[str, JSONValue] | None = ...,
    skip_postprocessing: bool = ...,
    **kwargs: object,
) -> tuple[pd.DataFrame, DataMeta]: ...


def read_and_process_data(
    desc: str | dict[str, Any] | DataDescription,
    return_meta: bool = False,
    constants: Mapping[str, JSONValue] | None = None,
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
            # Note: expects merge files to have corresponding meta in global DataMeta file
            oldcols = globs["df"].columns
            globs["df"] = _perform_merges(globs["df"], desc_obj.merge, constants, meta_obj)
            if kwargs.get("data_meta", False):
                newcols = [col for col in globs["df"].columns if col not in oldcols]
                globs["df"].loc[:, newcols] = fix_df_with_meta(globs["df"][newcols], kwargs["data_meta"])
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
    schema_metadata = schema.metadata or {}
    if custom_meta_key.encode() in schema_metadata:
        restored_meta_json = schema_metadata[custom_meta_key.encode()]
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
    schema_metadata = restored_table.schema.metadata or {}
    if custom_meta_key.encode() in schema_metadata:
        restored_meta_json = schema_metadata[custom_meta_key.encode()]
        restored_meta_payload = cast(dict[str, object], json.loads(restored_meta_json))
        restored_meta = soft_validate(restored_meta_payload, ParquetMeta)
    else:
        restored_meta = None

    return restored_df, restored_meta
