"""Loading of data files into per-file dataframes, including nested annotated sources."""

import os
import warnings
from collections.abc import Iterable
from typing import Any, cast

import pandas as pd
import pyreadstat  # type: ignore[import-untyped]

from salk_toolkit import utils
from salk_toolkit.utils import (
    warn,
)
from salk_toolkit.validation import (
    DataMeta,
    FileDesc,
    soft_validate,
)

from salk_toolkit.io import readers
from salk_toolkit.io.core import ROW_ID, _deterministic_categories_and_values
from salk_toolkit.io.meta import _fix_meta_categories


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


def _assign_row_id(df: pd.DataFrame, file_code: str, id_col: str | None, data_file: str) -> None:
    """Assign or extend the stable ``ROW_ID`` column in place.

    Nested/annotated sources already carry a row id (as a column after the caller lifts it
    from the index) - we only prepend this level's ``file_code``. Raw leaf files get a fresh
    ``{file_code}::{leaf}`` id, where ``leaf`` is the declared ``id_col`` value (validated
    unique + non-null) or the 0-based within-file position.
    """
    prefix = f"{file_code}::"
    if ROW_ID in df.columns:  # nested annotated source: extend the existing path
        df[ROW_ID] = prefix + df[ROW_ID].astype(str)
        return
    if id_col is not None:  # natural key: survives row reorders / re-exports of the source
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not found in {data_file}")
        col = df[id_col]
        if col.isna().any():
            raise ValueError(f"id_col '{id_col}' has null values in {data_file}")
        if col.duplicated().any():
            raise ValueError(f"id_col '{id_col}' is not unique within {data_file}")
        df[ROW_ID] = prefix + col.astype(str)
    else:  # fall back to within-file position (deterministic for a byte-stable source)
        df[ROW_ID] = [f"{prefix}{i}" for i in range(len(df))]


def _load_data_files(
    data_files: list[FileDesc],
    path: str | None,
    read_opts: dict[str, Any] | None = None,
    ignore_exclusions: bool = False,
    only_fix_categories: bool = False,
    add_original_inds: bool = False,
    id_col: str | None = None,
) -> tuple[dict[str, pd.DataFrame], DataMeta | None, dict[str, object]]:
    """Internal helper to load files defined in metadata or descriptions.

    Returns per-file dataframes keyed by file code, keeping them separate for per-file processing.
    """

    from salk_toolkit.io.datasets import read_annotated_data  # deferred import breaks the sources<->datasets cycle

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
        mapped_file = readers.stk_file_map.get(cast(str, data_file), cast(str, data_file))

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

        readers.stk_loaded_files_set.add(mapped_file)

        # A nested/annotated source (or a parquet with embedded ids) comes back indexed by
        # ROW_ID; lift it to a column so it survives the internal reset_index(drop=True) churn
        # (re-indexed at the outer boundary) and so _assign_row_id can extend the path below.
        if raw_data.index.name == ROW_ID:
            raw_data = raw_data.reset_index()

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

        # Stamp the stable row id (per-file id_col overrides the meta-level default).
        _assign_row_id(raw_data, file_code, fd.id_col or id_col, cast(str, data_file))

        raw_data_dict[file_code] = raw_data

    if metas:  # Do we have any metainfo?
        meta = metas[-1]

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
        # TODO: one should also merge the structures in case the columns don't match
        return raw_data_dict, meta, einfo
    else:
        return raw_data_dict, None, einfo
