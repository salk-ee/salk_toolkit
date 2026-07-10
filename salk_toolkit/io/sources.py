"""Loading of data sources into per-file SourceBundles, including nested annotated sources."""

import os
from collections.abc import Iterable
from typing import Any, cast

import pandas as pd

from salk_toolkit import utils
from salk_toolkit.utils import (
    read_json,
    read_yaml,
    warn,
)
from salk_toolkit.validation import (
    DataMeta,
    FileDesc,
    soft_validate,
)

from salk_toolkit.io import readers
from salk_toolkit.io.core import Dataset, ProcessOpts, SourceBundle, _deterministic_categories_and_values
from salk_toolkit.io.meta import _fix_meta_categories, _merge_data_metas
from salk_toolkit.io.parquet import read_parquet_with_metadata
from salk_toolkit.io.pipeline import process


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


def _load_data_files(
    data_files: list[FileDesc],
    path: str | None,
    read_opts: dict[str, Any] | None = None,
    opts: ProcessOpts = ProcessOpts(),
) -> SourceBundle:
    """Internal helper to load files defined in metadata or descriptions.

    Returns a SourceBundle keeping the per-file dataframes separate for per-file processing.
    """

    raw_data_dict: dict[str, pd.DataFrame] = {}
    metas: list[DataMeta] = []
    einfo: dict[str, object] = {}
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
        fopts = fd.opts or read_opts
        result_meta: DataMeta | None = None
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
            raw_data, result_meta = _load_dataset(cast(str, data_file), opts)
            if result_meta is not None:
                metas.append(soft_validate(result_meta, DataMeta, warnings=True))
        elif extension in readers._TABULAR_EXTENSIONS:
            raw_data, fenv = readers._read_tabular(
                cast(str, mapped_file), extension, cast(dict[str, Any], fopts) if fopts else {}
            )
            einfo.update(fenv)  # Allow the fields in reader meta to be used just like self-defined constants
        else:
            raise Exception(f"Not a known file format for {data_file}: {extension}")

        readers.stk_loaded_files_set.add(mapped_file)

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
            result_meta is not None and result_meta.files is not None and len(result_meta.files) > 1
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

    if not metas:  # Do we have any metainfo?
        return SourceBundle(frames=raw_data_dict, env=einfo)

    # Union block structure (blocks + column lists) across all annotated sources in file order
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
    return SourceBundle(frames=raw_data_dict, env=einfo, meta=meta)


def _read_meta_input(meta_fname: str | None, meta: DataMeta | dict[str, object] | None) -> DataMeta:
    """Load the metafile (or accept an in-memory meta dict) and validate to DataMeta."""
    meta_input: DataMeta | dict[str, object] | None = meta
    if meta_fname is not None:
        metafile = cast(dict[str, str], readers.stk_file_map).get(meta_fname, meta_fname)
        ext = os.path.splitext(metafile)[1]
        if ext == ".yaml":
            meta_raw = read_yaml(metafile)
        elif ext == ".json":
            meta_raw = read_json(metafile)
        else:
            raise Exception(f"Unknown meta file format {ext} for file: {meta_fname}")
        assert isinstance(meta_raw, dict), "Meta file must contain a dict"
        meta_input = dict(meta_raw)  # Cast to ensure object values
    if meta_input is None:
        raise ValueError("Metadata cannot be None")
    return soft_validate(meta_input, DataMeta, warnings=True)


def _load_meta_sources(
    meta_obj: DataMeta, meta_fname: str | None, data_file: str | None, opts: ProcessOpts
) -> SourceBundle:
    """Resolve the meta's file list (or a data_file override) and load it into a SourceBundle."""
    if data_file is not None:
        # data_file override: use it directly
        files_list = [FileDesc(file=data_file, opts=meta_obj.read_opts)]
    elif meta_obj.files is not None:
        files_list = meta_obj.files
    else:
        raise ValueError("No files provided in metadata")

    bundle = _load_data_files(
        files_list,
        path=meta_fname if meta_fname is not None else data_file,
        read_opts=meta_obj.read_opts,
        opts=opts,
    )
    if bundle.meta is not None:
        warn("Processing main meta file")  # Print this to separate warnings for input jsons from main
    return bundle


def _load_dataset(path: str, opts: ProcessOpts) -> Dataset:
    """Load an already-annotated source: a metafile or a parquet with embedded meta. No inference."""
    ext = os.path.splitext(path)[1]
    if ext in {".json", ".yaml"}:
        return _process_annotated_data(meta_fname=path, opts=opts)
    df, full_meta = read_parquet_with_metadata(path)
    return Dataset(df, full_meta.data if full_meta is not None else None)


# Default usage with mature metafile: _process_annotated_data(<metafile name>)
# When figuring out the metafile, it can also be run as: _process_annotated_data(meta=<dict>, data_file=<>)


def _process_annotated_data(
    meta_fname: str | None = None,
    meta: DataMeta | dict[str, object] | None = None,
    data_file: str | None = None,
    opts: ProcessOpts = ProcessOpts(),
    return_raw: bool = False,
) -> Dataset:
    """Process annotated data according to metadata specifications."""
    meta_obj = _read_meta_input(meta_fname, meta)
    bundle = _load_meta_sources(meta_obj, meta_fname, data_file, opts)
    if return_raw:  # Concatenated raw data, for debugging metafiles
        return Dataset(bundle.concat(), meta_obj)
    return process(bundle, meta_obj, opts)
