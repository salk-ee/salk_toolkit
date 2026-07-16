"""Public entry points: read annotated datasets and DataDescription-driven data assembly."""

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Callable, Literal, cast, overload

import numpy as np
import pandas as pd
import pyreadstat  # type: ignore[import-untyped]
import scipy as sp

import salk_toolkit as stk
from salk_toolkit import utils
from salk_toolkit.utils import (
    JSONValue,
    cached_fn,
    is_datetime,
    warn,
)
from salk_toolkit.validation import (
    DataDescription,
    DataMeta,
    SingleMergeSpec,
    soft_validate,
)

from salk_toolkit.io.core import (
    PROVENANCE_COLUMNS,
    ROW_ID,
    ProcessedDataReturn,
    _str_from_list,
    assert_row_id_intact,
    finalize_row_index,
    mint_positional_row_id,
)
from salk_toolkit.io.meta import _fix_meta_categories, _is_categorical, fix_df_with_meta
from salk_toolkit.io.parquet import read_parquet_with_metadata
from salk_toolkit.io.pipeline import _process_annotated_data
from salk_toolkit.io.sources import _load_data_files


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
        # Idempotent for the json/yaml path (already row_id-indexed); mints positional ids for a
        # legacy parquet written before stable ids existed.
        data = finalize_row_index(data)
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
    # finalize_row_index keeps this path's return shape consistent with the annotated branch
    # above (the return_raw path in _process_annotated_data does not index by row_id on its own).
    if return_meta:
        idata, imeta = _process_annotated_data(
            data_file=fname,
            meta=inferred_meta,
            return_meta=True,
            return_raw=return_raw,
            ignore_exclusions=ignore_exclusions,
            only_fix_categories=only_fix_categories,
            add_original_inds=add_original_inds,
        )
        return (finalize_row_index(idata), imeta)
    else:
        return finalize_row_index(
            _process_annotated_data(
                data_file=fname,
                meta=inferred_meta,
                return_meta=False,
                return_raw=return_raw,
                ignore_exclusions=ignore_exclusions,
                only_fix_categories=only_fix_categories,
                add_original_inds=add_original_inds,
            )
        )


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
            df = pd.read_parquet(data_file, **read_opts)  # type: ignore[call-overload]
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


def _repair_merge_row_ids(mdf: pd.DataFrame, tag: str) -> None:
    """Restore ROW_ID uniqueness after a horizontal merge, in place (corner case).

    A merge keeps the left row's id, but ``right``/``outer`` joins introduce right-only rows
    with no left id, and one-to-many / ``cross`` joins duplicate an id across the fanned-out
    rows. Right-only rows get a fresh ``{tag}::{position}`` id; duplicated ids get a ``::m{k}``
    per-group suffix. Both are deterministic since merge output order is deterministic.
    """
    rid = mdf[ROW_ID]
    na_mask = rid.isna()
    if na_mask.any():
        mdf.loc[na_mask, ROW_ID] = f"{tag}::" + pd.RangeIndex(int(na_mask.sum())).astype(str)
        rid = mdf[ROW_ID]
    dup_mask = rid.duplicated(keep=False)
    if dup_mask.any():
        cc = mdf.loc[dup_mask].groupby(ROW_ID).cumcount()
        mdf.loc[dup_mask, ROW_ID] = rid[dup_mask].astype(str) + "::m" + cc.astype(str)


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

    for merge_idx, ms in enumerate(merges):
        # ms is always a SingleMergeSpec Pydantic model
        file = ms.file
        on = ms.on if isinstance(ms.on, list) else [ms.on]
        add = ms.add
        how = ms.how

        ndf = read_and_process_data(file, constants=constants).reset_index()  # lift row_id out of the index
        if add:
            ms_on = on
            ms_add = list(add) if isinstance(add, list) else [add]
            ndf = ndf[ms_on + ms_add]
        # Drop system provenance columns from the merge-side to avoid suffix collisions.
        # Keep them on the left df (the main dataset) only.
        ndf = ndf.drop(
            columns=[c for c in [*PROVENANCE_COLUMNS, ROW_ID] if c in ndf.columns and c not in on],
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
        if ROW_ID in mdf.columns:
            _repair_merge_row_ids(mdf, f"merge{merge_idx}")

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
        mint_positional_row_id(df)  # inline data has no source-file ids
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
            assert_row_id_intact(globs["df"], "DataDescription preprocessing")

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
            assert_row_id_intact(globs["df"], "DataDescription postprocessing")
        df = globs["df"]

    # Stable, unique, deterministic index at the return boundary.
    df = finalize_row_index(df)

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
