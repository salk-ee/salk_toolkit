"""The annotation pipeline: _process_annotated_data turns loaded source files into a processed dataframe + meta."""

import os
from typing import Literal, cast, overload

import numpy as np
import pandas as pd
import scipy as sp

import salk_toolkit as stk
from salk_toolkit.utils import (
    read_json,
    read_yaml,
    warn,
)
from salk_toolkit.validation import (
    ColumnBlockMeta,
    ColumnMeta,
    DataMeta,
    FileDesc,
    soft_validate,
)

from salk_toolkit.io import readers
from salk_toolkit.io.core import (
    ROW_ID,
    ProcessedDataReturn,
    _deterministic_categories_and_values,
    _is_series_of_lists,
    _str_from_list,
    assert_row_id_intact,
    finalize_row_index,
    mint_positional_row_id,
)
from salk_toolkit.io.create_blocks import _create_new_columns_and_metas
from salk_toolkit.io.meta import _fix_meta_categories
from salk_toolkit.io.sources import _load_data_files


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
    metafile = cast(dict[str, str], readers.stk_file_map).get(meta_fname, meta_fname)  # type: ignore[call-overload]
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
            id_col=meta_obj.id_col,
        )
        if inp_meta is not None:
            warn("Processing main meta file")  # Print this to separate warnings for input jsons from main
    elif isinstance(raw_data, dict):
        # Directly-injected frames may lack ids: mint a positional one per file_code.
        raw_data_dict = {
            fc: (df if ROW_ID in df.columns else mint_positional_row_id(df.copy(), fc)) for fc, df in raw_data.items()
        }
        einfo = {}
    else:
        # Backward compatibility: single DataFrame -> treat as single-file dict
        single = raw_data if ROW_ID in raw_data.columns else mint_positional_row_id(raw_data.copy(), "F0")
        raw_data_dict = {"F0": single}
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
            assert_row_id_intact(raw_data_dict[file_code], f"preprocessing of {file_code}")

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
    sys_block = soft_validate(sys_block_dict, ColumnBlockMeta)
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
                    dropped = set(s.dropna().unique()) - set(cats)
                    if dropped:
                        warn(f"Values for {cn} not in categories and will be dropped: {dropped}")
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
            create_source_df = ndf_df.combine_first(raw_data_concat) if not raw_data_concat.empty else ndf_df
            for newdf, newmeta_dict in _create_new_columns_and_metas(
                create_source_df,
                group,
                topics=constants.get("topics", None),
                sets=constants.get("sets", None),
            ):
                ndf_df = ndf_df.combine_first(newdf)
                new_group_meta = soft_validate(newmeta_dict, ColumnBlockMeta)
                new_structure[new_group_meta.name] = new_group_meta
            # Create a copy of group without the create field
            group = group.model_copy(update={"create": None})

        # Add processed group to structure
        new_structure[group.name] = group

    # Carry the stable row id through: it is a system column (not part of the meta structure)
    # so it is not rebuilt column-by-column. ndf_df is positionally aligned with raw_data_concat
    # here (same rows, same order), so copy it over by position.
    if not raw_data_concat.empty:
        if len(ndf_df) != len(raw_data_concat):
            raise ValueError("row count changed during column processing - cannot align stable row ids")
        ndf_df[ROW_ID] = raw_data_concat[ROW_ID].to_numpy()

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
        assert_row_id_intact(ndf_df, "postprocessing")

    # Update meta with new structure (including any groups created during processing)
    meta_obj = meta_obj.model_copy(update={"structure": new_structure})

    # Fix categories after postprocessing
    # Also replaces infer with the actual categories
    meta_obj = _fix_meta_categories(meta_obj, ndf_df, warnings=True)
    # Ensure we have a DataMeta object
    if not isinstance(meta_obj, DataMeta):
        meta_obj = soft_validate(meta_obj, DataMeta)

    # Positional counter over the processed frame - kept only on request; no longer drives exclusions.
    ndf_df["original_inds"] = np.arange(len(ndf_df))

    # Apply meta exclusions by stable row id (composes across nesting - an id already filtered
    # by an inner meta simply matches nothing here).
    if meta_obj.excluded and not ignore_exclusions:
        excl_ids = [rid for rid, _ in meta_obj.excluded]
        present = set(ndf_df[ROW_ID])
        missing = [rid for rid in excl_ids if rid not in present]
        if missing:
            warn(f"{len(missing)} excluded row_id(s) not present in data (already filtered upstream?): {missing[:5]}")
        ndf_df = ndf_df[~ndf_df[ROW_ID].isin(excl_ids)]

    if not add_original_inds:
        ndf_df.drop(columns=["original_inds"], inplace=True)

    # Stable, unique, deterministic index at the return boundary.
    ndf_df = finalize_row_index(ndf_df)

    # Return with meta as dict if requested (for backward compatibility)
    if return_meta:
        return (ndf_df, meta_obj)
    return ndf_df
