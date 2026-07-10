"""The annotation pipeline: stages that turn a SourceBundle + DataMeta into a processed Dataset."""

from typing import cast

import numpy as np
import pandas as pd

from salk_toolkit.utils import warn
from salk_toolkit.validation import (
    ColumnBlockMeta,
    ColumnMeta,
    DataMeta,
    MaxDiffBlock,
    OneHotBlock,
    TopKBlock,
    soft_validate,
)

from salk_toolkit.io.core import (
    Dataset,
    HookEnv,
    ProcessOpts,
    SourceBundle,
    _deterministic_categories_and_values,
    _is_series_of_lists,
)
from salk_toolkit.io.create_blocks import (
    _combine_first_preserving_order,
    _demote_to_plain,
    _process_block,
)
from salk_toolkit.io.meta import _fix_meta_categories


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


def _inject_files_block(bundle: SourceBundle, meta_obj: DataMeta, file_names: dict[str, str]) -> DataMeta:
    """(Re)stamp provenance columns and add the generated `files` block to the structure.

    The columns are overwritten to guarantee correctness even if preprocessing mutated/dropped them.
    The block gets explicit ordered categories (no "infer") for determinism, so the system file
    columns can be used downstream (e.g. plotting/pipeline).
    """
    for fc, fdf in bundle.frames.items():
        fdf["file_code"] = str(fc)
        fdf["file_name"] = file_names[fc]

    sys_block_name = "files"
    sys_block_hidden = len(bundle.frames) <= 1
    sys_block_dict: dict[str, object] = {
        "name": sys_block_name,
        "generated": True,
        "hidden": sys_block_hidden,
        "columns": {
            "file_code": {"categories": [str(fc) for fc in bundle.frames], "ordered": True},
            "file_name": {"categories": [file_names[fc] for fc in bundle.frames], "ordered": True},
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
    return meta_obj.model_copy(update={"structure": structure2})


def _gather_source(
    bundle: SourceBundle, source_spec: str | dict[str, str], orig_cn: str, cn: str, generated: bool
) -> pd.Series | None:
    """Resolve and concatenate a column's source series across all files.

    Returns None when the column is missing/empty everywhere: declared in meta but absent
    from the source files -> skipped in df (meta categories are reconciled later).
    """
    per_file_series: list[pd.Series] = []
    for file_code, file_raw_data in bundle.frames.items():
        # Determine source column name for this file:
        # a dict maps file codes (with 'default' then orig_cn as fallbacks); a string applies to all files
        if isinstance(source_spec, dict):
            sn = source_spec.get(file_code, source_spec.get("default", orig_cn))
        else:
            sn = source_spec

        if sn not in file_raw_data:
            if not generated:  # bypass warning for columns marked as being generated later
                warn(f"Column {sn} not found in file {file_code}")
            s = pd.Series(index=file_raw_data.index, dtype=object, name=cn)  # Empty series with same index
        elif file_raw_data[sn].isna().all():
            warn(f"Column {sn} is empty in file {file_code} and thus ignored")
            s = pd.Series(index=file_raw_data.index, dtype=object, name=cn)  # Empty series with same index
        else:
            s = file_raw_data[sn].copy()
            s.name = cn  # Set name early
        per_file_series.append(s)

    if not per_file_series:
        return None
    s = pd.concat(per_file_series).reset_index(drop=True)
    return None if s.isna().all() else s


def _apply_transforms(
    s: pd.Series, mcm: ColumnMeta, bundle: SourceBundle, ndf_df: pd.DataFrame, cn: str, hooks: HookEnv
) -> pd.Series:
    """Apply translate -> transform -> translate_after -> dtype coercion to a gathered series."""
    if _is_series_of_lists(s):
        return s
    if s.dtype.name == "category":
        s = s.astype("object")  # This makes it easier to use common ops like replace and fillna
    if mcm.translate:
        s = s.astype("str").replace(mcm.translate).replace("nan", None).replace("None", None)
    if mcm.transform is not None and isinstance(mcm.transform, str):
        # Transforms are evaluated per-file so they can reference the file's raw data (df)
        # and the per-file view of the columns processed so far (ndf)
        transformed_parts: list[pd.Series] = []
        ranges = bundle.ranges()
        for file_code, file_raw_data in bundle.frames.items():
            s_local = s.iloc[ranges[file_code]]
            transformed = hooks.eval(mcm.transform, s=s_local, df=file_raw_data, ndf=ndf_df.iloc[ranges[file_code]])
            transformed_parts.append(pd.Series(transformed, index=s_local.index, name=cn))
        s = pd.concat(transformed_parts, ignore_index=True)
    if mcm.translate_after:
        s = pd.Series(s).astype("str").replace(mcm.translate_after).replace("nan", None).replace("None", None)

    if mcm.datetime:
        s = pd.to_datetime(s, errors="coerce")
    elif mcm.continuous:
        s = pd.to_numeric(s, errors="coerce")
    return s


def _resolve_categories(s: pd.Series, mcm: ColumnMeta, cn: str) -> tuple[pd.Series, ColumnMeta]:
    """Infer or coerce the series' categories, returning the updated column meta."""
    if not mcm.categories or _is_series_of_lists(s):
        return s, mcm

    if mcm.categories == "infer":
        # Check for ordered warning before we modify s
        should_warn_ordered = mcm.ordered and not pd.api.types.is_numeric_dtype(s)
        # s is the source of truth: it has the final translate -> transform -> translate_after values
        present = set(s.dropna().unique())
        if mcm.translate and not mcm.transform and set(mcm.translate.values()) >= present:
            # Infer order from the translation dict; mapping can be many-to-one so dedup preserving order
            cats = list(dict.fromkeys(str(c) for c in mcm.translate.values() if c in present))
        else:
            # Deterministic ordering: numeric dtype and numeric-like strings sort numerically,
            # otherwise lexicographically
            if should_warn_ordered:
                warn(
                    f"Ordered category {cn} had category: infer. "
                    "This only works correctly if you want lexicographic ordering!"
                )
            s, cats = _deterministic_categories_and_values(s)
        mcm = mcm.model_copy(update={"categories": cats})  # Persist inferred categories into metadata
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
            distances = np.abs(s_values_arr.reshape(-1, 1) - fcats.reshape(1, -1))
            s = pd.Series(
                np.array(cats)[distances.argmin(axis=1)],
                index=s.index,
                name=s.name,
                dtype=pd.CategoricalDtype(categories=cats, ordered=mcm.ordered if mcm.ordered is not None else False),
            )
            return s, mcm  # Snapping cannot drop values, and s is already the right Categorical
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
        final_ordered = mcm.ordered if mcm.ordered is not None else False
        s = pd.Series(pd.Categorical(s, categories=cats, ordered=final_ordered), name=cn)
    return s, mcm


def _apply_subgroup_transform(
    bundle: SourceBundle, ndf_df: pd.DataFrame, transform: str, g_cols: list[str], hooks: HookEnv
) -> pd.DataFrame:
    """Apply a block-level subgroup transform per-file over the block's columns."""
    # TODO: Add a subgroups field to ColumnBlockMeta if per-subgroup application is needed
    transformed_parts: list[pd.DataFrame] = []
    ranges = bundle.ranges()
    for file_code, file_raw_data in bundle.frames.items():
        idx_range = ranges[file_code]
        transformed_gdf = hooks.eval(
            transform,
            gdf=ndf_df[g_cols].iloc[idx_range].copy(),
            ndf=ndf_df.iloc[idx_range].copy(),  # Per-file processed data view
            df=file_raw_data,  # Per-file raw data
        )
        transformed_parts.append(cast(pd.DataFrame, transformed_gdf))
    transformed_combined = pd.concat(transformed_parts).reset_index(drop=True)
    ndf_df[g_cols] = transformed_combined[g_cols].values
    return ndf_df


def _build_columns(bundle: SourceBundle, meta_obj: DataMeta, hooks: HookEnv) -> tuple[pd.DataFrame, DataMeta]:
    """Build the processed dataframe column by column from the meta structure.

    Returns the new dataframe and meta with the processed structure (create blocks
    expanded into their generated groups).
    """
    raw_data_concat = bundle.concat(reset_index=True)
    ndf_df = pd.DataFrame()  # Start empty, built column by column

    all_cns: dict[str, str] = {}  # column/group name -> group that claimed it, for duplicate detection
    # Build new structure dict as we process groups (Pydantic models are immutable)
    new_structure: dict[str, ColumnBlockMeta] = {}
    for group in meta_obj.structure.values():
        if group.name in all_cns:
            raise Exception(f"Group name {group.name} duplicates a column name in group {all_cns[group.name]}")
        all_cns[group.name] = group.name

        # Col prefix is used to avoid name clashes when different groups naturally share same column names
        col_prefix = group.scale.col_prefix if group.scale is not None else None

        # Note: scale metadata is already merged with column metadata by the
        # ColumnBlockMeta.merge_scale_with_columns validator. Missing columns are not removed
        # from meta, only skipped in df; their categories are reconciled later by _fix_meta_categories.
        g_cols = []
        for orig_cn, mcm in group.columns.items():
            source_spec = mcm.source if mcm.source is not None else orig_cn
            cn = col_prefix + orig_cn if col_prefix is not None else orig_cn

            # Detect duplicate columns in meta - including among those missing or generated
            if cn in all_cns:
                raise Exception(f"Duplicate column name found: '{cn}' in {all_cns[cn]} and {group.name}")
            all_cns[cn] = group.name

            s = _gather_source(bundle, source_spec, orig_cn, cn, group.generated)
            if s is None:
                continue
            s = _apply_transforms(s, mcm, bundle, ndf_df, cn, hooks)
            s.name = cn  # Ensure name is set
            s, mcm = _resolve_categories(s, mcm, cn)

            ndf_df[cn] = s
            g_cols.append(cn)

        if group.subgroup_transform is not None:
            ndf_df = _apply_subgroup_transform(bundle, ndf_df, group.subgroup_transform, g_cols, hooks)

        if isinstance(group, (TopKBlock, MaxDiffBlock, OneHotBlock)):
            # Specialized blocks: any explicitly declared raw columns were already processed
            # as plain columns above; they are kept under a demoted plain block while the
            # transform fans the block out into its derived sibling blocks.
            demoted = _demote_to_plain(group)
            new_structure[demoted.name] = demoted
            source_df = _combine_first_preserving_order(ndf_df, raw_data_concat)
            for sdf, smeta in _process_block(group, source_df):
                for c in sdf.columns:
                    ndf_df[c] = sdf[c]
                new_structure[smeta.name] = smeta
        else:
            new_structure[group.name] = group

    # Update meta with new structure (including any groups created during processing)
    return ndf_df, meta_obj.model_copy(update={"structure": new_structure})


def _recast_hook_created_columns(ndf_df: pd.DataFrame, meta_obj: DataMeta) -> None:
    """Snap columns that postprocessing created (or replaced) back to their declared
    categorical dtype. Columns already categorical were cast during the main pass and
    are left untouched, preserving their inferred category order."""
    for group in meta_obj.structure.values():
        if group.type != "plain":
            continue
        prefix = group.scale.col_prefix if group.scale and group.scale.col_prefix else ""
        for orig_cn, mcm in group.columns.items():
            cn = prefix + orig_cn
            if cn not in ndf_df.columns or ndf_df[cn].dtype.name == "category":
                continue
            s, _ = _resolve_categories(ndf_df[cn], mcm, cn)
            ndf_df[cn] = s


def _apply_exclusions(ndf_df: pd.DataFrame, meta_obj: DataMeta, opts: ProcessOpts) -> Dataset:
    """Filter out rows listed in meta `excluded` and optionally keep the original_inds column."""
    ndf_df["original_inds"] = np.arange(len(ndf_df))
    if meta_obj.excluded and not opts.ignore_exclusions:
        excl_inds = [i for i, _ in meta_obj.excluded]
        ndf_df = ndf_df[~ndf_df["original_inds"].isin(excl_inds)]
    if not opts.add_original_inds:
        ndf_df.drop(columns=["original_inds"], inplace=True)
    return Dataset(ndf_df, meta_obj)


def process(bundle: SourceBundle, meta_obj: DataMeta, opts: ProcessOpts) -> Dataset:
    """Run the annotation pipeline stages on loaded source data."""
    hooks = HookEnv(bundle.env, meta_obj.constants)
    file_names = _file_meta_map(bundle.frames)

    # Run preprocessing per file
    if meta_obj.preprocessing is not None:
        for fc in bundle.frames:
            bundle.frames[fc] = hooks.exec_df(
                meta_obj.preprocessing, bundle.frames[fc], file_code=fc, file_name=file_names[fc]
            )

    # File provenance columns must survive end-to-end (also if preprocessing dropped/mutated them)
    meta_obj = _inject_files_block(bundle, meta_obj, file_names)

    ndf_df, meta_obj = _build_columns(bundle, meta_obj, hooks)

    if meta_obj.postprocessing is not None:
        ndf_df = hooks.exec_df(meta_obj.postprocessing, ndf_df)
        _recast_hook_created_columns(ndf_df, meta_obj)

    # Fix categories after postprocessing; also replaces "infer" with the actual categories
    meta_obj = _fix_meta_categories(meta_obj, ndf_df, warnings=True)

    return _apply_exclusions(ndf_df, meta_obj, opts)
