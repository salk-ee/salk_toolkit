"""Metadata introspection, category reconciliation and dtype-fixing for DataMeta structures."""

from collections.abc import Sequence
from typing import TypeVar

import pandas as pd

from salk_toolkit import utils
from salk_toolkit.utils import (
    is_datetime,
    warn,
)
from salk_toolkit.validation import (
    ColumnBlockMeta,
    ColumnMeta,
    DataMeta,
    GroupOrColumnMeta,
    soft_validate,
)

from salk_toolkit.io.core import _deterministic_categories_and_values


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


def _is_categorical(col: pd.Series) -> bool:
    """Check if a pandas Series is categorical."""
    return col.dtype.name in ["object", "str", "category"] and not is_datetime(col)
