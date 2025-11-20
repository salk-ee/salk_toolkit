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
    "get_loaded_files",
    "reset_file_tracking",
    "get_file_map",
    "set_file_map",
    "process_annotated_data",
    "read_annotated_data",
    "fix_df_with_meta",
    "extract_column_meta",
    "group_columns_dict",
    "list_aliases",
    "change_df_to_meta",
    "update_meta_with_model_fields",
    "replace_data_meta_in_parquet",
    "fix_meta_categories",
    "fix_parquet_categories",
    "infer_meta",
    "data_with_inferred_meta",
    "read_and_process_data",
    "find_type_in_dict",
    "write_parquet_with_metadata",
    "read_parquet_metadata",
    "read_parquet_with_metadata",
]

import json
import os
import warnings
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import scipy as sp
import polars as pl


import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

import salk_toolkit as stk
from salk_toolkit.utils import (
    replace_constants,
    is_datetime,
    warn,
    cached_fn,
    read_yaml,
    read_json,
)
from salk_toolkit.validation import DataMeta, DataDescription, soft_validate

# Ignore fragmentation warnings
warnings.filterwarnings("ignore", "DataFrame is highly fragmented.*", pd.errors.PerformanceWarning)


def str_from_list(val: list[str] | object) -> str:
    """Convert a list to a newline-separated string, or return string representation.

    Args:
        val: Value to convert (list or other object).

    Returns:
        Newline-separated string if val is a list, otherwise str(val).
    """
    if isinstance(val, list):
        return "\n".join(val)
    return str(val)


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


# Read files listed in meta['file'] or meta['files']
def read_concatenate_files_list(
    meta: dict[str, object], data_file: str | None = None, path: str | None = None, **kwargs: object
) -> tuple[pd.DataFrame, dict[str, object] | None, dict[str, object]]:
    """Read and concatenate multiple data files listed in metadata.

    Args:
        meta: Metadata dictionary containing file paths and read options.
        data_file: Optional single data file path (overrides meta['file']).
        path: Optional base path for relative file paths.
        **kwargs: Additional arguments passed to file readers.

    Returns:
        Tuple of (concatenated DataFrame, merged metadata, extra info dictionary).
    """
    global stk_loaded_files_set, stk_file_map

    opts = meta["read_opts"] if "read_opts" in meta else {}
    if data_file:
        data_files = [{"file": data_file, "opts": opts}]
    elif meta.get("file"):
        data_files = [{"file": meta["file"], "opts": opts}]
    elif meta.get("files"):
        data_files = meta["files"]
    else:
        raise Exception("No files provided")

    data_files = [{"opts": opts, **f} if isinstance(f, dict) else {"opts": opts, "file": f} for f in data_files]

    cat_dtypes = {}
    raw_dfs, metas, einfo = [], [], {}
    for fi, fd in enumerate(data_files):
        data_file, opts = fd["file"], fd["opts"]
        file_code = fd.get("code", f"F{fi}")  # Default to F0, F1, F2, etc.
        if path:
            data_file = os.path.join(os.path.dirname(path), data_file)
        mapped_file = stk_file_map.get(data_file, data_file)

        extension = os.path.splitext(data_file)[1][1:].lower()
        if extension in [
            "json",
            "parquet",
            "yaml",
        ]:  # Allow loading metafiles or annotated data
            if extension == "json":
                warn(f"Processing {data_file}")  # Print this to separate warnings for input jsons from main
            # Pass in orig_data_file here as it might loop back to this function here and we need to preserve paths
            raw_data, meta = read_annotated_data(data_file, infer=False, return_meta=True, **kwargs)
            if meta is not None:
                metas.append(meta)
        elif extension in ["csv", "gz"]:
            raw_data = pd.read_csv(mapped_file, low_memory=False, **opts)
        elif extension in ["sav", "dta"]:
            read_fn = getattr(pyreadstat, "read_" + (mapped_file[-3:]).lower())
            with warnings.catch_warnings():  # While pyreadstat has not been updated to pandas 2.2 standards
                warnings.simplefilter("ignore")
                raw_data, fmeta = read_fn(
                    mapped_file,
                    **{"apply_value_formats": True, "dates_as_pandas_datetime": True},
                    **opts,
                )
                einfo.update(fmeta.__dict__)  # Allow the fields in meta to be used just like self-defined constants
        elif extension in ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]:
            raw_data = pd.read_excel(mapped_file, **opts)
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
            cat_dtypes["file_code"] = None  # Mark file_code as categorical
        for k, v in fd.items():
            if k in ["opts", "code"]:
                continue  # Skip opts and code (already handled)
            if len(data_files) <= 1 and k in ["file"]:
                continue
            raw_data[k] = v
            if isinstance(v, str):
                cat_dtypes[k] = None

        # Handle categorical types for more complex situations
        for c in raw_data.columns:
            if raw_data[c].dtype.name == "object" and not isinstance(raw_data[c].dropna().iloc[0], list):
                cat_dtypes[c] = cat_dtypes.get(c, None)  # Infer a categorical type unless already given
            elif (
                raw_data[c].dtype.name == "category" and len(data_files) > 1
            ):  # Strip categories when multiple files involved
                if c not in cat_dtypes or len(cat_dtypes[c].categories) <= len(raw_data[c].dtype.categories):
                    cat_dtypes[c] = raw_data[c].dtype
                raw_data[c] = raw_data[c].astype("object")

        raw_dfs.append(raw_data)

    fdf = pd.concat(raw_dfs)

    # Restore categoricals
    if len(cat_dtypes) > 0:
        for c, dtype in cat_dtypes.items():
            if dtype is None:  # Added as an extra field, infer categories
                s = fdf[c].dropna()
                if (
                    s.dtype.name == "object"
                    and not isinstance(s.iloc[0], str)  # Check for string as string is also iterable
                    and isinstance(s.iloc[0], Iterable)
                ):
                    continue  # Skip if it's a list or tuple or ndarray
                dtype = pd.Categorical([], list(s.unique())).dtype
            elif not set(fdf[c].dropna().unique()) <= set(
                dtype.categories
            ):  # If the categories are not the same, create a new dtype
                # print(set(fdf[c].dropna().unique()), set(dtype.categories))
                dtype = pd.Categorical([], list(fdf[c].dropna().unique())).dtype
                warn(f"Categories for {c} are different between files - merging to total {len(dtype.categories)} cats")
            fdf[c] = pd.Categorical(fdf[c], dtype=dtype)

    if metas:  # Do we have any metainfo?
        meta = metas[-1]
        # This will fix categories inside meta too
        fix_meta_categories(meta, fdf, warnings=False)
        # TODO: one should also merge the structures in case the columns don't match
        return fdf, meta, einfo
    else:
        return fdf, None, einfo


# convert number series to categorical, avoiding long and unweildy fractions like 24.666666666667
# This is a practical judgement call right now - round to two digits after comma and remove .00 from integers
def convert_number_series_to_categorical(s: pd.Series) -> pd.Series:
    """Convert numeric series to categorical with formatted strings.

    Args:
        s: Numeric pandas Series to convert.

    Returns:
        Series with formatted string values (2 decimal places, .00 removed).
    """
    return s.astype("float").map("{:.2f}".format).str.replace(".00", "").replace({"nan": None})


def is_series_of_lists(s: pd.Series) -> bool:
    """Check if a pandas Series contains lists or arrays as values.

    Args:
        s: Pandas Series to check.

    Returns:
        True if the series contains lists or numpy arrays, False otherwise.
    """
    s_rep = s.dropna().iloc[0]  # Find a non-na element
    return isinstance(s_rep, list) or isinstance(s_rep, np.ndarray)


def throw_vals_left(df: pd.DataFrame) -> None:
    """Move all NaN values to the right in each row (in-place).

    Args:
        df: DataFrame to modify in-place.
    """
    # Helper fun to move inplace all nan values to right.
    df.iloc[:, :] = df.apply(lambda row: sorted(row, key=pd.isna), axis=1).to_list()


# We want to show docs for this
def create_topk_metas_and_dfs(
    df: pd.DataFrame,
    dict_with_create: dict[str, object],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    r"""Create top K aggregations from DataFrame.

    Creates new DataFrames and metadata structures for top-K aggregations
    based on regex patterns and column transformations.

    Args:
        df: DataFrame to create top K aggregations from.
        dict_with_create: Dictionary with create block and meta args.

    Returns:
        Tuple of (list of DataFrames, list of metadata dictionaries).

    Meta args:

    **Meta args:**

    - `from_columns` (str)
      Regex template with groups to select df cols from.
      If more than one group, then separate into subgroups.

    - `res_cols` (str)
      Regex template with groups to create new cols. Note that new structure is created for each subgroup.
      Currently res_cols assumes same number of groups in from_columns.

    - `na_vals` (list)
      List of values to consider as NA.

    - `type` (str)
      Type of aggregation to create. If `"topk"`, this function is called.

    - `scale` (dict, optional)
      Scale block to be added to each subgroup.

    - `name` (str, optional)
      Name of the new columns. Defaults to dict_with_create['name'] + "_topk".

    - `k` (int, optional)
      Number of new cols to create per subgroup. Defaults to max and selects columns that include other vals than NA.

    - `agg_index` (int, optional)
      Index of the from_cols group to be aggregated. Note that 1 is first group, defaults to last group.
    """
    create = dict_with_create["create"]
    name = create.get("name", f"{dict_with_create['name']}_{create['type']}")
    # Compile regex to catch df.columns that we want to apply topk for.
    regex_from = re.compile(create["from_columns"])
    from_cols = list(filter(lambda s: regex_from.match(s), df.columns))
    # agg_ind is index of the regex group that we want to aggregate over.
    # All other indeces are unique identifiers for each subgroup
    # e.g. ['A'] and ['B'] (that are also added to meta names)
    # Recall that regex group 0 is the whole match.
    # Note that re.Match.groups() does not include the whole match.
    # This means that we need to subtract 1 from agg_ind.
    agg_ind = create.get("agg_index", -1)
    agg_ind = agg_ind - 1 if agg_ind > 0 else agg_ind
    n_groups = len(regex_from.match(from_cols[0]).groups())
    has_subgroups = n_groups >= 2  # Multiple aggregations needed?
    regex_to = create.get("res_cols", "")
    kmax = create.get("k", None)
    na_vals = create.get("na_vals", [])
    if has_subgroups:
        # collect all subgroups, later aggregate each subgroup separately
        def get_subgroup_id(column: str) -> list[str]:
            """Discard the aggregation index and collect the rest as identifier."""
            subgroup_id = list(regex_from.match(column).groups())
            subgroup_id.pop(agg_ind)
            return tuple(subgroup_id)

        subgroups_ids = dict.fromkeys(map(get_subgroup_id, from_cols))
        subgroups = [[col for col in from_cols if get_subgroup_id(col) == subgroup_id] for subgroup_id in subgroups_ids]
    else:
        subgroups = [from_cols]
    topk_dfs, subgroup_metas = [], []

    # select group at agg_ind in col name to allow translate if spec-d in scale
    # e.g. {A_11: selected} |-> {11: selected}, later by using mask |-> {11: 11}
    # this fun is def-d in current fun, so agg_ind acts as global var
    def get_regex_group_at_agg_ind(s: str) -> str:
        """Get regex group at aggregation index."""
        return regex_from.match(s).groups()[agg_ind]

    for subgroup in subgroups:
        sdf = df[subgroup].astype("object").replace(na_vals, None)
        newcols = [
            # from_cols names map to res_cols names
            # note regex groups stay the same: e.g A_11 |-> A_R11
            regex_from.match(col).expand(regex_to)
            for col in sdf.columns
        ]

        # Convert one-hot encoded columns into a list-of-selected format
        sdf.columns = sdf.columns.map(get_regex_group_at_agg_ind)
        sdf = sdf.mask(
            ~sdf.isna(), other=pd.Series(sdf.columns, index=sdf.columns), axis=1
        )  # replace cell with column name where not NA
        throw_vals_left(sdf)  # changes df in place, Nones go to rightmost side

        sdf.columns = newcols  # set column names per the regex_to template
        sdf = sdf.dropna(axis=1, how="all")  # drop rightmost cols that are all NA
        sdf = sdf.iloc[:, :kmax] if kmax else sdf  # up to kmax columns if spec-d
        sname = name
        if has_subgroups:
            sname += "_" + "_".join(map(str, get_subgroup_id(subgroup[0])))
        meta_subgroup = {
            "name": sname,
            "scale": deepcopy(create.get("scale", {})),
            "columns": sdf.columns.tolist(),
        }
        topk_dfs.append(sdf)
        subgroup_metas.append(meta_subgroup)
    return subgroup_metas, topk_dfs  # note: each df has one meta for zip later


create_block_type_to_create_fn = {
    "topk": create_topk_metas_and_dfs,
    "maxdiff": NotImplementedError("Maxdiff not implemented yet"),
}


def create_new_columns_and_metas(
    df: pd.DataFrame, group: dict[str, object]
) -> Iterator[tuple[pd.DataFrame, dict[str, object]]]:
    """Create new columns and metadata from a group definition.

    One group can create multiple metas if it has >1 groups spec-d in regex.

    Args:
        df: DataFrame to add columns to.
        group: Group definition dictionary with create block.

    Returns:
        Zip object of (DataFrame, metadata) tuples.
    """
    type = group["create"]["type"]
    if type not in create_block_type_to_create_fn:
        raise NotImplementedError(f"Create block {type} not supported")
    metas, dfs = create_block_type_to_create_fn[type](df, group)
    return zip(dfs, metas)


def _apply_categories(s: pd.Series, cd: dict[str, object], cn: str, sn: str, raw_data: pd.DataFrame) -> pd.Series:
    """Apply category transformations to a series. Isolated to avoid pyright performance issues."""
    na_sum = s.isna().sum()

    if cd["categories"] == "infer":
        if s.dtype.name == "category":
            cd["categories"] = list(s.dtype.categories)
        elif "translate" in cd and "transform" not in cd and set(cd["translate"].values()) >= set(s.dropna().unique()):
            cats = [str(c) for c in cd["translate"].values() if c in s.unique()]
            cd["categories"] = list(dict.fromkeys(cats))
            s = s.astype("str")
        else:
            if cd.get("ordered", False) and not pd.api.types.is_numeric_dtype(s):
                warn(
                    f"Ordered category {cn} had category: infer. "
                    "This only works correctly if you want lexicographic ordering!"
                )
            if not pd.api.types.is_numeric_dtype(s):
                s.loc[~s.isna()] = s[~s.isna()].astype(str)
            cinds = s.drop_duplicates().sort_values().index
            if pd.api.types.is_numeric_dtype(s):
                s = convert_number_series_to_categorical(s)
            cd["categories"] = [c for c in s[cinds] if pd.notna(c)]
    elif pd.api.types.is_numeric_dtype(s):
        try:
            fcats = np.array(cd["categories"]).astype(float)
            s_vals, s_idx, s_name = s.values, s.index, s.name
            nearest_cats = np.array(cd["categories"])[np.abs(s_vals[:, None] - fcats[None, :]).argmin(axis=1)]
            s = pd.Series(
                nearest_cats,
                index=s_idx,
                name=s_name,
                dtype=pd.CategoricalDtype(categories=cd["categories"], ordered=cd.get("ordered")),
            )
        except (ValueError, TypeError):
            raise ValueError(f"Categories for {cn} are not numeric: {cd['categories']}")

    cats = cd["categories"]
    ns = pd.Series(
        pd.Categorical(s, categories=cats, ordered=cd["ordered"] if "ordered" in cd else False),
        name=cn,
        index=raw_data.index,
    )

    new_nas = ns.isna().sum() - na_sum
    if new_nas > 0:
        unlisted_cats = set(s.dropna().unique()) - set(cats)
        col_info = f"({sn}) " if cn != sn else ""
        warn(f"Column {cn} {col_info}had unknown categories {unlisted_cats} for {new_nas / len(ns):.1%} entries")

    return ns


# Default usage with mature metafile: process_annotated_data(<metafile name>)
# When figuring out the metafile, it can also be run as: process_annotated_data(meta=<dict>, data_file=<>)
def process_annotated_data(
    meta_fname: str | None = None,
    meta: dict[str, object] | None = None,
    data_file: str | None = None,
    raw_data: pd.DataFrame | None = None,
    return_meta: bool = False,
    ignore_exclusions: bool = False,
    only_fix_categories: bool = False,
    return_raw: bool = False,
    add_original_inds: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object] | None]:
    """Process annotated data according to metadata specifications.

    Args:
        meta_fname: Path to metadata JSON/YAML file.
        meta: Metadata dictionary (alternative to meta_fname).
        data_file: Path to data file (if not in metadata).
        raw_data: Pre-loaded DataFrame (alternative to data_file).
        return_meta: Whether to return metadata along with data.
        ignore_exclusions: Whether to ignore exclusion rules in metadata.
        only_fix_categories: Whether to only fix categories without other processing.
        return_raw: Whether to return raw unprocessed data.
        add_original_inds: Whether to add original index column.

    Returns:
        Processed DataFrame, or tuple of (DataFrame, metadata) if return_meta=True.
    """
    # Read metafile
    metafile = stk_file_map.get(meta_fname, meta_fname)
    if meta_fname is not None:
        ext = os.path.splitext(metafile)[1]
        if ext == ".yaml":
            meta = read_yaml(metafile)
        elif ext == ".json":
            meta = read_json(metafile)
        else:
            raise Exception(f"Unknown meta file format {ext} for file: {meta_fname}")

    # Print any issues with the meta without raising an error - for now
    soft_validate(meta, DataMeta)

    # Setup constants with a simple replacement mechanic
    constants = meta["constants"] if "constants" in meta else {}
    meta = replace_constants(meta)

    # Read datafile(s)
    if raw_data is None:
        raw_data, inp_meta, einfo = read_concatenate_files_list(meta, data_file, path=meta_fname)
        if inp_meta is not None:
            warn("Processing main meta file")  # Print this to separate warnings for input jsons from main
    else:
        einfo = {}

    if return_raw:
        return (raw_data, meta) if return_meta else raw_data

    globs = {
        "pd": pd,
        "np": np,
        "sp": sp,
        "stk": stk,
        "df": raw_data,
        **einfo,
        **constants,
    }

    if "preprocessing" in meta and not only_fix_categories:
        exec(str_from_list(meta["preprocessing"]), globs)
        raw_data = globs["df"]

    ndf: pd.DataFrame = pd.DataFrame()
    all_cns = dict()
    for group in meta["structure"]:
        if group["name"] in all_cns:
            raise Exception(f"Group name {group['name']} duplicates a column name in group {all_cns[group['name']]}")
        all_cns[group["name"]] = group["name"]
        g_cols = []
        if "create" in group:
            for newdf, newmeta in create_new_columns_and_metas(raw_data, group):
                # If same name cols, we overwrite the old ones
                raw_data = newdf.combine_first(raw_data)
                # Note we are appending to the list that we are iterating over
                meta["structure"].append(newmeta)
            del group["create"]  # clean up the meta
        for tpl in group["columns"]:
            if isinstance(tpl, list):
                cn = tpl[0]  # column name
                sn = tpl[1] if len(tpl) > 1 and isinstance(tpl[1], str) else cn  # source column
                # metadata
                if len(tpl) == 3:
                    o_cd = tpl[2]
                elif len(tpl) == 2 and isinstance(tpl[1], dict):
                    o_cd = tpl[1]
                else:
                    o_cd = {}
            else:
                cn = sn = tpl
                o_cd = {}

            cd = {**group.get("scale", {}), **o_cd}

            # Col prefix is used to avoid name clashes when different groups naturally share same column names
            if "col_prefix" in cd:
                cn = cd["col_prefix"] + cn

            # Detect duplicate columns in meta - including among those missing or generated
            # Only flag if they are duplicates even after prefix
            if cn in all_cns:
                raise Exception(f"Duplicate column name found: '{cn}' in {all_cns[cn]} and {group['name']}")
            all_cns[cn] = group["name"]

            if only_fix_categories:
                sn = cn
            g_cols.append(cn)

            if sn not in raw_data:
                if not group.get("generated"):  # bypass warning for columns marked as being generated later
                    warn(f"Column {sn} not found")
                continue

            if raw_data[sn].isna().all():
                warn(f"Column {sn} is empty and thus ignored")
                continue

            s = raw_data[sn]
            if not only_fix_categories and not is_series_of_lists(s):
                if s.dtype.name == "category":
                    s = s.astype("object")  # This makes it easier to use common ops like replace and fillna
                if "translate" in cd:
                    s = s.astype("str").replace(cd["translate"]).replace("nan", None).replace("None", None)
                if "transform" in cd:
                    s = eval(
                        cd["transform"],
                        {
                            "s": s,
                            "df": raw_data,
                            "ndf": ndf,
                            "pd": pd,
                            "np": np,
                            "stk": stk,
                            **constants,
                        },
                    )
                if "translate_after" in cd:
                    s = (
                        pd.Series(s)
                        .astype("str")
                        .replace(cd["translate_after"])
                        .replace("nan", None)
                        .replace("None", None)
                    )

                if cd.get("datetime"):
                    s = pd.to_datetime(s, errors="coerce")
                elif cd.get("continuous"):
                    s = pd.to_numeric(s, errors="coerce")

            s: pd.Series = pd.Series(s, name=cn)  # In case transformation removes the name or renames it

            if cd.get("categories") and not is_series_of_lists(s):
                s = _apply_categories(s, cd, cn, sn, raw_data)

            # Update ndf in real-time so it would be usable in transforms for next columns
            if cn in ndf.columns:
                ndf = ndf.drop(columns=cn)  # Overwrite existing instead of duplicates
            ndf[cn] = s

        if "subgroup_transform" in group:
            subgroups = group.get("subgroups", [g_cols])
            for sg in subgroups:
                ndf[sg] = eval(
                    group["subgroup_transform"],
                    {
                        "gdf": ndf[sg],
                        "df": raw_data,
                        "ndf": ndf,
                        "pd": pd,
                        "np": np,
                        "stk": stk,
                        **constants,
                    },
                )

    if "postprocessing" in meta and not only_fix_categories:
        globs["df"] = ndf
        exec(str_from_list(meta["postprocessing"]), globs)
        ndf = globs["df"]

    # Fix categories after postprocessing
    # Also replaces infer with the actual categories
    fix_meta_categories(meta, ndf, warnings=True)

    ndf["original_inds"] = np.arange(len(ndf))
    if "excluded" in meta and not ignore_exclusions:
        excl_inds = [i for i, _ in meta["excluded"]]
        ndf = ndf[~ndf["original_inds"].isin(excl_inds)]
    if not add_original_inds:
        ndf.drop(columns=["original_inds"], inplace=True)

    return (ndf, meta) if return_meta else ndf


# Read either a json annotation and process the data, or a processed parquet with the annotation attached
# Return_raw is here for easier debugging of metafiles and is not meant to be used in production
def read_annotated_data(
    fname: str, infer: bool = True, return_raw: bool = False, return_meta: bool = False, **kwargs: object
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object] | None]:
    """Read annotated data from JSON/YAML or Parquet file.

    Args:
        fname: Path to data file (JSON, YAML, or Parquet).
        infer: Whether to infer metadata if not found (default: True).
        return_raw: Whether to return raw unprocessed data (for debugging).
        return_meta: Whether to return metadata along with data.
        **kwargs: Additional arguments passed to processing functions.

    Returns:
        DataFrame, or tuple of (DataFrame, metadata) if return_meta=True.
    """
    _, ext = os.path.splitext(fname)
    meta = None
    if ext == ".json" or ext == ".yaml":
        data, meta = process_annotated_data(fname, return_meta=True, return_raw=return_raw, **kwargs)
    elif ext == ".parquet":
        data, full_meta = read_parquet_with_metadata(fname)
        meta = (full_meta or {}).get("data")

    if meta is not None or not infer:
        return (data, meta) if return_meta else data
    else:
        warn(f"Warning: using inferred meta for {fname}")
        meta = infer_meta(fname, meta_file=False)
        return process_annotated_data(data_file=fname, meta=meta, return_meta=return_meta)


# Fix df dtypes etc using meta - needed after a lazy load


def fix_df_with_meta(df: pd.DataFrame, dmeta: dict[str, object]) -> pd.DataFrame:
    """Fix DataFrame dtypes and categories using metadata.

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
        if cd.get("categories"):
            cats = list(df[c].unique()) if cd["categories"] == "infer" else cd["categories"]
            df[c] = pd.Categorical(df[c], categories=cats, ordered=cd.get("ordered", False))
    return df


# Helper functions designed to be used with the annotations


# Convert data_meta into a dict where each group and column maps to their metadata dict
def extract_column_meta(data_meta: dict[str, object]) -> dict[str, dict[str, object]]:
    """Extract column metadata from data_meta structure.

    Args:
        data_meta: Data metadata dictionary with structure field.

    Returns:
        Dictionary mapping column/group names to their metadata.
    """
    res = defaultdict(lambda: {})
    for g in data_meta["structure"]:
        base = g["scale"].copy() if "scale" in g else {}
        res[g["name"]] = {
            **base,
            "columns": [base.get("col_prefix", "") + (t[0] if isinstance(t, list) else t) for t in g["columns"]],
        }
        base["label"] = None  # Don't let that be carried over to individual columns
        for cd in g["columns"]:
            if isinstance(cd, str):
                cd = [cd]
            res[base.get("col_prefix", "") + cd[0]] = {**base, **cd[-1]} if isinstance(cd[-1], dict) else base.copy()
    return res


# Convert data_meta into a dict of group_name -> [column names]
# TODO: deprecate - info available in extract_column_meta


def group_columns_dict(data_meta: dict[str, object]) -> dict[str, list[str]]:
    """Get dictionary mapping group names to their column lists.

    Args:
        data_meta: Data metadata dictionary.

    Returns:
        Dictionary mapping group names to lists of column names.
    """
    return {k: d["columns"] for k, d in extract_column_meta(data_meta).items() if "columns" in d}

    # return { g['name'] : [(t[0] if type(t)!=str else t) for t in g['columns']] for g in data_meta['structure'] }


# Take a list and a dict and replace all dict keys in list with their corresponding lists in-place


def list_aliases(lst: list[str], da: dict[str, list[str]]) -> list[str]:
    """Expand aliases in a list using a dictionary mapping.

    Args:
        lst: List of strings that may contain aliases.
        da: Dictionary mapping aliases to lists of expanded values.

    Returns:
        List with aliases expanded to their corresponding lists.
    """
    return [fv for v in lst for fv in (da[v] if isinstance(v, str) and v in da else [v])]


# Creates a mapping old -> new
def get_original_column_names(dmeta: dict[str, object]) -> dict[str, str]:
    """Get mapping of original column names (before any transformations).

    Args:
        dmeta: Data metadata dictionary.

    Returns:
        Dictionary mapping current column names to original names.
    """
    res = {}
    for g in dmeta["structure"]:
        for c in g["columns"]:
            if isinstance(c, str):
                res[c] = c
            elif isinstance(c, list):
                if len(c) == 1:
                    res[c[0]] = c[0]
                elif len(c) >= 2 and isinstance(c[1], str):
                    res[c[1]] = c[0]  # This is a rename: [new_name, old_name, ...]
                elif len(c) >= 1:
                    res[c[0]] = c[0]  # This is a regular column with metadata: [name, {...}]
    return res


# Map ot backwards and nt forwards to move from one to the other


def change_mapping(ot: dict[str, str], nt: dict[str, str], only_matches: bool = False) -> dict[str, str]:
    """Create mapping from old translation to new translation.

    Args:
        ot: Old translation dictionary.
        nt: New translation dictionary.
        only_matches: Whether to only include keys present in both dictionaries.

    Returns:
        Dictionary mapping old values to new values.

    Note:
        TODO: warn about non-bijective mappings.
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


# Change an existing dataset to correspond better to a new meta_data
# This is intended to allow making small improvements in the meta even after a model has been run
# It is by no means perfect, but is nevertheless a useful tool to avoid re-running long pymc models
# for simple column/translation changes
def change_df_to_meta(df: pd.DataFrame, old_dmeta: dict[str, object], new_dmeta: dict[str, object]) -> pd.DataFrame:
    """Update DataFrame to match new metadata structure.

    Args:
        df: DataFrame to update.
        old_dmeta: Old data metadata dictionary.
        new_dmeta: New data metadata dictionary.

    Returns:
        Updated DataFrame matching new metadata structure.

    Note:
        This tool handles only simple cases of column name, translation and category order changes.
    """
    warn("This tool handles only simple cases of column name, translation and category order changes.")

    # Ready the metafiles for parsing
    old_dmeta = replace_constants(old_dmeta)
    new_dmeta = replace_constants(new_dmeta)

    # Rename columns
    ocn, ncn = (
        get_original_column_names(old_dmeta),
        get_original_column_names(new_dmeta),
    )
    name_changes = change_mapping(ocn, ncn, only_matches=True)
    if name_changes != {}:
        print(f"Renaming columns: {name_changes}")
    df.rename(columns=name_changes, inplace=True)

    rev_name_changes = {v: k for k, v in name_changes.items()}

    # Get metadata for each column
    ocm = extract_column_meta(old_dmeta)
    ncm = extract_column_meta(new_dmeta)

    for c in ncm.keys():
        if c not in df.columns:
            continue  # probably group
        if c not in ocm.keys():
            continue  # new column

        ncd, ocd = ncm[c], ocm[rev_name_changes[c] if c in rev_name_changes else c]

        # Warn about transformations and don't touch columns where those change
        if ocd.get("transform") != ncd.get("transform"):
            warn(f"Column {c} has a different transformation. Leaving it unchanged")
            continue

        # Handle translation changes
        ot, nt = ocd.get("translate", {}), ncd.get("translate", {})
        remap = change_mapping(ot, nt)
        if remap != {}:
            # Validate that mapping keys exist in current categories
            invalid_keys = set(remap.keys()) - set(df[c].cat.categories)
            if invalid_keys:
                raise ValueError(
                    f"Translation mapping keys {invalid_keys} not found in current categories "
                    f"{list(df[c].cat.categories)} for column {c}"
                )
            print(f"Remapping {c} with {remap}")
            df[c] = df[c].cat.rename_categories(remap)

        # Reorder categories and/or change ordered status
        if (
            ncd.get("categories", "infer") != "infer" and list(df[c].dtype.categories) != ncd.get("categories")
        ) or ocd.get("ordered") != ncd.get("ordered"):
            cats = ncd.get("categories") if ncd.get("categories", "infer") != "infer" else df[c].dtype.categories
            if isinstance(cats, list):
                print(f"Changing {c} to Cat({cats},ordered={ncd.get('ordered')})")
                df[c] = pd.Categorical(df[c], categories=cats, ordered=ncd.get("ordered"))

    # Column order changes
    gcdict = group_columns_dict(new_dmeta)

    cols = ["draw", "obs_idx", "training_subsample"] + [c for g in new_dmeta["structure"] for c in gcdict[g["name"]]]
    cols.append(new_dmeta["weight_col"] if new_dmeta.get("weight_col") else "N")
    cols = [c for c in cols if c in df.columns]

    if len(set(df.columns) - set(cols)) > 0:
        print("Dropping columns:", set(df.columns) - set(cols))

    return df[cols]


def update_meta_with_model_fields(meta: dict[str, object], donor: dict[str, object]) -> dict[str, object]:
    """Update metadata with fields from a donor metadata (e.g., from a model).

    Args:
        meta: Target metadata dictionary to update.
        donor: Source metadata dictionary with additional fields.

    Returns:
        Updated metadata dictionary.
    """
    # Add the groups added by the model before to data_meta
    existing_grps = {g["name"] for g in meta["structure"]}
    meta["structure"] += [
        grp for grp in donor["structure"] if grp.get("generated") and grp["name"] not in existing_grps
    ]

    # Add back the fields added/changed by the model in sampling
    meta["draws_data"] = donor.get("draws_data", [])
    if "total_size" in donor:
        meta["total_size"] = donor["total_size"]
    if "weight_col" in donor:
        meta["weight_col"] = donor["weight_col"]

    return meta


def replace_data_meta_in_parquet(
    parquet_name: str, metafile_name: str, advanced: bool = True
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace metadata in a Parquet file with metadata from a JSON file.

    Args:
        parquet_name: Path to Parquet file.
        metafile_name: Path to JSON metadata file.
        advanced: Whether to use advanced metadata replacement.

    Returns:
        Tuple of (DataFrame, updated metadata).
    """
    df, meta = read_parquet_with_metadata(parquet_name)

    ometa = meta["data"]
    nmeta = read_json(metafile_name)
    nmeta = replace_constants(nmeta)

    nmeta = update_meta_with_model_fields(nmeta, ometa)

    # Perform the column name changes and category translations
    # Do this before inferring meta as categories might change in this step
    if advanced:
        df = change_df_to_meta(df, ometa, nmeta)

    nmeta = fix_meta_categories(nmeta, df)  # replace infer with values

    meta["original_data"] = meta.get("original_data", meta["data"])
    meta["data"] = nmeta

    write_parquet_with_metadata(df, meta, parquet_name)

    return df, meta


# A function to infer categories (and validate the ones already present)
# Works in-place
def fix_meta_categories(
    data_meta: dict[str, object], df: pd.DataFrame, infers_only: bool = False, warnings: bool = True
) -> None:
    """Fix and infer categories in metadata based on DataFrame (modifies metadata in-place).

    Args:
        data_meta: Data metadata dictionary to update.
        df: DataFrame to extract categories from.
        infers_only: Whether to only fix "infer" categories, not validate existing ones.
        warnings: Whether to emit warnings for missing categories.
    """
    if "structure" not in data_meta:
        return

    for g in data_meta["structure"]:
        all_cats = set()
        prefix = g.get("scale", {}).get("col_prefix", "")
        for c in g.get("columns", []):
            if isinstance(c, str):
                c = [c]
            if prefix + c[0] in df.columns and df[prefix + c[0]].dtype.name == "category":
                cats, cm = list(df[prefix + c[0]].dtype.categories), c[-1]
                if isinstance(cm, dict) and cm.get("categories") == "infer":
                    cm["categories"] = cats
                elif (
                    (not infers_only)
                    and isinstance(cm, dict)
                    and cm.get("categories")
                    and not set(cm["categories"]) >= set(cats)
                ):
                    diff = set(cats) - set(cm["categories"])
                    if warnings:
                        warn(f"Fixing missing categories for {c[0]}: {diff}")
                    cm["categories"] = cats
                all_cats |= set(cats)

        if g.get("scale") and g["scale"].get("categories") == "infer":
            # IF they all share same categories, keep the category order
            tr = g["scale"].get("translate", {})
            if all_cats == set(tr.values()):  # First prefer translate order
                scats = list(dict.fromkeys(tr.values()))
            elif all_cats == set(cats):  # Then prefer order from single col
                scats = list(cats)
            else:  # Otherwise, sort categories
                scats = sorted(list(all_cats))
            g["scale"]["categories"] = scats
        elif (
            (not infers_only)
            and g.get("scale")
            and g["scale"].get("categories")
            and not set(g["scale"]["categories"]) >= all_cats
        ):
            diff = all_cats - set(g["scale"]["categories"])
            if warnings:
                warn(f"Fixing missing categories for group {g['name']}: {diff}")
            g["scale"]["categories"] = list(all_cats)

    return data_meta


def fix_parquet_categories(parquet_name: str) -> None:
    """Fix categories in a Parquet file by reading, fixing, and rewriting.

    Args:
        parquet_name: Path to Parquet file.
    """
    df, meta = read_parquet_with_metadata(parquet_name)
    meta["data"] = fix_meta_categories(meta["data"], df, infers_only=False)
    write_parquet_with_metadata(df, meta, parquet_name)


def is_categorical(col: pd.Series) -> bool:
    """Check if a pandas Series is categorical.

    Args:
        col: Pandas Series to check.

    Returns:
        True if the series is categorical, False otherwise.
    """
    return col.dtype.name in ["object", "str", "category"] and not is_datetime(col)


max_cats = 50

# Create a very basic metafile for a dataset based on it's contents
# This is not meant to be directly used, rather to speed up the annotation process


def infer_meta(
    data_file: str | None = None,
    meta_file: bool | str = True,
    read_opts: dict[str, object] | None = None,
    df: pd.DataFrame | None = None,
    translate_fn: Callable[[str], str] | None = None,
    translation_blacklist: list[str] | None = None,
) -> dict[str, object]:
    """Infer metadata structure from a data file.

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
        meta["file"] = fname
        if ext in ["csv", "gz"]:
            df = pd.read_csv(data_file, low_memory=False, **read_opts)
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
            df = pd.read_excel(data_file, **read_opts)
        else:
            raise Exception(f"Not a known file format {data_file}")

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
        if not is_categorical(df[cn]):
            continue
        cats[cn] = (
            sorted(list(df[cn].dropna().unique())) if df[cn].dtype.name != "category" else list(df[cn].dtype.categories)
        )

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
    def cat_meta(cn: str) -> dict[str, object]:
        """Create metadata for a categorical column.

        Args:
            cn: Column name.

        Returns:
            Metadata dictionary for the categorical column.
        """
        m = {"categories": cats[cn] if len(cats[cn]) <= max_cats else "infer"}
        if cn in df.columns and df[cn].dtype.name == "category" and df[cn].dtype.ordered:
            m["ordered"] = True
        if translate_fn is not None and cn not in translation_blacklist and len(cats[cn]) <= max_cats:
            tdict = {c: translate_fn(c) for c in m["categories"]}
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

        grp = {"name": ";".join(kl), "scale": cat_meta(str(kl)), "columns": m_cols}

        meta["structure"].append(grp)
        handled_cols.update(g_cols)

    # Put the rest of variables into main category
    main_cols = [c for c in cols if c not in handled_cols]
    for cn in main_cols:
        if cn in cats:
            cdesc = cat_meta(cn)
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


# Small convenience function to have a meta available for any dataset


def data_with_inferred_meta(data_file: str, **kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read data file and infer metadata if not present.

    Args:
        data_file: Path to data file.
        **kwargs: Additional arguments passed to processing functions.

    Returns:
        Tuple of (DataFrame, inferred metadata).
    """
    meta = infer_meta(data_file, meta_file=False, **kwargs)
    return process_annotated_data(meta=meta, data_file=data_file, return_meta=True)


def perform_merges(
    df: pd.DataFrame, merges: dict[str, object] | list[dict[str, object]], constants: dict[str, object] | None = None
) -> pd.DataFrame:
    """Perform merge operations on a DataFrame.

    Args:
        df: DataFrame to merge into.
        merges: Single merge specification or list of merge specifications.
        constants: Dictionary of constants to pass to merge operations.

    Returns:
        DataFrame with merges applied.
    """
    if constants is None:
        constants = {}
    if not isinstance(merges, list):
        merges = [merges]
    for ms in merges:
        ndf = read_and_process_data(ms["file"], constants=constants)
        on = ms["on"] if isinstance(ms["on"], list) else [ms["on"]]
        if ms.get("add"):
            ndf = ndf[ms["on"] + ms["add"]]
        # print(df.columns,ndf.columns,ms['on'])
        mdf = pd.merge(df, ndf, on=on, how=ms.get("how", "inner"))

        for c in on:
            mdf[c] = mdf[c].astype(df[c].dtype)
        if len(df) != len(mdf):
            missing = set(list(df[on].drop_duplicates().itertuples(index=False, name=None))) - set(
                list(ndf[on].drop_duplicates().itertuples(index=False, name=None))
            )
            warn(f"Merge with {ms['file']} removes {1 - len(mdf) / len(df):.1%} rows with missing merges on: {missing}")
        df = mdf
    return df


def read_and_process_data(
    desc: str | dict[str, object],
    return_meta: bool = False,
    constants: dict[str, object] | None = None,
    skip_postprocessing: bool = False,
    **kwargs: object,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """Read and process data according to a description dictionary.

    Args:
        desc: Data description (string file path or dictionary).
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
        desc = {"file": desc}  # Allow easy shorthand for simple cases

    # Validate the data desc format
    desc = DataDescription.model_validate(desc).model_dump(mode="json")

    if desc.get("data") is not None:
        df, meta, einfo = pd.DataFrame(data=desc["data"]), None, {}
    else:
        df, meta, einfo = read_concatenate_files_list(desc, **kwargs)

    if meta is None and return_meta:
        raise Exception("No meta found on any of the files")

    # Perform transformation and filtering
    globs = {"pd": pd, "np": np, "sp": sp, "stk": stk, "df": df, **einfo, **constants}
    if desc.get("preprocessing"):
        exec(str_from_list(desc["preprocessing"]), globs)

    if desc.get("filter"):
        globs["df"] = globs["df"][eval(desc["filter"], globs)]
    if desc.get("merge"):
        globs["df"] = perform_merges(globs["df"], desc.get("merge"), constants)
    if desc.get("postprocessing") and not skip_postprocessing:
        exec(str_from_list(desc["postprocessing"]), globs)
    df = globs["df"]

    return (df, meta) if return_meta else df


# Small debug tool to help find where jsons become non-serializable
def find_type_in_dict(d: object, dtype: type, path: str = "") -> None:
    """Debug function to find values of a specific type in a nested dictionary.

    Args:
        d: Dictionary or value to search.
        dtype: Type to search for.
        path: Current path string for error reporting.

    Raises:
        Exception: If a value of the specified type is found.
    """
    print(d, path)
    if isinstance(d, dict):
        for k, v in d.items():
            find_type_in_dict(v, dtype, path + f"{k}:")
    if isinstance(d, list):
        for i, v in enumerate(d):
            find_type_in_dict(v, dtype, path + f"[{i}]")
    elif isinstance(d, dtype):
        raise Exception(f"Value {d} of type {dtype} found at {path}")


# These two very helpful functions are borrowed from https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e

custom_meta_key = "salk-toolkit-meta"


def write_parquet_with_metadata(df: pd.DataFrame, meta: dict[str, object], file_name: str) -> None:
    """Write DataFrame to Parquet file with embedded metadata.

    Args:
        df: DataFrame to write.
        meta: Metadata dictionary to embed.
        file_name: Path to output Parquet file.
    """
    table = pa.Table.from_pandas(df)

    # find_type_in_dict(meta,np.int64)

    custom_meta_json = json.dumps(meta)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)

    pq.write_table(table, file_name, compression="ZSTD")


# Just load the metadata from the parquet file


def read_parquet_metadata(file_name: str) -> dict[str, object] | None:
    """Read metadata from a Parquet file.

    Args:
        file_name: Path to Parquet file.

    Returns:
        Metadata dictionary, or None if no metadata found.
    """
    schema = pq.read_schema(file_name)
    if custom_meta_key.encode() in schema.metadata:
        restored_meta_json = schema.metadata[custom_meta_key.encode()]
        restored_meta = json.loads(restored_meta_json)
    else:
        restored_meta = None
    return restored_meta


# Load parquet with metadata


def read_parquet_with_metadata(
    file_name: str, lazy: bool = False, **kwargs: object
) -> tuple[pd.DataFrame | pl.LazyFrame, dict[str, object] | None]:
    """Read Parquet file with embedded metadata.

    Args:
        file_name: Path to Parquet file.
        lazy: Whether to return Polars LazyFrame instead of pandas DataFrame.
        **kwargs: Additional arguments passed to Parquet reader.

    Returns:
        Tuple of (DataFrame/LazyFrame, metadata dictionary).
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
        restored_meta = json.loads(restored_meta_json)
    else:
        restored_meta = None

    return restored_df, restored_meta
