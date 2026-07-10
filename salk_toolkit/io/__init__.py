"""Data Annotation I/O
---------------------

This package covers:

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

from salk_toolkit.io.core import Dataset, HookEnv, ProcessOpts, SourceBundle  # noqa: F401
from salk_toolkit.io.datasets import infer_meta, read_and_process_data, read_annotated_data
from salk_toolkit.io.meta import (
    extract_column_meta,
    fix_df_with_meta,  # noqa: F401  # imported from here by dashboard
    group_columns_dict,
    list_aliases,
    update_meta_with_model_fields,
)
from salk_toolkit.io.parquet import (
    get_stk_commit,  # noqa: F401
    read_parquet_metadata,
    read_parquet_with_metadata,
    replace_data_meta_in_parquet,
    write_parquet_with_metadata,
)
from salk_toolkit.io.readers import get_file_map, get_loaded_files, reset_file_tracking, set_file_map
from salk_toolkit.utils import read_json, read_yaml  # noqa: F401  # historically re-exported via this module
