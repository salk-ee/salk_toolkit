"""Parquet read/write with embedded DataMeta, and meta replacement on existing files."""

import json
import os
import subprocess
from copy import deepcopy
from functools import lru_cache
from typing import Literal, cast, overload

import pandas as pd
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from salk_toolkit.utils import (
    read_json,
    replace_constants,
)
from salk_toolkit.validation import (
    DataMeta,
    ParquetMeta,
    soft_validate,
)

from salk_toolkit.io.meta import _change_df_to_meta, _fix_meta_categories, update_meta_with_model_fields


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


def _fix_parquet_categories(parquet_name: str) -> None:
    """Fix categories in a Parquet file by reading, fixing, and rewriting."""
    df, meta = read_parquet_with_metadata(parquet_name)
    if meta is None:
        raise ValueError(f"Parquet file {parquet_name} has no metadata")
    meta.data = _fix_meta_categories(meta.data, df, infers_only=False)
    write_parquet_with_metadata(df, meta, parquet_name)


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


@lru_cache(maxsize=1)
def get_stk_commit() -> str | None:
    """Short git commit of the salk_toolkit checkout, or None when not running from a git repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:
        return None


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

    # Stamp the writing salk_toolkit commit so files are self-describing; callers may override.
    if not meta_payload.get("stk_commit"):
        stk_commit = get_stk_commit()
        if stk_commit:
            meta_payload["stk_commit"] = stk_commit

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
