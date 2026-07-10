"""Column metadata extraction and plot-descriptor overrides."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from salk_toolkit.io import extract_column_meta
from salk_toolkit.io import group_columns_dict
from salk_toolkit.validation import (
    BlockScaleMeta,
    ColumnBlockMeta,
    DataMeta,
    GroupOrColumnMeta,
    PlotDescriptor,
    soft_validate,
)


# extract_column_meta runs a pydantic validation per column, which is a real per-plot cost
# on datasets with hundreds of columns - and it gets called several times per plot render.
# Cache per meta instance; keys hold a strong reference so id() stays valid while cached.
_col_meta_cache: Dict[int, tuple[DataMeta, Dict[str, GroupOrColumnMeta]]] = {}


def _extract_column_meta_cached(data_meta: DataMeta) -> Dict[str, GroupOrColumnMeta]:
    """Cached ``extract_column_meta``; returns a fresh dict so callers can replace entries."""

    key = id(data_meta)
    hit = _col_meta_cache.get(key)
    if hit is None or hit[0] is not data_meta:
        if len(_col_meta_cache) > 8:
            _col_meta_cache.clear()
        _col_meta_cache[key] = (data_meta, extract_column_meta(data_meta))
    return dict(_col_meta_cache[key][1])


def _update_data_meta_with_pp_desc(
    data_meta: DataMeta,
    pp_desc: PlotDescriptor,
) -> tuple[Dict[str, GroupOrColumnMeta], Dict[str, List[str]]]:
    """Allow pp_desc to modify data meta by merging plot-descriptor overrides into the canonical data metadata."""

    # Ensure data_meta is a DataMeta object (handle legacy dict inputs from tests)
    if not isinstance(data_meta, DataMeta):
        meta_obj = soft_validate(data_meta, DataMeta)
    else:
        meta_obj = data_meta
    desc_obj = pp_desc

    working_meta = meta_obj
    structure = dict(meta_obj.structure or {})

    res_meta_raw = desc_obj.res_meta
    if res_meta_raw:
        if isinstance(res_meta_raw, dict):
            res_meta = soft_validate(res_meta_raw, ColumnBlockMeta)
        else:
            # Already a ColumnBlockMeta object
            res_meta = res_meta_raw

        if res_meta.scale is None and res_meta.columns:
            base_col_name = next(iter(res_meta.columns.keys()), None)
            base_col_meta = _extract_column_meta_cached(meta_obj).get(base_col_name) if base_col_name else None
            if base_col_meta is not None:
                scale_payload = base_col_meta.model_dump(mode="python")
                scale_payload["col_prefix"] = ""
                res_meta = res_meta.model_copy(update={"scale": soft_validate(scale_payload, BlockScaleMeta)})

        structure[res_meta.name] = res_meta
        working_meta = working_meta.model_copy(update={"structure": structure})

    col_meta = _extract_column_meta_cached(working_meta)
    gc_dict = group_columns_dict(working_meta)

    col_meta_override = desc_obj.col_meta or {}
    if col_meta_override:
        assert isinstance(col_meta_override, dict), "col_meta must be a dict"
        for key, update in col_meta_override.items():
            if key in col_meta:
                # Ensure update is a dict (convert ColumnMeta to dict if needed)
                if isinstance(update, BaseModel):
                    update = update.model_dump(mode="python", exclude_unset=True)
                elif not isinstance(update, dict):
                    update = dict(update)
                col_meta[key] = col_meta[key].model_copy(update=update)

    return col_meta, gc_dict
