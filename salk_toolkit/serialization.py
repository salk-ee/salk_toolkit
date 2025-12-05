"""Serialization helpers for Pydantic models.

This module contains helper functions for custom serialization logic,
keeping the validation.py file clean by separating serialization concerns.
"""

from typing import Any, Callable, Dict, Sequence, Tuple, Union

from pydantic import BaseModel, SerializationInfo

# Type aliases for column specifications
ColumnSpecMeta = Dict[str, Any]
ColumnSpecInput = Union[str, list[object]]
ParsedColumnSpec = Tuple[str, str, ColumnSpecMeta]


def serialize_pbase(
    model: BaseModel,
    handler: Callable[[BaseModel], dict[str, Any]],
    info: SerializationInfo,
) -> dict[str, Any]:
    """Serialize PBase model and remove keys where values match defaults or are optional with None values."""
    serialized = handler(model)

    # Get serialization mode
    mode = getattr(info, "mode", "python") if hasattr(info, "mode") else "python"

    # Build default values dict by iterating through fields
    from pydantic_core import PydanticUndefined

    default_values: dict[str, Any] = {}
    for field_name, field_info in model.__class__.model_fields.items():
        if field_name in serialized:
            # Get default value - check for default_factory first
            factory = getattr(field_info, "default_factory", ...)
            if factory is not ... and factory is not None and callable(factory):
                # default_factory is a callable with no arguments
                from typing import cast

                default_val = cast(Callable[[], Any], factory)()
            elif field_info.default is not ... and field_info.default is not PydanticUndefined:
                default_val = field_info.default
            else:
                continue  # No default, skip comparison

            # Serialize default value if it's a BaseModel
            if isinstance(default_val, BaseModel):
                default_values[field_name] = default_val.model_dump(mode=mode)
            else:
                default_values[field_name] = default_val

    # Remove keys where values match defaults, are None, or are empty collections
    result: dict[str, Any] = {}
    for key, value in serialized.items():
        default_val = default_values.get(key)
        # Exclude if value matches default, is None, or is empty collection
        if value != default_val and value is not None and value != {} and value != []:
            result[key] = value

    return result


def serialize_column_meta(
    model: BaseModel,
    handler: Callable[[BaseModel], dict[str, Any]],
    info: SerializationInfo,
) -> dict[str, Any]:
    """Serialize ColumnMeta, excluding fields that match block_scale from context if present."""
    # Check for block_scale in context first
    context = getattr(info, "context", None) or {}
    block_scale = context.get("block_scale")

    if block_scale is not None:
        # Skip parent default removal - serialize directly and compare against block_scale
        mode = getattr(info, "mode", "python") if hasattr(info, "mode") else "python"
        # Serialize without going through PBase default removal
        # Use BaseModel's model_dump directly to bypass PBase serializer
        serialized = BaseModel.model_dump(model, mode=mode, exclude_defaults=False)

        # Serialize block_scale for comparison
        if isinstance(block_scale, BaseModel):
            block_scale_dict = block_scale.model_dump(mode=mode)
        else:
            block_scale_dict = block_scale

        # Remove keys where values match block_scale
        result: dict[str, Any] = {}
        for key, value in serialized.items():
            block_scale_val = block_scale_dict.get(key)
            if value != block_scale_val:
                result[key] = value
        return result
    else:
        # Calling parent PBase serializer
        return serialize_pbase(model, handler, info)


def serialize_column_block_meta(
    model: BaseModel,
    handler: Callable[[BaseModel], dict[str, Any]],
    info: SerializationInfo,
) -> dict[str, Any]:
    """Serialize ColumnBlockMeta and pass block_scale to context for ColumnMeta serialization."""

    # Set up context with block_scale if it exists
    if model.scale is not None:  # type: ignore[attr-defined]
        # Get current context or create new one
        context = getattr(info, "context", None) or {}
        context = dict(context)  # Make a copy
        context["block_scale"] = model.scale  # type: ignore[attr-defined]

        # Get serialization mode
        mode = getattr(info, "mode", "python") if hasattr(info, "mode") else "python"

        # Calling parent PBase serializer
        serialized = serialize_pbase(model, handler, info)

        # Re-serialize columns with context if scale exists
        if "columns" in serialized and isinstance(serialized["columns"], dict):
            # Pass the original ColumnMeta objects to cs_dict_to_lst
            # But we need to apply context serialization for block_scale exclusion
            # cs_dict_to_lst will handle serializing with context
            serialized["columns"] = _cs_dict_to_lst(model.columns, context=context, mode=mode)  # type: ignore[attr-defined]
        else:
            # Convert columns dict to list format even if no scale
            if "columns" in serialized and isinstance(serialized["columns"], dict):
                serialized["columns"] = _cs_dict_to_lst(serialized["columns"])
    else:
        # Calling parent PBase serializer
        serialized = serialize_pbase(model, handler, info)
        # Convert columns dict to list format
        if "columns" in serialized and isinstance(serialized["columns"], dict):
            serialized["columns"] = _cs_dict_to_lst(serialized["columns"])

    return serialized


def serialize_data_meta(
    model: BaseModel,
    handler: Callable[[BaseModel], dict[str, Any]],
    info: SerializationInfo,
) -> dict[str, Any]:
    """Serialize DataMeta with structure and columns converted to list format."""
    # Calling parent PBase serializer
    serialized = serialize_pbase(model, handler, info)

    # Convert structure from dict to list format
    if "structure" in serialized and isinstance(serialized["structure"], dict):
        structure_list = []
        for block_dict in serialized["structure"].values():
            block_dict = dict(block_dict)  # Make a copy
            # Convert columns from dict to list format
            if "columns" in block_dict and isinstance(block_dict["columns"], dict):
                block_dict["columns"] = _cs_dict_to_lst(block_dict["columns"])
            structure_list.append(block_dict)
        serialized["structure"] = structure_list

    return serialized


def _cspec(tpl: ColumnSpecInput) -> ParsedColumnSpec:
    """Column specification conversion functions.

    Parse column specification tuple/list into [column_name, source_name, metadata].
    """
    if isinstance(tpl, list | tuple):
        if not tpl:
            raise TypeError("Column specification lists must contain at least the new column name.")

        raw_cn = tpl[0]
        if not isinstance(raw_cn, str):
            raise TypeError("Column specification must start with the target column name.")
        cn = raw_cn  # column name

        raw_sn = tpl[1] if len(tpl) > 1 else cn
        sn = raw_sn if isinstance(raw_sn, str) else cn  # source column

        if len(tpl) == 3:
            raw_meta = tpl[2]
        elif len(tpl) == 2 and isinstance(tpl[1], dict):
            raw_meta = tpl[1]
        else:
            raw_meta = {}

        if not isinstance(raw_meta, dict):
            raise TypeError("Column metadata must be provided as a dictionary when present.")
        o_cd = raw_meta
    else:
        cn = sn = tpl
        o_cd = {}

    # Add source to metadata dict
    o_cd["source"] = sn
    return (cn, sn, o_cd)


def _cs_lst_to_dict(
    lst: Sequence[ColumnSpecInput] | dict[str, ColumnSpecMeta],
) -> dict[str, ColumnSpecMeta]:
    """Transform list of column specs to dictionary format."""

    # If already a dict, return as-is
    if isinstance(lst, dict):
        return lst

    parsed_specs = [_cspec(item) for item in lst]
    # Import here to avoid circular dependency
    from salk_toolkit.validation import ColumnMeta

    result: dict[str, ColumnSpecMeta] = {}
    for cn, _, meta_dict in parsed_specs:
        # Create ColumnMeta with source already in the dict
        result[cn] = ColumnMeta.model_validate(meta_dict)
    return result


def _cs_dict_to_lst(
    d: dict[str, ColumnSpecMeta], context: dict[str, Any] | None = None, mode: str = "json"
) -> list[ColumnSpecInput]:
    """Transform dict of column specs back to list format (inverse of cs_lst_to_dict)."""
    from pydantic import BaseModel

    result: list[ColumnSpecInput] = []
    for cn, col_meta in d.items():
        # Extract source from ColumnMeta
        sn = col_meta.source if hasattr(col_meta, "source") and col_meta.source is not None else cn

        # Serialize metadata, excluding source (it's in the tuple position)
        if isinstance(col_meta, BaseModel):
            # Use model_dump to leverage the serializers (exclude defaults, etc.)
            # Pass context if provided (for block_scale exclusion)
            if context is not None:
                meta_dict = col_meta.model_dump(mode=mode, context=context)
            else:
                meta_dict = col_meta.model_dump(mode=mode)
        else:
            meta_dict = dict(col_meta) if col_meta else {}
        meta_dict.pop("source", None)

        if sn == cn:
            if meta_dict:
                result.append([cn, meta_dict])
            else:
                result.append(cn)
        else:
            if meta_dict:
                result.append([cn, sn, meta_dict])
            else:
                result.append([cn, sn])
    return result
