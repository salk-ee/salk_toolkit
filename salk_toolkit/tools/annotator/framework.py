"""Annotator framework utilities (state, IO, caching, widget wrapper, undo/redo)."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable, get_args, get_origin

import pandas as pd
import streamlit as st

import numpy as np
import scipy as sp

import salk_toolkit as stk
from salk_toolkit.io import _load_data_files, _str_from_list
from salk_toolkit.validation import DataMeta, soft_validate
from pydantic import BaseModel, ValidationError


def get_path_value(obj: Any, path: list[str | int]) -> Any:  # noqa: ANN401
    """Get value from nested object using dot notation."""
    current = obj
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            except (ValueError, IndexError):
                return None
        else:  # Object, use hasattr
            if isinstance(part, str):
                current = getattr(current, part, None)
            else:
                return None

        if current is None:
            return None
    return current


def set_path_value(obj: Any, path: list[str | int], value: Any) -> None:  # noqa: ANN401
    """Set value in nested object using dot notation."""
    current = obj
    # Navigate to parent of target
    for i, part in enumerate(path[:-1]):
        next_part = path[i + 1]
        if isinstance(current, dict):
            if part not in current or current.get(part) is None:
                # Create container if missing to allow setting deep paths
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid list index in path: {part}")

            # Extend list if needed
            while len(current) <= idx:
                current.append(None)
            if current[idx] is None:
                current[idx] = [] if isinstance(next_part, int) else {}
            current = current[idx]
        else:
            if isinstance(part, str):
                child = getattr(current, part, None)
                if child is None:
                    child = [] if isinstance(next_part, int) else {}
                    setattr(current, part, child)
                current = child
            else:
                raise ValueError(f"Cannot access attribute with non-string key: {part}")

    last_part = path[-1]
    if isinstance(current, dict):
        current[last_part] = value
    elif isinstance(current, list):
        idx = int(last_part)
        while len(current) <= idx:
            current.append(None)
        current[idx] = value
    else:
        if isinstance(last_part, str):
            setattr(current, last_part, value)
        else:
            raise ValueError(f"Cannot set attribute with non-string key: {last_part}")


def wrap(  # noqa: ANN401
    inp_func: Callable,
    *args: Any,  # noqa: ANN401
    path: list[str | int],
    i_to_o: Callable | None = None,
    on_change: Callable[[Any, Any, list[str | int]], None] | None = None,
    default_value: Any = None,  # noqa: ANN401
    o_to_i: Callable | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Wrap a Streamlit widget to sync `master_meta` + history."""
    if "master_meta" not in st.session_state:
        return inp_func(*args, **kwargs)

    # Resolve path to get current value
    current_value = get_path_value(st.session_state.master_meta, path)

    # Ensure key is provided
    if "key" not in kwargs:
        raise ValueError("wrap() requires a 'key' parameter for Streamlit widgets")
    key = kwargs["key"]

    # Initialize widget value from current state
    # This enforces current state being the thing always on display

    # Set initial value from master_meta
    if inp_func == st.multiselect:
        kwargs["default"] = current_value if current_value is not None else []
    elif inp_func in (st.selectbox, st.radio):
        # For selectbox/radio, we need to find the index of current_value in options
        options = kwargs.get("options")
        if options is None and len(args) > 1:
            options = args[1]

        index = 0
        if options is not None and current_value is not None:
            # Try to find current_value in options
            try:
                options_list = list(options)
                if current_value in options_list:
                    index = options_list.index(current_value)
            except ValueError:
                pass
        kwargs["index"] = index
    else:
        # text_input, number_input, checkbox, toggle, text_area, color_picker
        # Apply o_to_i conversion first, then default_value logic
        input_value = o_to_i(current_value) if o_to_i is not None else current_value

        # Apply default_value logic: if result is None, use default_value
        if input_value is None and default_value is not None:
            kwargs["value"] = default_value
        elif input_value is not None:
            kwargs["value"] = input_value
        # Otherwise let widget use its default

    def on_change_callback() -> None:
        # Don't save state during initialization or state restoration
        if st.session_state.get("_initializing", False) or st.session_state.get("_restoring", False):
            return

        old_val = get_path_value(st.session_state.master_meta, path)
        new_val = st.session_state.get(key)

        # Handle default_value: if input value equals default_value, convert to None
        if default_value is not None and new_val == default_value:
            new_val = None
        # Special-case: color picker returns hex strings (often lowercased). Treat default_value
        # comparison as case-insensitive so "#ffffff" clears when default is "#FFFFFF".
        elif (
            inp_func == st.color_picker
            and isinstance(default_value, str)
            and isinstance(new_val, str)
            and new_val.lower() == default_value.lower()
        ):
            new_val = None

        if i_to_o is not None:
            try:
                new_val = i_to_o(new_val)
            except Exception as e:
                st.error(f"Conversion error: {e}")
                return

        def normalize_for_compare(val: Any) -> Any:  # noqa: ANN401
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return tuple(val)
            if isinstance(val, str) and val == "":
                return None
            return val

        old_normalized = normalize_for_compare(old_val)
        new_normalized = normalize_for_compare(new_val)

        if old_normalized != new_normalized:
            old_empty = old_normalized is None or old_normalized == tuple()
            new_empty = new_normalized is None or new_normalized == tuple()
            if old_empty and new_empty:
                return

            save_state(path)
            # If the target is a dict key and the new value is None, prefer deleting the key
            # (keeps JSON cleaner than `"key": null` and matches translate-like semantics).
            if new_val is None:
                parent = get_path_value(st.session_state.master_meta, path[:-1])
                last_part = path[-1]
                if isinstance(parent, dict):
                    parent.pop(last_part, None)
                else:
                    set_path_value(st.session_state.master_meta, path, new_val)
            else:
                set_path_value(st.session_state.master_meta, path, new_val)

            if on_change is not None:
                on_change(new_val, old_val, path)

    kwargs["on_change"] = on_change_callback
    return inp_func(*args, **kwargs)


def init_state(meta_path: str) -> None:
    """Initialize session state."""
    if "meta_path" not in st.session_state:
        st.session_state.meta_path = meta_path

    try:
        st.session_state._initializing = True

        with open(meta_path, "r") as f:
            raw_meta = json.load(f)

        st.session_state.master_meta = soft_validate(raw_meta, DataMeta, context={"track_constants": True})

        initial_state = st.session_state.master_meta.model_dump(mode="json")
        st.session_state.history = [(initial_state, "initial")]
        st.session_state.history_index = 0

        raw_data_dict = _load_raw_data(st.session_state.master_meta, meta_path)
        st.session_state.raw_data_dict = raw_data_dict

        cache_hash = _compute_cache_hash(raw_data_dict, st.session_state.master_meta)
        if "data_cache" in st.session_state and isinstance(st.session_state.data_cache, tuple):
            cached_hash, cached_data = st.session_state.data_cache
            if cached_hash == cache_hash:
                st.session_state.data_cache = (cache_hash, cached_data)
            else:
                _clear_column_data_cache()
                processed_data = _process_for_editor(raw_data_dict, st.session_state.master_meta)
                st.session_state.data_cache = (cache_hash, processed_data)
        else:
            _clear_column_data_cache()
            processed_data = _process_for_editor(raw_data_dict, st.session_state.master_meta)
            st.session_state.data_cache = (cache_hash, processed_data)

        st.session_state._initializing = False

    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        st.stop()


def _load_raw_data(meta: DataMeta, meta_path: str) -> dict[str, pd.DataFrame]:
    """Load raw data files using salk_toolkit.io._load_data_files."""
    if meta.files is None:
        return {}

    raw_data_dict, _, _ = _load_data_files(
        meta.files,
        path=meta_path,
        read_opts=meta.read_opts,
        ignore_exclusions=True,
    )
    return raw_data_dict


def _compute_cache_hash(raw_data_dict: dict[str, pd.DataFrame], meta: DataMeta) -> str:
    """Compute a hash of inputs that should trigger cache invalidation."""
    import hashlib

    hash_parts: list[object] = []
    hash_parts.append(tuple(sorted(raw_data_dict.keys())))
    if meta.preprocessing:
        hash_parts.append(_str_from_list(meta.preprocessing))

    # Structure fingerprint: column names + source mappings affect the processed data shape.
    struct_parts: list[tuple[str, str, object, object]] = []
    for block_name, group in sorted(meta.structure.items()):
        scale_meta = group.scale
        col_prefix = scale_meta.col_prefix if scale_meta is not None else None
        for orig_cn, col_meta in sorted(group.columns.items()):
            source_spec: object = col_meta.source if col_meta.source is not None else orig_cn
            struct_parts.append((block_name, orig_cn, source_spec, col_prefix))
    hash_parts.append(tuple(struct_parts))

    transform_scripts: list[str] = []
    for group in meta.structure.values():
        for col_meta in group.columns.values():
            if col_meta.transform:
                transform_scripts.append(col_meta.transform)
    hash_parts.append(tuple(sorted(transform_scripts)))

    hash_str = str(hash_parts)
    return hashlib.md5(hash_str.encode()).hexdigest()


def rebuild_data_cache() -> None:
    """Rebuild processed data cache from `raw_data_dict` + current `master_meta`."""
    if "raw_data_dict" not in st.session_state or "master_meta" not in st.session_state:
        return
    raw_data_dict: dict[str, pd.DataFrame] = st.session_state.raw_data_dict
    _clear_column_data_cache()
    processed_data = _process_for_editor(raw_data_dict, st.session_state.master_meta)
    cache_hash = _compute_cache_hash(raw_data_dict, st.session_state.master_meta)
    st.session_state.data_cache = (cache_hash, processed_data)


def _process_for_editor(raw_data_dict: dict[str, pd.DataFrame], meta: DataMeta) -> dict[str, pd.DataFrame]:
    """Custom processing pipeline for the editor."""
    data_dict = {k: v.copy() for k, v in raw_data_dict.items()}
    constants = dict(meta.constants)

    if meta.preprocessing is not None:
        for file_code, df in data_dict.items():
            globs = {
                "pd": pd,
                "np": np,
                "sp": sp,
                "stk": stk,
                "df": df,
                "file_code": file_code,
                **constants,
            }
            exec(_str_from_list(meta.preprocessing), globs)
            data_dict[file_code] = globs["df"]

    ndf_dict = {k: pd.DataFrame(index=v.index) for k, v in data_dict.items()}
    for fc, ndf in ndf_dict.items():
        ndf["file_code"] = fc

    for _, group in meta.structure.items():
        g_cols: list[str] = []

        scale_meta = group.scale
        col_prefix = scale_meta.col_prefix if scale_meta is not None else None

        for orig_cn, col_meta in group.columns.items():
            cn = col_prefix + orig_cn if col_prefix is not None else orig_cn
            g_cols.append(cn)

            source_spec = col_meta.source if col_meta.source is not None else orig_cn

            for file_code, file_raw in data_dict.items():
                ndf = ndf_dict[file_code]

                if isinstance(source_spec, dict):
                    sn = source_spec.get(file_code, source_spec.get("default", orig_cn))
                else:
                    sn = source_spec

                if sn not in file_raw:
                    ndf[cn] = np.nan
                    continue

                s = file_raw[sn].copy()
                s.name = cn

                if col_meta.transform is not None:
                    s = eval(
                        col_meta.transform,
                        {
                            "s": s,
                            "df": file_raw,
                            "ndf": ndf,
                            "pd": pd,
                            "np": np,
                            "stk": stk,
                            **constants,
                        },
                    )

                if col_meta.continuous:
                    s = pd.to_numeric(s, errors="coerce")
                elif col_meta.datetime:
                    s = pd.to_datetime(s, errors="coerce")

                ndf[cn] = s

        if group.subgroup_transform is not None:
            subgroups = [g_cols]
            for sg in subgroups:
                for file_code, ndf in ndf_dict.items():
                    gdf_local = ndf[sg].copy()
                    file_raw = data_dict[file_code]

                    transformed_gdf = eval(
                        group.subgroup_transform,
                        {
                            "gdf": gdf_local,
                            "ndf": ndf,
                            "df": file_raw,
                            "pd": pd,
                            "np": np,
                            "stk": stk,
                            **constants,
                        },
                    )
                    ndf[sg] = transformed_gdf[sg]

    return ndf_dict


def save_state(path: list[str | int] | None = None) -> None:
    """Save current state to history with optional path information."""
    try:
        if "history" not in st.session_state:
            st.session_state.history = []
        if "history_index" not in st.session_state:
            st.session_state.history_index = -1

        st.session_state.history = st.session_state.history[: st.session_state.history_index + 1]

        current_state = st.session_state.master_meta.model_dump(mode="json")
        path_str = "unknown" if path is None else ".".join(str(p) for p in path)
        st.session_state.history.append((current_state, path_str))
        st.session_state.history_index += 1
        st.toast("Changes saved. Use Undo to revert.", icon="💾")
    except Exception as e:
        st.error(f"Error saving state: {e}")
        import traceback

        st.code(traceback.format_exc())


def undo() -> None:
    """Undo the last change."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        restore_state()


def redo() -> None:
    """Redo the last undone change."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        restore_state()


def restore_state() -> None:
    """Restore state from history."""
    st.session_state._restoring = True
    try:
        history_entry = st.session_state.history[st.session_state.history_index]
        if isinstance(history_entry, tuple):
            state_dump, path_str = history_entry
        else:
            state_dump = history_entry
            path_str = "unknown"
        st.session_state.master_meta = soft_validate(state_dump, DataMeta, context={"track_constants": True})
        st.session_state._last_restored_path = path_str

        # The processed data cache depends on structure/transform metadata; ensure it matches the restored state.
        rebuild_data_cache()
    finally:
        st.session_state._restoring = False


def save_meta() -> None:
    """Save metadata to file with backup."""
    path = st.session_state.meta_path

    i = 0
    while True:
        backup_path = f"{path}.orig.{i}.json"
        if not os.path.exists(backup_path):
            break
        i += 1

    try:
        if os.path.exists(path):
            shutil.copy(path, backup_path)
            st.toast(f"Backup created at {os.path.basename(backup_path)}")

        with open(path, "w") as f:
            dump = st.session_state.master_meta.model_dump(mode="json", exclude_none=True)
            json.dump(dump, f, indent=2)

        st.success(f"Saved to {os.path.basename(path)}")

    except Exception as e:
        st.error(f"Failed to save: {e}")


def _clear_column_data_cache() -> None:
    """Clear cached column data when data cache is invalidated."""
    keys_to_remove = [
        k
        for k in st.session_state.keys()
        if k.startswith("column_data_") or k.startswith("col_counts_") or k.startswith("col_trans_")
    ]
    for k in keys_to_remove:
        del st.session_state[k]


def get_all_raw_columns() -> list[str]:
    """Get all raw column names from loaded data."""
    cols = set()
    if "raw_data_dict" in st.session_state:
        for df in st.session_state.raw_data_dict.values():
            cols.update(df.columns)
    return sorted(list(cols))


def sidebar() -> None:
    """Render the sidebar UI."""
    with st.sidebar:
        st.header("Annotator")

        st.radio("Mode", ["Blocks", "Constants", "Files"], key="mode")

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            undo_disabled = st.session_state.history_index <= 0
            if not undo_disabled:
                prev_entry = st.session_state.history[st.session_state.history_index]
                prev_path = prev_entry[1] if isinstance(prev_entry, tuple) else "unknown"
            else:
                prev_path = "unknown"
            st.button("Undo", on_click=undo, disabled=undo_disabled, help=f"Path: {prev_path}")

        with col2:
            redo_disabled = st.session_state.history_index >= len(st.session_state.history) - 1
            if not redo_disabled:
                next_entry = st.session_state.history[st.session_state.history_index + 1]
                next_path = next_entry[1] if isinstance(next_entry, tuple) else "unknown"
            else:
                next_path = "unknown"
            st.button("Redo", on_click=redo, disabled=redo_disabled, help=f"Path: {next_path}")

        st.button("Save Changes", on_click=save_meta, type="primary")

        st.divider()

        if st.session_state.get("mode") == "Blocks":
            if "raw_data_dict" in st.session_state:
                num_files = len(st.session_state.raw_data_dict)
                if num_files > 1:
                    st.toggle("Separate files", value=False, key="separate_files")
                else:
                    st.session_state.separate_files = False


def get_column_data(col_name: str) -> pd.DataFrame:
    """Get concatenated data for a column from cache, including file_code."""
    if "data_cache" not in st.session_state:
        return pd.DataFrame()

    if isinstance(st.session_state.data_cache, tuple):
        _, data_dict = st.session_state.data_cache
    else:
        data_dict = st.session_state.data_cache

    cache_key = f"column_data_{col_name}"
    if cache_key not in st.session_state:
        frames = []
        for file_code, df in data_dict.items():
            if col_name in df.columns:
                sub = df[[col_name, "file_code"]].copy()
                sub.columns = ["value", "file_code"]
                frames.append(sub)

        if not frames:
            st.session_state[cache_key] = pd.DataFrame(columns=["value", "file_code"])
        else:
            st.session_state[cache_key] = pd.concat(frames)

    return st.session_state[cache_key]


def normalize_translate_dict(translate_dict: dict[Any, Any]) -> dict[str, str]:  # noqa: ANN401
    """Normalize translate dict to dict[str, str] (keeps '' as explicit missing mapping)."""
    out: dict[str, str] = {}
    for k, v in translate_dict.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def updated_categories_from_translate(
    current_categories: list[str],
    translate_dict: dict[str, str],
    old_val: str | None,
    new_val: str | None,
) -> list[str]:
    """Compute updated categories list based on translate dict, preserving order when possible."""
    translated_values = {v for v in translate_dict.values() if v != ""}

    if not translated_values:
        return list(current_categories)

    old_val_str = str(old_val) if old_val is not None and old_val != "" else None
    new_val_str = str(new_val) if new_val is not None and new_val != "" else None

    old_val_still_used = bool(old_val_str and old_val_str in translated_values)

    remaining = set(translated_values)
    new_categories: list[str] = []
    replacement_done = False

    for cat in current_categories:
        if old_val_str and cat == old_val_str and new_val_str and new_val_str != old_val_str and not replacement_done:
            if old_val_still_used and old_val_str in translated_values:
                new_categories.append(old_val_str)
                remaining.discard(old_val_str)
                new_categories.append(new_val_str)
                remaining.discard(new_val_str)
            else:
                new_categories.append(new_val_str)
                remaining.discard(new_val_str)
            replacement_done = True
            continue

        if cat in translated_values:
            new_categories.append(cat)
            remaining.discard(cat)

    if new_val_str and new_val_str in remaining and new_val_str not in new_categories:
        insert_pos = None
        if old_val_str and old_val_str in new_categories:
            insert_pos = new_categories.index(old_val_str) + (1 if old_val_still_used else 0)
        if insert_pos is None:
            new_categories.append(new_val_str)
        else:
            new_categories.insert(min(insert_pos, len(new_categories)), new_val_str)
        remaining.discard(new_val_str)

    new_categories.extend(sorted(remaining))
    return new_categories


def _unwrap_optional(tp: object) -> object:
    """Return non-None inner type for Optional/Union types (best-effort)."""
    origin = get_origin(tp)
    if origin is None:
        return tp
    if origin is list or origin is dict:
        return tp
    if origin is tuple:
        return tp
    if origin is type(None):  # noqa: E721
        return tp
    if origin is None:
        return tp
    if origin is object:
        return tp
    if origin is Callable:
        return tp
    if origin is type:
        return tp
    if origin is getattr(__import__("typing"), "Union", object()):
        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0]
    return tp


def _infer_model_class_for_path(path: list[str | int]) -> type[BaseModel] | None:
    """Infer the pydantic model class expected at `path` (best-effort)."""
    obj: Any = st.session_state.master_meta
    expected: object = DataMeta

    for part in path:
        # If we have a model instance, prefer its runtime type.
        if isinstance(obj, BaseModel):
            expected = obj.__class__

        # Step through BaseModel field access.
        if isinstance(part, str) and isinstance(obj, BaseModel):
            field = obj.__class__.model_fields.get(part)
            if field is not None:
                expected = field.annotation
            obj = getattr(obj, part, None)
            continue

        # Dict key access: use expected type if we have it.
        if isinstance(obj, dict):
            exp = _unwrap_optional(expected)
            if get_origin(exp) is dict:
                args = get_args(exp)
                if len(args) == 2:
                    expected = args[1]
            obj = obj.get(part)  # type: ignore[arg-type]
            continue

        # List index access: use expected type if we have it.
        if isinstance(obj, list):
            exp = _unwrap_optional(expected)
            if get_origin(exp) is list:
                args = get_args(exp)
                if len(args) == 1:
                    expected = args[0]
            try:
                obj = obj[int(part)]  # type: ignore[arg-type]
            except Exception:
                obj = None
            continue

        # Fallback: attempt attribute access for string parts.
        if isinstance(part, str):
            obj = getattr(obj, part, None)
            continue

        return None

    # End: if we have an instance, that's authoritative.
    if isinstance(obj, BaseModel):
        return obj.__class__

    exp = _unwrap_optional(expected)
    if isinstance(exp, type) and issubclass(exp, BaseModel):
        return exp
    return None


@st.dialog("Edit JSON")
def edit_json_modal(path: list[str | int]) -> None:
    """Edit a pydantic object at `path` by JSON round-trip + `soft_validate`."""
    if "master_meta" not in st.session_state:
        st.error("No metadata loaded.")
        return

    val = get_path_value(st.session_state.master_meta, path)
    model_cls = _infer_model_class_for_path(path)
    if model_cls is None:
        st.error("Could not infer a pydantic model type for this path.")
        return

    # Serialize current value into editable JSON.
    if isinstance(val, BaseModel):
        payload: Any = val.model_dump(mode="json")
    else:
        payload = val
    json_text = json.dumps(payload, indent=2)

    path_key = ".".join(str(p) for p in path)
    editor_key = f"_json_modal_{path_key}"
    err_key = f"_json_modal_err_{path_key}"
    prev_key = f"_json_modal_prev_{path_key}"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = json_text

    st.text_area("JSON", key=editor_key, height=400)

    # Clear validation error when user edits the JSON.
    curr_text = st.session_state.get(editor_key, "")
    if st.session_state.get(prev_key) != curr_text:
        st.session_state[prev_key] = curr_text
        if err_key in st.session_state:
            del st.session_state[err_key]

    # Render validation error (scrollable).
    if err_key in st.session_state and st.session_state[err_key]:
        st.error("Validation error")
        st.text_area(
            "Details",
            value=str(st.session_state[err_key]),
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Save", type="primary", key=f"_json_modal_save_{path_key}"):
            try:
                parsed = json.loads(st.session_state[editor_key])
            except Exception as e:
                st.session_state[err_key] = f"Invalid JSON: {e}"
                return

            try:
                validated = soft_validate(parsed, model_cls, context={"track_constants": True})
            except ValidationError as e:
                # Store full error in session state so it renders in a scrollable container.
                try:
                    st.session_state[err_key] = e.json(indent=2)
                except Exception:
                    st.session_state[err_key] = str(e)
                return
            except Exception as e:
                st.session_state[err_key] = f"Validation failed: {e}"
                return

            save_state(path)
            set_path_value(st.session_state.master_meta, path, validated)
            if err_key in st.session_state:
                del st.session_state[err_key]
            st.rerun()

    with c2:
        if st.button("Cancel", key=f"_json_modal_cancel_{path_key}"):
            if err_key in st.session_state:
                del st.session_state[err_key]
            st.rerun()
