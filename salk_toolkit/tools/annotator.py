"""Streamlit-based annotation editor for survey metadata."""

import streamlit as st
import sys
import os
import json
import shutil
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from salk_toolkit.validation import DataMeta, soft_validate, ColumnBlockMeta, ColumnMeta
from salk_toolkit.io import _load_data_files
import numpy as np
import scipy as sp
import salk_toolkit as stk
import altair as alt

from streamlit_sortables import sort_items


def _str_from_list(val: list[str] | object) -> str:
    """Convert a list to a newline-separated string, or return string representation."""
    if isinstance(val, list):
        return "\n".join(val)
    return str(val)


def get_path_value(obj: Any, path: List[str | int]) -> Any:  # noqa: ANN401
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
        else:
            if isinstance(part, str):
                current = getattr(current, part, None)
            else:
                return None

        if current is None:
            return None
    return current


def set_path_value(obj: Any, path: List[str | int], value: Any) -> None:  # noqa: ANN401
    """Set value in nested object using dot notation."""
    current = obj
    # Navigate to parent of target
    for i, part in enumerate(path[:-1]):
        if isinstance(current, dict):
            # If key missing in dict, create it?
            # For Pydantic models structure, we rely on fields existing.
            # For pure dicts (like translate), we might need to create?
            # But here we are just traversing.
            if part not in current and i < len(path) - 1:
                # If we are traversing and key is missing, we can't proceed unless we create.
                # But we don't know the type of the next object usually unless we infer.
                # Assuming existing structure for now.
                pass
            current = current.get(part)
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid list index in path: {part}")
        else:
            if isinstance(part, str):
                current = getattr(current, part)
            else:
                raise ValueError(f"Cannot access attribute with non-string key: {part}")

    last_part = path[-1]
    if isinstance(current, dict):
        current[last_part] = value
    elif isinstance(current, list):
        idx = int(last_part)
        current[idx] = value
    else:
        if isinstance(last_part, str):
            setattr(current, last_part, value)
        else:
            raise ValueError(f"Cannot set attribute with non-string key: {last_part}")


def wrap(  # noqa: ANN401
    inp_func: Callable,
    *args: Any,  # noqa: ANN401
    path: List[str | int],
    converter: Optional[Callable] = None,
    extra_on_change: Optional[Callable[[Any, Any, List[str | int]], None]] = None,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """
    Wraps a Streamlit input widget to automatically update master_meta and history.

    Args:
        inp_func: The Streamlit input function (e.g. st.text_input)
        *args: Positional args for inp_func
        path: List of keys/indices to the value in master_meta (e.g. ["structure", "block", "columns", "col", "label"])
        converter: Optional function to convert the input value before saving (e.g. json.loads)
        extra_on_change: Optional callback function(new_val, old_val, path) called after value is updated
        **kwargs: Keyword args for inp_func
    """
    if "master_meta" not in st.session_state:
        return inp_func(*args, **kwargs)

    # Resolve path to get current value
    current_value = get_path_value(st.session_state.master_meta, path)

    # Generate a stable key based on path if not provided
    # Join path elements with dot for key
    path_str = ".".join(str(p) for p in path)
    key = kwargs.get("key", f"wrap_{path_str}")
    kwargs["key"] = key

    # Initialize widget value from current state
    # Special handling for color_picker: always re-read from master_meta to avoid stale values
    # For other widgets: if key already exists in session_state, use that value to avoid triggering on_change
    # during initialization
    force_reread = inp_func == st.color_picker

    if key in st.session_state and not st.session_state.get("_initializing", False) and not force_reread:
        # Key exists and we're not initializing - widget was already created
        # Don't override the value, let Streamlit use the existing one
        pass
    else:
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
                    # Need to handle different types potentially
                    options_list = list(options)
                    if current_value in options_list:
                        index = options_list.index(current_value)
                except ValueError:
                    pass
            kwargs["index"] = index
        elif inp_func == st.color_picker:
            # Special case for color_picker: None -> #FFFFFF for display
            # Always re-read from master_meta to ensure we have the latest value
            # This prevents stale session_state values from persisting
            if current_value is None:
                kwargs["value"] = "#FFFFFF"
            else:
                # Convert Color object to hex string if needed
                if hasattr(current_value, "as_hex"):
                    hex_val = current_value.as_hex()
                    # Ensure it's a valid hex string
                    kwargs["value"] = hex_val if hex_val else "#FFFFFF"
                elif isinstance(current_value, str):
                    # Already a string, use it directly (but ensure it's valid)
                    kwargs["value"] = current_value if current_value.startswith("#") else "#FFFFFF"
                else:
                    # Try to convert to string, fallback to white
                    try:
                        str_val = str(current_value)
                        kwargs["value"] = str_val if str_val.startswith("#") else "#FFFFFF"
                    except Exception:
                        kwargs["value"] = "#FFFFFF"
        else:
            # text_input, number_input, checkbox, toggle, text_area
            # Only set value if not None, otherwise let widget use its default (usually empty/False)
            if current_value is not None:
                if converter and inp_func == st.text_area and not isinstance(current_value, str):
                    # Special case for JSON text area: convert object back to string for display
                    kwargs["value"] = json.dumps(current_value, indent=2)
                else:
                    kwargs["value"] = current_value

    def on_change() -> None:
        # Don't save state during initialization or state restoration
        if st.session_state.get("_initializing", False) or st.session_state.get("_restoring", False):
            return

        # Get the old value from master_meta FIRST, before reading widget value
        # This ensures we're comparing against the actual stored value
        old_val = get_path_value(st.session_state.master_meta, path)

        # Get the new value from widget
        new_val = st.session_state.get(key)

        if converter:
            try:
                new_val = converter(new_val)
            except Exception as e:
                st.error(f"Conversion error: {e}")
                return

        # Normalize values for comparison (handle None, empty strings, etc.)
        def normalize_for_compare(val: Any) -> Any:  # noqa: ANN401
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                # Convert to tuple for comparison
                # Keep empty lists/tuples as empty tuple to distinguish from None
                return tuple(val)
            if isinstance(val, str) and val == "":
                return None
            return val

        old_normalized = normalize_for_compare(old_val)
        new_normalized = normalize_for_compare(new_val)

        # Only update if values actually changed
        # Use a more strict comparison to avoid false positives
        if old_normalized != new_normalized:
            # Double-check: if both are "empty" (None or empty tuple), treat as equal
            old_empty = old_normalized is None or old_normalized == tuple()
            new_empty = new_normalized is None or new_normalized == tuple()
            if old_empty and new_empty:
                return

            save_state(path)  # Save state BEFORE change (for Undo), include path
            set_path_value(st.session_state.master_meta, path, new_val)

            # Call custom extra_on_change callback if provided
            if extra_on_change is not None:
                extra_on_change(new_val, old_val, path)

            # No need to rerun explicitly if we are in a callback, Streamlit reruns after callback

    kwargs["on_change"] = on_change

    return inp_func(*args, **kwargs)


# Page config
st.set_page_config(
    layout="wide",
    page_title="SALK Annotator",
    initial_sidebar_state="expanded",
)


def init_state(meta_path: str) -> None:
    """Initialize session state."""
    if "meta_path" not in st.session_state:
        st.session_state.meta_path = meta_path

    try:
        # Set initialization flag to prevent save_state during widget initialization
        st.session_state._initializing = True

        with open(meta_path, "r") as f:
            raw_meta = json.load(f)

        # Initial soft validation to get DataMeta object
        # We track constants usage via context
        st.session_state.master_meta = soft_validate(raw_meta, DataMeta, context={"track_constants": True})

        # Initialize history with the initial state
        # We store serialized state in history (with path info)
        initial_state = st.session_state.master_meta.model_dump(mode="json")
        st.session_state.history = [(initial_state, "initial")]
        st.session_state.history_index = 0

        # Load data
        raw_data_dict = _load_raw_data(st.session_state.master_meta, meta_path)
        st.session_state.raw_data_dict = raw_data_dict

        # Process data with caching
        cache_hash = _compute_cache_hash(raw_data_dict, st.session_state.master_meta)
        if "data_cache" in st.session_state and isinstance(st.session_state.data_cache, tuple):
            cached_hash, cached_data = st.session_state.data_cache
            if cached_hash == cache_hash:
                # Cache hit - reuse existing data
                st.session_state.data_cache = (cache_hash, cached_data)
            else:
                # Cache miss - recompute and clear column data caches
                _clear_column_data_cache()
                processed_data = _process_for_editor(raw_data_dict, st.session_state.master_meta)
                st.session_state.data_cache = (cache_hash, processed_data)
        else:
            # First time - compute
            _clear_column_data_cache()
            processed_data = _process_for_editor(raw_data_dict, st.session_state.master_meta)
            st.session_state.data_cache = (cache_hash, processed_data)

        # Clear initialization flag after everything is set up
        st.session_state._initializing = False

    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        st.stop()


def _load_raw_data(meta: DataMeta, meta_path: str) -> Dict[str, pd.DataFrame]:
    """Load raw data files using salk_toolkit.io._load_data_files."""
    if meta.files is None:
        return {}

    raw_data_dict, _, _ = _load_data_files(
        meta.files,
        path=meta_path,
        read_opts=meta.read_opts,
        ignore_exclusions=True,  # Editor sees all data
    )
    return raw_data_dict


def _compute_cache_hash(raw_data_dict: Dict[str, pd.DataFrame], meta: DataMeta) -> str:
    """Compute a hash of inputs that should trigger cache invalidation."""
    import hashlib

    # Hash: file codes, preprocessing script, transform scripts per column
    hash_parts = []
    hash_parts.append(tuple(sorted(raw_data_dict.keys())))
    if meta.preprocessing:
        hash_parts.append(_str_from_list(meta.preprocessing))
    # Hash transform scripts
    transform_scripts = []
    for group in meta.structure.values():
        for col_meta in group.columns.values():
            if col_meta.transform:
                transform_scripts.append(col_meta.transform)
    hash_parts.append(tuple(sorted(transform_scripts)))
    # Create stable hash
    hash_str = str(hash_parts)
    return hashlib.md5(hash_str.encode()).hexdigest()


def _process_for_editor(raw_data_dict: Dict[str, pd.DataFrame], meta: DataMeta) -> Dict[str, pd.DataFrame]:
    """Custom processing pipeline for the editor.

    Returns:
        Dict mapping file_code to processed DataFrame.
    """
    # Work on copies to avoid modifying cached raw data
    data_dict = {k: v.copy() for k, v in raw_data_dict.items()}
    constants = dict(meta.constants)
    # TODO: extract einfo from _load_data_files if needed, but it returns it.
    # For now assume no extra info needed from loading.

    # 1. Preprocessing
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

    # 2. Build per-file result DataFrames
    ndf_dict = {k: pd.DataFrame(index=v.index) for k, v in data_dict.items()}

    # Add file_code column
    for fc, ndf in ndf_dict.items():
        ndf["file_code"] = fc

    # 3. Process structure
    all_cns = {}
    for group_name, group in meta.structure.items():
        all_cns[group.name] = group.name
        g_cols = []

        scale_meta = group.scale
        col_prefix = scale_meta.col_prefix if scale_meta is not None else None

        for orig_cn, col_meta in group.columns.items():
            cn = orig_cn
            if col_prefix is not None:
                cn = col_prefix + orig_cn

            g_cols.append(cn)

            # Determine source spec
            source_spec = col_meta.source if col_meta.source is not None else orig_cn

            # Process per file
            for file_code, file_raw in data_dict.items():
                ndf = ndf_dict[file_code]

                # Resolve source name
                if isinstance(source_spec, dict):
                    sn = source_spec.get(file_code, source_spec.get("default", orig_cn))
                else:
                    sn = source_spec

                if sn not in file_raw:
                    # Missing column
                    ndf[cn] = np.nan
                    continue

                s = file_raw[sn].copy()
                s.name = cn

                # Apply transform
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

                # Type coercion
                if col_meta.continuous:
                    s = pd.to_numeric(s, errors="coerce")
                elif col_meta.datetime:
                    s = pd.to_datetime(s, errors="coerce")

                # Do NOT translate
                ndf[cn] = s

        # 4. Subgroup transform
        if group.subgroup_transform is not None:
            # subgroups is not in ColumnBlockMeta, assume all columns in group
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
                    # Assign back
                    ndf[sg] = transformed_gdf[sg]

    return ndf_dict


def save_state(path: List[str | int] | str | None = None) -> None:
    """Save current state to history with optional path information.

    Args:
        path: Either a list of path components (e.g., ["structure", "block", "columns", "col"]),
              a descriptive string (e.g., "reorder categories"), or None for unknown paths.
    """
    try:
        # Ensure history exists
        if "history" not in st.session_state:
            st.session_state.history = []
        if "history_index" not in st.session_state:
            st.session_state.history_index = -1

        # Truncate history if we are in the middle
        st.session_state.history = st.session_state.history[: st.session_state.history_index + 1]

        # Serialize current state
        current_state = st.session_state.master_meta.model_dump(mode="json")
        # Store as tuple: (state, path_string)
        if path is None:
            path_str = "unknown"
        elif isinstance(path, str):
            path_str = path
        else:
            path_str = ".".join(str(p) for p in path)
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
    history_entry = st.session_state.history[st.session_state.history_index]
    # Handle both old format (just state) and new format (tuple with path)
    if isinstance(history_entry, tuple):
        state_dump, path_str = history_entry
    else:
        state_dump = history_entry
        path_str = "unknown"
    st.session_state.master_meta = soft_validate(state_dump, DataMeta, context={"track_constants": True})
    # Store the path for display
    st.session_state._last_restored_path = path_str
    # Streamlit will automatically rerun after the callback


def save_meta() -> None:
    """Save metadata to file with backup."""
    path = st.session_state.meta_path

    # Backup
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

        # Save
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


def get_all_raw_columns() -> List[str]:
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

        mode = st.radio("Mode", ["Blocks", "Constants", "Files"], key="mode")

        st.divider()

        # Undo/Redo with path info
        col1, col2 = st.columns([1, 1])
        with col1:
            undo_disabled = st.session_state.history_index <= 0
            if not undo_disabled:
                # Show path for the state we would undo to
                prev_entry = st.session_state.history[st.session_state.history_index]
                prev_path = prev_entry[1] if isinstance(prev_entry, tuple) else "unknown"
            else:
                prev_path = "unknown"
            st.button("Undo", on_click=undo, disabled=undo_disabled, help=f"Path: {prev_path}")

        with col2:
            redo_disabled = st.session_state.history_index >= len(st.session_state.history) - 1
            if not redo_disabled:
                # Show path for the state we would redo to
                next_entry = st.session_state.history[st.session_state.history_index + 1]
                next_path = next_entry[1] if isinstance(next_entry, tuple) else "unknown"
            else:
                next_path = "unknown"
            st.button("Redo", on_click=redo, disabled=redo_disabled, help=f"Path: {next_path}")

        st.button("Save Changes", on_click=save_meta, type="primary")

        st.divider()

        if mode == "Blocks":
            # Separate toggle - only show if multiple files loaded
            if "raw_data_dict" in st.session_state:
                num_files = len(st.session_state.raw_data_dict)
                if num_files > 1:
                    st.toggle("Separate files", value=False, key="separate_files")
                else:
                    # Single file - set to False and don't show toggle
                    st.session_state.separate_files = False


def get_column_data(col_name: str) -> pd.DataFrame:
    """Get concatenated data for a column from cache, including file_code."""
    if "data_cache" not in st.session_state:
        return pd.DataFrame()

    # Extract actual data dict from cache tuple
    if isinstance(st.session_state.data_cache, tuple):
        _, data_dict = st.session_state.data_cache
    else:
        # Legacy format - assume it's the dict directly
        data_dict = st.session_state.data_cache

    # Cache concatenated results per column in session state
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


@st.dialog("Reorder Categories")
def reorder_dialog(col_meta: ColumnMeta) -> None:
    """Dialog for reordering categories."""
    current_cats = col_meta.categories
    if not isinstance(current_cats, list):
        st.warning("Categories are inferred or not a list. Cannot reorder explicitly.")
        return

    sorted_items = sort_items(current_cats)
    if st.button("Apply Order"):
        col_meta.categories = sorted_items
        save_state("reorder categories")
        # Streamlit will automatically rerun after the callback


def column_editor(block_name: str, col_name: str, col_meta: ColumnMeta) -> None:
    """Editor for column metadata."""
    st.write(f"Column: {col_name}")

    # Get latest metadata from session state to ensure we have the most recent version
    if "master_meta" in st.session_state:
        block = st.session_state.master_meta.structure.get(block_name)
        if col_name == "scale":
            if block and block.scale is not None:
                col_meta = block.scale
        elif block and col_name in block.columns:
            col_meta = block.columns[col_name]

    # Handle scale vs column paths
    if col_name == "scale":
        base_path = ["structure", block_name, "scale"]
    else:
        base_path = ["structure", block_name, "columns", col_name]

    # Renaming (structural change, manual) - only for columns, not scale
    if col_name != "scale":
        new_name = st.text_input("Rename Column", value=col_name, key=f"rename_{block_name}_{col_name}")
        if new_name != col_name:
            # TODO: Handle column rename in structure (not implemented in block actions yet)
            st.warning("Column renaming not fully implemented in this fragment yet.")

    # Warnings for usage in other blocks (placeholder as per spec scanning master_meta is acceptable but expensive?)
    # ...

    # Properties
    col1, col2 = st.columns(2)
    with col1:
        wrap(st.text_input, "Label", path=base_path + ["label"], key=f"label_{block_name}_{col_name}")

    # Metadata toggles
    col1, col2, col3 = st.columns(3)
    with col1:
        wrap(st.checkbox, "Ordered", path=base_path + ["ordered"], key=f"ordered_{block_name}_{col_name}")
    with col2:
        wrap(st.checkbox, "Likert", path=base_path + ["likert"], key=f"likert_{block_name}_{col_name}")
    with col3:
        # Nonordered multiselect
        # Needs categories to select from
        cats = col_meta.categories if isinstance(col_meta.categories, list) else []
        wrap(
            st.multiselect,
            "Non-ordered categories",
            options=cats,
            path=base_path + ["nonordered"],
            key=f"nonordered_{block_name}_{col_name}",
        )

    # Check if we have data (scale doesn't have individual data)
    if col_name == "scale":
        # Scale editor - show scale-specific fields
        st.info(
            "Scale metadata applies to all columns in this block. Individual column metadata overrides scale settings."
        )
        # Show scale-specific fields like col_prefix
        if hasattr(col_meta, "col_prefix"):
            wrap(st.text_input, "Column Prefix", path=base_path + ["col_prefix"], key=f"col_prefix_{block_name}_scale")
        # Don't show data visualizations for scale
        return

    df = get_column_data(col_name)
    if df.empty:
        st.warning(f"No data found for {col_name}")
        st.json(col_meta.model_dump(mode="json"))
        return

    separate = st.session_state.get("separate_files", True)

    # Cache expensive computations - but invalidate when metadata changes
    # Use a hash of relevant metadata to detect changes
    meta_hash = hash(str(col_meta.model_dump(mode="json")))
    cache_key_counts = f"col_counts_{block_name}_{col_name}_{separate}_{meta_hash}"

    # Invalidate counts cache if data, separate mode, or metadata changed
    if cache_key_counts not in st.session_state:
        # Clear old cache entries for this column
        old_keys = [k for k in st.session_state.keys() if k.startswith(f"col_counts_{block_name}_{col_name}_")]
        for k in old_keys:
            if k != cache_key_counts:
                del st.session_state[k]

        if separate:
            counts = df.groupby(["value", "file_code"]).size().unstack(fill_value=0)
        else:
            counts = df["value"].value_counts().to_frame(name="count")
        st.session_state[cache_key_counts] = counts
    else:
        counts = st.session_state[cache_key_counts]

    # Visualizations
    if col_meta.continuous:
        # Check if we have valid numeric data
        df_valid = df.dropna(subset=["value"])
        if df_valid.empty:
            st.warning("No valid numeric data for density plot")
        else:
            # Density plot
            # Use different output field names to avoid confusion
            base_chart = alt.Chart(df_valid)
            if separate:
                chart = (
                    base_chart.transform_density(
                        "value", as_=["density_value", "density"], groupby=["file_code"], steps=200
                    )
                    .mark_area(opacity=0.5)
                    .encode(
                        x=alt.X("density_value:Q", title="Value"),
                        y=alt.Y("density:Q", title="Density"),
                        color="file_code:N",
                        tooltip=["density_value:Q", "density:Q", "file_code:N"],
                    )
                )
            else:
                chart = (
                    base_chart.transform_density("value", as_=["density_value", "density"], steps=200)
                    .mark_area(opacity=0.5)
                    .encode(
                        x=alt.X("density_value:Q", title="Value"),
                        y=alt.Y("density:Q", title="Density"),
                        color=alt.value("steelblue"),
                        tooltip=["density_value:Q", "density:Q"],
                    )
                )
            st.altair_chart(chart, use_container_width=True)

        # Stats
        stats = df.groupby("file_code")["value"].describe() if separate else df["value"].describe().to_frame().T
        st.dataframe(stats)

    elif col_meta.datetime:
        # Datetime visualization
        df_valid = df.dropna(subset=["value"])
        if df_valid.empty:
            st.warning("No valid datetime data for visualization")
        else:
            # Time series or histogram for datetime
            base_chart = alt.Chart(df_valid)
            if separate:
                chart = base_chart.mark_bar(opacity=0.7).encode(
                    x=alt.X("value:T", title="Date", bin=alt.Bin(maxbins=50)),
                    y="count()",
                    color="file_code:N",
                    tooltip=["value:T", "count()", "file_code:N"],
                )
            else:
                chart = base_chart.mark_bar(opacity=0.7).encode(
                    x=alt.X("value:T", title="Date", bin=alt.Bin(maxbins=50)),
                    y="count()",
                    color=alt.value("steelblue"),
                    tooltip=["value:T", "count()"],
                )
            st.altair_chart(chart, use_container_width=True)

        # Stats for datetime
        if separate:
            stats = df.groupby("file_code")["value"].agg(["count", "min", "max"]).T
        else:
            stats = pd.DataFrame(
                {"count": [df["value"].count()], "min": [df["value"].min()], "max": [df["value"].max()]}
            )
        st.dataframe(stats)

    else:  # Categorical
        # Apply translation to show translated values in the plot
        df_plot = df.copy()
        if col_meta.translate:
            # Map original values to translated values
            # Handle both string and original type keys
            translate_dict = {}
            for k, v in col_meta.translate.items():
                # Add both string and original type as keys to handle type mismatches
                translate_dict[str(k)] = str(v)
                if k != str(k):
                    translate_dict[k] = str(v)

            # Apply translation
            df_plot["translated_value"] = df_plot["value"].astype(str).replace(translate_dict)
            # For values not in translate, try original type
            missing_mask = df_plot["translated_value"] == df_plot["value"].astype(str)
            if missing_mask.any():
                # Try with original value types
                for idx in df_plot[missing_mask].index:
                    orig_val = df_plot.loc[idx, "value"]
                    if orig_val in col_meta.translate:
                        df_plot.loc[idx, "translated_value"] = str(col_meta.translate[orig_val])
                    else:
                        df_plot.loc[idx, "translated_value"] = str(orig_val) if pd.notna(orig_val) else "NA"
        else:
            df_plot["translated_value"] = df_plot["value"].astype(str)

        # Use categories order from metadata for sorting
        sort_order = None
        if col_meta.categories and isinstance(col_meta.categories, list):
            # Use the categories list directly as sort order (these are translated values)
            sort_order = [str(cat) for cat in col_meta.categories]
            # Add any translated values not in categories list
            all_trans = set(df_plot["translated_value"].dropna().unique())
            for trans in all_trans:
                if trans not in sort_order:
                    sort_order.append(trans)

        # Build color scale from metadata if available
        # Colors are keyed by translated category names
        color_scale = None
        if col_meta.colors:
            # Build color domain and range in category order
            color_domain = []
            color_range = []
            if sort_order:
                for cat in sort_order:
                    if cat in col_meta.colors:
                        color_domain.append(cat)
                        color_val = col_meta.colors[cat]
                        if hasattr(color_val, "as_hex"):
                            color_range.append(color_val.as_hex())
                        else:
                            color_range.append(str(color_val))
            else:
                # No sort order, use all colors
                for cat, color_val in col_meta.colors.items():
                    color_domain.append(str(cat))
                    if hasattr(color_val, "as_hex"):
                        color_range.append(color_val.as_hex())
                    else:
                        color_range.append(str(color_val))

            if color_domain:
                color_scale = alt.Scale(domain=color_domain, range=color_range)

        # Histogram using translated values
        encode_dict = {
            "y": alt.Y("translated_value:N", axis=alt.Axis(title="Value"), sort=sort_order if sort_order else "-x"),
            "x": "count()",
        }

        # Add color encoding
        if separate:
            encode_dict["color"] = "file_code:N"
            encode_dict["yOffset"] = "file_code:N"
        elif color_scale:
            encode_dict["color"] = alt.Color("translated_value:N", scale=color_scale)
        else:
            encode_dict["color"] = alt.value("steelblue")

        chart = alt.Chart(df_plot).mark_bar().encode(**encode_dict)
        st.altair_chart(chart, use_container_width=True)

        # Reorder Button (only for categorical columns)
    if not col_meta.datetime and st.button("Reorder Categories", key=f"reorder_btn_{block_name}_{col_name}"):
        reorder_dialog(col_meta)

    # Category Editor (Table)
    if not col_meta.continuous and not col_meta.datetime:
        st.subheader("Category Mapping")

        # Get all unique original values from data only
        # Only use values that actually appear in the data, not just keys from translate dict
        # Keep both string and original type for proper lookup
        unique_vals_str = set(counts.index.astype(str))
        # Also keep original index for proper type matching in counts lookup
        counts_index_map = {str(idx): idx for idx in counts.index}

        # Build mapping: original -> translated (using string keys for consistency)
        orig_to_trans = {}
        for orig_str in unique_vals_str:
            trans = None
            if col_meta.translate:
                # Try exact match first
                if orig_str in col_meta.translate:
                    trans_val = col_meta.translate[orig_str]
                    # None or empty string means NA
                    if trans_val is None or trans_val == "":
                        trans = None
                    else:
                        trans = str(trans_val)
                else:
                    # Try int/float conversion if applicable
                    try:
                        orig_num = int(orig_str)
                        if orig_num in col_meta.translate:
                            trans_val = col_meta.translate[orig_num]
                            if trans_val is None or trans_val == "":
                                trans = None
                            else:
                                trans = str(trans_val)
                    except (ValueError, TypeError):
                        try:
                            orig_float = float(orig_str)
                            if orig_float in col_meta.translate:
                                trans_val = col_meta.translate[orig_float]
                                if trans_val is None or trans_val == "":
                                    trans = None
                                else:
                                    trans = str(trans_val)
                        except (ValueError, TypeError):
                            pass
            # If no translation found, use identity (original value)
            if trans is None and orig_str not in (col_meta.translate if col_meta.translate else {}):
                trans = orig_str
            orig_to_trans[orig_str] = trans

        # Group original values by their translated value
        trans_to_orig = {}
        for orig, trans in orig_to_trans.items():
            trans_key = trans if trans else "__NO_TRANSLATION__"
            if trans_key not in trans_to_orig:
                trans_to_orig[trans_key] = []
            trans_to_orig[trans_key].append(orig)

        # Sort groups according to category order, then build rows
        # First, determine group order based on categories list
        group_order = []
        if col_meta.categories and isinstance(col_meta.categories, list):
            # Add groups in category order
            for cat in col_meta.categories:
                cat_str = str(cat)
                if cat_str in trans_to_orig:
                    group_order.append(cat_str)
            # Add any other translated values not in categories
            for trans_key in trans_to_orig:
                if trans_key not in group_order and trans_key != "__NO_TRANSLATION__":
                    group_order.append(trans_key)
        else:
            # No categories list, use alphabetical order of translated values
            group_order = sorted([k for k in trans_to_orig.keys() if k != "__NO_TRANSLATION__"])

        # Add untranslated group at the end
        if "__NO_TRANSLATION__" in trans_to_orig:
            group_order.append("__NO_TRANSLATION__")

        # Check for missing values
        na_count = df["value"].isna().sum()
        has_na = na_count > 0

        # Build rows: group by translated value, sort originals alphabetically within groups
        rows = []
        for trans_key in group_order:
            origs_in_group = sorted(trans_to_orig[trans_key])
            trans_value = trans_key if trans_key != "__NO_TRANSLATION__" else None

            for i, orig in enumerate(origs_in_group):
                row = {"original": orig}
                # Counts - use original index type for lookup
                orig_idx = counts_index_map.get(orig, orig)
                if separate:
                    for col in counts.columns:
                        try:
                            row[col] = int(counts.loc[orig_idx, col])
                        except (KeyError, IndexError):
                            row[col] = 0
                else:
                    try:
                        row["count"] = int(counts.loc[orig_idx, "count"])
                    except (KeyError, IndexError):
                        row["count"] = 0

                # Translation (only first row of group shows it, others are empty for grouping)
                row["translation"] = trans_value if trans_value else ""
                # Mark if this is the first row of the group (for color picker and num_values)
                row["is_first_in_group"] = i == 0
                row["translated_value"] = trans_value if trans_value else None

                rows.append(row)

        # Add NA row at the end if there are missing values
        if has_na:
            row = {"original": "NA"}
            if separate:
                # Count NA per file
                na_counts = df[df["value"].isna()].groupby("file_code").size()
                for col in counts.columns:
                    row[col] = int(na_counts.get(col, 0))
            else:
                row["count"] = int(na_count)
            row["translation"] = ""
            row["is_first_in_group"] = False
            row["translated_value"] = None
            rows.append(row)

        # Table Header
        # Check if num_values should be shown (only for ordered categoricals)
        show_num_values = col_meta.ordered and not col_meta.continuous and not col_meta.datetime

        if separate:
            # Multiple columns for counts (one per file)
            num_count_cols = len(counts.columns) if separate else 1
            if show_num_values:
                col_widths = [2] + [1] * num_count_cols + [2, 1, 1]
            else:
                col_widths = [2] + [1] * num_count_cols + [2, 1]
            cols = st.columns(col_widths)
            cols[0].markdown("**Original Value**")
            for i, file_col in enumerate(counts.columns):
                cols[i + 1].markdown(
                    f"<div style='text-align: right;'><strong>Count ({file_col})</strong></div>",
                    unsafe_allow_html=True,
                )
            cols[-2 if show_num_values else -1].markdown("**Translation**")
            if show_num_values:
                cols[-2].markdown(
                    "<div style='text-align: right;'><strong>Num Value</strong></div>",
                    unsafe_allow_html=True,
                )
            cols[-1].markdown("**Color**")
        else:
            if show_num_values:
                cols = st.columns([2, 1, 2, 1, 1])
            else:
                cols = st.columns([2, 1, 2, 1])
            cols[0].markdown("**Original Value**")
            cols[1].markdown("<div style='text-align: right;'><strong>Count</strong></div>", unsafe_allow_html=True)
            cols[2].markdown("**Translation**")
            if show_num_values:
                cols[3].markdown(
                    "<div style='text-align: right;'><strong>Num Value</strong></div>",
                    unsafe_allow_html=True,
                )
                color_col_idx = 4
            else:
                color_col_idx = 3
            cols[color_col_idx].markdown("**Color**")

        # Render Rows
        # show_num_values already computed above for header
        for i, row in enumerate(rows):
            orig = row["original"]
            if separate:
                # Multiple columns for counts
                num_count_cols = len(counts.columns)
                if show_num_values:
                    col_widths = [2] + [1] * num_count_cols + [2, 1, 1]
                else:
                    col_widths = [2] + [1] * num_count_cols + [2, 1]
                cols = st.columns(col_widths)
            else:
                if show_num_values:
                    cols = st.columns([2, 1, 2, 1, 1])
                else:
                    cols = st.columns([2, 1, 2, 1])

            cols[0].text(orig)

            # Count display - right-justified
            if separate:
                # Show one count per file column
                for j, file_col in enumerate(counts.columns):
                    count_val = row.get(file_col, 0)
                    cols[j + 1].markdown(f"<div style='text-align: right;'>{count_val}</div>", unsafe_allow_html=True)
            else:
                cols[1].markdown(f"<div style='text-align: right;'>{row.get('count', 0)}</div>", unsafe_allow_html=True)

            # Translation Input with wrap (last column)
            # Need to determine the key in translate dict
            if orig != "NA":
                # Find best key
                translate_key = orig
                if col_meta.translate:
                    # Check if key exists as is
                    if orig in col_meta.translate:
                        translate_key = orig
                    else:
                        # Check int
                        try:
                            orig_int = int(orig)
                            if orig_int in col_meta.translate:
                                translate_key = orig_int
                            else:
                                # Check float
                                try:
                                    orig_float = float(orig)
                                    if orig_float in col_meta.translate:
                                        translate_key = orig_float
                                except (ValueError, TypeError):
                                    pass
                        except (ValueError, TypeError):
                            pass

                # We need to render the input.
                # If it's the first in group, it has a translation. But we want to allow editing for ALL.
                # The grouping logic was for display in table, but here we might want to allow
                # editing each original value's mapping.
                # If we stick to the grouping visual: "Translation (only first row of group shows
                # it, others are empty for grouping)"
                # But here we are building an editor. If we want to change mapping for one item in
                # a group, we should be able to.
                # So we should show current translation for EVERY row.

                # Re-calculate current translation for this specific original value because
                # row['translation'] was based on grouping logic
                # Default to original value (identity mapping) if no translation exists
                curr_trans = orig
                if col_meta.translate:
                    if translate_key in col_meta.translate:
                        trans_val = col_meta.translate[translate_key]
                        # Empty string means map to NA/None
                        if trans_val == "" or trans_val is None:
                            curr_trans = ""
                        else:
                            curr_trans = str(trans_val)
                    else:
                        # Try numeric type matching
                        try:
                            orig_num = int(orig)
                            if orig_num in col_meta.translate:
                                trans_val = col_meta.translate[orig_num]
                                if trans_val == "" or trans_val is None:
                                    curr_trans = ""
                                else:
                                    curr_trans = str(trans_val)
                        except (ValueError, TypeError):
                            try:
                                orig_float = float(orig)
                                if orig_float in col_meta.translate:
                                    trans_val = col_meta.translate[orig_float]
                                    if trans_val == "" or trans_val is None:
                                        curr_trans = ""
                                    else:
                                        curr_trans = str(trans_val)
                            except (ValueError, TypeError):
                                # No translation found, use identity (original value)
                                curr_trans = orig

                # If the key doesn't exist in translate yet, wrap will create it when value changes.
                # But we need to use the right key type. If it doesn't exist, default to string (orig is string here).

                # Translation input goes in the last column
                trans_col_idx = len(counts.columns) + 1 if separate else 2
                with cols[trans_col_idx]:

                    def update_categories_from_translations(
                        new_val: str | None, old_val: str | None, path: List[str | int]
                    ) -> None:
                        """Update categories list when translations change, preserving order when possible."""
                        # Get the current column metadata
                        block = st.session_state.master_meta.structure.get(block_name)
                        if not block or col_name not in block.columns:
                            return

                        col_meta = block.columns[col_name]
                        translate_path = base_path + ["translate"]
                        translate_dict = col_meta.translate if col_meta.translate else {}

                        # Handle empty string as mapping to None/NA
                        # If new_val is empty string, set it to None in the translate dict
                        if new_val == "":
                            # The path should be ["translate", key], so get the key from path
                            if len(path) >= 2 and path[-2] == "translate":
                                translate_key = path[-1]
                                translate_dict[translate_key] = None
                                set_path_value(st.session_state.master_meta, translate_path, translate_dict)
                                # Update col_meta reference
                                col_meta = block.columns[col_name]
                                translate_dict = col_meta.translate if col_meta.translate else {}

                        # Get all unique original values from the data
                        df = get_column_data(col_name)
                        unique_original_vals = set()
                        if not df.empty:
                            unique_original_vals = set(df["value"].dropna().astype(str).unique())

                        # Also include keys from existing translate dict
                        if translate_dict:
                            # Try to convert keys to strings for comparison
                            for k in translate_dict.keys():
                                unique_original_vals.add(str(k))

                        # Ensure all original values have translations (add identity mappings for missing ones)
                        updated_translate = False

                        for orig_val in unique_original_vals:
                            # Check if translation exists (try both string and original type)
                            needs_identity = True
                            if orig_val in translate_dict:
                                # Check if it's None (which means NA mapping) - don't add identity for those
                                if translate_dict[orig_val] is not None:
                                    needs_identity = False
                            else:
                                # Try int/float conversion
                                try:
                                    orig_num = int(orig_val)
                                    if orig_num in translate_dict:
                                        if translate_dict[orig_num] is not None:
                                            needs_identity = False
                                except (ValueError, TypeError):
                                    try:
                                        orig_float = float(orig_val)
                                        if orig_float in translate_dict:
                                            if translate_dict[orig_float] is not None:
                                                needs_identity = False
                                    except (ValueError, TypeError):
                                        pass

                            if needs_identity:
                                # Add identity mapping - use the original value as both key and value
                                # Try to preserve the original type if possible
                                translate_key = orig_val
                                try:
                                    # Try to use numeric type if it's a number
                                    orig_num = int(orig_val)
                                    translate_key = orig_num
                                except (ValueError, TypeError):
                                    try:
                                        orig_float = float(orig_val)
                                        translate_key = orig_float
                                    except (ValueError, TypeError):
                                        pass

                                # Add identity mapping
                                translate_dict[translate_key] = orig_val
                                updated_translate = True

                        # Update translate dict if we added identity mappings
                        if updated_translate:
                            set_path_value(st.session_state.master_meta, translate_path, translate_dict)
                            # Update col_meta reference to reflect the change
                            col_meta = block.columns[col_name]

                        # Get all unique translated values from the translate dict
                        # Exclude None values (which represent NA mappings)
                        translated_values = set()
                        for trans_val in translate_dict.values():
                            if trans_val is not None and trans_val != "":
                                translated_values.add(str(trans_val))

                        # If no translations, don't update categories
                        if not translated_values:
                            return

                        # Get current categories to preserve order
                        current_categories = []
                        if col_meta.categories and isinstance(col_meta.categories, list):
                            current_categories = [str(c) for c in col_meta.categories]

                        # Check if we're replacing an old value with a new value
                        # If old_val was in categories and new_val is different, replace it in place
                        old_val_str = str(old_val) if old_val is not None and old_val != "" else None
                        new_val_str = str(new_val) if new_val is not None and new_val != "" else None

                        # Check if old_val_str is still used by other translations
                        # (in case multiple original values map to the same translated value)
                        # Note: translate_dict already has the new value, so we check if old_val_str appears elsewhere
                        old_val_still_used = False
                        if old_val_str:
                            for trans_val in translate_dict.values():
                                if trans_val is not None and trans_val != "" and str(trans_val) == old_val_str:
                                    old_val_still_used = True
                                    break

                        # Build new categories list preserving order
                        new_categories = []
                        replacement_done = False

                        # First, iterate through current categories and handle replacements
                        for cat in current_categories:
                            if (
                                old_val_str
                                and cat == old_val_str
                                and new_val_str
                                and new_val_str != old_val_str
                                and not replacement_done
                            ):
                                if not old_val_still_used:
                                    # Replace old value with new value in the same position
                                    # Only if old value is no longer used by other translations
                                    new_categories.append(new_val_str)
                                    translated_values.discard(new_val_str)
                                else:
                                    # Old value is still used, keep it and add new value right after
                                    new_categories.append(cat)
                                    new_categories.append(new_val_str)
                                    translated_values.discard(new_val_str)
                                    translated_values.discard(cat)
                                replacement_done = True
                            elif cat in translated_values:
                                # Keep existing category that still exists in translations
                                new_categories.append(cat)
                                translated_values.discard(cat)
                            # If cat is not in translated_values and wasn't replaced, skip it (it was removed)

                        # If replacement didn't happen in the loop but new_val should be added
                        if new_val_str and new_val_str not in new_categories and new_val_str in translated_values:
                            # If old_val was in categories, try to insert new_val in its position
                            if old_val_str and old_val_str in current_categories:
                                old_idx = current_categories.index(old_val_str)
                                # Insert at the position where old_val was (or right after if old_val still used)
                                insert_pos = old_idx if not old_val_still_used else old_idx + 1
                                new_categories.insert(min(insert_pos, len(new_categories)), new_val_str)
                            else:
                                # New value that wasn't in old categories - add it at the end
                                new_categories.append(new_val_str)
                            translated_values.discard(new_val_str)

                        # Then, add any other new translated values that weren't in the old categories
                        # Sort them for consistency
                        new_categories.extend(sorted(translated_values))

                        # Update categories if they changed
                        if new_categories != current_categories:
                            categories_path = base_path + ["categories"]
                            set_path_value(st.session_state.master_meta, categories_path, new_categories)

                    wrap(
                        st.text_input,
                        "Trans",
                        value=curr_trans,
                        label_visibility="collapsed",
                        path=base_path + ["translate", translate_key],
                        extra_on_change=update_categories_from_translations,
                        key=f"trans_{block_name}_{col_name}_{orig}_{i}",
                    )
            else:
                trans_col_idx = len(counts.columns) + 1 if separate else 2
                cols[trans_col_idx].text("NA")

            # Num values input (only for ordered categoricals, first row of group, and not for NA)
            if show_num_values:
                num_val_col_idx = len(counts.columns) + 2 if separate else 3
                if row.get("is_first_in_group", False) and orig != "NA" and row.get("translated_value"):
                    trans_cat = str(row["translated_value"])

                    # Find the index of this category in the categories list
                    cat_index = None
                    if col_meta.categories and isinstance(col_meta.categories, list):
                        try:
                            cat_index = col_meta.categories.index(trans_cat)
                        except ValueError:
                            # Category not in list, try string comparison
                            for idx, cat in enumerate(col_meta.categories):
                                if str(cat) == trans_cat:
                                    cat_index = idx
                                    break

                    if cat_index is not None:
                        # Get current num_values list or create empty one (handles None case)
                        num_values_list = col_meta.num_values if col_meta.num_values is not None else []
                        # Ensure list is long enough
                        while len(num_values_list) <= cat_index:
                            num_values_list.append(None)

                        # Get current value - use consecutive number (1-indexed) as default if None
                        current_num_val = num_values_list[cat_index]
                        default_num_val = float(cat_index + 1)  # Consecutive number as default/placeholder
                        display_val = current_num_val if current_num_val is not None else default_num_val

                        with cols[num_val_col_idx]:
                            num_val_key = f"num_val_{block_name}_{col_name}_{trans_cat}_{cat_index}"

                            # Use number_input directly and handle update manually
                            def on_num_val_change() -> None:
                                """Handle num_values update when number input changes."""
                                init_flag = st.session_state.get("_initializing", False)
                                restore_flag = st.session_state.get("_restoring", False)
                                if init_flag or restore_flag:
                                    return

                                # Save state before change
                                save_state(base_path + ["num_values"])

                                # Get current num_values list
                                block = st.session_state.master_meta.structure.get(block_name)
                                if not block or col_name not in block.columns:
                                    return

                                col_meta = block.columns[col_name]
                                num_values_path = base_path + ["num_values"]
                                num_values_list = col_meta.num_values if col_meta.num_values is not None else []

                                # Ensure list is long enough for all categories
                                if col_meta.categories and isinstance(col_meta.categories, list):
                                    while len(num_values_list) < len(col_meta.categories):
                                        num_values_list.append(None)

                                # Update the value at cat_index
                                new_val = st.session_state.get(num_val_key)
                                if cat_index is not None and cat_index < len(num_values_list):
                                    # Convert to float or None
                                    if new_val is None:
                                        num_values_list[cat_index] = None
                                    else:
                                        num_values_list[cat_index] = float(new_val)
                                    set_path_value(st.session_state.master_meta, num_values_path, num_values_list)

                            # Don't set session_state before calling number_input to avoid Streamlit warning
                            # Instead, compute the value and pass it directly
                            # Only update session_state if it exists and needs updating (for undo/redo)
                            if num_val_key in st.session_state:
                                # Update from master_meta if it changed externally (e.g., after undo/redo)
                                stored_val = st.session_state[num_val_key]
                                if stored_val is None or abs(stored_val - display_val) > 1e-10:
                                    st.session_state[num_val_key] = display_val

                            st.number_input(
                                "Num Value",
                                value=display_val,
                                label_visibility="collapsed",
                                key=num_val_key,
                                on_change=on_num_val_change,
                            )
                    else:
                        # Category not in categories list, can't set num_value
                        cols[num_val_col_idx].empty()
                else:
                    # Empty cell for non-first rows or NA
                    cols[num_val_col_idx].empty()

            # Color picker (only for first row of group, and not for NA)
            if show_num_values:
                color_col_idx = len(counts.columns) + 3 if separate else 4
            else:
                color_col_idx = len(counts.columns) + 2 if separate else 3
            if row.get("is_first_in_group", False) and orig != "NA" and row.get("translated_value"):
                trans_cat = str(row["translated_value"])
                color_path = base_path + ["colors", trans_cat]

                with cols[color_col_idx]:

                    def color_converter(val: str) -> str | None:
                        """Convert color picker value: #FFFFFF -> None, otherwise keep as is."""
                        if val == "#FFFFFF":
                            return None
                        return val

                    # wrap() will handle reading the color from master_meta and converting Color objects
                    wrap(
                        st.color_picker,
                        "Color",
                        label_visibility="collapsed",
                        path=color_path,
                        converter=color_converter,
                        key=f"color_{block_name}_{col_name}_{trans_cat}",
                    )
            else:
                # Empty cell for non-first rows or NA
                cols[color_col_idx].empty()


def block_editor() -> None:
    """Editor for block metadata."""
    if "master_meta" not in st.session_state:
        st.info("No metadata loaded")
        return

    # Block and column selectors at the top
    block_names = list(st.session_state.master_meta.structure.keys())
    if not block_names:
        st.info("No blocks defined")
        return

    # Initialize selected_block if not set
    if "selected_block" not in st.session_state or st.session_state.selected_block not in block_names:
        st.session_state.selected_block = block_names[0]

    block_name = st.session_state.selected_block
    block = st.session_state.master_meta.structure.get(block_name)
    if not block:
        st.error(f"Block {block_name} not found")
        return

    # Selectors on the same row
    col1, col2 = st.columns([1, 1])
    with col1:
        st.selectbox("Select Block", block_names, key="selected_block")
    with col2:
        # Column selector - include scale if it exists
        column_options = []
        if block.scale is not None:
            column_options.append("scale")
        column_options.extend(list(block.columns.keys()))

        # Initialize selected_column if not set
        if "selected_column" not in st.session_state or st.session_state.selected_column not in column_options:
            st.session_state.selected_column = column_options[0] if column_options else None

        if column_options:
            st.selectbox("Select Column", column_options, key="selected_column")
        else:
            st.info("No columns in this block")
            return

    # Show selected column editor (no expander)
    selected_col = st.session_state.selected_column
    if selected_col == "scale" and block.scale is not None:
        # Show scale editor - use column_editor which now handles scale
        st.subheader("Scale Settings")
        column_editor(block_name, "scale", block.scale)
    elif selected_col in block.columns:
        # Show column editor
        column_editor(block_name, selected_col, block.columns[selected_col])
    else:
        st.warning(f"Column {selected_col} not found")
        return

    st.divider()

    # Block Actions
    with st.expander("Block Settings", expanded=False):
        # Rename block
        # Use wrap to update block.name attribute, with custom on_change to handle dictionary key update
        def handle_block_rename(new_name: str | None, old_name: str | None, path: List[str | int]) -> None:
            """Handle block rename: update dictionary key when block.name changes."""
            if new_name == old_name:
                return

            new_name_str = str(new_name) if new_name is not None else ""

            # Check if new name conflicts with an existing block
            if new_name_str in st.session_state.master_meta.structure and new_name_str != block_name:
                st.error(f"Block {new_name_str} already exists")
                # Revert: restore old name
                set_path_value(st.session_state.master_meta, path, old_name)
                return

            # Valid rename: update dictionary key in structure
            if new_name_str != block_name:
                block = st.session_state.master_meta.structure[block_name]
                st.session_state.master_meta.structure[new_name_str] = block
                del st.session_state.master_meta.structure[block_name]
                st.session_state.selected_block = new_name_str

        wrap(
            st.text_input,
            "Rename Block",
            path=["structure", block_name, "name"],
            extra_on_change=handle_block_rename,
            key=f"rename_block_{block_name}",
        )

        # Add Column
        st.subheader("Add Column")
        all_raw_cols = get_all_raw_columns()
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            col_to_add = st.selectbox("Select column", all_raw_cols, key=f"add_col_{block_name}")
        with col2:
            new_col_name = st.text_input("New column name (optional)", "", key=f"new_col_name_{block_name}")
        with col3:
            if st.button("Add", key=f"add_btn_{block_name}"):
                final_name = new_col_name if new_col_name else col_to_add
                if final_name in block.columns:
                    st.error("Column already exists")
                else:
                    new_meta = ColumnMeta()
                    new_meta.source = col_to_add
                    block.columns[final_name] = new_meta
                    save_state(f"add column: {final_name}")
                    # Streamlit will automatically rerun after the callback

        # Split Block
        st.subheader("Split Block")
        split_name = st.text_input("New block name", key=f"split_name_{block_name}")
        cols_to_split = st.multiselect(
            "Select columns to move",
            list(block.columns.keys()),
            key=f"split_cols_{block_name}",
        )
        if st.button("Split", key=f"split_btn_{block_name}"):
            if not split_name:
                st.error("Please provide a name")
            elif split_name in st.session_state.master_meta.structure:
                st.error("Block exists")
            else:
                new_block = ColumnBlockMeta(name=split_name, columns={})
                for c in cols_to_split:
                    new_block.columns[c] = block.columns[c]
                    del block.columns[c]

                st.session_state.master_meta.structure[split_name] = new_block
                save_state(f"split block: {split_name}")
                # Streamlit will automatically rerun after the callback


def constants_editor() -> None:
    """Editor for constants metadata."""
    st.header("Constants")

    wrap(
        st.text_area,
        "Edit Constants (JSON)",
        height=400,
        path=["constants"],
        converter=json.loads,
        key="constants_editor",
    )


def files_editor() -> None:
    """Editor for files metadata."""
    st.header("Files")

    if st.session_state.master_meta.files is None:
        st.write("No files defined.")
        return

    for i, fd in enumerate(st.session_state.master_meta.files):
        with st.expander(f"File: {fd.file} ({fd.code})"):
            wrap(st.text_input, "Code", path=["files", i, "code"], key=f"file_code_{i}")

            # Read options editing could be complex (dict)
            st.json(fd.opts)


def main() -> None:
    """Main entry point for the annotator tool."""
    if len(sys.argv) < 2:
        st.error("Usage: stk_annotator <path_to_meta.json>")
        st.info("Please provide the path to the annotation file you want to edit.")
        st.stop()

    meta_path = sys.argv[1]

    if "master_meta" not in st.session_state:
        init_state(meta_path)

    # Clear restoring flag if it was set (after rerun from undo/redo)
    if st.session_state.get("_restoring", False):
        st.session_state._restoring = False

    sidebar()

    st.title(f"Annotator: {os.path.basename(meta_path)}")

    if "mode" in st.session_state:
        if st.session_state.mode == "Blocks":
            block_editor()
        elif st.session_state.mode == "Constants":
            constants_editor()
        elif st.session_state.mode == "Files":
            files_editor()


if __name__ == "__main__":
    main()
