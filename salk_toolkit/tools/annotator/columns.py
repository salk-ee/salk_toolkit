"""Columns view: block and column editor UI.

Note: This file was renamed from `blocks.py` to `columns.py` to free up the
`blocks` name for future block-focused functionality.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from salk_toolkit.utils import rename_dict_key
from salk_toolkit.validation import BlockScaleMeta, ColumnBlockMeta, ColumnMeta, soft_validate
from pydantic import ValidationError

from salk_toolkit.tools.annotator.framework import (
    edit_json_modal,
    get_all_raw_columns,
    get_column_data,
    normalize_translate_dict,
    rebuild_data_cache,
    save_state,
    set_path_value,
    wrap,
)


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
        save_state(["reorder categories"])


@st.dialog("Remove Column")
def remove_column_dialog(block_name: str, col_name: str) -> None:
    """Dialog for removing a column from a block."""
    st.warning(f"Remove column '{col_name}' from block '{block_name}'?")
    st.caption("This is a structural change and will delete the column metadata entry.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Remove", type="primary", key=f"confirm_remove_{block_name}_{col_name}"):
            block = st.session_state.master_meta.structure.get(block_name)
            if not block or col_name not in block.columns:
                st.error("Column no longer exists.")
                return

            save_state(["structure", block_name, "columns", col_name])
            del block.columns[col_name]

            options: list[str] = []
            options.append("Block settings")
            if block.scale is not None:
                options.append("scale")
            options.extend(list(block.columns.keys()))
            st.session_state.selected_column = options[0] if options else None
            st.rerun()
    with col2:
        st.button("Cancel", key=f"cancel_remove_{block_name}_{col_name}")


def _render_continuous_visualization(df: pd.DataFrame, separate: bool) -> None:
    """Render visualization for continuous/numeric data."""
    df_valid = df.dropna(subset=["value"])
    if df_valid.empty:
        st.warning("No valid numeric data for density plot")
        return

    base_chart = alt.Chart(df_valid)
    if separate:
        chart = (
            base_chart.transform_density("value", as_=["density_value", "density"], groupby=["file_code"], steps=200)
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

    stats = df.groupby("file_code")["value"].describe() if separate else df["value"].describe().to_frame().T
    st.dataframe(stats)


def _render_datetime_visualization(df: pd.DataFrame, separate: bool) -> None:
    """Render visualization for datetime data."""
    df_valid = df.dropna(subset=["value"])
    if df_valid.empty:
        st.warning("No valid datetime data for visualization")
        return

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

    if separate:
        stats = df.groupby("file_code")["value"].agg(["count", "min", "max"]).T
    else:
        stats = pd.DataFrame({"count": [df["value"].count()], "min": [df["value"].min()], "max": [df["value"].max()]})
    st.dataframe(stats)


def _render_categorical_visualization(df: pd.DataFrame, col_meta: ColumnMeta, separate: bool) -> None:
    """Render visualization for categorical data."""
    df_plot = df.copy()
    translate_map = normalize_translate_dict(col_meta.translate) if col_meta.translate else {}
    base_vals = df_plot["value"].astype("string")
    translated = base_vals.replace(translate_map) if translate_map else base_vals

    translated = translated.mask(translated.isna() | (translated == ""), "NA")
    df_plot["translated_value"] = translated

    sort_order = None
    if col_meta.categories and isinstance(col_meta.categories, list):
        sort_order = [str(cat) for cat in col_meta.categories]
        all_trans = set(df_plot["translated_value"].dropna().unique())
        for trans in all_trans:
            if trans not in sort_order:
                sort_order.append(trans)

    color_scale = None
    if col_meta.colors:
        # Altair's default single-color is steelblue.
        default_altair_color = "#4682b4"

        if sort_order:
            domain = list(sort_order)
        else:
            # No explicit ordering: use observed translated values.
            domain = sorted({str(v) for v in df_plot["translated_value"].dropna().unique()})

        color_range: list[str] = []
        for cat in domain:
            color_val = col_meta.colors.get(cat)
            if color_val is None:
                color_range.append(default_altair_color)
            elif hasattr(color_val, "as_hex"):
                color_range.append(color_val.as_hex())
            else:
                color_range.append(str(color_val))

        if domain:
            color_scale = alt.Scale(domain=domain, range=color_range)

    encode_dict: dict[str, Any] = {
        "y": alt.Y("translated_value:N", axis=alt.Axis(title="Value"), sort=sort_order if sort_order else "-x"),
        "x": "count()",
    }

    if separate:
        encode_dict["color"] = "file_code:N"
        encode_dict["yOffset"] = "file_code:N"
    elif color_scale:
        encode_dict["color"] = alt.Color("translated_value:N", scale=color_scale)
    else:
        encode_dict["color"] = alt.value("steelblue")

    chart = alt.Chart(df_plot).mark_bar().encode(**encode_dict)
    st.altair_chart(chart, use_container_width=True)


def column_editor(block_name: str, col_name: str, col_meta: ColumnMeta) -> None:
    """Editor for column metadata."""
    if "master_meta" in st.session_state:
        block = st.session_state.master_meta.structure.get(block_name)
        if col_name == "scale":
            if block and block.scale is not None:
                col_meta = block.scale
        elif block and col_name in block.columns:
            col_meta = block.columns[col_name]

    if col_name == "scale":
        base_path: list[str | int] = ["structure", block_name, "scale"]
    else:
        base_path = ["structure", block_name, "columns", col_name]

    if col_name != "scale":
        rename_key = f"rename_{block_name}_{col_name}"
        if rename_key not in st.session_state:
            st.session_state[rename_key] = col_name

        type_key = f"type_{block_name}_{col_name}"
        current_type = "continuous" if col_meta.continuous else "datetime" if col_meta.datetime else "categorical"
        if type_key not in st.session_state:
            st.session_state[type_key] = current_type

        def _apply_column_rename() -> None:
            """Rename the column key in the block columns mapping."""
            new_name = str(st.session_state.get(rename_key, "")).strip()
            if new_name == col_name:
                return
            if new_name == "":
                st.error("Column name cannot be empty.")
                st.session_state[rename_key] = col_name
                return

            block = st.session_state.master_meta.structure.get(block_name)
            if not block or col_name not in block.columns:
                st.error("Column no longer exists.")
                return

            if new_name in block.columns:
                st.error(f"Column {new_name} already exists in this block.")
                st.session_state[rename_key] = col_name
                return

            save_state(["structure", block_name, "columns", col_name])
            # If the column used the implicit default source (== old column key), make it explicit
            # so that renaming the key does not break the link to the raw data column.
            meta_obj = block.columns[col_name]
            if getattr(meta_obj, "source", None) is None:
                meta_obj.source = col_name
            block.columns = rename_dict_key(block.columns, col_name, new_name)
            st.session_state.selected_column = new_name
            st.session_state[f"rename_{block_name}_{new_name}"] = new_name
            rebuild_data_cache()

        def _apply_column_type() -> None:
            """Update column type flags and keep categorical metadata consistent."""
            block = st.session_state.master_meta.structure.get(block_name)
            if not block or col_name not in block.columns:
                return

            meta_obj = block.columns[col_name]
            old_type = "continuous" if meta_obj.continuous else "datetime" if meta_obj.datetime else "categorical"
            new_type = str(st.session_state.get(type_key, old_type))
            if new_type == old_type:
                return

            save_state(base_path + ["continuous"])

            meta_obj.continuous = new_type == "continuous"
            meta_obj.datetime = new_type == "datetime"

            if new_type == "categorical":
                # Set explicit categories based on observed values so the categorical UI can render
                # without collapsing everything into NA defaults.
                col_prefix_local = getattr(getattr(block, "scale", None), "col_prefix", None)
                data_cn = f"{col_prefix_local}{col_name}" if col_prefix_local else col_name
                df_vals = get_column_data(data_cn)
                if not df_vals.empty:
                    cats = sorted({str(v) for v in df_vals["value"].dropna().unique()})
                    meta_obj.categories = cats
                else:
                    meta_obj.categories = "infer"
            else:
                # Clear categorical-only fields to satisfy ColumnMeta validation when categories is None.
                meta_obj.categories = None
                meta_obj.ordered = False
                meta_obj.likert = False
                meta_obj.groups = {}
                meta_obj.colors = {}
                meta_obj.num_values = None
                meta_obj.topo_feature = None

            rebuild_data_cache()

        name_col, type_col, remove_col = st.columns([7, 3, 2])
        with name_col:
            st.text_input("Column name", key=rename_key, on_change=_apply_column_rename)
        with type_col:
            st.selectbox(
                "Type",
                options=["categorical", "continuous", "datetime"],
                key=type_key,
                on_change=_apply_column_type,
                label_visibility="collapsed",
            )

        with remove_col:
            if st.button("Remove Column…", type="secondary", key=f"remove_btn_{block_name}_{col_name}"):
                remove_column_dialog(block_name, col_name)
    else:
        st.info(
            "Scale metadata applies to all columns in this block. Individual column metadata overrides scale settings."
        )

    label_col, edit_col = st.columns([10, 2])
    with label_col:
        wrap(st.text_input, "Label", path=base_path + ["label"], key=f"label_{block_name}_{col_name}")
    with edit_col:
        if st.button(
            "Edit Column Meta (JSON)…",
            type="secondary",
            key=f"edit_meta_btn_{block_name}_{col_name}",
        ):
            edit_json_modal(base_path)

    # Data fetch
    block_local = st.session_state.master_meta.structure.get(block_name) if "master_meta" in st.session_state else None
    col_prefix = None
    if block_local is not None and getattr(block_local, "scale", None) is not None:
        col_prefix = getattr(block_local.scale, "col_prefix", None)

    if col_name == "scale":
        if hasattr(col_meta, "col_prefix"):
            wrap(st.text_input, "Column Prefix", path=base_path + ["col_prefix"], key=f"col_prefix_{block_name}_scale")

        # Mirror per-column display, but aggregate data across *all* columns in the block.
        frames: list[pd.DataFrame] = []
        if block_local is not None:
            for orig_cn in block_local.columns.keys():
                data_cn = f"{col_prefix}{orig_cn}" if col_prefix else orig_cn
                sub = get_column_data(data_cn)
                if not sub.empty:
                    frames.append(sub)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["value", "file_code"])
        data_key_for_cache = f"{block_name}__scale__{len(frames)}"
    else:
        data_cn = f"{col_prefix}{col_name}" if col_prefix else col_name
        df = get_column_data(data_cn)
        data_key_for_cache = data_cn

    is_categorical = not col_meta.continuous and not col_meta.datetime
    if is_categorical:
        col1, col2, col3 = st.columns(3)
        with col1:
            ordered = wrap(st.checkbox, "Ordered", path=base_path + ["ordered"], key=f"ordered_{block_name}_{col_name}")

        if ordered:
            with col2:
                wrap(st.checkbox, "Likert", path=base_path + ["likert"], key=f"likert_{block_name}_{col_name}")
            with col3:
                cats = col_meta.categories if isinstance(col_meta.categories, list) else []
                wrap(
                    st.multiselect,
                    "Non-ordered categories",
                    options=cats,
                    path=base_path + ["nonordered"],
                    key=f"nonordered_{block_name}_{col_name}",
                )

    if df.empty:
        st.warning(f"No data found for {col_name}")
        st.json(col_meta.model_dump(mode="json"))
        return

    separate = st.session_state.get("separate_files", True)
    # Counts depend both on scale/column metadata and on which underlying data columns are included.
    meta_hash = hash(str(col_meta.model_dump(mode="json")) + str(data_key_for_cache))
    cache_key_counts = f"col_counts_{block_name}_{data_key_for_cache}_{separate}_{meta_hash}"

    if cache_key_counts not in st.session_state:
        prefix_key = f"col_counts_{block_name}_{data_key_for_cache}_"
        old_keys = [k for k in st.session_state.keys() if k.startswith(prefix_key)]
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

    if col_meta.continuous:
        _render_continuous_visualization(df, separate)
    elif col_meta.datetime:
        _render_datetime_visualization(df, separate)
    else:
        _render_categorical_visualization(df, col_meta, separate)

    if (not col_meta.continuous and not col_meta.datetime) and st.button(
        "Reorder Categories", key=f"reorder_btn_{block_name}_{col_name}"
    ):
        reorder_dialog(col_meta)

    if not col_meta.continuous and not col_meta.datetime:
        st.subheader("Category Mapping")

        unique_vals_str = set(counts.index.astype(str))
        counts_index_map = {str(idx): idx for idx in counts.index}

        trans_to_orig: defaultdict[str, list[str]] = defaultdict(list)
        translate_dict = normalize_translate_dict(col_meta.translate) if col_meta.translate else {}

        init_flag = f"_translate_initialized_{block_name}_{col_name}"
        if not st.session_state.get(init_flag, False):
            # Do not auto-populate `translate` in metadata. Missing mappings are treated as identity.
            st.session_state[init_flag] = True

        for orig_str in unique_vals_str:
            trans_key = translate_dict.get(orig_str, orig_str)
            trans_to_orig[trans_key].append(orig_str)

        if col_meta.categories and isinstance(col_meta.categories, list):
            not_in_cats = [k for k in trans_to_orig.keys() if k not in col_meta.categories]
            group_order = [str(cat) for cat in (col_meta.categories + not_in_cats)]
        else:
            group_order = sorted(set(trans_to_orig.keys()))

        na_count = df["value"].isna().sum()
        has_na = na_count > 0

        rows: list[dict[str, Any]] = []
        for trans_value in group_order:
            origs_in_group = sorted(trans_to_orig[trans_value])
            for i, orig in enumerate(origs_in_group):
                row: dict[str, Any] = {"original": orig}
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

                row["translation"] = trans_value if trans_value else ""
                row["is_first_in_group"] = i == 0
                row["translated_value"] = trans_value if trans_value else None
                rows.append(row)

        if has_na:
            row = {"original": "NA"}
            if separate:
                na_counts = df[df["value"].isna()].groupby("file_code").size()
                for col in counts.columns:
                    row[col] = int(na_counts.get(col, 0))
            else:
                row["count"] = int(na_count)
            row["translation"] = ""
            row["is_first_in_group"] = False
            row["translated_value"] = None
            rows.append(row)

        show_num_values = col_meta.ordered and not col_meta.continuous and not col_meta.datetime

        if separate:
            num_count_cols = len(counts.columns)
            col_widths = [2] + [1] * num_count_cols + ([2, 1, 1] if show_num_values else [2, 1])
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
            cols = st.columns([2, 1, 2, 1, 1] if show_num_values else [2, 1, 2, 1])
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

        for i, row in enumerate(rows):
            orig = row["original"]
            if separate:
                num_count_cols = len(counts.columns)
                col_widths = [2] + [1] * num_count_cols + ([2, 1, 1] if show_num_values else [2, 1])
                cols = st.columns(col_widths)
            else:
                cols = st.columns([2, 1, 2, 1, 1] if show_num_values else [2, 1, 2, 1])

            cols[0].text(orig)

            if separate:
                for j, file_col in enumerate(counts.columns):
                    count_val = row.get(file_col, 0)
                    cols[j + 1].markdown(f"<div style='text-align: right;'>{count_val}</div>", unsafe_allow_html=True)
            else:
                cols[1].markdown(f"<div style='text-align: right;'>{row.get('count', 0)}</div>", unsafe_allow_html=True)

            if orig != "NA":
                translate_key = orig
                trans_col_idx = len(counts.columns) + 1 if separate else 2
                with cols[trans_col_idx]:

                    def update_categories_from_translations(
                        new_val: str | None,
                        old_val: str | None,
                        path: list[str | int],
                    ) -> None:
                        """Update categories/colors when translations change (rename/merge/split/realign)."""
                        block = st.session_state.master_meta.structure.get(block_name)
                        if not block or col_name not in block.columns:
                            return

                        if not (len(path) >= 2 and path[-2] == "translate"):
                            return

                        changed_key = str(path[-1])

                        # Work off the *already updated* translate mapping (wrap() applied the change).
                        col_meta_local = block.columns[col_name]
                        translate_dict = (
                            normalize_translate_dict(col_meta_local.translate) if col_meta_local.translate else {}
                        )

                        # Only auto-maintain explicit order if categories is already explicit.
                        if not (col_meta_local.categories and isinstance(col_meta_local.categories, list)):
                            return

                        cats = [str(c) for c in col_meta_local.categories]

                        def _label(val: str | None) -> str:
                            if val is None:
                                return changed_key  # identity mapping
                            if val == "":
                                return ""  # maps to NA
                            return str(val)

                        old_lbl = _label(old_val)
                        new_lbl = _label(new_val)

                        # Determine which translated labels are used by observed raw values after the edit.
                        used: set[str] = set()
                        for orig in unique_vals_str:
                            mapped = translate_dict.get(orig, orig)
                            if mapped != "":
                                used.add(mapped)

                        old_used = old_lbl != "" and old_lbl in used
                        new_used = new_lbl != "" and new_lbl in used
                        old_in = old_lbl in cats
                        new_in = new_lbl in cats

                        colors_path = base_path + ["colors"]
                        colors = dict(col_meta_local.colors or {})

                        # Update ordering + colors based on the four-case model.
                        if new_lbl == "":
                            # Map-to-NA: drop old category/color if it is no longer used.
                            if old_in and not old_used:
                                cats = [c for c in cats if c != old_lbl]
                                colors.pop(old_lbl, None)
                        elif not old_used and (not new_in) and new_used:
                            # Rename: old disappears, new appears (preserve position; move old color if new missing).
                            if old_in:
                                cats[cats.index(old_lbl)] = new_lbl
                            else:
                                cats.append(new_lbl)
                            if old_lbl in colors and new_lbl not in colors:
                                colors[new_lbl] = colors.pop(old_lbl)
                            else:
                                colors.pop(old_lbl, None)
                        elif not old_used and new_in and new_used:
                            # Merge: old disappears, new already present (do NOT touch new color).
                            if old_in:
                                cats = [c for c in cats if c != old_lbl]
                            colors.pop(old_lbl, None)
                        elif old_used and (not new_in) and new_used:
                            # Split: old remains, new appears (insert new after old).
                            if old_in:
                                cats.insert(cats.index(old_lbl) + 1, new_lbl)
                            else:
                                cats.append(new_lbl)
                        else:
                            # Realign (or no-op): keep order as-is.
                            pass

                        # Ensure all used labels exist in categories (append missing deterministically).
                        missing = sorted(used - set(cats))
                        if missing:
                            cats.extend(missing)

                        set_path_value(st.session_state.master_meta, base_path + ["categories"], cats)
                        set_path_value(st.session_state.master_meta, colors_path, colors)

                    wrap(
                        st.text_input,
                        "Trans",
                        label_visibility="collapsed",
                        path=base_path + ["translate", translate_key],
                        # Show identity mapping by default, but keep it implicit (not stored).
                        default_value=orig,
                        i_to_o=lambda s, _orig=orig: None if s is None or str(s).strip() in ("", _orig) else s,
                        on_change=update_categories_from_translations,
                        key=f"trans_{block_name}_{col_name}_{orig}",
                    )
            else:
                trans_col_idx = len(counts.columns) + 1 if separate else 2
                cols[trans_col_idx].text("NA")

            if show_num_values:
                num_val_col_idx = len(counts.columns) + 2 if separate else 3
                if row.get("is_first_in_group", False) and orig != "NA" and row.get("translated_value"):
                    trans_cat = str(row["translated_value"])

                    cat_index = None
                    if col_meta.categories and isinstance(col_meta.categories, list):
                        try:
                            cat_index = col_meta.categories.index(trans_cat)
                        except ValueError:
                            for idx, cat in enumerate(col_meta.categories):
                                if str(cat) == trans_cat:
                                    cat_index = idx
                                    break

                    if cat_index is not None:
                        with cols[num_val_col_idx]:

                            def num_o_to_i(val: Any) -> str:  # noqa: ANN401
                                """Render numeric category mapping in a compact form (ints stay ints)."""
                                if isinstance(val, int):
                                    return str(val)
                                if isinstance(val, float):
                                    return str(int(val)) if val.is_integer() else str(val)
                                return ""  # Other cases - return ""

                            def num_i_to_o(txt: str | None) -> int | float | None:
                                """Parse numeric category mapping from text ('' -> None)."""
                                if txt is None:
                                    return None
                                try:
                                    f = float(txt)
                                except ValueError:
                                    return None
                                return int(f) if f.is_integer() else f

                            default_num_val = cat_index + 1
                            wrap(
                                st.text_input,
                                "Num Value",
                                label_visibility="collapsed",
                                path=base_path + ["num_values", cat_index],
                                default_value=str(default_num_val),
                                o_to_i=num_o_to_i,
                                i_to_o=num_i_to_o,
                                key=f"num_val_{block_name}_{col_name}_{trans_cat}_{cat_index}",
                            )
                    else:
                        cols[num_val_col_idx].empty()
                else:
                    cols[num_val_col_idx].empty()

            if show_num_values:
                color_col_idx = len(counts.columns) + 3 if separate else 4
            else:
                color_col_idx = len(counts.columns) + 2 if separate else 3

            if row.get("is_first_in_group", False) and orig != "NA" and row.get("translated_value"):
                trans_cat = str(row["translated_value"])
                color_path = base_path + ["colors", trans_cat]

                with cols[color_col_idx]:

                    def color_o_to_i(val: Any) -> str | None:  # noqa: ANN401
                        """Convert Color object to hex string for color picker input."""
                        if val is None:
                            return None
                        if hasattr(val, "as_hex"):
                            return val.as_hex()
                        if isinstance(val, str):
                            return val if val.startswith("#") else None
                        try:
                            str_val = str(val)
                            return str_val if str_val.startswith("#") else None
                        except Exception:
                            return None

                    wrap(
                        st.color_picker,
                        "Color",
                        label_visibility="collapsed",
                        path=color_path,
                        default_value="#FFFFFF",
                        o_to_i=color_o_to_i,
                        key=f"color_{block_name}_{col_name}_{trans_cat}",
                    )
            else:
                cols[color_col_idx].empty()


def block_editor() -> None:
    """Editor for block metadata."""
    if "master_meta" not in st.session_state:
        st.info("No metadata loaded")
        return

    block_names = list(st.session_state.master_meta.structure.keys())
    if not block_names:
        st.info("No blocks defined")
        return

    if "selected_block" not in st.session_state or st.session_state.selected_block not in block_names:
        st.session_state.selected_block = block_names[0]

    block_name = st.session_state.selected_block
    block = st.session_state.master_meta.structure.get(block_name)
    if not block:
        st.error(f"Block {block_name} not found")
        return

    def block_settings_view() -> None:
        """Render block-level settings UI for the selected block."""

        # Shared scale toggle:
        # - When toggled OFF: stash existing scale in session_state and set `block.scale=None`.
        # - When toggled ON: restore stashed scale (or create a default) and set `block.scale=<BlockScaleMeta>`.
        stash_key = "_block_scale_stash"
        if stash_key not in st.session_state:
            st.session_state[stash_key] = {}
        stash: dict[str, object] = st.session_state[stash_key]

        if block.scale is not None:
            stash[block_name] = block.scale.model_dump(mode="json")

        toggle_key = f"_shared_scale_{block_name}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = block.scale is not None

        def _toggle_shared_scale() -> None:
            if st.session_state.get("_initializing", False) or st.session_state.get("_restoring", False):
                return

            want_scale = bool(st.session_state.get(toggle_key))
            block_local = st.session_state.master_meta.structure.get(block_name)
            if block_local is None:
                return

            if want_scale and block_local.scale is None:
                payload = stash.get(block_name, {})
                try:
                    scale_obj = soft_validate(payload, BlockScaleMeta, context={"track_constants": True})
                except ValidationError:
                    scale_obj = BlockScaleMeta()
                save_state(["structure", block_name, "scale"])
                block_local.scale = scale_obj

            if (not want_scale) and block_local.scale is not None:
                stash[block_name] = block_local.scale.model_dump(mode="json")
                save_state(["structure", block_name, "scale"])
                block_local.scale = None

        st.toggle("Shared scale", key=toggle_key, on_change=_toggle_shared_scale)

        def handle_block_rename(new_name: str | None, old_name: str | None, path: list[str | int]) -> None:
            """Handle block rename: update dictionary key when block.name changes."""
            if new_name == old_name:
                return

            new_name_str = str(new_name) if new_name is not None else ""

            if new_name_str in st.session_state.master_meta.structure and new_name_str != block_name:
                st.error(f"Block {new_name_str} already exists")
                set_path_value(st.session_state.master_meta, path, old_name)
                return

            if new_name_str != block_name:
                block_obj = st.session_state.master_meta.structure[block_name]
                st.session_state.master_meta.structure[new_name_str] = block_obj
                del st.session_state.master_meta.structure[block_name]
                st.session_state.selected_block = new_name_str

        wrap(
            st.text_input,
            "Rename Block",
            path=["structure", block_name, "name"],
            on_change=handle_block_rename,
            key=f"rename_block_{block_name}",
        )

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
                    save_state(["structure", block_name, "columns", final_name])

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
                save_state(["split_block", split_name])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.selectbox("Select Block", block_names, key="selected_block")
    with col2:
        column_options: list[str] = ["Block settings"]
        if block.scale is not None:
            column_options.append("scale")
        column_options.extend(list(block.columns.keys()))

        if "selected_column" not in st.session_state or st.session_state.selected_column not in column_options:
            st.session_state.selected_column = column_options[0] if column_options else None

        if column_options:
            st.selectbox("Select Column", column_options, key="selected_column")
        else:
            st.info("No columns in this block")
            return

    selected_col = st.session_state.selected_column
    if selected_col == "Block settings":
        st.subheader("Block settings")
        block_settings_view()
    elif selected_col == "scale" and block.scale is not None:
        st.subheader("Scale Settings")
        column_editor(block_name, "scale", block.scale)
    elif selected_col in block.columns:
        column_editor(block_name, selected_col, block.columns[selected_col])
    else:
        st.warning(f"Column {selected_col} not found")
        return
