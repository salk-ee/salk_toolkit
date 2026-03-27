"""Constants view: edit DataMeta.constants."""

from __future__ import annotations

import json

import streamlit as st

from salk_toolkit.tools.annotator.framework import wrap


def constants_editor() -> None:
    """Editor for constants metadata."""
    st.header("Constants")

    wrap(
        st.text_area,
        "Edit Constants (JSON)",
        height=400,
        path=["constants"],
        i_to_o=json.loads,
        o_to_i=lambda val: json.dumps(val, indent=2) if not isinstance(val, str) else val,
        key="constants_editor",
    )
