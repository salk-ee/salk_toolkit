"""Streamlit-based annotation editor for survey metadata."""

from __future__ import annotations

import os
import sys

import streamlit as st

from salk_toolkit.tools.annotator.columns import block_editor
from salk_toolkit.tools.annotator.constants import constants_editor
from salk_toolkit.tools.annotator.files import files_editor
from salk_toolkit.tools.annotator.framework import init_state, sidebar


st.set_page_config(
    layout="wide",
    page_title="SALK Annotator",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main entry point for the annotator tool."""
    if len(sys.argv) < 2:
        st.error("Usage: stk_annotator <path_to_meta.json>")
        st.info("Please provide the path to the annotation file you want to edit.")
        st.stop()

    meta_path = sys.argv[1]

    if "master_meta" not in st.session_state:
        init_state(meta_path)

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
