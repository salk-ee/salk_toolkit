"""Files view: edit DataMeta.files."""

from __future__ import annotations

import streamlit as st

from salk_toolkit.tools.annotator.framework import wrap


def files_editor() -> None:
    """Editor for files metadata."""
    st.header("Files")

    if st.session_state.master_meta.files is None:
        st.write("No files defined.")
        return

    for i, fd in enumerate(st.session_state.master_meta.files):
        with st.expander(f"File: {fd.file} ({fd.code})"):
            wrap(st.text_input, "Code", path=["files", i, "code"], key=f"file_code_{i}")
            st.json(fd.opts)
