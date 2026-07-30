"""Tiny scratch file for local Streamlit/Altair theme debugging."""

# pyright: ignore[reportMissingImports, reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]

import streamlit as st
import altair as alt


print(alt.theme.get()())
print(alt.theme.options)

print(st.context.theme.type)
