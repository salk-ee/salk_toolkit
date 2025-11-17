# TOOL-001: Explorer Tool Spec

**Last Updated**: 2025-11-13
**Status**: ✅ Complete
**Module**: Tool
**Tags**: `#tool`, `#explorer`, `#plotting`
**Dependencies**: None

## Overview

Explorer is an interactive Streamlit tool for exploratory analysis on annotated datasets. It wraps the plotting pipeline, enabling rapid faceting/filtering and seamless handoff of `plot_desc` into dashboard modules. The tool must also function as a secured, shareable mini-dashboard for sharing data with external partners.

## Business Context

- Provide analysts with a fast, code-free interface to inspect annotated survey data and validate preprocessing outputs.
- Accelerate dashboard development by letting teams prototype charts and copy generated `plot_desc` JSON directly into `sdb.plot`.
- Support partner deployments where Explorer runs as a gated dashboard instance with Frontegg authentication and event logging aligned with existing dashboards.
- Maintain parity with plotting pipeline features (filters, faceting, export, aggregations) while handling large parquet inputs efficiently.

### Clarifications (Q&A)

- **Q**: What are the prioritized workflows?  
  **A**: Exploratory charting with flexible filters/facets, plus generating `plot_desc` for dashboards.
- **Q**: Should Explorer operate like other dashboards with authentication/logging?  
  **A**: Yes, it can be deployed as a dashboard with the standard access controls for partner datasets.

## Requirements

**Files to Create/Modify:**

- `salk_toolkit/tools/explorer.py`: Maintain core Streamlit app, syncing with plotting pipeline capabilities and authentication hooks.
- `nbs/11_commands.ipynb`: Add a function that executes it as a streamlit app that can be referenced from `settings.ini`
- `settings.ini`: Ensure `stk_explorer` entry point remains registered for CLI launches.
- Deployment configs (`.streamlit/secrets.example`, partner runbooks) as needed to document auth/logging setup.

**Functionality:**

- Load one or more parquet datasets (with optional external `meta` JSON) and expose them for selection, aligning column metadata via `read_parquet_with_metadata`.
- Persist UI state (localStorage + `st.session_state`) so analysts can refresh without losing configuration.
- Provide sidebar controls for observation selection, facet dimensions, filters (including grouped filters), aggregation toggles, and advanced overrides.
- Surface available plots from the plotting pipeline (`matching_plots`, `get_plot_meta`) with auto-default behavior and plot-argument controls.
- Render plots for each selected dataset or multi-dataset facet, honoring export width controls and offering HTML/CSV downloads.
- Emit `plot_desc`/args payload preview that can be copied directly into dashboards (`sdb.plot`).
- Optional admin panel for authenticated users leveraging `FronteggAuthenticationManager`, S3 logging, and organization whitelists.
- Maintain performance safeguards (lazy loading via Polars, `altair` transformer settings, warning suppression) for large datasets.
- If comparing multiple files, add a dummy facet choice `input_files` that combines all the files and facets over the file dimension to make it easy to compare the files. 

## Implementation Plan

### Foundation Setup

- [x] Catalogue current plotting pipeline APIs  (plot registry, filters, transforms) and document expected contracts.
- [x] Document authentication/logging requirements and shared utilities (`FronteggAuthenticationManager`, `log_event`) for dashboard parity.

### Core Development

- [x] Audit UI controls to ensure full coverage of plotting pipeline options (facets, transforms, aggregation, overrides, export).
- [x] Clarify handling of multi-file comparisons (categorical alignment, schema harmonization) and document expected behavior.

## Definition of Done

- [x] Explorer supports prioritized workflows: exploratory analysis, `plot_desc` generation, partner deployments.
- [x] Authentication, logging, and export features align with dashboard standards.
- [x] Plotting pipeline compatibility verified (matching plots, transforms, filters).
- [x] Regression coverage added/updated for critical Explorer pathways.

## Implementation Notes

- Keep Streamlit app self-contained; avoid importing from notebooks except through exported modules.
- Monitor performance impacts of new features; large parquet files must remain responsive.
- Retain localStorage/session-state persistence patterns when refactoring the sidebar.
- Capture future clarifications in this section to keep spec a living reference.

