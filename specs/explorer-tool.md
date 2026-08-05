# Explorer tool

**Modules:** `salk_toolkit/tools/explorer.py`, `salk_toolkit/commands.py`
(`run_explorer` / the `stk_explorer` entry point)

## Goal

Explorer is a Streamlit front end for the plot pipeline: point it at processed parquet files
and every plot the pipeline can draw becomes reachable through sidebar controls, with no code.
Its two jobs are exploratory analysis of annotated survey data and generating the
`pp_desc` descriptor that a dashboard then hard-codes — the descriptor shown in the sidebar is
literally the dict handed to `pp_transform_data`, so a chart that looks right in Explorer
looks the same in `sdb.plot`.

## Design

**Input.** `stk_explorer` shells out to `streamlit run tools/explorer.py`, forwarding CLI
arguments. A leading `.json` argument is loaded as a global `DataMeta` override (soft-validated
after `replace_constants`) that replaces each file's embedded meta; remaining path arguments
become the default file selection, and any other `*.parquet` in the working directory is
offered in the sidebar multiselect. Files load through
`read_parquet_with_metadata(..., lazy=True)` behind `@st.cache_resource`, so the frames stay
polars LazyFrames until the pipeline collects them.

**State.** All widgets are keyed into `st.session_state`, which is mirrored to browser
`localStorage` on every rerun and restored on first load, so a refresh preserves the whole
configuration. "Reset choices" clears both. `stss_safety` drops any restored selection that is
no longer a legal option for the current dataset — otherwise a stale key from another file
crashes the sidebar before it can be reset.

**Descriptor construction.** The sidebar builds up an `args` dict incrementally and validates
it into a `PlotDescriptor` only at the point of use:

- `get_dimensions` walks `data_meta.structure`, skipping hidden blocks, and yields block names
  for scaled blocks (with columns present in the data) and bare column names otherwise. That
  list feeds the **Observation** selectbox (`res_col`, temporal columns filtered out) and the
  facet picker; the observation column's `modifiers` are prepended to the facetable dimensions.
- `facet_ui` and `filter_ui` (shared with `dashboard.py`) produce `facet_dims` and `filter`;
  `impute_facet_dims` makes the facet list explicit, once before plot selection and again
  after, since the chosen plot can change it.
- `matching_plots` narrows the registry to what the descriptor supports and the **Plot type**
  selectbox offers those plus a `default` that resolves to the first match. The chosen plot's
  `PlotMeta.args` are rendered as typed widgets (bool → toggle, int → number input, list →
  selectbox) into `plot_args`.
- Advanced controls add `convert_res`, `cont_transform`, `agg_fn`, `sort` and explicit
  question position, plus a free-form **Override keys** text area `eval`'d into `args` —
  which also feeds `_update_data_meta_with_pp_desc` so a `res_meta`/`col_meta` override
  reshapes the column metadata the rest of the UI reads.

**Rendering.** With one file selected, or several without file faceting, each file gets its
own column: the descriptor is filtered down to that file's columns, run through
`pp_transform_data` + `create_plot`, and drawn with `draw_plot_matrix`; the header reports what
share of the data survived filtering (`filtered_size / total_size`). Selecting the synthetic
`input_file` facet instead transforms each file separately, concatenates the results, adds an
`input_file` Categorical, and renders one faceted chart — the multi-file comparison path.

**Export** (behind a toggle, since generating it is slow) offers custom width/height, publish
mode, a Vega Editor link, HTML and data-CSV downloads, and an iframe snippet with the HTML
inlined as base64. A pasted Vega-Lite spec is stored as `custom_spec` and overrides rendering
for the first file, so a chart can be hand-tuned in the Vega Editor and brought back.

**Inspection.** Expanders show the current `pp_desc`, the parsed data meta, and — for parquets
written by SIP — the model meta, split by `sequence` step.

## Implementation notes

- Explorer is excluded from pdoc builds and imports its heavy dependencies inside a
  `st.spinner` block, because the module executes on import.
- Altair schema validation is disabled (`alt.utils.schemapi.debug_mode(False)`) and the max-rows
  limit lifted — altair re-validates far more than it needs to and dominates render time on
  survey-sized frames.
- `get_plot_width` deliberately overrides the `st_dimensions`-based sizing: measuring the real
  container width triggers a rerun that can measure again, and the resulting refresh loop is
  worse than a fixed `min(800, 1200/ncols)`.
- When files disagree on a column's dtype, the concatenated multi-file frame is coerced to the
  first file's categorical dtype — differing `ordered` flags across waves are common and
  otherwise silently split the axis.
- The tool carries no authentication or logging of its own; the legacy Frontegg/admin hooks
  were removed. Deploying it for external partners means putting it behind a dashboard's auth.
