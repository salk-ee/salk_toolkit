# Annotation block editor tool

**Modules:** `salk_toolkit/tools/annotation_editor.py` (planned), helpers in
`salk_toolkit/io/`, `salk_toolkit/validation.py` (`DataMeta`)

## Goal

Data annotation metafiles are edited by hand, which is error-prone and gives no feedback on
what the data behind a block actually looks like — the common case being a new wave of a
multi-wave survey whose coding drifted. This is a Streamlit editor for those metafiles:
block-level structure editing, a category/translation table backed by live value counts, undo,
and validated save. Not built yet — this is the design of record, and the io helper names below
are targets rather than existing functions.

## Design

**Shape.** The metafile path is a CLI argument (no in-app upload). The app is one sidebar
mode selectbox — `Blocks`, `Constants`, `Files` — driving the main pane, plus a save section.

**State.** `st.session_state['master_meta']` holds the full annotation JSON as the single
mutable copy, with the unresolved original kept alongside in `original_meta` and the extracted
constants in `constants`. Widgets do not own state: a binding wrapper around
`selectbox`/`multiselect`/`checkbox`/`text_input` takes a `path/in/meta/structure`, reads its
current value out of `master_meta` on render, and writes it back on change. Every edit takes
effect immediately; only `st.dialog` modals (raw-JSON edit, category reorder) have save/cancel.
At the end of each run the end state is compared to the start state, and any difference pushes
a full `master_meta` snapshot onto the undo stack — no diffing, no history cap.

**Blocks mode.** Sidebar picks the block and toggles `Separate` (per-file breakdowns, on by
default). The main pane renames the block, lists its columns as expanders — plus a pseudo-`All`
entry that edits the block `scale` through *the same function* as a column — and offers "add
column" (source column selectbox, optional new name) and "split block" (new name + multiselect
of columns to move).

**Column editor**, one `st.fragment` per column so a 40-column block stays responsive:

- Header shows the label and the original source column names, with a warning icon and an
  explicit warning listing any other block using the same source column.
- Visualization of post-`transform`, post-`translate` values, in Altair: a density plot for
  continuous columns with count-of-missing plus mean/median/std (a table by `file_code` when
  Separate), or a horizontal histogram for categoricals in declared category order, using the
  annotation `colors`, split by `yOffset` on `file_code` when Separate, with NA appended as a
  final category in `#cc4560` if any values are missing.
- `ordered` / `likert` toggles, the `nonordered` multiselect, and the reorder-categories
  dialog button (`streamlit_sortables`) share one row.
- Category table, one row per *original* value: the raw value, its count (a column per file
  when Separate, headed by `file_code`), a text input for its `translate` target (empty string
  meaning `None`), the `num_values` entry, and a colour picker. Rows that translate to the same
  value are grouped together — groups ordered by the column's category order, alphabetically
  within a group — and the numeric value and colour are editable only on the group's first row,
  since they belong to the translated category, not the raw value. Values absent from a file
  show a `0` on yellow; NA gets its own row when present.
- A dialog for raw JSON editing of the column meta (validated against `ColumnMeta`), and a
  remove button.

**Constants mode** lists constants as JSON text areas with parse errors surfaced inline, plus
add and delete. Editing a field that *originated* from a constant edits the constant, not the
resolved local copy: `infer_constant_source(meta, path)` takes the unresolved meta and a
path like `structure['issues'].scale.categories` and returns the constant key or `None`, and
the binding wrapper redirects the write accordingly. Any constant edit re-resolves
`master_meta`.

**Files mode** lists the `files` entries with their codes, paths and read options; codes are
renameable, options editable as JSON, entries removable, and data files sitting next to the
metafile are auto-detected and offered.

**Save** takes an output path (defaulting to the original), re-inserts constants as references
in place of their resolved values, serializes, and writes. Overwriting the original first
copies it to `{original_name}.orig.{i}.json` with `i` incremented until unique.

**Shared helpers.** The tool loads data through the io package rather than reimplementing it,
which needs a small amount of extraction:

- `load_annotation_meta(meta_path) -> DataMeta`-shaped raw dict — file-map lookup, JSON/YAML
  parse, `soft_validate` — returning the *unresolved* meta, since file resolution needs the
  constants intact.
- `resolve_annotation_constants(meta) -> (resolved_meta, constants)`.
- `load_annotation_inputs(...) -> (raw_df, env)` over the io loading layer, for the raw values
  the category table counts.
- `build_block_column_index(meta) -> {block_to_columns, column_to_block, duplicate_columns}`.

## Implementation notes

- `DataMeta` is the contract, not an implementation detail: this tool is a UI *for that
  format*, so any apparent mismatch between UI and model is resolved by changing the UI.
- The rule for extraction is 5+ line logic blocks become shared helpers in `io/`; one-liners
  stay duplicated. Divergence between the tool's view of a metafile and the pipeline's is the
  failure mode worth paying an indirection for.
- Scanning the whole `master_meta` on every render to find cross-block column usage is fine —
  annotation files are small enough that no index or cache is warranted.
- Beware rerun loops: writes go through the bindings on change only, and dialogs commit once.
