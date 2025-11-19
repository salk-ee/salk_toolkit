# TOOL-002: JSON Block Editor

**Last Updated**: 2025-11-13
**Status**: 🚧 Planning
**Module**: Tool
**Tags**: `#tool`, `#streamlit`, `#data-annotation`
**Dependencies**: salk_toolkit/io.py, salk_toolkit/validation.py

## Overview

Streamlit-based utility for inspecting and editing data annotation JSON metafiles used by the data ingestion pipeline. Provides block-level editing, category management, and persistence so analysts can adjust annotations without hand-editing JSON.

## Problem Context

- Problem being solved: Manual JSON edits for block metadata are error prone and lack immediate feedback on survey data structure.
- Intended use cases:
  - Data analysts iteratively improving data annotations
  - Adding a new wave of data in a multi-wave survey where coding might change from wave to wave
- Technical constraints and requirements:
  - Must validate annotations against `DataMeta`
  - Reuse `process_annotated_data` to load data
  - Operate within Streamlit’s session state model.
- Integration points with existing systems:
  - Consumes annotations defined in `salk_toolkit/io.py`
  - Relies on validation models from `salk_toolkit/validation.py`

## Requirements

**Important context functions/files**
- `salk_toolkit/validation.py`: The pydantic model of annotation files is defined here as `DataMeta`. This structure should not be altered without explicit confirmation, as this tool is intended as a UI for editing this format. 

**Files to Create/Modify:**
- `salk_toolkit/tools/json_block_editor.py`: Main Streamlit app orchestrating all functionality.
- `salk_toolkit/io.py`: abstract out helper functions from inside `process_annotated_data` to use in the tool as needed.
  - `load_annotation_meta(meta_path: str) -> dict` performs the file-map lookup, YAML/JSON parsing, and `soft_validate` call so both CLI and Streamlit paths rely on the same loader. Returns the raw (unresolved) meta dict.
  - `resolve_annotation_constants(meta: dict) -> tuple[dict, dict]` extracts constants from meta dict (before replacement), then applies `replace_constants`. Returns `(resolved_meta_with_constants_removed, constants_dict)` tuple.
  - `load_annotation_inputs(meta: dict, meta_path: str | None, data_file: str | None, raw_data: pd.DataFrame | None)` wraps `read_concatenate_files_list` and warning handling, returning `(raw_df, einfo)` for both pipeline and UI consumers. Note: `meta` should be unresolved (with constants) for proper file resolution.
  - `build_block_column_index(meta: dict) -> dict` exposes the block/column enumeration. Returns a dict with keys:
    - `'block_to_columns'`: dict mapping block name -> list of column names in that block
    - `'column_to_block'`: dict mapping column name -> block name it belongs to
    - `'duplicate_columns'`: dict mapping column name -> list of block names where it appears (for warning badges)

**Shared:**

- File loading: Use `load_annotation_meta` helper to load and validate JSON/YAML, then use `load_annotation_inputs` to get raw data. For processed data for visualizations, use `process_annotated_data(..., return_meta=True)` (after constants are resolved).
- The annotation file path is passed in as a CLI argument when launching the tool (no in-app upload/browser)
- Three modes: `Blocks`, `Constants`, `Files` modes, with a selectbox choice in sidebar. Each mode drives main-pane content

**Blocks:**
- Sidebar contains
 - Choice of block - selectbox
 - "Separate" toggle, on by default
- Main pane functionality:
  - Rename block
  - Column editors as expanders (more below)
    - a pseudo "All" entry for the block `scale` + one per column
  - Add column(s)
    - selectbox for column (from list of all columns from all files combined)
    - text_input for name (by default, blank in which case use the name of original column)
    - "Add" button to confirm the action
  - Split block
    - text_input for name of the new block
    - multi-select to choose which columns to split off from this block
    - "Split" button to confirm the action

**Column editor:**

- Expander text also shows `label` (if present) and names of original columns 
  - Original columns have a warning icon if it is also used in another block
  - It is acceptable to scan the entire `master_meta` each render to determine other-block usage; dataset sizes are small enough that no cache/index is required.
- rename column text_input
- warning with list of other blocks this column is in
- Visualization - visualize the values of the column after transform and translate
  - All plots use Altair. Rely on Altair’s aggregations; no extra downsampling guardrails are needed for large datasets at this stage.
  - For `continuous` columns - show density plot
    - Colored by `file_code` if "Separate"
    - Show count of missing values + mean, median, std
      - or a table of these values by `file_code` if "Separate"
  - For categorical columns - show horizontal histogram of values 
    - with `yOffset` on `file_code` if separate
    - Show categories in the same order they are ordered
    - Use the annotation `colors` to color the histogram
    - Add NA as a category to end of histogram if any missing values present (color it `#cc4560`)
- reorder categories dialog button via `streamlit_sortables`,
- Ensure `streamlit_sortables` dependency is listed in `pyproject.toml` extras (or dev requirements).
- `ordered`, `likert` toggles, `nonordered` multi-select - all on the same row as 'reorder'
- Category editor in a table format
  - With the following columns:
    - Original value before `translate`
    - Count of that original value
      - Either in total or separately for each file (column per file) based on "Separate" toggle
        - If separately, header of this column should be `file_code` for that file
    - Text_input for what the value should `translate` to
     - Treat empty string as a translation to `None`
    - Numerical value for that translated category (`num_values`)
      - default empty, showing consecutive numbers in translated category order as placeholder
    - Color picker for the color of that category
      - default is pure white `#FFFFFF` which is translated to `None` in `master_meta`
  - Group the categories that translate to the same value together in ordering
    - Alphabetic ordering on original names inside the groups
    - Groups ordered according to category order given for the column
    - Numerical value and color picker shown only for the first row of the group
  - Show categories from all input files
    - If some category not present, show a 0 on a yellow background
    - Also show NA as a table row with counts if there are any missing values in that column
- Button to open JSON metadata edit dialog (validated against `ColumnMeta`)
- "Remove column" button

**Constants:**
- list, edit, delete, or add constants via JSON text areas, with validation through `json.loads` wrapped in try/except (or create `parse_json_safe` helper if needed for reusable error handling).

**Files:**
- show linked input files, allow code renaming, read option editing, removal, and auto-detection of additional data files alongside the metafile.
- Dialog workflows: modal helpers for JSON edit, metadata edit, and category reorder handling session state keys, rerun logic, and validation feedback.
- Multi-file translation editor: when split-by-file mode is active, show per-file raw-value counts in the translation/color table to highlight discrepancies between sources.
- Full undo/redo functionality on all edits to the annotations
- Ability to save the current state of annotations as a json via a textbox for file name and a save button
  - Prompt for output path; default to the original file path. When overwriting, create a backup `{original_name}.orig.{i}.json`, incrementing `i` each save.

**Architecture:**

- Keep a full master copy of the annotation json in `session_state` as `st.session_state['master_meta']`
  - Create a wrapper that wraps main streamlit controls (selectbox, multiselect, checkbox, text_input) so that it can be given a `path/in/meta/structure` and make it always load the session state value from `master_meta` on display and write it back into it on change to ensure the current master state is properly displayed and updated.
  - On each run, compare the end state with the start state and if they differ, create a new undo point by snapshotting the full `master_meta`. No diffing or history cap is required beyond the Streamlit session.
- Use `st.dialog` for edits that might override/change/conflict with regular editing
  - Create a dialog that just allows raw editing of a part of the annotations json as text. Validate on save before updating `master_meta`
  - Other cases where this is needed: reordering categories;
- Create a separate `st.fragment` for each column block so blocks with lots of columns would still be responsive
- Make sure the editor for block `scale` reuses the code for editing a column, i.e. they are both the same function
- By default, change state when input changes without further confirmation
  - Dialog modals are the exception here - they should have a save/cancel button
- Avoid code duplication with the functionality of io.ipynb module
  - Instead, whenever possible, create a helper function inside `01_io.ipynb` that is used both there and in 
  the new tool being built
  - Don't go overboard - 1 line things do not need a helper function, but 5+ line logic blocks probably do. 
- Always make sure the editor UI and the values created by it are in alignment with `DataMeta` model. If there seems to be a mismatch, clarify it explicitly. 
- Handling constants: when editing part of the meta that originates from a constant, edit that constant value instead of the local copy. Use a helper function `infer_constant_source(meta: dict, path: str) -> str | None` (add to `01_io`) that takes the unresolved meta and a JSONPath-like path (e.g., `'structure['issues'].scale.categories'`) and returns the constant key name if that field was originally a constant reference, or None if not. This requires storing original constant references before resolution or reverse-engineering from the resolved value.
- Be mindful of not creating infinite reload loops in streamlit

## Implementation Plan

### Foundation Setup

- [ ] Add helper utilities to `salk_toolkit/io.py` (no nbdev export step is needed anymore).
  - [ ] `load_annotation_meta(meta_path: str) -> dict`: wrap file-map lookup, JSON/YAML parsing, and `soft_validate(meta)` (takes single arg). Returns unresolved meta (with constants still present).
  - [ ] `resolve_annotation_constants(meta: dict) -> tuple[dict, dict]`: extract constants dict, apply `replace_constants`, return `(resolved_meta, constants_dict)`.
  - [ ] `load_annotation_inputs(meta: dict, meta_path: str | None, data_file: str | None, raw_data: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]`: wrap `read_concatenate_files_list`, warnings, return `(raw_df, einfo)`. Note: expects unresolved meta.
  - [ ] `build_block_column_index(meta: dict) -> dict`: derive block ↔ column mappings. Returns dict with `'block_to_columns'`, `'column_to_block'`, `'duplicate_columns'` keys. Works on resolved meta.
  - [ ] `infer_constant_source(meta: dict, path: str) -> str | None`: helper to determine if a field path came from a constant. Takes unresolved meta and path (e.g., `'structure['issues'].scale.categories'`), returns constant key or None.
- [ ] Declare any new dependencies in `pyproject.toml` (if not already present).
  - [ ] Verify `streamlit-sortables` is in requirements (already present as of review).
  - [ ] Re-run nbdev export / dependency sync so generated modules pick up any new requirements.

### Core Development

- [ ] Implement `json_block_editor.py` Streamlit entry and core infrastructure.
  - [ ] Parse CLI meta path argument using `sys.argv`
  - [ ] Load unresolved meta via `load_annotation_meta`, store in `st.session_state['original_meta']`.
  - [ ] Resolve constants via `resolve_annotation_constants`, store resolved meta in `st.session_state['master_meta']` and constants in `st.session_state['constants']`.
  - [ ] Load raw data via `load_annotation_inputs` (using unresolved meta for file paths), store in `st.session_state['raw_data']` and `st.session_state['einfo']`.
  - [ ] Initialize undo/redo stacks: `st.session_state['undo_stack'] = []`, `st.session_state['redo_stack'] = []`, snapshot initial state.
  - [ ] Create widget binding helper function that reads from/writes to `master_meta` at given paths (e.g., `bound_selectbox(path, options, ...)`).
  - [ ] Wire sidebar selectors (mode selectbox, block selector, Separate toggle) and global alerts/validation messages.
- [ ] Implement undo/redo system.
  - [ ] Create snapshot function that deep-copies `master_meta` and appends to undo stack.
  - [ ] Create undo/redo handlers that restore snapshots and manage stack state.
  - [ ] Hook snapshot creation into all edit operations (widget bindings should trigger snapshots on change).
  - [ ] Add undo/redo buttons to sidebar UI.
- [ ] Build Blocks mode UI/logic.
  - [ ] Block selector: use `build_block_column_index` to populate selectbox with block names.
  - [ ] Block rename: text input that updates block name in `master_meta['structure']` with validation.
  - [ ] Add column workflow: selectbox (all columns from raw_data), text_input for name, "Add" button.
  - [ ] Split block workflow: text_input for new block name, multiselect for columns to split, "Split" button.
  - [ ] Column fragment factory using `st.fragment` per column (and one for block scale pseudo-column).
  - [ ] Shared column editor function (reused for both columns and block scale):
    - [ ] Expander header with label and original column names, warning icons for duplicate usage.
    - [ ] Rename column text_input.
    - [ ] Warning display showing other blocks using same column.
    - [ ] Visualization: conditional on column type (continuous vs categorical).
      - [ ] Continuous: density plot (Altair), colored by `file_code` if Separate toggle, stats table.
      - [ ] Categorical: horizontal histogram (Altair), yOffset by `file_code` if Separate, use annotation colors, add NA category if present.
    - [ ] Reorder categories button opening `st.dialog` with `streamlit_sortables`.
    - [ ] `ordered`, `likert` toggles and `nonordered` multiselect on same row.
    - [ ] Category editor table: original values, counts (per-file if Separate), translation inputs, numeric values, color pickers, grouped by translated value.
    - [ ] JSON metadata edit dialog (validated against `ColumnMeta`).
    - [ ] Remove column button.
- [ ] Implement Constants mode.
  - [ ] Display constants from `st.session_state['constants']` as editable JSON textareas.
  - [ ] Validate JSON on edit using `json.loads` with try/except (or create `parse_json_safe` helper).
  - [ ] Use `infer_constant_source` to detect when editing a field that originates from a constant, redirect edits to constant value in widget wrappers.
  - [ ] After constant edit, re-resolve `master_meta` via `resolve_annotation_constants` and update state.
  - [ ] Add constant button: dialog with JSON textarea for new constant entry.
  - [ ] Delete constant button: confirmation dialog, then remove from constants dict and re-resolve.
- [ ] Implement Files mode.
  - [ ] Display files from `master_meta['files']` (or `master_meta['file']` converted to list format) with file codes, paths, read options.
  - [ ] Allow renaming file codes, editing read options (as JSON), removing files.
  - [ ] Auto-detection: scan directory of metafile for additional data files, offer to add them.
  - [ ] When Separate toggle active in Blocks mode, show per-file translation/colour tables highlighting discrepancies.
- [ ] Implement save + backup flow.
  - [ ] Save section in sidebar: text_input for output path (default to original file path).
  - [ ] Save button: reconstruct unresolved meta by merging `constants` back into `master_meta` (or use a helper function to invert `replace_constants`), serialize to JSON, write to file.
    - Note: If constants were edited, need to update references in `master_meta` structure to point to updated constant values. May need helper to detect constant-originated fields and restore them as constant references rather than resolved values.
  - [ ] Backup logic: if overwriting original, create `{original_name}.orig.{i}.json` backup, incrementing `i` until unique.
  - [ ] Success/error feedback after save.

### Integration & Testing

- [ ] Write unit tests for new helper functions in `tests/test_io.py` (or create new test file).
  - [ ] Test `load_annotation_meta` with JSON and YAML files, file-map lookup, validation errors.
  - [ ] Test `resolve_annotation_constants` with various constant structures.
  - [ ] Test `load_annotation_inputs` with single and multiple files.
  - [ ] Test `build_block_column_index` returns correct structure and detects duplicates.
  - [ ] Test `infer_constant_source` with various paths and constant references.
- [ ] Manual QA of Streamlit app against representative annotation files.
  - [ ] Test with single-file and multi-file annotations.
  - [ ] Test Blocks mode: rename, split, add columns, edit categories, reorder.
  - [ ] Test Constants mode: edit, add, delete constants, verify propagation.
  - [ ] Test Files mode: rename codes, edit options, add/remove files.
  - [ ] Test undo/redo across all operations.
  - [ ] Test save/backup functionality.
- [ ] Smoke-test CLI invocation: `streamlit run salk_toolkit/tools/json_block_editor.py <meta_path>`.
  - [ ] Verify helpers load correctly.
  - [ ] Verify Streamlit UI initializes without errors.
  - [ ] Verify all modes are accessible and responsive.

