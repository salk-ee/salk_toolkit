# TOOL-002: Annotations Editor

**Last Updated**: 2025-12-18
**Status**: ✅ Complete
**Module**: Tool
**Tags**: `#tool`, `#streamlit`, `#data-annotation`
**Dependencies**: None

## Overview

Streamlit-based utility for inspecting and editing data annotation JSON metafiles used by the data ingestion pipeline. Provides block-level editing, category management, and persistence so analysts can adjust annotations without hand-editing JSON.

## Problem Context

- Problem being solved: Manual JSON edits for block metadata are error prone and lack immediate feedback on survey data structure.
- Intended use cases:
  - Data analysts iteratively improving data annotations
  - Adding a new wave of data in a multi-wave survey where coding might change from wave to wave
- Technical constraints and requirements:
  - Must validate annotations against `DataMeta`
  - Reuse existing IO logic where possible but implement custom processing for the editor's specific needs (per-file visualization, pre-translation state).
  - Operate within Streamlit’s session state model.
- Integration points with existing systems:
  - Consumes annotations defined in `salk_toolkit/io.py`
  - Relies on validation models from `salk_toolkit/validation.py`

## Requirements

**Relevant rules files and documentation**
- @salk_toolkit.mdc for whole-project rules
- @tools.mdc for tool-specific guidelines
- @data_annotations.mdc for annotation structure and logic

**Important context functions/files**
- `salk_toolkit/validation.py`: The pydantic model of annotation files is defined here as `DataMeta`. This structure should not be altered without explicit confirmation. `soft_validate` is key for state restoration.
- `salk_toolkit/io.py`: Contains `_load_data_files` which should be used to load raw data.

**Files to Create/Modify:**
- `salk_toolkit/tools/annotator.py`: Main Streamlit app orchestrating all functionality. Contains the custom data processing logic.
- `salk_toolkit/commands.py`: Register CLI entry point.
- `salk_toolkit/validation.py`: Update `soft_validate` to accept `context` for Pydantic validation (to track constant usage).

**Shared:**

- **File loading**:
  - Load the annotation file as JSON.
  - Load input data files using `salk_toolkit.io._load_data_files` (accessing the internal function).
- **Data Processing**:
  - Implement a custom processing pipeline in `annotator.py` that mirrors `_process_annotated_data` but with key differences:
    - **Per-file processing only**: Do not aggregate into a single DataFrame.
    - **Skip translation**: The editor needs to see original values to map them. Skip `translate` and `translate_after` steps.
    - **Applies**: `preprocessing`, `transform` (per-file), and `subgroup_transform`.
  - **Caching**: The resulting dictionary of per-file DataFrames should be cached.
    - Recompute ONLY if: loaded file codes change, `preprocessing` changes, or column `transform` scripts change.
    - Do NOT recompute if: categories, translations, or other metadata changes. This allows fast UI updates when editing mappings.
- **State Management**:
  - **Undo/Redo**: The history stack should store the *serialized* `DataMeta` (result of `model_dump()`).
  - **Current State**: On restore (undo/redo), use `soft_validate(snapshot, DataMeta)` to reconstitute the Pydantic object.
  - **Constants**: `soft_validate` must support a context to track constant usage during validation.
  - **Input Wrapping**:
    - All Streamlit inputs must be wrapped using a helper function `wrap` to ensure instantaneous updates to `DataMeta` and automatic history tracking.

**Functionality (as implemented)**

- **CLI**:
  - `stk_annotator <path_to_meta.json>` runs the Streamlit app.
- **Modes**:
  - Sidebar `Mode` radio: `Blocks`, `Constants`, `Files`.
  - Sidebar includes Undo/Redo buttons, plus `Save Changes` (writes the annotation JSON back to disk, with backup).
- **Blocks mode**:
  - Block selection via `selectbox` in the main pane.
  - Column selection via `selectbox` in the main pane.
    - Includes a pseudo-column `scale` when `block.scale` exists.
  - `Separate files` toggle lives in the sidebar, is shown only when multiple files exist, and is off by default.
  - Block actions:
    - Rename block (renames the `structure` dict key).
    - Add column: choose a raw column name (from all loaded raw files), optionally rename, and create a new `ColumnMeta(source=<raw_col>)`.
    - Split block: move selected columns to a new `ColumnBlockMeta`.
- **Column editor**:
  - Column-level fields:
    - `label` (`text_input`)
    - `ordered`, `likert` (`checkbox`)
    - `nonordered` (`multiselect`) from the current categories list
  - Visualisations (Altair) computed from cached per-file processed data:
    - Continuous: density plot + summary table (`separate` optionally groups by `file_code`)
    - Datetime: binned count plot + min/max table
    - Categorical: histogram of **translated display values** (based on `translate` and `categories` ordering), optionally split by `file_code`
  - Category mapping table:
    - Lists original observed values and counts (total or per-file).
    - Provides per-original `translate` text input:
      - Empty string means “translate to NA” (stored as `""`).
      - Leaving translation equal to the original value is treated as “no explicit translate entry” (removes key).
    - Updates `categories` ordering when translations change to preserve order where possible.
    - Provides `num_values` editor for ordered categoricals (indexed by translated category order).
    - Provides `colors` editor via `color_picker` for translated categories; picking pure white `#FFFFFF` stores `None`.
    - Includes NA row counts when missing values exist.
  - Reorder categories dialog via `streamlit_sortables`.
- **Constants mode**:
  - A single JSON text area editor for `constants` (`json.loads` validation).
  - Note: changing constants updates `constants` only; it does not re-resolve constant references already materialised in the current `DataMeta` instance.
- **Files mode**:
  - Lists `files`; supports editing per-file `code`.
  - Displays `opts` (read-only).
- **Save**:
  - `Save Changes` writes to the original `meta_path`.
  - When overwriting, creates a backup `{original_name}.orig.{i}.json`.
  - Writes `master_meta.model_dump(mode="json", exclude_none=True)` (no separate hard-validate step).

**Architecture:**

- **State**:
  - `st.session_state['master_meta']`: The live `DataMeta` Pydantic object.
  - `st.session_state['history']`: List of serialized (JSON-compatible dict) snapshots of `DataMeta`.
  - `st.session_state['data_cache']`: Tuple of `(hash_of_input_params, dict[file_code, DataFrame])`.
- **Data Loading & Custom Pipeline**:
  - Use `salk_toolkit.io._load_data_files` to get raw dataframes.
  - Implement `_process_for_editor(raw_data_dict, meta)` in `annotator.py`:
    - Iterates over `meta.structure`.
    - Runs `preprocessing` on raw data.
    - Runs `transform` (column-level) and `subgroup_transform`.
    - **Crucially**: Does NOT run `translate` or `translate_after`. It keeps original values.
    - Does NOT merge into a single dataframe.
- **Components**:
  - Use `st.dialog` for raw JSON edits and category reordering.
  - Use `st.fragment` for column blocks to ensure responsiveness.
  - **Input Wrapper**:
    - `wrap(inp, *args, path: str | list[str], **kwargs)` helper function:
      - Wraps Streamlit input widgets (e.g., `st.selectbox`, `st.text_input`).
      - Arguments:
        - `inp`: The Streamlit input function to call.
        - `*args`: Positional arguments for the input widget.
        - `path`: Dot-notation string (e.g., "structure.block.col.translate") pointing to the location in `master_meta`.
        - `**kwargs`: Keyword arguments for the input widget.
      - Responsibilities:
        - Reads the current value from `master_meta` at `path` to initialize the widget.
        - Updates `master_meta` at `path` immediately upon change.
        - Pushes the previous state to the undo history before modification.
- **Validation Context**:
  - Pass `context={'track_constants': True}` to `soft_validate`.
  - Validators write constant usage to this context to trace which fields are backed by constants.
- **CLI**:
  - Use `sys.argv` to get the meta path.

## Implementation Plan

### Foundation Setup

- [x] Update `salk_toolkit/validation.py`:
  - [x] Modify `soft_validate` to accept `context: dict[str, Any] | None = None`.
  - [x] Ensure validators in `DataMeta` (and children) respect context for tracing constants.
- [x] Declare dependencies in `pyproject.toml`.
  - [x] Add `streamlit-sortables`.
  - [x] Register script `stk_annotator` in `[project.scripts]`.

### Core Development

- [x] Implement `annotator.py` backbone.
  - [x] CLI arg parsing.
  - [x] State initialization (`master_meta`, `history`, `constants`).
  - [x] `wrap` helper function for inputs (auto-update `master_meta` & history).
  - [x] `_load_raw_data` helper using `io._load_data_files`.
  - [x] `_process_for_editor` helper (custom pipeline, no translate, cached).
- [x] Implement Undo/Redo.
  - [x] `save_state()`: serialize `master_meta` -> history.
  - [x] `restore_state()`: pop history -> `soft_validate` -> `master_meta`.
- [x] Build Blocks Mode.
  - [x] Mode switcher + navigation.
  - [x] Block Editor (Rename, Add, Split).
  - [x] Column editor (selectbox-driven, includes scale).
  - [x] Visualizations (uses cached processed data).
  - [x] Category table / mapping editor.
  - [x] Reorder Dialog.
- [x] Build Constants Mode.
  - [x] JSON editor for constants.
- [x] Build Files Mode.
  - [x] List files and allow code editing.
- [x] Implement Save.
  - [x] Backup, Write.

### Integration & Testing

- [x] Write unit tests for new validation context logic in `tests/test_io.py` (or `test_validation.py`).
- [ ] Manual QA (recommended):
  - [ ] Verify editor works on a real annotation+data pair.
  - [ ] Verify undo/redo for wrapped widget edits.
  - [ ] Verify save produces a valid JSON file and a backup when overwriting.

## Definition of Done

- [x] `soft_validate` supports validation context.
- [x] `annotator.py` implements custom non-aggregating, non-translating pipeline.
- [x] Editor UI correctly maps original values to categories.
- [x] Tool is registered and runnable.
- [x] Undo/Redo relies on serialized state and `soft_validate`.
- [x] Saving produces valid JSON.

## Implementation Notes

**Future enhancements (not implemented)**

- Column rename that updates `structure[block].columns` keys
- Cross-block “column used elsewhere” warnings
- “Edit raw ColumnMeta JSON” dialog for a column
- Remove column action
- Missing-category highlighting (e.g. yellow 0s) across files
- Constants propagation by re-validating `master_meta` after edits
- Files editing: add/remove entries, edit `opts`, auto-detect data files next to the meta file
- Save-as prompt and/or explicit `hard_validate` before writing
