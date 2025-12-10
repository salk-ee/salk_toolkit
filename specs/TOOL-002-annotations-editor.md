# TOOL-002: Annotations Editor

**Last Updated**: 2025-12-10
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
  - Load the annotation file using standard JSON/YAML loaders.
  - Load input data files using `salk_toolkit.io._load_data_files` (accessing the internal function).
- **Data Processing**:
  - Implement a custom processing pipeline in `annotator.py` that mirrors `_process_annotated_data` but with key differences:
    - **Per-file processing only**: Do not aggregate into a single DataFrame.
    - **Skip translation**: The editor needs to see original values to map them. Skip `translate` and `translate_after` steps.
    - **Applies**: `preprocessing`, `transform` (per-file), and `subgroup_transform`.
  - **Caching**: The resulting dictionary of per-file DataFrames should be cached.
    - Recompute ONLY if: files list changes, `preprocessing` script changes, or `transform` scripts change.
    - Do NOT recompute if: categories, translations, or other metadata changes. This allows fast UI updates when editing mappings.
- **State Management**:
  - **Undo/Redo**: The history stack should store the *serialized* `DataMeta` (result of `model_dump()`).
  - **Current State**: On restore (undo/redo), use `soft_validate(snapshot, DataMeta)` to reconstitute the Pydantic object.
  - **Constants**: `soft_validate` must support a context to track constant usage during validation.
  - **Input Wrapping**:
    - All Streamlit inputs must be wrapped using a helper function `wrap` to ensure instantaneous updates to `DataMeta` and automatic history tracking.

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
  - It is acceptable to scan the entire `master_meta` each render to determine other-block usage.
- rename column text_input
- warning with list of other blocks this column is in
- Visualization - visualize the values of the column (using the cached pre-translation data)
  - All plots use Altair.
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
  - Ensure `streamlit-sortables` dependency is listed in `pyproject.toml`.
- `ordered`, `likert` toggles, `nonordered` multi-select - all on the same row as 'reorder'
- Category editor in a table format
  - With the following columns:
    - Original value (from cached data, before `translate`)
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
- list, edit, delete, or add constants via JSON text areas, with validation through `json.loads`.
- Updates to constants require re-validation of the `DataMeta` object to propagate changes.

**Files:**
- show linked input files, allow code renaming, read option editing, removal, and auto-detection of additional data files alongside the metafile.
- Multi-file translation editor: when split-by-file mode is active (in Blocks mode), show per-file raw-value counts.
- Full undo/redo functionality on all edits.
- Ability to save the current state of annotations as a json.
  - Prompt for output path; default to the original file path. When overwriting, create a backup `{original_name}.orig.{i}.json`.
  - Validate against `DataMeta` before saving.

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

- [ ] Update `salk_toolkit/validation.py`:
  - [ ] Modify `soft_validate` to accept `context: dict[str, Any] | None = None`.
  - [ ] Ensure validators in `DataMeta` (and children) respect context for tracing constants.
- [ ] Declare dependencies in `pyproject.toml`.
  - [ ] Add `streamlit-sortables`.
  - [ ] Register script `stk_annotator` in `[project.scripts]`.

### Core Development

- [ ] Implement `annotator.py` backbone.
  - [ ] CLI arg parsing.
  - [ ] State initialization (`master_meta`, `history`, `constants`).
  - [ ] `wrap` helper function for inputs (auto-update `master_meta` & history).
  - [ ] `_load_raw_data` helper using `io._load_data_files`.
  - [ ] `_process_for_editor` helper (custom pipeline, no translate, cached).
- [ ] Implement Undo/Redo.
  - [ ] `save_state()`: serialize `master_meta` -> history.
  - [ ] `restore_state()`: pop history -> `soft_validate` -> `master_meta`.
- [ ] Build Blocks Mode.
  - [ ] Sidebar (Block selector, Separate toggle).
  - [ ] Block Editor (Rename, Add, Split).
  - [ ] Column Editor Fragment.
    - [ ] Visualizations (using cached pre-translate data).
    - [ ] Category Table (mapping original -> translated).
    - [ ] Reorder Dialog.
- [ ] Build Constants Mode.
  - [ ] JSON editors for constants.
  - [ ] Update logic (edit constant -> re-validate `master_meta`).
- [ ] Build Files Mode.
  - [ ] File management UI.
- [ ] Implement Save.
  - [ ] Validate, Backup, Write.

### Integration & Testing

- [ ] Write unit tests for new validation context logic in `tests/test_io.py` (or `test_validation.py`).
- [ ] Manual QA:
  - [ ] Verify editor works with pre-translation values.
  - [ ] Verify changing a constant updates the resolved value in the UI.
  - [ ] Verify undo/redo works across all modes.
- [ ] Smoke test CLI.

## Definition of Done

- [ ] `soft_validate` supports validation context.
- [ ] `annotator.py` implements custom non-aggregating, non-translating pipeline.
- [ ] Editor UI correctly maps original values to categories.
- [ ] Tool is registered and runnable.
- [ ] Undo/Redo relies on serialized state and `soft_validate`.
- [ ] Saving produces valid JSON.

## Implementation Notes

Track decisions and pattern deviations as you work.
