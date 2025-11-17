# TOOL-002: JSON Block Editor

**Last Updated**: 2025-11-13
**Status**: 🚧 Planning
**Module**: Tool
**Tags**: `#tool`, `#streamlit`, `#data-annotation`
**Dependencies**: nbs/01_io.ipynb, nbs/06_validation.ipynb

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
  - Consumes annotations defined in `nbs/01_io.ipynb`
  - Relies on validation models from `nbs/06_validation.ipynb`

## Requirements

**Important context functions/files**
- `nbs/06_validation.ipynb`: The pydantic model of annotation files is defined here as `DataMeta`. This structure should not be altered without explicit confirmation, as this tool is intended as a UI for editing this format. 

**Files to Create/Modify:**
- `salk_toolkit/tools/json_block_editor.py`: Main Streamlit app orchestrating all functionality.
- `nbs/01_io.ipynb`: abstract out helper functions from inside `process_annotated_data` to use in the tool as needed. 

**Shared:**

- File loading: open JSON, run `soft_validate(meta, DataMeta)`, load the annotated dataset via `process_annotated_data(..., return_meta=True, return_raw=True)` to supply both metadata and data frame for visualisations.
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
- rename column text_input
- warning with list of other blocks this column is in
- Visualization - visualize the values of the column after transform and translate
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
- list, edit, delete, or add constants via JSON text areas, with validation through `parse_json_safe`.

**Files:**
- show linked input files, allow code renaming, read option editing, removal, and auto-detection of additional data files alongside the metafile.
- Dialog workflows: modal helpers for JSON edit, metadata edit, and category reorder handling session state keys, rerun logic, and validation feedback.
- Multi-file translation editor: when split-by-file mode is active, show per-file raw-value counts in the translation/color table to highlight discrepancies between sources.
- Full undo/redo functionality on all edits to the annotations
- Ability to save the current state of annotations as a json via a textbox for file name and a save button
  - If overwriting original file, first create `{original_name}.orig.{i}.json` to back up the starting state

**Architecture:**

- Keep a full master copy of the annotation json in `session_state` as `st.session_state['master_meta']`
  - Create a wrapper that wraps main streamlit controls (selectbox, multiselect, checkbox, text_input) so that it can be given a `path/in/meta/structure` and make it always load the session state value from `master_meta` on display and write it back into it on change to ensure the current master state is properly displayed and updated.
  - On each run, compare the end state with the start state and if they differ, create a new undo point
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
- Be mindful of not creating infinite reload loops in streamlit
- Always make sure the editor UI and the values created by it are in alignment with `DataMeta` model. If there seems to be a mismatch, clarify it explicitly. 
  
## Implementation Plan

### Foundation Setup

- [ ] Ensure Streamlit configuration (`st.set_page_config`) sets wide layout and title.
- [ ] Establish session state defaults for metadata, dataframe, file path, and initial load flag.

### Core Development

- [ ] Implement file loading pipeline (CLI arg handling, JSON parse, validation, `process_annotated_data`).
- [ ] Build sidebar modes (Blocks, Constants, Files) with associated handlers and shared save section.
- [ ] Develop block editor: column rendering, scale aggregation, column addition/removal, block splitting, metadata editing dialog.
- [ ] Implement category/translation/color management UI with modal reorder, constants integration, and charting.
- [ ] Add constants and files management panels mirroring metadata schema expectations.

### Integration & Testing

- [ ] Exercise tool against representative metafiles, confirming `soft_validate` warnings display and columns sync with processed data.
- [ ] Verify Altair charts render for categorical and continuous columns, including multi-file splits.
- [ ] Smoke-test saving workflow and ensure metadata modifications persist to disk and download artefact.

## Definition of Done

- [ ] json_block_editor reproduces current block editing feature set and navigation.
- [ ] Metadata changes update session state and persist on save/download without schema violations.
- [ ] Category reordering, translation, and colour edits update both inline metadata and referenced constants.
- [ ] Constants and file managers support add/edit/remove flows with validation feedback.
- [ ] Tool works with multi-file annotations, showing split statistics without errors.
- [ ] `soft_validate` errors surface to the user while allowing continued editing.

## Implementation Notes

- Assumes Altair is the plotting stack; the tool standardises colour handling by normalising named colours to hex (`color_to_hex`).
- Uses modal dialogs (`@st.dialog`) to isolate larger edits; reruns clear relevant session state keys to force widget refresh.
- Category inference respects translate dictionaries and supports `categories: "infer"` by querying the loaded dataframe.
- Split-by-file logic depends on `process_annotated_data` adding `file_code`; multi-file aggregation is handled with per-file charts and counts.
- Constants are resolved through `resolve_constant`/`update_metadata_field`, ensuring modifications propagate to the shared `constants` dictionary rather than duplicating inline values.

## Known Gaps

- `soft_validate` warnings currently only emit to server logs; UI surfacing is still required.
- The "Has scale" checkbox in the block header is not wired to create or remove `block['scale']`; either connect this control or adjust the interaction.
