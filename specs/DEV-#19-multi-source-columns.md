# DEV-#19: Multi-source columns and per-file processing

**Last Updated**: 2025-01-27
**Status**: ✅ Complete
**Module**: Annotations
**Tags**: `#feature`, `#io`, `#multi-file`
**Dependencies**: None

## Overview

Enable processing of multiple data files with per-file column mappings. Allow `ColumnMeta.source` to map different column names from different files to a single output column, and ensure preprocessing/transforms run with correct per-file context.

## Problem Context

- **Problem**: Currently, `ColumnMeta.source` only supports a single string, requiring identical column names across all files. Multi-wave surveys or merged datasets often have different column names for the same concept across files.
- **Intended use cases**:
  - Multi-wave surveys where column names differ between waves
  - Merged datasets from different sources with different naming conventions
  - Per-file preprocessing that needs access to individual file context
- **Technical constraints**:
  - Must maintain backward compatibility with single-file inputs
  - `preprocessing` and `transform` functions need access to per-file dataframes
  - Category inference must work across all files before applying back to individual files
- **Integration points**:
  - `ColumnMeta.source` in `validation.py`
  - `DataMeta` normalization in `validation.py`
  - `_load_data_files` and `_process_annotated_data` in `io.py`

## Requirements

**Important context functions/files:**
- `salk_toolkit/validation.py`: `ColumnMeta`, `DataMeta`, `FileDesc`
- `salk_toolkit/io.py`: `_load_data_files`, `_read_concatenate_files_list`, `_read_files_from_description`, `_process_annotated_data`, `read_and_process_data`

**Files to Create/Modify:**
- `salk_toolkit/validation.py`: Update `ColumnMeta.source` type, add `DataMeta` validator
- `salk_toolkit/io.py`: Refactor `_load_data_files` and `_process_annotated_data` for dict-based processing

**Functionality:**
- `ColumnMeta.source`: Support `str | dict[str, str] | None`
  - String: Applies to all files (backward compatible)
  - Dict: Maps file code (`F0`, `wave1`) to column name in that file
    - `default` key holds the default column name in cases it only changes for very few input files
  - Missing code in dict = missing values for that file
- `DataMeta` normalization:
  - Convert `file` + `read_opts` → `files` list via `model_validator(mode='before')`
  - Ensure all `FileDesc` entries have a `code` (default `F{i}`)
- Processing logic:
  - `_load_data_files` returns `(dict[str, pd.DataFrame], DataMeta | None, dict[str, object])` keyed by file code
  - `_read_concatenate_files_list` and `_read_files_from_description` deprecated
  - `_process_annotated_data` handles `raw_data` as `dict[str, pd.DataFrame]`
  - Per-file context:
    - `preprocessing`: runs per file with `df` = file's raw data, `file_code` = code
    - `transform`: runs per file by slicing concatenated series using index ranges, with `df` = file's raw data, `ndf` = view of concatenated processed data for that file's range
  - Column extraction and processing:
    - Extract per-file series based on `source` (str/dict), concatenate immediately
    - Apply `translate` on concatenated series
    - Apply `transform` per-file using index ranges (slicing concatenated series)
    - Apply `translate_after` on concatenated series
    - Category inference runs on concatenated data
    - Add column directly to single concatenated `ndf_df` DataFrame
  - Category reconciliation:
    - Extract category merging logic from `_load_data_files` into `_reconcile_categories` helper
    - Merge categorical dtypes across files, preserving all categories
    - Applied in `read_and_process_data` before concatenation

**Architecture:**
- Maintain backward compatibility: single-file inputs internally treated as 1-file dict
- Separate category reconciliation logic from file loading (`_reconcile_categories` helper)
- Single concatenated `ndf_df` DataFrame built column-by-column (not per-file DataFrames)
- Index ranges computed upfront to enable per-file slicing of concatenated data during transforms

## Implementation Plan

### Foundation Setup

- [x] Update `ColumnMeta.source` type to `str | dict[str, str] | None` in `salk_toolkit/validation.py`
- [x] Add `model_validator(mode='before')` to `DataMeta`:
  - [x] Normalize `file` + `read_opts` → `files` list
  - [x] Ensure `files` is populated even if `file` was used
  - [x] Deprecate `FileDescriptionProtocol` as we can now assume `list[FileDesc]`
- [x] Ensure `FileDesc` always has a `code` via validator (default `F0`, `F1`...)

### Core Development

- [x] Create `_reconcile_categories` helper in `salk_toolkit/io.py`:
  - [x] Extract category merging logic from current `_load_data_files`
  - [x] Takes `dict[str, pd.DataFrame]` and `dict[str, pd.CategoricalDtype | None]`
  - [x] Returns dict with reconciled categorical dtypes
- [x] Refactor `_load_data_files` in `salk_toolkit/io.py`:
  - [x] Return `(dict[str, pd.DataFrame], DataMeta | None, dict[str, object])` keyed by file code
  - [x] Keep per-file dataframes separate (no concatenation)
  - [x] Track categorical dtypes per file for later reconciliation
- [x] Deprecate `_read_concatenate_files_list` and `_read_files_from_description`
- [x] Update `read_and_process_data`:
  - [x] Handle dict return from `_load_data_files`
  - [x] Run `_reconcile_categories` and apply to per-file dataframes
  - [x] Return concatenated result
- [x] Refactor `_process_annotated_data`:
  - [x] Accept `raw_data` as `dict[str, pd.DataFrame]` from `_load_data_files`
  - [x] Initialize single concatenated `ndf_df` DataFrame (built column-by-column)
  - [x] Compute and store index ranges for each file_code upfront
  - [x] Update `preprocessing` loop to iterate over `raw_data` dict items
  - [x] Update column processing loop:
    - [x] Extract per-file series based on `source` (str/dict), concatenate immediately
    - [x] Apply `translate` on concatenated series
    - [x] Apply `transform` per-file using index ranges (slicing concatenated series, context: `df`=file_raw, `ndf`=sliced view of `ndf_df`)
    - [x] Apply `translate_after` on concatenated series
    - [x] Run category inference on concatenated series
    - [x] Add column directly to `ndf_df`
  - [x] Move `_create_new_columns_and_metas` to run on `ndf_df`
  - [x] Run `postprocessing` on `ndf_df`
  - [x] Update `subgroup_transform` to use index ranges for per-file processing

### Integration & Testing

- [x] Create test in `tests/test_io.py`:
  - [x] Test with 3 CSV files with different column names
  - [x] Use `source` dict to map different column names from each file to single output column
  - [x] Test missing file code in dict (should result in missing values)
  - [x] Test missing column in file when `source` is str (should result in missing values)
  - [x] Verify `preprocessing` ran per file (e.g., add flag column)
  - [x] Verify `transform` sees correct per-file context (check `df` shape/content)
  - [x] Verify category reconciliation works across files
  - [x] Verify final concatenation is correct
  - [x] Verify backward compatibility with single-file inputs

## Definition of Done

- [x] All files created/updated
- [x] `pytest tests/test_io.py` passes
- [x] `read_and_process_data` handles `source` dicts correctly
- [x] Preprocessing and transforms have access to correct per-file context
- [x] Backward compatibility maintained for single-file inputs
- [x] Category inference works correctly across files
- [x] Follows established patterns in `io.py`

## Implementation Notes

- Updated `ColumnMeta.source` to support `str | dict[str, str] | None` for per-file column mapping
- Added `model_validator` to `DataMeta` to normalize `file` + `read_opts` → `files` list automatically
- Added validator to `FileDesc` to ensure `code` is always set (defaults to "F0" for single files)
- Created `_reconcile_categories` helper to merge categorical dtypes across files
- Refactored `_load_data_files` to return `dict[str, pd.DataFrame]` keyed by file code, keeping files separate
- Updated `_process_annotated_data` for per-file processing:
  - Preprocessing runs per file with `file_code` in context
  - Column processing: extract per-file series, concatenate immediately, process on concatenated data
  - Transforms run per-file by slicing concatenated series using precomputed index ranges
  - Transform context: `df` = per-file raw data, `ndf` = sliced view of concatenated `ndf_df` for that file
  - Category inference runs on concatenated data
  - Single `ndf_df` DataFrame built column-by-column (not separate per-file DataFrames)
  - Subgroup transforms also use index ranges for per-file processing
- Preserved backward compatibility: single-file inputs work as before
- Added comprehensive tests covering all scenarios
