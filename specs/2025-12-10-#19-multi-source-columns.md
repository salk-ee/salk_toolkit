# Multi-source columns and per-file processing (PR #19)

**Modules:** `salk_toolkit/io.py`, `salk_toolkit/validation.py`

## Goal

`ColumnMeta.source` accepted only a single string, so every input file had to name a column
identically — which multi-wave surveys and merged datasets never do. This PR lets one output
column draw from differently-named source columns per file, and reworks the processing loop
so preprocessing and transforms see the correct per-file context instead of whichever frame
happened to be in scope.

## Design

**`source: str | dict[str, str] | None`.** A string applies to all files (unchanged
behaviour). A dict maps file code (`F0`, `wave1`, …) to the column name in that file, with an
optional `default` key for the common case where only a few files deviate. A code absent from
the dict, or a named column missing from a file, yields missing values for that file's rows
with a warning rather than an error.

**File codes are guaranteed.** A `model_validator(mode="before")` on both `DataMeta` and
`DataDescription` runs `_normalize_files_dict`, which folds the legacy `file` + `read_opts`
pair into a `files` list and assigns each entry a `code` (`F0`, `F1`, …) if it has none.
Downstream code can therefore assume `list[FileDesc]` with populated codes, and the
`FileDescriptionProtocol` indirection is retired.

**Loading keeps files apart.** `_load_data_files` returns
`(dict[str, pd.DataFrame], DataMeta | None, dict[str, object])` keyed by file code — no
concatenation — tracking each frame's categorical dtypes.
`_read_concatenate_files_list` and `_read_files_from_description` are deleted.
`_reconcile_categories` is factored out of the loader as its own helper: it merges the
categorical dtypes across frames so a category present in only one file survives, and
`read_and_process_data` applies it to the per-file frames before concatenating.

**Processing builds one frame column by column.** `_process_annotated_data` takes the dict of
raw frames, computes `file_index_ranges: dict[str, slice]` upfront, and grows a single
`ndf_df`:

- `preprocessing` runs once per file, with `df` = that file's raw frame and `file_code` in
  scope.
- For each column: gather the per-file series according to `source` and concatenate
  immediately → `translate` on the concatenated series → `transform` per file, slicing the
  concatenated series by its range (`df` = that file's raw frame, `ndf` = the matching slice
  of the frame built so far) → `translate_after` → category inference → append to `ndf_df`.
- `subgroup_transform` uses the same ranges; `_create_new_columns_and_metas` and
  `postprocessing` run once on the finished `ndf_df`.

## Implementation notes

- Category inference runs on the *concatenated* series, so a column can never end up with
  per-file category sets that disagree — which is what made cross-wave merges silently lossy
  before.
- Translation is applied to the concatenated series but transforms are applied per file: a
  translation dict is a property of the column, whereas a transform routinely needs the
  file's own raw frame (a wave-specific recode, a per-file scale flip).
- A single-file input is internally just a one-entry dict, so there is no separate legacy
  path to keep in sync.
