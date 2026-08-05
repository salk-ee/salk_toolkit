# Stable row index and id-keyed exclusions (PR #60)

**Modules:** `salk_toolkit/io/` (`core.py`, `sources.py`, `pipeline.py`, `datasets.py`),
`salk_toolkit/validation.py`, `salk_toolkit/pp.py`, `.agents/skills/stk-data-annotations/`

## Goal

Every DataFrame the io layer returns carries a unique, deterministic index naming each row's
provenance, and meta exclusions are keyed on that identity. Previously row identity was a
RangeIndex rebuilt by `reset_index(drop=True)` at every concat, and `excluded` entries were
absolute integer positions into the post-concat frame — so any inner filter, file reorder, or
preprocessing row change silently re-aimed an outer meta's exclusions. With identity assigned
at load time, meta-within-meta exclusions compose to arbitrary depth: each level filters by
id, and a change in an inner filter can never move an outer one.

## Design

**Row id.** Each row gets a string id in the reserved `ROW_ID = "row_id"` column, assigned in
`_load_data_files` immediately after each file is read — before any concat, preprocessing, or
column processing can move rows:

- Raw file (csv/sav/xlsx/…): `{file_code}::{leaf}`, where `leaf` is the declared `id_col`
  value if set, else the 0-based row position within that file (`mint_positional_row_id`).
- Annotated inner dataset (json/yaml/parquet loaded as a `files` entry): rows already carry
  ids and the outer level prepends its own code — `wave1::37` stacked under code `F0` becomes
  `F0::wave1::37`. Variable nesting depth is why this is a delimited string, not a MultiIndex.
- Ids are opaque: nothing splits them, code only prepends prefixes and matches exactly, so a
  `::` inside an `id_col` value is harmless.

The id rides as a *column* through the positionally-indexed middle of the pipeline (the
`file_index_ranges`/`.iloc` machinery is untouched). At the return boundary of
`read_annotated_data` / `read_and_process_data`, `finalize_row_index` sets it as the index,
asserting uniqueness and non-nullness.

**`id_col` meta syntax.** `DataMeta.id_col` names a raw-data column that uniquely identifies
rows within each file (a respondent id); `FileDesc.id_col` overrides it per file. Validated at
load — the column must exist, be non-null and unique within the file, hard error otherwise.
Positional leaf ids are deterministic only for a byte-stable source file; a declared natural
key survives re-exports and reorders, which is why the annotation skill now actively looks for
one.

**Exclusions.** `DataMeta.excluded` becomes `List[Tuple[str, str]]` — `(row_id, reason)` —
applied at the same pipeline point as before via `~df[ROW_ID].isin(ids)`. An id matching
nothing emits a `warn` naming the unmatched ids but is not an error: an inner meta may
legitimately have filtered that row already, so no-match must not break the load — while a
typo'd id is otherwise undetectable, so it is surfaced. Legacy integer entries fail validation
with a message pointing here; no positional fallback is kept.

**Parquet round-trip.** `write_parquet_with_metadata` already preserves the index via
`pa.Table.from_pandas`. On read, a restored frame carrying a `row_id` index (or column) has it
lifted back into the column before the stacking concat would destroy it; frames without one
(raw parquet sources) get fresh positional ids like any other raw file.

**Merges.** `_perform_merges` runs after exclusions, so ids never need referencing post-merge
— only the output invariant matters. `_repair_merge_row_ids` keeps left ids where already
unique (many-to-one enrich); right-only rows (NaN id from `right`/`outer` joins) collapse to
the merge `tag`, and an id duplicated by a one-to-many or `cross` join gets a `::m{k}`
per-group suffix. One `fillna` + `duplicated` + `groupby().cumcount()` pass, deterministic
because merge output order is.

**Row-count discipline.** After each user-code hook (`preprocessing`, `postprocessing`,
frame-level `subgroup_transform`), `assert_row_id_intact` checks the column still exists and
is non-null and unique. Dropping or reordering rows is fine; rows *added* by user code have no
identity and fail loudly.

**Scope of the guarantee.** io-layer returns only. The pp bootstrap `_augment_draws`
duplicates rows by design and resets its own index; there the id is provenance ("which source
row generated this draw"), not a unique key.

**`original_inds` unchanged.** `add_original_inds=True` still produces the positional
post-processing `np.arange` column; it just plays no role in exclusions any more.

**Annotation skill.** `stk-data-annotations` gains an id-column step: scan the raw data for
candidate respondent-id columns (`id`, `resp_id`, `caseid`, `ResponseId`, `uuid`; verify
`is_unique` and non-null), confirm the choice with the user, set `id_col`. Added to the
definition-of-done checklist and the JSON quick reference.

## Implementation notes

- `FileDesc` has `extra="allow"` and extra fields are injected into the data as constant
  categorical columns, so `id_col` must be a *declared* field to stay out of
  `__pydantic_extra__`.
- The nested lift must happen before `pd.concat(...).reset_index(drop=True)` in the stacking
  path — that concat is what destroyed inner identity.
- `combine_first` in the create-block path aligns on index labels and worked by RangeIndex
  luck; with `row_id` as a column it keeps working, but the index must not be set before the
  return boundary or the positional `.iloc` slicing breaks.
- Leaf positional ids are within-file, so adding, removing or reordering *files* never shifts
  another file's ids; row edits inside one raw file shift only that file's tail — the residual
  fragility `id_col` exists to remove.
- `read_and_process_data("path.json")` wraps the bare path in a synthetic `F0` FileDesc, so
  its ids gain a redundant prefix (`F0::M::F0::37`). Harmless and still unique; the canonical
  nested loader `read_annotated_data` yields the clean `M::F0::37`, and exclusions are always
  authored and applied at that level.
- `FileDesc` omits `id_col` from serialization when `None`, mirroring how the col-meta
  serializer drops defaults.
- The hard failure on hook-added rows proved too strict in practice: PR #65 re-mints fresh
  ids when a hook rebuilds the frame, and moves exclusions ahead of postprocessing.
