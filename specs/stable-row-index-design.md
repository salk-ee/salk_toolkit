# Stable row index and id-keyed exclusions

**Modules:** `salk_toolkit/validation.py`, `salk_toolkit/io/` (`sources.py`, `pipeline.py`, `datasets.py`, `parquet.py`), `.cursor/skills/stk-data-annotations/`

## Goal

Every DataFrame returned by the io layer carries a unique, deterministic index that names
each row's provenance, and meta exclusions are keyed on that identity. Today row identity
is a RangeIndex rebuilt by `reset_index(drop=True)` at every concat, and `excluded` entries
are absolute integer positions into the post-concat frame — so any inner filter, file
reorder, or preprocessing row change silently shifts what an outer meta's exclusions point
at. With identity assigned at load time, meta-within-meta exclusions compose to arbitrary
depth: each level filters by id, and a change in an inner filter can never re-aim an outer
one.

## Design

**Row id.** Each row gets a string id in a reserved system column `row_id` (joining
`file_code`/`file_name`), assigned in `_load_data_files` immediately after each file is
read — before any concat, preprocessing, or column processing can move rows:

- Raw file (csv/sav/xlsx/…): `{file_code}::{leaf}` where `leaf` is the value of the
  declared `id_col` if set, else the 0-based row position within that file.
- Annotated inner dataset (json/yaml/parquet loaded as a `files` entry): rows already
  carry ids; the outer level prepends its own code — `wave1::37` stacked under code `F0`
  becomes `F0::wave1::37`. Variable nesting depth is why this is a delimited string, not a
  MultiIndex.
- Ids are opaque: nothing ever splits them, code only prepends prefixes and does exact
  string matching, so `::` inside id_col values is harmless.

The id rides as a column through the positionally-indexed middle of the pipeline (the
`file_index_ranges`/`.iloc` machinery is untouched). At the return boundary of
`read_annotated_data` / `read_and_process_data` it becomes the index
(`set_index("row_id")`), asserted unique and non-null.

**`id_col` meta syntax.** `DataMeta.id_col` names a raw-data column that uniquely
identifies rows within each file (a respondent id); `FileDesc.id_col` overrides it
per-file. Validated at load: the column must exist, be non-null, and unique within the
file — hard error otherwise. Positional leaf ids are deterministic only for a byte-stable
source file; a declared natural key survives re-exports and reorders, so the annotation
skill actively looks for one.

**Exclusions.** `DataMeta.excluded` becomes `List[Tuple[str, str]]` — `(row_id, reason)` —
applied at the same pipeline point as today via `~df["row_id"].isin(ids)`. An exclusion id
that matches nothing emits a `warn` (naming the unmatched ids) but is not an error: an
inner meta may legitimately have already filtered that row, so no-match must not break the
load — but a typo'd id is otherwise undetectable, so it is surfaced. Legacy integer entries
fail validation with a message pointing here —
no positional fallback is kept (the few metas in the wild that used ints are migrated by
hand as encountered; the exclusion-writing tools in SIP are being rebuilt against the new
syntax on data-quality-v2).

**Parquet round-trip.** `write_parquet_with_metadata` already preserves the index via
`pa.Table.from_pandas`. On read, if the restored frame carries a `row_id` index (or
column), it is lifted back into the `row_id` column before the stacking concat would
destroy it; frames without one (raw parquet sources) get fresh positional ids like any
other raw file. At the outer return boundary the column is set back as the index as usual.

**Merges.** `_perform_merges` runs after exclusions, so ids never need to be referenced
post-merge — only the output invariant matters. `_repair_merge_row_ids` keeps left ids where
already unique (many-to-one enrich); right-only rows (NaN id from `right`/`outer` joins)
collapse to the merge `tag` and any id duplicated by a one-to-many/`cross` join gets a
`::m{k}` per-group suffix — one `fillna` + `duplicated` + `groupby().cumcount()` pass,
deterministic because merge output order is.

**Row-count discipline.** After each user-code hook (`preprocessing`, `postprocessing`,
`subgroup_transform` on the frame level), assert `row_id` still exists, non-null, unique.
Dropping or reordering rows is fine; rows *added* by user code have no identity and fail
loudly. No auto-minting of ids for synthesized rows until a real use case demands it.

**Scope of the guarantee.** io-layer returns only. The pp layer bootstrap
(`_augment_draws`) duplicates rows by design and redraws/resets its own index; there the
id is provenance ("which source row generated this draw"), not a unique key.

**`original_inds` unchanged.** `add_original_inds=True` keeps producing the positional
post-processing `np.arange` column; it no longer plays any role in exclusions. SIP's runner
is unaffected: it already resets `sdf`'s index right after load
(`reset_index(names="orig_index")`), so the string `row_id` index is captured as
`orig_index` provenance and replaced with a clean RangeIndex for SIP's positional work.

**Annotation skill.** `stk-data-annotations` gains an id-column step: when annotating,
scan the raw data for candidate respondent-id columns (names like `id`, `resp_id`,
`caseid`, `ResponseId`, `uuid`; verify `is_unique` and non-null), confirm the choice with
the user, and set `id_col` in the meta. Added to the definition-of-done checklist and the
JSON quick reference.

## Implementation notes

- `FileDesc` has `extra="allow"` and extra fields are injected into the data as constant
  categorical columns (`sources.py` extra-field loop) — `id_col` must be a *declared*
  field so it stays out of `__pydantic_extra__`.
- The nested lift must happen before `pd.concat(...).reset_index(drop=True)` in the
  stacking path — that concat is what currently destroys inner identity.
- `combine_first` in the create-block path aligns on index labels and currently works by
  RangeIndex luck; with `row_id` as a column it keeps working unchanged, but do not set
  the index earlier than the return boundary or the positional `.iloc` slicing breaks.
- The `raw_data` direct-injection paths (`raw_data=` DataFrame or dict in
  `_process_annotated_data`) mint positional ids at ingestion, same rule as raw files.
- Leaf positional ids are within-file, so adding/removing/reordering *files* never shifts
  another file's ids; row edits inside one raw file shift only that file's tail — the
  residual fragility `id_col` exists to remove.
- `read_and_process_data("path.json")` wraps the bare path in a synthetic `F0` FileDesc, so
  its returned ids gain a redundant `F0::` prefix (`F0::M::F0::37`). Harmless and still
  unique/deterministic; the canonical nested loader `read_annotated_data` yields the clean
  `M::F0::37`. Exclusions are always authored and applied at the `read_annotated_data`
  level (clean ids), so the wrapper prefix never affects exclusion keys.
- `FileDesc` omits `id_col` from serialization when `None`, keeping saved metas free of
  `id_col: null` noise (mirrors how the col-meta serializer drops defaults).
