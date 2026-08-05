# io pipeline refactor (PRs #53, #74, #72)

**Modules:** `salk_toolkit/io.py` → `salk_toolkit/io/` (`core`, `readers`, `sources`,
`pipeline`, `meta`, `parquet`, `create_blocks`, `datasets`)

## Goal

`salk_toolkit/io.py` was 2397 lines built around four mutually-recursive functions with mode
flags (`return_meta`, `return_raw`, `only_fix_categories`, `raw_data=`) threaded through the
recursion, so no function had one signature or one return type. It becomes a package with a
layered pipeline — format readers → source bundle → annotation pipeline → consumption —
where every internal function has one signature and returns one thing. The public API is
unchanged.

## Design

**Conceptual model.** Two description languages, one value type flowing between them:

- **`DataMeta`** is a *construction* recipe: raw files → clean typed dataset. Its product,
  `(df, resolved meta)`, is exactly what a processed parquet stores.
- **`DataDescription`** is a *consumption* recipe: already-built datasets → analysis frame
  (preprocessing / filter / merge / postprocessing).
- A **source** is one of three maturity levels: raw tabular (csv/sav/xlsx/…), an annotation
  file (json/yaml = recipe), or a processed parquet (= finished product). Sources recurse: a
  `files` entry may itself be an annotation file or a parquet.

**Core types (`core.py`).**

- `Dataset(NamedTuple)` — `(df, meta: DataMeta | None)`, what every internal function returns.
- `SourceBundle` — per-file frames before concat: `frames` keyed by file_code and
  insertion-ordered, `env` (reader metadata such as sav value labels, visible to hooks), and
  the `meta` carried by annotated sources; `ranges() -> dict[str, slice]` and `concat()` are
  how the per-file view and the concatenated view stay reconcilable.
- `ProcessOpts` — frozen `(ignore_exclusions, add_original_inds)`, travelling with the load
  recursion as one value instead of two flags.
- `HookEnv` — the single environment for user code in metafiles: `exec_df(code, df, **extra)`
  for preprocessing/postprocessing and `eval(expr, **names)` for transform/subgroup_transform,
  over the historical namespaces exactly (exec: `pd`, `np`, `sp`, `stk` + `bundle.env` +
  `meta.constants`; eval: `pd`, `np`, `stk` + constants).

`core.py` also holds the row-id machinery and the series helpers shared by loading and
processing (`_deterministic_categories_and_values`, `_is_series_of_lists`, the
number/datetime → categorical converters).

**Module layout and DAG.** `readers` (file tracking, file map, raw tabular readers) ·
`sources` (the recursive loading layer) · `pipeline` (`process` and its column stages) ·
`meta` (`extract_column_meta`, `group_columns_dict`, `fix_df_with_meta`, …) · `parquet` ·
`create_blocks` (topk / topk-regex / topk-list / maxdiff builders, moved verbatim) ·
`datasets` (public entry points) · `__init__` (compat re-exports). Imports are acyclic:
`core` ← `readers`/`meta`/`create_blocks` ← `parquet`/`pipeline` ← `sources` ← `datasets` ←
`__init__`. The inherent recursion — an annotated source inside a `files` list — is contained
in `sources.py`: `_load_data_files` → `_load_dataset` → `_process_annotated_data` →
`pipeline.process`; `pipeline` never calls back into loading.

**Loading layer.** `readers._read_tabular(path, opts) -> (df, env)` is a single dispatch for
csv/gz, sav/dta (pyreadstat metadata lands in `env`) and the excel family, including the
MultiIndex column flatten. `sources._load_dataset(path, opts) -> Dataset` handles the nested
source path — parquet via `read_parquet_with_metadata`, json/yaml via
`_process_annotated_data` — and never infers. `_load_data_files` routes each entry through one
of those two, then does what it always did: stamp provenance columns, inject extra `FileDesc`
fields as Categoricals with category lists pre-scanned across files, reconcile categories
across frames (`_reconcile_categories`, now with only one home), and `_fix_meta_categories` on
the carried meta. `Dataset.meta` being non-`None` is what identifies a nested multi-file
source, replacing a `"result_meta" in locals()` introspection.

**Annotation pipeline.** `process(bundle, meta, opts) -> Dataset` is a composition of named
stages, reducing the former 460-line `_process_annotated_data` body to ~20 lines of
orchestration; all user code runs through one `HookEnv`.

- Preprocessing — per-file `exec_df` with `df`, `file_code`, `file_name` in scope (names
  resolved via `_file_meta_map` *before* preprocessing can drop the columns).
- `_inject_files_block` — rewrites `file_code`/`file_name` after preprocessing and adds the
  generated `files` block: explicit ordered categories in file order, `hidden` iff ≤ 1 file,
  `setdefault`-merged into a user-defined `files` block if present.
- `_build_columns` per block/column: `_gather_source` (per-file resolution of a str or
  code-keyed dict with `default` fallback, missing/empty warnings, concat; `None` when NA
  across all files) → `_apply_transforms` (translate → per-file `transform` eval with `s`/`df`/
  `ndf` views from `bundle.ranges()` → translate_after → datetime/continuous coercion) →
  `_resolve_categories` (`infer` including the translation-dict-order path, numeric
  snap-to-nearest, plain categorical coercion, dropped-value warnings) → `_apply_subgroup_transform`.
  Create blocks delegate to `create_blocks.py`; generated blocks land in the new structure and
  the `create` field is cleared.
- Postprocessing — one `exec_df` on the built frame — then `_fix_meta_categories` and
  `_apply_exclusions`, which filters via `original_inds` and keeps the column iff
  `opts.add_original_inds`.

**Public API.** `read_annotated_data(fname, infer=True, return_raw=False, return_meta=False, *,
ignore_exclusions=False, add_original_inds=False)` — explicit keywords replace `**kwargs`
digging; `return_raw` short-circuits to `bundle.concat()` right after loading (it is "stop
after load", not a pipeline mode); the `infer=True` fallback builds a meta with `infer_meta`
and processes a single-file bundle. `read_and_process_data(desc, return_meta=False,
constants=None, skip_postprocessing=False, ignore_exclusions=False, add_original_inds=False)`
normalizes its desc, then loads through the *same* `_load_data_files` as everything else —
its private load-concat path and its duplicate `_fix_meta_categories` pass are gone, since
category reconciliation now happens once — followed by the consumption stages through
`HookEnv`: preprocessing, `filter`, `_perform_merges` (semantics unchanged), postprocessing.
`io/__init__.py` re-exports the previous `__all__` plus the names other packages import from
`salk_toolkit.io` today: `fix_df_with_meta`, `read_json`, `_fix_meta_categories`.

## Implementation notes

- Deleted rather than ported: `only_fix_categories` (threaded through all four functions,
  never passed `True` by any caller in STK, SIP or tools), the
  `_process_annotated_data(raw_data=)` `DataFrame | dict` compat branch, the
  `"result_meta" in locals()` check, and the byte-identical `if return_meta:` call
  duplication behind the overload towers.
- The `**kwargs` swallow on `read_and_process_data` is gone entirely, so unknown arguments
  are now a `TypeError`. That removed `data_meta=`, which had one live caller — SIP's
  `generate_population`, which passes the survey meta so a population frame and the survey
  frame agree on the categories of a column a model-level `merge` adds to both. PR #77
  restored it as an explicit parameter.
- Output frames, resolved metas and warning texts are identical to the pre-refactor `main`
  for every existing metafile; `tests/test_io.py` (several tests assert warning output via
  capsys) passes unmodified through the pure-move phases.
- The split landed as pure moves first (#53 creating the package, #74 extracting the column
  build and exclusion stages) with every function body byte-identical, and all logic changes
  in the cleanup (#72). The only deviations a pure move could not avoid: module-qualifying
  the two rebindable globals as `readers.stk_file_map` / `readers.stk_loaded_files_set` (a
  `from readers import …` captures a stale binding and silently breaks `set_file_map`), and a
  deferred import inside `_load_data_files` to break the one `sources ↔ datasets` cycle —
  later replaced by a proper `_load_dataset` call.
- `_apply_subgroup_transform`'s `df` is now the matching per-file raw frame; it was
  previously whichever frame the preceding column loop last touched, which was wrong for
  multi-file data.
