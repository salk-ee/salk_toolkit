# IO pipeline refactor — Design

**Date:** 2026-07-09
**Branch:** `io-pipeline-refactor` (off `main`)
**Status:** Implemented (all phases), pending review

## Goal

Restructure `salk_toolkit/io.py` (2372 lines; four mutually-recursive functions with mode
flags threaded through the recursion) into a `salk_toolkit/io/` package with a layered
pipeline: format readers → source bundle → annotation pipeline → consumption. The public
API is unchanged; every internal function has one signature and returns one thing.

## Conceptual model

Two description languages, one value type flowing between them:

- **`DataMeta`** is a *construction* recipe: raw files → clean typed dataset. Its product,
  `(df, resolved meta)`, is exactly what a processed parquet stores.
- **`DataDescription`** is a *consumption* recipe: already-built datasets → analysis frame
  (preprocessing / filter / merge / postprocessing).
- A **source** is one of three maturity levels: raw tabular (csv/sav/xlsx/…), an
  annotation file (json/yaml = recipe), or a processed parquet (= finished product).
  Sources recurse: a `files` entry may itself be an annotation file or parquet.

## Core types (`core.py`)

```python
class Dataset(NamedTuple):          # what every internal function returns
    df: pd.DataFrame
    meta: DataMeta | None

@dataclass
class SourceBundle:                 # per-file frames before concat
    frames: dict[str, pd.DataFrame]   # keyed by file_code, insertion-ordered
    env: dict[str, object]            # reader metadata (sav value labels etc), visible to hooks
    meta: DataMeta | None             # meta carried by annotated sources
    # ranges() -> dict[str, slice] over the concat frame
    # concat(reset_index=False) -> pd.DataFrame
    # (file names are derived from the stamped provenance columns via _file_meta_map)

@dataclass(frozen=True)
class ProcessOpts:                  # travels with the load recursion as one value
    ignore_exclusions: bool = False
    add_original_inds: bool = False

class HookEnv:                      # the single environment for user code in metafiles
    # exec base: pd, np, sp, stk + bundle.env + meta.constants; eval base: pd, np, stk + constants
    # (matching the historical hook namespaces exactly)
    def exec_df(self, code, df, **extra) -> pd.DataFrame   # preprocessing / postprocessing
    def eval(self, expr, **names) -> object                # transform / subgroup_transform
```

`core.py` also holds the shared series helpers used by both loading and processing:
`_deterministic_categories_and_values`, `_is_series_of_lists`, and the
number/datetime → categorical converters.

## Module layout

```
salk_toolkit/io/
    __init__.py       compat re-exports (see Public API)
    core.py           Dataset, SourceBundle, ProcessOpts, HookEnv, shared series helpers
    readers.py        file tracking + file map (get/set/reset), raw tabular readers
    sources.py        _load_data_files(), _load_dataset(), _process_annotated_data(),
                      _reconcile_categories() — the recursive loading layer
    pipeline.py       process(bundle, meta, opts) and its column stages
    create_blocks.py  topk / topk-regex / topk-list / maxdiff create-block builders (moved verbatim)
    datasets.py       public entry points: read_annotated_data, read_and_process_data,
                      _perform_merges, infer_meta
    parquet.py        read/write parquet with embedded meta, replace_data_meta_in_parquet,
                      _fix_parquet_categories
    meta.py           extract_column_meta, group_columns_dict, list_aliases, fix_df_with_meta,
                      _fix_meta_categories, update_meta_with_model_fields, _change_df_to_meta
```

The import DAG is acyclic: `core` ← `readers`/`meta`/`create_blocks` ← `parquet`/`pipeline`
← `sources` ← `datasets` ← `__init__`. The inherent recursion (annotated source inside a
`files` list) is contained in `sources.py`: `_load_data_files` → `_load_dataset` →
`_process_annotated_data` → `pipeline.process(bundle)`; `pipeline` never calls back into
loading. `stk_file_map`/`stk_loaded_files_set` are accessed as `readers` module attributes
so `set_file_map` rebinding stays visible across modules.

## Loading layer

`readers.py` keeps the module-level file tracking (`stk_loaded_files_set`, `stk_file_map`
and their accessors) and a single `_read_tabular(path, opts) -> (df, env)` dispatch for
csv/gz, sav/dta (pyreadstat metadata lands in `env`), and the excel family, including the
MultiIndex column flatten.

`sources.py`:

- `_load_dataset(path, opts: ProcessOpts) -> Dataset` — parquet → `parquet.read_parquet_with_metadata`;
  json/yaml → `_process_annotated_data` (= `_read_meta_input` + `_load_meta_sources` +
  `pipeline.process`). No inference: this is the nested-source path, which never infers.
- `_load_data_files(files: list[FileDesc], path, read_opts, opts) -> SourceBundle` — per entry:
  annotated formats go through `_load_dataset` (the returned `Dataset.meta` directly answers
  "is this a nested multi-file source", replacing the `"result_meta" in locals()` check);
  raw formats through `_read_tabular`. Then, exactly as today: provenance columns
  (`file_code`/`file_name`, not overwritten for nested annotated multi-file sources), extra
  `FileDesc` fields injected as Categoricals with category lists pre-scanned across all
  files, category reconciliation across frames (`_reconcile_categories` — its only home),
  and `_fix_meta_categories` on the carried meta.

## Annotation pipeline (`pipeline.py`)

`process(bundle: SourceBundle, meta: DataMeta, opts: ProcessOpts) -> Dataset` is a
composition of named stages; the former 460-line `_process_annotated_data` body is ~20
lines of orchestration. All user-code execution goes through one `HookEnv` built from
`bundle.env` + `meta.constants`.

- Preprocessing — per-file `exec_df` with `df`, `file_code`, `file_name` in scope
  (file names via `_file_meta_map`, computed before preprocessing can drop the columns).
- `_inject_files_block(bundle, meta, file_names) -> DataMeta` — (re)write
  `file_code`/`file_name` columns after preprocessing and add the generated `files` block:
  explicit ordered categories in file order, `hidden` iff ≤ 1 file, `setdefault`-merged
  into a user-defined `files` block if present.
- `_build_columns(bundle, meta, hooks)` — per block/column:
  - `_gather_source(bundle, source_spec, orig_cn, cn, generated)` — per-file source
    resolution (str, or dict keyed by file code with `default` fallback), missing/empty
    warnings, concat; returns `None` (column skipped in df) when NA across all files.
  - `_apply_transforms(s, cmeta, bundle, ndf, cn, hooks)` — translate → per-file
    `transform` eval (with per-file `s`/`df`/`ndf` views via `bundle.ranges()`) →
    translate_after → datetime/continuous coercion.
  - `_resolve_categories(s, cmeta, cn) -> (s, cmeta)` — `infer` (including the
    translation-dict-order path), numeric snap-to-nearest for numeric series with explicit
    categories, plain categorical coercion; dropped-value warnings; returns the updated
    `ColumnMeta` with inferred categories materialized.
  - `_apply_subgroup_transform` per-file via `bundle.ranges()` (its `df` is now the
    matching per-file raw frame — previously it was whichever frame the preceding column
    loop last touched, wrong for multi-file data); duplicate-name detection across blocks
    and columns as before.
  - Create blocks delegate to `create_blocks.py` (moved verbatim); generated blocks land
    in the new structure and the `create` field is cleared.
- Postprocessing — one `exec_df` on the built frame.
- `_fix_meta_categories` + `_apply_exclusions(ndf, meta, opts) -> Dataset` — exclusion
  filtering via `original_inds`, kept as a column iff `opts.add_original_inds`.

No `return_meta`, `return_raw` or `only_fix_categories` anywhere inside: internals always
return `Dataset`.

## Public API (`datasets.py`, `__init__.py`)

- `read_annotated_data(fname, infer=True, return_raw=False, return_meta=False, *,
  ignore_exclusions=False, add_original_inds=False)` — explicit keywords replace the
  `**kwargs` digging. Dispatch: json/yaml → `_process_annotated_data`; parquet →
  `read_parquet_with_metadata`; `return_raw` short-circuits to `bundle.concat()` right
  after loading (it is "stop after load", not a pipeline mode); the `infer=True` fallback
  builds a meta with `infer_meta` and processes a single-file bundle. Keeps its
  `@overload` pair on `return_meta` — the only overloads remaining, without the duplicated
  `if return_meta:` double-call.
- `read_and_process_data(desc, return_meta=False, constants=None,
  skip_postprocessing=False, *, ignore_exclusions=False, add_original_inds=False)`
  — desc normalization (str / dict / `DataDescription`), inline `data` dicts,
  then loading via the same `_load_data_files` as everything else (its private
  load-concat path and its duplicate `_fix_meta_categories` pass are deleted —
  category reconciliation happens once, inside `_load_data_files`) followed by the consumption stages through `HookEnv`:
  preprocessing, `filter`, `_perform_merges` (semantics unchanged: provenance columns
  dropped on the merge side, overlap error, row-loss warning, `fix_df_with_meta` on the
  merged frame), postprocessing gated by `skip_postprocessing`. The `**kwargs` swallow is gone
  entirely: unknown arguments are now a TypeError. (Its only caller was the ineffective
  `file_map=` in SIP's `proxy_builder.py`, removed by salk_internal_package#95.)
- `io/__init__.py` re-exports the current `__all__` plus the names other modules import
  from `salk_toolkit.io` today: `fix_df_with_meta` (dashboard), `read_json` (explorer,
  tests) and `_fix_meta_categories` (SIP `sampling/meta.py`). In-repo test imports of
  private helpers (`_convert_number_series_to_categorical`,
  `_convert_datetime_series_to_categorical`, `_get_original_column_names`) are updated to
  the new module paths.

## Removed

Dead code, deleted rather than ported:

- `only_fix_categories` — threaded through all four functions, never passed as `True` by
  any caller in STK, SIP, or tools.
- `_process_annotated_data(raw_data=...)` and its `DataFrame | dict` / `{"F0": df}`
  compatibility branch — no callers.
- The `kwargs["data_meta"]` post-merge fixup in `read_and_process_data` — no callers, and
  `_perform_merges` already applies `fix_df_with_meta`.
- The `"result_meta" in locals()` introspection in `_load_data_files`.
- The duplicated overload towers on `_process_annotated_data` and the byte-identical
  `if return_meta:` call duplication in `read_annotated_data`.

## Behavioral invariants

Output frames, resolved metas, and warning texts are identical to `main` for every
existing metafile. Concretely: the full `tests/test_io.py` suite (several tests assert
warning output via capsys) passes unmodified through phase (a); phase (b) only re-points
three private test imports to their new submodule paths and adds one regression test.

## Phasing

Two phases, each landing green, cleanly separable for review:

**(a) Pure package creation.** `salk_toolkit/io.py` is split into the `salk_toolkit/io/`
package by *relocation only* — every function/class body is byte-identical to `main`. An
automated body-diff proves it: the sole deviations are the mechanical necessities of
crossing a module boundary — module-qualifying the two rebindable globals as
`readers.stk_file_map` / `readers.stk_loaded_files_set` (a `from readers import …` would
capture a stale binding and silently break `set_file_map`), a deferred
`from …datasets import read_annotated_data` inside `_load_data_files` to break the one
`sources ↔ datasets` import cycle, and per-module import headers. No logic changes, so this
phase touches no tests and no callers.

**(b) Cleanup refactor.** All logic changes land here, on the package:

- Deadwood — delete `only_fix_categories`, the `_process_annotated_data(raw_data=)` compat
  branch, the `data_meta` kwarg fixup, the `"result_meta" in locals()` check, the kwargs
  digging, and the duplicated overload/`if return_meta:` calls.
- Decomposition — introduce `Dataset`, `SourceBundle`, `ProcessOpts`, `HookEnv`; split
  `_process_annotated_data` into the named stages (`process` + `_gather_source` /
  `_apply_transforms` / `_resolve_categories` / …); factor `readers._read_tabular` and
  `sources._load_dataset` out of `_load_data_files` (the latter turning the deferred
  cycle-break import into a proper call); internals return `Dataset`.
- Unification — `read_and_process_data` consumes the shared loader; its duplicate
  category-fixup pass and its `**kwargs` swallow are deleted; the per-file `read_opts` leak
  is fixed; the private test imports move to submodule paths.

## Verification

- `pytest tests/test_io.py` and the full STK suite green after every phase.
- pyright: 0 errors on touched files; ruff: clean.
- SIP quick suite run against the refactored STK checkout (integration gate — SIP imports
  only public names, verified by grep).
- `grep` gate: no remaining references to `only_fix_categories`, `raw_data=`, or
  `return_raw` outside `sources.py`/`datasets.py`.
- Determinism: verification surfaced a pre-existing hash-order flakiness in
  `plots._cat_to_cont_axis` (pandas 3 guesses date formats from the first element;
  `test_lines_date` failed on ~1/15 runs on `main` too). Fixed alongside this work with
  `format="mixed"`, matching the `is_date_str_series` gate.

## Out of scope

- Honoring `file_map=` as a real per-call parameter (the global `set_file_map` remains the
  mechanism; SIP's ineffective `file_map=` threading is removed by salk_internal_package#95).
- Any behavior change to hooks, create blocks, category inference, or merge semantics.
