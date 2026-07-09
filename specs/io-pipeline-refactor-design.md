# IO pipeline refactor — Design

**Date:** 2026-07-09
**Branch:** `io-pipeline-refactor` (off `main`)
**Status:** Ready for review

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
    names: dict[str, str]             # file_code -> file basename
    env: dict[str, object]            # reader metadata (sav value labels etc), visible to hooks
    meta: DataMeta | None             # meta carried by annotated sources
    # ranges() -> dict[str, slice] over the concat frame
    # concat() -> pd.DataFrame
    # split(s) -> iterator of (file_code, per-file view) aligned via ranges()

@dataclass(frozen=True)
class ProcessOpts:                  # travels with the load recursion as one value
    ignore_exclusions: bool = False
    add_original_inds: bool = False

class HookEnv:                      # the single environment for user code in metafiles
    # base globals: pd, np, sp, stk + bundle.env + meta.constants
    def exec_df(self, code, df, **extra) -> pd.DataFrame   # preprocessing / postprocessing
    def eval(self, expr, **names) -> object                # transform / subgroup_transform / filter
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
    sources.py        _load_sources(), load_dataset() — the recursive loading layer
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
`files` list) is contained in `sources.py`: `_load_sources` → `load_dataset` →
`pipeline.process(bundle)`; `pipeline` never calls back into loading.

## Loading layer

`readers.py` keeps the module-level file tracking (`stk_loaded_files_set`, `stk_file_map`
and their accessors) and a single `_read_tabular(path, opts) -> (df, env)` dispatch for
csv/gz, sav/dta (pyreadstat metadata lands in `env`), and the excel family, including the
MultiIndex column flatten.

`sources.py`:

- `load_dataset(path, opts: ProcessOpts) -> Dataset` — parquet → `parquet.read_parquet_with_metadata`;
  json/yaml → read the metafile, `_load_sources(meta.files)`, `pipeline.process`.
- `_load_sources(files: list[FileDesc], base_path, opts) -> SourceBundle` — per entry:
  annotated formats go through `load_dataset` (the returned `Dataset.meta` directly answers
  "is this a nested multi-file source", replacing the `"result_meta" in locals()` check);
  raw formats through `_read_tabular`. Then, exactly as today: provenance columns
  (`file_code`/`file_name`, not overwritten for nested annotated multi-file sources), extra
  `FileDesc` fields injected as Categoricals with category lists pre-scanned across all
  files, category reconciliation across frames (`_reconcile_categories` — its only home),
  and `_fix_meta_categories` on the carried meta.

## Annotation pipeline (`pipeline.py`)

`process(bundle: SourceBundle, meta: DataMeta, opts: ProcessOpts) -> Dataset` is a
composition of named stages; the current 460-line `_process_annotated_data` body becomes
~20 lines of orchestration. All user-code execution goes through one `HookEnv` built from
`bundle.env` + `meta.constants`.

- `_run_preprocessing(bundle, meta, hooks)` — per-file `exec_df` with `df`, `file_code`,
  `file_name` in scope.
- `_inject_provenance(bundle, meta) -> DataMeta` — (re)write `file_code`/`file_name`
  columns after preprocessing and add the generated `files` block: explicit ordered
  categories in file order, `hidden` iff ≤ 1 file, `setdefault`-merged into a user-defined
  `files` block if present.
- `_build_block(bundle, ndf, block, hooks)` — per column:
  - `_gather_source(bundle, source_spec, cn)` — per-file source resolution (str, or dict
    keyed by file code with `default` fallback), missing/empty-column warnings, concat;
    returns `None` (column dropped from df and meta) when NA across all files.
  - `_apply_transforms(s, cmeta, bundle, ndf, hooks)` — translate → per-file `transform`
    eval (with per-file `s`/`df`/`ndf` views via `bundle.split`) → translate_after →
    datetime/continuous coercion.
  - `_resolve_categories(s, cmeta) -> (s, cmeta)` — `infer` (including the
    translation-dict-order path), numeric snap-to-nearest for numeric series with explicit
    categories, plain categorical coercion; dropped-value warnings; returns the updated
    `ColumnMeta` with inferred categories materialized.
  - `subgroup_transform` per-file via `bundle.split`; duplicate-name detection across
    blocks and columns as today.
- Create blocks delegate to `create_blocks.py` (moved verbatim); generated blocks land in
  the new structure and the `create` field is cleared.
- `_run_postprocessing(ndf, meta, hooks)`.
- `_finalize(ndf, meta, opts) -> Dataset` — `_fix_meta_categories`, exclusion filtering
  via `original_inds`, which is kept as a column iff `opts.add_original_inds`.

No `return_meta`, `return_raw` or `only_fix_categories` anywhere inside: internals always
return `Dataset`.

## Public API (`datasets.py`, `__init__.py`)

- `read_annotated_data(fname, infer=True, return_raw=False, return_meta=False, *,
  ignore_exclusions=False, add_original_inds=False)` — explicit keywords replace the
  `**kwargs` digging. Dispatch: parquet/json/yaml → `load_dataset`; `return_raw` is
  implemented as `_load_sources(...).concat()` in the wrapper (it is "stop after load",
  not a pipeline mode); the `infer=True` fallback builds a meta with `infer_meta` and
  processes a single-file bundle. Keeps its `@overload` pair on `return_meta` — the only
  place overloads remain, and without the duplicated `if return_meta:` double-call.
- `read_and_process_data(desc, return_meta=False, constants=None,
  skip_postprocessing=False, *, ignore_exclusions=False, add_original_inds=False,
  **kwargs)` — desc normalization (str / dict / `DataDescription`), inline `data` dicts,
  then loading via the same `_load_sources` as everything else (its private
  load-concat-fix path is deleted) followed by the consumption stages through `HookEnv`:
  preprocessing, `filter`, `_perform_merges` (semantics unchanged: provenance columns
  dropped on the merge side, overlap error, row-loss warning, `fix_df_with_meta` on the
  merged frame), postprocessing gated by `skip_postprocessing`. Unknown `**kwargs` keys
  produce a warning instead of being silently swallowed (this surfaces the existing
  `file_map=` call in SIP's `proxy_builder.py`, which has never had an effect).
- `io/__init__.py` re-exports the current `__all__` plus the names other modules import
  from `salk_toolkit.io` today: `fix_df_with_meta` (dashboard) and `read_json` (explorer,
  tests). In-repo test imports of private helpers
  (`_convert_number_series_to_categorical`, `_convert_datetime_series_to_categorical`,
  `_get_original_column_names`) are updated to the new module paths.

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
existing metafile. Concretely: the full `tests/test_io.py` suite (90 tests, several of
which assert warning output via capsys) passes unmodified except for the three private
import paths above.

## Phasing

Each phase lands as its own commit with the full test suite green:

1. **Deadwood** — delete `only_fix_categories`, `raw_data`, the `data_meta` kwarg fixup,
   the `locals()` check, the kwargs digging, and the double-call; still in `io.py`.
2. **Decomposition in place** — introduce `Dataset`, `SourceBundle`, `ProcessOpts`,
   `HookEnv`; split `_process_annotated_data` into the named stages; internals return
   `Dataset`.
3. **Package split** — move the stages into `salk_toolkit/io/` with the compat
   `__init__.py`; `io.py` is deleted.
4. **Unify loading** — `read_and_process_data` consumes `_load_sources`/`process`; its
   parallel load path is deleted.

## Verification

- `pytest tests/test_io.py` and the full STK suite green after every phase.
- pyright: 0 errors on touched files; ruff: clean.
- SIP quick suite run against the refactored STK checkout (integration gate — SIP imports
  only public names, verified by grep).
- `grep` gate: no remaining references to `only_fix_categories`, `raw_data=`, or
  `return_raw` outside `datasets.py`.

## Out of scope

- Fixing `proxy_builder.py`'s ineffective `file_map=` argument (a SIP change; the new
  unknown-kwarg warning makes it visible).
- Honoring `file_map=` as a real per-call parameter (the global `set_file_map` remains the
  mechanism).
- Any behavior change to hooks, create blocks, category inference, or merge semantics.
