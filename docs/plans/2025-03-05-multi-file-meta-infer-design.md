# Multi-File Meta Inference — Design Document

**Date:** 2025-03-05

## Goal

Add functionality in salk_toolkit to infer metadata for multiple input files, not just a single file. Comprising files may already have annotations (JSON or parquet). The logic uses the last existing annotation as a basis and merges inferred metas from other files.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Basis file | Last in `files` list order that has an annotation |
| Category mismatch (basis vs new) | Merge categories (union), warn |
| Module location | New `meta_infer.py`; move current infer logic there |
| New block order | Don't care; just add |
| CLI | `stk_infer` for single and multi-file |
| Basis annotation prep | Fix categories against each file's data before use |
| Column matching (infer-from-basis) | Only if (1) type matches, (2) ≥50% category overlap, (3) source name matches — all must hold |
| Matched columns | Carry over everything from basis; don't overwrite inferred labels |
| Basis translations | Joint dict, applied before translate_fn |
| infer_meta_from_basis | Not separate; add `basis_meta` param to `infer_meta` |

## Architecture

- **New module:** `salk_toolkit/meta_infer.py`
- **Functions:** `infer_meta` (extended with `basis_meta`), `merge_meta_into_basis`, `infer_meta_multi`
- **io.py:** Import `infer_meta` from `meta_infer`, re-export for backward compatibility
- **CLI:** `stk_infer` in `commands.py`, entry point in `pyproject.toml`

## Merge Logic (`merge_meta_into_basis`)

1. **Shared blocks/columns:** For each column in both: if categories differ → union, preserve basis order, append new, warn.
2. **New columns in existing blocks:** Add to basis block with inferred metadata.
3. **New blocks:** Add to basis (order unspecified).

## Infer-from-Basis Logic (inside `infer_meta` when `basis_meta` set)

- Match only if: type same, ≥50% category overlap, source name matches.
- When matched: carry over all basis metadata; don't overwrite inferred labels.
- Basis translate dicts → joint dict, applied before translate_fn.
- Unmatched columns: use existing infer_meta heuristic.

## Orchestration (`infer_meta_multi`)

1. Scan files to find last with annotation.
2. Load basis: data + meta (or infer). Fix categories.
3. For each other file: load data, `infer_meta(df=..., basis_meta=basis)`, `merge_meta_into_basis(basis, inferred)`.
4. Return merged meta.

## CLI (`stk_infer`)

- `stk_infer file1 [file2 ...] -o meta.json`
- Single file → infer_meta; multi-file → infer_meta_multi
- Options: `-o`, `--no-write`, `--path`

## Integration: Replace Current Multi-File Logic

Replace/adjust the current logic in `read_and_process_data` and `_load_data_files`:

**When multiple files:** Always use `infer_meta_multi`. It handles both (1) no meta and (2) some/all files with meta: finds last with annotation as basis (or infers from last if none), merges inferred metas from other files. Single path, no special cases.

**When single file:** Keep current behavior (use meta if present, else infer from single file).

`infer_meta_multi` should accept an optional `raw_data_dict` to avoid re-loading when called from read_and_process_data.

## col_prefix Conflict Detection

When merging, a shared block name with **differing `col_prefix`** across files produces different output column names for what is nominally the same block. This cannot be silently reconciled.

**Rule:** In `merge_meta_into_basis`, for each shared block, if `basis_block.scale.col_prefix != inferred_block.scale.col_prefix`, raise a `ValueError` with an explicit message indicating the block name and the two conflicting prefixes. This is not yet handled; users must align prefixes in their meta files manually.

## Error Handling & Testing

- Missing file → FileNotFoundError
- Empty files list → ValueError
- Unit tests: `infer_meta` with basis_meta, `merge_meta_into_basis`, `infer_meta_multi`
- Integration test: `read_and_process_data` with multi-file, no meta → infers; multi-file, multiple metas → merges
- CLI smoke test
