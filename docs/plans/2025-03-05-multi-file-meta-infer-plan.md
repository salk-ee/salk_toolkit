# Multi-File Meta Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-file meta inference to salk_toolkit: infer metadata for multiple input files, using the last existing annotation as basis and merging inferred metas from other files.

**Architecture:** New `meta_infer.py` module with `infer_meta` (extended with `basis_meta`), `merge_meta_into_basis`, `infer_meta_multi`. io.py imports and re-exports `infer_meta`. CLI `stk_infer` for single and multi-file. **Integration:** Replace current multi-file logic in `read_and_process_data`/`_load_data_files`—infer when no meta, merge when multiple metas.

**Tech Stack:** Python 3.12, pandas, pydantic, salk_toolkit validation/io

**Reference:** Design doc: `docs/plans/2025-03-05-multi-file-meta-infer-design.md`

---

## Task 1: Create meta_infer.py and move infer_meta

**Files:**
- Create: `salk_toolkit/salk_toolkit/meta_infer.py`
- Modify: `salk_toolkit/salk_toolkit/io.py` (remove infer_meta, add import from meta_infer)

**Step 1: Create meta_infer.py with infer_meta moved**

Copy infer_meta and its helpers from io.py into meta_infer.py. Include:
- `_is_categorical`, `max_cats` (or import from io if shared)
- `infer_meta` function body
- Imports: `os`, `json`, `defaultdict`, `cast`, `pd`, `np`, `pyreadstat`, `utils` (from salk_toolkit.utils), `utils.get_categories`, `utils.cached_fn`, `is_datetime`, `_deterministic_categories_and_values` (from io - may need to move or expose)

**Step 2: Update io.py**

- Remove `infer_meta` and `_is_categorical`, `max_cats` (if moved)
- Add: `from salk_toolkit.meta_infer import infer_meta` (or `from .meta_infer import infer_meta`)
- Add `infer_meta` to `__all__` if not already
- Ensure `_data_with_inferred_meta` still works (it calls infer_meta)

**Step 3: Run tests**

Run: `pytest tests/test_io.py -v -k infer`
Expected: All existing infer tests pass

**Step 4: Commit**

```bash
git add salk_toolkit/meta_infer.py salk_toolkit/io.py
git commit -m "refactor: move infer_meta to meta_infer.py"
```

---

## Task 2: Add merge_meta_into_basis

**Files:**
- Create: `tests/test_meta_infer.py`
- Modify: `salk_toolkit/salk_toolkit/meta_infer.py`

**Step 1: Write failing test for merge_meta_into_basis**

```python
# tests/test_meta_infer.py
import pytest
from salk_toolkit.meta_infer import merge_meta_into_basis
from salk_toolkit.validation import soft_validate, DataMeta, ColumnBlockMeta, ColumnMeta

def test_merge_shared_block_category_union():
    basis = soft_validate({
        "structure": {"main": {"name": "main", "columns": {"age": {"categories": ["18-24", "25-34"]}}}},
        DataMeta,
    )
    inferred = soft_validate({
        "structure": {"main": {"name": "main", "columns": {"age": {"categories": ["18-24", "35-44"]}}}},
        DataMeta,
    )
    result = merge_meta_into_basis(basis, inferred)
    cats = result.structure["main"].columns["age"].categories
    assert set(cats) == {"18-24", "25-34", "35-44"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_meta_infer.py::test_merge_shared_block_category_union -v`
Expected: FAIL (merge_meta_into_basis not defined or wrong signature)

**Step 3: Implement merge_meta_into_basis**

```python
# In meta_infer.py
def merge_meta_into_basis(
    basis: DataMeta | dict,
    inferred: DataMeta | dict,
) -> DataMeta:
    """Merge inferred meta into basis. See design doc for rules."""
    from salk_toolkit.validation import soft_validate, DataMeta, ColumnBlockMeta, ColumnMeta
    basis = soft_validate(basis, DataMeta) if isinstance(basis, dict) else basis
    inferred = soft_validate(inferred, DataMeta) if isinstance(inferred, dict) else inferred
    new_structure = dict(basis.structure or {})
    for block_name, block in (inferred.structure or {}).items():
        if block_name not in new_structure:
            new_structure[block_name] = block
            continue
        basis_block = new_structure[block_name]
        # Fail loudly on col_prefix conflicts - output column names would differ
        b_prefix = basis_block.scale.col_prefix if basis_block.scale else None
        i_prefix = block.scale.col_prefix if block.scale else None
        if b_prefix != i_prefix:
            raise ValueError(
                f"Block '{block_name}' has conflicting col_prefix between files: "
                f"'{b_prefix}' vs '{i_prefix}'. Align prefixes in your meta files manually."
            )
        new_columns = dict(basis_block.columns)
        for col_name, col_meta in block.columns.items():
            if col_name not in new_columns:
                new_columns[col_name] = col_meta
                continue
            b_cats = getattr(new_columns[col_name], "categories", None)
            i_cats = getattr(col_meta, "categories", None)
            if b_cats and i_cats and isinstance(b_cats, list) and isinstance(i_cats, list):
                union = list(b_cats) + [c for c in i_cats if c not in b_cats]
                if set(union) != set(b_cats):
                    warn(f"Categories for {col_name} differ - merging to {len(union)} cats")
                new_columns[col_name] = col_meta.model_copy(update={"categories": union})
        new_structure[block_name] = basis_block.model_copy(update={"columns": new_columns})
    return basis.model_copy(update={"structure": new_structure})
```

**Step 4: Run test**

Run: `pytest tests/test_meta_infer.py::test_merge_shared_block_category_union -v`
Expected: PASS

**Step 5: Add tests for new columns and new blocks**

```python
def test_merge_new_column_in_existing_block():
    basis = soft_validate({"structure": {"main": {"name": "main", "columns": {"age": {"categories": ["18-24"]}}}}}, DataMeta)
    inferred = soft_validate({"structure": {"main": {"name": "main", "columns": {"age": {"categories": ["18-24"]}, "gender": {"categories": ["M","F"]}}}}}, DataMeta)
    result = merge_meta_into_basis(basis, inferred)
    assert "gender" in result.structure["main"].columns

def test_merge_new_block():
    basis = soft_validate({"structure": {"main": {"name": "main", "columns": {}}}}, DataMeta)
    inferred = soft_validate({"structure": {"main": {"name": "main", "columns": {}}, "new_block": {"name": "new_block", "columns": {"x": {"categories": ["a"]}}}}}, DataMeta)
    result = merge_meta_into_basis(basis, inferred)
    assert "new_block" in result.structure

def test_merge_col_prefix_conflict_raises():
    basis = soft_validate({"structure": {"parties": {"name": "parties", "scale": {"col_prefix": "p_"}, "columns": {"a": {"categories": ["x"]}}}}}, DataMeta)
    inferred = soft_validate({"structure": {"parties": {"name": "parties", "scale": {"col_prefix": "q_"}, "columns": {"a": {"categories": ["x"]}}}}}, DataMeta)
    with pytest.raises(ValueError, match="col_prefix"):
        merge_meta_into_basis(basis, inferred)
```

**Step 6: Commit**

```bash
git add salk_toolkit/meta_infer.py tests/test_meta_infer.py
git commit -m "feat: add merge_meta_into_basis"
```

---

## Task 3: Extend infer_meta with basis_meta parameter

**Files:**
- Modify: `salk_toolkit/salk_toolkit/meta_infer.py`
- Modify: `tests/test_meta_infer.py`

**Step 1: Write failing test**

```python
def test_infer_meta_with_basis_matching_column():
    import pandas as pd
    from salk_toolkit.meta_infer import infer_meta
    basis = {"structure": {"main": {"name": "main", "columns": {"gender": {"categories": ["Male","Female"], "source": "Q1"}}}}, "constants": {}, "read_opts": {}}
    df = pd.DataFrame({"Q1": ["M","F","M"]})  # Same source name, 50%+ overlap
    result = infer_meta(df=df, basis_meta=basis, meta_file=False)
    # Should map Q1 to gender with basis categories
    assert "gender" in str(result["structure"]) or "main" in result["structure"]
```

**Step 2: Implement basis_meta logic in infer_meta**

Add `basis_meta: DataMeta | dict | None = None` to infer_meta signature.

When basis_meta is provided:
1. Get `_get_original_column_names(basis_meta)` from io (need to import or duplicate)
2. Build joint translate dict from all basis column translate dicts
3. For each column in df: check if (1) type matches, (2) if categorical ≥50% overlap, (3) source name matches. If all match, use basis metadata.

**Step 3: Run tests**

Run: `pytest tests/test_meta_infer.py tests/test_io.py -v -k infer`
Expected: All pass

**Step 4: Commit**

```bash
git add salk_toolkit/meta_infer.py tests/test_meta_infer.py
git commit -m "feat: extend infer_meta with basis_meta for column matching"
```

---

## Task 4: Add infer_meta_multi

**Files:**
- Modify: `salk_toolkit/salk_toolkit/meta_infer.py`
- Modify: `tests/test_meta_infer.py`

**Step 1: Write failing test**

```python
def test_infer_meta_multi_single_file():
    import tempfile
    import pandas as pd
    from salk_toolkit.meta_infer import infer_meta_multi
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        pd.DataFrame({"a": [1,2], "b": ["x","y"]}).to_csv(f.name, index=False)
        result = infer_meta_multi([f.name], meta_file=False)
    assert "structure" in result
```

**Step 2: Implement infer_meta_multi**

```python
def infer_meta_multi(
    files: list[str | dict],
    path: str | None = None,
    meta_file: bool | str = True,
    **kwargs,
) -> dict:
    """Infer meta for multiple files. Uses last file with annotation as basis."""
    if not files:
        raise ValueError("files list cannot be empty")
    # Resolve paths, normalize to FileDesc-like
    # Scan for last with annotation (read_parquet_metadata for parquet, read_json for json)
    # Load basis, fix categories
    # For each other: infer_meta(df=..., basis_meta=basis), merge_meta_into_basis
    # Return merged
```

**Step 3: Run tests**

Run: `pytest tests/test_meta_infer.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add salk_toolkit/meta_infer.py tests/test_meta_infer.py
git commit -m "feat: add infer_meta_multi"
```

---

## Task 5: Add stk_infer CLI

**Files:**
- Modify: `salk_toolkit/salk_toolkit/commands.py`
- Modify: `pyproject.toml`

**Step 1: Add stk_infer to commands.py**

```python
def stk_infer():
    """CLI for inferring meta from single or multiple files."""
    import argparse
    parser = argparse.ArgumentParser(description="Infer metadata for data files")
    parser.add_argument("files", nargs="+", help="Data files to infer from")
    parser.add_argument("-o", "--output", help="Output meta JSON path")
    parser.add_argument("--no-write", action="store_true", help="Do not write to file")
    parser.add_argument("--path", help="Base path for relative file paths")
    args = parser.parse_args()
    from salk_toolkit.meta_infer import infer_meta, infer_meta_multi
    if len(args.files) == 1:
        meta = infer_meta(args.files[0], meta_file=not args.no_write and (args.output or True))
    else:
        meta = infer_meta_multi(args.files, path=args.path, meta_file=args.output or (not args.no_write))
    if args.output and not args.no_write:
        import json
        with open(args.output, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
```

**Step 2: Add entry point to pyproject.toml**

```toml
[project.scripts]
stk_explorer = "salk_toolkit.commands:run_explorer"
stk_translate_dashboard = "salk_toolkit.commands:translate_dashboard"
stk_infer = "salk_toolkit.commands:stk_infer"
```

**Step 3: Smoke test**

Run: `pip install -e . && stk_infer --help`
Expected: Usage prints

**Step 4: Commit**

```bash
git add salk_toolkit/commands.py pyproject.toml
git commit -m "feat: add stk_infer CLI"
```

---

## Task 6: Integrate into read_and_process_data / _load_data_files

**Files:**
- Modify: `salk_toolkit/salk_toolkit/io.py`
- Modify: `salk_toolkit/salk_toolkit/meta_infer.py` (add optional raw_data_dict to infer_meta_multi)
- Modify: `tests/test_io.py`

**Step 1: Extend infer_meta_multi to accept optional raw_data_dict**

Add `raw_data_dict: dict[str, pd.DataFrame] | None = None` to `infer_meta_multi`. When provided, use it instead of loading files (avoids re-loading when called from read_and_process_data).

**Step 2: Update read_and_process_data / _load_data_files for multiple files**

When `len(files_list) > 1`: always call `infer_meta_multi(files_list, raw_data_dict=raw_data_dict, meta_file=False)` to get meta. Use result as `meta_obj`. Remove the raise `"No meta found on any of the files"` for multi-file. Infer_meta_multi handles both no-meta and multiple metas internally (single path).

**Step 4: Add integration tests**

```python
# tests/test_io.py
def test_read_and_process_data_multi_file_no_meta_infers():
    """When no file has meta, infer_meta_multi is used."""
    # Two csv files, no meta -> should infer and return meta
    ...

def test_read_and_process_data_multi_file_multiple_metas_merges():
    """When multiple files have meta, structures are merged."""
    # Two json/parquet files with meta -> merged structure
    ...
```

**Step 5: Run tests**

Run: `pytest tests/test_io.py tests/test_meta_infer.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add salk_toolkit/io.py salk_toolkit/meta_infer.py tests/test_io.py
git commit -m "feat: integrate infer_meta_multi into read_and_process_data"
```

---

## Task 7: Export and final validation

**Files:**
- Modify: `salk_toolkit/salk_toolkit/__init__.py` (if needed)
- Modify: `salk_toolkit/salk_toolkit/meta_infer.py` (add __all__)

**Step 1: Ensure infer_meta_multi and merge_meta_into_basis are exported**

Add to meta_infer.py: `__all__ = ["infer_meta", "infer_meta_multi", "merge_meta_into_basis"]`

**Step 2: Run full test suite**

Run: `ruff check salk_toolkit && pyright salk_toolkit && pytest tests -v`
Expected: All pass

**Step 3: Commit**

```bash
git add salk_toolkit/meta_infer.py
git commit -m "chore: export meta_infer functions"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-03-05-multi-file-meta-infer-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open a new session with executing-plans, batch execution with checkpoints

Which approach?
