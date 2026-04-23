# Unified Block-Processing Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the three per-block-type `_create_*_metas_and_dfs` wrappers and their duplicated `_apply_pre_transform_translate` calls into a single universal driver `_process_block` that executes the four stages named in `specs/2026-04-20-block-processing-pipeline.md` (match → explode → transform → build-output) plus a universal stage 5 (`translate_after`). Makes adding a new block type a matter of supplying only a transform and a build-output function. Simultaneously, simplify MaxDiff: drop the `choice_mapping` field and collapse index→display-name translation into `scale.translate`. Deprecate `scale.translate_after` on MaxDiff blocks (hard fail).

**Architecture:** Three orthogonal changes. **(1)** Push regex-resolution down into `_subgroup_explode` via a `resolve_role_columns` method so that MaxDiff's `best_columns` / `worst_columns` / `set_columns` are concrete lists on every sibling before any transform runs — the `io.py:772-805` narrowing block disappears. **(2)** Remove `MaxDiffBlock.choice_mapping`; `scale.translate` replaces it in its dual role (index→display-name lookup + cell translation). Inline set cells prefer integer-list tokens (`[1, 2, 3]`) over comma-separated strings. **(3)** Lift `_apply_pre_transform_translate` and `translate_after` out of every transform body into a universal driver that calls `transform(block, df)` followed by `_apply_post_transform_translate(block, sdf, meta)`; MaxDiff rejects `scale.translate_after` with a hard fail.

**Tech Stack:** Python 3.11+, pandas, Pydantic v2, pytest. All work lives in `salk_toolkit/salk_toolkit/io.py`, `salk_toolkit/salk_toolkit/validation.py`, and `salk_toolkit/tests/test_io.py`.

**Non-goals:**
- TopK/OneHot/plain blocks preserve their existing `scale.translate_after` semantics. Only MaxDiff deprecates `translate_after`.
- No changes to `_subgroup_explode`'s sibling-naming logic.
- No changes to the `CreateBlockModel` discriminated union.

---

## File Structure

- **Modify `salk_toolkit/salk_toolkit/validation.py`:** Add two methods to `ColumnBlockMeta` with overrides on `MaxDiffBlock`.
  - `input_df_columns(self, df: pd.DataFrame) -> list[str]` — df-columns the block reads; default: matched `from_columns`; MaxDiff override: union of best/worst/sets after resolution.
  - `resolve_role_columns(self, df: pd.DataFrame, sibling_label: str) -> dict[str, object]` — per-sibling regex→list resolution for non-`from_columns` role fields; default: `{}`; MaxDiff override: resolves `best_columns`, `worst_columns`, `set_columns`.

- **Modify `salk_toolkit/salk_toolkit/io.py`:**
  - Extend `_subgroup_explode` (currently 523–566) to apply `resolve_role_columns` to each sibling via `model_copy`.
  - Add `_apply_post_transform_translate(block, sdf, meta) -> tuple[pd.DataFrame, ColumnBlockMeta]` near `_apply_pre_transform_translate` (464–471).
  - Add `_process_block(block, df) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]` — the universal driver.
  - Delete the three `_create_*_metas_and_dfs` wrappers (569–582, 756–809, 1108–1115).
  - Rename each transform-body function to `_apply_transform_<type>(block, df, source_block)` and strip `_apply_pre_transform_translate(...)` / `translate_after` handling from each body.
  - Rewrite `_create_new_columns_and_metas` (1220–1227) to call `_process_block` directly (drop the dispatch dict; dispatch happens inside `_apply_transform`).

- **Modify `salk_toolkit/salk_toolkit/validation.py` (MaxDiff):**
  - Delete `choice_mapping` field from `MaxDiffBlock`.
  - Update class docstring to reflect `scale.translate` as the index→name source.

- **Modify `salk_toolkit/tests/test_io.py`:**
  - Update every existing maxdiff test that references `choice_mapping` / `items`: replace with `scale.translate` in the meta, and convert cell data to integer-index form where that shortens the test.
  - Delete `test_maxdiff_single_sibling_rejects_keyed_choice_mapping` (the field no longer exists).
  - Update `test_maxdiff_schema_has_input_format_and_renamed_fields` accordingly.
  - Add four new tests (Task 7) that establish the token-path distinctions crisply.
  - Add one test that `scale.translate_after` on a MaxDiff block is a hard fail.

- **Modify `salk_toolkit/specs/2026-04-20-block-processing-pipeline.md`:** Strike `choice_mapping` from the MaxDiffBlock schema listing (lines 173–181, 204–213). Update the `translate_after` universality claim (lines 142–158) to note the MaxDiff exception.

---

## Task 1 — Add `resolve_role_columns` method and teach `_subgroup_explode` to pre-expand per-sibling

> **Concept:** "Resolving role columns" means taking MaxDiff's regex-valued `best_columns` / `worst_columns` / `set_columns` and, for a given sibling, replacing them with the concrete df-column lists that belong to that sibling. It is the "pre-expand at explode time" step discussed in design. The default on `ColumnBlockMeta` is a no-op — only MaxDiff has additional role-regex fields beyond `from_columns`. After this task, transforms never see regex in those fields.

**Files:**
- Modify: `salk_toolkit/salk_toolkit/validation.py:229-358`
- Modify: `salk_toolkit/salk_toolkit/io.py:512-566`
- Test: `salk_toolkit/tests/test_io.py` (new test in the MaxDiff class)

### Steps

- [ ] **Step 1: Write the failing test — siblings get concrete role-column lists**

Add to `salk_toolkit/tests/test_io.py` in the `TestMaxDiffCreate` (or equivalent) class. Use the existing `meta_file` / `csv_file` fixtures:

```python
def test_maxdiff_explode_resolves_role_columns_per_sibling(self, meta_file, csv_file):
    """After subgroup_explode the source regex in best/worst/set_columns is replaced
    by per-sibling concrete lists; the transform never sees regex."""
    from salk_toolkit.io import _subgroup_explode
    from salk_toolkit.validation import MaxDiffBlock
    df = pd.DataFrame(
        {
            "Q_A_1best": ["x"], "Q_A_1worst": ["x"], "Q_A_1set": [["x"]],
            "Q_A_2best": ["x"], "Q_A_2worst": ["x"], "Q_A_2set": [["x"]],
            "Q_B_1best": ["x"], "Q_B_1worst": ["x"], "Q_B_1set": [["x"]],
        }
    )
    block = MaxDiffBlock(
        name="md",
        from_columns=r"Q_([AB])_\d+best",  # used by explode to enumerate siblings
        best_columns=r"Q_([AB])_(\d+)best",
        worst_columns=r"Q_([AB])_(\d+)worst",
        set_columns=r"Q_\1_\2set",
        choice_mapping={"1": "Alpha"},
    )
    siblings = _subgroup_explode(block, df)
    by_label = {s.name.removeprefix("md_"): s for s in siblings}
    assert set(by_label) == {"A", "B"}
    sib_a = by_label["A"]
    assert isinstance(sib_a.best_columns, list) and sib_a.best_columns == ["Q_A_1best", "Q_A_2best"]
    assert isinstance(sib_a.worst_columns, list) and sib_a.worst_columns == ["Q_A_1worst", "Q_A_2worst"]
    assert isinstance(sib_a.set_columns, list) and sib_a.set_columns == ["Q_A_1set", "Q_A_2set"]
```

- [ ] **Step 2: Run test — verify it fails**

Run:
```
cd salk_toolkit && python -m pytest tests/test_io.py::TestMaxDiffCreate::test_maxdiff_explode_resolves_role_columns_per_sibling -v
```
Expected: FAIL. Today `_narrow_sibling` only sets `from_columns`; `best_columns` stays as the source regex.

- [ ] **Step 3: Add `resolve_role_columns` on `ColumnBlockMeta` (default no-op)**

First, add a `TYPE_CHECKING` block near the top of `salk_toolkit/salk_toolkit/validation.py` (after the `typing` import block at lines 48-66) — `Any` and `Dict` are already imported; pandas is not:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
```

Then append inside `ColumnBlockMeta` (near line 280, before the `@model_serializer`):

```python
    def resolve_role_columns(self, df: "pd.DataFrame", sibling_label: str) -> Dict[str, Any]:
        """Return a dict of field-name -> concrete-list updates that narrow any
        regex-valued column-role fields to this sibling's columns. Default: no roles
        beyond `from_columns` (already handled by `_narrow_sibling`)."""
        return {}
```

- [ ] **Step 4: Override on `MaxDiffBlock`**

Edit `salk_toolkit/salk_toolkit/validation.py:325-358` — add inside the `MaxDiffBlock` class (after `segments`):

```python
    def resolve_role_columns(self, df: "pd.DataFrame", sibling_label: str) -> Dict[str, Any]:
        """Resolve `best_columns` / `worst_columns` / `set_columns` to this sibling's
        concrete df-columns. `set_columns` may be a substitution template
        (`re.Pattern.expand`-style) applied to matched `best_columns`."""
        import re as _re
        updates: Dict[str, Any] = {}

        def _label_match(col: str, patt: "_re.Pattern[str]") -> bool:
            m = patt.match(col)
            return m is not None and (not sibling_label or m.group(1) == sibling_label)

        if isinstance(self.best_columns, str):
            best_re = _re.compile(self.best_columns)
            sib_best = [c for c in df.columns if _label_match(c, best_re)]
            updates["best_columns"] = sib_best
            if isinstance(self.worst_columns, str):
                worst_re = _re.compile(self.worst_columns)
                updates["worst_columns"] = [c for c in df.columns if _label_match(c, worst_re)]
            if isinstance(self.set_columns, str):
                # set_columns may be a substitution template against best_re.
                updates["set_columns"] = [best_re.sub(self.set_columns, c) for c in sib_best]
        return updates
```

- [ ] **Step 5: Teach `_subgroup_explode` to apply role-column narrowing**

Edit `salk_toolkit/salk_toolkit/io.py:523-566`. Replace the current `_subgroup_explode` body's return value with a version that calls `resolve_role_columns` on each sibling. Concretely, append (inside the same function, right before each `return` statement) a final pass that rewrites each sibling:

```python
def _subgroup_explode(block: ColumnBlockMeta, df: pd.DataFrame) -> list[ColumnBlockMeta]:
    matched_cols = _match_columns(block, df)
    pattern = block.from_columns
    if not isinstance(pattern, str):
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    regex = re.compile(pattern)
    first = regex.match(matched_cols[0])
    assert first is not None
    n_groups = len(first.groups())

    agg_idx = getattr(block, "agg_index", None)
    agg_pos = None
    if agg_idx is not None:
        agg_pos = agg_idx - 1 if agg_idx > 0 else agg_idx
        if agg_pos < 0:
            agg_pos = n_groups + agg_pos
        if not (0 <= agg_pos < n_groups):
            raise ValueError(
                f"Block {block.name!r}: agg_index={agg_idx} out of range for {n_groups} capture group(s)"
            )
    non_agg_positions = [i for i in range(n_groups) if i != agg_pos]

    if not non_agg_positions:
        siblings = [_narrow_sibling(block, matched_cols, label_suffix="")]
        return [_apply_role_resolution(s, block, df) for s in siblings]

    def _key(col: str) -> tuple[str, ...]:
        m = regex.match(col)
        assert m is not None
        g = m.groups()
        return tuple(g[i] for i in non_agg_positions)

    sibling_cols: dict[tuple[str, ...], list[str]] = {}
    for c in matched_cols:
        sibling_cols.setdefault(_key(c), []).append(c)

    labels = block.subgroup_labels or {}

    def _label(key: tuple[str, ...]) -> str:
        parts = []
        for val, pos in zip(key, non_agg_positions, strict=True):
            parts.append(str(labels.get(str(pos + 1), {}).get(val, val)))
        return "_".join(parts)

    return [
        _apply_role_resolution(_narrow_sibling(block, cols, label_suffix=_label(key)), block, df)
        for key, cols in sibling_cols.items()
    ]


def _apply_role_resolution(sib: ColumnBlockMeta, source: ColumnBlockMeta, df: pd.DataFrame) -> ColumnBlockMeta:
    """Apply per-type role-column narrowing. Label is derived from sibling vs source name."""
    sib_label = sib.name.removeprefix(source.name).lstrip("_")
    updates = sib.resolve_role_columns(df, sib_label)
    return sib.model_copy(update=updates) if updates else sib
```

- [ ] **Step 6: Run the new test and the existing MaxDiff suite**

Run:
```
cd salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```
Expected: new test PASSES; every existing maxdiff test PASSES. (Regex narrowing in `_create_maxdiff_metas_and_dfs:772-805` is now redundant — still there but dead — which is fine; Task 3 removes it.)

- [ ] **Step 7: Commit**

```bash
git add salk_toolkit/salk_toolkit/validation.py salk_toolkit/salk_toolkit/io.py salk_toolkit/tests/test_io.py
git commit -m "Pre-expand MaxDiff role columns per sibling during explode"
```

(Occurrences of the word "narrow" in the `io.py:772-805` block being retired in Task 2 refer to the *old* in-body resolution logic; the new code is a resolution/expansion step, not a narrowing.)
```

---

## Task 2 — Delete the dead regex-narrowing block in `_create_maxdiff_metas_and_dfs`

**Files:**
- Modify: `salk_toolkit/salk_toolkit/io.py:756-809`

### Steps

- [ ] **Step 1: Run existing maxdiff tests to establish green baseline**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```
Expected: all pass.

- [ ] **Step 2: Delete the regex-narrowing block**

Edit `salk_toolkit/salk_toolkit/io.py:756-809`. Replace the body of `_create_maxdiff_metas_and_dfs` with:

```python
def _create_maxdiff_metas_and_dfs(
    df: pd.DataFrame,
    block: MaxDiffBlock,
    **_: object,
) -> tuple[list[pd.DataFrame], list[MaxDiffBlock]]:
    dfs: list[pd.DataFrame] = []
    metas: list[MaxDiffBlock] = []
    siblings: list[ColumnBlockMeta]
    if block.from_columns is None:
        # No explode: still run role-narrowing so transforms see concrete lists.
        siblings = [_apply_role_resolution(block, block, df)]
    else:
        siblings = _subgroup_explode(block, df)
    for sib in siblings:
        assert isinstance(sib, MaxDiffBlock)
        cs = _pick_subgroup_field(block.choice_sets, sib.name, block.name)
        cm = _pick_subgroup_field(block.choice_mapping, sib.name, block.name)
        sdf, out = _maxdiff_apply_transform(sib, df, cs, cm, source_block=block)
        dfs.append(sdf)
        metas.append(out)
    return dfs, metas
```

- [ ] **Step 3: Run the full test_io suite**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass. If any maxdiff test fails, re-verify Task 1's role-narrowing override handles the failing case's column pattern.

- [ ] **Step 4: Commit**

```bash
git add salk_toolkit/salk_toolkit/io.py
git commit -m "Drop dead regex-narrowing in MaxDiff create; sibling roles are concrete"
```

---

## Task 2b — Simplify `_maxdiff_transform_choice_sets` for the concrete-lists invariant

**Files:**
- Modify: `salk_toolkit/salk_toolkit/io.py:900-969`

**Context:** After Tasks 1 and 2, every `MaxDiffBlock` reaching `_maxdiff_transform_choice_sets` has `best_columns` / `worst_columns` / `set_columns` as concrete `list[str]` (resolved by `resolve_role_columns`). The in-body regex-resolution branch at io.py:930-965 (`if best_is_str: ... best_template = re.compile(...); best_cols = list(filter(...), df.columns))` etc., plus `_expand_set_col` and `_get_group_index` helpers) becomes unreachable and should be deleted. This keeps transform bodies pure data-shape work, not regex resolution.

### Steps

- [ ] **Step 1: Confirm green baseline**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```
Expected: all pass.

- [ ] **Step 2: Add an assertion that roles are concrete lists at entry**

Edit `salk_toolkit/salk_toolkit/io.py:900-919`. After the opening lines of `_maxdiff_transform_choice_sets`, before any branch on `best_is_str`, add:

```python
    if not (isinstance(block.best_columns, list)
            and isinstance(block.worst_columns, list)
            and isinstance(block.set_columns, list)):
        raise TypeError(
            f"_maxdiff_transform_choice_sets expects resolved role columns; got "
            f"best={type(block.best_columns).__name__}, "
            f"worst={type(block.worst_columns).__name__}, "
            f"set={type(block.set_columns).__name__}"
        )
```

Run the suite to prove the assertion holds:
```
cd salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```
Expected: all pass. If any fail on the assertion, the role resolution in Task 1 missed a case — fix there, not here.

- [ ] **Step 3: Delete the regex-resolution branch**

Edit `salk_toolkit/salk_toolkit/io.py:930-969`. Replace the block:

```python
    best_is_str = isinstance(best_cols, str)
    set_is_str = isinstance(set_cols, str)
    if set_cols is None:
        raise ValueError("Maxdiff create blocks must define 'set_columns'.")
    if best_is_str != set_is_str:
        raise ValueError(...)
    if best_is_str:
        best_cols_str = cast(str, best_cols)
        # ... ~25 lines of regex resolution, _expand_set_col, _get_group_index,
        # sorted(best_cols, key=_get_group_index), etc.
    else:
        best_cols = list(best_cols)
        worst_cols = list(worst_cols)
        set_cols = list(set_cols)
```

with:

```python
    if set_cols is None:
        raise ValueError("Maxdiff create blocks must define 'set_columns'.")
    best_cols = list(best_cols)
    worst_cols = list(worst_cols)
    set_cols = list(set_cols)
```

This deletes the `best_template = re.compile(...)` branch, `_expand_set_col`, `_get_group_index`, and the `set_cols = list(map(_expand_set_col, sorted(...)))` line.

- [ ] **Step 4: Run the full test_io suite**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add salk_toolkit/salk_toolkit/io.py
git commit -m "Simplify _maxdiff_transform_choice_sets: roles are always concrete lists"
```

---

## Task 3 — Drop `choice_mapping` from MaxDiffBlock; `scale.translate` is the sole translation mechanism

**Files:**
- Modify: `salk_toolkit/salk_toolkit/validation.py:325-358` (drop field, update docstring)
- Modify: `salk_toolkit/salk_toolkit/io.py:756-1105` (stop reading `choice_mapping`; read `scale.translate`)
- Modify: `salk_toolkit/tests/test_io.py` (update every maxdiff test that used `choice_mapping` / `items`; delete `test_maxdiff_single_sibling_rejects_keyed_choice_mapping`)
- Modify: `salk_toolkit/specs/2026-04-20-block-processing-pipeline.md` (strike `choice_mapping` references)

**Context:** Under the new design, `scale.translate` maps 1-based-index strings (`"1"`, `"2"`, …) to display-language names. That dict serves three jobs at once:
1. Source for the `topics` universe fed to the setindex_column lookup (was `choice_mapping.values()`).
2. Element-wise translator for raw best/worst df cells when they contain index-string values.
3. Element-wise translator for inline set-column cells whether they contain index tokens or name tokens.

Inline set cells **prefer integer-list tokens** (`[1, 2, 3]`) over comma-string tokens (`"1,2,3"`); both forms are accepted by `_tokens_from_value` today and both route through the same translation. All new tests use integer-list form.

### Steps

- [ ] **Step 1: Baseline**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```
Expected: all pass.

- [ ] **Step 2: Remove the `choice_mapping` field from `MaxDiffBlock`**

Edit `salk_toolkit/salk_toolkit/validation.py:325-358`. Delete the line `choice_mapping: Optional[Union[Dict[str, str], Dict[str, Dict[str, str]]]] = None`. Update the class docstring to point users at `scale.translate`:

```python
class MaxDiffBlock(ColumnBlockMeta):
    """Block for MaxDiff best-worst scaling experiments. The stored output
    block is an instance of this class; `best_columns` / `worst_columns` /
    `set_columns` are resolved to `List[str]` by :mod:`salk_toolkit.io`,
    index-aligned by question. Input-only directives are cleared on output.

    Translation: `scale.translate` is a `Dict[str, str]` mapping 1-based-index
    strings (``"1"``, ``"2"``, …) to target-language display names. It is
    applied as the topic universe for ``setindex_column`` lookups AND as an
    element-wise translator for raw best/worst/set cells. `scale.translate_after`
    is not supported on MaxDiff blocks and raises ``ValueError`` at read time.
    """

    type: Literal["maxdiff"] = "maxdiff"  # type: ignore[assignment]

    columns: ColSpec = DF(dict)
    best_columns: Union[str, List[str]]
    worst_columns: Union[str, List[str]]
    set_columns: Optional[Union[str, List[str]]] = None
    setindex_column: Optional[Union[str, List[object]]] = None

    input_format: Literal["choice_sets", "resolved"] = "choice_sets"

    choice_sets: Optional[Union[List[List[List[int]]], Dict[str, List[List[List[int]]]]]] = None
```

Keep `segments()` as-is.

- [ ] **Step 3: Rewrite `_maxdiff_transform_choice_sets` to read `scale.translate`**

Edit `salk_toolkit/salk_toolkit/io.py:900-1105`. The changes:

1. Drop the `choice_mapping` argument from the function signature.
2. Replace the `items = cm; topics = [items[k] for k in sorted(items, key=int)]` block with:

```python
    translate = dict(block.scale.translate) if (block.scale and block.scale.translate) else {}
    if not translate:
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate is required (maps 1-based "
            f"index strings to display names). Got empty translate."
        )
    topics: list[str] = [translate[k] for k in sorted(translate, key=int)]
```

3. Delete the `effective_topics` vs `topics` split entirely. `topics` is now always the display-name list — no second pass. Replace all `effective_topics` uses with `topics`.
4. In `_convert_tokens_to_topics` inner function, index-token path becomes:
   ```python
   converted.append(topics[idx - 1])
   ```
   String-token path still applies `.replace()`-style translation via the outer `translate` dict (for mixed cells where names are stored directly).
5. At the best/worst Categorical build (was lines 1066-1072), drop the explicit `s.map(translate.get)` step. Pre-translate (stage 3, added in Task 5) will have already translated scalar best/worst cells when they hold index strings.

- [ ] **Step 4: Drop the `choice_mapping` arg from callers**

Edit `salk_toolkit/salk_toolkit/io.py:756-809` — `_create_maxdiff_metas_and_dfs`. Remove the `cm = _pick_subgroup_field(block.choice_mapping, …)` line and the `choice_mapping` argument to `_maxdiff_apply_transform`. Similarly in `_maxdiff_apply_transform` (line 812) and `_maxdiff_transform_resolved` (line 871) — drop the arg.

- [ ] **Step 4b: Add model validator rejecting `scale.translate_after` on MaxDiffBlock**

Edit `salk_toolkit/salk_toolkit/validation.py` — inside `MaxDiffBlock`, after field declarations, before `segments()`:

```python
    @model_validator(mode="after")
    def _reject_translate_after(self, info: ValidationInfo) -> Self:
        if self.scale is not None and self.scale.translate_after:
            raise ValueError(
                f"MaxDiffBlock {self.name!r}: scale.translate_after is deprecated for "
                f"maxdiff; use scale.translate (pre-transform) instead."
            )
        return self
```

This triggers at `read_and_process_data` time (during soft_validate) — users get a clear message before any transform runs. Task 7's deprecation test asserts against this ValueError.

- [ ] **Step 5: Update existing maxdiff tests**

Systematically in `salk_toolkit/tests/test_io.py`:

- Replace `"choice_mapping": {...}` (or the older `"items": {...}`) with the equivalent under `"scale": {"translate": {...}}`.
- Where a test previously combined `choice_mapping = {"1": "Ekonomika", ...}` + `translate_after = {"Ekonomika": "Economy", ...}`, collapse to the single dict `scale.translate = {"1": "Economy", ...}` and rewrite raw best/worst cells as index strings (`"1"`, `"2"`, `"3"`) if needed.
- Delete `test_maxdiff_single_sibling_rejects_keyed_choice_mapping` entirely — `choice_mapping` no longer exists, so the rejection test is moot.
- Update `test_maxdiff_schema_has_input_format_and_renamed_fields`: remove the `choice_mapping` presence-assertion; assert the field is **not** on the schema.

- [ ] **Step 6: Update the design spec**

Edit `salk_toolkit/specs/2026-04-20-block-processing-pipeline.md`. In the MaxDiff schema summary (lines ~173–181), strike the `choice_mapping: Dict[str, str]` bullet and replace with:

> - `scale.translate: Dict[str, str]` — 1-based-index string → display name. Required under `input_format="choice_sets"`, optional under `"resolved"`.

Lines 204–213 (multi-subgroup scaffold, keyed form): remove the `choice_mapping` entry from the `MaxDiffBlock` union block (only `choice_sets` remains in that block; `scale.translate` is flat per-block).

In the Translation section (lines 142–158), append:

> **MaxDiff exception:** `scale.translate_after` is not supported on MaxDiff blocks — use `scale.translate` (pre-transform) instead. Writing `translate_after` on a MaxDiff scale raises `ValueError` at read time.

- [ ] **Step 7: Run the full test_io suite**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all existing tests (as updated) pass. If a translate-related test fails it's usually because best/worst cells still hold raw-language names that aren't keys in the new `scale.translate` — fix the test fixture to use index strings or extend the dict.

- [ ] **Step 8: Commit**

```bash
git add salk_toolkit/salk_toolkit/validation.py salk_toolkit/salk_toolkit/io.py salk_toolkit/tests/test_io.py salk_toolkit/specs/2026-04-20-block-processing-pipeline.md
git commit -m "Drop MaxDiff.choice_mapping; scale.translate is the sole index→name source"
```

---

## Task 4 — Add `input_df_columns` method for universal pre-translate

**Files:**
- Modify: `salk_toolkit/salk_toolkit/validation.py`
- Test: `salk_toolkit/tests/test_io.py` (new test)

### Steps

- [ ] **Step 1: Write the failing test**

```python
def test_input_df_columns_topk_onehot_maxdiff(self):
    """Each block type reports the df-columns it reads: topk/onehot use from_columns;
    maxdiff uses the union of best/worst/set."""
    from salk_toolkit.validation import TopKBlock, MaxDiffBlock, OneHotBlock
    df = pd.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})
    tk = TopKBlock(name="t", from_columns=["a", "b"], res_columns=["R1", "R2"])
    assert tk.input_df_columns(df) == ["a", "b"]

    oh = OneHotBlock(name="o", from_columns=["a", "c"])
    assert oh.input_df_columns(df) == ["a", "c"]

    md = MaxDiffBlock(
        name="m",
        best_columns=["a"],
        worst_columns=["b"],
        set_columns=["c"],
        choice_mapping={"1": "Alpha"},
    )
    assert set(md.input_df_columns(df)) == {"a", "b", "c"}
```

- [ ] **Step 2: Run test — verify fail**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k input_df_columns -v
```
Expected: `AttributeError: ... has no attribute 'input_df_columns'`.

- [ ] **Step 3: Add default on `ColumnBlockMeta`**

Edit `salk_toolkit/salk_toolkit/validation.py` — add inside `ColumnBlockMeta`, near `resolve_role_columns`:

```python
    def input_df_columns(self, df: "pd.DataFrame") -> List[str]:
        """Return every df-column this block reads. Default: `from_columns`,
        resolved via regex if necessary."""
        import re as _re
        pattern = self.from_columns
        if pattern is None:
            return [c for c in self.columns.keys() if c in df.columns]
        if isinstance(pattern, list):
            return list(pattern)
        regex = _re.compile(pattern)
        return [c for c in df.columns if regex.match(c)]
```

- [ ] **Step 4: Override on `MaxDiffBlock`**

Add inside `MaxDiffBlock` (after `resolve_role_columns`):

```python
    def input_df_columns(self, df: "pd.DataFrame") -> List[str]:
        """Union of best/worst/set columns; each may be regex or list. `set_columns`
        as a substitution template is treated as derived from `best_columns` and
        already resolved at explode time — non-list values here are a bug."""
        out: list[str] = []

        def _collect(spec: object) -> None:
            if spec is None:
                return
            if isinstance(spec, list):
                out.extend(spec)
                return
            if isinstance(spec, str):
                import re as _re
                r = _re.compile(spec)
                out.extend(c for c in df.columns if r.match(c))
                return
            raise TypeError(f"MaxDiff column role must be str or list; got {type(spec)}")

        _collect(self.best_columns)
        _collect(self.worst_columns)
        _collect(self.set_columns)
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        uniq: list[str] = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq
```

- [ ] **Step 5: Run the test**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k input_df_columns -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add salk_toolkit/salk_toolkit/validation.py salk_toolkit/tests/test_io.py
git commit -m "Add input_df_columns method for universal pre-translate"
```

---

## Task 5 — Introduce `_process_block` driver and move pre-translate out of transforms

**Files:**
- Modify: `salk_toolkit/salk_toolkit/io.py`

### Steps

- [ ] **Step 1: Run the full test_io suite to confirm green baseline**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass.

- [ ] **Step 2: Add `_process_block` driver near `_apply_pre_transform_translate` (io.py:464)**

Insert after `_apply_pre_transform_translate`:

```python
def _process_block(
    block: ColumnBlockMeta, df: pd.DataFrame, **kwargs: object
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Universal driver. Stage 1+2 (match + explode) happens in
    `_subgroup_explode`; stage 3 (pre-translate) happens here; stage 4 (transform)
    is dispatched by block type; stage 5 (post-translate) runs after. Build-output
    still lives inside each transform for now."""
    if isinstance(block, (TopKBlock, MaxDiffBlock, OneHotBlock)):
        if getattr(block, "from_columns", None) is None and isinstance(block, MaxDiffBlock):
            siblings = [_apply_role_resolution(block, block, df)]
        else:
            siblings = _subgroup_explode(block, df)
    else:
        siblings = _subgroup_explode(block, df)

    for sib in siblings:
        cols = sib.input_df_columns(df)
        df_t = _apply_pre_transform_translate(sib, df, cols)
        sdf, meta = _apply_transform(sib, df_t, source_block=block, **kwargs)
        yield sdf, meta


def _apply_transform(
    block: ColumnBlockMeta,
    df: pd.DataFrame,
    *,
    source_block: ColumnBlockMeta,
    **kwargs: object,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Dispatch to the per-type transform. Dispatches on block.type."""
    if isinstance(block, TopKBlock):
        assert isinstance(source_block, TopKBlock)
        source_pattern = source_block.from_columns if isinstance(source_block.from_columns, str) else None
        return _topk_apply_transform(block, df, source_pattern=source_pattern, source_block=source_block)
    if isinstance(block, MaxDiffBlock):
        assert isinstance(source_block, MaxDiffBlock)
        cs = _pick_subgroup_field(source_block.choice_sets, block.name, source_block.name)
        cm = _pick_subgroup_field(source_block.choice_mapping, block.name, source_block.name)
        return _maxdiff_apply_transform(block, df, cs, cm, source_block=source_block)
    if isinstance(block, OneHotBlock):
        assert isinstance(source_block, OneHotBlock)
        chs = source_block.choices if isinstance(source_block.choices, list) or source_block.choices is None else list(source_block.choices)
        return _onehot_apply_transform(block, df, chs)
    raise TypeError(f"Unsupported block type for _apply_transform: {type(block)}")
```

- [ ] **Step 3: Strip `_apply_pre_transform_translate` calls out of every transform body**

Edit each listed site — remove the `df = _apply_pre_transform_translate(block, df, ...)` line (no replacement; `_process_block` has already translated):
- `salk_toolkit/salk_toolkit/io.py:605` (inside `_topk_transform_skip`)
- `salk_toolkit/salk_toolkit/io.py:645` (inside `_topk_transform_onehot`)
- `salk_toolkit/salk_toolkit/io.py:880` (inside `_maxdiff_transform_resolved`)
- `salk_toolkit/salk_toolkit/io.py:1033` (inside `_maxdiff_transform_choice_sets`)
- `salk_toolkit/salk_toolkit/io.py:1136` (inside `_onehot_transform_leftpacked`)
- `salk_toolkit/salk_toolkit/io.py:1183` (inside `_onehot_transform_wide`)

- [ ] **Step 4: Rewrite `_create_new_columns_and_metas` to call `_process_block`**

Replace `salk_toolkit/salk_toolkit/io.py:1220-1227`:

```python
def _create_new_columns_and_metas(
    df: pd.DataFrame, group: ColumnBlockMeta, **kwargs: dict[str, str]
) -> Iterator[tuple[pd.DataFrame, ColumnBlockMeta]]:
    """Create new columns and metadata from a specialized block."""
    if not isinstance(group, (TopKBlock, MaxDiffBlock, OneHotBlock)):
        raise ValueError("Group must be a TopKBlock, MaxDiffBlock, or OneHotBlock to generate new columns")
    return _process_block(df=df, block=group, **kwargs)
```

Delete `create_block_type_to_create_fn` and the three `_create_*_metas_and_dfs` wrappers (now unused): lines 569-582, 756-809 as rewritten in Task 2, and 1108-1115. `_process_block` has absorbed all three.

- [ ] **Step 5: Run the full test_io suite**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass.

- [ ] **Step 6: Run ruff and pyright**

```
cd salk_toolkit && ruff check && pyright salk_toolkit
```
Expected: clean (or no new issues vs. baseline).

- [ ] **Step 7: Commit**

```bash
git add salk_toolkit/salk_toolkit/io.py
git commit -m "Introduce _process_block driver; pre-translate is now universal"
```

---

## Task 6 — Extract `translate_after` into a universal stage-5; MaxDiff rejects it

**Files:**
- Modify: `salk_toolkit/salk_toolkit/io.py`
- Test: `salk_toolkit/tests/test_io.py` (no changes expected; existing tests enforce behavior parity)

### Steps

- [ ] **Step 1: Run full test_io suite (baseline)**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass.

- [ ] **Step 2: Add `_apply_post_transform_translate`**

Insert in `salk_toolkit/salk_toolkit/io.py` after `_apply_pre_transform_translate`:

```python
def _apply_post_transform_translate(
    block: ColumnBlockMeta,
    sdf: pd.DataFrame,
    meta: ColumnBlockMeta,
) -> tuple[pd.DataFrame, ColumnBlockMeta]:
    """Universal stage 5. If `scale.translate_after` is set on the *output* meta:
    - scalar-valued cells: `.replace(translate_after)` per output column.
    - list-valued cells (e.g. MaxDiff set columns): map elements element-wise.
    - Rewrite `meta.scale.categories` to `list(translate_after.values())` unless
      `categories` is already set to something compatible.
    - For columns declared as `pd.Categorical` with categories matching the raw
      topic list, rebuild the Categorical with translated categories in the same
      order.
    Blocks with no scale or no translate_after pass through unchanged.
    MaxDiff blocks reject translate_after at pre-transform via
    `scale.translate_after` validation (see Task 3); this function will see
    `scale.translate_after` is unset for MaxDiff outputs in practice."""
    scale = meta.scale
    if scale is None or not scale.translate_after:
        return sdf, meta
    if isinstance(block, MaxDiffBlock):
        raise ValueError(
            f"MaxDiffBlock {block.name!r}: scale.translate_after is deprecated for "
            f"maxdiff; use scale.translate (pre-transform) instead."
        )
    t = dict(scale.translate_after)

    def _map_scalar(v: object) -> object:
        if isinstance(v, str):
            return t.get(v, v)
        return v

    for col in sdf.columns:
        s = sdf[col]
        if _is_series_of_lists(s):
            sdf[col] = s.map(
                lambda lst, _t=t: None
                if lst is None or (isinstance(lst, float) and pd.isna(lst))
                else [_t.get(x, x) if isinstance(x, str) else x for x in lst]
            )
        elif isinstance(s.dtype, pd.CategoricalDtype):
            new_cats = [t.get(c, c) if isinstance(c, str) else c for c in s.cat.categories]
            sdf[col] = s.cat.rename_categories(new_cats)
        else:
            sdf[col] = s.map(_map_scalar)

    scale_dict = scale.model_dump(mode="python")
    if not scale_dict.get("categories") or scale_dict.get("categories") == "infer":
        scale_dict["categories"] = list(dict.fromkeys(t.values()))
    else:
        scale_dict["categories"] = [t.get(c, c) if isinstance(c, str) else c for c in scale_dict["categories"]]
    meta_out = meta.model_copy(update={"scale": type(scale).model_validate(scale_dict)})
    return sdf, meta_out
```

- [ ] **Step 3: Call `_apply_post_transform_translate` from `_process_block`**

Edit `_process_block` (added in Task 5). The loop becomes:

```python
    for sib in siblings:
        cols = sib.input_df_columns(df)
        df_t = _apply_pre_transform_translate(sib, df, cols)
        sdf, meta = _apply_transform(sib, df_t, source_block=block, **kwargs)
        sdf, meta = _apply_post_transform_translate(sib, sdf, meta)
        yield sdf, meta
```

- [ ] **Step 4: Strip `translate_after` handling from `_topk_transform_onehot`**

Edit `salk_toolkit/salk_toolkit/io.py:706-714`. Remove the `translate_values = block.scale.translate_after ...` block and everything it drives (the `sdf.replace(translate_values)`, the loop rebuilding Categoricals, and the `scale_dict["categories"] = effective_cats` line). The downstream `_build_topk_output_block` must continue to work without these — if it relied on translated categories being in `scale_dict`, move that reliance: `_apply_post_transform_translate` will overwrite `scale.categories` at stage 5.

Expected shape of the edit — after the `sdf = sdf.iloc[:, :kmax]` line, replace lines 706-714 with just:

```python
    scale_dict = deepcopy(block.scale.model_dump(mode="python") if block.scale else {})
```

(Leave the rest of the function unchanged; `_build_topk_output_block` continues to consume `scale_dict`.)

- [ ] **Step 5: Strip `translate_after` handling from `_maxdiff_transform_choice_sets`**

Edit `salk_toolkit/salk_toolkit/io.py:971-1092`. This is the most intricate removal. Specifically:

- Delete lines 971-972 (`_t = block.scale.translate_after ...; translate = dict(_t) if _t else {}`).
- In `_convert_tokens_to_topics` (lines 1008-1030), replace `translate.get(t, t)` at line 1017 with `t`:
  ```python
  if all(isinstance(token, str) for token in tokens) and not all(_is_int_like(token) for token in tokens):
      stripped = [token.strip() for token in tokens]
      return stripped  # translate_after runs at stage 5
  ```
- At line 1036, replace the translated topics build with raw topics:
  ```python
  effective_topics: list[str] | None = list(topics) if topics is not None else None
  ```
- At line 1042, the setindex path uses `effective_topics` which is now raw — fine; stage 5 translates them element-wise across list-valued cells.
- At lines 1066-1072, remove the `if translate: s = s.map(...)` branch; keep the `pd.Categorical(s, categories=effective_topics)` line. Stage 5's Categorical-rename branch will translate categories and cells together.
- At line 1090, keep `scale_dict["categories"] = effective_topics` (stage 5 will rewrite it to translated values).

- [ ] **Step 6: Verify onehot transforms don't touch `translate_after`**

Grep:
```
grep -n translate_after salk_toolkit/salk_toolkit/io.py
```
Expected: hits only in `_apply_pre_transform_translate`, `_apply_post_transform_translate`, and possibly comments — NOT inside `_topk_transform_*`, `_maxdiff_transform_*`, or `_onehot_transform_*`.

- [ ] **Step 7: Run the full test_io suite**

```
cd salk_toolkit && python -m pytest tests/test_io.py -v
```
Expected: all pass. If `test_maxdiff_with_translate` fails, debug by comparing intermediate state: before stage 5, cells should be raw topic names; after, translated. If cell values are translated but `scale.categories` shows raw names, the scale-rewrite branch in `_apply_post_transform_translate` is wrong.

- [ ] **Step 8: Run ruff and pyright**

```
cd salk_toolkit && ruff check && pyright salk_toolkit
```
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add salk_toolkit/salk_toolkit/io.py
git commit -m "Extract translate_after as universal stage-5 post-translate"
```

---

## Task 7 — Four comprehensive MaxDiff tests exposing the token-path and translate distinctions

**Files:**
- Test: `salk_toolkit/tests/test_io.py`

**Context:** These four tests establish the MaxDiff contract crisply under the new design. Add them to the `TestMaxDiffCreate` class (or equivalent). Prefer integer-list tokens for inline set cells (`[1, 2, 3]`) — the tests assert that form works and the implementation supports it.

### Steps

- [ ] **Step 1: Add the four tests to `salk_toolkit/tests/test_io.py`**

```python
def test_maxdiff_inline_index_tokens(self, meta_file, csv_file):
    """Inline MaxDiff with integer-list tokens in set cells and index strings in
    best/worst cells. scale.translate maps index strings to display names."""
    meta = {
        "file": "test.csv",
        "structure": [{
            "type": "maxdiff",
            "name": "md",
            "columns": [],
            "best_columns": ["Q_1best"],
            "worst_columns": ["Q_1worst"],
            "set_columns": ["Q_1set"],
            "scale": {
                "categories": "infer",
                "translate": {"1": "Economy", "2": "Health", "3": "Education"},
            },
        }],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q_1best":  ["1", "3", "2"],
        "Q_1worst": ["2", "1", "1"],
        "Q_1set":   [[1, 2, 3], [1, 2, 3], [1, 2, 3]],  # integer-list tokens
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    assert data_df["Q_1best"].tolist()  == ["Economy", "Education", "Health"]
    assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]
    assert list(data_df["Q_1set"].iloc[0]) == ["Economy", "Health", "Education"]
    block = data_meta.structure["md"]
    assert set(block.scale.categories or []) == {"Economy", "Health", "Education"}


def test_maxdiff_inline_name_tokens(self, meta_file, csv_file):
    """Inline MaxDiff with raw-language names in best/worst and set cells.
    scale.translate maps raw names to display names — same dict-key space
    as cell contents, NOT integer positions."""
    meta = {
        "file": "test.csv",
        "structure": [{
            "type": "maxdiff",
            "name": "md",
            "columns": [],
            "best_columns": ["Q_1best"],
            "worst_columns": ["Q_1worst"],
            "set_columns": ["Q_1set"],
            "scale": {
                "categories": "infer",
                "translate": {
                    "Ekonomika": "Economy",
                    "Sveikata":  "Health",
                    "Svietimas": "Education",
                },
            },
        }],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q_1best":  ["Ekonomika", "Svietimas", "Sveikata"],
        "Q_1worst": ["Sveikata",  "Ekonomika", "Ekonomika"],
        "Q_1set": [
            ["Ekonomika", "Sveikata", "Svietimas"],
            ["Ekonomika", "Sveikata", "Svietimas"],
            ["Ekonomika", "Sveikata", "Svietimas"],
        ],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    assert data_df["Q_1best"].tolist()  == ["Economy", "Education", "Health"]
    assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]
    assert list(data_df["Q_1set"].iloc[0]) == ["Economy", "Health", "Education"]


def test_maxdiff_setindex_lookup(self, meta_file, csv_file):
    """MaxDiff driven by setindex_column + choice_sets metadata. scale.translate
    replaces the old choice_mapping: it's both the index→name source for the
    setindex lookup AND the cell translator for best/worst."""
    choice_sets = [
        [[1, 2, 3], [2, 3, 1], [1, 3, 2]],  # version 1
        [[3, 1, 2], [1, 2, 3], [3, 2, 1]],  # version 2
    ]
    meta = {
        "file": "test.csv",
        "structure": [{
            "type": "maxdiff",
            "name": "md",
            "columns": [],
            "best_columns":    r"Q_(\d+)best",
            "worst_columns":   r"Q_(\d+)worst",
            "set_columns":     r"Q_\1set",
            "setindex_column": ["Q_Version", {"continuous": True}],
            "choice_sets":     choice_sets,
            "scale": {
                "categories": "infer",
                "translate": {"1": "Economy", "2": "Health", "3": "Education"},
            },
        }],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q_Version": [1, 2, 1],
        "Q_1best":   ["1", "3", "2"],
        "Q_1worst":  ["2", "1", "1"],
        "Q_2best":   ["3", "1", "3"],
        "Q_2worst":  ["1", "2", "1"],
        "Q_3best":   ["1", "3", "1"],
        "Q_3worst":  ["3", "1", "3"],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # row 0, version 1, Q_1 → choice_sets[0][0] = [1,2,3] → [Economy, Health, Education]
    # row 1, version 2, Q_1 → choice_sets[1][0] = [3,1,2] → [Education, Economy, Health]
    assert list(data_df["Q_1set"].iloc[0]) == ["Economy", "Health", "Education"]
    assert list(data_df["Q_1set"].iloc[1]) == ["Education", "Economy", "Health"]
    assert data_df["Q_1best"].tolist()  == ["Economy", "Education", "Health"]
    assert data_df["Q_1worst"].tolist() == ["Health", "Economy", "Economy"]

    block = data_meta.structure["md"]
    assert block.choice_sets is None  # input-only, cleared on output
    assert isinstance(block.best_columns, list)
    assert set(block.scale.categories or []) == {"Economy", "Health", "Education"}


def test_maxdiff_translate_after_is_deprecated(self, meta_file, csv_file):
    """scale.translate_after on a MaxDiff block must be a hard fail with a
    message pointing at scale.translate."""
    meta = {
        "file": "test.csv",
        "structure": [{
            "type": "maxdiff",
            "name": "md",
            "columns": [],
            "best_columns":  ["Q_1best"],
            "worst_columns": ["Q_1worst"],
            "set_columns":   ["Q_1set"],
            "scale": {"translate_after": {"1": "Economy"}},
        }],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({"Q_1best": ["1"], "Q_1worst": ["1"], "Q_1set": [[1]]})
    df.to_csv_file(csv_file)

    with pytest.raises(ValueError, match="translate_after.*deprecated.*maxdiff.*scale.translate"):
        read_and_process_data(str(meta_file), return_meta=True)
```

- [ ] **Step 2: Run the four tests**

```
cd salk_toolkit && python -m pytest tests/test_io.py -k "maxdiff_inline_index_tokens or maxdiff_inline_name_tokens or maxdiff_setindex_lookup or maxdiff_translate_after_is_deprecated" -v
```
Expected: all four PASS. If the `translate_after_is_deprecated` test fails, confirm Task 6 added the MaxDiff branch in `_apply_post_transform_translate` (or a `model_validator` on `MaxDiffBlock`) that raises.

- [ ] **Step 3: Commit**

```bash
git add salk_toolkit/tests/test_io.py
git commit -m "MaxDiff: four tests pinning token paths, setindex lookup, translate_after deprecation"
```

---

## Task 8 — Final sweep: ruff, pyright, reference plots, and full test suite

**Files:**
- None (verification only)

### Steps

- [ ] **Step 1: Run ruff**

```
cd salk_toolkit && ruff check
```
Expected: clean.

- [ ] **Step 2: Run pyright**

```
cd salk_toolkit && pyright salk_toolkit
```
Expected: no new errors vs. main.

- [ ] **Step 3: Run the full pytest suite**

```
cd salk_toolkit && python -m pytest tests/ -v
```
Expected: all pass. Plot-reference tests in `test_plots.py` must pass without `--recompute` — if any fail, that means a behavior change leaked into plot output. Investigate before recomputing references.

- [ ] **Step 4: Confirm no stray references to deleted helpers**

```
grep -n "_create_topk_metas_and_dfs\|_create_maxdiff_metas_and_dfs\|_create_onehot_metas_and_dfs\|create_block_type_to_create_fn" salk_toolkit/salk_toolkit/
```
Expected: no hits (helpers fully replaced by `_process_block`).

- [ ] **Step 5: Confirm `_apply_pre_transform_translate` is only called from one site**

```
grep -rn "_apply_pre_transform_translate" salk_toolkit/salk_toolkit/
```
Expected: only the definition plus one call inside `_process_block`.

- [ ] **Step 6: If clean, no final commit needed — previous commits land the work.**

---

## Self-Review Notes

- **Spec coverage** — every delta in `specs/2026-04-20-block-processing-pipeline.md` lines 19–46 (the four-stage decomposition) is addressed: Task 1+2 finishes stage 2 (explode now hands out fully-concrete siblings across all role fields), Task 5 adds the driver and moves stage 3 out of transform bodies, Task 6 adds stage 5 (`translate_after`) and strips translate handling from transform bodies. Stage 4 (`_build_output_block`) is not factored out by this plan — that's a follow-up, since the current `soft_validate({...}, <BlockClass>)` calls inside each transform do not duplicate meaningful logic across types and the spec does not require extraction.
- **User-facing changes** — two, both deliberate: `MaxDiffBlock.choice_mapping` is removed (migrate to `scale.translate`), and `scale.translate_after` on MaxDiff blocks now raises `ValueError`. TopK/OneHot/plain keep `translate_after` semantics unchanged.
- **Risk points to watch during execution:**
  - Categorical rename in `_apply_post_transform_translate` (Task 6, step 2) must preserve category order; `rename_categories(new_cats)` does, but verify behavior with the rewritten `test_maxdiff_with_translate`-descendant tests.
  - `_is_series_of_lists` checks only the first non-null element — fine for this data but worth noting if tests introduce mixed-type columns.
  - `_process_block` loses the existing special-case `if block.from_columns is None: siblings = [block]` for MaxDiff; the replacement wraps `block` in `_apply_role_resolution(block, block, df)` so role regex still resolves. If a non-explode MaxDiff test fails, check that branch.

---

## Execution Handoff

Plan complete and saved to `salk_toolkit/specs/2026-04-23-unified-block-processing-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
