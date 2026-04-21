# Block Processing Pipeline

Internal decomposition of annotated-data block processing into four ordered
stages. The user-facing surface stays at the level of block types (`plain`,
`topk`, `maxdiff`, `onehot`); primitives are never written by the user.

## Motivation

`_create_topk_metas_and_dfs` and `_create_maxdiff_metas_and_dfs` today bundle
three concerns into each function: regex-based subgroup explosion,
column-shape transformation, and obs-model structure hints. Adding a new
block type (e.g. `onehot` for the am_data Q12 social-media case) duplicates
all three. This spec factors the concerns internally, generalises
`subgroup_explode` to every block type, and lets `input_format` express
"upstream ETL already did the transform, skip it" for `topk` and `maxdiff`.

## The pipeline (internal only)

Every block's processing decomposes into four ordered stages:

```
match_columns     (regex on from_columns, or explicit list)
  → subgroup_explode   (universal: 1 source block → N sibling blocks of same type)
  → _apply_transform   (0 or 1 transform; dispatched by type + input_format)
  → _build_output_block (resolved block carries obs-model hints via segments())
```

This decomposition is **not** user-visible. The user writes a block type
plus the relevant fields; each type has a fixed desugaring dispatched in
`io.py`. There is no user-writable pipeline field and no `transform[]`
array.

Helper signatures:

```python
_match_columns(block, df)      -> list[str]
_subgroup_explode(block, df)   -> list[Block]              # same type as input; narrowed
_apply_transform(block, df)    -> tuple[pd.DataFrame, Block]  # Block is output-shape
_build_output_block(block, …)  -> Block                    # finalise + attach segments()
```

Siblings from `_subgroup_explode` are instances of the same block class as
the source (e.g. `TopKBlock` in, `list[TopKBlock]` out), each with
narrowed `name` / `from_columns` / `columns`. Stage 3 never sees the
subgroup machinery — it receives one single-subgroup block at a time.

## Stage 1 — match_columns

Regex on `from_columns`, or an explicit column list. Unchanged from today
other than `from_columns` moving up to `ColumnBlockMeta` so every block
type can use it.

## Stage 2 — subgroup_explode (universal)

When `from_columns` is a regex with more than one capture group, the
non-aggregation captures define subgroups; one source block produces one
sibling block per Cartesian combination of non-agg capture values that
actually appear in matched columns.

Only one per-type hook:

```python
# TopK aggregates columns over the subgroup, should not explode
agg_idx = getattr(block, "agg_index", None)
```

`agg_index` is consumed by the explode helper and excluded from the label
suffix. Every other non-agg capture contributes one label slot.

### Fields controlling explode (at `ColumnBlockMeta` level)

- `from_columns: Optional[Union[str, List[str]]] = None` — regex or list.
- `subgroup_labels: Optional[Dict[str, Dict[str, str]]] = None` — 1-based
  regex-group index (as string) → capture value → human label. Used in
  generated sibling names. Not a translation — these labels identify
  *which* question a sibling represents.

```json
"subgroup_labels": {
  "1": {"1": "economics", "2": "healthcare", "3": "education"}
}
```

### Sibling naming (universal)

Sibling block name = `block.name` + `_` + non-agg labels joined with `_`.

Label resolution per capture (left-to-right, skipping `agg_index`):

1. `subgroup_labels["<1-based-group-idx>"][<capture_value>]` if present.
2. Otherwise the raw capture value, stringified.

For a block producing a single sibling (no non-agg captures, or explicit
list input) the output name is just `block.name` — no `_type` suffix, no
underscore-trailing name. This is a **break from current behaviour**,
where `TopKBlock` outputs were named `{block.name}_topk_{labels}`.

Example (leedu issue-ownership):

```text
from_columns="Q7r(\\d+)c(\\d+)", agg_index=2,
subgroup_labels={"1": {"1":"economics","2":"healthcare",...}}
  → siblings: issue_ownership_economics, issue_ownership_healthcare, …
```

## Stage 3 — transform (1:1 with block type)

| Block type | Default transform | Purpose |
|---|---|---|
| `plain` *(default)* | none | columns pass through unchanged |
| `topk` | one-hot → left-packed rank columns (agg over `agg_index`) | rank/multi-select collapse |
| `maxdiff` | resolve `choice_sets` via `setindex_column` into per-row set columns | set resolution |
| `onehot` | left-packed → one bool column per item | widen to per-item booleans |

Transform parameters live as normal fields on the block class. The block
type discriminator picks the transform function in `io.py`.

### `input_format` — skip the transform when the data is already in shape

`input_format` is a per-block-type `Literal` field declaring the shape of
incoming data. When input is already the transform's output shape, the
transform is skipped; explode and structure still run.

| Block type | `input_format` | Meaning | Rank-bearing? |
|---|---|---|---|
| `topk` | `"onehot"` *(default)* | Per-choice selection cells (strings, or `"selected"`/`"NO TO: …"`). Leftpack runs. | no |
| `topk` | `"ranked_onehot"` *(scaffold)* | Cells carry rank integers (`1` = first pick, `2` = second, `0`/NA = unselected). Leftpack sorts by cell rank. | yes |
| `topk` | `"leftpacked"` | Raw cells already in `R1..Rk`, arbitrary column order. Transform skipped; `res_columns` must match existing names. | no |
| `topk` | `"ranked_leftpack"` | Same shape as `"leftpacked"`; column position encodes rank (R1 = first pick …). Transform skipped. | yes (column-position rank) |
| `maxdiff` | `"choice_sets"` *(default)* | Best/worst cells + `choice_sets` + `setindex_column`. Set resolution runs. | yes (intrinsic) |
| `maxdiff` | `"resolved"` | `best_columns` / `worst_columns` / `set_columns` are each independent regexes (aligned by capture value) or explicit aligned `List[str]`. Transform skipped. | yes |
| `onehot` | `"leftpacked"` *(default)* | Rank-position columns carrying item codes (e.g. `vQ12_M_1..7`). Explode runs. | no |
| `onehot` | `"wide"` | Already one bool column per item. Transform skipped. | no |

`plain` has no transform and no `input_format` field.

`ranked_onehot` is scaffolded: schema + `segments()` shape are specified
so callers don't need to special-case it; the transform body raises
`NotImplementedError("ranked_onehot transform not yet implemented — no
production user")`. Wired up for real when a user appears.

### Translation (universal, scale-level)

`ColumnMeta` / `BlockScaleMeta` already carry `translate` and
`translate_after`:

- `translate` — applied to raw cell values **before** the transform
  runs.
- `translate_after` — applied to cell values **after** the transform
  produces output.

Both are the primary mechanism for all block types:

- `TopKBlock.translate_values` is **removed**. Users write
  `scale.translate_after` to map leftpacked cell values to display names.
- `MaxDiffBlock.translate` is **removed**. Users write `scale.translate`
  (pre-transform, e.g. to normalise input_format=`"resolved"` cells) or
  `scale.translate_after` (post-transform).

When `translate_after` is present and `scale.categories` is unset,
categories are derived from `translate_after.values()`.

### Per-block-type schema summary

**`TopKBlock`:**
- `k`, `agg_index`, `res_columns`, `na_vals`, `from_prefix` — as today.
- `input_format: Literal["onehot", "ranked_onehot", "leftpacked", "ranked_leftpack"] = "onehot"`.
- `groups` removed → universal `subgroup_labels`.
- `translate_values` removed → use `scale.translate_after`.
- No `ordered` override, no `tie_policy` enum.

**`MaxDiffBlock`:**
- `best_columns`, `worst_columns`, `set_columns`, `setindex_column` — as today.
- `input_format: Literal["choice_sets", "resolved"] = "choice_sets"`.
- `choice_mapping: Dict[str, str]` — renamed from `items`. Maps 1-based
  item index (as string) to item value as it appears in raw data cells.
  Required under `"choice_sets"`; optional (validation-only) under
  `"resolved"`.
- `choice_sets: List[List[List[int]]]` for single-subgroup blocks
  (flat form, today's shape).
- `translate` removed → use `scale.translate_after`.

**`OneHotBlock` (new):**
- `type: Literal["onehot"]`.
- `input_format: Literal["leftpacked", "wide"] = "leftpacked"`.
- `choices: Optional[List[str]] = None` — item list. When `None`,
  derived as the sorted union of unique non-null cell values across all
  matched columns (excluding `na_vals`).
- `res_prefix: Optional[str] = None` — prefix for generated output
  column names (`f"{res_prefix}{choice}"`). Default derived from block
  name.
- `na_vals: Optional[List[str]] = None`.
- No `segments()`; autodetect handles the output as per-column
  Categorical/boolean.

### Multi-subgroup scaffold — `maxdiff` and `onehot`

`maxdiff` is currently single-subgroup in production. For the future
case of multi-subgroup `maxdiff` (and analogously for `onehot` when the
unique-value set differs per subgroup), per-subgroup configuration is
expressed as a dict keyed by sibling label:

```python
# MaxDiffBlock
choice_sets: Optional[Union[
    List[List[List[int]]],                     # flat: single subgroup
    Dict[str, List[List[List[int]]]]           # keyed by sibling label
]]
choice_mapping: Optional[Union[
    Dict[str, str],                            # flat: single subgroup
    Dict[str, Dict[str, str]]                  # keyed by sibling label
]]

# OneHotBlock
choices: Optional[Union[
    List[str],                                 # flat
    Dict[str, List[str]]                       # keyed by sibling label
]]
```

**Strict rule (no broadcasting):**

- Explode produces 1 sibling → only the flat form is valid. Passing the
  keyed form is a hard fail.
- Explode produces N>1 siblings → only the keyed form is valid. Every
  sibling label must appear as a key; passing the flat form, or omitting
  any sibling's key, is a hard fail.

Selection happens at stage 3 dispatch, per sibling. Explode itself has
no maxdiff-specific knowledge.

### Error handling (hard fails)

- `topk`, any format: truncation to `k` that would drop a non-NA cell
  → hard fail (not silent truncate).
- `topk input_format="leftpacked"` / `"ranked_leftpack"`: `res_columns`
  must match existing column names; mismatch → hard fail.
- `topk input_format="ranked_onehot"`: two cells with the same rank
  within one row → hard fail; non-integer or negative rank cells → hard
  fail; gaps compressed with warning; overflow truncated with warning.
- `onehot`: observed cell value outside `choices` (and outside
  `na_vals`) when `choices` is explicit → hard fail.
- Empty matched-columns set after regex match → hard fail naming the
  regex.
- Maxdiff subgroup fields: see "strict rule" above.

## Stage 4 — give_structure

The output block carries enough metadata for downstream consumers
(`AutodetectOM.prepare` in
`salk_internal_package/obs_models/autodetect.py`) to pick the right
observation model. Wiring AutodetectOM itself is **out of scope** for
this spec — the toolkit side exposes `segments()` of the right shape and
AutodetectOM's `prepare()` extension lives in the salk_internal_package
repo.

`segments()` returns a list of `(a, b, ordered)` triples consumed by
`OrdinalRanking`:

- `a`, `b` are lists of column names on the output block.
- `b = None` means "`a` is the selected set out of the full agg-axis
  universe" — unordered multi-select.
- `ordered = True` means every column in `a` is ranked strictly above
  every column in `b`.

Per block type:

| Block | `input_format` | `segments()` |
|---|---|---|
| `plain` | — | not defined; autodetect → Categorical/Continuous |
| `onehot` | any | not defined; autodetect per-column |
| `topk` | `onehot` / `leftpacked` | `[(all_R_cols, None, False)]` — unordered multi-select |
| `topk` | `ranked_onehot` / `ranked_leftpack` | ordered chain: `[([R1],[R2..Rk],True), ([R2],[R3..Rk],True), …, (all_R_cols, None, False)]` |
| `maxdiff` | any | `[([best_k],[set_k],True), ([set_k],[worst_k],True)]` per question, concatenated |

`segments()` reads `self.input_format`; callers flip format and the
right segment shape follows automatically.

## Adding a new block type

A new block type, not a new primitive. Steps:

1. Add a Pydantic class (e.g. `OneHotBlock`) inheriting
   `ColumnBlockMeta`, with a `type: Literal["…"]` discriminator and
   its own `input_format` enum if it admits more than one input shape.
2. Register a transform function and a build function under the type
   discriminator in `io.py`.
3. Add `segments()` only if the output needs non-default structure.

The universal `subgroup_explode` path and universal translate pipeline
apply automatically.

## Tests

`tests/test_io.py` additions (keeping existing golden cases intact by
adding explicit `input_format` where today's defaults apply):

**Explode:**
- Plain multi-capture regex → N siblings, correct labels, no
  `_{type}` suffix.
- TopK single non-agg capture → N siblings.
- TopK with explicit list → 1 sibling, bare block name.

**TopK transform:**
- `input_format="onehot"` — existing golden.
- `input_format="leftpacked"` — transform skipped; res_columns validated;
  mismatch → hard fail test.
- `input_format="ranked_leftpack"` — transform skipped; `segments()`
  returns the ordered chain.
- `input_format="ranked_onehot"` — `NotImplementedError`.
- Truncation to `k` that drops non-NA cells → hard fail.

**MaxDiff transform:**
- `input_format="choice_sets"` — existing golden, renamed fields.
- `input_format="resolved"` with three independent regexes aligned by
  capture value.
- Flat `choice_sets` + N>1 siblings → hard fail.
- Dict `choice_sets` + N=1 sibling → hard fail.
- Dict `choice_sets` with a missing sibling key → hard fail.

**OneHot:**
- `input_format="leftpacked"` with explicit `choices`.
- `input_format="leftpacked"` with `choices=None` (derive from cells).
- `input_format="wide"` — pass-through.
- Observed value outside explicit `choices` → hard fail.

**Structure (`segments()`):**
- TopK unranked / ranked shapes.
- MaxDiff best>set>worst pairs.
- OneHot / plain — method absent.

**Translate:**
- `scale.translate` applied pre-transform (maxdiff `"resolved"`,
  topk `"leftpacked"`).
- `scale.translate_after` applied post-transform (topk `"onehot"`,
  maxdiff `"choice_sets"`).
- Categories derived from `translate_after.values()` when unset.

## Migration

See `specs/2026-04-20-block-processing-migration.md` for the full
field-by-field rewrite guide (covers rename of `groups`,
`translate_values`, `items`, `translate`, and the sibling-name change
that drops the `_{type}` suffix). That document is the reference when
pointing an LLM at old annotation files to rewrite.

## Non-goals

- No user-exposed pipeline or primitive vocabulary.
- No `transform[]` array.
- No named-transform dispatch separate from block type.
- No exposure of `segments()` as a user-written field.
- No `ordered` override flag on `TopKBlock` — `input_format` carries
  rank-bearing information for both topk and maxdiff.
- No `tie_policy` enum — ties in `ranked_onehot` are always a hard fail.
- Wiring `AutodetectOM` to call `block.segments()` is tracked in the
  salk_internal_package repo, not here.
