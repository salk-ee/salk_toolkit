# Block Processing Pipeline

How salk_toolkit decomposes annotated-data block processing into ordered
stages, and why the user-facing surface stays at the level of block types
(`plain`, `topk`, `maxdiff`) rather than exposing the primitives.

## Motivation

Today, TopK and MaxDiff bundle three concerns: regex-based subgroup
explosion, column-shape transformation, and obs-model structure hints.
Adding a new block type (e.g. `onehot` for the am_data Q12 social
media case) duplicates all three. This spec separates the concerns
internally while keeping the user-facing surface simple.

## The pipeline (internal only)

Every block's processing decomposes into four ordered stages:

```
match_columns   (regex or list)
  → subgroup_explode   (optional: 1 source block → N sibling blocks)
  → transform          (0 or 1 transform; fixed by block type)
  → give_structure     (resolved block carries obs-model hints)
```

**This decomposition is internal.** The user writes a block type plus
the relevant fields; each block type has a fixed desugaring. There is no
user-writable pipeline field and no `transform[]` array.

## Stage 1 — match_columns

Regex pattern on `from_columns`, or an explicit column list. Unchanged
from today.

## Stage 2 — subgroup_explode

When `from_columns` is a regex with more than one capture group, non-agg
captures define subgroups; one source block produces one sibling block
per combination of non-agg capture values.

**Generalised to all block types**, not tied to topk. A `plain` block
whose columns share a regex shape can use subgroup_explode to fan out
into per-subgroup siblings — useful for any parallel-question layout.

Controlled by fields at the top level of any block:

- `agg_index` — which regex group is the aggregation axis (stays within
  a block; the rest fan out). Only Top K specific. 
- `subgroup_labels` — human-readable names per non-agg capture value,
  used in generated sibling-block names.

## Stage 3 — transform (1:1 with block type)

| Block type | Transform |
|---|---|
| `plain` *(default)* | none — columns pass through unchanged |
| `topk` | one-hot selection → left-packed rank columns (agg over `agg_index`) |
| `maxdiff` | resolve `choice_sets` via `setindex_column` into per-row set columns |

Transform parameters (e.g. `choice_sets` for maxdiff, `na_vals` for
topk) stay as normal fields on the block, not inside a generic
`transform: {...}` wrapper. The block-type discriminator dispatches to
the right handler in `io.py`.

**One transform per block, always.** Two-transform pipelines are
expressed by chaining blocks: the second block's columns reference the
first's output via the existing `source` mechanism. Schema stays flat.

Why 1:1 instead of a named-transform vocabulary: maxdiff requires
`choice_sets`, which is transform-specific state. Carving a per-type
block class keeps the schema for each transform honest about its own
required fields, versus a generic `transform: "maxdiff", params: {...}`
that would need ad-hoc validation.

### Input format — skipping a transform that's already been done

Sometimes the upstream ETL already performed the transform and the raw
data arrives in the post-transform shape. Each transforming block type
takes an `input_format` field declaring the shape of its incoming data.
When the input is already in the output shape, the transform becomes a
no-op; downstream stages (subgroup_explode, structure) still run.

| Block type | `input_format` values | Meaning |
|---|---|---|
| `topk` | `"onehot"` *(default)* | Raw cells are per-choice selection values (strings, or `"selected"`/`"NO TO: ..."`). The leftpack transform runs. |
| `topk` | `"leftpacked"` | Raw cells already in `R1`, `R2`, ... columns carrying agg-axis values directly. Transform skipped; `res_columns` must match existing column names. Unlocks the optional `ordered: true` flag, since source rank survives into `segments()` — see Stage 4. |
| `maxdiff` | `"choice_sets"` *(default)* | Raw best/worst cells plus `choice_sets` + `setindex_column`. Set resolution runs. |
| `maxdiff` | `"resolved"` | Set columns already populated per row with shown-item lists. Transform skipped; `choice_sets` / `setindex_column` unused. `best_columns` / `worst_columns` / `set_columns` are each supplied either as independent regexes whose captured question-index aligns across the three, or as explicit `List[str]`. |

**Column matching under `"resolved"`.** Each of `best_columns`,
`worst_columns`, `set_columns` accepts its own regex. Alignment across
the three roles is by matching capture-group value (typically the
question index), not by substitution templating. This covers weird
provider naming cleanly:

```json
{
  "type": "maxdiff",
  "input_format": "resolved",
  "best_columns":  "Q6_(\\d+)_1",
  "set_columns":   "Q6_(\\d+)_set_abc",
  "worst_columns": "Q6_(\\d+)_2",
  "choice_mapping": {"1": "Economy", "2": "Health", "3": "Education"}
}
```

Here `Q6_3_1`, `Q6_3_set_abc`, `Q6_3_2` all capture `"3"` and are
grouped as one question.

Explicit `List[str]` is the fallback when no single regex captures the
alignment (e.g. the three roles genuinely don't share a naming stem);
one entry per question, index-aligned across the three lists.

**Substitution-template form** (`set_columns: "Q6_\\1set"` as a backref
into `best_columns`'s captures) stays supported for the default
`"choice_sets"` path, where our own transform picks the naming and the
shortcut is convenient. Under `"resolved"`, independent regex is the
expected shape.

**Wide one-hot-per-item shape.** If a provider ships sets as one-hot
"item present/absent" columns instead of list-valued cells, they should
flatten to list-valued cells upstream in a preprocessing hook. The
block only consumes list-valued set columns.

`input_format` is a per-type field (`TopKBlock.input_format`,
`MaxDiffBlock.input_format`) with its own enum on each class — consistent
with the 1:1 block-to-transform mapping. `plain` blocks have no
transform and thus no `input_format` field.

## Stage 4 — give_structure

The resolved output block carries enough metadata for
`AutodetectOM.prepare` (`salk_internal_package/obs_models/autodetect.py:22`)
to pick the right observation model downstream.

`segments()` returns a list of `(a, b, ordered)` triples consumed by
`OrdinalRanking`:

- `a`, `b` are lists of column names drawn from the resolved block.
- `b = None` means "`a` is the selected set out of the full agg-axis
  universe" — a multi-select, no ordering among the items of `a`.
- `ordered = True` means "every column in `a` is ranked strictly above
  every column in `b`". `ordered = False` with a non-`None` `b` would
  mean a set comparison without intra-set rank; not currently used.

Per block type:

- `plain` → no `segments()`. Categorical (or Continuous if
  high-cardinality numeric). Autodetect handles it.
- `topk` with `input_format: "onehot"` → unordered multi-select.
  `segments()` returns a single `(all_R_cols, None, False)`: the
  selected set, no rank among positions. This matches the current
  leftpack behaviour — raw selection order is destroyed by
  `_throw_vals_left` (`salk_toolkit/io.py:443`), so there is no rank
  to expose.
- `topk` with `input_format: "leftpacked"` and `ordered: true` →
  ranked positions. Source rank is preserved (transform skipped), so
  `segments()` returns the chain
  `[([R1], [R2, R3, ...], True), ([R2], [R3, ...], True), ...,
  ([R_{k-1}], [R_k], True), (all_R_cols, None, False)]`: each position
  beats every lower position, plus the overall selected-vs-unselected
  multi-select segment. This is the shape `OrdinalRanking` consumes to
  fit a rank-weighted model. Mirrors `MaxDiffBlock.segments()`
  (`salk_toolkit/validation.py:345`), which emits analogous ordered
  pairs `(best, set, True)` and `(set, worst, True)`.
- `topk` with `input_format: "leftpacked"` and `ordered: false`
  *(default when `leftpacked`)* → same unordered shape as the `onehot`
  path. Use this when the data happens to be in R-column form but the
  positions don't encode rank (e.g. arbitrary storage order from the
  upstream system).
- `maxdiff` → `OrdinalRanking` via `segments()` on the resolved
  `MaxDiffBlock`. Two ordered segments per question: best > set,
  set > worst.

Structure is read off the resolved block; users do not write it
directly. Block types that need non-default structure implement
`segments()`; the rest leave it to autodetect.

### The `ordered` field on `topk`

`ordered: bool = False` lives on `TopKBlock` alongside `input_format`.
Only meaningful (and only allowed) when `input_format == "leftpacked"`,
because the `onehot` path destroys source rank before `segments()`
sees the data. Hard-fail if `ordered: true` is combined with
`input_format: "onehot"`.

The `am_data` A10 case (`/am_data/am_26_apr_meta.json:394-423`) is the
canonical user of `input_format: "leftpacked"` + `ordered: true`:
three rank-position columns `vA10_M_1/2/3` whose position *is* the
rank.

## Adding a new transform (e.g. onehot-expand)

A new block type, not a new primitive. Steps:

1. Add a Pydantic block class (e.g. `OneHotBlock`) with a
   `type: Literal["onehot"]` discriminator.
2. Implement the transform once in `io.py`, dispatched by the
   discriminator.
3. Implement `segments()` only if the output needs non-default
   structure; otherwise autodetect handles it.

Users adopt it by writing `"type": "onehot"` — same uniform surface as
`topk` and `maxdiff`.

### The `onehot` case

Motivating example: `am_data` Q12 social media
(`/am_data/am_26_apr_meta.json:603-624`). Raw shape is leftpacked
rank-position columns (`vQ12_M_1..7`, each cell a platform code), but
the desired output is one boolean column per platform — a multi-select
wide form, not a rank. Today this is done with handwritten
preprocessing code (`preprocessing` lines 15-18 of that file).

A `OneHotBlock` would:

- Take leftpacked input columns plus an explicit item list (or a
  regex-captured index set).
- Emit one boolean column per item (`sm_Facebook`, `sm_TikTok`, …).
- Default `segments()` to the multi-select form:
  `[(all_item_cols, None, False)]`.

Contrast with `topk`: `topk` narrows the agg-axis domain to a fixed
width (`R1..Rk`) irrespective of item identity; `onehot` widens by
item, keeping every item addressable by name. Neither is a reversal
of the other in a strict sense — they produce different output
shapes — which is why "reverse_topk" is the wrong name.

## Non-goals

- No user-exposed pipeline or primitive vocabulary.
- No `transform[]` array.
- No named-transform dispatch separate from block type.
- No exposure of `segments()` as a user-written field.
