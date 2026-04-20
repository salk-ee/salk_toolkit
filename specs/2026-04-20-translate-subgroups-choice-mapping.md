# Translations, Subgroups & Choice Mapping

A simplification of how TopK and MaxDiff blocks express value remapping,
subgroup naming, and MaxDiff's index-to-value lookup.

## What's wrong today

Value-remapping is spread across five fields (`translate`, `translate_after`,
`translate_values`, `scale.translate`, `MaxDiffBlock.translate`), each tied to
an implicit position in the processing pipeline. Structural labeling
(TopK subgroup names via `groups`, MaxDiff item lookup via `items`) is
conflated with translation. Readers must memorise which field fires when.

## Three separable concerns

1. **Structural labeling** — irreducibly nested / per-regex-group. Stays as
   its own field: `subgroup_labels` (TopK), `choice_mapping` (MaxDiff).
2. **Value translation** — a flat `old → new` dict. Position in the pipeline
   is usually incidental; expose only when it matters.
3. **Output structure** — segment/category hints for observation models.
   Out of scope here.

## New fields

### `subgroup_labels` — TopK only (was `groups`)

Maps each non-agg regex group's capture values to labels used in the
generated sibling-block names. Not a translation.

```json
"subgroup_labels": {
  "1": {"1": "economics", "2": "healthcare", "3": "education"}
}
```

### `choice_mapping` — MaxDiff only (was `items`)

Maps 1-based item index to the value **as it appears in raw data cells**.
Pairs with `choice_sets` (which indices were shown).

```json
"choice_mapping": {"1": "Ekonomika", "2": "Sveikata", "3": "Švietimas"}
```

Values are in the language of the raw data. Localisation happens via
`translate`.

### `translate` + `translate_strategy` — all block types

One flat remap dict, one strategy flag:

```json
"translate": {"Ekonomika": "Economy", "Sveikata": "Health"},
"translate_strategy": "auto"
```

| Strategy | Meaning |
|---|---|
| `"auto"` *(default)* | Apply at the pipeline stage where every key resolves. If both stages would resolve, prefer `"output"` and emit a warning. |
| `"raw"` | Apply to raw cell values before the block's transform. |
| `"output"` | Apply to cell values after the block's transform. |

Only one `translate` block per `ColumnMeta` / block. A second stage is never
needed in practice — the block's transform already collapses raw values to
codes before translation runs. If a future need appears, the schema can be
extended to a stage-keyed dict-of-dicts without breaking the current form.

### Failure rules

1. **Every key in `translate` must match at the chosen stage.** Dead keys
   indicate a stale or mistyped annotation. Hard fail.
2. **Every cell must be covered by `translate`.** Uncovered cells are silently
   lost data. Hard fail — unless opted out (see below).

### Opting into partial coverage

`accept_partial: bool = False` — when `true`, cells not covered by
`translate` pass through unchanged instead of failing. Rule 1 (dead keys)
still applies.

```json
"translate": {"yes": 1, "no": 0},
"accept_partial": true
```

Default `false` everywhere keeps the safe behaviour. Opt-in `true` is the
right setting for legitimate partial renames (e.g. touching up a handful
of categories on a `ColumnMeta` without enumerating the rest).

The same `translate` + `translate_strategy` + `accept_partial` triple lives
on `ColumnMeta`, replacing `translate_after` (applied post-`transform`).
`ColumnMeta.translate` retains its current semantics as the default
pre-`transform` remap.

### Why only two explicit stages?

Of the four theoretically possible positions, only cell-value translations
have realistic use cases:

| Case | Example | Keep? |
|---|---|---|
| Cell, pre-transform | Raw Lithuanian party strings → short codes before TopK collapse | ✅ |
| Cell, post-transform | TopK agg codes → party names; MaxDiff Lithuanian → English | ✅ |
| Column name, pre-transform | — raw column names are opaque regex identifiers | ❌ |
| Column name, post-transform | — `res_columns` regex already controls output naming | ❌ |

## Examples

**TopK, post-transform (the typical case):**

```json
{
  "type": "topk",
  "from_columns": "Q7r(\\d+)c(\\d+)", "res_columns": "Q7r\\1_R\\2",
  "agg_index": 2,
  "subgroup_labels": {"1": {"1": "economics", "2": "healthcare"}},
  "na_vals": ["NO TO: ..."],
  "translate": {"1": "TS-LKD", "2": "LSDP", "99": "No party"},
  "translate_strategy": "auto"
}
```

**TopK, pre-transform (raw Lithuanian cells):**

```json
{
  "type": "topk", "from_columns": "Q7r(\\d+)c(\\d+)", "agg_index": 2,
  "translate": {
    "Tėvynės sąjunga...": "TS-LKD",
    "Lietuvos socialdemokratų...": "LSDP"
  },
  "translate_strategy": "raw"
}
```

**MaxDiff (post-transform):**

```json
{
  "type": "maxdiff",
  "best_columns": "Q6_(\\d+)best", "worst_columns": "Q6_(\\d+)worst",
  "set_columns": "Q6_\\1set",
  "choice_mapping": {"1": "Ekonomika", "2": "Sveikata"},
  "translate": {"Ekonomika": "Economy", "Sveikata": "Health"},
  "translate_strategy": "auto"
}
```

Note that translate_strategy "auto" is implicit and succeeds for all those examples. The strategy option covers for cases when there might be multiple options which stage to translate.

**Pitfall: numeric-key collision.** When `translate` keys are numeric strings,
they can match *both* raw cell values and post-transform regex captures.
`"auto"` picks output + warns — but if the user actually meant raw, the
result is silently wrong.

```json
{
  "type": "topk",
  "from_columns": "Q(\\d+)",
  "agg_index": 1,
  "translate": {"1": "Running", "2": "Swimming", "3": "Cycling"}
}
```

Suppose raw cells are pre-coded numerically (`"1"` = respondent picked
activity 1 in Q1, `"2"` = activity 2 in Q2, etc.) *and* the regex captures
are also `"1"`/`"2"`/`"3"` (the Q-number). Both stages resolve the same
numeric keys. `"auto"` applies translation at output — labelling cells by
column position rather than by the respondent's coded answer.

Mitigations: (a) `"translate_strategy": "raw"` pinned explicitly when the
user means raw; (b) the `"auto"` warning on ambiguity is a hard signal to
inspect; (c) as a general rule, avoid numeric keys in `translate` on blocks
whose regex captures are also numeric — prefer a `ColumnMeta.translate`
upstream that maps pre-coded numbers to labels before the block sees them.

## Migration map

| Old | New |
|---|---|
| `TopKBlock.groups` | `TopKBlock.subgroup_labels` |
| `TopKBlock.translate_values` | `translate` + `translate_strategy="output"` (or `"auto"`) |
| `MaxDiffBlock.items` | `MaxDiffBlock.choice_mapping` |
| `MaxDiffBlock.translate` | `translate` + `translate_strategy="output"` (or `"auto"`) |
| `scale.translate` used as pre-step on topk | `translate` on block + `translate_strategy="raw"` (or `"auto"`) |
| `ColumnMeta.translate_after` | `ColumnMeta.translate` + `translate_strategy="output"` (or `"auto"`) |
| *(new)* | `accept_partial: true` on any `translate` that is not meant to be exhaustive |
