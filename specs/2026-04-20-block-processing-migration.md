# Block Processing — Migration Guide

Companion to `2026-04-20-block-processing-pipeline.md` and
`2026-04-23-unified-block-processing-plan.md`. This document is the
authoritative checklist when rewriting pre-refactor annotation JSON files
(data meta) to the new block schema. Point an LLM at it when migrating old
`*_meta.json` files.

## TL;DR field map

| Old location | Old field | New location | New field | Notes |
|---|---|---|---|---|
| `TopKBlock` | `groups` | `ColumnBlockMeta` (any block) | `subgroup_labels` | Same shape; moved up so any block type can use it. Key is 1-based regex group index (string). |
| `TopKBlock` | `translate_values` | `scale` | `translate_after` | Applied after leftpack; maps capture-group values to display names. |
| `MaxDiffBlock` | `items` | `scale` | `translate` | **Moved to `scale.translate`.** Maps 1-based index string → display name. See MaxDiff section below. |
| `MaxDiffBlock` | `translate` | `scale` | `translate` | **Merged into the same `scale.translate` dict** as `items`. The union acts as both the index→name source for `setindex_column` lookups and an element-wise translator for raw best/worst/set cells. `translate_after` is NOT valid for MaxDiff. |
| `MaxDiffBlock` | `choice_mapping` | `scale` | `translate` | Intermediate name during refactor; folded into `scale.translate`. Delete the field. |

## Generated sibling-block names

Old rule (topk only): `"{block.name}_topk_{label}"` per sibling, e.g.
`issue_ownership_topk_economics`.

New rule (all types): `"{block.name}_{label}"` per sibling, e.g.
`issue_ownership_economics`. Blocks producing a single sibling keep the
bare block name with no suffix.

**Only affects downstream references to the generated names.** If some
sibling block is referenced by name in a model description or other
annotation, drop the `_{type}` segment. In production files today, the
only cases of this are internal to `salk_toolkit`; no existing LT / AM
annotation file needs rewrites for this.

## New fields (no migration needed — optional additions)

- `input_format` on `TopKBlock`, `MaxDiffBlock`, `OneHotBlock`. Default
  values (`"onehot"` / `"choice_sets"` / `"leftpacked"`) match current
  behaviour, so omitting the field is correct for existing data. Set it
  explicitly when the upstream ETL has already produced the transform
  output shape.
- `OneHotBlock` (`type: "onehot"`) is a new block type. Use it when
  replacing handwritten preprocessing that flattens leftpacked item
  columns into per-item booleans (cf. am_data Q12 social media).

## Worked examples

### 1. TopK — issue ownership (leedu)

**Before** (`LT_WEB_2026_variant.original.json`, paraphrased):

```json
{
  "type": "topk",
  "name": "issue ownership",
  "scale": {
    "categories": "infer",
    "translate": {
      "Tėvynės sąjunga – Lietuvos krikščionys demokratai (…)": "TS-LKD",
      "…": "…"
    }
  },
  "from_columns": "Q7r(\\d+)c(\\d+)",
  "res_columns": "Q7r\\1_R\\2",
  "agg_index": 2,
  "na_vals": ["NO TO: …", "…"],
  "translate_values": {
    "1": "TS-LKD", "2": "LSDP", "…": "…"
  },
  "groups": {
    "1": {"1": "economics", "2": "healthcare", "…": "…"}
  }
}
```

**After:**

```json
{
  "type": "topk",
  "name": "issue ownership",
  "scale": {
    "categories": "infer",
    "translate_after": {
      "1": "TS-LKD", "2": "LSDP", "…": "…"
    }
  },
  "from_columns": "Q7r(\\d+)c(\\d+)",
  "res_columns": "Q7r\\1_R\\2",
  "agg_index": 2,
  "na_vals": ["NO TO: …", "…"],
  "subgroup_labels": {
    "1": {"1": "economics", "2": "healthcare", "…": "…"}
  }
}
```

Changes:

- `translate_values` → `scale.translate_after`. (The old top-level
  `scale.translate` mapping full Lithuanian party names to short codes
  is no longer needed because the topk transform emits capture-group
  values `"1", "2", …`, not raw cell strings. The `translate_after`
  above handles the short-code mapping on the output side.)
- `groups` → `subgroup_labels`. Same shape, renamed and moved up.

### 2. MaxDiff — leedu Q6

MaxDiff changed significantly. The `items`, `choice_mapping`, and
block-level `translate` / `scale.translate_after` fields all collapse
into a single `scale.translate` dict. This dict plays three roles:

1. The `topics` universe used by `setindex_column` lookups (was
   `choice_mapping.values()` or `items.values()`).
2. Element-wise translator for raw best/worst cells (when cells hold
   index strings like `"1"`).
3. Element-wise translator for inline set-column cells (whether cells
   hold index tokens or name tokens — the dict's keys must match the
   token space of the cells).

**`scale.translate_after` is not supported on MaxDiff** and will raise
`ValueError` at read time. Any pre-refactor MaxDiff annotation that used
`translate_after` (or the older block-level `translate`) must move that
mapping into `scale.translate`, composed with any existing `items` /
`choice_mapping` if present.

**Before:**

```json
{
  "type": "maxdiff",
  "name": "maxdiff",
  "scale": { "categories": "infer" },
  "best_columns":  "Q6_(\\d+?)best",
  "worst_columns": "Q6_(\\d+?)worst",
  "set_columns":   "Q6_\\1set",
  "setindex_column": ["Q6_Version", {"continuous": true}],
  "choice_sets": [[[1, 2, 3], [1, 4, 5], "…"], "…"],
  "items": {
    "1": "Ekonomika ir verslo aplinka",
    "2": "…"
  },
  "translate": {
    "Ekonomika ir verslo aplinka": "Economy and business environment",
    "…": "…"
  }
}
```

**After:**

```json
{
  "type": "maxdiff",
  "name": "maxdiff",
  "scale": {
    "categories": "infer",
    "translate": {
      "1": "Economy and business environment",
      "2": "…"
    }
  },
  "best_columns":  "Q6_(\\d+?)best",
  "worst_columns": "Q6_(\\d+?)worst",
  "set_columns":   "Q6_\\1set",
  "setindex_column": ["Q6_Version", {"continuous": true}],
  "choice_sets": [[[1, 2, 3], [1, 4, 5], "…"], "…"]
}
```

Changes:

- `items` and block-level `translate` collapse into `scale.translate`.
  The new dict keys are 1-based index strings (like the old `items`)
  and the values are the target display names (composed from the old
  `items[k]` passed through the old `translate`).
- **Raw best/worst cell values must match the keys of
  `scale.translate`.** Typically this means: if your raw cells
  contained Lithuanian topic names (`"Ekonomika ir verslo aplinka"`),
  either (a) continue keying translate by the raw names (the "inline
  name tokens" path — include both index-string keys AND raw-name
  keys if you also need `setindex_column` lookups), or (b) rewrite the
  raw data so best/worst cells hold index strings (`"1"`, `"2"`, …).
  The simplest production pattern is (b) when you control upstream
  ETL.
- `choice_mapping` is **not** a valid field and must be removed.

### 3. MaxDiff — inline set cells (no `setindex_column`)

For MaxDiff data without version metadata, set columns carry the topic
set directly per row. Prefer integer-list tokens over comma-separated
strings; both are accepted but integer lists avoid string-parsing
ambiguity.

```json
{
  "type": "maxdiff",
  "name": "q1",
  "scale": {
    "categories": "infer",
    "translate": {"1": "Economy", "2": "Health", "3": "Education"}
  },
  "best_columns":  ["Q_1best"],
  "worst_columns": ["Q_1worst"],
  "set_columns":   ["Q_1set"]
}
```

Cell shapes for `Q_1set`:
- Preferred: `[1, 2, 3]` (integer list) — matches `scale.translate`
  keys by index position.
- Accepted: `"1,2,3"` (string with separator) — parsed and each token
  translated.
- Also accepted: `["Economy", "Health", "Education"]` (list of
  already-translated names) or `["Ekonomika", "Sveikata",
  "Svietimas"]` (list of raw names) — in the latter case `scale.translate`
  must key on the raw names rather than indices.

`scale.translate` keys must cover whatever token space the cells use.
Mixed cells (some index tokens, some name tokens) work if the dict has
both key forms.

## Pure rename checklist (for a find-and-replace pass)

For each annotation file:

1. In every `TopKBlock`: rename `groups` → `subgroup_labels`.
2. In every `TopKBlock`: move `translate_values` into its `scale` block
   as `translate_after`. Remove the block-level field.
3. In every `MaxDiffBlock`: merge `items` (or the intermediate-name
   `choice_mapping`) and any block-level `translate` /
   `scale.translate_after` into a single `scale.translate` dict.
   Remove `items`, `choice_mapping`, and `translate_after` from the
   MaxDiff block.
4. In every `MaxDiffBlock`: if raw best/worst cells are not in the key
   space of the new `scale.translate`, either extend the dict or rewrite
   the raw data to use matching tokens (typically 1-based index
   strings). Run the data pipeline and inspect whether best/worst
   output cells are in the target language; an all-NaN column is the
   usual symptom of a key-space mismatch.

No other structural edits are required to adopt the refactor. New
features (`input_format`, `OneHotBlock`) are opt-in.
