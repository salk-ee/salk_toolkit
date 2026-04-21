# Block Processing — Migration Guide

Companion to `2026-04-20-block-processing-pipeline.md`. This document is
the authoritative checklist when rewriting pre-refactor annotation JSON
files (data meta) to the new block schema. Point an LLM at it when
migrating old `*_meta.json` files.

## TL;DR field map

| Old location | Old field | New location | New field | Notes |
|---|---|---|---|---|
| `TopKBlock` | `groups` | `ColumnBlockMeta` (any block) | `subgroup_labels` | Same shape; moved up so any block type can use it. Key is 1-based regex group index (string). |
| `TopKBlock` | `translate_values` | `scale` | `translate_after` | Applied after leftpack; maps capture-group values to display names. |
| `MaxDiffBlock` | `items` | `MaxDiffBlock` | `choice_mapping` | Same shape: 1-based item index (string) → raw-data item value. |
| `MaxDiffBlock` | `translate` | `scale` | `translate_after` | Applied to best/worst cell values and resolved set cells. |

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
    "translate_after": {
      "Ekonomika ir verslo aplinka": "Economy and business environment",
      "…": "…"
    }
  },
  "best_columns":  "Q6_(\\d+?)best",
  "worst_columns": "Q6_(\\d+?)worst",
  "set_columns":   "Q6_\\1set",
  "setindex_column": ["Q6_Version", {"continuous": true}],
  "choice_sets": [[[1, 2, 3], [1, 4, 5], "…"], "…"],
  "choice_mapping": {
    "1": "Ekonomika ir verslo aplinka",
    "2": "…"
  }
}
```

Changes:

- `items` → `choice_mapping`. Same shape, renamed for consistency with
  `choice_sets`.
- `translate` on the block → `scale.translate_after`. (Block-level
  `translate` today is applied to cell values post-transform, which is
  exactly what `scale.translate_after` does universally.)

## Pure rename checklist (for a find-and-replace pass)

For each annotation file:

1. In every `TopKBlock`: rename `groups` → `subgroup_labels`.
2. In every `TopKBlock`: move `translate_values` into its `scale` block
   as `translate_after`. Remove the block-level field.
3. In every `MaxDiffBlock`: rename `items` → `choice_mapping`.
4. In every `MaxDiffBlock`: move `translate` into its `scale` block as
   `translate_after` (merge with any existing `scale.translate_after`).
   Remove the block-level field.

No other structural edits are required to adopt the refactor. New
features (`input_format`, `OneHotBlock`) are opt-in.
