# Migrating legacy create-block annotations

Read this when an annotation fails to load with a "legacy nested 'create' field" /
"removed block field(s)" `ValueError`, or when you are asked to update a metafile
written before the 2026-04 block-processing refactor. The loader rejects the old
shapes **loudly** — a legacy file cannot silently mis-process — so every old
annotation must be migrated by hand using the rules below. The new-schema reference
is [specs/block-processing.md](../../../specs/block-processing.md); the new syntax
essentials are in [SKILL.md](SKILL.md).

## The shape change

Old: a plain block carried a nested `create` sub-object describing the derivation.
New: the block **is** the typed thing — hoist `create.type` to a top-level `type`
discriminator and flatten the remaining `create` fields onto the block itself.

```jsonc
// OLD                                     // NEW
{ "name": "x",                             { "type": "topk",
  "create": { "type": "topk",                "name": "x",
              "from_columns": "...",         "from_columns": "...",
              ... },                         ...,
  "scale": { ... },                          "scale": { ... },
  "columns": [] }                            "columns": [] }
```

## Field migration table

| Removed | Replacement |
|---|---|
| nested `create: { ... }` | hoist `create.type` to top-level `type`, flatten the rest |
| TopK `create.translate_after` / `translate_values` | `scale.translate_after` |
| TopK `groups` | `subgroup_labels` |
| MaxDiff `topics` / `items` / `choice_mapping` / `row_labels` | folded into `scale.translate` (see semantic change below) |
| MaxDiff `sets` | `choice_sets` (same `[version][set][item-index]` shape, now inline on the block, not a constant) |
| — (new, optional) | `input_format` — declares the raw-data shape; defaults (`topk: "onehot"`, `maxdiff: "choice_sets"`) match the old behavior |

Unchanged: `from_columns`, `res_columns`, `agg_index`, `na_vals`, `k`, `from_prefix`
(TopK); `best_columns`, `worst_columns`, `set_columns`, `setindex_column` (MaxDiff);
`scale`, `columns`, and all plain-block fields.

## MaxDiff: `scale.translate` changed meaning (IMPORTANT)

This is the one migration that is a **semantic recode**, not a rename.

- **Old**: `topics` was a source-language list (usually a constant), and
  `scale.translate` mapped *local topic name → English name*.
- **New**: `scale.translate` maps *1-based index string → English name* and is the
  single source of truth: it defines the topic universe for `setindex_column` /
  `choice_sets` lookups **and** element-wise translates raw best/worst cells.

To migrate, compose the two old structures:

```jsonc
// OLD: "topics": ["Ekonomika", "Sveikata"],  (order matters!)
//      "scale": { "translate": { "Ekonomika": "Economy", "Sveikata": "Healthcare" } }
// NEW: index i+1 of old topics → its English translation
"scale": { "translate": { "1": "Economy", "2": "Healthcare" } }
```

If the raw best/worst cells hold *names* rather than index strings (so integer-keyed
translate can't apply), use `input_format: "resolved"` with a name-keyed translate
instead — see the "Two maxdiff routes" note in SKILL.md.

## Output block names changed

Old create blocks emitted derived blocks named `<block>_<type>`
(e.g. `issue_importance_topk`, `maxdiff_maxdiff`); new typed blocks keep the
block's own name (`issue_importance`). **Grep model descriptions, dashboards and
downstream configs for the suffixed names when migrating** — they must be updated
to the unsuffixed name.

## Worked example: TopK

```jsonc
// OLD
{ "name": "issue_importance",
  "create": { "type": "topk",
              "from_columns": "Q6r(\\d+)", "res_columns": "Q6p_R\\1",
              "agg_index": 1, "na_vals": ["NO TO: ..."],
              "translate_after": { "1": "Cost of living", "2": "Healthcare" } },
  "scale": { "categories": "infer" }, "columns": [] }

// NEW
{ "type": "topk", "name": "issue_importance",
  "from_columns": "Q6r(\\d+)", "res_columns": "Q6p_R\\1",
  "agg_index": 1, "na_vals": ["NO TO: ..."],
  "input_format": "onehot",
  "scale": { "categories": "infer",
             "translate_after": { "1": "Cost of living", "2": "Healthcare" } },
  "columns": [] }
```

## Worked example: MaxDiff (setindex + sets)

```jsonc
// OLD (topics/sets usually via constants)
{ "name": "maxdiff",
  "create": { "type": "maxdiff",
              "best_columns": "Q4_(\\d+?)best", "worst_columns": "Q4_(\\d+?)worst",
              "set_columns": "Q4_\\1set",
              "setindex_column": ["Q4_Version", { "continuous": true }],
              "topics": ["Ekonomika", "Sveikata", "Gynyba"],
              "sets": [[[1,2,3]], [[2,3,1]]] },
  "scale": { "categories": "infer",
             "translate": { "Ekonomika": "Economy", "Sveikata": "Healthcare",
                            "Gynyba": "Defence" } },
  "columns": [] }

// NEW
{ "type": "maxdiff", "name": "maxdiff",
  "best_columns": "Q4_(\\d+?)best", "worst_columns": "Q4_(\\d+?)worst",
  "set_columns": "Q4_\\1set",
  "setindex_column": ["Q4_Version", { "continuous": true }],
  "input_format": "choice_sets",
  "choice_sets": [[[1,2,3]], [[2,3,1]]],
  "scale": { "categories": "infer",
             "translate": { "1": "Economy", "2": "Healthcare", "3": "Defence" } },
  "columns": [] }
```

After migrating: delete any now-unused `topics`/`sets` constants, then verify with
`read_annotated_data(meta_file)` — it must load with no warnings and the derived
block must carry the same English category list as before.
