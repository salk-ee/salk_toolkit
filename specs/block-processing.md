# Block processing & the DataMeta block schema

Authoritative reference for how `salk_toolkit` turns annotated survey columns into
processed output, and how to author the JSON for TopK / MaxDiff / OneHot blocks.
This supersedes the earlier `specs/2026-04-*` planning docs.

The Pydantic models in `salk_toolkit/validation.py` are the source of truth for the
exact fields and defaults; this doc explains the concepts and shows working JSON.

## What a block is

A *block* is one entry in a DataMeta `structure`: a named group of columns that
share a scale and a processing rule. Every block has a top-level `type`
discriminator:

| `type`    | Class           | Purpose                                                              |
|-----------|-----------------|---------------------------------------------------------------------|
| `plain`   | `ColumnBlockMeta` | Pass-through columns with metadata (the default if `type` is omitted). |
| `topk`    | `TopKBlock`     | Aggregate multi-select / ranked columns into top-K ranked slots.    |
| `maxdiff` | `MaxDiffBlock`  | Best–worst (MaxDiff) experiments.                                    |
| `onehot`  | `OneHotBlock`   | Widen rank-position columns into one boolean column per choice.      |

`type` lives at the **top level** of the block. The old nested `"create": {...}`
form and the removed block-level fields below are rejected at load time with a
migration hint — they no longer silently no-op.

## The pipeline (internal)

Plain blocks are processed column-by-column by `salk_toolkit/io/pipeline.py`
(gather → translate/transform → resolve categories). Specialized blocks
(`topk`/`maxdiff`/`onehot`) go through `_process_block` in
`salk_toolkit/io/create_blocks.py`, which runs five stages:

1. **Match** — resolve `from_columns` (regex or list) to concrete df columns.
2. **Explode** — fan a regex with capture groups out into one *sibling* block per subgroup.
3. **Pre-translate** — map raw cell values through `scale.translate` (index → name).
4. **Transform** — the type-specific aggregation/widening, producing the output columns.
5. **Post-translate** — map output cells and categories through `scale.translate_after`.

### `scale.translate` vs `scale.translate_after`

- **`translate`** runs *before* the transform. Use it when raw cells hold index
  strings (`"1"`, `"2"`, …) that need to become names before aggregating — this is
  the norm for MaxDiff, where it doubles as the topic universe.
- **`translate_after`** runs *after* the transform, on the output cells. Use it to
  map aggregated index/code values to display names — the norm for TopK. It is
  **not** allowed on `maxdiff` blocks (use `translate`).

## TopK

```json
{
  "type": "topk",
  "name": "issue_importance",
  "columns": [],
  "from_columns": "q(\\d+)_(\\d+)",
  "res_columns": "q\\1_R\\2",
  "na_vals": ["not_selected"],
  "input_format": "onehot",
  "scale": { "translate_after": { "1": "USA", "2": "Canada", "3": "Mexico" } }
}
```

- `from_columns` regex capture groups index subgroups and items; siblings explode
  per leading group. `res_columns` is a substitution template (`\1`, `\2`).
- `agg_index` (default `-1`) selects which capture group is the item index.
- `input_format`: `onehot` (one 0/1 column per item, the default), `leftpacked`
  (`R1..Rk` already hold chosen names — transform is skipped), or the `ranked_*`
  variants which additionally treat slot order as a ranking (`segments()`).

## MaxDiff

```json
{
  "type": "maxdiff",
  "name": "maxdiff",
  "columns": [],
  "best_columns": "Q2_(\\d+?)best",
  "worst_columns": "Q2_(\\d+?)worst",
  "set_columns": "Q2_\\1set",
  "setindex_column": ["Q2_Version", { "continuous": true, "categories": null }],
  "input_format": "choice_sets",
  "choice_sets": [[[1, 2, 3, 4, 5]]],
  "scale": { "categories": ["A", "B", "C"], "translate": { "1": "A", "2": "B", "3": "C" } }
}
```

- `scale.translate` maps 1-based index strings to topic display names and is the
  topic universe for `setindex_column` lookups.
- `input_format`: `choice_sets` (best/worst cells hold indices, `choice_sets` /
  set columns define each question's options) or `resolved` (best/worst/set
  columns already aligned per question).

> **Note — two maxdiff routes.** This `MaxDiffBlock` transform (int-index cells,
> required `set_columns`) is distinct from how maxdiff is usually modelled in
> production, where the best/worst/set columns are kept as plain name-categorical
> columns and fed to the SIP `ordinal_ranking` observation model via a hand-written
> `structure` (with the shown-set column as the comparison set). Don't assume a
> survey's maxdiff goes through this STK transform — check the model_desc.

## OneHot

```json
{
  "type": "onehot",
  "name": "social_media",
  "from_columns": "vQ12_M_(\\d+)",
  "input_format": "leftpacked",
  "choices": ["Facebook", "TikTok"],
  "res_prefix": "sm_",
  "na_vals": ["99"]
}
```

- `input_format`: `leftpacked` (`M_1..M_n` hold chosen choice names packed left) or
  `wide` (one column per choice already).
- `choices` is optional; if omitted it's the sorted union of non-null cell values
  (excluding `na_vals`).

## Migrating pre-refactor annotations

These block-level fields were **removed** and now raise a `ValueError` at load:

| Removed field                | Where it goes now                                          |
|------------------------------|-----------------------------------------------------------|
| nested `create: { ... }`     | Hoist `create.type` to top-level `type`, flatten the rest.|
| MaxDiff `topics` / `items` / `choice_mapping` / `row_labels` | `scale.translate` (index → name). |
| MaxDiff `sets`               | `set_columns` / `setindex_column`.                        |
| TopK `translate_values`      | `scale.translate_after`.                                  |
| TopK `groups`                | `subgroup_labels`.                                         |

Because the schema only *ignores* genuinely-unknown fields, these named legacy
fields are detected explicitly so stale files fail loudly instead of mis-processing.

## Multi-file structure merge

When a DataMeta loads several files, `_merge_data_metas` (`salk_toolkit/io/meta.py`)
unions the block
*structure* across all file metas in file order (not just the last file):

- **Columns**: first-seen union — file 1's order is preserved, later files' new
  columns are appended.
- **Categories**: unioned (preserving order); `"infer"` on either side stays `"infer"`.
- **Scalars**: last-file-wins, with a warning on disagreement (`source` is exempt).
- **Hard conflicts that raise**: block-`type` mismatch, scale-kind mismatch, or a
  `num_values` length that disagrees with the merged categories.
