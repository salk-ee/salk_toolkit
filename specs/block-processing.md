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

### `model_spec` — routing a block into the model

Every block accepts an optional `model_spec`: the observation-model description
(the dict a SIP `res_cols` entry would hold — any OM, not just `ordinal_ranking`)
that the block name should resolve to when passed to a model. Typed blocks stamp
a default onto their processed output blocks:

- **topk** — `{"structure": [[<output cols>, null]], "ordered": <ranked?>}`:
  picked items rank above the rest of the item pool; the `ranked_*` input
  formats set `ordered: true` so slot order counts as a ranking.
- **maxdiff** — `{"structure": [[[best_k], [set_k], [worst_k]], …]}`: one
  weak-order chain per question (best > shown set > worst).
- **onehot / plain** — no default; an authored `model_spec` passes through as-is.

An authored `model_spec` on a typed block wins over the default. Setting one on
a block that explodes into multiple subgroup siblings raises (ambiguous — the
siblings get per-sibling defaults). The spec travels with the DataMeta (parquet
round-trips included) and is exposed on the group entry by `extract_column_meta`,
which is how SIP's AutodetectOM resolves a block name to the described OM. In a
model desc, a `res_cols` entry of `{"name": "<block>", "model_spec": {…}}`
shallow-merges its dict over the block's spec — override parameters (`mode`,
`model`, …) without restating the structure.

### `na_labels` — global not-asked codes

A meta-level `na_labels` lists raw cell values meaning the question was **not
asked / not collected** (mode/filter skips: `"Nicht erhoben: Filter"`, …). They
are mapped to NA in every block before translation, so they never surface as
categories. A block-level `na_labels` overrides the global list (`[]` opts the
block out). This is **not** for substantive non-responses — "Don't know" /
"Refused" that were actually offered and picked stay as categories, flagged via
`nonresponse`. Distinct from the typed blocks' `na_vals`, which marks
*not-selected* cells within a multi-select. Replaces global
`df.replace(na_labels, nan)` preprocessing.

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
- `cell_values: true` (onehot): the cells hold the item value itself (e.g.
  LimeSurvey multi-choice: `"Social inequalities"` / `"Not mentioned"`), so the
  values are leftpacked as-is instead of mapped to column identity. `res_columns`
  then acts as a slot-name prefix template — `"q4a_"` yields `q4a_1..q4a_k`, and
  backrefs resolve per sibling (`"Q9_\\1b"` → `Q9_1b1`, `Q9_1b2`, …). This replaces
  `stk.aggregate_multiselect(colnames_as_values=False)` preprocessing.
- Translate/na_vals matching is round-trip tolerant: integer-string keys/values
  (`"1"`, `"0"`) also match the `1` / `1.0` / `"1.0"` cell forms CSV round-trips
  produce, so no `astype` coercion preprocessing is needed.
- Column matching follows **raw survey order** even when some matched columns are
  also declared as plain columns elsewhere in the meta.

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

- The topic universe is an index-keyed `scale.translate` (1-based index → name),
  **or** `scale.categories` when the data already holds display names (a name-keyed
  `translate` is then a plain recode).
- `input_format`: `choice_sets` (best/worst cells hold indices or names,
  `choice_sets` / set columns define each question's options) or `resolved`
  (best/worst/set columns already aligned per question).
- `setindex_column` cells may be **design-name strings** instead of version
  numbers: `choice_sets` is then a dict keyed by design name, each value one item
  list per question (indices or names): `{"block 1": [["Economy", "Health"], …]}`.
  The setindex column stays categorical over the design names.

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
  `wide` (one column per choice already: 0/1 dummies or mention markers).
- Output cells are coded via `coding` — default `["No", "Yes"]`, stamped as ordered
  categories (the negative-pole-first house convention); `"coding": null` keeps raw
  booleans. The block scale can add `likert` / `num_values` on top.
- In `wide` mode the choice identity is the first regex capture group of
  `from_columns` (or the bare column name), named through `scale.translate`
  (`{"1": "Facebook", …}`); cells are *not* translated. Choices the universe
  expects but the data lacks become all-unselected columns with a warning
  (cross-wave dummy-column drift). A wide cell is selected iff non-NA and, when
  numeric, nonzero (after `na_vals` replacement).
- `choices` is optional; if omitted it's derived from `scale.translate` values
  (wide) or the sorted union of non-null cell values (leftpacked).
- Replaces `stk.deaggregate_multiselect` (leftpacked) and hand-rolled
  `(df[dummy] == 1).map({True: "Yes", False: "No"})` preprocessing (wide).

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

When a DataMeta loads several files, `_merge_data_metas` unions the block *structure*
across all file metas in file order (not just the last file). It runs on the raw metas
in `_load_data_files`, before any block processing, so the merged block is
what the create/typed-block stages actually see.

- **Columns**: first-seen union — file 1's order is preserved, later files' new
  columns are appended.
- **Categories**: unioned (preserving order); `"infer"` on either side stays `"infer"`.
- **Column/scale meta fields** (`_MERGE_SCALAR_FIELDS`): last-file-wins, with a warning
  on disagreement.
- **All other block fields** (`from_columns`, `res_columns`, `k`, `na_vals`, …):
  first-file-wins, silently — `_merge_blocks` copies the accumulated block and only
  overrides `columns`/`scale`. A typed block whose source pattern changed between waves
  therefore keeps the earlier wave's pattern; such blocks need distinct names.
- **Top-level fields**: taken from the last file (only `structure` is merged).
- **Hard conflicts that raise**: block-`type` mismatch, scale-kind mismatch, or a
  `num_values` length that disagrees with the merged categories.
