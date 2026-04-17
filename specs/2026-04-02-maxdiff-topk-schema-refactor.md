# MaxDiff/TopK Schema Refactor — Design Document

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the TopK and MaxDiff schemas to cleanly separate three distinct concerns: subgroup naming (`groups`), aggregation-axis value translation (`translate_values`), and item labeling (`items`). Remove the confusing overlap between `row_labels`, `translate_after`, `topics`, `sets`, and `scale.translate`. TopK/MaxDiff blocks live at the top level of `DataMeta.structure` as specialized block types (Part 2, Option A), dispatched by a `type` discriminator — not nested inside a `create` field on a wrapper block. Downstream consumers (e.g. salk_internal_package) retrieve them by block name and use them directly.

**No backwards compatibility.** The old fields `topics`, `sets`, `row_labels`, and `translate_after` are removed. Existing annotation files must be updated to the new schema.

**Scope:** salk_toolkit only. No changes to salk_internal_package — that package will do its own wiring to consume the processed blocks.

**Tech Stack:** Python, Pydantic v2, pandas, pytest.

**Conventions:** Code as documentation. No comments unless they clarify non-obvious behavior.

---

## Processing walkthrough: from raw column to output

### TopK example: `Q7r1c2` in issue ownership

**Annotation (TopKBlock top-level — `type` discriminator dispatches to the TopK path):**
```json
{
  "type": "topk",
  "name": "issue ownership",
  "from_columns": "Q7r(\\d+)c(\\d+)",  "res_columns": "Q7r\\1_R\\2",
  "agg_index": 2,
  "groups": {"1": {"1": "economics", "2": "healthcare"}},
  "na_vals": ["NO TO: ..."],
  "translate_values": {"1": "TS-LKD", "2": "LSDP", "99": "No party"}
}
```

**Step-by-step processing of column `Q7r1c2`:**

```
1. REGEX MATCH
   Column "Q7r1c2" matches "Q7r(\d+)c(\d+)"
   → group 1 = "1", group 2 = "2"

2. IDENTIFY ROLES
   agg_index = 2, so group 2 is the aggregation axis (party).
   Non-agg groups: group 1 = "1" → subgroup identifier.

3. SUBGROUP ASSIGNMENT
   groups["1"]["1"] = "economics"
   → This column belongs to the "economics" subgroup.
   → Block name: "issue ownership_topk_economics"

4. NA FILTERING
   Raw cell: "Lietuvos socialdemokratų partija (Mindaugas Sinkevičius)"
   Matches na_vals? No → selected.
   Raw cell: "NO TO: Lietuvos socialdemokratų partija..."
   Matches na_vals? Yes → replaced with None (not selected).

5. COLLAPSE (LEFT-PACK, UNORDERED)
   Within the "economics" subgroup (Q7r1c1, Q7r1c2, ..., Q7r1c99):
   Replace non-NA cells with their agg-group value (group 2 capture).
   "Lietuvos socialdemokratų..." → "2" (from c2)
   "NO TO: ..." → None
   Then throw NAs right, producing columns Q7r1_R1, Q7r1_R2, ...
   The R-index is NOT a ranking — it is just the position after left-packing the
   selected values in their original agg-group column order. R1 holds the lowest
   agg-group value that the respondent selected, R2 the next, etc. Consumers
   should treat the R-columns as an unordered multi-select of the agg-axis.

6. TRANSLATE VALUES
   translate_values: "2" → "LSDP"
   Cell "2" in Q7r1_R1 becomes "LSDP".

7. OUTPUT
   Block "issue ownership_topk_economics":
     columns: {Q7r1_R1: ColumnMeta(...), Q7r1_R2: ColumnMeta(...)}
     Cell values: "TS-LKD", "LSDP", "No party", etc.
```

### MaxDiff example: `Q6_3best` in WEB maxdiff

**Annotation (MaxDiffBlock top-level — `type` discriminator dispatches to the MaxDiff path):**
```json
{
  "type": "maxdiff",
  "name": "maxdiff",
  "best_columns": "Q6_(\\d+?)best",  "worst_columns": "Q6_(\\d+?)worst",
  "set_columns": "Q6_\\1set",
  "setindex_column": ["Q6_Version", {"continuous": true}],
  "items": {"1": "Ekonomika", "2": "Sveikata", "3": "Švietimas"},
  "choice_sets": [[[1, 2, 3], [2, 3, 1]], [[3, 1, 2], [1, 3, 2]]],
  "translate": {"Ekonomika": "Economy", "Sveikata": "Health", "Švietimas": "Education"}
}
```

**Step-by-step processing of column `Q6_3best`:**

```
1. REGEX MATCH
   Column "Q6_3best" matches "Q6_(\d+?)best"
   → group 1 = "3", this is a "best" column for question 3.

2. DERIVE SET COLUMN
   set_columns pattern "Q6_\1set" → "Q6_3set" (the set shown for question 3).

3. RESOLVE SETS VIA setindex_column
   Row has Q6_Version = 1 (respondent saw version 1).
   choice_sets[0] = [[1,2,3], [2,3,1]] → question 3's set is choice_sets[0][2] = [2,3,1].
                                           (question index is 0-based internally)
   Items at indices 2,3,1:
     items["2"] = "Sveikata", items["3"] = "Švietimas", items["1"] = "Ekonomika"
   → Q6_3set cell becomes ["Sveikata", "Švietimas", "Ekonomika"]

4. TRANSLATE CELL VALUES
   Q6_3best raw cell: "Švietimas" (respondent picked this as best in question 3).
   translate: "Švietimas" → "Education"
   → Q6_3best cell becomes "Education".

   Q6_3set cell: ["Sveikata", "Švietimas", "Ekonomika"]
   → translated to ["Health", "Education", "Economy"]

5. CATEGORICAL DTYPE
   effective_topics = ["Economy", "Health", "Education"] (items values run through translate)
   Q6_3best column gets Categorical dtype with these categories.

6. OUTPUT
   Block "maxdiff_maxdiff":
     columns: {Q6_1best, Q6_1set, Q6_1worst, Q6_2best, Q6_2set, Q6_2worst,
               Q6_3best, Q6_3set, Q6_3worst, Q6_Version}
     Q6_3best values: "Education", "Economy", ... (translated)
     Q6_3set values: ["Health", "Education", "Economy"], ... (translated lists)
```

### How a downstream consumer (e.g. salk_internal_package) uses the output

Under Option A (see Part 2), the processed block *is* a `TopKBlock` or `MaxDiffBlock`. Consumers isinstance-check on the stored block and call `segments()` directly — no `.create` traversal, no regex parsing.

```python
# TopK: one block per subgroup; one segment per block.
block = data_meta.structure["ownership_economics"]
assert isinstance(block, TopKBlock)
segments = block.segments()
# → [(["Q7r1_R1","Q7r1_R2"], None, False)]

# MaxDiff: one block, two segments per question (best > set > worst).
block = data_meta.structure["maxdiff_maxdiff"]
assert isinstance(block, MaxDiffBlock)
segments = block.segments()
# → [([Q6_1best],[Q6_1set],True), ([Q6_2best],[Q6_2set],True), ...,
#    ([Q6_1set],[Q6_1worst],True), ([Q6_2set],[Q6_2worst],True), ...]
```

The `segments()` method takes no arguments — each block carries its own resolved column lists (`from_columns` / `res_columns` for TopK; `best_columns` / `worst_columns` / `set_columns` for MaxDiff). See **Part 2** at the end of this document for the full wiring story.

---

## Background: what was wrong

Three distinct concerns were tangled across multiple fields:

| Concern | Old field(s) | Problem |
|---|---|---|
| **Subgroup naming** — which regex groups form subgroups, what human name for each | `row_labels` (topk), nothing (maxdiff) | `row_labels` only handled 1 dimension; no multi-dimensional support |
| **Agg-axis value translation** — after topk collapses to rankings, map the numeric values to display names | `translate_after` (topk), `scale.translate` (topk) | Two fields doing the same job depending on whether data has raw strings or codes |
| **Item labeling** — what does index 1 mean as a topic/item | `topics` or `row_labels` (maxdiff), `translate_after` (topk) | `row_labels` meant different things on topk vs maxdiff |

The CATI issue_importance_top3 used `translate_after: {"1": "Cost of living"}` for the same purpose that the WEB maxdiff used `row_labels: {"1": "Pragyvenimo..."}` — there was no consistent vocabulary.

---

## New Schema

### TopKBlock

```python
class TopKBlock(PBase):
    type: Literal["topk"] = "topk"
    k: Union[int, Literal["max"]] = "max"
    from_columns: Union[str, List[str]]
    res_columns: Union[str, List[str]]
    agg_index: int = -1
    na_vals: Optional[List[str]] = []
    from_prefix: Optional[str] = None

    # Subgroup naming: one entry per non-agg regex group.
    # Key = regex group number (as string, 1-based, skipping agg_index).
    # Value = dict mapping each group value → human-readable label.
    # Labels become part of the generated block name.
    groups: Optional[Dict[str, Dict[str, str]]] = None

    # Agg-axis value translation: maps the aggregation group value
    # (after topk ranking) to a display name.
    # Applied to cell values in the result DataFrame.
    translate_values: Optional[Dict[str, str]] = None
```

### MaxDiffBlock

```python
class MaxDiffBlock(PBase):
    type: Literal["maxdiff"] = "maxdiff"
    name: Optional[str] = None
    best_columns: Union[str, List[str]]
    worst_columns: Union[str, List[str]]
    set_columns: Optional[Union[str, List[str]]] = None
    setindex_column: Optional[Union[str, List[object]]] = None

    # Item labeling: maps the 1-based index to the item name.
    # These are the "topics" being compared in the maxdiff experiment.
    # Values should be in the original language of the data.
    items: Dict[str, str]

    # Choice sets: sets[version][question] = list of item indices shown.
    choice_sets: Optional[List[List[List[int]]]] = None

    # Translation: maps original-language item names → display names.
    # Applied to cell values (best/worst columns) and categories.
    # If omitted, items values are used as-is.
    translate: Optional[Dict[str, str]] = None
```

### Removed fields

| Field | Replacement |
|---|---|
| `MaxDiffBlock.topics` | `items` (dict, not list) |
| `MaxDiffBlock.sets` | `choice_sets` (unchanged) |
| `MaxDiffBlock.row_labels` | `items` |
| `TopKBlock.row_labels` | `groups` |
| `TopKBlock.translate_after` | `translate_values` |
| `scale.translate` on maxdiff blocks | `MaxDiffBlock.translate` |
| `constants.topics` / `constants.sets` | moved onto the MaxDiff block as `items` / `choice_sets` |

---

## How each field works

### `groups` on TopKBlock

Given `from_columns: "Q7r(\\d+)c(\\d+)"` with `agg_index: 2`:

- Group 1 = issue area, Group 2 = party (aggregated over)
- `groups` labels the non-agg groups. Here only group 1 is non-agg:

```json
"groups": {
  "1": {"1": "economics", "2": "healthcare", "3": "education"}
}
```

This produces separate blocks named:
- `issue ownership_topk_economics` (columns from Q7r1c*)
- `issue ownership_topk_healthcare` (columns from Q7r2c*)
- `issue ownership_topk_education` (columns from Q7r3c*)

**Multi-dimensional example:** Given `from_columns: "Q6_(\\w+)_(\\d+)_(\\d+)"`, `agg_index: 3`:

```json
"groups": {
  "1": {"A": "Estonia", "B": "Latvia"},
  "2": {"1": "economics", "2": "healthcare"}
}
```

Produces blocks: `..._topk_Estonia_economics`, `..._topk_Estonia_healthcare`, `..._topk_Latvia_economics`, `..._topk_Latvia_healthcare`. The label is built by joining group labels in group-number order with `_`.

**No subgroups:** When `from_columns` has only one regex group (which is the agg group), `groups` is omitted. A single block is produced named `{name}_topk`.

### `translate_values` on TopKBlock

After topk processing, cell values are the agg-axis group values ("1", "2", "99", ...). `translate_values` maps these to display names:

```json
"translate_values": {"1": "TS-LKD", "2": "LSDP", "99": "No party"}
```

This replaces both `translate_after` and the pattern of using `scale.translate` for the same purpose. The field name makes clear it operates on **values** (cell contents), not on group/block names.

### `items` on MaxDiffBlock

Maps 1-based indices to the item labels in the **original language** of the data:

```json
"items": {
  "1": "Pragyvenimo išlaidos ir kainos",
  "2": "Atlyginimai ir perkamoji galia"
}
```

These are the items being ranked. When `setindex_column` is used, choice_sets reference items by these indices. When set columns contain explicit topic names, the values must match the `items` values.

### `translate` on MaxDiffBlock

Optional. Maps original-language item names to display names:

```json
"translate": {
  "Pragyvenimo išlaidos ir kainos": "Cost of living and prices",
  "Atlyginimai ir perkamoji galia": "Wages and purchasing power"
}
```

If provided, categories and cell values in best/worst columns are mapped through this dict. If omitted, item names are used as-is (useful when items are already in the target language).

---

## Generated block naming convention

TopK and MaxDiff blocks produce output siblings with predictable names:

| Block type | Naming pattern | Example |
|---|---|---|
| TopK, no subgroups | `{block_name}_topk` | `issue_importance_top3_topk` |
| TopK, 1 subgroup | `{block_name}_topk_{group_label}` | `issue ownership_topk_economics` |
| TopK, N subgroups | `{block_name}_topk_{label1}_{label2}_...` | `survey_topk_Estonia_economics` |
| MaxDiff | `{block_name}_maxdiff` | `maxdiff_maxdiff` |

These names are stable identifiers. Downstream consumers (e.g. model_desc in salk_internal_package) reference blocks by these names and retrieve the specialized block from `data_meta.structure[block_name]`. The block's `.columns` dict contains all processed result columns.

### What a processed block provides

Under Option A (Part 2), the processed sibling *is* a `TopKBlock` or `MaxDiffBlock` — not a `ColumnBlockMeta` wrapping one. Each output block contains:

- `type`: `"topk"` or `"maxdiff"` — the discriminator on the specialized subclass.
- `name`: the predictable block name (see table above).
- `columns`: dict of column_name → `ColumnMeta` (all result columns, fully processed).
- `scale`: `BlockScaleMeta` with categories (translated if `translate`/`translate_values` was applied).
- **Resolved column lists** in place of regex patterns: `from_columns` / `res_columns` (TopK) or `best_columns` / `worst_columns` / `set_columns` (MaxDiff), as `List[str]`.
- **Input-only directives cleared to `None`:** `groups`, `translate_values` (TopK); `items`, `translate`, `choice_sets` (MaxDiff). These are processing instructions, not facts about the output.

Consumers call `block.segments()` to obtain `OrdinalRanking`-shaped segments directly (see Part 2). The preserved input block (under the original `name`) is demoted to a plain `ColumnBlockMeta` that keeps the raw source columns but none of the processing directives.

### ColumnBlockMeta output: TopK (issue ownership, economics subgroup)

This is what `data_meta.structure["issue ownership_topk_economics"]` looks like after processing. The Leedu annotation has `Q7r(\d+)c(\d+)` with `agg_index: 2`, and the economics subgroup corresponds to group 1 value "1" (i.e. columns Q7r1c*).

```python
ColumnBlockMeta(
    name="issue ownership_topk_economics",
    scale=BlockScaleMeta(
        categories=["TS-LKD", "LSDP", "NA", "DSVL", "LVŽS", "LS", "Other", "No party"],
        colors="party_colors",
    ),
    columns={
        "Q7r1_R1": ColumnMeta(
            categories=["TS-LKD", "LSDP", "NA", "DSVL", "LVŽS", "LS", "Other", "No party"],
        ),
        "Q7r1_R2": ColumnMeta(
            categories=["TS-LKD", "LSDP", "NA", "DSVL", "LVŽS", "LS", "Other", "No party"],
        ),
        # ... up to Q7r1_RN where N depends on how many parties each respondent selected
    },
    generated=False,
    hidden=False,
    create=TopKBlock(
        type="topk",
        from_columns="Q7r(\\d+)c(\\d+)",
        res_columns="Q7r\\1_R\\2",
        agg_index=2,
        groups={"1": {"1": "economics", "2": "healthcare", "3": "education",
                       "4": "foreign_policy", "5": "nato"}},
        na_vals=["NO TO: Tėvynės sąjunga...", "..."],
        translate_values={"1": "TS-LKD", "2": "LSDP", "3": "NA", "4": "DSVL",
                          "5": "LVŽS", "6": "LS", "7": "Other", "99": "No party"},
    ),
)
```

**Key for salk_internal_package wiring (Option A, Part 2):**
- The processed block **is** a `TopKBlock`. `block.segments()` → `[(["Q7r1_R1","Q7r1_R2"], None, False)]` — one segment per subgroup block, reproducing `leedu_2026_full_package/model_desc.original.json:288-297`.
- Each of the 5 sibling blocks (economics, healthcare, education, foreign_policy, nato) is an independent `TopKBlock` with its own narrowed `from_columns` / `res_columns`. Input-only directives (`groups`, `translate_values`) are cleared on the siblings.
- `block.scale.categories` → the category list for the agg-axis values (e.g. the party set). The R-columns are an unordered left-packed multi-select, so `ordered=False` on the segment.
- Analyst writes `"res_cols": ["ownership_economics", ...]` in model_desc; SIP resolves each block name to its single segment via `block.segments()`.

### ColumnBlockMeta output: MaxDiff (Leedu WEB)

This is what `data_meta.structure["maxdiff_maxdiff"]` looks like after processing. The annotation has `Q6_(\d+?)best/worst`, 18 items in Lithuanian, translated to English.

```python
ColumnBlockMeta(
    name="maxdiff_maxdiff",
    scale=BlockScaleMeta(
        categories=[
            "Cost of living and prices",
            "Wages and purchasing power",
            "Tax burden",
            "Social benefits",
            "Access to healthcare",
            "Pension system",
            "Quality of education",
            "National defence capabilities",
            "Relations with Russia and Belarus",
            "Relations with the European Union",
            "Corruption and political transparency",
            "Immigration",
            "Population decline",
            "Regional development",
            "Business environment",
            "Environmental and climate policy",
            "Energy security",
            "Family and social values",
        ],
    ),
    columns={
        "Q6_Version": ColumnMeta(continuous=True, categories=[...]),  # set version index
        "Q6_1best":   ColumnMeta(categories=[...]),  # translated English categories
        "Q6_1set":    ColumnMeta(categories=[...]),  # set columns contain topic lists
        "Q6_1worst":  ColumnMeta(categories=[...]),
        "Q6_2best":   ColumnMeta(categories=[...]),
        "Q6_2set":    ColumnMeta(categories=[...]),
        "Q6_2worst":  ColumnMeta(categories=[...]),
        # ... Q6_3 through Q6_8 (8 question blocks × 3 columns each)
    },
    generated=False,
    hidden=False,
    create=MaxDiffBlock(
        type="maxdiff",
        best_columns="Q6_(\\d+?)best",
        worst_columns="Q6_(\\d+?)worst",
        set_columns="Q6_\\1set",
        setindex_column=["Q6_Version", {"continuous": True}],
        items={
            "1": "Pragyvenimo išlaidos ir kainos",
            "2": "Atlyginimai ir perkamoji galia",
            # ... all 18 Lithuanian item names
        },
        choice_sets=[[[1, 2, 4, 10, 12], [1, 6, 7, 15, 17], "..."]],
        translate={
            "Pragyvenimo išlaidos ir kainos": "Cost of living and prices",
            "Atlyginimai ir perkamoji galia": "Wages and purchasing power",
            # ... all 18 translations
        },
    ),
)
```

**Key for salk_internal_package wiring (Option A, Part 2):**
- The processed block **is** a `MaxDiffBlock`. `block.best_columns` / `worst_columns` / `set_columns` are `List[str]`, index-aligned by question.
- `block.segments()` returns the full `OrdinalRanking.structure` — two segments per question:
  ```python
  [([best_k], [set_k],   True) for k in range(Q)] +
  [([set_k],  [worst_k], True) for k in range(Q)]
  ```
  This reproduces `leedu_2026_full_package/model_desc.original.json:386-514`.
- `items` / `translate` are input-only directives; they are cleared on the output `MaxDiffBlock`. The translated item vocabulary lives in `block.scale.categories`. `OrdinalRanking.prepare()` infers `omc` from observed data values (`salk_internal_package/obs_models/ordinal_ranking.py:101-115`), so no separate vocabulary plumbing is needed.
- `choice_sets` is also input-only; a future latent model that needs per-row shown sets would consume it from the input annotation, not the output block.

---

## Concrete annotation examples (Leedu 2026)

### CATI issue_importance_top3 — topk, no subgroups

Raw data: `Q6r1`="Pragyvenimo išlaidos ir kainos" or "NO TO: ...", `Q6r2`=..., etc.

```json
{
  "type": "topk",
  "name": "issue_importance_top3",
  "columns": [],
  "from_columns": "Q6r(\\d+)",
  "res_columns": "Q6p_R\\1",
  "agg_index": 1,
  "na_vals": [
    "NO TO: Pragyvenimo išlaidos ir kainos",
    "NO TO: Prieiga prie sveikatos priežiūros paslaugų",
    "NO TO: Pensijų sistema",
    "NO TO: Švietimo kokybė",
    "NO TO: Valstybės gynybos pajėgumai",
    "NO TO: Santykiai su Rusija ir Baltarusija",
    "NO TO: Imigracija",
    "NO TO: Šeimos ir socialinės vertybės",
    "Nei viena iš šių",
    "NO TO: Nei viena iš šių",
    "Nežinau (instrukcija interviuotojui: neskaityti)",
    "NO TO: Nežinau (instrukcija interviuotojui: neskaityti)"
  ],
  "translate_values": {
    "1": "Cost of living and prices",
    "2": "Access to healthcare",
    "3": "Pension system",
    "4": "Quality of education",
    "5": "National defence capabilities",
    "6": "Relations with Russia and Belarus",
    "7": "Immigration",
    "8": "Family and social values",
    "98": "None of these",
    "99": "Don't know"
  },
  "scale": {"categories": "infer"}
}
```

**Produces:** block `issue_importance_top3_topk` with columns `Q6p_R1`, `Q6p_R2`, `Q6p_R3`. Cell values are "Cost of living and prices", "Access to healthcare", etc.

**What `translate_values` gives the user:** the mapping `"1" → "Cost of living and prices"` tells you that regex group value 1 (from Q6r**1**) means "Cost of living and prices". The original Lithuanian question text for Q6r1 was "Pragyvenimo išlaidos ir kainos" — this was the raw cell value before NA filtering. After topk processing, cell values become the group capture ("1", "2", ...), then `translate_values` maps them to English.

### Issue ownership — topk, 1 subgroup dimension

Raw data: `Q7r1c1` = "Tėvynės sąjunga..." or "NO TO: Tėvynės sąjunga..."

```json
{
  "type": "topk",
  "name": "issue ownership",
  "columns": ["Q7r1c1", "Q7r1c2", "...", "Q7r5c99"],
  "from_columns": "Q7r(\\d+)c(\\d+)",
  "res_columns": "Q7r\\1_R\\2",
  "agg_index": 2,
  "groups": {
    "1": {
      "1": "economics",
      "2": "healthcare",
      "3": "education",
      "4": "foreign_policy",
      "5": "nato"
    }
  },
  "na_vals": [
    "NO TO: Tėvynės sąjunga – Lietuvos krikščionys demokratai (Laurynas Kasčiūnas)",
    "NO TO: Lietuvos socialdemokratų partija (Mindaugas Sinkevičius)",
    "NO TO: Nemuno aušra (Remigijus Žemaitatis)",
    "NO TO: Demokratų sąjunga „Vardan Lietuvos\" (Saulius Skvernelis)",
    "NO TO: Lietuvos valstiečių ir žaliųjų sąjunga (Aurelijus Veryga)",
    "NO TO: Liberalų sąjūdis (Viktorija Čmilytė Nielsen)",
    "NO TO: Kitai partijai",
    "NO TO: Nepatikėčiau šios srities nei vienai partijai"
  ],
  "translate_values": {
    "1": "TS-LKD",
    "2": "LSDP",
    "3": "NA",
    "4": "DSVL",
    "5": "LVŽS",
    "6": "LS",
    "7": "Other",
    "99": "No party"
  },
  "scale": {"categories": "infer", "colors": "party_colors"}
}
```

**Produces 5 blocks:**
- `issue ownership_topk_economics` — columns `Q7r1_R1`, `Q7r1_R2`
- `issue ownership_topk_healthcare` — columns `Q7r2_R1`, `Q7r2_R2`
- `issue ownership_topk_education` — columns `Q7r3_R1`, `Q7r3_R2`
- `issue ownership_topk_foreign_policy` — columns `Q7r4_R1`, `Q7r4_R2`
- `issue ownership_topk_nato` — columns `Q7r5_R1`, `Q7r5_R2`

Cell values are "TS-LKD", "LSDP", etc. (applied by `translate_values`).

**What `groups` gives the user:** `"1": "economics"` tells you that Q7r**1**c* asks "Which party do you trust most on **economic** issues?". The model_desc can reference `issue ownership_topk_economics` by name.

### WEB maxdiff — maxdiff with translation

Raw data: `Q6_1best` = "Pragyvenimo išlaidos ir kainos" (Lithuanian), `Q6_Version` = 1..N.

```json
{
  "type": "maxdiff",
  "name": "maxdiff",
  "columns": [],
  "best_columns": "Q6_(\\d+?)best",
  "worst_columns": "Q6_(\\d+?)worst",
  "set_columns": "Q6_\\1set",
  "setindex_column": ["Q6_Version", {"continuous": true}],
  "items": {
    "1": "Pragyvenimo išlaidos ir kainos",
    "2": "Atlyginimai ir perkamoji galia",
    "3": "Mokesčių našta",
    "4": "Socialinės išmokos",
    "5": "Prieiga prie sveikatos priežiūros",
    "6": "Pensijų sistema",
    "7": "Švietimo kokybė",
    "8": "Valstybės gynybos pajėgumai",
    "9": "Santykiai su Rusija ir Baltarusija",
    "10": "Santykiai su Europos Sąjunga",
    "11": "Korupcija ir politinis skaidrumas",
    "12": "Imigracija",
    "13": "Gyventojų skaičiaus mažėjimas",
    "14": "Regioninė plėtra",
    "15": "Verslo aplinka",
    "16": "Aplinkos ir klimato politika",
    "17": "Energetinis saugumas",
    "18": "Šeimos ir socialinės vertybės"
  },
  "choice_sets": [
    [[1, 2, 4, 10, 12], [1, 6, 7, 15, 17], "..."]
  ],
  "translate": {
    "Pragyvenimo išlaidos ir kainos": "Cost of living and prices",
    "Atlyginimai ir perkamoji galia": "Wages and purchasing power",
    "Mokesčių našta": "Tax burden",
    "Socialinės išmokos": "Social benefits",
    "Prieiga prie sveikatos priežiūros": "Access to healthcare",
    "Pensijų sistema": "Pension system",
    "Švietimo kokybė": "Quality of education",
    "Valstybės gynybos pajėgumai": "National defence capabilities",
    "Santykiai su Rusija ir Baltarusija": "Relations with Russia and Belarus",
    "Santykiai su Europos Sąjunga": "Relations with the European Union",
    "Korupcija ir politinis skaidrumas": "Corruption and political transparency",
    "Imigracija": "Immigration",
    "Gyventojų skaičiaus mažėjimas": "Population decline",
    "Regioninė plėtra": "Regional development",
    "Verslo aplinka": "Business environment",
    "Aplinkos ir klimato politika": "Environmental and climate policy",
    "Energetinis saugumas": "Energy security",
    "Šeimos ir socialinės vertybės": "Family and social values"
  },
  "scale": {"categories": "infer"}
}
```

**Produces:** block `maxdiff_maxdiff` with columns `Q6_1best`, `Q6_1set`, `Q6_1worst`, ..., `Q6_8best`, `Q6_8set`, `Q6_8worst`, `Q6_Version`. Cell values and categories are English ("Cost of living and prices", etc.).

**What `items` + `translate` give the user:** `items` shows the original Lithuanian question/item text mapped to each index. `translate` shows how each Lithuanian item maps to the English display name. Together they provide a complete audit trail from index → original language → display language.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `salk_toolkit/validation.py` | Modify | New `TopKBlock` and `MaxDiffBlock` schemas per above. Remove `row_labels`, `topics`, `sets`, `translate_after`, backwards-compat validators. |
| `salk_toolkit/serialization.py` | Modify | Update serialization for renamed fields. Remove `topics`/`sets` alias handling. |
| `salk_toolkit/io.py` (topk) | Modify | `_create_topk_metas_and_dfs_regex`: use `groups` for multi-dimensional subgroup labeling, `translate_values` instead of `translate_after`. Emit output siblings as `TopKBlock` instances with resolved column lists (Option A, Part 2). |
| `salk_toolkit/io.py` (maxdiff) | Modify | `_create_maxdiff_metas_and_dfs`: use `items` instead of `topics`, `translate` (on the MaxDiff block itself) instead of `scale.translate`. Remove constants fallback. Emit output as a `MaxDiffBlock` instance with resolved column lists (Option A, Part 2). |
| `salk_toolkit/io.py` (constants) | Modify | Remove `topics`/`sets` constants fallback and deprecation warnings. |
| `tests/test_io.py` | Rewrite | Replace `TestBlockSchemaUnification` and existing topk/maxdiff tests with new comprehensive tests (see Task 2). |

---

## Task 1: Update validation.py and serialization.py schemas

**Files:** `salk_toolkit/validation.py`, `salk_toolkit/serialization.py`

- [ ] **Step 1: Rewrite TopKBlock**

Replace:
```python
class TopKBlock(PBase):
    type: Literal["topk"] = "topk"
    k: Union[int, Literal["max"]] = "max"
    from_columns: Union[str, List[str]]
    res_columns: Union[str, List[str]]
    agg_index: int = -1
    na_vals: Optional[List[str]] = []
    from_prefix: Optional[str] = None
    groups: Optional[Dict[str, Dict[str, str]]] = None
    translate_values: Optional[Dict[str, str]] = None
```

Remove `translate_after`, `row_labels`.

- [ ] **Step 2: Rewrite MaxDiffBlock**

Replace:
```python
class MaxDiffBlock(PBase):
    type: Literal["maxdiff"] = "maxdiff"
    name: Optional[str] = None
    best_columns: Union[str, List[str]]
    worst_columns: Union[str, List[str]]
    set_columns: Optional[Union[str, List[str]]] = None
    setindex_column: Optional[Union[str, List[object]]] = None
    items: Dict[str, str]
    choice_sets: Optional[List[List[List[int]]]] = None
    translate: Optional[Dict[str, str]] = None
```

Remove `topics`, `sets`, `row_labels`, `_migrate_legacy_fields` validator.

- [ ] **Step 3: Update serialization.py**

Remove any `topics`/`sets`/`row_labels`/`translate_after` alias handling.

- [ ] **Step 4: Run tests to see what breaks**

```bash
cd /Users/erik/salk/salk_toolkit && python -m pytest tests/test_io.py -x -v 2>&1 | head -60
```

Expected: many failures — tests still use old field names.

---

## Task 2: Rewrite tests

**Files:** `tests/test_io.py`

Remove the `TestBlockSchemaUnification` class entirely. Remove `test_maxdiff_constants_fallback_warns`. Update existing tests and add new ones to cover every combination.

- [ ] **Step 1: Rewrite topk test — no subgroups**

This tests `issue_importance_top3` pattern: single regex group = agg group, `translate_values` for cell value mapping.

```python
def test_topk_no_subgroups(self, meta_file, csv_file):
    """TopK with single regex group (= agg group), no subgroups.
    Pattern: Q6r(\\d+) — each column is an issue, selected or not.
    translate_values maps the numeric group value to English issue names.
    """
    meta = {
        "file": "test.csv",
        "structure": [
            {
                "type": "topk",
                "name": "issue_importance",
                "columns": ["Q6r1", "Q6r2", "Q6r3"],
                "from_columns": r"Q6r(\d+)",
                "res_columns": r"Q6p_R\1",
                "agg_index": 1,
                "na_vals": ["NO TO: Cost of living", "NO TO: Healthcare", "NO TO: Pensions"],
                "translate_values": {
                    "1": "Cost of living",
                    "2": "Healthcare",
                    "3": "Pensions",
                },
                "scale": {"categories": "infer"},
            }
        ],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q6r1": ["Cost of living", "NO TO: Cost of living", "Cost of living"],
        "Q6r2": ["NO TO: Healthcare", "Healthcare", "Healthcare"],
        "Q6r3": ["Pensions", "Pensions", "NO TO: Pensions"],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # Single block produced, named with _topk suffix (no subgroup label)
    assert "issue_importance_topk" in data_meta.structure

    # Cell values are the translated issue names
    assert data_df["Q6p_R1"].tolist() == ["Cost of living", "Healthcare", "Cost of living"]

    # Columns accessible by block name
    block = data_meta.structure["issue_importance_topk"]
    assert set(block.columns.keys()) == {"Q6p_R1", "Q6p_R2"}
```

- [ ] **Step 2: Rewrite topk test — 1 subgroup dimension with translate_values**

This tests the `issue ownership` pattern: 2 regex groups, agg over group 2, `groups` labels group 1, `translate_values` maps party codes.

```python
def test_topk_one_subgroup(self, meta_file, csv_file):
    """TopK with 2 regex groups, 1 subgroup dimension.
    Pattern: Q7r(\\d+)c(\\d+) — group 1 = issue area, group 2 = party (aggregated).
    groups labels group 1; translate_values maps party numbers to abbreviations.
    """
    meta = {
        "file": "test.csv",
        "structure": [
            {
                "type": "topk",
                "name": "issue ownership",
                "columns": ["Q7r1c1", "Q7r1c2", "Q7r2c1", "Q7r2c2"],
                "from_columns": r"Q7r(\d+)c(\d+)",
                "res_columns": r"Q7r\1_R\2",
                "agg_index": 2,
                "groups": {
                    "1": {"1": "economics", "2": "healthcare"},
                },
                "na_vals": ["not_selected"],
                "translate_values": {"1": "Party A", "2": "Party B"},
                "scale": {"categories": "infer"},
            }
        ],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q7r1c1": ["selected", "not_selected", "selected"],
        "Q7r1c2": ["not_selected", "selected", "not_selected"],
        "Q7r2c1": ["selected", "not_selected", "selected"],
        "Q7r2c2": ["selected", "selected", "not_selected"],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # Two subgroup blocks produced
    assert "issue ownership_topk_economics" in data_meta.structure
    assert "issue ownership_topk_healthcare" in data_meta.structure

    # Economics block has Q7r1_R* columns
    econ_block = data_meta.structure["issue ownership_topk_economics"]
    assert all(c.startswith("Q7r1_R") for c in econ_block.columns.keys())

    # Cell values are translated party names
    econ_cols = [c for c in data_df.columns if c.startswith("Q7r1_R")]
    assert data_df[econ_cols[0]].tolist() == ["Party A", "Party B", "Party A"]

    # Output sibling IS a TopKBlock with narrowed resolved column lists; input-only
    # directives (`groups`, `translate_values`) are cleared.
    assert isinstance(econ_block, TopKBlock)
    assert econ_block.type == "topk"
    assert isinstance(econ_block.from_columns, list)
    assert econ_block.groups is None
    assert econ_block.translate_values is None
```

- [ ] **Step 3: Add topk test — 2 subgroup dimensions (multi-dimensional)**

Tests the hypothetical `Q_(country)_(issue)_(party)` pattern.

```python
def test_topk_two_subgroup_dimensions(self, meta_file, csv_file):
    """TopK with 3 regex groups, 2 subgroup dimensions.
    Pattern: Q_(\\w+)_(\\d+)_(\\d+) — group 1 = country, group 2 = issue, group 3 = party (agg).
    groups labels both non-agg groups; block names combine labels.
    """
    meta = {
        "file": "test.csv",
        "structure": [
            {
                "type": "topk",
                "name": "survey",
                "columns": ["Q_A_1_1", "Q_A_1_2", "Q_A_2_1", "Q_A_2_2",
                             "Q_B_1_1", "Q_B_1_2", "Q_B_2_1", "Q_B_2_2"],
                "from_columns": r"Q_(\w+)_(\d+)_(\d+)",
                "res_columns": r"Q_\1_\2_R\3",
                "agg_index": 3,
                "groups": {
                    "1": {"A": "Estonia", "B": "Latvia"},
                    "2": {"1": "economics", "2": "healthcare"},
                },
                "na_vals": ["no"],
                "translate_values": {"1": "Party X", "2": "Party Y"},
                "scale": {"categories": "infer"},
            }
        ],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q_A_1_1": ["yes", "no"],  "Q_A_1_2": ["no", "yes"],
        "Q_A_2_1": ["yes", "yes"], "Q_A_2_2": ["no", "no"],
        "Q_B_1_1": ["no", "yes"],  "Q_B_1_2": ["yes", "no"],
        "Q_B_2_1": ["yes", "no"],  "Q_B_2_2": ["no", "yes"],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # Four blocks: all combinations of country × issue
    assert "survey_topk_Estonia_economics" in data_meta.structure
    assert "survey_topk_Estonia_healthcare" in data_meta.structure
    assert "survey_topk_Latvia_economics" in data_meta.structure
    assert "survey_topk_Latvia_healthcare" in data_meta.structure

    # Estonia_economics block has Q_A_1_R* columns
    ee_block = data_meta.structure["survey_topk_Estonia_economics"]
    assert all(c.startswith("Q_A_1_R") for c in ee_block.columns.keys())
```

- [ ] **Step 4: Rewrite maxdiff test — with items and translate**

Tests the WEB maxdiff pattern with Lithuanian items + English translation.

```python
def test_maxdiff_with_items_and_translate(self, meta_file, csv_file):
    """MaxDiff with items (original language) and translate (to display language).
    items maps indices to Lithuanian topic names.
    translate maps Lithuanian → English.
    """
    items = {"1": "Ekonomika", "2": "Sveikata", "3": "Švietimas"}
    translate = {"Ekonomika": "Economy", "Sveikata": "Health", "Švietimas": "Education"}
    choice_sets = [
        [[1, 2, 3], [2, 3, 1], [1, 3, 2]],  # version 1: 3 questions, 3 items each
        [[3, 1, 2], [1, 2, 3], [3, 2, 1]],  # version 2
    ]
    meta = {
        "file": "test.csv",
        "structure": [
            {
                "type": "maxdiff",
                "name": "maxdiff",
                "columns": [],
                "best_columns": r"Q_(\d+)best",
                "worst_columns": r"Q_(\d+)worst",
                "set_columns": r"Q_\1set",
                "setindex_column": ["Q_Version", {"continuous": True}],
                "items": items,
                "choice_sets": choice_sets,
                "translate": translate,
                "scale": {"categories": "infer"},
            }
        ],
    }
    write_json(meta_file, meta)
    df = pd.DataFrame({
        "Q_Version": [1, 2, 1],
        "Q_1best": ["Ekonomika", "Švietimas", "Sveikata"],
        "Q_1worst": ["Sveikata", "Ekonomika", "Ekonomika"],
        "Q_2best": ["Švietimas", "Ekonomika", "Švietimas"],
        "Q_2worst": ["Ekonomika", "Sveikata", "Ekonomika"],
        "Q_3best": ["Ekonomika", "Švietimas", "Ekonomika"],
        "Q_3worst": ["Švietimas", "Ekonomika", "Švietimas"],
    })
    df.to_csv_file(csv_file)
    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # Output block IS a MaxDiffBlock (not a wrapper with a nested 'create')
    block = data_meta.structure["maxdiff_maxdiff"]
    assert isinstance(block, MaxDiffBlock)
    assert "Q_1best" in block.columns
    assert "Q_1set" in block.columns

    # Cell values and categories are in the display language
    assert data_df["Q_1best"].tolist() == ["Economy", "Education", "Health"]
    assert set(data_df["Q_1best"].cat.categories) == {"Economy", "Health", "Education"}

    # Resolved column lists replace regex patterns on output; input-only directives cleared
    assert isinstance(block.best_columns, list)
    assert isinstance(block.worst_columns, list)
    assert isinstance(block.set_columns, list)
    assert block.items is None
    assert block.translate is None
    assert block.choice_sets is None
```

- [ ] **Step 5: Add maxdiff test — items without translate**

Tests when items are already in the display language (no translation needed).

```python
def test_maxdiff_items_no_translate(self, meta_file, csv_file):
    """MaxDiff with items already in target language, no translate needed."""
    items = {"1": "Economy", "2": "Health", "3": "Education"}
    meta = {
        "file": "test.csv",
        "structure": [
            {
                "type": "maxdiff",
                "name": "maxdiff",
                "columns": [],
                "best_columns": r"Q_(\d+)best",
                "worst_columns": r"Q_(\d+)worst",
                "set_columns": r"Q_\1set",
                "items": items,
                "scale": {"categories": "infer"},
            }
        ],
    }
    # ... build df with English item names as best/worst values,
    # set columns contain explicit topic lists ...

    data_df, data_meta = read_and_process_data(str(meta_file), return_meta=True)

    # Cell values are the item names directly (no translation)
    assert data_df["Q_1best"].iloc[0] in items.values()
```

- [ ] **Step 6: Update existing test_topk_create_block**

Update the existing test to use `translate_values` instead of `translate_after`. Keep the same data and assertions, just change the field name.

- [ ] **Step 7: Remove old tests**

Remove:
- `TestBlockSchemaUnification` class entirely (all 6 tests)
- `test_maxdiff_constants_fallback_warns`

- [ ] **Step 8: Run full test suite**

```bash
cd /Users/erik/salk/salk_toolkit && python -m pytest tests/test_io.py -v
```

---

## Task 3: Update io.py topk processing

**Files:** `salk_toolkit/io.py`

- [ ] **Step 1: Update `_create_topk_metas_and_dfs_regex`**

Replace `_subgroup_label` to use `groups` dict-of-dicts:

```python
def _subgroup_label(
    subgroup_id: tuple[str, ...],
    groups: dict[str, dict[str, str]] | None,
    n_groups: int,
    agg_ind: int,  # 0-based (already adjusted from user-facing agg_index)
) -> str:
    # Map subgroup_id tuple positions back to original 1-based regex group numbers,
    # skipping the agg group.
    non_agg_group_numbers = [i + 1 for i in range(n_groups) if i != agg_ind]
    if groups is None:
        return "_".join(subgroup_id)
    labels = []
    for pos, val in enumerate(subgroup_id):
        group_key = str(non_agg_group_numbers[pos])
        if group_key in groups and val in groups[group_key]:
            labels.append(groups[group_key][val])
        else:
            labels.append(val)
    return "_".join(labels)
```

The `groups` dict keys are the original (1-based) regex group numbers. `subgroup_id` is built by `subgroup_id.pop(agg_ind)` on the raw groups tuple, so tuple position `p` maps to the `p`-th non-agg group. The helper above explicitly computes that mapping.

Replace `translate_after` usage with `translate_values`:

```python
if create.translate_values is not None:
    sdf = sdf.replace(create.translate_values)
```

- [ ] **Step 2: Emit the output subgroup as a `TopKBlock` instance (not a plain dict)**

Under Option A (Part 2), each output subgroup *is* a `TopKBlock` — not a `ColumnBlockMeta` with a nested `create`. Build it through the `_build_topk_output_block` helper with resolved `from_columns` / `res_columns` lists, and leave input-only directives (`groups`, `translate_values`) cleared so they don't survive onto the siblings. Downstream consumers read the output block directly via `isinstance(block, TopKBlock)` + `block.segments()`.

- [ ] **Step 3: Remove `scale.translate` application from topk**

The `scale.translate` cell-value mapping currently applied at io.py:598-608 is replaced by `translate_values`. Remove the `scale_translate` logic.

- [ ] **Step 4: Run topk tests**

```bash
cd /Users/erik/salk/salk_toolkit && python -m pytest tests/test_io.py -k topk -v
```

---

## Task 4: Update io.py maxdiff processing

**Files:** `salk_toolkit/io.py`

- [ ] **Step 1: Update `_create_maxdiff_metas_and_dfs`**

Accept a `MaxDiffBlock` directly (Option A, Part 2) instead of unwrapping a nested `create`. Read `items`/`translate` from the block itself:
```python
items = block.items  # Dict[str, str], required
translate = block.translate or {}
topics = [items[k] for k in sorted(items, key=int)]
effective_topics = [translate.get(t, t) for t in topics] if translate else list(topics)
```

Remove `topics` and `sets` function parameters. Remove constants fallback logic and deprecation warnings.

- [ ] **Step 2: Replace `scale.translate` with `block.translate`**

Input-side translation lives on the MaxDiff block itself (no longer piggy-backing on `scale.translate`).

- [ ] **Step 3: Emit the output as a `MaxDiffBlock` instance**

Under Option A (Part 2), the output is a `MaxDiffBlock` with resolved `best_columns` / `worst_columns` / `set_columns` lists (index-aligned by question). Input-only directives (`items`, `translate`, `choice_sets`) are cleared on output; translated vocabulary lives in `scale.categories`. Downstream consumers read the output block directly via `isinstance(block, MaxDiffBlock)` + `block.segments()`.

- [ ] **Step 4: Remove constants fallback in caller**

In the function that calls `_create_maxdiff_metas_and_dfs` (around io.py:1318-1331), remove the `constants.get("topics")` / `constants.get("sets")` fallback.

- [ ] **Step 5: Run maxdiff tests**

```bash
cd /Users/erik/salk/salk_toolkit && python -m pytest tests/test_io.py -k maxdiff -v
```

---

## Task 5: Final validation

- [ ] **Step 1: Run full salk_toolkit test suite**

```bash
cd /Users/erik/salk/salk_toolkit && python -m pytest tests/ -v
```

- [ ] **Step 2: Type check**

```bash
cd /Users/erik/salk/salk_toolkit && pyright salk_toolkit
```

- [ ] **Step 3: Lint**

```bash
cd /Users/erik/salk/salk_toolkit && ruff check
```

- [ ] **Step 4: Fix any issues and commit**

---

# Part 2: Wiring to salk_internal_package model_desc

**Status:** Schema refactor (Part 1) is landed on the `create-refactor` branch. This part describes the remaining work on the salk_toolkit side so that processed TopK/MaxDiff blocks are directly consumable by `salk_internal_package` (SIP), and sketches (without implementing) the one-hook change on the SIP side.

**Target:** Given a processed block carrying TopK or MaxDiff structure, SIP's `run_stack` accepts `"res_cols": ["<block_name>"]` and implicitly constructs the same `OrdinalRanking` that `leedu_2026_full_package/model_desc.original.json:248-355` (issue_ownership) and `:358-525` (maxdiff_model) currently specify by hand.

**Scope (this part):**
- salk_toolkit — promote `TopKBlock` / `MaxDiffBlock` to first-class block types that *are* the stored block (Option A below). Drop `ColumnBlockMeta.create`. Add a `type` discriminator on the block union. Resolve regex to explicit column lists. Add `segments()` methods. Migrate tests. No changes to salk_internal_package.
- salk_internal_package — not implemented here. A sketch is included so the reader can see the intended consumption pattern and understand why the toolkit changes take the shape they do.

---

## Motivation: the shape we want to reproduce

The leedu production `model_desc.json` writes out ordinal-ranking structure by hand. For issue_ownership (`model_desc.original.json:288-297`):

```json
{
  "name": "ownership_economics",
  "structure": [[["Q7r1_R1", "Q7r1_R2"], null]],
  "ordered": false
}
```

For the maxdiff model (`model_desc.original.json:386-514`):

```json
"structure": [
  [["Q6_1best"], ["Q6_1set"]],
  [["Q6_2best"], ["Q6_2set"]],
  ...
  [["Q6_1set"], ["Q6_1worst"]],
  [["Q6_2set"], ["Q6_2worst"]],
  ...
]
```

Both shapes are mechanically derivable from the processed block alone, once three things hold:
1. The stored block **is** a `TopKBlock` or `MaxDiffBlock` — no wrapper `ColumnBlockMeta` + nested `create` field.
2. Its `*_columns` fields carry **explicit column lists** (regex resolved), aligned by question index for MaxDiff.
3. It exposes a `segments()` method returning `(a, b, ordered)` tuples ready for `OrdinalRanking.structure`.

Reference for the segment shape: `salk_internal_package/tests/obs_models/test_variant.py:68` (`test_variant_two_ordinal_rankings_maxdiff_and_top3`) constructs explicit `ComplexOrankingSegment(a=..., b=..., ordered=...)` lists by hand — the shape `segments()` must produce.

---

## Option A: Promote `TopKBlock` / `MaxDiffBlock` to full block types

Today, `data_meta.structure[name]` is always a `ColumnBlockMeta`, which may carry a nested `create: TopKBlock | MaxDiffBlock | None`. Consumers reach into `.create`.

Under Option A, `TopKBlock` and `MaxDiffBlock` **inherit from `ColumnBlockMeta`** and are themselves the stored block. A Pydantic discriminated union on a new `type` field (`"plain" | "topk" | "maxdiff"`) dispatches validation. `ColumnBlockMeta.create` is removed.

```python
class ColumnBlockMeta(PBase):
    type: Literal["plain"] = "plain"
    name: str
    scale: Optional[BlockScaleMeta] = None
    columns: ColSpec
    subgroup_transform: Optional[str] = None
    generated: bool = False
    hidden: bool = False

    @model_validator(mode="after")
    def merge_scale_with_columns(self, info: ValidationInfo) -> Self: ...


class TopKBlock(ColumnBlockMeta):
    type: Literal["topk"] = "topk"  # type: ignore[assignment]
    k: Union[int, Literal["max"]] = "max"
    from_columns: Union[str, List[str]]
    res_columns: Union[str, List[str]]
    agg_index: int = -1
    na_vals: Optional[List[str]] = []
    from_prefix: Optional[str] = None
    groups: Optional[Dict[str, Dict[str, str]]] = None         # input-only: subgroup naming
    translate_values: Optional[Dict[str, str]] = None          # input-only: cell translation

    def segments(self) -> list[tuple[list[str], list[str] | None, bool]]:
        return [(list(self.columns.keys()), None, False)]


class MaxDiffBlock(ColumnBlockMeta):
    type: Literal["maxdiff"] = "maxdiff"  # type: ignore[assignment]
    best_columns: Union[str, List[str]]
    worst_columns: Union[str, List[str]]
    set_columns: Optional[Union[str, List[str]]] = None
    setindex_column: Optional[Union[str, List[object]]] = None
    items: Dict[str, str]                                       # input-only: index → item
    choice_sets: Optional[List[List[List[int]]]] = None
    translate: Optional[Dict[str, str]] = None                  # input-only: language map

    def segments(self) -> list[tuple[list[str], list[str], bool]]:
        # Precondition: best/worst/set_columns resolved to List[str] by io.py,
        # index-aligned (question k = list[k]).
        best = self.best_columns; worst = self.worst_columns; sets = self.set_columns
        assert isinstance(best, list) and isinstance(worst, list) and isinstance(sets, list)
        return (
            [([best[k]], [sets[k]],  True) for k in range(len(best))] +
            [([sets[k]], [worst[k]], True) for k in range(len(best))]
        )


BlockUnion = Annotated[
    Union[TopKBlock, MaxDiffBlock, ColumnBlockMeta],
    Field(discriminator="type"),
]
BlockSpec = Annotated[Dict[str, BlockUnion], BeforeValidator(_cb_lst_to_dict)]
```

`segments()` takes no arguments — every block carries its own resolved column lists.

### Subgroup fan-out under Option A

TopK with a `groups` directive produces N sibling blocks — one per subgroup. Each sibling is a fresh `TopKBlock` instance with *its own* narrowed `from_columns` / `res_columns` / `columns`, independent of the other siblings. The annotation-level `groups` / `translate_values` fields are processing directives; they are consumed during io.py expansion and **not carried onto the output siblings** (the narrowed column lists already encode which subgroup a sibling represents).

```python
# Input: one TopKBlock with regex + groups directive
TopKBlock(
    type="topk", name="issue ownership",
    from_columns=r"Q7r(\d+)c(\d+)",
    res_columns=r"Q7r\1_R\2",
    agg_index=2,
    groups={"1": {"1": "economics", "2": "healthcare", ...}},
    translate_values={"1": "TS-LKD", ...},
    scale=BlockScaleMeta(categories="infer", colors="party_colors"),
    columns={},
)

# Output: N siblings, each a TopKBlock with resolved per-subgroup columns
data_meta.structure["ownership_economics"]   # TopKBlock
  .name          = "ownership_economics"
  .from_columns  = ["Q7r1c1", "Q7r1c2", ...]
  .res_columns   = ["Q7r1_R1", "Q7r1_R2"]
  .columns       = {"Q7r1_R1": ColumnMeta(...), "Q7r1_R2": ColumnMeta(...)}
  .scale         = BlockScaleMeta(categories=[...])
  .segments()    → [(["Q7r1_R1","Q7r1_R2"], None, False)]

data_meta.structure["ownership_healthcare"]  # TopKBlock, same shape, Q7r2_* columns
...
```

MaxDiff has no subgroup dimension — a single input `MaxDiffBlock` produces a single output `MaxDiffBlock` with resolved index-aligned `best_columns` / `worst_columns` / `set_columns`.

### Why the `type` discriminator is a `plain` / `topk` / `maxdiff` enum

Annotations today look either like `{"name": "...", "columns": [...]}` (plain) or `{"name": "...", "create": {"type": "topk", ...}}` (specialized). Under Option A the inner `create` disappears and the outer block carries the discriminator directly. `type: "plain"` is the explicit default on `ColumnBlockMeta` so that:

1. **Omitted `type` still validates as plain.** Pydantic's discriminated union accepts the default when no discriminator is present — existing plain annotations need no edits.
2. **Specialized blocks migrate by hoisting `create.type` up.** An old `{"name":"X", "create":{"type":"topk", "from_columns":...}}` becomes `{"type":"topk", "name":"X", "from_columns":...}`. Non-backward-compatible, which matches this document's "no backwards compatibility" stance.
3. **Union members are exhaustive.** Adding a future specialized block (e.g. `ConjointBlock`) means registering a new `type` value and appending to the union — no separate "create" plumbing.

Serialization emits the `type` field only when it's non-default (i.e. topk / maxdiff), so plain blocks stay visually unchanged in serialized JSON.

### Input directives vs output state

Fields split into two populations. Input-only fields (`groups`, `translate_values`, `items`, `translate`, `choice_sets` as currently used) are consumed by io.py during processing and have no meaning on output blocks. Output-facing fields (`from_columns`, `res_columns`, `best_columns`, `worst_columns`, `set_columns`, `columns`, `scale`) carry resolved state that downstream consumers read.

We keep both on the same class to avoid duplicating the schema. io.py is responsible for clearing / populating fields appropriately at the input→output boundary. Two options for the cleanup:
- **Clear on emit** — io.py constructs output blocks with `groups=None`, `translate_values=None`, etc.
- **Subclass** — split into `TopKInput` / `TopKOutput`. Rejected for now; doubles the schema surface without buying type safety that runtime consumers actually need.

We go with **clear on emit**. Tests assert that `groups` / `translate_values` / `items` / `translate` / `choice_sets` are `None` on output siblings.

---

## Step A — salk_toolkit schema changes

**File:** `salk_toolkit/validation.py`

- Add `type: Literal["plain"] = "plain"` to `ColumnBlockMeta`.
- Make `TopKBlock(ColumnBlockMeta)` and `MaxDiffBlock(ColumnBlockMeta)`. Override `type` with their literal values. Add `segments()`.
- Remove `ColumnBlockMeta.create`.
- Replace the current `BlockSpec = Annotated[Dict[str, ColumnBlockMeta], BeforeValidator(_cb_lst_to_dict)]` with a discriminated union: `BlockSpec = Annotated[Dict[str, Annotated[Union[TopKBlock, MaxDiffBlock, ColumnBlockMeta], Field(discriminator="type")]], BeforeValidator(_cb_lst_to_dict)]`.
- Keep `__all__` exports of `TopKBlock`, `MaxDiffBlock`.

**File:** `salk_toolkit/serialization.py`

- `serialize_column_block_meta` must continue to work for subclasses (it already receives the concrete class via the `@model_serializer(mode="wrap")` on `ColumnBlockMeta`; subclass instances should pick the same serializer via inheritance).
- Ensure `serialize_pbase` skips `type` when it equals the class default (so `"plain"` doesn't leak into otherwise unchanged JSON, while `"topk"` / `"maxdiff"` are emitted).

---

## Step B — salk_toolkit io.py changes

**File:** `salk_toolkit/io.py`

- `_create_new_columns_and_metas` dispatches on `isinstance(block, TopKBlock | MaxDiffBlock)` instead of reading `block.create`.
- `_create_topk_metas_and_dfs` and `_create_maxdiff_metas_and_dfs` accept a `TopKBlock` / `MaxDiffBlock` directly (not a `ColumnBlockMeta` wrapping one).
- The regex branches resolve `from_columns` / `res_columns` / `best_columns` / `worst_columns` / `set_columns` to `List[str]` and emit the output block with those lists.
- Output siblings are `TopKBlock` / `MaxDiffBlock` instances (not plain `ColumnBlockMeta`). `groups` / `translate_values` / `items` / `translate` / `choice_sets` are set to `None` on the output (input directives only).
- The main structure loop (around io.py:1326) replaces `if group.create is not None:` with `if isinstance(group, (TopKBlock, MaxDiffBlock)):`.

After Step B, `data_meta.structure[name]` is a `TopKBlock` / `MaxDiffBlock` / `ColumnBlockMeta`, never a `ColumnBlockMeta` with a `.create`. Every `*_columns` field on an output block is `List[str]`, never regex.

---

## Step C — salk_internal_package consumption sketch (not implemented here)

**File (for later):** `salk_internal_package/salk_internal_package/sampling.py` (`list_aliases_res_cols`) or `obs_models/autodetect.py` (`AutodetectOM.prepare`).

Today, a bare block name in `res_cols` fans out to one `AutodetectOM` per column via `list_aliases_res_cols` (sampling.py:617-626), producing N independent Categoricals. For TopK/MaxDiff this is the wrong shape — we want one `OrdinalRanking` per block.

**Future hook** (left as a note; not to be implemented in this toolkit change):

```python
# When resolving a block name in res_cols:
block = data_meta.structure.get(name)
if isinstance(block, (TopKBlock, MaxDiffBlock)):
    return [OrdinalRanking(name=block.name, structure=block.segments())]
# else: existing fan-out behaviour
```

`OrdinalRanking.prepare()` already infers `omc` from observed data (`salk_internal_package/obs_models/ordinal_ranking.py:101-115`) — no vocabulary plumbing required. `salk_internal_package` imports `TopKBlock` / `MaxDiffBlock` from `salk_toolkit.validation` and never references `create`.

After this hook, the analyst writes:

```json
"res_cols": [
  {"factors": 3},
  "ownership_economics", "ownership_healthcare", "ownership_education",
  "ownership_foreign_policy", "ownership_nato"
]
```

and
```json
"res_cols": ["maxdiff_score"]
```

instead of the hand-written `structure` lists at `leedu_2026_full_package/model_desc.original.json:281-355` and `:383-525`. The `Factors(3)` prefix stays explicit because it's a modeling choice, not an annotation artefact.

---

## Step D — Tests

**Files:** `salk_toolkit/tests/test_io.py`, `salk_toolkit/tests/test_serialization.py`

- [ ] **Annotation-shape migration.** All existing tests that wrote `{"name": "X", "create": {"type": "topk", ...}}` migrate to the flat form `{"type": "topk", "name": "X", ...}`. `TestReadAnnotatedData::test_topk_*` and `test_maxdiff_*` are the primary sites.

- [ ] **Output-class assertions.** After `read_and_process_data`, assert `isinstance(data_meta.structure[<name>], TopKBlock)` / `MaxDiffBlock` on the relevant blocks. Assert `*_columns` fields are `List[str]` (regex resolved). Assert input-only directives (`groups`, `translate_values`, `items`, `translate`, `choice_sets`) are `None` on outputs.

- [ ] **`segments()` shape.** One test per block type, asserting the tuple shape matches leedu's hand-written structure: TopK `ownership_economics` → `[(["Q7r1_R1","Q7r1_R2"], None, False)]`; MaxDiff 8-question block → 16 segments in best→set + set→worst order.

- [ ] **Discriminated-union round-trip.** test_serialization.py asserts JSON `model_dump` → `model_validate` is stable for all three block types. Plain blocks emit no `type` field in JSON; topk/maxdiff blocks emit `"type"` once and validate back to the right subclass.

- [ ] **SIP integration test (future, not implemented here).** Once the SIP hook lands, a test in SIP asserts `"res_cols": ["<maxdiff_block_name>"]` builds an `OrdinalRanking.structure` matching what `tests/obs_models/test_variant.py:68` constructs by hand.

---

## Non-goals

- **No `choice_sets` consumption in the initial wiring.** `OrdinalRanking` today does not need it. If a future latent model wants per-row shown sets, it can read `block.choice_sets` directly on the input-side MaxDiff annotation.
- **No per-respondent available-set (`local_b`) handling.** None of the leedu blocks need it; `b=None` already expresses "chosen beats the full axis".
- **No subclass split between input / output blocks.** Input directives are cleared on emit; the output carries the same class, just with specialized fields populated and directives `None`.
- **No change to how `Factors(N)` is specified.** It remains an explicit prefix in `res_cols`; the number of shared factors is a modeling choice, not an annotation artefact.

---

## Order of operations

1. **Part 2 Step A + B** (salk_toolkit) — schema + io.py + serialization + tests. Single landing on `create-refactor` branch.
2. **Migrate annotations in consumer packages** — the flat `type: topk|maxdiff` shape replaces the old nested `create` form wherever it appears (leedu, other surveys).
3. **Step C** (salk_internal_package, out of scope for this branch) — lands on SIP's `2026_refacto` branch later.
4. **Step D SIP integration test** (out of scope for this branch) — locks down the full pipeline, unblocks short-form `leedu_2026_full_package/model_desc.json`.
