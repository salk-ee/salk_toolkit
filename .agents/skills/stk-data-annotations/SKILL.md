---
name: stk-data-annotations
description: Create, validate, align and audit stk (salk_toolkit) JSON data meta annotations for survey datasets. Use when working with _meta.json files, infer_meta, read_annotated_data, read_and_process_data, or when the user mentions survey annotations, metafiles, data alignment, or category mapping.
---

# STK Data Meta Annotations

## Overview

STK annotations are JSON files (`*_meta.json`) that describe how to transform raw survey data (`.sav`, `.csv`, `.parquet`) into a standardised, English-language, typed DataFrame. The authoritative schema lives in `salk_toolkit/validation.py` (`DataMeta`); processing logic lives in `salk_toolkit/io.py`.

Always read these two files before starting annotation work — the schema evolves.

**IMPORTANT**: When in doubt about the semantics of a survey question — what categories mean, whether something is ordered, topk or somethi— **always ask the user rather than assuming**. Wrong semantic assumptions (e.g. treating an unordered category as ordered, or merging categories that shouldn't be merged) produce silent errors that are extremely hard to detect later in the modeling pipeline.

**NEVER edit the raw data file (VERY IMPORTANT).** The raw data (`.sav`, `.csv`, `.xlsx`, `.parquet`) is the immutable source of truth — never modify it, overwrite it, or save a "cleaned" copy over it. All corrections, recodings, merges, synthetic columns, filters and fix-ups happen inside the annotation, in this order of preference:

1. **`translate` / `translate_after`** — for plain value → value remappings (e.g. merging `"Don't remember"` and `"Difficult to answer"` into `"Don't know"`, renaming categories, fixing typos).
2. **`transform`** (per column) — for expression-level fixes that need the cell / column in scope (casting, regex, `stk.cut_nice`, rule-based recoding).
3. **`preprocessing`** (top-level code block) — last resort, for changes that need multiple source columns at once, row filtering, or cross-column derivations before any column-level processing runs.

If you think you need to edit the raw file, you're wrong — use translate/transform/preprocessing instead.

## Use Cases

### 1. Creating a new annotation

**Definition of done:**
- [ ] Matches census in category names and granularity — ask for census file if not provided
- [ ] `id_col` set to a stable respondent-id column if the data has one (see Identifying the id column)
- [ ] All relevant columns annotated (demographics, opinions, scales, etc.)
- [ ] Ordered categories correctly ordered (likerts always go negative → positive pole, e.g. disagree → agree); `num_values` set for all ordered columns (centered on zero for likerts, 1–N otherwise, `null` for nonordered entries)
- [ ] `nonresponse` marks the true non-response categories ("Don't know", "No answer", "Refused") on **every** categorical block — ordered or not — so those semantics stay available to data-quality and other tooling; `nonordered` lists only the extra off-scale-but-substantive categories ("Other", "none", "Would not participate")
- [ ] All conventions followed (see below)
- [ ] Loads cleanly via `read_annotated_data(meta_file)` with no warnings
- [ ] Everything translated to English (exception: party acronyms)
- [ ] If a questionnaire / data description is available, add `label` entries with the exact question wording (per-item text on individual columns, shared lead-in text on `scale.label` for item-battery blocks)
- [ ] Region fields have a `topo_feature` attached — ask the user for a link to the map JSON if not provided
- [ ] Party brand colors collected and wired up wherever parties appear (see Colors section) — search the web if not provided by user

### 2. Aligning to an existing annotation

**Definition of done:**
- [ ] New annotation loads on its own (same criteria as above)
- [ ] Both files load together via `read_and_process_data` with no errors
- [ ] Shared columns have identical category names, order, and types
- [ ] `col_prefix` usage matches between files

### 3. Auditing / cleaning up an existing annotation

Same criteria as creating. Focus on correctness of category lists, ordered flags, translations, and consistency between party preference / thermometer / issue ownership blocks.

**Review & fix protocol** (follow this order strictly when editing existing annotations):

1. **Read everything first.** Read all annotation files, census meta, and any alignment targets fully before making a single edit.
2. **Produce one consolidated issue list.** After reading, output a single list of all issues found. No inline self-corrections or "wait, actually…" — if unsure, verify before listing.
3. **Batch all fixes.** Apply every fix in one pass. Do not stop partway through and wait for the user — complete all edits before moving on.
4. **Verify once.** Run `read_annotated_data` (and `read_and_process_data` if aligning) exactly once after all fixes are applied. If new issues surface, fix and re-verify — but the goal is one clean pass.
5. **Report.** Output: (a) changes made, (b) remaining warnings and whether they are actionable, (c) ambiguity report per the workflow below.

## Gathering Inputs

Before doing any annotation work, gather the required inputs. First, **search the directory of the provided data file** (and nearby folders) for these — only prompt the user for what you can't find:

1. **Data file** (`.sav`, `.csv`, `.parquet`, `.xlsx`) — the raw survey data. Required. Always provided or referenced.
2. **Data description** (Word/Excel/PDF document) — describes the survey questions, answer codes, and structure. This might not exist, especially if .sav file is provided as that often contains most of the required metadata. Nevertheless, always ask for this file if not found/provided. 
3. **Census file** — the country's census parquet/meta defining demographic categories and granularity. Look in the `census/` repo or ask the user. Usually present but might not be in very rare cases. 
4. **Previous wave / existing annotation** — if aligning, the `*_meta.json` from the prior wave or partner survey. Search nearby folders. Might not be present, ex for first wave in each country.
5. **DeepL API key + source language code** (e.g. `LT`, `ET`, `RO`) — needed for automatic translation during bootstrap.

When creating a new meta, ALWAYS ask the user about all 5 in sequence (have him confirm the file if you found one yourself).
For other use cases, ask as needed. 

## Typical Workflow

```python
import salk_toolkit as stk
from salk_toolkit.io import infer_meta, read_annotated_data, read_and_process_data
from salk_toolkit.validation import hard_validate, soft_validate, DataMeta
import json

# 1. Bootstrap from raw data with DeepL translation
meta = infer_meta("raw_data.sav", deepl_key="<key>", source_lang="LT")

# 2. Edit the *_meta.json to fix structure, ordering, conventions (AI does this)

# 3. Validate
hard_validate(json.load(open("data_meta.json")))

# 4. Test loading — iterate on step 2 until this passes cleanly
df = read_annotated_data("data_meta.json")

# 5. Write an ambiguity report: list every semantic judgement call made
#    (ordering decisions, category merges, what was marked nonordered, etc.)
#    so the user can verify assumptions in one pass

# 6. Hand off to user for review only after step 4 passes with no warnings

# 7. Multi-file alignment test (if applicable)
df = read_and_process_data({
    "files": [
        {"file": "wave1_meta.json", "code": "W1"},
        {"file": "wave2_meta.json", "code": "W2"}
    ]
})
```

## JSON Structure Quick Reference

```json
{
  "description": "...",
  "source": "...",
  "collection_start": "2026-01-15",
  "collection_end": "2026-02-01",
  "author": "...",
  "constants": { "party_colors": { "PartyA": "#ff0000" } },
  "files": [{ "file": "data.sav", "opts": {}, "code": "F0" }],
  "read_opts": {},
  "id_col": "respondent_id",
  "preprocessing": "df = df[df['age'] >= 18]",
  "postprocessing": null,
  "weight_col": null,
  "excluded": [],
  "structure": [
    {
      "name": "demographics",
      "scale": { "...shared column meta..." },
      "columns": [
        ["new_name", "source_col", { "...column meta..." }],
        ["new_name", { "...meta, source defaults to new_name..." }],
        ["new_name"],
        "bare_col_name"
      ]
    }
  ]
}
```

### Column entry formats (inside `columns` list)

| Format | Meaning |
|--------|---------|
| `"col"` or `["col"]` | Keep column as-is (name = source name in data) |
| `["new_name", "source"]` | Rename: read `source` from data, expose as `new_name` |
| `["new_name", { meta }]` | Same name in data, add/override metadata |
| `["new_name", "source", { meta }]` | Combines the two above |

Column-level `{ meta }` should only contain fields that **differ** from the block's `scale`. The scale is merged as defaults into every column, so don't repeat what's already set there.


### Key ColumnMeta fields

**Type declaration** — exactly one of these should apply:

| Field | Type | Purpose |
|-------|------|---------|
| `categories` | `list \| "infer"` | Categorical column. `"infer"` only valid with `translate` (order from translate dict). |
| `continuous` | `bool` | Numeric real-valued column |
| `datetime` | `bool` | Datetime column |

**Ordering** — only meaningful for ordered categorical columns:

| Field | Type | Purpose |
|-------|------|---------|
| `ordered` | `bool` | Whether categories are naturally ordered (age, income, likerts) |
| `likert` | `bool` | Symmetric ordered scale (requires `ordered: true`) |
| `neutral_middle` | `str` | Which category is the neutral middle for likert |
| `num_values` | `list[float|null]` | Numeric mapping for ordered categories (same length as `categories`). Use `null` for out-of-order entries (`nonresponse` + `nonordered`) like "Don't know". |

**Non-response & off-scale** — for **every** categorical column, ordered or not:

| Field | Type | Purpose |
|-------|------|---------|
| `nonresponse` | `list` | Categories that are true non-responses ("Don't know", "No answer", "Refused", "Hard to say"). **Recommended field going forward** — annotate on all categorical blocks so the nonresponse semantics feed data-quality checks and other tooling, not just modeling. The model treats non-responses as out-of-order automatically, so do not repeat them in `nonordered`. |
| `nonordered` | `list` | Categories that fall **outside the order but are still substantive** — *not* non-responses ("Other", "none", "Would not participate", "Did not vote"). List only these extras; the effective out-of-order set is `nonresponse` ∪ `nonordered`. For ordered scales the union is excluded from the order (gets `null` in `num_values`). |

**Transformations** — applied in order: `translate` → `transform` → `translate_after`:

| Field | Type | Purpose |
|-------|------|---------|
| `translate` | `dict` | Map source values → output values |
| `transform` | `str` | Python expression with `s`, `df`, `ndf`, `pd`, `np`, `stk`, constants in scope |
| `translate_after` | `dict` | Like translate, applied after transform |

**Display & modeling context:**

| Field | Type | Purpose |
|-------|------|---------|
| `label` | `str` | Column description for tooltips/headers |
| `colors` | `dict \| str` | Category value → color mapping (or constant name). See Colors section. |
| `question_colors` | `dict \| str` | Block-scale only: column name → color for unpivoted plots. See Colors section. |
| `groups` | `dict` | Named category groupings for filtering |
| `topo_feature` | `[url, type, col]` | Link to topojson for geographic columns |
| `modifiers` | `list[str]` | Columns that modify responses (private inputs for modeling) |

### Block-level fields

| Field | Purpose |
|-------|---------|
| `name` | Block identifier (must not collide with any column name in the annotation) |
| `scale` | Shared `ColumnMeta` defaults merged into every column in block |
| `columns` | List of column specs |
| `col_prefix` | On scale: prefix prepended to column names (disambiguates shared names) |
| `hidden` | Hide from explorer dashboards |
| `generated` | Column data produced by model, not in source file |
| `create` | TopK or MaxDiff block spec (see below) |
| `subgroup_transform` | Python code applied to all columns in block as `gdf` |

### Constants

Any value in the structure can be a string matching a key in `constants`. It gets replaced at parse time. Use for colors, topic lists, and translation dicts shared across blocks.

**Only define a constant if it is referenced two or more times.** Single-use constants add indirection and hurt readability — inline them at the use site instead. When auditing an annotation, remove any constant used zero or one times.

### Colors

Two fields, orthogonal dimensions:

| Field | Where | Maps |
|-------|-------|------|
| `colors` | Column meta (or `scale` as default) | category value → hex |
| `question_colors` | Block `scale` only | column name → hex; becomes `colors` on the synthetic `question` column after unpivot (see `pp.py::_question_meta_clone`, ~line 1110) |

Both accept an inline dict or a string referencing a constant. For `question_colors`, the block's column **names must match the keys** in the referenced dict. If a block's `scale` is a string reference to a shared constant (e.g. `"scale": "trust_scale"`), inline the scale to add `question_colors` — string refs are whole-value replacements.

**Party colors — always collect them.** Whenever an annotation has party data (`party_preference`, per-party `thermometer` / `ownership`), define a `party_colors` constant and reference it via `colors` on party-valued columns and via `scale.question_colors` on blocks whose columns are parties. If the user didn't supply colors, **search the web**:

> Wikipedia's "Opinion polling for the [YEAR] [COUNTRY] parliamentary election" pages are the canonical source for exact hex codes. Open the page source on the polling table and look for `{{party color|PartyName}}` templates — these pull from a shared CSS database of hex codes used by news organizations. That gives you per-country, per-election, match-the-press colors in one place.

Fall back to distinct placeholder hues (documented in a `comment`) only if a reliable hex can't be found. Use neutral greys for ballot meta-options (`other`, `spoil_ballot`, `Against_Everyone`, `none`, `Don't know`, `No answer`).

See `examples/example_web_meta.json` for a worked pattern — `party_colors` constant, `colors: "party_colors"` on `party_preference`, and `scale.question_colors: "party_colors"` on the thermometer block.

### Comments

Every block in the annotation (the top-level `DataMeta`, any entry in `structure`, any `scale`, any per-column meta dict, `create` blocks, etc.) accepts an optional `"comment"` field. JSON has no native comment syntax, so this field is the canonical place to leave notes.

- Value is either a single string or a list of strings (one per line) — both render fine in the JSON.
- The field is ignored by all processing code: it carries no semantic meaning and has zero runtime effect.
- It is preserved on load/save round-trips through the pydantic models.

```json
{
  "name": "attitudes",
  "comment": "5-point Likert collapsed from original 7-point in CATI wave — see below",
  "scale": {
    "categories": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
    "ordered": true,
    "likert": true
  },
  "columns": [
    ["future", { "comment": ["'optimism about the future' in questionnaire", "kept singular name to match previous waves"] }]
  ]
}
```

**Use `comment` to document any decision that deviates from best practice or is non-obvious.** This includes (but is not limited to):

- Non-standard mappings via translate, especially if they lose information
- Unusual `transform` logic, especially when a simpler form would look correct but be wrong
- Placeholder values, known-broken columns, or anything the next editor would otherwise "fix" incorrectly

If you find yourself wanting to explain a choice to the user in chat, write that explanation into `comment` as well — future readers of the JSON will thank you.

## TopK Blocks

For "select top K" questions (e.g. "which 3 issues matter most?"):

```json
{
  "name": "issue_importance_top3",
  "create": {
    "type": "topk",
    "from_columns": "Q6r(\\d+)",
    "res_columns": "Q6p_R\\1",
    "agg_index": 1,
    "na_vals": ["NO TO: ...", "..."],
    "translate_after": { "1": "Cost of living", "2": "Healthcare" }
  },
  "scale": { "categories": "infer" },
  "columns": []
}
```

- `from_columns`: regex matching source columns (or explicit list)
- `res_columns`: output column template (or explicit list matching from_columns)
- `agg_index`: which regex group indexes the items (1-indexed; -1 = last)
- `na_vals`: values meaning "not selected" — replaced with NA
- `translate_after`: map item indices to readable names (applied first)
- `from_prefix`: if `from_columns` is a list, strip this prefix for translation

The `columns` list in a topk block is usually **empty** — output columns are auto-generated. However, some topk blocks (e.g. issue ownership) list the raw source columns alongside the `create` block when those columns are also needed for other purposes.

### TopK translate pipeline

After the one-hot columns are reshaped (cell value becomes the column's regex-group label), translations are applied in order:

1. **`create.translate_after`** — maps raw regex-group labels (typically numeric indices like `"1"`, `"2"`) to readable names.
2. **`scale.translate`** — maps those names (or the original text if `translate_after` was not used) to final English output names. When `scale.translate` is present, its values become the output `categories` list.

In practice you use **one or the other**, not both:
- Numeric one-hot columns → use `translate_after` to go from index → English name.
- Text-valued one-hot columns (e.g. party names in the local language) → use `scale.translate` to go from local name → English short code.

## MaxDiff Blocks

For best-worst scaling / maxdiff experiments:

```json
{
  "name": "maxdiff",
  "create": {
    "type": "maxdiff",
    "best_columns": "Q6_(\\d+?)best",
    "worst_columns": "Q6_(\\d+?)worst",
    "set_columns": "Q6_\\1set",
    "setindex_column": ["Q6_Version", { "continuous": true }],
    "topics": null,
    "sets": null
  },
  "scale": {
    "categories": "infer",
    "translate": { "Local topic 1": "English topic 1", "...": "..." }
  },
  "columns": []
}
```

- `best_columns` / `worst_columns`: regex or list matching best/worst choice columns
- `set_columns`: regex template or list for the set-membership columns
- `setindex_column`: column containing set version index (with optional meta). Mutually exclusive with explicit set_columns data in the file.
- `topics`: list of all topic strings (typically in `constants`)
- `sets`: list of lists of 1-indexed topic indices per version (typically in `constants`)
- Scale `translate` maps local-language topics to English

### MaxDiff translate pipeline

All translation happens through **`scale.translate`** (there is no `translate_after` for maxdiff). The flow:

1. **`topics`** defines the full topic list (usually via `constants`) in the source language.
2. **`scale.translate`** maps each source-language topic to its English name, producing `effective_topics`.
3. `effective_topics` is used everywhere: best/worst column values are translated and cast to categorical with this list; set columns resolve topic indices through this list; the output meta carries `effective_topics` as its categories.

So `scale.translate` is where all the naming happens for maxdiff — it controls both the cell values and the category list.

When using `setindex_column`, `topics` and `sets` must be defined (usually via constants). The columns list should be **empty**.

## Conventions (MUST follow)

1. **English**: All category names, labels, and column names in English.
   - Exception: party names/acronyms kept as originals (e.g. "TS-LKD", "LSDP")
   - Exception: geographic names (counties, municipalities) may stay in the local language — match whatever the census uses
2. **Column names**: short, `snake_case`, single identifier where possible. Put the full human-readable name in `label` when the column name is a shortening/change.
   - **Default**: lowercase (e.g. `age`, `gender`, `pol_interest`).
   - **Proper nouns** (people, parties, organisations) stay capitalized (e.g. `Putin`, `Macron`, `Civil_Contract`, `Fidesz`). For people prefer last name only. If **any** name in a block needs a first-name prefix to disambiguate, use full `First_Last` names for **every** person in that block.
   - **Acronyms** stay fully uppercase (e.g. `ARF`, `ANC`, `LSDP`, `TS-LKD`).
3. **Standard block/column naming**: use these names whenever the concept applies, so blocks line up across surveys:
   - `party_preference` — who the respondent would vote for (single column or block).
   - `thermometer` — per-party rating / likability / trust scale (one likert-style column per party).
   - `importance` — issue-importance ranking, usually pick-top-K or maxdiff.
   - `ownership` — which party is trusted most to handle each issue.
4. **`categories: "infer"`**: Only use together with `translate`. Order is derived from `translate` dict key order.
5. **`translate`**: Only include if actually performing translation or value mapping. Don't add identity translations unless needed for order disambiguation with `categories: "infer"`.
6. **Ordered categories**: Naturally ordered data (age, income, education, likerts) must be `ordered: true`. Any **bipolar** ordered scale — one with opposing poles (agree/disagree, trust/distrust, positive/negative, better/worse) — must be marked `likert: true` with `num_values` centred on zero, regardless of whether a neutral middle exists. Set `neutral_middle` when a middle category does exist.

   **Dichotomous choices are likerts too.** Any 2-way choice — yes/no, for/against, approve/disapprove, support/oppose, stay/leave, EU/EAEU, etc. — must be marked `ordered: true, likert: true` with `num_values: [-1, 1]` (plus nulls for DK/NA), not left as unordered categorical. This applies to both opinion bipolars (agree vs disagree) and factual/choice binaries (yes vs no, A vs B).

   **Pick the positive pole by this priority** (documented with a `comment` when non-obvious):
   1. Explicit valence: trust, agree, approve, support, positive, better, more, yes → positive; distrust, disagree, disapprove, oppose, negative, worse, less, no → negative.
   2. Affirmative / pro-action: yes, for, support, change-to-new > no, against, oppose, keep-status-quo.
   3. For A vs B choices without explicit valence, pick the pole aligned with the survey's analytical reference direction (e.g. Western/EU orientation as positive in Eastern-European polling) and document with `comment`.

   **Always order likert categories from the negative pole to the positive pole** (disagree → agree, distrust → trust, no → yes, against → for, leave → stay, EAEU → EU); `num_values` increase monotonically from negative to positive. Flip with `translate` if the source data codes the other way.
7. **Non-response categories** (`nonresponse`): Mark true non-responses ("Don't know", "No answer", "Refused", "Hard to say") in `nonresponse` on **every** categorical block — ordered **and** nominal (unordered) — not only on ordered scales. This is the recommended field going forward; the signal is relied on by data-quality checks and other tooling. The model treats non-responses as out-of-order automatically, so do **not** repeat them in `nonordered`. Reserve `nonordered` for categories that are off-scale but still *substantive* — **not** non-responses (e.g. "Other", "none", "Would not participate", "Did not vote"). On an ordered scale the union (`nonresponse` ∪ `nonordered`) is lifted out of the order and gets `null` in `num_values`; on a nominal categorical neither changes modeling but both record structure worth keeping.
8. **Party consistency**: Party names must be identical across `party_preference`, `thermometer`, and `ownership` blocks.
9. **Discrete scales**: Use categorical (not continuous) for scales with <20 values, even if numeric.
10. **`col_prefix`**: Use to disambiguate columns that share names across blocks (e.g. `attitude_`, `issue_`, `therm_`).
11. **Auto-inferred blocks from topk/maxdiff**: Delete any blocks that were auto-generated by `infer_meta` for columns that belong to topk/maxdiff `create` blocks — those get regenerated.
12. **Document non-obvious decisions with `comment`**: Any choice that deviates from best practice or is non-obvious (unusual merges, ambiguous ordering calls, deliberate category mismatches, tricky transforms) must be noted in a `comment` field on the block, scale, or column where it applies. See the Comments subsection above.

## Common Pitfalls

- **Category order matters**: `categories: ["Never", "Sometimes", "Usually", "Always"]` defines the modeling/display order. Check it matches the natural ordering.
- **Many-to-one translate**: Multiple source values can map to the same output (e.g. merging districts). This is fine but be aware `categories: "infer"` deduplicates while preserving first-seen order.
- **Missing na_vals in topk**: If `na_vals` don't match the actual "not selected" values in the data, topk processing will fail or produce wrong results.
- **Scale vs column precedence**: Column-level meta overrides scale. If a column needs different categories than the block, specify them on the column.
- **`education` ordering**: `["Primary", "Secondary", "Higher"]` not alphabetical. Always verify ordered categories make substantive sense.
- **num_values alignment**: Must have same length as categories list and correspond 1:1.
- **Variant files**: When CATI and WEB surveys share questions but with different scales (5-point vs 7-point), use a `_p` suffix for the phone variant columns and create separate blocks with appropriate scale transforms.

## Inspecting Raw Data

Before annotating, examine the source file:

```python
import pyreadstat
df, meta = pyreadstat.read_sav("data.sav", apply_value_formats=True)
# meta.column_names, meta.column_labels — useful for labels
# df['Q1'].value_counts() — check actual category values
# df.columns.tolist() — all column names
```

For SAV files, `meta.column_labels` often contains the question text in the original language — feed these to a translation function for initial labels.

## Identifying the id column

Every row STK returns is stamped with a stable id and the returned DataFrame is indexed by it. By default that id is the within-file row position (`F0::0`, `F0::1`, …), which is only stable while the source file's bytes don't change. If the survey has a genuine respondent identifier, set **`id_col`** so ids become `{file_code}::{respondent_id}` — stable across re-exports and row reorders, and the correct anchor for `excluded`.

- **Scan for a candidate** while inspecting the raw data: look for columns named like `id`, `respondent_id`, `resp_id`, `caseid`, `ResponseId`, `uuid`, `interview_id`, or a running record number. Verify it: `df['id'].is_unique and df['id'].notna().all()`.
- **Set it** at the `DataMeta` level (`"id_col": "respondent_id"`), or per-file on a `FileDesc` (`{"file": "...", "code": "W1", "id_col": "resp_id"}`) when different waves name it differently. The per-file value overrides the meta-level default.
- **Confirm with the user** before committing to a column — picking the wrong one silently misattributes exclusions. If no unique non-null column exists, leave `id_col` unset (positional ids are fine); note the absence in a `comment`.
- STK hard-errors if the declared `id_col` is missing, has nulls, or is non-unique within a file — so a failing load tells you the candidate was wrong.

`excluded` is keyed on these ids: each entry is `["{row_id}", "reason"]` (e.g. `["F0::102", "careless battery"]` with `id_col`, or `["F0::5", "..."]` positionally). The legacy positional-integer form (`[5, "..."]`) is rejected. Exclusions compose across nested metas — an outer meta references an inner row by its full prefixed id (`M::F0::102`) and is unaffected by what the inner meta itself filters.

## Validation Commands

```python
# Quick validation
hard_validate(meta_dict)  # Raises on any issue

# Load test (most thorough — runs full processing pipeline)
df = read_annotated_data("my_meta.json")

# Multi-file alignment test
df = read_and_process_data({
    "files": [{"file": "meta1.json"}, {"file": "meta2.json"}]
})
```

Warnings during `read_annotated_data` are important — they flag missing columns, dropped categories, and category mismatches. Resolve all of them.

## Aligning With Census

Census files define the ground-truth category names and granularity for demographic columns. When annotating:

1. Load the census parquet/meta to see its column names and categories
2. Ensure demographic columns (age_group, gender, education, county, municipality, etc.) use **exactly** the same category strings
3. Match any computed columns like `county+` that combine geography levels
4. **`age_group` is typically derived from a continuous `age` column** using `stk.cut_nice` with breakpoints matching the census granularity. The survey data usually has raw age — you create the correct grouped column via `transform`:

```json
["age_group", "age", {
  "categories": "infer",
  "transform": "stk.cut_nice(s, [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])",
  "ordered": true,
  "label": "age group"
}]
```

The breakpoint list must match what the census uses. Check the census `age_group` categories to determine the right bins.

## Aligning Two Meta Files

When two surveys (e.g. CATI + WEB, or two waves) need to load together via `read_and_process_data`, their annotations must be compatible:

1. **Shared columns must have identical names, categories, and category order.** This includes demographics (`gender`, `age_group`, `education`, `county`, etc.) and any columns used as model inputs.
2. **`col_prefix` must match** for blocks that should merge (e.g. both files use `attitude_` for attitudes).
3. **Different scales for the same question** are handled with separate blocks and a `_p` suffix on column names. For example, WEB uses a 7-point scale (`attitudes` block with `attitude_` prefix), CATI uses 5-point (`attitudes_p` block with the same `attitude_` prefix but columns like `pol_interest_p`). The shared prefix means they land in the same namespace; the `_p` suffix distinguishes the reduced-scale variant.
4. **The `method` column** should be added to distinguish data sources (e.g. `"categories": ["web", "cati"]`). Include it in both files.
5. **Translate dicts for party names** must produce identical output strings across files — even if the source-language strings differ slightly between surveys.
6. **Test alignment** by loading both together and checking for warnings:

```python
df = read_and_process_data({
    "files": [{"file": "web_meta.json"}, {"file": "cati_meta.json"}]
})
```

Any category mismatch or duplicate column name will surface as a warning or error. Fix these iteratively until the load is clean.

7. **The last file is the basis for the combined meta.** `read_and_process_data` uses the last file's annotation as the combined schema. If blocks exist in file A but not in file B (the last file), they won't appear in the output — even though the data is present. To fix this, add the missing blocks to the last file with `"generated": true` on each such block. This suppresses "no matching columns in data" warnings for that file while letting the block's schema carry through to the combined result.

## Worked Example

A complete minimal example lives in `.cursor/skills/stk-data-annotations/examples/`:

| File | Description |
|------|-------------|
| `example_web_meta.json` | WEB survey annotation — 7-point attitudes, topk, maxdiff |
| `example_cati_meta.json` | CATI survey annotation — 5-point attitudes (same questions) |
| `example_web_data.csv` | 60-row synthetic raw data for WEB |
| `example_cati_data.csv` | 40-row synthetic raw data for CATI |
| `example_census.csv` | 30-row census cross-tab (gender × education × age_group) |

**Key patterns demonstrated:**

- **Demographics aligned with census**: `gender`, `age_group` (via `stk.cut_nice` transform), `education` — category names and age bins match `example_census.csv` exactly.
- **`method` column**: Synthetic column created via `transform` — WEB file produces `'web'`, CATI file produces `'cati'`; both share `"categories": ["web", "cati"]`.
- **`categories: "infer"` + `translate`**: `party_preference` — category order comes from translate dict key order. Translate dicts are identical across both files for alignment.
- **Likert `_p` variant pattern**: WEB has 7-point `attitudes` block (columns `pol_interest`, `future`); CATI has 5-point `attitudes` block (columns `pol_interest_p`, `future_p`). Both use `col_prefix: "attitude_"` so columns land in the same namespace.
- **`generated: true` for alignment**: WEB includes `attitudes_p` block with `generated: true` — this block has no matching data in the WEB file, but its schema lets the 5-point CATI columns carry through when loading both files together.
- **TopK with `translate_after`**: `issue_importance` block uses regex `from_columns`, `na_vals` to filter unselected items, and `translate_after` to map numeric regex groups to English names.
- **MaxDiff with `scale.translate`**: `maxdiff` block (WEB only) uses `setindex_column` + `topics`/`sets` constants (2 versions × 3 sets of 3 topics). `scale.translate` maps Lithuanian topic names to English — this single dict controls both cell values and the output category list.
- **Colors — `colors` vs `question_colors`**: `party_colors` constant is referenced by `colors` on `party_preference` (values are parties) and by `scale.question_colors` on the `thermometer` block (columns are parties, so each party gets its brand color when the block is unpivoted into a `question` dimension). Thermometer column names must match the `party_colors` keys.

## For more details

- Schema: `salk_toolkit/validation.py` — `DataMeta`, `ColumnMeta`, `ColumnBlockMeta`, `TopKBlock`, `MaxDiffBlock`
- Processing: `salk_toolkit/io.py` — `_process_annotated_data`, `infer_meta`, `_fix_meta_categories`
- Cursor rule: `salk_toolkit/.cursor/rules/data_annotations.mdc`
- Examples: look at recent `*_meta.json` files in the sandbox repo for real-world patterns
