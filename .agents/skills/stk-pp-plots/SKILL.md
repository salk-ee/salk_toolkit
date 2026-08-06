---
name: stk-pp-plots
description: Author stk `pp_desc` plot descriptors for `salk_toolkit.pp.e2e_plot` — picking a plot, wiring facets / filters / transforms / weights / expression stats, choosing between a chart, `return_data=True` and a `create_plot_payload`, and debugging `matching_plots` rejections.
---

# stk `pp_desc` authoring

## Overview

`e2e_plot(pp_desc, full_df=..., data_meta=...)` is the single entry point — given a survey dataframe plus its annotations, it picks a registered plot function, builds the right aggregation, and returns either an Altair chart (`to_dict()` → Vega-Lite) or the aggregated pandas frame (`return_data=True`).

All business logic — tooltips, colors, category orders, labels, translations — is read from `data_meta` via `cmeta`. **Never hand-configure any of that at the descriptor level.** If a label or color is wrong, fix the annotation (`stk-data-annotations` skill), not the descriptor.

Pipeline summary (authoritative flow: `salk_toolkit/pp/`, `wrangle.py` → `plotting.py`):

1. `matching_plots` — checks the selected plot is feasible given the data + annotations.
2. `pp_transform_data` / `wrangle_data` — lazy polars filter / unpivot / aggregate → pandas frame + `pparams`.
3. `create_plot` — adds tooltips / colors / translations, dispatches to the registered plot function.

Descriptor schema: `salk_toolkit.validation.PlotDescriptor`. Read it when anything below is ambiguous — it is the source of truth.

## pp expresses ~all plot data prep

The 1% it doesn't is genuine row-level modelling whose output is not an aggregate — a clustering, a per-respondent imputation. Everything else is a descriptor, including things that do not look like it: argmax over a battery, exclusive reach, a scope's `n`, an election simulation.

When pp appears unable to do what you want, the cause is almost always that **you named the wrong thing** — a plot carrying its own defaults, a column where a block was wanted, a draws-resolved plot for a plain proportion. The symptom then looks like a library limitation, and writing a workaround plus a plausible explanation is very tempting. The check is cheap: **read the `@stk_plot(...)` registration** — `data_format`, `draws`, `agg_fn`, `transform_fn`, `requires` — and `grep -rn "@stk_plot" --include=*.py` across the package, not just `plots.py`. If your explanation of pp's behaviour is not something you read in the code, it is a guess.

- `reference/excuses.md` — the misdiagnoses that have actually happened, symptom → cause → fix.
- `reference/perf.md` — cost model, `draws` cost, expression-stat limits, measured numbers.

## Minimal descriptor

```python
pp_desc = {"plot": "columns", "res_col": "vote_intent"}
e2e_plot(pp_desc, full_df=af.df, data_meta=af.meta).to_dict()
```

- `plot` — registered plot type (see catalog below).
- `res_col` — response column, or a *block* name when the block unpivots into a `question` column (maxdiff, thermometer, issue importance).
- Everything else is optional; `impute_factor_cols` fills `factor_cols` sensibly based on the plot's `PlotMeta`.

## Plot registry catalog

All entries come from `salk_toolkit/plots.py` via `@stk_plot(...)`. The `data_format` column tells you whether the plot consumes aggregated long-form rows (one value per facet-cell) or raw rows. `draws=True` means the plot leverages posterior draws when the data carries them; most plots are fine either way.

| plot name | data_format | n_facets | typical `res_col` | notes |
|---|---|---|---|---|
| `columns` | longform | 1–2 | categorical (likert, party) | simple bar chart; top-level comparison of category shares |
| `stacked_columns` | longform | 2 | categorical | sums proportions per facet; `plot_args={"normalized": True}` for 0–1 stack |
| `diff_columns` | longform | 2 | 2-category (likert dichotomy) | paired bars with delta; `plot_args={"sort_descending": True}` |
| `likert_bars` | longform | 1–2 | `likert: true` column | divergent positive/negative bars about the neutral middle |
| `likert_rad_pol` | longform | varies | `likert: true` block | radial polarisation view of likert batteries |
| `boxplots` | longform | 1–2 | continuous (or `convert_res="continuous"`) | Tukey whiskers; uses draws when present |
| `massplot` | longform | 1–2 | categorical | bubble/mass chart — useful for 3+ dimensional breakdowns |
| `marimekko` | longform | 2 | categorical | rows sum to 1 within each first-facet level; shows composition × size |
| `matrix` | longform | 2 | categorical | heatmap of category frequencies |
| `corr_matrix` | raw | — | a *block* with multiple numeric columns | pairwise correlations; input is raw (not aggregated) |
| `density` | longform | up to 3 factor cols | continuous | KDE / smoothed density |
| `violin` | raw-ish (as_is) | 1–2 | continuous | violin plot; supports `plot_args={"bw": 0.3}` |
| `lines` / `line` | longform | 1–2 | continuous | multi-series line (`lines`) or single line (`line`) over an ordered factor |
| `lines_hdi` / `line_hdi` | longform | 1–2 | continuous | as above with posterior HDI ribbons — requires draws |
| `area_smooth` | longform | 1–2 | continuous | stacked smooth areas |
| `maxdiff` | longform | 1–2 | maxdiff block | columns of best-minus-worst scores; ships `transform_fn="ordered-topbot1"` + `agg_fn="posneg_mean"` by default |
| `barbell` | longform | 2 | categorical or likert | two-point barbell — good for before/after, variant A/B |
| `facet_dist` | raw | — | mixture of columns | facet grid of distributions |
| `ordered_population` | raw | — | ordered categorical | population pyramid for ordered categories |
| `geoplot` / `geobest` | longform | 1–2 | categorical or continuous | chloropleth and "best category per region"; requires `topo_feature` on a geographic `factor_col` |

Election plots (`mandate_plot`, `party_mandates`, `coalition_applet`) are registered in `salk_toolkit/election_models.py`. To see the live list and the full `PlotMeta` for each:

```python
from salk_toolkit.pp import registry_meta, _ensure_plot_registry_loaded
_ensure_plot_registry_loaded()
{k: v.model_dump() for k, v in registry_meta.items()}
```

## Descriptor fields — authoring patterns

### `factor_cols` (facets)

One or more columns to break the plot down by. Order matters — first factor is typically the primary axis (x in `columns`, color in `lines`, etc.). Leave empty to let `impute_factor_cols` backfill a sensible default for the chosen plot.

```python
{"plot": "columns", "res_col": "vote_intent", "factor_cols": ["age_group"]}
{"plot": "likert_bars", "res_col": "pol_interest", "factor_cols": ["gender", "education"]}
```

A **numeric factor col is binned, not grouped by value** — by default into quantiles, with integer ties broken by a jitter drawn from a hash of the row index (deterministic across engines, chunkings and thread counts, and uncorrelated with how the file happens to be sorted). For value-exact groups set `bin_breaks` / `bin_labels` through `col_meta`, as under `convert_res` below.

### `filter`

`{column: selection}`. Applied **before** aggregation on the lazy frame. Three value shapes:

- **Scalar** — single category: `{"gender": "Female"}`
- **List** — category subset: `{"education": ["Higher", "Secondary"]}`
- **Range** — inclusive `[None, min, max]` (either bound can be `None`): `{"age": [None, 25, 65]}`, `{"age": [None, 18, None]}`

Group aliases declared on the column's `groups` meta are resolved too: `{"party_preference": ["left_bloc"]}` expands per `cmeta[party_preference].groups`.

**Filtering a block is load-bearing, not cosmetic.** It narrows the melt, and `ordered-top*` / rank transforms compare a respondent's columns *to each other* — so it decides both which columns compete and, through them, which respondents count. A block carrying `Don't know` / `Other` / a count column sits in everyone's top-k cutoff and quietly lowers every share:

```python
# argmax over the 7 tracked parties only — not over every column in the block
{"plot": "columns", "res_col": "maksud", "cont_transform": "ordered-top1",
 "agg_fn": "mean", "filter": {"maksud": TRACKED_PARTIES}}
```

The same filter expresses a share **among** a subset: `columns` renormalizes `percent` within each cell *after* filtering, so restricting a categorical `res_col` turns "share of all respondents" into "share of named-party voters".

For expressions polars can evaluate but `filter` can't encode, use `pl_filter` (a polars expression string evaluated on the LazyFrame) — an escape hatch, not the default.

### `weights`

pp weights by the column `data_meta.weight_col` names. If the annotation declares one and the parquet does not have it — a vintage renaming `N_voters` → `N`, say — that is an **error**, not a silent fallback to unweighted numbers. An annotation declaring no weight column is unweighted, unless the data carries the conventional `row_weights` column.

- `weights: True` — the default: the declared `weight_col`, required if declared.
- `weights: "<column>"` — a specific column (also required-or-error).
- `weights: False` — deliberately unweighted: every row counts 1, and `total_size` / `filtered_size` come back as **plain row counts**. This is how respondent `n`s come out of the same descriptor that produces the shares.
- `weights: "<polars expression>"` — a string **containing `pl.`** is evaluated as an expression building the per-row weight (the `pl_filter` contract): `"pl.col('N') * pl.col('turnout_prob')"`, or a window expression (`.sum().over(...)`) renormalizing within a partition. A column name that is not a Python identifier (`w.2024`) still resolves as a column; a missing one still errors. Null results are not filled — that policy makes sense for a declared weight column and none at all for a computed design weight.

Any override (everything except `True`) recomputes `total_size` from the actual weights: the annotation's declared population total describes the *declared* weighting, so it no longer applies.

### `convert_res`

**`"continuous"`** turns an ordered categorical response numeric for plots that expect it (`boxplots`, `density`, `lines`, `violin`). `num_values` comes from the annotation; override per-descriptor when the analytic scale must differ. On an already-continuous `res_col` it is a no-op cast, so it is safe to leave on; it errors on a column that is neither (a `datetime`, one with no numbers to parse). The annotated `val_range` is preserved; an unannotated one stays unset.

```python
{"plot": "boxplots", "res_col": "pol_interest", "convert_res": "continuous",
 "factor_cols": ["age_group"]}
```

**`"categorical"`** goes the other way: bucket a numeric response and take shares, using the same discretization numeric facets get. Edges come from `bin_breaks` / `bin_labels` (an int means that many quantiles), set per plot through `col_meta` — `res_meta` declares a *block*, `col_meta` overrides a single column.

```python
# exact 0-10 histogram of an integer column
{"plot": "columns", "res_col": "vote_prob", "convert_res": "categorical",
 "col_meta": {"vote_prob": {"bin_breaks": [b + 0.5 for b in range(10)],
                            "bin_labels": [str(b) for b in range(11)]}}}
```

Binning is per column, so it needs a **single-column** `res_col`: on a block each column would get its own quantiles and labels and only the last one's would survive the shared axis.

**With a `cont_transform`, `"categorical"` describes the result**, not the input: convert to continuous as needed, transform, *then* categorize. Without bin specs that is **literal bins** — one ordered category per distinct transformed value, ordered numerically (rank 10 after 9, not after 1). Labels and ordering are resolved after aggregation on the small aggregated frame, so no second scan; past 50 distinct values it errors, since genuinely continuous output wants bins. This is the one categorization that **does** work on a block, and the point of it — the transform puts every column on one shared domain, so the category list is derived globally rather than per column:

```python
# distribution over each topic's rank: which rank does each topic get, and how often
{"plot": "columns", "res_col": "topic_importance", "factor_cols": ["question"],
 "convert_res": "categorical", "cont_transform": "ordered-avgrank"}
```

A plot can register `convert_res` (with `transform_fn` / `agg_fn`) so a bare descriptor gets this shape by default; the descriptor still wins. Sorting a facet on such a distribution orders by the share-weighted mean of the category scale — for ranks, the mean rank — not the near-constant mean of the shares. Shares under `agg_fn="mean"`, weighted counts under `"sum"`. Only expressible on a longform plot: a raw-format or `stats` descriptor raises instead of silently returning untransformed values.

### `cont_transform`

Applied **after** `convert_res` when a rescaling / summary is desired; it only runs on a continuous response (see `reference/excuses.md` for the silent-no-op case). Names are validated against the live registries, so a transform registered from outside stk is accepted.

- Scale-level: `center`, `zscore`, `01range`, `proportion`
- Softmax family: `softmax`, `softmax-ratio`, `softmax-avgrank`
- Ordered helpers: `ordered-avgrank` (1 = lowest of the battery) and `ordered-avgrank-desc` (1 = highest), `ordered-warf`, `ordered-top1`, `ordered-bot1`, `ordered-topbot1`, `ordered-top2`, `ordered-top3`
- Threshold family: `ge:<x>` — replaces the column with a 0/1 indicator, so `mean` gives the share past the cutoff and `sum` the weighted count (value format follows: `.1%` / `.0f`). The cutoff is compared *after* `convert_res`, so on an ordered categorical it is a `num_values` number, not a category index. `ge:-inf` is the "answered at all" indicator, so its weighted sum is the response count. `gt:`, `le:` and non-numeric cutoffs are rejected at validation.
- Top-k family: `ordered-top-ties:<k>` selects everything reaching the row's k-th best *value*, so ties can select more than k (`ordered-top1` is its k=1 case); `ordered-top2` / `ordered-top3` select exactly k, ties broken by column order. All rank among the row's *answered* columns, so a partly-answered row still has a top-k.

```python
# share who rate each party at least 1 on a -5..5 thermometer, all parties in one scan
{"plot": "columns", "res_col": "thermometer", "agg_fn": "mean",
 "convert_res": "continuous", "cont_transform": "ge:1"}
```

Affine pre-steps (a temperature divisor, an additive log-prior) fold into the descriptor. Most plots that need a transform declare a default via `transform_fn` on the registration; override only when that isn't what you want.

### `agg_fn`

One of `mean | sum | posneg_mean | median | min | max`. Override the plot's registered default when the analytic question needs a different summary — e.g. switching `columns` from count-proportions to the mean of a continuous conversion:

```python
{"plot": "columns", "res_col": "approval", "convert_res": "continuous",
 "agg_fn": "mean", "factor_cols": ["party_preference"]}
```

### `stats` — several statistics in one pass

Mean + threshold shares + n for every column of a battery is *one* question, not five. Never issue one descriptor per statistic — each re-melts the block. In order of preference:

1. **Same rows, discrete scale (likert, ordered categorical): take the level distribution.** One *categorical* descriptor (no `convert_res`) gives `(question, level) → count`; mean, shares, exact neutral and n are arithmetic over ~a hundred cells, and the cached distribution serves the block's other consumers for free.

2. **Different row sets per cell, or continuous values: expression stats.** Each statistic is a row-level polars expression aggregated in one `group_by`:

```python
{"plot": "columns", "res_col": "media_television", "weights": False, "stats": [
    {"name": "tv",         "expr": "(pl.col('media_television') > 0)"},
    {"name": "tv_not_web", "expr": "((pl.col('media_television') > 0) & ~(pl.col('media_web') > 0))"},
    {"name": "tv_n",       "expr": "(pl.col('media_television') > 0)", "agg_fn": "sum"},
]}  # data-only: pass return_data=True or return_input=True
```

This covers the filter-varying class — overlap matrices ("uses i but not j"), audience masks, marginal reach — where a `pl_filter` per cell costs a descriptor per cell. `agg_fn` folds the declared (or overridden) weighting in per stat; referenced columns are harvested from the expressions; output is a column per stat.

Expressions see the frame as stored, the same vocabulary `pl_filter` sees. Unlike `filter` they get no meta resolution — no `groups` aliases, no category expansion — so write annotation-declared values literally.

`stats` refuses the shapes it cannot serve: a non-longform plot, a block `res_col` (the block aggregation drops the columns the expressions name), a stat name colliding with a facet dimension, a descriptor-level `agg_fn` / `cont_transform` (each stat carries its own), and a render with neither `return_data` nor `return_input`.

They are not always the right answer — there is no common-subexpression elimination, and they lose to `cont_transform` on within-respondent transforms over a wide block. Numbers in `reference/perf.md`.

### `sort`

Force facet ordering. Two shapes:

- **List** — explicit order: `"sort": ["Left", "Center", "Right"]`
- **Dict** — per-factor ascending flag: `"sort": {"age_group": True, "education": False}`

Leave unset to inherit the annotation's category order. Only set when the annotation is right but you want a different per-chart order.

### `plot_args`

Extra kwargs forwarded to the concrete plot function; allowed keys come from the `args` map on `@stk_plot(...)`. Examples: `{"normalized": True}` for `stacked_columns`, `{"bw": 0.3}` for `violin`, `{"sort_descending": True}` for `diff_columns`.

### `val_name` / `val_format` / `val_range`

Display-level overrides on the aggregated value: rename the value column, change its format string (e.g. `"0.1%"`), or clamp the numeric range. Use when a plot's default axis labelling is close but not quite right.

### `n_facet_cols` / `internal_facet`

Grid layout controls when a plot wraps multiple facets. Rarely needed — the defaults follow the registered `factor_columns` count.

### `col_meta` — descriptor-local annotation overrides

**Temporary** overrides for a one-off chart that needs a different scale than the annotation, *without* editing it:

```python
{"plot": "likert_bars", "res_col": "pol_interest",
 "col_meta": {"pol_interest": {"neutral_middle": "Somewhat interested"}}}
```

The merged result is revalidated, so an override contradicting the annotation raises. To read a categorical column as numbers use `convert_res="continuous"`, not `col_meta: {"continuous": true}`. `col_meta` **cannot introduce a column** — it only overrides meta on columns that exist. If you set the same override from several call sites, the annotation is wrong; fix it there.

### `res_meta` — a virtual block

Loose columns that aren't in an annotation block are not a wall: `res_meta` builds a block descriptor-locally, injected into `meta.structure` before processing, behaving exactly like a declared one — unpivots into `question`, crosses with `factor_cols`, takes block filters:

```python
# a dozen "do you use X" media columns, no block in the annotation
{"plot": "columns", "res_col": "media", "factor_cols": ["question", "age_group"], "agg_fn": "mean",
 "res_meta": {"name": "media", "scale": {"continuous": True},
              "columns": [["tv_daily"], ["radio_daily"], ["web_daily"]]}}
```

`scale` omitted is inherited from the first column's existing meta. `res_meta` is for one-off shapes — a virtual block declared from several call sites belongs in the annotation.

## Registering a custom transform

`custom_row_transforms` is a public, mutable registry, and `cont_transform` validates against the live registries — a name registered at import time in a dashboard is accepted without patching stk:

```python
from salk_toolkit.pp import custom_row_transforms

def _therm_destination(p: np.ndarray) -> np.ndarray:
    """(n_rows, n_cols) -> (n_rows, n_cols); rows are respondents, cols the battery."""
    ...

custom_row_transforms["therm-destination"] = (_therm_destination, ".1%")
```

The callable takes the battery as an `(n_rows, n_cols)` numpy array *while the data is still wide* and returns the same shape; the second tuple element is the display format.

**Prefer a polars expression to a numpy callback.** If the transform is expressible with `max_horizontal` / `min_horizontal` / `sum_horizontal` / `concat_list(...).list.eval(...)`, put it in the sibling `ordered_expr_transforms` registry and it stays in the query plan — `custom_row_transforms` runs through `map_batches`, which is slower and deadlocks under new streaming (`reference/perf.md`).

Two constraints before planning a conversion:

- The callback sees *only* the `res_col` battery, so it cannot reach another column. A different or combined **weighting** is not a limitation — that belongs in the descriptor's `weights`.
- **Keep aggregation out of the transform.** A row transform maps respondents → respondents; anything summing *across* respondents belongs in `agg_fn` + the weighting. Only genuinely conditional model logic (fallback branches, masking on per-respondent state) stays in code around the pipeline.

## Outputs: `return_data`, `return_input`, payload

Default path is the chart: `e2e_plot(pp_desc, ...).to_dict()` (Vega-Lite) or the Altair object for notebooks.

**`return_data=True`** gives the aggregated pandas frame instead — the *pre-shaping* aggregate. Use it when the aggregation + filter are right but the rendering isn't, when building a raw-data API, or when testing the numbers rather than the chart JSON. In a dashboard, `salk_dashboard_tools.plot.pp_data(pp_desc, af)` is the direct wrapper.

**`return_input=True`** returns the whole `PlotInput` — the same aggregate plus `filtered_size` (post-filter) and `total_size` (pre-filter) weight mass, exact row counts under `weights: False`. That is how a scope's `n` becomes a descriptor rather than a separate scan. Both count the scope, whatever the res_col's dtype or nullity. `return_input` takes precedence if both flags are passed.

**Do not compute "how many answered" — it is `filtered_size`.** That is strictly the scope size, but on model-generated data every column is scored for every respondent (verified across three dashboards: 62 lt26 columns, 24 am salience columns, rk's thermometer — zero nulls, zero NaNs), so a valid-count descriptor re-derives the scope size it was handed.

### `create_plot_payload` — plot-shaped data + metadata (PlotPayload v1)

`return_data=True` stops *before* the plot function — you get the aggregate, not the geometry. `create_plot_payload` runs the plot's own shaping code and serializes the result plus everything a renderer needs, per facet-grid cell.

```python
from salk_toolkit.pp import pp_transform_data, create_plot_payload, UnsupportedPayloadError

pi = pp_transform_data(full_df, data_meta, pp_desc)   # same wrangle e2e_plot uses
payload = create_plot_payload(pi, pp_desc)            # PlotPayload v1 dict
```

Contents: `cells` (2D grid of `{title, keys, columns, data}`, column-wise JSON-safe data), `facets` (order / plain-hex `colors`, default palette synthesized when the annotation has none / `neutrals`), `value_col` / `cat_col` / `val_format` / `value_range` / `filtered_size`, `grid` layout, and plot-specific `scale` (resolved hex stops + domain for matrix/geoplot) and `geo` (topojson url/object/join keys). Labels come through unescaped — no Vega-escape artifacts.

Use it when **another engine renders** (the ECharts path in dms-plots-api `/plot-data`), or for **CSV exports of "the numbers behind the chart"** — `pd.DataFrame(cell["data"]).to_csv(...)` matches what the chart displays (boxplot whisker stats, likert segments, maxdiff Most/Least split) rather than the raw aggregate.

Coverage is universal: `payload=True` plots early-return their prepared frame on `return_df` (authoritative), every other chart-producing plot falls back to building its Altair chart and reading the frame / color-scale / geo back off it. `UnsupportedPayloadError` fires only when a plot returns no chart or frame at all (e.g. `coalition_applet`, a streamlit-only widget) — catch it and fall back to the Vega path. `get_plot_meta(name).payload` is **not** a coverage gate; adding it is an optimization/robustness choice (restructure the fn so all frame shaping precedes the `return_df` early-return, replace rather than mutate shared facet objects, pin the frame against the chart in `tests/test_plot_payload.py`). A plot whose chart layers carry *different* frames should opt in, to declare the canonical one.

## `matching_plots` — use it before forcing a plot

```python
from salk_toolkit.pp import matching_plots
matching_plots(pp_desc, af.df, af.meta, details=True)
# -> {plot_name: (priority, [reasons])}  when details=True
```

If your chosen plot isn't in the list, the metadata doesn't support it. Common culprits: the plot needs `draws=True` but the data has no `draw` column; it needs a continuous `res_col` (pick one or set `convert_res="continuous"`); it needs `requires_factor=True` (add a `factor_cols` entry); it needs an ordered facet (geo, `likert_rad_pol`). Fix the metadata / descriptor, don't bypass the check.

## Testing authored descriptors

- e2e plot tests live in `tests/test_plots.py`. `_run_plot_test` renders the chart and diffs normalised Altair JSON against `tests/reference_plots/*.json`.
- For new plot types, add a reference test following the existing ones; for new `PlotDescriptor` options, ensure at least one e2e test exercises them.
- Regenerate references with `pytest --recompute` **only after** confirming only the intended tests fail.
- Unit-test sub-helpers in `tests/test_pp.py`.

## Anti-patterns

- **Hand-writing polars aggregation** when a `pp_desc` can express it — you're re-implementing `pp_transform_data` and you will drift from tooltip / color / label conventions.
- **Concluding pp cannot do something without reading the registration.** The recurring failure is naming the wrong plot/column and then explaining the surprising result as a library limitation — see `reference/excuses.md`.
- **Patching stk to make a descriptor win over a registration.** Name a plot that registers neither `transform_fn` nor `agg_fn` instead; making registrations overridable lets callers silently break pairs the renderer depends on.
- **Hand-writing a Vega-Lite dict** for something `e2e_plot(pp_desc).to_dict()` would produce.
- **Reading labels / colors / orders from the descriptor instead of the annotation.**
- **One descriptor per statistic** over a battery, or per cell of a filter-varying matrix — see `stats`.
- **Setting `factor_cols` when the default is fine**, and **using `pl_filter` when `filter` would work.**
- **Ignoring `matching_plots` rejections.** If it says no, the plot won't render correctly.
- **Mixing `convert_res="continuous"` with `agg_fn="sum"` without thought** — `sum` of a numeric score across respondents is rarely what you want.
- **Passing `data_file=` for a dashboard endpoint** — pass the `AnnotatedFrame`'s already-open `LazyFrame` via `full_df=` so caching / auth / scope filtering stay in play.

## For more details

- Registry and pipeline: `salk_toolkit/pp/` — `registry_meta`, `e2e_plot`, `matching_plots`, `impute_factor_cols`, `pp_transform_data`, `wrangle_data`.
- Transform registries: `salk_toolkit/pp/transforms.py` — `custom_row_transforms`, `ordered_expr_transforms`.
- Payload serializer: `salk_toolkit/payload.py` — `create_plot_payload`, `UnsupportedPayloadError` (also importable from `pp`).
- Plot implementations: `salk_toolkit/plots.py` — one `@stk_plot(...)` per registered name.
- Descriptor schema: `salk_toolkit/validation.py` — `PlotDescriptor`, `FilterSpec`, `SortSpec`, `ConvertResOption`, `ContTransformOption`, `AggFnOption`.
- Annotation authoring (labels, colors, orders, `num_values`): `stk-data-annotations` skill.
- Dashboard integration (`pp_spec`, `pp_data`, `AnnotatedFrame`): `salk-dashboard` skill in `salk_dashboard_tools`.
