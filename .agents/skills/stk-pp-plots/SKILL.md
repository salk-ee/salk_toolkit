---
name: stk-pp-plots
description: Author stk `pp_desc` plot descriptors for `salk_toolkit.pp.e2e_plot`. Use when writing or editing `pp_desc` dicts, choosing a `plot` type from the registry, wiring `factor_cols` / `filter` / `convert_res` / `agg_fn` / `sort`, deciding between a rendered chart, `return_data=True`, or a `create_plot_payload` data+metadata payload (other renderers, CSV exports), or debugging `matching_plots` rejections.
---

# stk `pp_desc` authoring

## Overview

`e2e_plot(pp_desc, full_df=..., data_meta=...)` is the single entry point — given a survey dataframe plus its annotations, it picks a registered plot function, builds the right aggregation, and returns either an Altair chart (`to_dict()` → Vega-Lite) or the aggregated pandas frame (`return_data=True`).

All business logic — tooltips, colors, category orders, labels, translations — is read from `data_meta` via `cmeta`. **Never hand-configure any of that at the descriptor level.** If a label or color is wrong, fix the annotation (`stk-data-annotations` skill), not the descriptor.

Pipeline summary (authoritative flow: `salk_toolkit/pp.py`):

1. `matching_plots` — checks the selected plot is feasible given the data + annotations.
2. `pp_transform_data` / `wrangle_data` — lazy polars filter / unpivot / aggregate → pandas frame + `pparams`.
3. `create_plot` — adds tooltips / colors / translations, dispatches to the registered plot function.

Descriptor schema: `salk_toolkit.validation.PlotDescriptor`. Read it when anything below is ambiguous — it is the source of truth.

## If you are preparing data for a plot, 99% of the time it can be prepared via pp

The 1% is genuine row-level modelling whose output is not an aggregate — a clustering, a per-respondent imputation. Everything else is a descriptor, including things that do not look like it.

When pp appears unable to do what you want, the cause is almost always that **you named the wrong thing** — a plot that carries its own defaults, a column where a block was wanted, a draws-resolved plot for a plain proportion. The symptom then looks like a library limitation, and it is very tempting to write a workaround and a plausible explanation for it. Every mode below has been hit for real, diagnosed wrongly, and "fixed" in the dashboard or in stk before someone checked the registration.

**The check is cheap. Before concluding pp cannot do something, read the `@stk_plot(...)` registration** — `data_format`, `draws`, `agg_fn`, `transform_fn`, `requires` — and `grep -rn "@stk_plot" --include=*.py` across the whole package, not just `plots.py`. If your explanation of pp's behaviour is not something you read in the code, it is a guess.

### "pp ignores my `cont_transform`"

You named a plot that registers its own `transform_fn`, and the registration is applied. `maxdiff` is the only plot in the registry that does this (`ordered-topbot1` + `posneg_mean`) — and those two are a *matched pair* its tornado renderer needs, not defaults to be swapped.

Wrong reading: "pp has a precedence bug, descriptors should win." Right reading: **ask for the plot whose statistic you actually want.** `boxplots` and `columns` register no transform, so the descriptor picks it:

```python
# top-3 share of an ordered battery — NOT the maxdiff tornado statistic
{"plot": "boxplots", "res_col": "maxdiff_score", "cont_transform": "ordered-top3", "agg_fn": "mean"}
```

### "pp can't consume this wide / distributional block"

You passed a *column* name where the *block* name was wanted. Wide per-category share columns (`party_preference_<Party>` × a weight) are a block: `res_col=<block name>` unpivots them into `question`. `res_col="party_preference"` fails; `res_col="party_preference_dist"` is the block and works.

**Check `af.meta.structure` for the block name before concluding the shape is unsupported.** A battery you are about to loop over column-by-column is nearly always a declared block — that is also how you get the whole battery in one pass instead of N.

### "pp is slower / heavier than my hand-rolled version"

Check `draws` on the plot you named. A `draws=True` plot resolves per-(question, draw) cells, which is correct for a posterior quantity that needs `group_size` weights — and pure waste for a plain proportion you are about to pool straight back down.

Measured on one 7-column ownership block, same numbers to the bit:

| descriptor | time | rows returned |
|---|---|---|
| `boxplots` (`draws=True`) + manual pooling | 0.47s | 1750 |
| `columns` (`draws=False`) | 0.23s | 7 |

**Rule of thumb: `draws=True` when the statistic is a posterior quantity; `draws=False` when it is a share.** Getting this wrong looks exactly like "pp is inherently slow".

### "this is a row-level computation, not a plot"

Check the transform registry first (`salk_toolkit/pp/transforms.py`): `custom_row_transforms` and `ordered_expr_transforms`. Several things that read as bespoke numpy are registered transforms:

- **argmax across columns** = `ordered-top1` (`pl.col(x) == pl.max_horizontal(cols)` per column). "Which option does this respondent rate highest", aggregated with `agg_fn="mean"`, is a share — no numpy needed.
- **argmin** = `ordered-bot1`; both poles at once = `ordered-topbot1`.
- **softmax over a battery** = `softmax` / `softmax-ratio`; expected Plackett-Luce rank = `softmax-avgrank`.
- **rank-based scores** = `ordered-avgrank`, `ordered-warf`, `ordered-topN`.

Affine pre-steps (a temperature divisor, an additive log-prior) fold into the descriptor. Only the genuinely non-affine model logic around a transform — turnout weighting, fallback branches — stays in code.

### "this is a domain simulation, pp can't do elections"

`mandate_plot`, `party_mandates` and `coalition_applet` are registered plots — they live in **`salk_toolkit/election_models.py`**, not `plots.py`, so a grep of `plots.py` alone finds nothing and looks like proof of absence. `simulate_election_pp` takes the longform `(draw, factor, category, value)` that pp already produces; `mandates` and `electoral_system` come through `plot_args` (and belong in the annotation, e.g. `cmeta["electoral_district"]`).

Building a `(draw, district, party)` tensor by hand to feed `simulate_election` is re-implementing the pp path.

### "my `cont_transform` silently does nothing"

`_transform_cont` only runs when the response is **continuous** (`if c_meta[res_col].continuous`). Point a `cont_transform` at a categorical block — thermometers stored as `-5…5` categories, a Likert battery — and it is skipped without an error, and you get category shares back instead. The result looks plausible (a value column, a `question` column, shares summing to 1), which is exactly what makes it dangerous.

```python
# WRONG: transform ignored, returns the distribution of thermometer *values*
{"plot": "columns", "res_col": "thermometer", "cont_transform": "softmax"}
# RIGHT: convert first, then the transform applies
{"plot": "columns", "res_col": "thermometer", "convert_res": "continuous", "cont_transform": "softmax"}
```

Needs `num_values` in the annotation for the conversion to be meaningful. If a transform appears to be a no-op, check `convert_res` before suspecting the transform.

### A missing `weight_col` is silently 1.0

pp weights by the column `data_meta.weight_col` names. If that column is **absent from the frame**, pp fills the weight with `1.0` and carries on — so a weighted `sum` quietly becomes an unweighted one, with no warning and a plausible-looking result.

This bites when a dataset vintage renames its weight column (`N_voters` → `N`) and the annotation and the parquet drift apart. Hand-rolled code that detected either name kept working; a descriptor trusts the annotation alone.

**If a weighted aggregation matters, assert the declared column is present** before building the descriptor, and fail loudly:

```python
if af.meta.weight_col not in af.df.collect_schema().names():
    raise ...  # otherwise you get unweighted numbers that look fine
```

### "my model is bespoke — pp has no transform for it"

Then register one. `custom_row_transforms` is a public, mutable registry:

```python
from salk_toolkit.pp import custom_row_transforms

def _therm_destination(p: np.ndarray) -> np.ndarray:
    """(n_rows, n_cols) -> (n_rows, n_cols); rows are respondents, cols the battery."""
    ...

custom_row_transforms["therm-destination"] = (_therm_destination, ".1%")
```

The callable takes the battery as a `(n_rows, n_cols)` numpy array *while the data is still wide* and returns the same shape; the second tuple element is the display format. `cont_transform` is validated against the live registries, so a name registered at import time in a dashboard is accepted — you do not need to patch stk. (`ordered_expr_transforms` is the sibling registry for transforms expressible as polars expressions.)

**Prefer a polars expression to a numpy callback.** `custom_row_transforms` entries run through `map_batches`, which forces `projection_pushdown=False`, needs a probe row to declare its output schema, and **deadlocks under `POLARS_FORCE_NEW_STREAMING=1`**. The `ordered-*` family was migrated off numpy for exactly these reasons. If your transform is expressible with `max_horizontal` / `min_horizontal` / `sum_horizontal` / `concat_list(...).list.eval(...)`, put it in `ordered_expr_transforms` instead and it stays in the query plan.

**Two things a custom transform cannot reach — check these before planning a conversion.** The callback receives *only* the `res_col` battery as an `(n_rows, n_cols)` array, so it cannot see any other column (a per-respondent turnout probability, a filter flag). And `weight_col` lives on `DataMeta`, not `PlotDescriptor`, so a descriptor cannot ask for a different weighting than the dataset declares — and pp returns the aggregate, so per-respondent values cannot be recovered downstream to weight them yourself. A model needing a *second* per-respondent weighting is the genuine exception that stays in code.

**Keep aggregation out of the transform.** A row transform maps respondents → respondents. Anything that sums *across* respondents belongs in `agg_fn` + the annotation's `weight_col`, not inside the callback. Affine pre-steps — a temperature divisor, an additive log-prior — fold into the transform or the descriptor; genuinely conditional model logic (fallback branches, masking on per-respondent state) is the one thing that legitimately stays in code around the pipeline.

### Restricting the candidate set: filter the block

Several transforms are *relative to the columns present* — `ordered-top1` argmaxes over whatever the block contains, top-k cutoffs are computed across the battery. So the block filter is **load-bearing, not cosmetic**: it decides both which columns compete and, through them, which respondents count.

```python
# argmax over the 7 tracked parties only — not over every column in the block
{"plot": "columns", "res_col": "maksud", "cont_transform": "ordered-top1",
 "agg_fn": "mean", "filter": {"maksud": TRACKED_PARTIES}}
```

Omitting it silently changes the answer (a block carrying `Other` / `Dont know`, or dashboard-disabled columns, shifts every share).

The same filter is how you express a share **among** a subset. `columns` renormalizes `percent` within each cell *after* filtering, so restricting a categorical `res_col` to the named parties turns "share of all respondents" into "share of named-party voters":

```python
{"plot": "columns", "res_col": "party_preference", "factor_cols": ["district"],
 "filter": {"party_preference": NAMED_PARTIES}}   # → support among named voters
```

### "my numbers are close but systematically off"

pp scans the **whole annotated dataset**. If the file is multi-wave, or carries any other partition the caller normally filters out, every descriptor must say so:

```python
{"plot": "columns", "res_col": "party_preference", "filter": {"t": latest_wave}}
```

A missing wave filter does not error — it silently averages the waves together. In rk2027 that moved party shares by up to 1.3 pp, with a *consistent sign per party*, which is exactly what a plausible-but-wrong number looks like. Any per-cell diff that is small, systematic and not noise-shaped is this bug until proven otherwise.

### "my `filter` matches nothing"

pp reads values as the **annotation** declares them. A loader that normalizes to internal keys (`Reformierakond` → `reform`, trimmed strings, merged categories) is invisible to pp, so a filter or a lookup written in the dashboard's vocabulary silently returns an empty frame or an empty dict — not an error.

Translate at the pp boundary, in one place, both ways: descriptor filters take canonical values; results get mapped back to internal keys before they meet the rest of the code.

### What is genuinely not expressible

Two gaps are real, and worth knowing before you go looking:

- **Proportion above a threshold** on a continuous column or battery ("share rating this party ≥ 1"). The registry has rank transforms and scale transforms; a threshold is neither. Register a `custom_row_transform` if the threshold is a stable part of your methodology — otherwise keep it in polars.
- **Binning a continuous column** into categories. `convert_res` only goes categorical → continuous.

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

To see the live list and the full `PlotMeta` for each, run:

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

### `filter`

`{column: selection}`. Applied **before** aggregation on the lazy frame. Three value shapes:

- **Scalar** — single category: `{"gender": "Female"}`
- **List** — category subset: `{"education": ["Higher", "Secondary"]}`
- **Range** — inclusive `[None, min, max]` (either bound can be `None`):
  - `{"age": [None, 25, 65]}` — age ∈ [25, 65]
  - `{"age": [None, 18, None]}` — age ≥ 18

Group aliases declared on the column's `groups` meta are resolved too:

```python
{"party_preference": ["left_bloc"]}   # expanded per cmeta[party_preference].groups["left_bloc"]
```

For expressions polars can evaluate but `filter` can't encode, use `pl_filter` (a polars expression string evaluated on the LazyFrame). Keep descriptors declarative whenever possible — `pl_filter` is an escape hatch.

### `convert_res="continuous"` + `num_values`

Turn an ordered categorical response into a numeric one for plots that expect a continuous `res_col` (`boxplots`, `density`, `lines`, `violin`, ...). By default `num_values` comes from the column's annotation; override per-descriptor via `num_values` when the analytic scale needs to differ.

```python
{
  "plot": "boxplots",
  "res_col": "pol_interest",          # likert, 5-point
  "convert_res": "continuous",
  "factor_cols": ["age_group"],
}
```

### `cont_transform`

Applied **after** `convert_res` when a rescaling / summary is desired. Literal values from `ContTransformOption`:

- Scale-level: `center`, `zscore`, `01range`, `proportion`
- Softmax family: `softmax`, `softmax-ratio`, `softmax-avgrank`
- Ordered helpers: `ordered-avgrank`, `ordered-warf`, `ordered-top1`, `ordered-bot1`, `ordered-topbot1`, `ordered-top2`, `ordered-top3`

Most plots that need one declare a sensible default via `transform_fn` on the registration; override only when that isn't what you want.

### `agg_fn`

One of `mean | sum | posneg_mean | median | min | max`. Override the plot's registered default when the analytic question needs a different summary. Example — switching `columns` from count-proportions to mean of a continuous conversion:

```python
{
  "plot": "columns",
  "res_col": "approval",
  "convert_res": "continuous",
  "agg_fn": "mean",
  "factor_cols": ["party_preference"],
}
```

### `sort`

Force facet ordering. Two shapes:

- **List** — explicit order: `"sort": ["Left", "Center", "Right"]`
- **Dict** — per-factor ascending flag: `"sort": {"age_group": True, "education": False}`

Leave unset to inherit the annotation's category order. Only set when the annotation is right but you want a different per-chart order.

### `plot_args`

Extra kwargs forwarded to the concrete plot function; the allowed keys come from the `args` map on `@stk_plot(...)`. Examples: `{"normalized": True}` for `stacked_columns`, `{"bw": 0.3}` for `violin`, `{"sort_descending": True}` for `diff_columns`.

### `val_name` / `val_format` / `val_range`

Display-level overrides on the aggregated value: rename the value column, change its format string (e.g. `"0.1%"`), or clamp the numeric range. Use when a plot's default axis labelling is close but not quite right.

### `n_facet_cols` / `internal_facet`

Grid layout controls when a plot wraps multiple facets. Rarely needed — the defaults follow the registered `factor_columns` count.

### `res_meta` / `col_meta`

**Temporary, descriptor-local annotation overrides.** Use when a one-off chart needs a different scale than the annotation, *without* editing the annotation:

```python
{
  "plot": "likert_bars",
  "res_col": "pol_interest",
  "col_meta": {
    "pol_interest": {"neutral_middle": "Somewhat interested"},
  },
}
```

If you find yourself setting the same override from multiple call sites, the annotation is wrong — fix it there.

## When to use `return_data=True`

Default path is the chart: `e2e_plot(pp_desc, ...).to_dict()` (Vega-Lite) or the Altair object for notebooks. Pass `return_data=True` to get the aggregated pandas frame instead. Use this when:

- The aggregation + filter are right but the rendering isn't — render with a custom Altair / Vega / D3 template.
- Building a raw-data API (e.g. a custom map, a table, a frontend D3 widget).
- Writing tests that assert on the aggregated numbers, not the chart JSON.

This is the *pre-shaping* aggregate. If you need the data as the plot function draws it (geometry columns, per-cell frames) plus display metadata, use `create_plot_payload` (next section).

```python
rows = e2e_plot(pp_desc, full_df=af.df, data_meta=af.meta, return_data=True)
# rows is a pandas DataFrame; .to_dict(orient="records") → list of dicts
```

In a dashboard context, `salk_dashboard_tools.plot.pp_data(pp_desc, af)` is the direct wrapper.

## `create_plot_payload` — plot-shaped data + metadata (PlotPayload v1)

`return_data=True` stops *before* the plot function — you get the aggregate, not the geometry. `create_plot_payload` runs the plot's own shaping code and serializes the result plus everything a renderer needs, per facet-grid cell. It uses two paths per plot: `payload=True` plots early-return their prepared frame on `return_df` (the authoritative path); every other chart-producing plot falls back to building its Altair chart and reading the frame / color-scale / geo back off it — so coverage is **universal**, no per-plot annotation required.

```python
from salk_toolkit.pp import pp_transform_data, create_plot_payload, UnsupportedPayloadError

pi = pp_transform_data(full_df, data_meta, pp_desc)   # same wrangle e2e_plot uses
payload = create_plot_payload(pi, pp_desc)            # PlotPayload v1 dict
```

Payload contents: `cells` (2D grid of `{title, keys, columns, data}` with column-wise JSON-safe data), `facets` (order / plain-hex `colors` — default palette synthesized when the annotation has none / `neutrals`), `value_col` / `cat_col` / `val_format` / `value_range` / `filtered_size`, `grid` layout, and plot-specific `scale` (resolved hex stops + domain for matrix/geoplot) and `geo` (topojson url/object/join keys). Labels come through unescaped (`escape_labels=False` internally) — no Vega-escape artifacts.

Use it when:

- **Another plotting engine renders** (the ECharts path in dms-plots-api `/plot-data`) — the payload is the full contract; no Vega spec scraping.
- **CSV / tabular exports of "the numbers behind the chart"** — each cell's `data` is column-wise: `pd.DataFrame(cell["data"]).to_csv(...)`. Prefer this over `return_data=True` when the export should match what the chart displays (e.g. boxplot whisker stats, likert start/end segments, maxdiff Most/Least split) rather than the raw aggregate.

Coverage is universal: any chart-producing plot yields a payload. `UnsupportedPayloadError` fires only when a plot returns no chart/frame at all (e.g. `coalition_applet`, a streamlit-only widget) and hasn't opted into `return_df` — catch it and fall back to the Vega path. `get_plot_meta(name).payload` is **not** a coverage gate; it just marks which plots take the authoritative `return_df` path (shares the plot's shaping code, decoupled from Altair internals) vs. the chart-introspection fallback. Adding `payload=True` to a plot is an optimization/robustness choice — restructure its fn so *all* frame shaping precedes the `return_df` early-return, replace (don't mutate) shared facet objects, and pin its frame against the chart in `tests/test_plot_payload.py`. The fallback reads data off `chart.data` (or a `transform_lookup`'s table for geo plots), so it's coupled to Altair's object model; a plot whose chart layers carry *different* frames should opt into `return_df` to declare the canonical one.

## `matching_plots` — use it before forcing a plot

```python
from salk_toolkit.pp import matching_plots
matching_plots(pp_desc, af.df, af.meta, details=True)
# -> {plot_name: (priority, [reasons])}  when details=True
```

If your chosen plot isn't in the list, `matching_plots` is telling you the metadata doesn't support it. Common culprits:

- Plot needs `draws=True` but the data has no `draw` column.
- Plot needs a continuous `res_col` — either pick a continuous column or set `convert_res="continuous"` on an ordered categorical.
- Plot needs `requires_factor=True` — add a `factor_cols` entry.
- Plot needs an ordered facet (geo, likert_rad_pol) — check `factor_cols` points at an ordered column.

Fix the metadata / descriptor, don't bypass the check.

## Testing authored descriptors

- e2e plot tests live in `tests/test_plots.py`. `_run_plot_test` renders the chart and diffs normalised Altair JSON against `tests/reference_plots/*.json`.
- For new plot types, add a reference test following the existing ones in `tests/test_plots.py`.
- For new `PlotDescriptor` options, ensure at least one e2e test exercises them.
- Regenerate references with `pytest --recompute` **only after** confirming only the intended tests fail.

Unit-test sub-helpers in `tests/test_pp.py`.

## Anti-patterns

- **Hand-writing polars aggregation** when a `pp_desc` can express it — you're re-implementing `pp_transform_data` and you will drift from tooltip / color / label conventions.
- **Concluding pp cannot do something without reading the registration.** See "99% of the time it can be prepared via pp" above — the recurring failure is naming the wrong plot/column and then explaining the surprising result as a library limitation.
- **Patching stk to make a descriptor win over a registration.** If a registered `transform_fn`/`agg_fn` is fighting you, name a plot that registers neither. Making registrations overridable lets callers silently break pairs the renderer depends on.
- **Hand-writing a Vega-Lite dict** for something `e2e_plot(pp_desc).to_dict()` would produce. Use `return_data=True` + a small custom template only when the rendering genuinely differs.
- **Reading labels / colors / orders from the descriptor instead of the annotation.** Fix the annotation instead — it is the single source of truth for all dashboards and tools.
- **Setting `factor_cols` when the default is fine** — noise.
- **Ignoring `matching_plots` rejections.** If it says no, the plot won't render correctly; pick a different plot or fix the metadata.
- **Using `pl_filter` when `filter` would work.** `filter` is declarative and auditable; `pl_filter` is a raw string.
- **Mixing `convert_res="continuous"` with `agg_fn="sum"` without thought.** `sum` of a numeric score across respondents is rarely what you want — prefer `mean` or `posneg_mean`.
- **Passing `data_file=` for a dashboard endpoint** — always pass the `AnnotatedFrame`'s already-open `LazyFrame` via `full_df=` so the dashboard's caching / auth / scope filtering stay in play.

## For more details

- Registry and pipeline: `salk_toolkit/pp.py` — `registry_meta`, `e2e_plot`, `matching_plots`, `impute_factor_cols`, `pp_transform_data`, `wrangle_data`.
- Payload serializer: `salk_toolkit/payload.py` — `create_plot_payload`, `UnsupportedPayloadError` (also importable from `pp`).
- Plot implementations: `salk_toolkit/plots.py` — one `@stk_plot(...)` per registered name.
- Descriptor schema: `salk_toolkit/validation.py` — `PlotDescriptor`, `FilterSpec`, `SortSpec`, `ConvertResOption`, `ContTransformOption`, `AggFnOption`.
- Annotation authoring (labels, colors, orders, `num_values`): `stk-data-annotations` skill.
- Dashboard integration (`pp_spec`, `pp_data`, `AnnotatedFrame`): `salk-dashboard` skill in `salk_dashboard_tools`.
