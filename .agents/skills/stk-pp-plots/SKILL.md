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
