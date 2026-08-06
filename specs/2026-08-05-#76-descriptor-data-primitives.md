# Descriptor data primitives (PR #76)

**Modules:** `salk_toolkit/validation.py`, `salk_toolkit/pp/` (`wrangle`, `transforms`,
`filters`, `plotting`, `matching`)

## Goal

pp is built for plots: one descriptor produces one value column for one chart. A dashboard
asks *data* questions — "how many rows are in this scope", "what share rate this ≥ 1",
"mean and positive share and n for every column of this battery", "what share use channel i
but not j" — and each of those either had no descriptor form or cost one descriptor per
cell, so the dashboards kept a parallel hand-written polars path beside pp. This PR widens
`PlotDescriptor` with the primitives those questions need, and fixes the defects the
conversion surfaced. Everything added is data-only or opt-in; the three SALK dashboards
moved their whole request path onto pp with 0-diff endpoint captures (68 / 87 / 57
endpoint-parameter pairs).

## Design

**Validation reaches the live registries.** `cont_transform` was a frozen `Literal`, so a
name registered from outside stk was rejected before dispatch saw it. A `field_validator`
now checks the live registries (`ordered_expr_transforms`, `custom_row_transforms`,
`SCALE_TRANSFORMS`), imported lazily to avoid the import cycle the frozen list existed to
dodge. `ContTransformOption` widens to `str`; runtime cover is unchanged and now tracks
what can actually be dispatched.

**Transform families.** `ge:<x>` produces an indicator column, so `agg_fn: "mean"` is the
share past the threshold and `"sum"` the weighted count — `ge:-inf` counts non-null
responses. `ordered-top-ties:<k>` marks every column reaching the row's k-th best *value*,
so a tie can select more than k; it generalizes the value-based `ordered-top1` the way the
rank-based `ordered-top2`/`top3` generalize `ordered-avgrank`.

**Weighting.** `weights` on the descriptor: `True` (the default) is the annotation's
declared `weight_col`, which must exist in the data — parquet and annotation drifting apart
is an error rather than a silent fallback to unweighted numbers; an annotation declaring no
weight column is unweighted. `False` is explicitly unweighted. A string is a column name,
or a polars expression building the weight per row if it contains `pl.`. Only `True` keeps
the annotation's declared `total_size`; the other modes recompute it, because a declared
population total describes the declared weighting only.

**Sizes out of the pipeline.** `e2e_plot(return_input=True)` returns the whole `PlotInput`
rather than just the aggregate, which is what makes `filtered_size` (post-filter weight
mass) and `total_size` (pre-filter) reachable. Both count the scope, independent of the
res_col's dtype or nullity, so under `weights: False` a scope's `n` is a descriptor rather
than a separate scan.

**Response binning.** `convert_res: "categorical"` buckets a numeric *response*, the inverse
of `"continuous"`; edges come from the column meta's `bin_breaks` / `bin_labels` (an int
means that many quantiles), which a descriptor's `col_meta` supplies per plot. Numeric
*facet* dimensions were already discretized this way; this adds the response side, and plot
matching and facet imputation treat a to-be-binned response as categorical.

**Expression statistics.** `stats: [{name, expr, agg_fn}]` — each entry a row-level polars
expression, all aggregated in one `group_by`. This is the primitive for cells over
*different row sets*: an overlap matrix's "uses i but not j", a war-room audience mask,
greedy marginal reach. Previously each such cell needed its own `pl_filter`, and a filter
defines a descriptor, so the row-set count became the descriptor count. `agg_fn` folds the
weighting in per statistic — `mean` is the weighted mean over rows where the expression is
non-null, `sum` the weighted sum — so boolean expressions aggregate to shares and counts
without hand-woven weights. Referenced columns are harvested via `expr.meta.root_names()`
into the projection. Output is data-only: one column per statistic, no single value column.

`stats` is rejected where it cannot be served rather than failing inside polars: on a
non-longform plot, on a block `res_col` (the block aggregation drops the columns the
expressions name), on a name colliding with a group-by dimension, alongside a
descriptor-level `agg_fn`/`cont_transform` (each statistic carries its own), and when
neither `return_data` nor `return_input` is set.

Measured on the dashboards: lt26's whole surface 112s → 83s, i.e. faster than the
hand-written polars it replaced; its media page 10.9s → 0.5s and one war-room media slice
16.1s → 2.1s (46 descriptors → 4); rk2027 156s → 115s.

**Categorizing a transformed result.** `convert_res="categorical"` combined with a
`cont_transform` describes the *result*: convert to continuous as needed, transform, then
categorize. Without `bin_breaks`/`bin_labels` that is **literal bins** — one ordered category
per distinct transformed value. This is the one categorization that works on a block, and the
reason to want it: the transform puts every column on one shared domain (a rank axis over a
battery), and the category list is derived globally instead of per column. `PlotMeta` gains
`convert_res` so a plot can register this shape as its default, `transform_fn`-style.

Literal bins never leave polars as strings. The value stays a numeric group key through the
lazy aggregation, and labels plus their numeric ordering are resolved after collection on the
aggregated frame — every distinct value survives aggregation as a group key, so the category
set is complete without a second scan. Sorting a facet on such a distribution orders by the
share-weighted mean of the category scale (for ranks, the mean rank) rather than the
near-constant mean of the shares.

## Implementation notes

- **Only the binning path is per column.** Literal bins are exempt from the single-column
  guard, and `ordered-avgrank-desc` is the descending sibling of `ordered-avgrank` (1 = the
  row's top item), which the pairwise ranking already supports.
- **NaN is not null to polars.** `proportion` and `01range` produce it on legitimate input (a
  zero-sum row, a constant column), where it would aggregate as a real "nan" category inside
  the denominator; the transformed columns get `fill_nan(None)` before categorizing.
- **Literal labels are `%g`, widened to `repr` when six significant digits are not injective** —
  two distinct values sharing a label crash `pd.Categorical`. Past 50 distinct values the
  categorization errors instead: continuous output wants bins.
- **The pair only reaches a categorization on a longform plot**, so it raises on raw/stats
  descriptors rather than switching the value format while silently skipping the transform.
- **Top-k and rank transforms count pairwise comparisons.** `pl.concat_list` + `list.eval`
  builds a list column and a sorted copy per row; summing `(col_j > col_i)` horizontally
  instead is 3.5x faster on a 500k × 18 block for identical output — `ordered-top3` 0.711s
  → 0.201s, `ordered-avgrank` 0.693s → 0.194s, `ordered-top-ties:3` 0.643s → 0.163s. The
  ranks are taken among the row's *non-null* values, so a respondent who answered 2 of 18
  columns has a top-3 rather than dropping out of the numerator while staying in the
  denominator.
- **`weights: False` must not sum a literal.** For `weights: False` the weight column is a
  synthesized `pl.lit(1.0)`, so recomputing the total summed a constant over the *pre-filter*
  frame — producing every row to learn how many there are. It takes `pl.len()` instead, off
  the scan's metadata. The cost was ~1s per descriptor *independent of data size* (450k rows
  and 19.5k rows both 1.20s), landed before any filtering, and hit every unweighted
  descriptor: single-column count 1.20s → 0.13s.
- **Integer-facet binning splits ties by rank, not by jitter.** Quantile-binning an integer
  facet has to break ties so the quantiles can split them. A random jitter makes bin
  membership depend on the RNG; seeding it inside `map_batches` only moves the dependency to
  the engine, chunking and thread count, which differ between the quantile scan (in-memory)
  and the aggregate (streaming). Spreading each tie group evenly over its unit interval by
  `cum_count().over(col)` is deterministic and exactly balanced.
- **Imputation sees virtual blocks.** `impute_facet_dims` read the raw annotation meta, so a
  `res_meta` block was invisible and raised on its own name; `e2e_plot` and `matching_plots`
  both route through `_impute_facet_dims`. Side effect: a descriptor's `col_meta` overrides
  are now visible to imputation, and specifically `categories`/`columns` overrides can change
  which dimensions get imputed.
- **Expression stats have no common-subexpression elimination.** A shared expensive
  subexpression is rebuilt inside every aggregate — an 18-column battery's top-k cutoff
  inlined into 144 statistics measured 384s against the 3.8s `cont_transform` path. Each
  statistic's own expression is materialized as a column before the `group_by`, so at least
  the weighted mean's denominator reuses it (169 cells 0.301s → 0.224s).
- **Expression stats are not universally faster.** They beat one descriptor per cell for
  cheap masks (a 169-cell exclusive-reach matrix is one 0.22s descriptor, exact to 1e-12
  against hand-written polars) and lose to `cont_transform` for within-respondent transforms
  over a wide block, where the whole battery is one vectorized pass.
- **A block is the annotation's grouping, not the analytical set.** For per-column
  aggregates a wider block is harmless; for anything computed within a respondent across the
  battery the extra columns enter the cutoff. Restrict with `filter: {block: [questions]}`,
  which narrows before the cutoff is computed.
