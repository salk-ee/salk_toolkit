# rank_columns: an area-proportional rank distribution (PR #75)

**Modules:** `salk_toolkit/plots.py`

## Goal

An ordinal-ranking model produces, per respondent, a score for every item in a battery. The question
it answers — "where does each item land in the ranking, and who puts it there" — has no plot: a mean
rank collapses the distribution and a maxdiff tornado shows only the two poles. `rank_columns` draws
the whole per-rank distribution, stacked by whoever you are comparing.

## Design

x is the discrete rank, each column stacked by the first facet, remaining facets to the outer grid.
`factor_cols=["party", "question"]` gives per-question panels stacked by party; dropping `question`
gives per-party panels stacked by topic.

The plot consumes the categorized-transform primitive from #76: it registers
`transform_fn="ordered-avgrank"`, `convert_res="categorical"` and `agg_fn="sum"`, so a bare
`{"plot": "rank_columns", "res_col": <block>}` delivers the weighted per-rank aggregate and the
function itself only does geometry. The axis is drawn best-rank-first, which is a draw-time
reversal only: the category scale keeps its own direction, so `sort` orders facets by mean rank the
way it reads on every other plot. Because the
aggregation stays in the lazy plan the plot receives one row per (panel, rank, stack) cell — 638 rows
for an 8-question battery over 200k respondents, against the 1.6M a raw-format delivery would hand
over — and the annotation's weight column applies.

**The height cap is the reason the plot exists.** On a shared y scale one dominant rank flattens every
other panel: in an 18-topic Estonian maxdiff ~36% of respondents rank one topic last, leaving the
consensus topics in the bottom quarter of their axes. So a column's height is capped at
`max_height_ratio` × the average column total (2/N of the panel) and the overflow goes into **width**,
keeping area proportional to the true share — a partial mosaic. Widths renormalize per panel so the
facet grid stays aligned, and the y domain is pinned to the tallest column rather than rounded up.

`split_groups=N` adds grid lines splitting the ranks into N equal-count groups (3 = top/middle/bottom
third); `center=True` grows columns symmetrically about a midline for a streamgraph-like silhouette.

## Implementation notes

- **Geometry is explicit `x0/x2/y0/y2` rects, not a band scale** — variable widths and centering both
  need it, so the rank labels are a text layer at each column's true centre rather than axis ticks.
- **The separator and label layers filter the plot's own frame.** The facet operator partitions only
  top-level data, so a separate frame would repeat identically in every panel.
- **The grid is completed with `observed=False`** before the geometry runs: a rank absent from one
  panel still occupies its slot, or the cumulative x layout would shift that panel out of alignment
  with its neighbours.
- **Tests assert invariants, not pinned pixels** — area proportional to share, widths tiling the
  panel, heights within the cap — which catches a 2% error in the squeeze that a chart fixture at the
  usual tolerance passes.
