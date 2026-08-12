# Observed-only categoricals in the plot layer (PR #84)

**Modules:** `salk_toolkit/plots.py`, `salk_toolkit/utils.py`

## Goal

A pandas groupby with `observed=False` spans every category the *dtype* carries, not the ones the
frame holds. `create_plot_payload` calls a plot function once per grid cell with the frame already
narrowed to one panel, so under that flag each cell reinstated every *other* panel's rows — and in
`facet_dist` the empty groups were not merely waste but a crash. This PR makes the whole plot layer
group observed-only, and moves `rank_columns`' grid completion to the one place it is load-bearing.

## Design

Every `observed=False` in `salk_toolkit/` is gone; `grep -rn "observed=False" salk_toolkit/` is empty.
The shared helpers `utils.gb_in` and `utils.gb_in_apply` group observed-only, and `gb_in_apply`'s
`observed` parameter is dropped — no caller in stk, sip, sdt or the dashboards passed it. Since #82
the project requires pandas>=3.0, where `observed=True` is already the pandas default and
`pivot`/`unstack` no longer expand categoricals, so each of these sites was an active opt-out.

The completions that remain are deliberate, and they now share one helper. `utils.complete_grid(df,
levels, keys, fill)` left-joins a frame onto every combination of `levels` *within each combination of
`keys` the frame observes* — so it completes a grid without inventing a panel — and returns the level
columns as ordered categoricals, so a downstream sort or groupby follows the given order rather than
whatever the merge resolved. It replaces three inline copies: `make_start_end`'s per-group likert
completion, `marimekko`'s mosaic grid, and `rank_columns`' x slots.

**What the flip changes per plot.** `facet_dist` stops crashing: `likert_aggregate`-style helpers that
return one row per group divided by an empty group's zero sum. `likert_rad_pol` stops emitting all-NaN
phantom rows — and because it z-scores across the surviving groups when `normalized=True`, the phantom
groups moved the plotted values of every real point. `likert_bars`, `corr_matrix`, `marimekko` and
`density` are output-inert: `groupby.apply` calls the function on an empty group but contributes no
rows, and `pivot_table` already ignores unobserved columns under pandas 3.

**rank_columns.** The aggregation groups observed-only and drops rows of zero weight, so nothing that
draws nothing reaches the wire; a panel whose rows all carry zero weight disappears with them, as it
did in #83. The rank axis is instead completed in the x layout: the cumulative layout cross-joins the
observed panels with the full rank labels, so a rank no row falls into still reserves an empty slot of
unit width and every panel spans the same `[0, n_ranks]` domain in the same rank order. (Widths are
normalized per panel, so a given rank does not start at the same x in every panel — only the slot
count, their order and the domain are shared.)

## Implementation notes

- **A payload cell's row count is not a cross-product.** Cells carry only the combinations their own
  panel observes, so counts vary across the grid (88 uniformly → 77–82 per cell on the test battery),
  and an outer-factor category with no rows yields a cell with an empty data array rather than a
  crash or a panel's worth of nulls. `payload.py` still enumerates cells from the dtype categories.
- **The x-slot reservation is what keeps the old geometry.** Regenerating `test_rank_columns.json`
  removes 924 lines and adds none: every surviving row, `x0`/`x1` included, is unchanged.
- **An empty slot still carries one flat row.** The rank-label and group-separator layers filter the
  plot's own frame on `f_order == 0`, so a slot with no rows would otherwise lose its label and its
  separator. One zero row per empty slot restores both — at most `n_ranks` per panel, not the
  `n_ranks x n_stack` the old completion added.
- **marimekko's outer facet was ordered alphabetically.** Its `it.product` grid flattened the outer
  key to plain strings, so the merge dropped the categorical and every groupby after it fell back to
  lexicographic order. Through `complete_grid` the key stays categorical and the annotation's order
  holds. The visible effect is one row: the axis-title hack (`ndata.iloc[0, -1] = xcol`) lands on the
  first panel in annotation order rather than the alphabetically first one.
