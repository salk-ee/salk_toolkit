# pp cost model and measurements

## Cost is descriptor count, not data volume

Per-descriptor Python overhead is ~0.6s, most of it `collect_all`. Endpoint cost is
**linear in the number of descriptors** and close to flat in rows scanned — lt26's media
page was 44 descriptors × 0.24s = 10.7s. Count descriptors when reviewing a conversion;
that is the number that moves.

The corollary is a diagnostic worth keeping: **cost that doesn't vary with data size isn't
doing work on the data.** A ~1s floor on every unweighted descriptor, identical at 450k and
20k rows, is what exposed the `weights: False` pre-filter scan (fixed: `pl.len()` off scan
metadata, 1.20s → 0.13s).

## `draws=True` only for posterior quantities

A `draws=True` plot resolves per-(question, draw) cells — correct for a posterior quantity
needing `group_size` weights, pure waste for a share you pool straight back down. One
7-column ownership block, same numbers to the bit:

| descriptor | time | rows returned |
|---|---|---|
| `boxplots` (`draws=True`) + manual pooling | 0.47s | 1750 |
| `columns` (`draws=False`) | 0.23s | 7 |

Getting this wrong looks exactly like "pp is inherently slow". When you do need draws, pool
the per-`(question, draw)` rows with `np.average(value, weights=group_size)` — `group_size`
is the post-filter weight mass, the correct pooling weight.

## Several statistics over a battery

Never one descriptor per statistic: each re-melts the block, 5× the cost measured. The
hierarchy is in SKILL.md (`stats`); these are the numbers behind it.

| shape | cost |
|---|---|
| 13-channel exclusive-reach matrix (169 cells) as expression `stats` | 0.22s |
| the same as one `pl_filter` descriptor per cell | 27 descriptors / 5.6s |
| 18-column battery's top-k cutoff inlined into 144 expression `stats` | 384s |
| the same via `cont_transform` | 3.8s |

Two limits behind the last two rows:

- **No common-subexpression elimination across aggregates.** A shared expensive
  subexpression — a per-respondent top-k cutoff, a normalizing max — is recomputed inside
  every statistic that names it.
- **Expression stats lose to `cont_transform`** when the statistic is a within-respondent
  transform over a wide block, where the whole battery is one vectorized pass. They win when
  the expressions are cheap (column masks, exclusive reach) and the alternative is a
  descriptor per cell. Measure before converting.

## Transform implementation

`custom_row_transforms` entries run through `map_batches`, which forces
`projection_pushdown=False`, needs a probe row to declare its output schema, and
**deadlocks under `POLARS_FORCE_NEW_STREAMING=1`**. The `ordered-*` family was migrated to
`ordered_expr_transforms` for exactly these reasons and got 3.5x faster on 500k × 18
(`ordered-top3` 0.711s → 0.201s, `ordered-avgrank` 0.693s → 0.194s).
