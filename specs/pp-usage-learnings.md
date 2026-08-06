# pp usage learnings from the dashboard refactors

**Modules:** the `salk-dashboard` skill in `salk_dashboard_tools` — raw material distilled
from converting rk2027 / lt26 / am_dashboard onto the plot pipeline.

## Goal

Three dashboards were moved from bespoke polars code onto `pp` descriptors. What that
exercise taught about *holding* pp correctly — where the boundary goes, what to memoize, how
to verify a conversion — is not in the dashboard skill yet. This is the source material for
getting it there. The descriptor-level half is done: it landed in `stk-pp-plots` with #76
(`reference/excuses.md` for the failure modes, `reference/perf.md` for the cost model).

## Design

**How a dashboard should hold pp:**

- **Scope is data, not a dataframe.** The single biggest structural win. The old convention
  passed pre-filtered eager frames around, which kept every dataset resident and made call
  sites unreadable. The right shape is rk's: **a dataset name plus a plain pp-syntax `filter`
  dict** — the base filter that later queries extend, in the same vocabulary descriptors
  already use. Do not invent a bespoke scope dataclass restricted to a couple of blessed
  fields (am's frozen `Scope(dataset, wave, province)` is this anti-pattern — it re-encodes a
  two-key filter in new syntax). Hashability for caching is a technical detail: key on
  `(dataset file identity, json.dumps(filter, sort_keys=True))`. Corollary: **compute
  functions take a scope, not a frame** — a frame parameter next to scope args can silently
  disagree with the pp-backed parts.

- **One `pp_support.py` module owns the boundary.** All descriptor construction, vocabulary
  translation (canonical ↔ internal keys), wave pinning and caching in one file. Nothing
  downstream ever sees both vocabularies or writes a descriptor. Wave pinning in particular
  belongs in the descriptor factory rather than at call sites: one `_scope_filter()` that
  always injects `{"t": latest_wave}` — deriving latest as min |t| from the data, not a
  hard-coded `0` — means no descriptor can forget it. (The failure mode is in the pp skill;
  the centralize-it pattern is not.)

- **Memoize on `(dataset file identity, descriptor-json)`.** Not an optimization — a
  requirement. War-room endpoints ask the same scope-level question for every party ×
  audience; one shared descriptor keyed this way turns ~35 endpoint hits into one scan.
  `frame_cache_key(dataset)` (mtime/size) as a cache-key argument makes invalidation
  automatic on parquet republish.

- **Counts come out of the same descriptor.** `weights: False` plus `return_input=True` gives
  `filtered_size` as an exact row count off the same scan that produced the shares, so the `n`
  and the share cannot describe different rows. No parallel `scoped_lf()` sibling to keep in
  step with the descriptor filter.

- **Genuine row-level models get a one-shot artifact, not a resident frame.** The
  undecided-destination and runoff models are the only paths needing per-respondent values.
  Build every scope's output in a subprocess that exits (memory back to the OS), key the JSON
  artifact on dataset file identity, have the server only read it —
  `am_dashboard/backend/model_artifact.py` and `rk2027/backend/model_artifact.py`.

**Verification:**

- **Endpoint capture is the gate, and the baseline comes from the branch point.** Capture
  every `/api/*` route to JSON before and after; diff at 1e-3. A kept fixture rots the moment
  main moves (rk: main flipped a default and the old baseline could no longer distinguish
  intended from accidental change). Recipe: worktree at the merge-base, symlink the parquets,
  copy the harness in (it usually doesn't exist on main), capture, diff.

- **A fast entrypoint smoke test catches what neither pytest nor the type checker does.**
  Changing a compute function's signature 500s every call site you didn't grep. rk hit this
  twice before adding `tests/test_compute_entrypoints.py` (every payload builder, national and
  scoped, ~13 s) — it reproduces in seconds what the 4-minute capture run finds.

- **Sub-tolerance residue must be *explained*, not waved off.** The only accepted non-zero
  residue was ~1e-4 salience share movement, traced to pp applying the annotation-declared
  `weight_col` where the old code was unweighted (weights near-constant). "Small and
  systematic" without an explanation is the wave-mixing bug until proven otherwise.

- **Measure hot-path conversions before shipping.** pp-correct is not pp-appropriate: rk's
  thermometer-block × factor crossings were numerically 0-diff and 2× slower end-to-end from
  melt amplification (8 cols × 500k rows unpivots to ~4M). Convert cache-amortised paths
  freely; measure per-request paths cold *and* warm first.

## Implementation notes

- **Non-injective key maps: sum the group, never invert with a dict comprehension.** In am,
  `Strong_Armenia` and `Our_Way` both map to `our_way`; inverting keeps only the last and
  reads as "that party got zero", quietly reordering every ranking downstream.
- **Deterministic ordering is on you.** `group_by` does not preserve order and ties are common
  (many regions at support 0). Every `.sort()` feeding a payload needs the label as a
  tie-break, or captures diff nondeterministically.
