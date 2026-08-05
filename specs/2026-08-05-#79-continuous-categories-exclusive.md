# Continuous and categorical are exclusive column types (PR #79)

**Modules:** `salk_toolkit/validation.py`, `salk_toolkit/pp/` (`wrangle.py`, `meta.py`,
`matching.py`, `common.py`), `salk_toolkit/io/create_blocks.py`, `salk_toolkit/tools/explorer.py`,
`.agents/skills/stk-data-annotations/`, `.agents/skills/stk-pp-plots/`

## Goal

A column is either continuous or categorical, and the annotation says which. Previously nothing
enforced that, so `continuous: true` alongside `categories` was constructible and each consumer
guessed differently — `matching_plots` checked `not continuous`, `impute_facet_dims` didn't, and
`pp_transform_data` built a category→number map for a column that had no meaningful categories.
Since #62 that last one nulled every row (`replace_strict({}, default=None)` over an empty map),
so `convert_res: continuous` on a natively-continuous response returned an empty grid.

## Design

**The invariant.** `ColumnMeta.check_categorical` raises when `continuous` is set together with
`categories`. Unlike the rest of that validator it runs *before* the soft-mode early return: every
io load path calls `soft_validate`, so a soft-only check would never fire on a real load. `datetime`
is orthogonal to both — it describes parsing, and parsed dates get bucketed into ordered
`"01 Dec 25"` categories, so a datetime column may legitimately be categorical.

**Scale inheritance is where the contradiction actually comes from.** No one writes both flags on
one column; they write `{"continuous": true}` on a column inside a block whose `scale` carries
`categories`, and `merge_scale_with_columns` merges the two. A column declaring its own type now
overrides the block's: the scale copy handed to `merge_pydantic_models` gets the counterpart
cleared (`categories` dropped for a continuous column, `continuous` for a categorical one). The
error is left for the case it was written for — both asserted at the same level.

**One predicate.** `ColumnMeta.is_categorical` (`categories is not None and not continuous`)
replaces five open-coded variants across `matching.py` (`nonneg`, the `convert_res` value scan,
`is_categorical`, `impute_facet_dims`) and `explorer.py`.

**`convert_res: "continuous"` dispatches on the column type** rather than on whether a map came
out non-empty:

- categorical → build `cmap` from `categories` × `num_values` and `replace_strict` (unchanged #62
  semantics: a category with no numeric value becomes null, not a failed cast);
- plain `datetime` → error;
- `continuous`, or any numeric dtype → parse the values (`Float64`, `strict=False`);
- anything else → error, rather than casting the column to all-null.

The resulting meta declares `continuous: True`, clears `categories`/`ordered`/`num_values`, and
also clears `datetime` — whatever it was parsed from, the converted column holds numbers.

**Descriptor overrides are revalidated.** `pp_desc.col_meta` was applied with
`model_copy(update=...)`, which skips validators, so an override could reconstruct the forbidden
state at plot time and silently empty the frame. The merged result now goes back through
`soft_validate`, so a contradicting override raises naming the offending categories. Reading a
categorical column as numbers is `convert_res`, not `col_meta: {"continuous": true}`.

**MaxDiff set index.** The generated block copies its source scale wholesale, which for maxdiff
carries the topic list. The set-index column holds the version number the respondent saw, so
`_create_maxdiff_metas_and_dfs` forces `continuous: True` on it — which under the merge rule is
enough to keep the topics off it. Topic vocabulary lives on the best/worst columns that contain
topics.

## Implementation notes

- The check must fire in soft mode, and the merge rule must run *before* `merge_pydantic_models`,
  not after: the merged model is validated on construction, so a post-hoc cleanup never gets to run.
- "No categories" is not expressible in a serialized column meta: `categories: null` equals the
  field default and `serialize_pbase` prunes defaults. An explicit `None` therefore cannot cancel
  an inherited scale value across a write/read round-trip — which is why the rule lives in the
  merge and not in what generators write.
- polars refuses to cast a Categorical column straight to a number, so non-numeric dtypes take a
  `cast(pl.String)` hop first. Numeric dtypes skip it: the detour stringifies every value for
  nothing and costs ~45% of `pp_transform_data` (87 → 25 ms per million rows). The mapped path
  stays `Float32` (small category codes); the parsed path is `Float64`, since those values are the
  data itself and `Float32` silently rounds anything past ~7 significant digits.
- An unannotated `val_range` stays `None` after conversion instead of being fabricated as
  `(0, 1)` — that default only ever made sense for the always-mapped path.
- `_question_meta_clone` clears `datetime` alongside `continuous`: the synthetic `question` column
  holds column names and is nominal regardless of what its group contained.
- Corpus effect, measured across 1250 parquet-embedded metas and every annotation JSON under
  `~/salk`: one genuine violation (lt `Q2status`, a 5-point living-standard rating annotated
  `continuous` + `categories: "infer"`; corrected to `ordered` categorical in the annotation and in
  the three stored artifacts carrying the old meta). Net load regressions against the previous
  `main`: zero.
