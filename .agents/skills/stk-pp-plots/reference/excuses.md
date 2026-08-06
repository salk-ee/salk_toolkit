# Misdiagnoses — symptom → cause → fix

Every entry here was hit for real, explained as a pp limitation, and worked around in a
dashboard or in stk before someone read the registration. They share a shape: **nothing
raises**, the result is the right shape and plausible, and only the numbers are wrong.

Before concluding pp cannot do something: read the `@stk_plot(...)` registration
(`data_format`, `draws`, `agg_fn`, `transform_fn`, `requires`) and
`grep -rn "@stk_plot" --include=*.py` across the package, not just `plots.py`.

| Symptom | Cause | Fix |
|---|---|---|
| Numbers close but systematically off, consistent sign per category | pp scans the **whole annotated dataset**; a multi-wave file averages the waves | filter the partition in every descriptor: `{"filter": {"t": latest_wave}}`. Moved rk2027 party shares up to 1.3pp |
| `filter` matches nothing, or a lookup returns `{}` | pp reads values as the **annotation** declares them; a loader's internal keys (`reform` for `Reformierakond`, trimmed strings, merged categories) are invisible to it | translate at the pp boundary, one place, both ways |
| `cont_transform` ignored | the named plot registers its own `transform_fn`. `maxdiff` is the only one (`ordered-topbot1` + `posneg_mean` — a matched pair its tornado renderer needs, not defaults to swap) | name a plot that registers neither: `boxplots`, `columns` |
| `cont_transform` is a silent no-op | `_transform_cont` runs only when the response is continuous; on a categorical block it is skipped without error and you get category shares back | add `convert_res: "continuous"` (needs `num_values`) |
| "pp can't consume this wide / distributional block" | a *column* name was passed where the *block* name was wanted | `res_col` takes the key in `meta.structure`: `party_preference_dist`, not `party_preference` |
| top-k / argmax shares uniformly a little low | within-respondent transforms compare the columns **in the melt**, and the melt is the whole block — `Don't know`, `No answer`, `Other` and count columns compete for the cutoff | restrict with a block filter (see `filter` in SKILL.md). Per-column stats (`mean`, `ge:`) are unaffected |
| a battery is missing items | a battery can span sibling blocks — lt26's positions are `issues` **plus** `issues_p` | check `meta.structure` for siblings before assuming one block is the battery |
| every block descriptor returns `{}` | a guard like `if res_col in lf.collect_schema().names()`: a block is a key in `meta.structure`, not a column in the frame | check both namespaces |
| weighted numbers came back unweighted | before #76 a declared `weight_col` missing from the parquet fell back to 1.0 silently | now an error; use `weights: False` when unweighted is intended |
| "this is a domain simulation, pp can't do elections" | `mandate_plot`, `party_mandates` and `coalition_applet` are registered in **`election_models.py`**, so grepping `plots.py` reads as proof of absence | `simulate_election_pp` consumes the longform pp already produces; `mandates` / `electoral_system` come through `plot_args` |
| "this is a row-level computation, not a plot" | argmax (`ordered-top1`), argmin (`ordered-bot1`), softmax and rank scores are registered transforms | check `pp/transforms.py` — `custom_row_transforms`, `ordered_expr_transforms` |
| `convert_res: "continuous"` yields all-NaN | pre-#79 an already-continuous response was mapped through categories it does not have | fixed; on an older stk convert only when `cmeta[res_col].continuous` is false |
| pp slower / heavier than the hand-rolled version | usually `draws=True` on a plot whose statistic is a plain share | see `reference/perf.md` |
