# Pydantic models through the io and plot pipelines (PR #15)

**Modules:** `salk_toolkit/validation.py`, `salk_toolkit/serialization.py` (new),
`salk_toolkit/io.py`, `salk_toolkit/pp.py`, `salk_toolkit/plots.py`,
`salk_toolkit/tools/explorer.py`

## Goal

After PR #14 the codebase was annotated but still passed `Dict[str, Any]` everywhere, which
is why five of pyright's seven disabled checks were disabled. This PR replaces the dict
contracts with Pydantic V2 models on both pipelines — typed metadata through io, a typed
parameter object through the plot pipeline — while keeping JSON metafiles loading exactly as
before.

## Design

**Validation models.** `validation.py` gains `ElectoralSystem(PBase)` (quotas, threshold,
`ed_threshold`, `body_size`, `first_quota_coef`, `dh_power`, exclusions, `special`) and
`MandatesDict = Dict[str, int]`, both wired into `ColumnMeta`, whose `translate` /
`translate_after` also tighten from `Optional[Dict]` to `Dict[str, str]`. Alongside
`ColumnMeta` sit `GroupOrColumnMeta` (adds `columns`, so one type covers both a block and a
bare column), `BlockScaleMeta`, and `FacetMeta` (`col`, `ocol`, `order`, `colors`,
`neutrals`, `meta`).

**Two-pass validation.** `PBase` is configured `extra="ignore"`, and `soft_validate(m: dict |
BaseModel, model: type[T]) -> T` runs the input twice: first through a strict clone built by
`_create_strict_model_class` (`extra="forbid"`) purely to collect and print warnings about
unknown fields, then through the real lenient model, returning the object. The strict clone
is built recursively, so an unknown key nested inside `structure` → block → column warns just
like a top-level one. `hard_validate` is the raising variant.

**Plot-side models live in `pp.py`, not `validation.py`,** because they reference altair
types and the registry:

- `PlotInput(PBase)` replaces the old `pparams` dict — data, `col_meta`, `value_col`,
  `cat_col`, formats and ranges, `facets: List[FacetMeta]`, `translate`, `tooltip`, colors,
  width, `alt_properties`, `outer_factors`, and a `plot_args` escape hatch for
  plot-specific options.
- `PlotMeta(PBase)` types the `@stk_plot` registry entry (`data_format`, `draws`,
  `continuous`, `n_facets`, `requires`, `agg_fn`, `sample`, `group_sizes`,
  `sort_numeric_first_facet`, `factor_columns`, `aspect_ratio`, `as_is`, `priority`, `args`,
  `hidden`, …); `get_plot_meta` returns it and `registry_meta` is `Dict[str, PlotMeta]`.

`pp_transform_data` returns a `PlotInput`; `create_plot(pi: PlotInput, pp_desc:
PlotDescriptor, ...)` consumes one; every `@stk_plot` function in `plots.py` takes a single
`p: PlotInput` instead of a long kwargs list. Descriptor `plot_args` keys are split on
arrival: those matching a `PlotInput` field override the model directly, the rest stay in
`plot_args`. `matching_plots` gets `@overload`s on its `details` flag. `e2e_plot` and
explorer soft-validate their descriptor into a `PlotDescriptor` at entry.

**Serialization.** A new `serialization.py` holds the wrap-serializers
(`serialize_pbase`, `serialize_column_meta`, `serialize_column_block_meta`,
`serialize_data_meta`) plus the column-spec list↔dict converters, so a model dumps back to
the historical JSON shape — list-form column specs, defaults omitted — rather than to
Pydantic's default rendering.

**io.** `process_annotated_data` soft-validates its input to `DataMeta` at entry;
`extract_column_meta` returns `dict[str, GroupOrColumnMeta]`; `ProcessedDataReturn` is
retyped from `MetaDict` to `DataMeta | None`, and the meta is dumped back to a dict only at
the `return_meta=True` boundary.

## Implementation notes

- `model_dump()` inside a function to dodge the type system is banned; the only sanctioned
  use is serializing a return value for backward compatibility.
- The plan called this parameter object `PlotParams`; it shipped as `PlotInput`.
- `create_plot` sniffs the callee signature, so plot functions in *other* packages can keep
  the old kwargs style and migrate on their own schedule.
- io only types its boundary. The internals still mutate the meta as a dict, because
  structure rewriting is extensive and interleaved; converting them is deferred to the io
  package refactor. For the same reason `col_meta` stays `Dict[str, Dict[str, Any]]`
  wherever the code mutates it, and typed only where it is read.
- Backward compatibility is the reason validation is two-pass at all: an old metafile with
  a since-renamed field warns and loads, instead of failing.
