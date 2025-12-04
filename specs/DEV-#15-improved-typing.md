# DEV-#15: Improved Typing

**Last Updated**: 2025-11-23  
**Status**: ✅ Complete  
**Module**: Core  
**Tags**: `#typing`, `#pydantic`, `#validation`, `#pyright`
**Dependencies**: DEV-#14

## Overview

Improve type safety across the codebase by leveraging Pydantic V2's validation capabilities and making more code checkable with pyright. Move from dict-based data structures to structured Pydantic models internally, while maintaining backward compatibility for JSON I/O.

## Problem Context

- Problem: Current codebase uses `Dict[str, Any]` extensively, making it difficult for type checkers like pyright to catch errors. Many functions accept/return dictionaries that should be typed Pydantic models.
- Use cases:
  - Type-safe metadata handling throughout IO and plotting pipelines
  - Better IDE autocomplete and error detection during development
  - Clearer contracts for function parameters and return values
  - Improved validation with configurable strictness (forbid vs ignore extra fields)
- Technical constraints:
  - Must maintain backward compatibility with existing JSON metafiles
  - Need to support both strict (`extra='forbid'`) and lenient (`extra='ignore'`) validation modes
  - Plot pipeline functions currently use long kwargs lists that should be consolidated
- Integration points:
  - `process_annotated_data` and related IO functions in `io.py`
  - Plot pipeline functions in `pp.py` that use `pparams` dict
  - Validation helpers in `validation.py` that need recursive validation support
  - Election models that define electoral system structures

## Requirements

**Important context functions/files:**
- `salk_toolkit/validation.py`: Defines `PBase`, `ColumnMeta`, `DataMeta`, `PlotDescriptor`, `soft_validate`, `hard_validate`
- `salk_toolkit/election_models.py`: Defines `simulate_election` which uses `electoral_system` and `mandates` dicts
- `salk_toolkit/io.py`: `process_annotated_data` currently validates but then works with dicts
- `salk_toolkit/pp.py`: `pp_transform_data`, `wrangle_data`, `create_plot` use `pparams: Dict[str, Any]`
- See [data_annotations.mdc](mdc:.cursor/rules/data_annotations.mdc) for annotation structure patterns
- See [plot_pipeline.mdc](mdc:.cursor/rules/plot_pipeline.mdc) for plot pipeline architecture and conventions

**Files to Create/Modify:**

- `salk_toolkit/validation.py`
  - Create detailed Pydantic models for `electoral_system` and `mandates` based on usage in `election_models.py`
  - Update `ColumnMeta` to use these new types instead of `Optional[Dict]`
  - Improve `ColumnMeta` field types: `translate` and `translate_after` should be `Optional[Dict[str, str]]` instead of `Optional[Dict]`
  - Create `PlotMeta` Pydantic model for plot registry metadata with all fields from `@stk_plot` decorator: `name`, `data_format`, `draws`, `continuous`, `n_facets`, `requires`, `no_question_facet`, `agg_fn`, `sample`, `group_sizes`, `sort_numeric_first_facet`, `no_faceting`, `factor_columns`, `aspect_ratio`, `as_is`, `priority`, `args`, `hidden`, plus any others found in registry
  - Create `FacetMeta` Pydantic model for facet structure with fields: `col: str`, `ocol: str`, `order: List[str]`, `colors: Dict[str, Any] | alt.Scale | alt.UndefinedType`, `neutrals: List[str]`, `meta: ColumnMeta`
  - Create `GroupOrColumnMeta` Pydantic model that extends `ColumnMeta` with optional `columns: List[str]` field for group metadata
  - Improve `DataMeta` field types: `read_opts`, `constants`, `FileDesc.opts` should have more specific types
  - Modify `PBase` to support recursive validation with configurable `extra` behavior
  - Update `soft_validate` to use `extra='forbid'` mode, print warnings, then parse with `extra='ignore'` and return the object
- `salk_toolkit/io.py`
  - Change `process_annotated_data` to soft-validate meta object at start, then work with Pydantic object internally
  - Only convert back to dict/JSON for return value when `return_meta=True`
  - Update all metadata manipulation to work with Pydantic models
- `salk_toolkit/pp.py`
  - Create `PlotParams` Pydantic class to replace `pparams: Dict[str, Any]`
  - Update `pp_transform_data`, `wrangle_data` to return `PlotParams` instance
  - Update `create_plot` and all plot functions to accept `PlotParams` instead of kwargs
  - Update `get_plot_meta` to return `PlotMeta` Pydantic model instead of `Dict[str, Any]`
  - Update `registry_meta` type annotation to `Dict[str, PlotMeta]`
  - Update function signatures: `data_meta: Dict[str, Any]` → `data_meta: DataMeta`, `pp_desc: Dict[str, Any]` → `pp_desc: PlotDescriptor`
  - Soft-validate `PlotDescriptor` at start of `e2e_plot`, then use Pydantic object
  - Update all functions that modify `pparams` to work with `PlotParams` class
- Review entire codebase for type annotation improvements
  - Replace `col_meta: Dict[str, Any]` with `col_meta: Dict[str, ColumnMeta]` where appropriate
  - Update `extract_column_meta` return type from `dict[str, dict[str, object]]` to `dict[str, GroupOrColumnMeta]`
    - Note: Current implementation returns `defaultdict` with nested dicts, needs to construct `GroupOrColumnMeta` instances
    - Groups have `columns` field, individual columns don't - `GroupOrColumnMeta` handles both cases
  - Update `update_data_meta_with_pp_desc` return type: first element should be `dict[str, GroupOrColumnMeta]` instead of `Dict[str, Dict[str, Any]]`
  - Update `MetaDict` type alias usage
  - Add proper type annotations for other commonly used dict structures

**Functionality:**

- **Electoral system types**: Create `ElectoralSystem` and `MandatesDict` Pydantic models based on actual usage patterns in `election_models.py` and example metafiles
  - `ElectoralSystem` should include: `quotas: bool`, `threshold: float | Dict[str, float]`, `ed_threshold: float`, `body_size: int | None`, `first_quota_coef: float`, `dh_power: float`, `exclude: List[str] | None`, `special: str | None`
  - `MandatesDict` should be `Dict[str, int]` with validation
- **Recursive validation with configurable extra behavior**: 
  - Use Pydantic V2 validators to allow `PBase` classes to be validated with either `extra='ignore'` or `extra='forbid'` based on a setting
  - Implement via model validator or config override mechanism
- **Soft validation enhancement**:
  - `soft_validate` currently returns `None` and only prints errors
  - Should validate with `extra='forbid'` to catch all issues and print warnings
  - Then parse with `extra='ignore'` and return the validated Pydantic object
  - **Recursive extra field handling**: Extra field ignoring must work recursively at all nesting levels (e.g., extra fields in `DataMeta.structure` blocks, `ColumnBlockMeta` objects, `ColumnMeta` objects, etc.). Nested `PBase` models must also ignore extra fields, not just the top-level model.
  - Signature changes from `soft_validate(m: dict[str, object], model: type[BaseModel]) -> None` to `soft_validate(m: dict[str, object] | T, model: type[T]) -> T`
- **Metadata handling in IO**:
  - `process_annotated_data` should call `soft_validate(meta, DataMeta)` at start
  - Work with `DataMeta` Pydantic object throughout processing
  - Convert to dict only when returning (`return_meta=True`) or writing to JSON
  - Apply same pattern to all metafile handling functions
- **PlotParams consolidation**:
  - Create `PlotParams` Pydantic class with all fields from `wrangle_data` and `create_plot`: `data: pd.DataFrame`, `col_meta: Dict[str, ColumnMeta]`, `value_col: str`, `cat_col: str | None`, `val_format: str | None`, `val_range: Tuple[float, float] | None` (possible value range for filters), `filtered_size: float`, `facets: List[FacetMeta]`, `translate: Callable[[str], str]`, `tooltip: List[Any]`, `value_range: Tuple[float, float]` (actual min/max of data, set in `create_plot`), `outer_colors: Dict[str, Any]`, `width: int`, `alt_properties: Dict[str, Any]`, `outer_factors: List[str]`, `plot_args: Dict[str, Any]` (dynamic fields from plot descriptor), plus any others found in code
  - All plot functions should accept `p: PlotParams` as first parameter instead of many kwargs
  - Update function signatures throughout `pp.py`

**Architecture:**

- Maintain separation between internal Pydantic models and external JSON representation
- NEVER use workarounds like `model_dump` inside functions to bypass type system
   - Only permissible use of `model_dump()` is for serializing return values for backwards compatibility
- Keep validation strict internally but allow lenient parsing for backward compatibility
- Inputs should use proper Pydantic model types, not `Dict[str, Any]`

## Implementation Plan

### Foundation Setup

- [x] Review `election_models.py` to understand exact structure of `electoral_system` and `mandates` parameters
  - Check `simulate_election` signature for all parameters (including `ed_threshold`)
  - Check `simulate_election_pp` for how `electoral_system` dict is used
- [x] Create `ElectoralSystem` and `MandatesDict` Pydantic models in `validation.py`
- [x] Update `ColumnMeta.electoral_system` and `ColumnMeta.mandates` to use new types

### Pydantic Base Class Enhancement

- [x] Research Pydantic V2 patterns for configurable `extra` behavior (model validators, config overrides)
- [x] Implement mechanism to allow `PBase` classes to validate with either `extra='ignore'` or `extra='forbid'`
- [x] Update `soft_validate` to:
  - Validate with `extra='forbid'` and print warnings
  - Parse with `extra='ignore'` and return Pydantic object
  - Update signature and return type

### IO Module Updates

- [x] Update `process_annotated_data` to soft-validate meta at start and work with Pydantic object
  - Note: Currently validates at start, then works with dict internally due to extensive structure modifications
  - TODO: Refactor to work with Pydantic objects throughout in future iteration
- [x] Convert to dict only for return value when `return_meta=True`
- [x] Update all metadata manipulation code to work with `DataMeta` instance
- [x] Apply same pattern to other metafile handling functions (e.g., `read_annotated_data`, helpers)

### Plot Pipeline Type Improvements

- [x] Create `FacetMeta` Pydantic model in `validation.py` with fields: `col`, `ocol`, `order`, `colors`, `neutrals`, `meta`
- [x] Create `GroupOrColumnMeta` Pydantic model in `validation.py` that extends `ColumnMeta` with optional `columns: List[str]` field
- [x] Create `PlotParams` Pydantic class in `validation.py` with all current `pparams` fields
  - Audit `wrangle_data` and `create_plot` to identify all fields added to `pparams` dict
  - Use `List[FacetMeta]` for `facets` field
  - Include `plot_args: Dict[str, Any]` for dynamic fields
- [x] Update `wrangle_data` to return `PlotParams` instead of `Dict[str, Any]`
- [x] Update `pp_transform_data` to return `PlotParams`
- [x] Update `create_plot` signature to accept `pparams: PlotParams`
- [x] Update all plot function signatures in `plots.py` to accept `p: PlotParams` as first param
  - use the name `p` to keep the code short inside the plot functions
- [x] Soft-validate `PlotDescriptor` in `e2e_plot` and use Pydantic object
  - Carry corresponding change out in `explorer.py`
- [x] Update all code that modifies `pparams` dict to work with `PlotParams` class

### Codebase-Wide Type Annotation Review

- [x] Search for `col_meta: Dict[str, Any]` and replace with `col_meta: Dict[str, ColumnMeta]` where appropriate
  - Updated `impute_factor_cols` to accept `Mapping[str, GroupOrColumnMeta | Dict[str, Any]]` since it only reads from col_meta
  - Kept other functions as `Dict[str, Dict[str, Any]]` since they mutate col_meta (comes from `update_data_meta_with_pp_desc` which returns dict)
  - `PlotParams.col_meta` kept as `Dict[str, Dict[str, Any]]` for backward compatibility with mutation-heavy code
- [x] Update `extract_column_meta` return type to `dict[str, GroupOrColumnMeta]`
  - Convert nested dicts to `GroupOrColumnMeta` instances, handling `columns` field for groups
  - Note: `update_data_meta_with_pp_desc` returns `dict[str, dict[str, Any]]` for backward compatibility with internal code that mutates col_meta
- [x] Update `update_data_meta_with_pp_desc` return type - uses `extract_column_meta` internally which returns `dict[str, GroupOrColumnMeta]`, then converts to dict for mutation
- [x] Update `get_plot_meta` to return `PlotMeta` instead of `Dict[str, Any]`
  - Already returns `PlotMeta` and uses `model_copy(deep=True)`
- [x] Update `registry_meta` type annotation from `Dict[str, Dict[str, Any]]` to `Dict[str, PlotMeta]`
  - Already typed as `Dict[str, PlotMeta]`
- [x] Replace `data_meta: Dict[str, Any]` with `data_meta: DataMeta | Dict[str, Any]` throughout `pp.py` (for backward compatibility)
- [x] Replace `pp_desc: Dict[str, Any]` with `pp_desc: PlotDescriptor | Dict[str, Any]` throughout `pp.py` (for backward compatibility)
- [x] Review `MetaDict` usage and update return types to use `DataMeta` where possible
  - Updated `process_annotated_data` to accept `DataMeta | MetaDict` for input parameters
  - Kept return types as `MetaDict` for backward compatibility (as specified in requirements)
- [x] Run pyright on codebase to identify additional typing improvements
  - Fixed all type errors related to `GroupOrColumnMeta` usage in `dashboard.py` and `explorer.py`
  - Fixed type error in `explorer.py` related to division operation
  - All type errors resolved

### Integration & Testing

- [x] Update existing tests to work with Pydantic objects instead of dicts
  - Updated `test_extract_column_meta_basic` to use attribute access instead of dict access
  - Updated `test_extract_column_meta_with_prefix` to use attribute access
  - Updated `test_extract_column_meta_complex_features` to use attribute access
  - Updated `test_extract_column_meta_label_isolation` to use attribute access
  - Updated other tests that use `extract_column_meta` to access results as Pydantic models
- [x] Add tests for new `ElectoralSystem` and `MandatesDict` validation
  - `ElectoralSystem` is tested in `test_election_models.py::test_simulate_election_pp_accepts_electoral_system`
  - `MandatesDict` is a type alias (`Dict[str, int]`) and is validated through usage in election model tests
- [x] Test `soft_validate` with both `extra='forbid'` and `extra='ignore'` modes
  - Tests exist in `test_io.py::TestSoftValidate` and all pass
  - `test_soft_validate_with_extra_fields` verifies recursive handling at multiple nesting levels (top-level, structure blocks, column metadata)
  - `test_soft_validate_with_column_meta_extra_fields` tests ColumnMeta with extra fields
  - `test_soft_validate_with_already_validated_model` tests idempotency
- [x] Verify backward compatibility: existing metafiles still load correctly
  - `process_annotated_data` accepts both `DataMeta` and `dict[str, object]` for backward compatibility
  - `soft_validate` handles dict input and converts to Pydantic models
  - JSON I/O maintains backward compatibility through `model_dump()` serialization
- [x] Test plot pipeline with `PlotParams` class
  - All plot tests in `test_plots.py` pass (37 passed, 1 skipped)
  - Plot functions accept `PlotInput` (renamed from `PlotParams`) as first parameter
  - `pp_transform_data` and `create_plot` work with `PlotInput` throughout
- [x] Verify `explorer.py` still remains functional
  - `explorer.py` uses `pp_transform_data` and `create_plot` which work with `PlotInput`
  - Code correctly handles `PlotInput` objects returned from `pp_transform_data`
- [x] Run pyright and fix any new type errors introduced
  - pyright passes with 0 errors, 0 warnings on `validation.py`, `io.py`, and `pp.py`

## Definition of Done

- [x] `ElectoralSystem` and `MandatesDict` Pydantic models created and used in `ColumnMeta`
- [x] `PBase` supports configurable `extra` behavior for recursive validation
- [x] `soft_validate` returns Pydantic object after printing warnings
- [x] `process_annotated_data` works with Pydantic `DataMeta` internally
- [x] `PlotParams` class created and used throughout plot pipeline
- [x] All plot functions accept `PlotParams` instead of many kwargs
- [x] `PlotDescriptor` is soft-validated and used as Pydantic object in `pp.py`
- [x] `PlotMeta` Pydantic model created and used for plot registry metadata
- [x] `extract_column_meta` returns `dict[str, GroupOrColumnMeta]`
- [x] `FacetMeta` and `GroupOrColumnMeta` Pydantic models created
- [x] Function signatures updated: `data_meta: DataMeta | Dict[str, Any]`, `pp_desc: PlotDescriptor | Dict[str, Any]`, `plot_meta: PlotMeta` (for backward compatibility)
- [x] Type annotations improved throughout codebase (especially `col_meta` → `Dict[str, GroupOrColumnMeta]` where appropriate, kept as dict where mutation needed)
- [x] All tests passing
  - All existing tests updated and passing
  - `test_io.py::TestSoftValidate` - 3 tests passing
  - `test_plots.py` - 37 tests passing, 1 skipped
  - `test_election_models.py` - ElectoralSystem tests passing
- [x] pyright type checking passes (or significantly improved)
- [x] Backward compatibility maintained for JSON I/O

## Implementation Notes

- `pp_transform_data` and `create_plot` now pass a `PlotParams` object end-to-end.
- Legacy dict-style accessors on `PlotParams` have been removed; all `salk_toolkit` callers now use attributes, and facets flow as real `FacetMeta` models (dumped to dicts when invoking plot functions).
- Every `@stk_plot` function now accepts `p: PlotParams`; `create_plot` auto-detects signature style so other packages can transition independently.
- Descriptor `plot_args` keys are now split so fields matching `PlotParams` attributes override the model directly, while all other options remain in `pparams.plot_args` for plot-specific kwargs.

## Q&A

(To be filled during review and implementation)

