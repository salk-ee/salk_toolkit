"""Validation Models
------------------

All pydantic models for annotated survey metadata now live in this module
(`06_validation.ipynb` was retired).  It defines:

- column/block metadata schemas (`ColumnMeta`, `ColumnBlockMeta`, `BlockScaleMeta`)
- helper types for filters, merges, and derived descriptors (`DataDescription`,
  `PlotDescriptor`, etc.)
- strict validation helpers (`soft_validate`, `hard_validate`) used throughout
  IO, plotting, and dashboard layers

Use these models directly when building new tooling or specs; docstrings here
replace the markdown commentary that used to sit in the notebook.
"""

__all__ = [
    "DataDescription",
    "MergeSpec",
    "FilterScalar",
    "FilterRange",
    "FilterCategories",
    "FilterValue",
    "FilterSpec",
    "SortSpec",
    "ConvertResOption",
    "ContTransformOption",
    "AggFnOption",
    "FileDesc",
    "DataMeta",
    "ParquetMeta",
    "hard_validate",
    "soft_validate",
    "SingleMergeSpec",
    "smc_ensure_list",
    "PlotDescriptor",
    "GroupOrColumnMeta",
    "ElectoralSystem",
    "MandatesDict",
]

from datetime import date, datetime
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    TypeAlias,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    ValidationInfo,
    model_validator,
    model_serializer,
    BeforeValidator,
    ValidationError,
)
from pydantic_extra_types.color import Color

from salk_toolkit.utils import replace_constants

DF = lambda dc: Field(default_factory=dc)

Scalar: TypeAlias = str | int | float | bool | None


# Define a new base that ignores extra fields by default (for backward compatibility)
# Warnings about extra fields are generated via soft_validate
class PBase(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), arbitrary_types_allowed=True)

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model and remove keys where values match defaults."""
        from salk_toolkit.serialization import serialize_pbase

        return serialize_pbase(self, handler, info)


# --------------------------------------------------------
#          ELECTORAL SYSTEM TYPES
# --------------------------------------------------------


class ElectoralSystem(PBase):
    """Electoral system parameters for election simulations."""

    quotas: bool = True  # Whether to use quota system
    threshold: Union[float, Dict[str, float]] = 0.0  # National threshold (float) or per-district (dict with 'default')
    ed_threshold: float = 0.0  # Electoral district threshold
    body_size: Optional[int] = None  # Total body size for compensation (default: sum of district mandates)
    first_quota_coef: float = 1.0  # Coefficient for first quota allocation
    dh_power: float = 1.0  # Power parameter for d'Hondt divisor
    exclude: Optional[List[str]] = ["Other"]  # List of party names to exclude from allocation
    special: Optional[str] = None  # Special system identifier (e.g., "cz" for Czech system)


MandatesDict = Dict[str, int]  # Dictionary mapping electoral districts to mandate counts


# --------------------------------------------------------
#          DATA META (JSON)
# --------------------------------------------------------


class ColumnMeta(PBase):
    # Source column name ()
    source: Optional[Union[str, Dict[str, str]]] = (
        None  # Name of the source column in the raw data. Can be a string (applies to all files),
        # a dict mapping file codes to column names, or None (defaults to the column name itself)
    )

    # Type specification
    continuous: bool = False  # For real numbers
    datetime: bool = False  # For datetimes
    categories: Optional[Union[List, Literal["infer"]]] = None  # For categoricals: List of categories or 'infer'
    ordered: bool = False  # If categorical data is ordered

    # Transformations
    translate: Dict[Scalar, Scalar] = DF(dict)  # Translate dict applied to categories
    transform: Optional[str] = None  # Transform function in python code, applied after translate
    translate_after: Dict[str, str] = DF(dict)  # Same as translate, but applied after transform

    # Model extras
    # List of columns that are meant to modify the responses on this col -> private_inputs
    modifiers: List[str] = DF(list)
    # List of categories that are outside the order (Like "Don't know") -> nonordered in ordered_outputs
    nonordered: List = DF(list)

    # Plot pipeline extras
    label: Optional[str] = None  # Longer description of the column for tooltips
    labels: Dict[str, str] = DF(dict)  # Dict matching categories to labels
    groups: Dict[str, List[str]] = DF(dict)  # Dict of lists of category values defining groups for easier filtering
    colors: Dict[str, Color] = DF(dict)  # Dict matching colors to categories
    num_values: Optional[List[Union[float, None]]] = None  # For categoricals - how to convert the categories to numbers
    val_format: Optional[str] = None  # Format string for the column values - only used with continuous display
    val_range: Optional[Tuple[float, float]] = (
        None  # Range of possible values for continuous variables - used for filter bounds etc
    )
    bin_breaks: Optional[Union[int, List[float]]] = None  # Optional manual breaks for discretization
    bin_labels: Optional[List[str]] = None  # Optional manual labels for discretization buckets
    question_colors: Dict[str, Color] = DF(dict)  # Question-level color overrides

    likert: bool = False  # For ordered categoricals - if they are likert-type (i.e. symmetric around center)
    neutral_middle: Optional[str] = (
        None  # For ordered categoricals - if there is a neutral category, which one should be in the middle?
    )

    topo_feature: Optional[Tuple[str, str, str]] = None  # Link to a geojson/topojson [url,type,col_name inside geodata]
    electoral_system: Optional[ElectoralSystem] = None  # Information about electoral system
    mandates: Optional[MandatesDict] = None  # Mandate count mapping for the electoral system

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model, excluding fields that match block_scale from context if present."""
        from salk_toolkit.serialization import serialize_column_meta

        return serialize_column_meta(self, handler, info)

    @model_validator(mode="after")
    def check_categorical(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("validation_mode") == "soft":
            return self
        if self.categories is None:
            # if not self.continuous and not self.datetime:
            #    raise ValueError('Column type undefined: need either categories, continuous or datetime')
            for f in [
                "ordered",
                "groups",
                "colors",
                "num_values",
                "likert",
                "topo_feature",
            ]:
                if getattr(self, f):
                    raise ValueError(f"Field {f} only makes sense for categorical columns {getattr(self, f)}")
        else:  # Is categorical
            if not self.ordered:
                for f in ["likert"]:  # ['num_values'] can be situationally useful in non-ordered settings
                    if getattr(self, f):
                        raise ValueError(f"Field {f} only makes sense for ordered categorical columns")
        return self


class GroupOrColumnMeta(ColumnMeta):
    """Column metadata that can optionally describe a grouped question."""

    columns: Optional[List[str]] = None
    col_prefix: Optional[str] = None


# This is for the block-level 'scale' group - basically same as ColumnMeta but with a few extras
class BlockScaleMeta(ColumnMeta):
    # Only useful in 'scale' block
    col_prefix: Optional[str] = None  # If column name should have the prefix added. Usually used in scale block
    question_colors: Dict[str, Color] = DF(dict)  # Dict mapping columns to different colors


class TopKBlock(PBase):
    type: Literal["topk"] = "topk"
    k: Union[int, Literal["max"]] = "max"
    from_columns: Union[str, List[str]]
    res_columns: Union[str, List[str]]  # Has to be list if from_columns is list
    agg_index: int = -1
    na_vals: Optional[List[str]] = []
    translate_after: Dict[str, str] = DF(dict)
    from_prefix: Optional[str] = None  # If from_columns is list, prefix will be removed to enable translation


class MaxDiffBlock(PBase):
    type: Literal["maxdiff"] = "maxdiff"
    name: Optional[str] = None
    best_columns: Union[str, List[str]]
    worst_columns: Union[str, List[str]]
    set_columns: Optional[Union[str, List[str]]] = None  # Mutually exclusive with setindex. Only one is specified.
    setindex_column: Optional[Union[str, List[object]]] = None  # Keeps metadata tuple used in annotations.
    topics: Optional[List[str]] = None
    sets: Optional[List[List[int]]] = None
    scale: Optional[Dict[str, Any]] = None


# Import _cs_lst_to_dict for BeforeValidator (needs to be at runtime)
from salk_toolkit.serialization import _cs_lst_to_dict  # noqa: E402

ColSpec = Annotated[Dict[str, ColumnMeta], BeforeValidator(_cs_lst_to_dict)]


class ColumnBlockMeta(PBase):
    name: str  # Name of the block
    scale: Optional[BlockScaleMeta] = None  # Shared column meta for all columns inside the block

    # List of columns, potentially with their ColummnMetas
    columns: ColSpec

    subgroup_transform: Optional[str] = None  # A block-level transform performed after column level transformations

    # Block level flags
    generated: bool = False  # This block is for data that is generated, i.e. not initially in the file.
    hidden: bool = False  # Use this to hide the block in explorer.py
    create: Union[TopKBlock, MaxDiffBlock, None] = None

    @model_validator(mode="after")
    def merge_scale_with_columns(self, info: ValidationInfo) -> Self:
        """Merge scale metadata with each column's metadata automatically on read.

        This ensures that column metadata inherits defaults from the block's scale,
        with column-specific metadata taking precedence.

        Special handling for 'label': scale labels don't propagate to columns
        unless the column explicitly sets a label.
        """
        if self.scale is None:
            return self

        from salk_toolkit.utils import merge_pydantic_models

        # Merge scale with each column's metadata
        merged_columns: dict[str, ColumnMeta] = {}
        for col_name, col_meta in self.columns.items():
            merged_meta = merge_pydantic_models(self.scale, col_meta, context=info.context)

            # Special case: Don't inherit label from scale unless explicitly set on column
            # This prevents scale-level labels from propagating to individual columns
            if col_meta.label is None and self.scale.label is not None:
                # Column didn't specify a label, but scale did - clear it
                merged_meta = merged_meta.model_copy(update={"label": None})

            merged_columns[col_name] = merged_meta

        # Update columns with merged metadata
        # Use object.__setattr__ because Pydantic models are frozen by default
        object.__setattr__(self, "columns", merged_columns)
        return self

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model and pass block_scale to context for ColumnMeta serialization."""
        from salk_toolkit.serialization import serialize_column_block_meta

        return serialize_column_block_meta(self, handler, info)


def _cb_lst_to_dict(lst: Sequence[dict[str, object]] | dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Again, convert list to dict for easier debugging in case errors get thrown.

    Transform list of block specs to dictionary format.
    """
    # If already a dict, return as-is
    if isinstance(lst, dict):
        return lst

    result: dict[str, dict[str, object]] = {}
    for block in lst:
        name = block.get("name")
        if not isinstance(name, str):
            raise TypeError("Each block specification must contain a 'name' field of type str.")
        result[name] = block
    return result


BlockSpec = Annotated[Dict[str, ColumnBlockMeta], BeforeValidator(_cb_lst_to_dict)]


class FileDesc(BaseModel):
    """Descriptor for a single data file in a multi-file data source."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    file: str
    opts: Dict = DF(dict)
    code: Optional[str] = None  # Short code identifier for the file (e.g., 'F1', 'F2' or 'wave1', 'wave2')


def _normalize_files_dict(meta: Any, read_opts_key: str = "read_opts") -> Any:  # noqa: ANN401
    """Normalize file → files list, ensuring all FileDesc entries have codes based on index.

    Args:
        meta: Dictionary or object to normalize (will be converted to dict if needed).
        read_opts_key: Key name for read options (default "read_opts" for DataMeta, not used for DataDescription).

    Returns:
        Normalized dictionary with files list and codes set based on index (F0, F1, F2, ...).
    """
    if isinstance(meta, dict):
        # If file is provided, convert to first entry in files list
        if meta.get("file") is not None:
            file_value = meta.get("file")
            read_opts = meta.get(read_opts_key, {}) if read_opts_key != "__no_read_opts__" else {}
            meta = dict(meta)  # Make a copy
            meta.pop("file", None)
            if read_opts_key != "__no_read_opts__":
                meta.pop("read_opts", None)  # read_opts is now in files
            if not meta.get("files"):
                meta["files"] = []
            meta["files"].insert(0, {"file": file_value, "opts": read_opts})

        # Ensure all files have codes based on their index in the list
        if meta.get("files") is not None:
            files = list(meta["files"])
            normalized_files = []
            for i, fd in enumerate(files):
                assert isinstance(fd, dict), "Files entries must be dicts"
                if fd.get("code") is None:
                    fd["code"] = f"F{i}"  # Set code based on index: F0, F1, F2, ...
                normalized_files.append(fd)
            meta["files"] = normalized_files

    return meta


class DataMeta(PBase):
    """Complete metadata specification for annotated survey data.

    Defines the structure, transformations, and metadata for survey datasets including
    column definitions, preprocessing steps, and categorical mappings.
    """

    #########################################################
    # Metadata
    #########################################################

    description: Optional[str] = None  # Description of the data
    source: Optional[str] = None  # Source of the data
    restrictions: Optional[str] = None  # Restrictions on the data use

    collection_start: Optional[str] = None  # Date in a way pd.to_datetime can parse it
    collection_end: Optional[str] = None  # Date in a way pd.to_datetime can parse it

    author: Optional[str] = None  # AUthor of the metafile

    ########################################################
    # Data source(s)
    ########################################################

    files: Optional[List[FileDesc]] = None
    read_opts: Dict = DF(dict)  # Additional options to pass to reading function (used by FileDesc)

    ########################################################
    # Data processing
    ########################################################

    # Main meat of data annotations
    structure: BlockSpec

    # A set of values that can be referenced in the file below
    constants: Dict = DF(dict)

    # Different global processing steps
    preprocessing: Optional[Union[str, List[str]]] = None  # Performed on raw data
    postprocessing: Optional[Union[str, List[str]]] = None  # Performed after columns and blocks have been processed

    weight_col: Optional[str] = None  # Column to use for weighting - overriden by model to population weight column

    # List of data points that should be excluded in alyses
    excluded: List[Tuple[int, str]] = []  # Index of row + str  reason for exclusion
    total_size: Optional[float] = None  # Optional total population size override
    draws_data: Dict[str, Tuple[str, int]] = DF(dict)  # Precomputed draws info keyed by column

    @model_validator(mode="before")
    @classmethod
    def normalize_files(cls, meta: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Normalize file + read_opts → files list, ensuring all FileDesc entries have a code."""
        return _normalize_files_dict(meta, read_opts_key="read_opts")

    @model_validator(mode="before")
    @classmethod
    def replace_constants(cls, meta: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Replace constant references in metadata with their actual values."""
        return replace_constants(meta, keep=True)

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model with structure and columns converted to list format."""
        from salk_toolkit.serialization import serialize_data_meta

        return serialize_data_meta(self, handler, info)


# --------------------------------------------------------
#          VALIDATION UTILITIES
# --------------------------------------------------------


def hard_validate(m: dict[str, object] | DataMeta) -> None:
    """Validate a DataMeta object with strict checking, raising errors on failure.

    Uses a strict model (extra='forbid') to ensure no extra fields are allowed.

    Args:
        m: Dictionary or DataMeta object to validate.

    Raises:
        ValidationError: If validation fails (including extra fields).
    """
    StrictDataMeta = _create_strict_model_class(DataMeta)
    StrictDataMeta.model_validate(m if isinstance(m, dict) else m.model_dump(mode="python"))


T = TypeVar("T", bound=BaseModel)


def _create_strict_model_class(base_model: type[BaseModel]) -> type[BaseModel]:
    """Create a strict version of a model class with extra='forbid' for validation warnings."""
    return _strict_model_class_cached(base_model)


@lru_cache(maxsize=None)
def _strict_model_class_cached(base_model: type[BaseModel]) -> type[BaseModel]:
    """Cached strict-model wrapper to avoid repeated dynamic class creation."""
    # If it's already strict, don't wrap again.
    if getattr(base_model, "model_config", None) and (
        cast(dict[str, Any], base_model.model_config).get("extra") == "forbid"
    ):
        return base_model

    if not issubclass(base_model, PBase):
        return base_model

    class StrictModel(base_model):  # type: ignore[valid-type, misc]
        model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    return StrictModel


def soft_validate(
    m: dict[str, object] | BaseModel,
    model: type[T],
    warnings: bool = False,
    *,
    context: dict[str, object] | None = None,
) -> T:
    """Validate dict/model against a pydantic model, printing warnings, then returning validated object.
    Validates with the normal model (extra='ignore') which allows extra fields but runs all validators.

    Args:
        m: Dictionary or Pydantic model instance to validate.
        model: Pydantic model class to validate against.
        warnings: Whether to print warnings about extra fields by doing a separate Hard validation pass

    Returns:
        Validated Pydantic model instance.
    """
    # If already a model instance of the correct type, return as-is
    if isinstance(m, model):
        return cast(T, m)

    # Convert to dict if needed
    if isinstance(m, BaseModel):
        m_dict = m.model_dump(mode="python")
    else:
        m_dict = m

    if warnings:
        # First, validate with a temporary strict model (extra='forbid') to catch extra fields
        # This generates warnings but doesn't affect the final result
        StrictModel = _create_strict_model_class(model)
        try:
            StrictModel.model_validate(m_dict)
        except ValidationError as e:
            # Print warnings for validation errors (mostly extra fields)
            print(f"Validation warnings for {model.__name__}:")
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                print(f"  {loc}: {msg}")

    # Now validate with the normal model (extra='ignore') which runs all validators
    # and allows extra fields at all nesting levels
    soft_context = dict(context or {})
    soft_context["validation_mode"] = "soft"
    inst = cast(T, model.model_validate(m_dict, strict=False, context=soft_context))
    return inst


class ParquetMeta(BaseModel):
    """Metadata bundle stored inside parquet files (data + miscellaneous extras)."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    data: DataMeta


DataSpec = Union[str, "DataDescription"]


class SingleMergeSpec(PBase):
    """Specification for merging an additional dataset with the main data."""

    file: DataSpec  # Filename to merge with
    on: Union[str, List[str]]  # Column(s) on which to merge
    add: Optional[List[str]] = None  # Column names to add with merge. If None, add all.
    how: Literal["inner", "outer", "left", "right", "cross"] = "inner"  # Type of merge. See pd.merge


# Make sure MergeSpec results in a list, even if input is a singular SingleMergeSpec


def smc_ensure_list(v: SingleMergeSpec | list[SingleMergeSpec]) -> list[SingleMergeSpec]:
    """Ensure merge spec is a list (convert single spec to list).

    Args:
        v: Single merge spec or list of merge specs.

    Returns:
        List of merge specs.
    """
    return v if isinstance(v, list) else [v]


MergeSpec = Annotated[List[SingleMergeSpec], BeforeValidator(smc_ensure_list)]

# This is the input for read_and_process_data, that allows some operations on top of data meta

# --------------------------------------------------------
#          DATA DESCRIPTION
# --------------------------------------------------------


class DataDescription(BaseModel):
    """Data source specification with optional preprocessing and filtering.

    Defines how to load data and apply transformations. Can reference a file,
    multiple files, or inline data dictionary. Supports preprocessing, filtering,
    merging, and postprocessing steps.

    Note: Uses BaseModel (not PBase) to allow for extensions like PopulationDescription.
    """

    files: Optional[List[FileDesc]] = None  # Multiple files to parse
    data: Optional[Dict[str, Any]] = None  # Alternative to file, files. Dictionary of column {name: values} pairs.
    preprocessing: Optional[Union[str, List[str]]] = None  # String of python code that can reference df
    filter: Optional[str] = None  # String of python code that can reference df and is evaluated as df[filter code]
    merge: MergeSpec = []  # Optionally merge another data source into this one
    postprocessing: Optional[Union[str, List[str]]] = None  # String of python code that can reference df

    @model_validator(mode="before")
    @classmethod
    def normalize_files(cls, meta: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Normalize file → files list, ensuring all FileDesc entries have a code based on index."""
        # DataDescription doesn't have read_opts, so use a non-existent key
        return _normalize_files_dict(meta, read_opts_key="__no_read_opts__")


# Filter spec:

# Primitive values accepted in filters
FilterScalar = Union[str, int, float, bool, date, datetime]

# Inclusive range encoded as [None, start, end]
FilterRange = Tuple[Literal[None], Optional[FilterScalar], Optional[FilterScalar]]
# List of selected values (usually categories)
FilterCategories = List[FilterScalar]
# Either single value, list of values, or range
FilterValue = Union[FilterScalar, FilterCategories, FilterRange]

# Column -> selection mapping consumed by pp_filter_data_lz
FilterSpec = Dict[str, FilterValue]

SortSpec = Union[List[str], Dict[str, bool]]
ConvertResOption = Literal["continuous"]
ContTransformOption = Literal[
    "center",
    "zscore",
    "01range",
    "proportion",
    "softmax",
    "softmax-ratio",
    "softmax-avgrank",
    "ordered-avgrank",
    "ordered-warf",
    "ordered-top1",
    "ordered-bot1",
    "ordered-topbot1",
    "ordered-top2",
    "ordered-top3",
]
AggFnOption = Literal["mean", "sum", "posneg_mean", "median", "min", "max"]


# --------------------------------------------------------
#          PLOT DESCRIPTION
# --------------------------------------------------------


class PlotDescriptor(PBase):
    """Descriptor for plot pipeline requests (``pp_desc``)."""

    # Main parameters
    plot: str  # Registered plot type (see `salk_toolkit.plots`)
    res_col: str  # Response column or question block name to visualise
    factor_cols: List[str] = []  # Facet dimensions applied to the plot
    filter: FilterSpec = {}  # Column filters applied before aggregation

    # Plotting choices
    convert_res: Optional[ConvertResOption] = None  # Convert categorical responses (currently only 'continuous')
    cont_transform: Optional[ContTransformOption] = None  # Continuous transform to apply before aggregation
    agg_fn: Optional[AggFnOption] = None  # Aggregation override for summary statistics
    sort: Optional[SortSpec] = None  # Sorting instructions for categorical dimensions
    n_facet_cols: Optional[int] = None  # Number of facet columns to display
    internal_facet: Optional[Union[bool, int]] = None  # Control inner facet (True/False or count)
    plot_args: Dict[str, Any] = {}  # Extra kwargs forwarded to the concrete plot function

    # Data meta overrides
    num_values: Optional[List[Union[int, float, None]]] = None  # Custom numeric mapping for ordered categories
    val_name: Optional[str] = None  # Rename the value column after aggregation
    val_format: Optional[str] = None  # Override value formatting string, ex '0.2f'
    val_range: Optional[Tuple[Optional[float], Optional[float]]] = None  # Override numeric bounds used downstream

    # Advanced
    pl_filter: Optional[str] = None  # Polars expression evaluated against the LazyFrame before selection
    sample: Optional[int] = None  # Sample size (with replacement) drawn before aggregation
    res_meta: Optional[ColumnBlockMeta] = None  # Temporary metadata block injected before processing
    col_meta: Dict[str, ColumnMeta] = DF(dict)  # Column-level metadata overrides

    # Internal / debugging
    calculated_draws: bool = True  # Whether to compute synthetic draws when metadata allows it
    data: Optional[str] = None  # Identifier for the data source (used for caching)
