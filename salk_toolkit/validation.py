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
    "PlotMeta",
    "FacetMeta",
    "GroupOrColumnMeta",
    "ElectoralSystem",
    "MandatesDict",
    "cs_dict_to_lst",
    "cb_dict_to_lst",
]

from datetime import date, datetime
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
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
    model_validator,
    BeforeValidator,
    ValidationError,
    PrivateAttr,
)
from pydantic_extra_types.color import Color
import altair as alt
from altair.utils.schemapi import UndefinedType

from salk_toolkit.utils import replace_constants

Scalar: TypeAlias = str | int | float | bool | None


# Define a new base that ignores extra fields by default (for backward compatibility)
# Warnings about extra fields are generated via soft_validate
class PBase(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), arbitrary_types_allowed=True)


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
    # Type specification
    continuous: bool = False  # For real numbers
    datetime: bool = False  # For datetimes
    categories: Optional[Union[List, Literal["infer"]]] = None  # For categoricals: List of categories or 'infer'
    ordered: bool = False  # If categorical data is ordered

    # Transformations
    translate: Dict[Scalar, Scalar] = Field(default_factory=dict)  # Translate dict applied to categories
    transform: Optional[str] = None  # Transform function in python code, applied after translate
    translate_after: Dict[str, str] = Field(default_factory=dict)  # Same as translate, but applied after transform

    # Model extras
    modifiers: List[str] = Field(
        default_factory=list
    )  # List of columns that are meant to modify the responses on this col -> private_inputs
    nonordered: List = Field(
        default_factory=list
    )  # List of categories that are outside the order (Like "Don't know") -> nonordered in ordered_outputs

    # Plot pipeline extras
    label: Optional[str] = None  # Longer description of the column for tooltips
    labels: Dict[str, str] = Field(default_factory=dict)  # Dict matching categories to labels
    groups: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # Dict of lists of category values defining groups for easier filtering
    colors: Dict[str, Color] = Field(default_factory=dict)  # Dict matching colors to categories
    num_values: Optional[List[Union[float, None]]] = None  # For categoricals - how to convert the categories to numbers
    val_format: Optional[str] = None  # Format string for the column values - only used with continuous display
    val_range: Optional[Tuple[float, float]] = (
        None  # Range of possible values for continuous variables - used for filter bounds etc
    )
    bin_breaks: Optional[Union[int, List[float]]] = None  # Optional manual breaks for discretization
    bin_labels: Optional[List[str]] = None  # Optional manual labels for discretization buckets
    question_colors: Dict[str, Color] = Field(default_factory=dict)  # Question-level color overrides

    likert: bool = False  # For ordered categoricals - if they are likert-type (i.e. symmetric around center)
    neutral_middle: Optional[str] = (
        None  # For ordered categoricals - if there is a neutral category, which one should be in the middle?
    )

    topo_feature: Optional[Tuple[str, str, str]] = None  # Link to a geojson/topojson [url,type,col_name inside geodata]
    electoral_system: Optional[ElectoralSystem] = None  # Information about electoral system
    mandates: Optional[MandatesDict] = None  # Mandate count mapping for the electoral system

    @model_validator(mode="after")
    def check_categorical(self) -> Self:
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
    question_colors: Dict[str, Color] = Field(default_factory=dict)  # Dict mapping columns to different colors


class TopKBlock(BaseModel):
    type: Literal["topk"] = "topk"
    k: Union[int, Literal["max"]] = "max"
    from_columns: Optional[Union[str, List[str]]] = None
    res_cols_prefix: Optional[str] = None
    res_cols: Optional[str] = None
    agg_index: int = -1  # TODO: Is this allowed to vary properly here?
    na_val: Optional[str] = None
    ordered: bool = False


class MaxDiffBlock(BaseModel):
    type: Literal["maxdiff"] = "maxdiff"
    best_columns: Optional[Union[str, List[str]]] = None
    worst_columns: Optional[Union[str, List[str]]] = None
    topics: Optional[str] = None
    sets: Optional[str] = None
    setindex: Optional[str] = None


ColumnSpecMeta = Dict[str, Any]
ColumnSpecInput = Union[str, list[object]]
ParsedColumnSpec = Tuple[str, str, ColumnSpecMeta]


# Transform the column tuple to (new name, old name, meta) format
def _cspec(tpl: ColumnSpecInput) -> ParsedColumnSpec:
    """Parse column specification tuple/list into [column_name, source_name, metadata].

    Args:
        tpl: Column specification (string, [name], [name, source], or [name, source, meta]).

    Returns:
        Tuple of (column_name, source_name, metadata_dict).
    """
    if isinstance(tpl, list):
        if not tpl:
            raise TypeError("Column specification lists must contain at least the new column name.")

        raw_cn = tpl[0]
        if not isinstance(raw_cn, str):
            raise TypeError("Column specification must start with the target column name.")
        cn = raw_cn  # column name

        raw_sn = tpl[1] if len(tpl) > 1 else cn
        sn = raw_sn if isinstance(raw_sn, str) else cn  # source column

        if len(tpl) == 3:
            raw_meta = tpl[2]
        elif len(tpl) == 2 and isinstance(tpl[1], dict):
            raw_meta = tpl[1]
        else:
            raw_meta = {}

        if not isinstance(raw_meta, dict):
            raise TypeError("Column metadata must be provided as a dictionary when present.")
        o_cd = raw_meta
    else:
        cn = sn = tpl
        o_cd = {}
    return (cn, sn, o_cd)


# Transform list to dict for better error readability


def _cs_lst_to_dict(
    lst: Sequence[ColumnSpecInput] | dict[str, Tuple[str, ColumnSpecMeta]],
) -> dict[str, Tuple[str, ColumnSpecMeta]]:
    """Transform list of column specs to dictionary format.

    Args:
        lst: List of column specifications, or already a dict.

    Returns:
        Dictionary mapping column names to (source_name, metadata).
    """
    # If already a dict, return as-is
    if isinstance(lst, dict):
        return lst

    parsed_specs = [_cspec(item) for item in lst]
    return {cn: (ocn, meta) for cn, ocn, meta in parsed_specs}


ColSpec = Annotated[Dict[str, Tuple[str, ColumnMeta]], BeforeValidator(_cs_lst_to_dict)]


def cs_dict_to_lst(d: dict[str, Tuple[str, ColumnSpecMeta]]) -> list[ColumnSpecInput]:
    """Transform dict of column specs back to list format (inverse of cs_lst_to_dict).

    Args:
        d: Dictionary mapping column names to (source_name, metadata).

    Returns:
        List of column specifications in original format.
    """
    result: list[ColumnSpecInput] = []
    for cn, (sn, meta) in d.items():
        if sn == cn:
            if meta:
                result.append([cn, meta])
            else:
                result.append(cn)
        else:
            if meta:
                result.append([cn, sn, meta])
            else:
                result.append([cn, sn])
    return result


class ColumnBlockMeta(PBase):
    name: str  # Name of the block
    scale: Optional[BlockScaleMeta] = None  # Shared column meta for all columns inside the block

    # List of columns, potentially with their ColummnMetas
    columns: ColSpec

    subgroup_transform: Optional[str] = None  # A block-level transform performed after column level transformations

    # Block level flags
    generated: bool = False  # This block is for data that is generated, i.e. not initially in the file.
    hidden: bool = False  # Use this to hide the block in explorer.py
    create: Optional[Union[TopKBlock, None]] = None  # TODO: None -> MaxDiff


class FacetMeta(PBase):
    """Facet definition consumed by the plotting pipeline."""

    col: str  # Column name used for faceting within the processed dataframe
    ocol: str  # Original column (before translations or label tweaks)
    order: List[str] = Field(default_factory=list)  # Ordered categories for the facet column
    colors: Optional[Dict[str, Any] | alt.Scale | UndefinedType] = None  # Altair-ready color definition
    neutrals: List[str] = Field(default_factory=list)  # Likert neutral categories to mute in gradients
    meta: ColumnMeta  # Full metadata reference for the facet column


# Again, convert list to dict for easier debugging in case errors get thrown
def _cb_lst_to_dict(lst: Sequence[dict[str, object]] | dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Transform list of block specs to dictionary format.

    Args:
        lst: List of block specification dictionaries, or already a dict.

    Returns:
        Dictionary mapping block names to block specifications.
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


def cb_dict_to_lst(d: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Transform dict of block specs back to list format (inverse of cb_lst_to_dict).

    Args:
        d: Dictionary mapping block names to block specifications.

    Returns:
        List of block specification dictionaries.
    """
    return list(d.values())


class FileDesc(BaseModel):
    """Descriptor for a single data file in a multi-file data source."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    file: str
    opts: Dict = Field(default_factory=dict)
    code: Optional[str] = None  # Short code identifier for the file (e.g., 'F1', 'F2' or 'wave1', 'wave2')


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

    # Single input file
    file: Optional[str] = None  # Name of the file, with relative path from this json file
    read_opts: Dict = Field(default_factory=dict)  # Additional options to pass to reading function

    # Multiple files
    files: Optional[List[FileDesc]] = None

    ########################################################
    # Data processing
    ########################################################

    # Main meat of data annotations
    structure: BlockSpec

    # A set of values that can be referenced in the file below
    constants: Dict = Field(default_factory=dict)

    # Different global processing steps
    preprocessing: Optional[Union[str, List[str]]] = None  # Performed on raw data
    postprocessing: Optional[Union[str, List[str]]] = None  # Performed after columns and blocks have been processed

    weight_col: Optional[str] = None  # Column to use for weighting - overriden by model to population weight column

    # List of data points that should be excluded in alyses
    excluded: List[Tuple[int, str]] = []  # Index of row + str  reason for exclusion
    total_size: Optional[float] = None  # Optional total population size override
    draws_data: Dict[str, Tuple[str, int]] = Field(default_factory=dict)  # Precomputed draws info keyed by column

    _constants_cache: dict[str, object] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def replace_constants(cls, meta: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Replace constant references in metadata with their actual values.

        Args:
            meta: Metadata dictionary potentially containing constant references.

        Returns:
            Metadata with all constant references replaced by their values.
        """
        return replace_constants(meta)

    @model_validator(mode="after")
    def check_file(self) -> Self:
        """Validate that either 'file' or 'files' is specified.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If neither 'file' nor 'files' is provided.
        """
        if self.file is None and self.files is None:
            raise ValueError("One of 'file' or 'files' has to be provided")
        return self


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
    """Create a strict version of a model class with extra='forbid' for validation warnings.

    If the model is a PBase subclass, creates a new class with extra='forbid'.
    Otherwise returns the original model class.

    Args:
        base_model: The model class to make strict.

    Returns:
        A new model class with extra='forbid' if base_model is a PBase subclass,
        otherwise returns base_model unchanged.
    """
    if issubclass(base_model, PBase):

        class StrictModel(base_model):  # type: ignore[valid-type, misc]
            model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

        return StrictModel
    return base_model


def soft_validate(m: dict[str, object] | BaseModel, model: type[T]) -> T:
    """Validate dict/model against a pydantic model, printing warnings, then returning validated object.

    First validates with a temporary strict model (extra='forbid') to catch extra fields and print warnings.
    Then validates with the normal model (extra='ignore') which allows extra fields but runs all validators.

    Args:
        m: Dictionary or Pydantic model instance to validate.
        model: Pydantic model class to validate against.

    Returns:
        Validated Pydantic model instance.
    """
    constants_cache: dict[str, object] | None = None
    if model is DataMeta:
        if isinstance(m, dict):
            consts = m.get("constants")
            if isinstance(consts, dict):
                constants_cache = dict(consts)
        elif isinstance(m, DataMeta):
            constants_cache = getattr(m, "_constants_cache", None)

    # If already a model instance of the correct type, return as-is
    if isinstance(m, model):
        return cast(T, m)

    # Convert to dict if needed
    if isinstance(m, BaseModel):
        m_dict = m.model_dump(mode="python")
    else:
        m_dict = m

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
    inst = cast(T, model.model_validate(m_dict, strict=False))
    if constants_cache is not None and isinstance(inst, DataMeta):
        inst._constants_cache = constants_cache
    return inst


class ParquetMeta(PBase):
    """Metadata bundle stored inside parquet files (data + model + miscellaneous extras)."""

    data: DataMeta
    model: Dict[str, Any] = Field(default_factory=dict)

    _extras: dict[str, object] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ParquetMeta":
        """Create a ParquetMeta instance from a raw metadata payload."""

        data_payload = payload.get("data")
        if data_payload is None:
            raise ValueError("Parquet metadata missing 'data' entry")
        data_meta = soft_validate(cast(dict[str, object] | DataMeta, data_payload), DataMeta)
        model_payload = payload.get("model")
        model_dict = cast(dict[str, Any], model_payload or {})
        extras = {k: v for k, v in payload.items() if k not in {"data", "model"}}
        inst = cls(data=data_meta, model=model_dict)
        inst._extras = extras
        return inst

    @property
    def extras(self) -> dict[str, object]:
        """Access additional metadata fields preserved from the parquet payload."""

        return self._extras

    def set_extra(self, key: str, value: object) -> None:
        """Set or update an extra metadata field."""

        self._extras[key] = value

    def payload(self) -> dict[str, object]:
        """Return a dict payload combining core fields and extras."""

        return {**self._extras, "data": self.data, "model": self.model}


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

    file: Optional[str] = None  # Single file to read
    files: Optional[List[FileDesc]] = None  # Multiple files to parse
    data: Optional[Dict[str, Any]] = None  # Alternative to file, files. Dictionary of column {name: values} pairs.
    preprocessing: Optional[Union[str, List[str]]] = None  # String of python code that can reference df
    filter: Optional[str] = None  # String of python code that can reference df and is evaluated as df[filter code]
    merge: MergeSpec = []  # Optionally merge another data source into this one
    postprocessing: Optional[Union[str, List[str]]] = None  # String of python code that can reference df


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
    col_meta: Dict[str, ColumnMeta] = Field(default_factory=dict)  # Column-level metadata overrides

    # Internal / debugging
    calculated_draws: bool = True  # Whether to compute synthetic draws when metadata allows it
    data: Optional[str] = None  # Identifier for the data source (used for caching)


class PlotMeta(PBase):
    """Metadata registered for each plot function via ``@stk_plot``."""

    name: str
    data_format: Literal["longform", "raw"] = "longform"
    draws: bool = False
    continuous: bool = False
    n_facets: Optional[Tuple[int, int]] = None
    requires: List[Dict[str, Any]] = Field(default_factory=list)
    no_question_facet: bool = False
    agg_fn: Optional[str] = None
    sample: Optional[int] = None
    group_sizes: bool = False
    sort_numeric_first_facet: bool = False
    no_faceting: bool = False
    factor_columns: int = 1
    aspect_ratio: Optional[float] = None
    as_is: bool = False
    priority: int = 0
    args: Dict[str, Any] = Field(default_factory=dict)
    hidden: bool = False
    transform_fn: Optional[str] = None
    nonnegative: bool = False
