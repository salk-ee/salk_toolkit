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
    "hard_validate",
    "soft_validate",
    "SingleMergeSpec",
    "smc_ensure_list",
    "PlotDescriptor",
]

from datetime import date, datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Self, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, model_validator, BeforeValidator
from pydantic_extra_types.color import Color
from salk_toolkit.utils import replace_constants


# Define a new base that is more strict towards unknown inputs
class PBase(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), arbitrary_types_allowed=True)


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
    translate: Optional[Dict] = None  # Translate dict applied to categories
    transform: Optional[str] = None  # Transform function in python code, applied after translate
    translate_after: Optional[Dict] = None  # Same as translate, but applied after transform

    # Model extras
    modifiers: Optional[List[str]] = (
        None  # List of columns that are meant to modify the responses on this col -> private_inputs
    )
    nonordered: Optional[List] = (
        None  # List of categories that are outside the order (Like "Don't know") -> nonordered in ordered_outputs
    )

    # Plot pipeline extras
    label: Optional[str] = None  # Longer description of the column for tooltips
    labels: Optional[Dict[str, str]] = None  # Dict matching categories to labels
    groups: Optional[Dict[str, List[str]]] = (
        None  # Dict of lists of category values defining groups for easier filtering
    )
    colors: Optional[Dict[str, Color]] = None  # Dict matching colors to categories
    num_values: Optional[List[Union[float, None]]] = None  # For categoricals - how to convert the categories to numbers
    val_format: Optional[str] = None  # Format string for the column values - only used with continuous display
    val_range: Optional[Tuple[float, float]] = (
        None  # Range of possible values for continuous variables - used for filter bounds etc
    )

    likert: bool = False  # For ordered categoricals - if they are likert-type (i.e. symmetric around center)
    neutral_middle: Optional[str] = (
        None  # For ordered categoricals - if there is a neutral category, which one should be in the middle?
    )

    topo_feature: Optional[Tuple[str, str, str]] = None  # Link to a geojson/topojson [url,type,col_name inside geodata]
    electoral_system: Optional[Dict] = None  # Information about electoral system (TODO: spec it out)
    mandates: Optional[Dict] = None  # Mandate count mapping for the electoral system

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


# This is for the block-level 'scale' group - basically same as ColumnMeta but with a few extras
class BlockScaleMeta(ColumnMeta):
    # Only useful in 'scale' block
    col_prefix: Optional[str] = None  # If column name should have the prefix added. Usually used in scale block
    question_colors: Optional[Dict[str, Color]] = None  # Dict mapping columns to different colors


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
def cspec(tpl: ColumnSpecInput) -> ParsedColumnSpec:
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


def cs_lst_to_dict(lst: Sequence[ColumnSpecInput]) -> dict[str, Tuple[str, ColumnSpecMeta]]:
    """Transform list of column specs to dictionary format.

    Args:
        lst: List of column specifications.

    Returns:
        Dictionary mapping column names to (source_name, metadata).
    """
    parsed_specs = [cspec(item) for item in lst]
    return {cn: (ocn, meta) for cn, ocn, meta in parsed_specs}


ColSpec = Annotated[Dict[str, Tuple[str, ColumnMeta]], BeforeValidator(cs_lst_to_dict)]


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


# Again, convert list to dict for easier debugging in case errors get thrown
def cb_lst_to_dict(lst: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Transform list of block specs to dictionary format.

    Args:
        lst: List of block specification dictionaries.

    Returns:
        Dictionary mapping block names to block specifications.
    """
    result: dict[str, dict[str, object]] = {}
    for block in lst:
        name = block.get("name")
        if not isinstance(name, str):
            raise TypeError("Each block specification must contain a 'name' field of type str.")
        result[name] = block
    return result


BlockSpec = Annotated[Dict[str, ColumnBlockMeta], BeforeValidator(cb_lst_to_dict)]


class FileDesc(PBase):
    """Descriptor for a single data file in a multi-file data source."""

    file: str
    opts: Optional[Dict] = None
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
    read_opts: Optional[Dict] = None  # Additional options to pass to reading function

    # Multiple files
    files: Optional[List[FileDesc]] = None

    ########################################################
    # Data processing
    ########################################################

    # Main meat of data annotations
    structure: BlockSpec

    # A set of values that can be referenced in the file below
    constants: Optional[Dict] = None

    # Different global processing steps
    preprocessing: Optional[Union[str, List[str]]] = None  # Performed on raw data
    postprocessing: Optional[Union[str, List[str]]] = None  # Performed after columns and blocks have been processed

    weight_col: Optional[str] = None  # Column to use for weighting - overriden by model to population weight column

    # List of data points that should be excluded in alyses
    excluded: List[Tuple[int, str]] = []  # Index of row + str  reason for exclusion

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
    """Validate a DataMeta object, raising errors on failure.

    Args:
        m: Dictionary or DataMeta object to validate.

    Raises:
        ValueError: If validation fails.
    """
    DataMeta.model_validate(m)


def soft_validate(m: dict[str, object], model: type[BaseModel]) -> None:
    """Validate a dictionary against a pydantic model, printing errors instead of raising.

    Args:
        m: Dictionary to validate.
        model: Pydantic model class to validate against.
    """
    try:
        model.model_validate(m)
    except ValueError as e:
        print(e)


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
    files: Optional[List[Union[str, Dict]]] = None  # Multiple files to parse
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
    col_meta: Optional[Dict[str, ColumnMeta]] = None  # Column-level metadata overrides

    # Internal / debugging
    calculated_draws: bool = True  # Whether to compute synthetic draws when metadata allows it
    data: Optional[str] = None  # Identifier for the data source (used for caching)
