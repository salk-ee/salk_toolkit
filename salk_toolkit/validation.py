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
    "MaxDiffBlock",
    "OneHotBlock",
    "TopKBlock",
]

import re
from collections.abc import Mapping
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
    get_args,
    get_origin,
    get_type_hints,
    TypeAlias,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    ValidationInfo,
    field_validator,
    model_validator,
    model_serializer,
    BeforeValidator,
    ValidationError,
)
from pydantic_extra_types.color import Color
import numpy as np

from salk_toolkit.utils import JSONValue, replace_constants

DF = lambda dc: Field(default_factory=dc)

Scalar: TypeAlias = str | int | float | bool | None


# Base model that ignores extra fields; strict checking is done via soft_validate warning pass.
class PBase(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=(), arbitrary_types_allowed=True)

    # Free-form human annotation; never consumed by code. A single string or list of lines.
    comment: Optional[Union[str, List[str]]] = None

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
    # Non-response / non-substantive answers ("Don't know", "No answer", "Refused"). Recommended field for marking
    # these on any categorical column, ordered or not: carries nonresponse semantics used by data-quality and other
    # tooling. Non-responses are always out-of-order; consumers that need the full out-of-order set take the union
    # of `nonresponse` and `nonordered` (handled model-side, see obs_models).
    nonresponse: List = DF(list)
    # Categories that fall outside the order but are NOT non-responses ("Other", "none", "Would not participate",
    # "Did not vote"). List only these extras here; `nonresponse` covers the rest. The union of the two is what ends
    # up out-of-order -> nonordered in ordered_outputs.
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
    col_prefix: Optional[str] = None  # Prefix prepended to column names in data (from scale block)

    @property
    def is_categorical(self) -> bool:
        """Has categories to work with. `datetime` is orthogonal: parsed dates can be bucketed into categories."""
        return self.categories is not None and not self.continuous

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model, excluding fields that match block_scale from context if present."""
        from salk_toolkit.serialization import serialize_column_meta

        return serialize_column_meta(self, handler, info)

    @model_validator(mode="after")
    def check_categorical(self, info: ValidationInfo) -> Self:
        # Continuous and categorical are exclusive, and this is never soft: a column claiming both is
        # pure confusion, so fail the load rather than pick a winner. (datetime is orthogonal - it
        # describes parsing, and the parsed values can be bucketed into categories.)
        if self.continuous and self.categories is not None:
            raise ValueError(f"Column is continuous, so it cannot have categories: {self.categories}")

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
    # Carried from the block by `extract_column_meta` (see ColumnBlockMeta.model_spec).
    model_spec: Optional[Dict[str, Any]] = None


# This is for the block-level 'scale' group - basically same as ColumnMeta but with a few extras
class BlockScaleMeta(ColumnMeta):
    question_colors: Dict[str, Color] = DF(dict)  # Dict mapping columns to different colors


# Import _cs_lst_to_dict for BeforeValidator (needs to be at runtime)
from salk_toolkit.serialization import _cs_lst_to_dict  # noqa: E402

ColSpec = Annotated[Dict[str, ColumnMeta], BeforeValidator(_cs_lst_to_dict)]


class ColumnBlockMeta(PBase):
    """Plain column block. Specialized blocks (`TopKBlock`, `MaxDiffBlock`) inherit from this
    and are dispatched on the `type` discriminator in :data:`BlockSpec`."""

    type: Literal["plain"] = "plain"
    name: str  # Name of the block
    scale: Optional[BlockScaleMeta] = None  # Shared column meta for all columns inside the block

    # List of columns, potentially with their ColummnMetas
    columns: ColSpec

    subgroup_transform: Optional[str] = None  # A block-level transform performed after column level transformations

    # Block level flags
    generated: bool = False  # This block is for data that is generated, i.e. not initially in the file.
    hidden: bool = False  # Use this to hide the block in explorer.py

    from_columns: Optional[Union[str, List[str]]] = None
    subgroup_labels: Optional[Dict[str, Dict[str, str]]] = None

    # Per-block override of the meta-level not_asked values (None = inherit, [] = opt out)
    not_asked: Optional[List[str]] = None

    # Observation-model description for modeling: the dict a SIP `res_cols` entry would hold
    # (any OM, e.g. {"structure": [...]} for ordinal_ranking). Typed blocks stamp a default onto
    # their output blocks; authors can set it on any block to route the block name to that OM.
    model_spec: Optional[Dict[str, Any]] = None

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
            # A column declaring its own type overrides the block's, so drop the counterpart it would inherit
            scale = self.scale
            if col_meta.continuous and scale.categories is not None:
                scale = scale.model_copy(update={"categories": None})
            elif col_meta.categories is not None and scale.continuous:
                scale = scale.model_copy(update={"continuous": False})
            merged_meta = merge_pydantic_models(scale, col_meta, context=info.context)

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

    def resolve_role_columns(self, df: "pd.DataFrame", raw_key: str) -> Dict[str, Any]:
        """Narrow regex-valued column-role fields to this sibling's columns, keyed by the raw
        capture value. Default: no roles beyond `from_columns` (handled by `_narrow_sibling`)."""
        return {}

    def source_columns(self, df: "pd.DataFrame") -> List[str]:
        """Every df-column whose cells this block reads. Default: `from_columns` resolved via regex."""
        pattern = self.from_columns
        if pattern is None:
            return [c for c in self.columns.keys() if c in df.columns]
        if isinstance(pattern, list):
            return list(pattern)
        regex = re.compile(pattern)
        return [c for c in df.columns if regex.match(c)]

    def translate_columns(self, df: "pd.DataFrame") -> List[str]:
        """Columns whose cell values `scale.translate` maps. Same as the source columns unless
        a block reads translate as something else (OneHot wide) or holds list cells (MaxDiff)."""
        return self.source_columns(df)

    def default_model_spec(self) -> Optional[Dict[str, Any]]:
        """Observation-model description this block resolves to when no explicit
        `model_spec` is authored. Plain blocks have none; typed blocks stamp an
        ordinal_ranking spec onto their processed output blocks."""
        return None

    @model_serializer(mode="wrap")
    def _serialize_model(
        self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo
    ) -> dict[str, Any]:  # type: ignore[type-arg]
        """Serialize model and pass block_scale to context for ColumnMeta serialization."""
        from salk_toolkit.serialization import serialize_column_block_meta

        return serialize_column_block_meta(self, handler, info)


class TopKBlock(ColumnBlockMeta):
    """Block for top-K aggregation of multi-select columns. The stored output
    block is an instance of this class; its `from_columns` / `res_columns`
    fields are resolved to `List[str]` by :mod:`salk_toolkit.io` (no regex on
    output). Input-only directives (`subgroup_labels` from the base class)
    are cleared on output."""

    type: Literal["topk"] = "topk"  # type: ignore[assignment]

    columns: ColSpec = DF(dict)
    k: int = Field(
        description="How many items the question let a respondent pick. Mandatory: it is also a data "
        "check - more picks than this in any row is an error, not a silent truncation."
    )
    # Source columns: an explicit list, or a regex whose capture group(s) index items/subgroups.
    from_columns: Union[str, List[str]]  # type: ignore[assignment]
    res_columns: Union[str, List[str]] = Field(
        description="Output column names; a regex substitution template (e.g. 'R\\1') when from_columns is a regex."
    )
    agg_index: int = Field(
        default=-1, description="Which regex capture group indexes the items (1-based; -1 = last group)."
    )
    not_selected: List[str] = Field(
        default_factory=list,
        description="Cell values meaning 'offered but not picked'. Nulled before packing - but they prove "
        "the question WAS asked (see not_asked).",
    )
    from_prefix: Optional[str] = None

    input_format: Literal["onehot", "ranked_onehot", "leftpacked", "ranked_leftpack"] = Field(
        default="onehot",
        description=(
            "Shape of the raw data: 'onehot' = one 0/1 column per item; 'leftpacked' = R1..Rk columns "
            "already holding chosen item names; 'ranked_*' variants additionally treat slot order as a ranking."
        ),
    )
    cell_values: bool = Field(
        default=False,
        description="onehot cells hold the item value itself (not a mention marker): leftpack the cell "
        "values instead of mapping cells to column identity. res_columns then acts as a slot-name prefix.",
    )

    def segments(self) -> List[Tuple[List[str], Optional[List[str]], bool]]:
        """Return ordinal-ranking segments for this resolved TopKBlock."""
        cols = list(self.columns.keys())
        if self.input_format in ("onehot", "leftpacked"):
            return [(cols, None, False)]
        if len(cols) < 2:
            return [(cols, None, False)] if cols else []
        chain: List[Tuple[List[str], Optional[List[str]], bool]] = []
        for i in range(len(cols) - 1):
            chain.append(([cols[i]], cols[i + 1 :], True))
        chain.append((cols, None, False))
        return chain

    def default_model_spec(self) -> Optional[Dict[str, Any]]:
        """ordinal_ranking: picked items rank above the rest of the item pool
        (`[cols, None]`); the ranked input formats additionally treat slot order
        as a ranking within the picks."""
        cols = list(self.columns.keys())
        if not cols:
            return None
        return {
            "structure": [[cols, None]],
            "ordered": self.input_format in ("ranked_onehot", "ranked_leftpack"),
        }


class MaxDiffBlock(ColumnBlockMeta):
    """Block for MaxDiff best-worst scaling experiments. The stored output
    block is an instance of this class; `best_columns` / `worst_columns` /
    `set_columns` are resolved to `List[str]` by :mod:`salk_toolkit.io`,
    index-aligned by question. Input-only directives are cleared on output.

    Translation: `scale.translate` is a `Dict[str, str]` mapping 1-based-index
    strings (``"1"``, ``"2"``, ...) to target-language display names. It is
    used as the topic universe for ``setindex_column`` lookups AND as an
    element-wise translator (via ``_apply_pre_transform_translate``) for raw
    best/worst/set cells when those cells hold index strings. ``scale.translate_after``
    is not supported on MaxDiff blocks and raises ``ValueError`` at read time.
    """

    type: Literal["maxdiff"] = "maxdiff"  # type: ignore[assignment]

    columns: ColSpec = DF(dict)
    best_columns: Union[str, List[str]] = Field(
        description="Columns (list or regex) holding the 'best' pick per question."
    )
    worst_columns: Union[str, List[str]] = Field(
        description="Columns (list or regex) holding the 'worst' pick per question."
    )
    set_columns: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Columns naming the items shown in each question. For input_format='choice_sets' a "
        "substitution template against best_columns; for 'resolved' an independent regex.",
    )
    setindex_column: Optional[Union[str, List[object]]] = None

    input_format: Literal["choice_sets", "resolved"] = Field(
        default="choice_sets",
        description=(
            "'choice_sets' = best/worst cells hold item indices and choice_sets/set lists define each question's "
            "options; 'resolved' = best/worst/set columns are already aligned per question."
        ),
    )

    # Flat: per-version per-question item lists (int indices). Dict: keyed by sibling label
    # (multi-sibling blocks), or - when setindex_column cells are strings - by design name,
    # with per-question item lists of indices or topic names.
    choice_sets: Optional[
        Union[
            List[List[List[int]]],
            Dict[str, Union[List[List[List[int]]], List[List[Union[int, str]]]]],
        ]
    ] = None

    @model_validator(mode="after")
    def _reject_translate_after(self, info: ValidationInfo) -> Self:
        if self.scale is not None and self.scale.translate_after:
            raise ValueError(
                f"MaxDiffBlock {self.name!r}: scale.translate_after is deprecated for "
                f"maxdiff; use scale.translate (pre-transform) instead."
            )
        return self

    def segments(self) -> List[Tuple[List[str], List[str], bool]]:
        """Return ordinal-ranking segments for this resolved MaxDiff block."""
        best = self.best_columns
        worst = self.worst_columns
        sets = self.set_columns
        if not (isinstance(best, list) and isinstance(worst, list) and isinstance(sets, list)):
            raise TypeError(
                f"MaxDiffBlock.segments() requires resolved lists; got best={best!r}, worst={worst!r}, sets={sets!r}"
            )
        if not best:
            return []
        return [([best[k]], [sets[k]], True) for k in range(len(best))] + [
            ([sets[k]], [worst[k]], True) for k in range(len(best))
        ]

    def default_model_spec(self) -> Optional[Dict[str, Any]]:
        """ordinal_ranking: one weak-order chain per question — best > shown set > worst."""
        best, worst, sets = self.best_columns, self.worst_columns, self.set_columns
        if not (isinstance(best, list) and isinstance(worst, list) and isinstance(sets, list)) or not best:
            return None
        return {"structure": [[[b], [s], [w]] for b, s, w in zip(best, sets, worst)]}

    def resolve_role_columns(self, df: "pd.DataFrame", raw_key: str) -> Dict[str, Any]:
        """Resolve best/worst/set roles to this sibling's columns, matching on the raw capture
        value (`raw_key`), not the display label. See specs/block-processing.md for the shapes."""
        updates: Dict[str, Any] = {}

        def _match_all(patt: "re.Pattern[str]") -> list[str]:
            return [c for c in df.columns if patt.match(c)]

        def _match_labeled(patt: "re.Pattern[str]") -> list[str]:
            hits = ((c, patt.match(c)) for c in df.columns)
            return [c for c, m in hits if m is not None and (not raw_key or m.group(1) == raw_key)]

        def _sort_by_last_group(cols: list[str], patt: "re.Pattern[str]") -> list[str]:
            def _key(col: str) -> tuple[int, Any]:
                m = patt.match(col)
                assert m is not None  # cols came from matching this same pattern
                if not m.groups():
                    return (1, col)
                last = m.groups()[-1]
                return (0, int(last)) if str(last).lstrip("-").isdigit() else (0, last)

            return sorted(cols, key=_key)

        if self.input_format == "resolved":
            all_regex = (
                isinstance(self.best_columns, str)
                and isinstance(self.worst_columns, str)
                and isinstance(self.set_columns, str)
            )
            if all_regex:
                bp = re.compile(cast(str, self.best_columns))
                wp = re.compile(cast(str, self.worst_columns))
                sp = re.compile(cast(str, self.set_columns))
                by_key: Dict[str, List[Optional[str]]] = {}
                for c in df.columns:
                    if (m := bp.match(c)) is not None:
                        by_key.setdefault(m.group(1), [None, None, None])[0] = c
                    if (m := wp.match(c)) is not None:
                        by_key.setdefault(m.group(1), [None, None, None])[1] = c
                    if (m := sp.match(c)) is not None:
                        by_key.setdefault(m.group(1), [None, None, None])[2] = c
                missing = [(k, t) for k, t in by_key.items() if None in t]
                if missing:
                    raise ValueError(f"MaxDiff resolved: incomplete alignment: {missing}")
                keys = sorted(by_key, key=lambda s: int(s) if s.isdigit() else s)
                triples = [by_key[k] for k in keys]
                updates["best_columns"] = [t[0] for t in triples]
                updates["worst_columns"] = [t[1] for t in triples]
                updates["set_columns"] = [t[2] for t in triples]
                return updates
            if isinstance(self.best_columns, str):
                updates["best_columns"] = _match_all(re.compile(self.best_columns))
            if isinstance(self.worst_columns, str):
                updates["worst_columns"] = _match_all(re.compile(self.worst_columns))
            if isinstance(self.set_columns, str):
                updates["set_columns"] = _match_all(re.compile(self.set_columns))
            return updates

        # input_format == "choice_sets"
        if isinstance(self.best_columns, str):
            best_re = re.compile(self.best_columns)
            sib_best = _sort_by_last_group(_match_labeled(best_re), best_re)
            updates["best_columns"] = sib_best
            if isinstance(self.worst_columns, str):
                worst_re = re.compile(self.worst_columns)
                updates["worst_columns"] = _sort_by_last_group(_match_labeled(worst_re), worst_re)
            if isinstance(self.set_columns, str):
                # substitution template against best_re
                updates["set_columns"] = [best_re.sub(self.set_columns, c) for c in sib_best]
        return updates

    def _role_columns(self, df: "pd.DataFrame", roles: List[object]) -> List[str]:
        """Concrete df-columns for the given role specs (explicit lists or regexes), deduped."""
        out: list[str] = []
        for spec in roles:
            if isinstance(spec, list):
                out.extend(spec)
            elif isinstance(spec, str):
                out.extend(c for c in df.columns if re.match(spec, c))
        return list(dict.fromkeys(out))

    def source_columns(self, df: "pd.DataFrame") -> List[str]:
        """Best + worst, plus set columns when those are read from the data rather than
        built from the design table a `setindex_column` points into."""
        roles: List[object] = [self.best_columns, self.worst_columns]
        if self.setindex_column is None:
            roles.append(self.set_columns)
        return self._role_columns(df, roles)

    def translate_columns(self, df: "pd.DataFrame") -> List[str]:
        """Set cells hold list values in choice_sets mode, so only best/worst pre-translate."""
        roles: List[object] = [self.best_columns, self.worst_columns]
        if self.input_format == "resolved":
            roles.append(self.set_columns)
        return self._role_columns(df, roles)


class OneHotBlock(ColumnBlockMeta):
    """Block producing one column per choice from a multi-select question.
    Output cells are coded via `coding` (default No/Yes categorical, negative pole first)."""

    type: Literal["onehot"] = "onehot"  # type: ignore[assignment]

    columns: ColSpec = DF(dict)
    # Source columns: an explicit list or a regex.
    from_columns: Union[str, List[str]]  # type: ignore[assignment]

    input_format: Literal["leftpacked", "wide"] = Field(
        default="leftpacked",
        description=(
            "'leftpacked' = M_1..M_n columns hold chosen choice names packed left; "
            "'wide' = one column per choice already (0/1 dummies or mention markers). In wide mode "
            "scale.translate names the choices (capture group / column name -> choice), not cell values."
        ),
    )

    choices: Optional[List[str]] = Field(
        default=None,
        description="Explicit choice list (also fixes category order); if None, derived from "
        "scale.translate values (wide) or observed cell values (leftpacked).",
    )
    res_prefix: Optional[str] = None
    not_selected: List[str] = Field(
        default_factory=list,
        description="Cell values meaning 'offered but not picked'; they prove the question WAS asked.",
    )
    coding: Optional[List[str]] = Field(
        default=["No", "Yes"],
        min_length=2,
        max_length=2,
        description="[false_label, true_label] for output cells, stamped as ordered categories; "
        "null keeps raw booleans.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_scale_categories_to_coding(cls, data: object) -> object:
        """A coded onehot's category set IS the coding - default it so scale fields like
        likert/num_values validate without the author restating ["No","Yes"]."""
        if isinstance(data, dict):
            scale, coding = data.get("scale"), data.get("coding", ["No", "Yes"])
            if isinstance(scale, dict) and coding is not None and not scale.get("categories"):
                scale["categories"] = list(coding)
        return data

    def translate_columns(self, df: "pd.DataFrame") -> List[str]:
        """In wide mode scale.translate names the choices, not the cell values."""
        return [] if self.input_format == "wide" else self.source_columns(df)


def _cb_lst_to_dict(lst: Sequence[object] | dict[str, object]) -> dict[str, object]:
    """Transform list of block specs to dictionary format keyed by block name,
    defaulting missing ``type`` to ``"plain"`` so the discriminated union validates
    old-shape annotations without an explicit ``type`` field."""
    if isinstance(lst, dict):
        return {k: _default_block_type(v) for k, v in lst.items()}

    result: dict[str, object] = {}
    for block in lst:
        if isinstance(block, BaseModel):
            name_val = getattr(block, "name", None)
        elif isinstance(block, dict):
            name_val = block.get("name")
        else:
            raise TypeError("Block specification must be a dict or BaseModel instance.")
        if not isinstance(name_val, str):
            raise TypeError("Each block specification must contain a 'name' field of type str.")
        result[name_val] = _default_block_type(block)
    return result


# Block-level fields removed by the create-refactor. Because ``PBase`` uses
# ``extra="ignore"``, leaving these unguarded would silently drop the directive and
# mis-process the block; we reject them loudly so stale annotations fail fast.
_LEGACY_BLOCK_FIELDS = {
    "topics": "MaxDiff topic names now come from scale.translate (1-based index -> name).",
    "sets": "MaxDiff set columns are declared via set_columns / setindex_column.",
    "choice_mapping": "Folded into scale.translate.",
    "items": "Folded into scale.translate.",
    "row_labels": "Folded into scale.translate.",
    "translate_values": "TopK index->name translation now lives in scale.translate_after.",
    "translate_after": "Block-level translation lives on the scale: scale.translate_after.",
    "translate": "Block-level translation lives on the scale: scale.translate.",
    "groups": "Subgroup naming now uses subgroup_labels.",
    "na_vals": "Renamed to not_selected ('offered but not picked'); see also meta-level not_asked.",
}


def _default_block_type(block: object) -> object:
    """Ensure a block dict carries a ``type`` discriminator (default ``"plain"``).
    Passes Pydantic model instances through untouched. Raises on legacy schema shapes
    (the nested ``create`` field or removed block-level fields) so silently-lost
    TopK/MaxDiff processing becomes a loud, actionable failure."""
    if isinstance(block, BaseModel):
        return block
    if not isinstance(block, dict):
        raise TypeError("Block specification must be a dict or BaseModel instance.")
    if "create" in block:
        raise ValueError(
            f"Block {block.get('name')!r} uses the legacy nested 'create' field, which is no "
            "longer supported. Hoist create.type to the top level as 'type' and flatten the "
            "create fields onto the block (e.g. {'type': 'topk', 'name': ..., 'from_columns': ...}). "
            "See specs/block-processing.md."
        )
    legacy = [f for f in _LEGACY_BLOCK_FIELDS if f in block]
    if legacy:
        hints = "; ".join(f"'{f}': {_LEGACY_BLOCK_FIELDS[f]}" for f in legacy)
        raise ValueError(
            f"Block {block.get('name')!r} uses removed block field(s) {legacy}. {hints} See specs/block-processing.md."
        )
    if "type" not in block:
        return {"type": "plain", **block}
    return block


_BlockUnion = Annotated[
    Union[TopKBlock, MaxDiffBlock, OneHotBlock, ColumnBlockMeta],
    Field(discriminator="type"),
]
BlockSpec = Annotated[Dict[str, _BlockUnion], BeforeValidator(_cb_lst_to_dict)]


class FileDesc(BaseModel):
    """Descriptor for a single data file in a multi-file data source."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    file: str
    opts: Dict = DF(dict)
    code: Optional[str] = None  # Short code identifier for the file (e.g., 'F1', 'F2' or 'wave1', 'wave2')
    id_col: Optional[str] = None  # Column in this file that uniquely identifies rows - overrides DataMeta.id_col

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Callable[[BaseModel], dict[str, Any]], info: SerializationInfo) -> dict[str, Any]:
        """Strip None/default fields (``id_col``, ``code``, empty ``opts``) via the shared serializer.

        Extra ``extra="allow"`` fields (e.g. wave labels) are not model fields, so they survive.
        """
        from salk_toolkit.serialization import serialize_pbase

        return serialize_pbase(self, handler, info)


def _normalize_data_desc_input(meta: Any, read_opts_key: str = "read_opts") -> Any:  # noqa: ANN401
    """Dict payload only: fold ``file`` and (for DataMeta) ``read_opts`` into ``files``,
        coerce FileDesc, default ``code`` to F0, F1, ….

    Non-dicts pass through unchanged.
    """
    if isinstance(meta, dict):
        # Never mutate the caller's dict: a `mode="before"` validator must be pure,
        # otherwise it leaks FileDesc objects back into the source model_desc (breaking
        # later json.dumps of that dict, e.g. in package_model).
        meta = dict(meta)

        # If file is provided, convert to first entry in files list
        if meta.get("file") is not None:
            file_value = meta.get("file")
            read_opts = meta.get(read_opts_key, {}) if read_opts_key != "__no_read_opts__" else {}
            meta.pop("file", None)
            if read_opts_key != "__no_read_opts__":
                meta.pop("read_opts", None)  # read_opts is now in files
            meta["files"] = [FileDesc(file=file_value, opts=read_opts), *(meta.get("files") or [])]

        # Ensure all files have codes based on their index in the list
        if meta.get("files") is not None:
            files = list(meta["files"])
            normalized_files: list[FileDesc] = []
            for i, fd in enumerate(files):
                try:
                    file_desc = FileDesc.model_validate(fd)
                except (ValidationError, TypeError) as exc:
                    raise TypeError("Files entries must be FileDesc-compatible objects") from exc
                if file_desc.code is None:
                    file_desc = file_desc.model_copy(update={"code": f"F{i}"})  # F0, F1, F2, ...
                normalized_files.append(file_desc)
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

    # Raw cell values meaning the question was NOT ASKED of this respondent (mode/filter skips,
    # 'Nicht erhoben', ...) - nulled in every block before translation, and the row counts as
    # unasked so blocks emit NA rather than a fabricated answer. NOT for substantive non-responses
    # like "Don't know" (keep those as categories flagged via nonresponse) and not for
    # "offered but not picked" (that is a block's not_selected). Blocks override ([] opts out).
    not_asked: Optional[List[str]] = None

    weight_col: Optional[str] = None  # Column to use for weighting - overriden by model to population weight column
    id_col: Optional[str] = None  # Raw column uniquely identifying rows within each file (e.g. respondent id)

    # List of data points that should be excluded in analyses
    excluded: List[Tuple[str, str]] = []  # (row_id, reason) - row_id is the stable string id, not a position
    total_size: Optional[float] = None  # Optional total population size override
    draws_data: Dict[str, Tuple[str, int]] = DF(dict)  # Precomputed draws info keyed by column

    @field_validator("excluded", mode="before")
    @classmethod
    def _reject_positional_excluded(cls, v: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Fail loudly on the legacy positional-int exclusion format.

        ``excluded`` is now ``(row_id, reason)`` keyed on the stable string row id, not an
        absolute position into the concatenated frame. Legacy ``[[int, reason], ...]`` metas
        must be migrated - see specs/2026-07-16-#60-stable-row-index.md.
        """
        if isinstance(v, list):
            for entry in v:
                # bool is an int subclass but never a valid position; np.integer is not an int subclass.
                first = entry[0] if isinstance(entry, (list, tuple)) and entry else None
                if isinstance(first, (int, np.integer)) and not isinstance(first, bool):
                    raise ValueError(
                        f"excluded entry {entry!r} uses a positional integer row index. "
                        "Exclusions are now keyed on the stable string row_id (e.g. 'F0::42' "
                        "or 'F0::<respondent_id>'). Migrate the meta to the new format - see "
                        "specs/2026-07-16-#60-stable-row-index.md."
                    )
        return v

    @model_validator(mode="before")
    @classmethod
    def normalize_files(cls, meta: Any) -> Any:  # noqa: ANN401  # pydantic validators require Any
        """Expand ``file``/``read_opts`` shorthands and coerce ``files`` to :class:`FileDesc`."""
        return _normalize_data_desc_input(meta, read_opts_key="read_opts")

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


def hard_validate(m: Mapping[str, JSONValue] | DataMeta) -> None:
    """Validate a DataMeta object with strict checking, raising errors on failure.

    Uses a strict model (extra='forbid') to ensure no extra fields are allowed.

    Args:
        m: Dictionary or DataMeta object to validate.

    Raises:
        ValidationError: If validation fails (including extra fields).
    """
    StrictDataMeta = _create_strict_model_class(DataMeta)
    payload = m.model_dump(mode="python") if isinstance(m, DataMeta) else dict(m)
    StrictDataMeta.model_validate(payload)


T = TypeVar("T", bound=BaseModel)


def _strictify_type(ann: object) -> object:  # noqa: ANN401  # annotation types are inherently dynamic
    """Recursively replace PBase subclasses with their strict twins inside a type annotation."""
    if isinstance(ann, type) and issubclass(ann, PBase):
        return _strict_model_class_cached(ann)

    origin = get_origin(ann)
    if origin is None:
        return ann

    args = get_args(ann)
    if not args:
        return ann

    new_args = tuple(_strictify_type(a) for a in args)
    if new_args == args:
        return ann

    if origin is Annotated:
        # Annotated[type, *metadata] — keep metadata unchanged, replace the base type only
        return Annotated.__class_getitem__((new_args[0],) + new_args[1:])

    # Generic types: List[X], Dict[K,V], Optional[X], Union[X,Y], Tuple[X,...], etc.
    if len(new_args) == 1:
        return origin[new_args[0]]
    return origin[new_args]


def _create_strict_model_class(base_model: type[BaseModel]) -> type[BaseModel]:
    """Create a strict version of a model class with extra='forbid' for validation warnings."""
    return _strict_model_class_cached(base_model)


@lru_cache(maxsize=None)
def _strict_model_class_cached(base_model: type[BaseModel]) -> type[BaseModel]:
    """Recursively build a strict twin of base_model where every nested PBase field is also strict.

    Creates a parallel strict hierarchy so that soft_validate's warning pass catches extra fields
    at all nesting levels, not just the top level.
    """
    if not issubclass(base_model, PBase):
        return base_model
    if cast(dict[str, Any], base_model.model_config).get("extra") == "forbid":
        return base_model

    # Collect field annotations that contain PBase subclasses and need strict twins.
    try:
        hints = get_type_hints(base_model, include_extras=True)
    except Exception:
        hints = {}

    new_annotations: dict[str, Any] = {}
    for fname in base_model.model_fields:
        ann = hints.get(fname)
        if ann is None:
            continue
        strict_ann = _strictify_type(ann)
        if strict_ann is not ann:
            new_annotations[fname] = strict_ann

    namespace: dict[str, Any] = {
        "model_config": ConfigDict(extra="forbid", arbitrary_types_allowed=True),
    }
    if new_annotations:
        namespace["__annotations__"] = new_annotations
        # Preserve defaults so overridden fields don't become required in the strict twin.
        for fname in new_annotations:
            fi = base_model.model_fields.get(fname)
            if fi is None or fi.is_required():
                continue
            if fi.default_factory is not None:
                namespace[fname] = Field(default_factory=fi.default_factory)
            else:
                namespace[fname] = fi.default

    strict_class = type(f"Strict{base_model.__name__}", (base_model,), namespace)
    strict_class.model_rebuild(force=True)  # type: ignore[union-attr]
    return strict_class


def soft_validate(
    m: Mapping[str, JSONValue] | BaseModel,
    model: type[T],
    warnings: bool = False,
    *,
    context: Mapping[str, JSONValue] | None = None,
) -> T:
    """Validate dict/model against a pydantic model, printing warnings, then returning validated object.
    When warnings=True, validates against a recursively strict twin first (extra='forbid' at all levels)
    to surface unknown keys as printed warnings, then validates with the normal model (extra='ignore')
    which allows extra fields and runs all validators so processing can continue.

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
        m_dict = dict(m)

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

    # Now validate with the normal model, which runs all validators and forbids extra fields.
    soft_context = dict(context) if context is not None else {}
    soft_context["validation_mode"] = "soft"
    inst = cast(T, model.model_validate(m_dict, strict=False, context=soft_context))
    return inst


class ParquetMeta(BaseModel):
    """Metadata bundle stored inside parquet files (data + miscellaneous extras)."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    data: DataMeta
    stk_commit: Optional[str] = None  # salk_toolkit git commit that wrote the file
    sip_commit: Optional[str] = None  # salk_internal_package git commit (set by run_stack)


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
        """Expand ``file`` shorthand and coerce ``files`` to :class:`FileDesc` (no top-level ``read_opts``)."""
        return _normalize_data_desc_input(meta, read_opts_key="__no_read_opts__")


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
ConvertResOption = Literal["continuous", "categorical"]

# The transform registries are extendable at runtime, so cont_transform is validated
# against the live ones rather than a frozen Literal - see `_valid_cont_transform`.
ContTransformOption = str
AggFnOption = Literal["mean", "sum", "posneg_mean", "median", "min", "max"]


def _valid_cont_transform(value: str) -> str:
    """Accept any transform the pipeline can dispatch, including ones a dashboard registered after import."""
    from salk_toolkit.pp.transforms import (
        TRANSFORM_FAMILIES,
        _ordered_topk,
        _threshold_cutoff,
        known_cont_transforms,
    )

    known = known_cont_transforms()
    if value in known:
        return value
    if _threshold_cutoff(value) is not None or _ordered_topk(value) is not None:  # raise on malformed parameters
        return value
    raise ValueError(
        f"unknown cont_transform {value!r}; registered: {', '.join(known)}; families: {TRANSFORM_FAMILIES}"
    )


# --------------------------------------------------------
#          PLOT DESCRIPTION
# --------------------------------------------------------


class StatSpec(PBase):
    """One named statistic: a row-level polars expression plus its weighted aggregation."""

    name: str  # Output column name
    expr: str  # Row-level polars expression, evaluated with {"pl": pl}
    agg_fn: Literal["mean", "sum"] = "mean"


class PlotDescriptor(PBase):
    """Descriptor for plot pipeline requests (``pp_desc``)."""

    # Main parameters
    plot: str  # Registered plot type (see `salk_toolkit.plots`)
    res_col: str  # Response column or question block name to visualise
    # Facet dimensions applied to the plot ("factor_cols" accepted as a legacy key)
    facet_dims: List[str] = Field(default=[], validation_alias=AliasChoices("facet_dims", "factor_cols"))
    filter: FilterSpec = {}  # Column filters applied before aggregation

    # Plotting choices
    convert_res: Optional[ConvertResOption] = None  # 'continuous' to number an ordinal, 'categorical' to bin a number
    cont_transform: Optional[ContTransformOption] = None  # Continuous transform to apply before aggregation

    agg_fn: Optional[AggFnOption] = None  # Aggregation override for summary statistics
    # Named row-level expressions aggregated in one group_by, so cells over different
    # row sets ride one scan. Data-only: a column per statistic, no single value column.
    stats: Optional[List[StatSpec]] = None
    # True = the declared weight_col (required if declared); False = unweighted; a string is a column
    # name, or a polars expression if it references pl. Anything but True recomputes total_size.
    weights: Union[bool, str] = True
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

    @field_validator("cont_transform")
    @classmethod
    def _check_cont_transform(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _valid_cont_transform(value)

    @model_validator(mode="after")
    def _check_stats(self) -> "PlotDescriptor":
        """`stats` replaces the single-statistic path rather than combining with it."""
        if not self.stats:
            return self
        clash = {k: v for k, v in (("agg_fn", self.agg_fn), ("cont_transform", self.cont_transform)) if v}
        if clash:
            drop = ", ".join(f"{k}={v!r}" for k, v in clash.items())
            raise ValueError(f"stats cannot combine with {drop}: each stat carries its own agg_fn and expression")
        names = [s.name for s in self.stats]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"stat name(s) {dupes} appear more than once; each names an output column")
        return self
