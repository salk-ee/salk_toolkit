"""Builds up to a function that validates metadata annotations"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_validation.ipynb.

# %% auto 0
__all__ = ['FileDesc', 'DataMeta', 'hard_validate', 'soft_validate']

# %% ../nbs/06_validation.ipynb 3
import numpy as np
from typing import *
from pydantic import BaseModel, ConfigDict, model_validator, BeforeValidator
from salk_toolkit.utils import replace_constants

# %% ../nbs/06_validation.ipynb 5
# Define a new base that is more strict towards unknown inputs
class PBase(BaseModel):
    model_config = ConfigDict(extra='forbid', protected_namespaces=(),arbitrary_types_allowed=True)

# %% ../nbs/06_validation.ipynb 6
class ColumnMeta(PBase):

    # Type specification
    continuous: bool = False # For real numbers
    datetime: bool = False # For datetimes
    categories: Optional[Union[List,Literal['infer']]] = None # For categoricals: List of categories or 'infer'
    ordered: bool = False # If categorical data is ordered

    # Transformations
    translate: Optional[Dict] = None # Translate dict applied to categories
    transform: Optional[str] = None # Transform function in python code, applied after translate
    translate_after: Optional[Dict] = None # Same as translate, but applied after transform

    # Model extras
    modifiers: Optional[List[str]] = None # List of columns that are meant to modify the responses on this col -> private_inputs
    nonresponses: Optional[List] = None # List of categories that are non-responses (Like "Don't know") -> unordered_categories 

    # Plot pipeline extras
    label: Optional[str] = None # Longer description of the column for tooltips
    groups: Optional[Dict[str,List[str]]] = None # Dict of lists of category values defining groups for easier filtering
    colors: Optional[Dict[str,str]] = None # Dict matching colors to categories
    num_values: Optional[List[Union[float,None]]] = None # For categoricals - how to convert the categories to numbers
    likert: bool = False # For ordered categoricals - if they are likert-type (i.e. symmetric around center)
    topo_feature: Optional[List[str]] = None # Link to a geojson/topojson [url,type,col_name inside geodata]
    electoral_system: Optional[Dict] = None # Information about electoral system (TODO: spec it out)
    mandates: Optional[Dict] = None # Mandate count mapping for the electoral system

    @model_validator(mode='after')
    def check_categorical(self) -> Self:
        if self.categories is None:
            #if not self.continuous and not self.datetime:
            #    raise ValueError('Column type undefined: need either categories, continuous or datetime')
            for f in ['ordered','groups','colors','num_values','likert','topo_feature']:
                if getattr(self,f):
                    raise ValueError(f'Field {f} only makes sense for categorical columns {getattr(self,f)}')
        else: # Is categorical
            if not self.ordered:
                for f in ['likert']: # ['num_values'] can be situationally useful in non-ordered settings
                    if getattr(self,f):
                        raise ValueError(f'Field {f} only makes sense for ordered categorical columns')
        return self


# This is for the block-level 'scale' group - basically same as ColumnMeta but with a few extras
class BlockScaleMeta(ColumnMeta):

    # Only useful in 'scale' block
    col_prefix: Optional[str] = None # If column name should have the prefix added. Usually used in scale block
    question_colors: Optional[Dict[str,str]] = None # Dict mapping columns to different colors


# %% ../nbs/06_validation.ipynb 7
# Transform the column tuple to (new name, old name, meta) format
def cspec(tpl):
    if type(tpl)==list:
        cn = tpl[0] # column name
        sn = tpl[1] if len(tpl)>1 and type(tpl[1])==str else cn # source column
        o_cd = tpl[2] if len(tpl)==3 else tpl[1] if len(tpl)==2 and type(tpl[1])==dict else {} # metadata
    else:
        cn = sn = tpl
        o_cd = {}
    return [cn,sn,o_cd]

# Transform list to dict for better error readability
def cs_lst_to_dict(lst):
    return { cn: [ocn,meta] for cn,ocn,meta in map(cspec,lst) }

ColSpec = Annotated[Dict[str,Tuple[str,ColumnMeta]],BeforeValidator(cs_lst_to_dict)]

# %% ../nbs/06_validation.ipynb 8
class ColumnBlockMeta(PBase):
    name: str # Name of the block
    scale: Optional[BlockScaleMeta] = None # Shared column meta for all columns inside the block
    
    # List of columns, potentially with their ColummnMetas
    columns: ColSpec

    subgroup_transform: Optional[str] = None # A block-level transform performed after column level transformations

    # Block level flags
    generated: bool = False # This block is for data that is generated, i.e. not initially in the file. 
    hidden: bool = False # Use this to hide the block in explorer.py
    virtual: bool = False # This block is for the virtual pass (i.e. works on already sampled data)

# %% ../nbs/06_validation.ipynb 9
# Again, convert list to dict for easier debugging in case errors get thrown
def cb_lst_to_dict(lst): return { c['name']:c for c in lst }
BlockSpec = Annotated[Dict[str,ColumnBlockMeta],BeforeValidator(cb_lst_to_dict)]

# %% ../nbs/06_validation.ipynb 10
class FileDesc(PBase):
    file: str
    opts: Optional[Dict] = None

class DataMeta(PBase):

    # Single input file
    file: Optional[str] = None # Name of the file, with relative path from this json file
    read_opts: Optional[Dict] = None # Additional options to pass to reading function

    # Multiple files
    files: Optional[List[FileDesc]] = None

     # Main meat of data annotations
    structure: BlockSpec

    # A set of values that can be referenced in the file below
    constants: Optional[Dict] = None

    # Different global processing steps
    preprocessing: Optional[str] = None # Performed on raw data
    postprocessing: Optional[str] = None # Performed after columns and blocks have been processed
    virtual_preprocessing: Optional[str] = None # Same as preprocessing, but only in virtual step
    virtual_postprocessing: Optional[str] = None # Same as postprocessing, but only in virtual step

    # List of data points that should be excluded in alyses
    excluded: List[Tuple[int,str]] = [] # Index of row + str  reason for exclusion

    @model_validator(mode='before')
    @classmethod
    def replace_constants(cls, meta: Any) -> Any:
        return replace_constants(meta)

    @model_validator(mode='after')
    def check_file(self) -> Self:
        if self.file is None and self.files is None:
            raise ValueError("One of 'file' or 'files' has to be provided")


# %% ../nbs/06_validation.ipynb 11
def hard_validate(m):
    DataMeta.validate(m)

def soft_validate(m):
    try:
        DataMeta.validate(m)
    except ValueError as e:
        print(e)
