# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_utils.ipynb.

# %% auto 0
__all__ = ['factorize_w_codes', 'batch', 'loc2iloc', 'replace_constants', 'index_encoder']

# %% ../nbs/10_utils.ipynb 3
import json, os
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

# %% ../nbs/10_utils.ipynb 4
# I'm surprised pandas does not have this function but I could not find it. 
def factorize_w_codes(s, codes):
    res = s.replace(dict(zip(codes,range(len(codes)))))
    if not s.isin(codes).all(): # Throw an exception if all values were not replaced
        vals = set(s) - set(codes)
        raise Exception(f'Codes for {s.name} do not match all values: {vals}')
    return res.to_numpy()

# %% ../nbs/10_utils.ipynb 5
# Simple batching of an iterable
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# %% ../nbs/10_utils.ipynb 6
# turn index values into order indices
def loc2iloc(index, vals):
    d = dict(zip(np.array(index),range(len(index))))
    return [ d[v] for v in vals ]

# %% ../nbs/10_utils.ipynb 7
# Allow 'constants' entries in the dict to provide replacement mappings
# This leads to much more readable jsons as repetitions can be avoided
def replace_constants(d, constants = {}):
    if type(d)==dict and 'constants' in d:
        constants = constants.copy() # Otherwise it would propagate back up through recursion - see test6 below
        constants.update(d['constants'])
        del d['constants']

    for k, v in (d.items() if type(d)==dict else enumerate(d)):
        if type(v)==str and v in constants:
            d[k] = constants[v]
        elif type(v)==dict or type(v)==list:
            replace_constants(v,constants)

# %% ../nbs/10_utils.ipynb 9
# JSON encoder needed to convert pandas indices into lists for serialization
def index_encoder(z):
    if isinstance(z, pd.Index):
        return list(z)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type {type_name} is not serializable")
