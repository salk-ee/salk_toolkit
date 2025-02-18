"""Utility functions with wider use potential"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_utils.ipynb.

# %% auto 0
__all__ = ['warn', 'default_color', 'factorize_w_codes', 'batch', 'loc2iloc', 'match_sum_round', 'min_diff', 'continify',
           'replace_cat_with_dummies', 'match_data', 'replace_constants', 'approx_str_match', 'index_encoder',
           'to_alt_scale', 'multicol_to_vals_cats', 'gradient_to_discrete_color_scale', 'color_ubound_luminosity',
           'is_datetime', 'rel_wave_times', 'stable_draws', 'deterministic_draws', 'clean_kwargs', 'censor_dict',
           'cut_nice_labels', 'cut_nice', 'rename_cats', 'str_replace', 'merge_series', 'aggregate_multiselect',
           'deaggregate_multiselect', 'gb_in', 'gb_in_apply', 'stk_defaultdict', 'cached_fn']

# %% ../nbs/10_utils.ipynb 3
import json, os, warnings, math, inspect
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import datetime as dt

import altair as alt
import matplotlib.colors as mpc
import colorsys
from copy import deepcopy
from hashlib import sha256

from typing import List, Tuple, Dict, Union, Optional
import Levenshtein

# %% ../nbs/10_utils.ipynb 4
pd.set_option('future.no_silent_downcasting', True)

# %% ../nbs/10_utils.ipynb 5
# convenience for warnings that gives a more useful stack frame (fn calling the warning, not warning fn itself)
warn = lambda msg,*args: warnings.warn(msg,*args,stacklevel=3)

# %% ../nbs/10_utils.ipynb 6
# I'm surprised pandas does not have this function but I could not find it. 
def factorize_w_codes(s, codes):
    res = s.astype('object').replace(dict(zip(codes,range(len(codes)))))
    if not s.dropna().isin(codes).all(): # Throw an exception if all values were not replaced
        vals = set(s) - set(codes)
        raise Exception(f'Codes for {s.name} do not match all values: {vals}')
    return res.fillna(-1).to_numpy(dtype='int')

# %% ../nbs/10_utils.ipynb 7
# Simple batching of an iterable
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# %% ../nbs/10_utils.ipynb 8
# turn index values into order indices
def loc2iloc(index, vals):
    d = dict(zip(np.array(index),range(len(index))))
    return [ d[v] for v in vals ]

# %% ../nbs/10_utils.ipynb 9
# Round in a way that preserves total sum
def match_sum_round(s):
    s = np.array(s)
    fs = np.floor(s)
    diff = round(s.sum()-fs.sum())
    residues = np.argsort(-(s%1))[:diff]
    fs[residues] = fs[residues]+1
    return fs.astype('int')

# %% ../nbs/10_utils.ipynb 11
# Find the minimum difference between two values in the array
def min_diff(arr):
    b = np.diff(np.sort(arr))
    if len(b)==0 or b.max()==0.0: return 0
    else: return b[b>0].min()

# Turn a discretized variable into a more smooth continuous one w a gaussian kernel
def continify(ar, bounded=False, delta=0.0):
    mi,ma = ar.min()+delta, ar.max()-delta
    noise = np.random.normal(0,0.5 * min_diff(ar),size=len(ar))
    res = ar + noise
    if bounded: # Reflect the noise on the boundaries
        res[res>ma] = ma - (res[res>ma] - ma)
        res[res<mi] = mi + (mi - res[res<mi])
    return res

# %% ../nbs/10_utils.ipynb 13
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# %% ../nbs/10_utils.ipynb 14
# Replace categorical with dummy values,
def replace_cat_with_dummies(df,c,cs):
    return pd.concat([df.drop(columns=[c]),pd.get_dummies(df[c])[cs[1:]].astype(float)],axis=1)

# Match data1 with data2 on columns cols as closely as possible
def match_data(data1,data2,cols=None):
    d1 = data1[cols].copy().dropna()
    d2 = data2[cols].copy().dropna()

    if len(d1)==0 or len(d2)==0: return [],[]

    ccols = [c for c in cols if d1[c].dtype.name=='category']
    for c in ccols:
        if d1[c].dtype.ordered: # replace categories with their index
            s1, s2 = set(d1[c].dtype.categories), set(d2[c].dtype.categories)
            if s1-s2 and s2-s1: # one-way imbalance is fine
                raise Exception(f"Ordered categorical columns differ in their categories on: {s1-s2} vs {s2-s1}")
            
            md = d1 if len(s2-s1)==0 else d2
            mdict = dict(zip(md[c].dtype.categories, np.linspace(0,2,len(md[c].dtype.categories))))
            d1[c] = d1[c].astype('object').replace(mdict)
            d2[c] = d2[c].astype('object').replace(mdict)
        else: # Use one-hot encoding instead
            cs = list(set(d1[c].unique()) | set(d2[c].unique()))
            d1[c], d2[c] = pd.Categorical(d1[c],cs), pd.Categorical(d2[c],cs)
            # Use all but the first category as otherwise mahalanobis fails because of full colinearity
            d1 = replace_cat_with_dummies(d1,c,cs)
            d2 = replace_cat_with_dummies(d2,c,cs)


    # Use pseudoinverse in case we have collinear columns
    cov = np.cov(np.vstack([d1.values, d2.values]).T.astype('float'))
    cov += np.eye(len(cov))*1e-5 # Add a small amount of noise to avoid singular matrix
    pVI = np.linalg.pinv(cov).T
    dmat = cdist(d1, d2, 'mahalanobis', VI=pVI)
    i1, i2 = linear_sum_assignment(dmat, maximize=False)
    ind1, ind2 = d1.index[i1], d2.index[i2]
    return ind1, ind2

# %% ../nbs/10_utils.ipynb 16
# Allow 'constants' entries in the dict to provide replacement mappings
# This leads to much more readable jsons as repetitions can be avoided
def replace_constants(d, constants = {}, inplace=False):
    if not inplace: d = deepcopy(d)
    if type(d)==dict and 'constants' in d:
        constants = constants.copy() # Otherwise it would propagate back up through recursion - see test6 below
        constants.update(d['constants'])
        del d['constants']

    for k, v in (d.items() if type(d)==dict else enumerate(d)):
        if type(v)==str and v in constants:
            d[k] = constants[v]
        elif type(v)==dict or type(v)==list:
            d[k] = replace_constants(v,constants, inplace=True)
            
    return d

# %% ../nbs/10_utils.ipynb 17
# Little function to do approximate string matching between two lists. Useful if things have multiple spellings. 
def approx_str_match(frm,to,dist_fn=None):
    if not isinstance(frm,list): frm = list(frm)
    if not isinstance(frm,list): to = list(to)
    if dist_fn is None: dist_fn = Levenshtein.distance 
    dmat = scipy.spatial.distance.cdist(np.array(frm)[:,None],np.array(to)[:,None],lambda x,y: dist_fn(x[0],y[0]))
    t1,t2 = scipy.optimize.linear_sum_assignment(dmat)
    return dict(zip([frm[i] for i in t1],[to[i] for i in t2]))

# %% ../nbs/10_utils.ipynb 20
# JSON encoder needed to convert pandas indices into lists for serialization
def index_encoder(z):
    if isinstance(z, pd.Index):
        return list(z)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type {type_name} is not serializable")

# %% ../nbs/10_utils.ipynb 21
default_color = 'lightgrey' # Something that stands out so it is easy to notice a missing color

# Helper function to turn a dictionary into an Altair scale (or None into alt.Undefined)
# Also: preserving order matters because scale order overrides sort argument
def to_alt_scale(scale, order=None):
    if scale is None: scale = alt.Undefined
    if isinstance(scale,dict):
        if order is None: order = scale.keys()
        #else: order = [ c for c in order if c in scale ]
        scale = alt.Scale(domain=list(order),range=[ (scale[c] if c in scale else default_color) for c in order ])
    return scale

# %% ../nbs/10_utils.ipynb 22
# Turn a question with multiple variants all of which are in distinct columns into a two columns - one with response, the other with which question variant was used

def multicol_to_vals_cats(df, cols=None, col_prefix=None, reverse_cols=[], reverse_suffixes=None, cat_order=None, vals_name='vals', cats_name='cats', inplace=False):
    if not inplace: df = df.copy()
    if cols is None: cols = [ c for c in df.columns if c.startswith(col_prefix)]
    
    if not reverse_cols and reverse_suffixes is not None:
        reverse_cols = list({ c for c in cols for rs in reverse_suffixes if c.endswith(rs)})
    
    if len(reverse_cols)>0:
        remap = dict(zip(cat_order,reversed(cat_order)))
        df.loc[:,reverse_cols] = df.loc[:,reverse_cols].astype('object').replace(remap)
    
    tdf = df[cols]
    cinds = np.argmax(tdf.notna(),axis=1)
    df.loc[:,vals_name] = np.array(tdf)[range(len(tdf)),cinds]
    df.loc[:,cats_name] = np.array(tdf.columns)[cinds]
    return df

# %% ../nbs/10_utils.ipynb 24
# Grad is a list of colors
def gradient_to_discrete_color_scale( grad, num_colors):
    cmap = mpc.LinearSegmentedColormap.from_list('grad',grad)
    return [mpc.to_hex(cmap(i)) for i in np.linspace(0, 1, num_colors)]

# %% ../nbs/10_utils.ipynb 26
def color_ubound_luminosity(color, l_value):
    rgb = mpc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    nrgb = colorsys.hls_to_rgb(h, min(l,l_value), s = s)
    return mpc.to_hex(nrgb)

# %% ../nbs/10_utils.ipynb 27
def is_datetime(col):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        return pd.api.types.is_datetime64_any_dtype(col) or (col.dtype.name in ['str','object'] and pd.to_datetime(col,errors='coerce').notna().any())

# %% ../nbs/10_utils.ipynb 28
# Convert a series of wave indices and a series of survey dates into a time series usable by our gp model
def rel_wave_times(ws, dts, dt0=None):
    df = pd.DataFrame({'wave':ws, 'dt': pd.to_datetime(dts)})
    adf = df.groupby('wave')['dt'].median()
    if dt0 is None: dt0 = adf.max() # use last wave date as the reference
    
    w_to_time = dict(((adf - dt0).dt.days/30).items())
    
    return pd.Series(df['wave'].replace(w_to_time),name='t')

# %% ../nbs/10_utils.ipynb 30
# Generate a random draws column that is deterministic in n, n_draws and uid
def stable_draws(n, n_draws, uid):
    # Initialize a random generator with a hash of uid
    bgen = np.random.SFC64(np.frombuffer(sha256(str(uid).encode("utf-8")).digest(), dtype='uint32'))
    gen = np.random.Generator(bgen)
    
    n_samples = int(math.ceil(n/n_draws))
    draws = np.tile(np.arange(n_draws),n_samples)[:n]
    return gen.permuted(draws)

# Use the stable_draws function to deterministicall assign shuffled draws to a df 
def deterministic_draws(df, n_draws, uid, n_total=None):
    if n_total is None: n_total = len(df)
    df.loc[:,'draw'] = pd.Series(stable_draws(n_total, n_draws, uid), index = np.arange(n_total))
    return df

# %% ../nbs/10_utils.ipynb 32
# Clean kwargs leaving only parameters fn can digest
def clean_kwargs(fn, kwargs):
    aspec = inspect.getfullargspec(fn)
    return { k:v for k,v in kwargs.items() if k in aspec.args } if aspec.varkw is None else kwargs

# %% ../nbs/10_utils.ipynb 33
# Simple one-liner to remove certain keys from a dict
def censor_dict(d,vs):
    return { k:v for k,v in d.items() if k not in vs }

# %% ../nbs/10_utils.ipynb 34
# Create nice labels for a cut
# Used by the cut_nice below as well as for a lazy polars version in pp
def cut_nice_labels(breaks, mi, ma, isint, format='', separator=' - '):
    
    # Extend breaks if needed
    lopen, ropen = False, False
    if ma > breaks[-1]:
        breaks.append(ma + 1)
        ropen = True
    if mi < breaks[0]:
        breaks.insert(0, mi)
        lopen = True
    
    if isint:
        breaks = list(map(int, breaks))
        format = ''  # No need for decimal places if all integers
    
    tuples = [(breaks[i], breaks[i + 1] - (1 if isint else 0)) for i in range(len(breaks) - 1)]
    labels = [f"{t[0]:{format}}{separator}{t[1]:{format}}" for t in tuples]
    
    if lopen: labels[0] = f"<{breaks[1]:{format}}"
    if ropen: labels[-1] = f"{breaks[-2]:{format}}+"

    return breaks,labels

# A nicer behaving wrapper around pd.cut
def cut_nice(s, breaks, format='', separator=' - '):
    s = np.array(s)
    mi, ma = s.min(), s.max()
    isint = np.issubdtype(s.dtype, np.integer) or (s % 1 == 0.0).all()
    breaks, labels = cut_nice_labels(breaks, mi, ma, isint, format, separator)
    return pd.cut(s, breaks, right=False, labels=labels, ordered=False)

# %% ../nbs/10_utils.ipynb 36
# Utility function to rename categories in pre/post processing steps as pandas made .replace unusable with categories
def rename_cats(df, col, cat_map):
    if df[col].dtype.name == 'category': 
        df[col] = df[col].cat.rename_categories(cat_map)
    else: df[col] = df[col].replace(cat_map)

# %% ../nbs/10_utils.ipynb 37
# Simplify doing multiple replace's on a column
def str_replace(s,d):
    s = s.astype('object')
    for k,v in d.items():
        s = s.str.replace(k,v)
    return s

# %% ../nbs/10_utils.ipynb 39
# Merge values from multiple columns, iteratively replacing values
# lst contains either a series or a tuple of (series, whitelist)
def merge_series(*lst):
    s = lst[0].astype('object').copy()
    for t in lst[1:]:
        if isinstance(t,tuple):
            ns, whitelist = t
            inds = ~ns.isna() & ns.isin(whitelist)
            s.loc[inds] = ns[inds]
        else: 
            ns = t
            s.loc[~ns.isna()] = ns[~ns.isna()]
    return s

# %% ../nbs/10_utils.ipynb 41
# Turn a list of selected/not seleced into a list of selected values in the same dataframe
def aggregate_multiselect(df, prefix, out_prefix, na_vals=[]):
    cols = [ c for c in df.columns if c.startswith(prefix) ]
    lst = list(map(lambda l: [ v for v in l if v is not None ],
         df[cols].astype('object').replace(dict(zip(na_vals,[None]*len(na_vals)))).values.tolist()))
    n_res = max(map(len,lst))
    df[[f'{out_prefix}{i+1}' for i in range(n_res)]] = pd.DataFrame(lst)

# %% ../nbs/10_utils.ipynb 42
# Take a list of values and create a one-hot matrix of them. Basically the inverse of previous
def deaggregate_multiselect(df, prefix, out_prefix=''):
    cols = [ c for c in df.columns if c.startswith(prefix) ]

    # Determine all categories
    ocols = set()
    for c in cols: ocols.update(df[c].dropna().unique())

    # Create a one-hot column for each
    for oc in ocols: df[out_prefix+oc] = (df[cols]==oc).any(axis=1)

# %% ../nbs/10_utils.ipynb 43
# Groupby if needed - this simplifies things quite often
def gb_in(df, gb_cols):
    return df.groupby(gb_cols,observed=False) if len(gb_cols)>0 else df

# Groupby apply if needed - similar to gb_in but for apply
def gb_in_apply(df, gb_cols, fn, cols=None, **kwargs):
    if cols is None: cols = list(df.columns)
    if len(gb_cols)==0: 
        res = fn(df[cols],**kwargs)
        if type(res)==pd.Series: res = pd.DataFrame(res).T
    else: res = df.groupby(gb_cols,observed=False)[cols].apply(fn,**kwargs)
    return res

# %% ../nbs/10_utils.ipynb 44
def stk_defaultdict(dv):
    if not isinstance(dv,dict): dv = {'default':dv}
    return defaultdict(lambda: dv['default'], dv)

# %% ../nbs/10_utils.ipynb 45
def cached_fn(fn):
    cache = {}
    def cf(x):
        if x not in cache:
            cache[x] = fn(x)
        return cache[x]
    return cf
