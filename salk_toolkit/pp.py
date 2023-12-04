# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_pp.ipynb.

# %% auto 0
__all__ = ['get_filtered_data', 'create_plot', 'e2e_plot', 'test_new_plot']

# %% ../nbs/02_pp.ipynb 3
import json, os, inspect
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

import altair as alt

from salk_toolkit.plots import stk_plot, stk_deregister, matching_plots, get_plot_fn, get_plot_meta
from salk_toolkit.utils import *
from salk_toolkit.io import load_parquet_with_metadata, extract_column_meta, group_columns_dict, list_aliases, read_annotated_data, read_json

# %% ../nbs/02_pp.ipynb 6
# Augment each draw with bootstrap data from across whole population to make sure there are at least <threshold> samples
def augment_draws(data, factors=None, n_draws=None, threshold=50):
    if n_draws == None: n_draws = data.draw.max()+1
    
    if factors: # Run recursively on each factor separately and concatenate results
        if data[ ['draw']+factors ].value_counts().min() >= threshold: return data # This takes care of large datasets fast
        return data.groupby(factors).apply(augment_draws,n_draws=n_draws,threshold=threshold).reset_index(drop=True) # Slow-ish, but only needed on small data now
    
    # Get count of values for each draw
    draw_counts = data['draw'].value_counts() # Get value counts of existing draws
    if len(draw_counts)<n_draws: # Fill in completely missing draws
        draw_counts = (draw_counts + pd.Series(0,index=range(n_draws))).fillna(0).astype(int)
        
    # If no new draws needed, just return original
    if draw_counts.min()>=threshold: return data
    
    # Generate an index for new draws
    new_draws = [ d for d,c in draw_counts[draw_counts<threshold].items() for _ in range(threshold-c) ]

    # Generate new draws
    new_rows = data.iloc[np.random.choice(len(data),len(new_draws)),:].copy()
    new_rows['draw'] = new_draws
    
    return pd.concat([data, new_rows])

# %% ../nbs/02_pp.ipynb 7
# Get all data required for a given graph
# Only return columns and rows that are needed
# This is self-contained so it can be moved to polars later
def get_filtered_data(full_df, data_meta, columns=None, **kwargs):
    
    # Figure out which columns we actually need
    meta_cols = ['draw', 'weight']
    cols = [ kwargs['res_col'] ]  + vod(kwargs,'factor_cols',[]) + list(vod(kwargs,'filter',{}).keys()) + [ c for c in meta_cols if c in full_df.columns ]
    
    # If any aliases are used, cconvert them to column names according to the data_meta
    gc_dict = group_columns_dict(data_meta)
    cols = [ c for c in np.unique(list_aliases(cols,gc_dict)) if c in full_df.columns ]
    
    #print("C",cols)
    
    df = full_df[cols]
    
    # Filter using demographics dict. This is very clever but hard to read. See:
    filter_dict = vod(kwargs,'filter',{})
    inds = np.full(len(df),True)
    for k, v in filter_dict.items():
        if isinstance(v,tuple): # Tuples specify a range
            inds = (((df[k]>=v[0]) & (df[k]<=v[1])) | df[k].isna()) & inds
        elif isinstance(v,list): # List indicates a set of values
            inds = df[k].isin(v) & inds
        else: # Just filter on single value
            inds = (df[k]==v) & inds
        #if not inds.any():
        #    print(f"None left after {k}:{v}")
        #    break
    filtered_df = df[inds]
    
    # If res_col is a group of questions
    # This might move to wrangle but currently easier to do here as we have gc_dict handy
    if kwargs['res_col'] in gc_dict:
        value_vars = [ c for c in gc_dict[kwargs['res_col']] if c in cols ]
        id_vars = [ c for c in cols if c not in value_vars ]
        filtered_df = filtered_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='question', value_name=kwargs['res_col'])
        filtered_df['question'] = pd.Categorical(filtered_df['question'],gc_dict[kwargs['res_col']])
    
    return filtered_df

# %% ../nbs/02_pp.ipynb 9
# Groupby if needed - this simplifies the wrangle considerably :)
def gb_in(df, gb_cols):
    return df.groupby(gb_cols) if len(gb_cols)>0 else df

def discretize_continuous(col, col_meta={}):
    # NB! qcut might be a better default - see where testing leads us
    cut = pd.cut(col, bins = vod(col_meta,'bins',5), labels = vod(col_meta,'bin_labels',None) )
    cut = pd.Categorical(cut.astype(str), map(str,cut.dtype.categories), True) # Convert from intervals to strings for it to play nice with altair
    return cut

# Helper function that handles reformating data for create_plot
def wrangle_data(raw_df, plot_meta, col_meta, res_col, factor_cols ,**kwargs):
    
    draws, continuous, data_format = (vod(plot_meta, n, False) for n in ['draws','continuous','data_format'])
    
    gb_dims = (['draw'] if draws else []) + (factor_cols if factor_cols else []) + (['question'] if 'question' in raw_df.columns else [])
    
    if 'weight' not in raw_df.columns and not continuous: raw_df['weight'] = 1
    
    if draws and 'draw' in raw_df.columns: # Draw present in data
        raw_df = augment_draws(raw_df,gb_dims[1:],threshold=50) # Augment draws so we always have 50 data points in each
        
    rv = { 'value_col': 'value' }
    
    if data_format=='raw':
        rv['value_col'] = res_col
        if vod(plot_meta,'sample'):
            rv['data'] = gb_in(raw_df[gb_dims+[res_col]],gb_dims).sample(plot_meta['sample'],replace=True)
        else: rv['data'] = raw_df[gb_dims+[res_col]]

    elif False and data_format=='table': # TODO: Untested. Fix when first needed
        ddf = pd.get_dummies(raw_df[res_col])
        res_cols = list(ddf.columns)
        ddf.loc[:,gb_dims] = raw_df[gb_dims]
        rv['data'] = gb_in(ddf,gb_dims)[res_cols].mean().reset_index()
        
    elif data_format=='longform':
        if continuous:
            rv['data'] = gb_in(raw_df,gb_dims)[res_col].mean().dropna().reset_index() 
            rv['value_col'] = res_col
        else: # categorical
            rv['cat_col'] = res_col 
            rv['value_col'] = 'percent'
            rv['data'] = (raw_df.groupby(gb_dims+[res_col])['weight'].sum()/gb_in(raw_df,gb_dims)['weight'].sum()).rename(rv['value_col']).dropna().reset_index()
            
    else:
        raise Exception("Unknown data_format")
        
    # Ensure all rv columns other than value are categorical
    for c in rv['data'].columns:
        if rv['data'][c].dtype.name != 'categorical' and c!=rv['value_col']:
            if vod(vod(col_meta,c,{}),'continuous'):
                rv['data'].loc[:,c] = discretize_continuous(rv['data'][c],vod(col_meta,c,{}))
            else: # Just assume it's categorical by any other name
                rv['data'].loc[:,c] = pd.Categorical(rv['data'][c])
            
    return rv

# %% ../nbs/02_pp.ipynb 11
ordered_gradient = ["#c30d24", "#f3a583", "#94c6da", "#1770ab"]

def meta_color_scale(cmeta,argname='colors',column=None):
    scale = vod(cmeta,argname)
    if scale is None and column is not None and column.dtype.name=='category' and column.dtype.ordered:
        cats = column.dtype.categories
        scale = dict(zip(cats,gradient_to_discrete_color_scale(ordered_gradient, len(cats))))
    return to_alt_scale(scale)

# %% ../nbs/02_pp.ipynb 12
# Function that takes filtered raw data and plot information and outputs the plot
# Handles all of the data wrangling and parameter formatting
def create_plot(filtered_df, data_meta, plot, alt_properties={}, dry_run=False, width=200, **kwargs):
    plot_meta = get_plot_meta(plot)
    col_meta = extract_column_meta(data_meta)
    col_meta['question'] = vod(col_meta[kwargs['res_col']],'question_colors',{})
    
    params = wrangle_data(filtered_df, plot_meta, col_meta, **kwargs)
    data = params['data']
    
    if 'plot_args' in kwargs: params.update(kwargs['plot_args'])
    params['color_scale'] = meta_color_scale(col_meta[kwargs['res_col']],'colors',data[kwargs['res_col']])

    # Handle factor columns 
    factor_cols = vod(kwargs,'factor_cols',[])
    
    # If we have a question column not handled by the plot, add it to factors:
    if 'question' in data.columns and not vod(plot_meta,'question'):
        factor_cols = factor_cols + ['question']
    # If we don't have a question column but need it, just fill it with res_col name
    elif 'question' not in data.columns and vod(plot_meta,'question'):
        data.loc[:,'question'] = pd.Categorical([kwargs['res_col']]*len(params['data']))
        
    if vod(plot_meta,'question'):
        params['question_color_scale'] = meta_color_scale(col_meta[kwargs['res_col']],'question_colors')
    
    if factor_cols:
        # See if we should use it as an internal facet?
        plot_args = vod(kwargs,'plot_args',{})
        if vod(kwargs,'internal_facet'):
            params['factor_col'] = factor_cols[0]
            params['factor_color_scale'] = meta_color_scale(col_meta[factor_cols[0]],'colors',data[factor_cols[0]])
            factor_cols = factor_cols[1:] # Leave rest for external faceting
        
        # If we still have more than 1 factor - merge the rest
        if len(factor_cols)>1:
            factor_col = '+'.join(factor_cols)
            data.loc[:,factor_col] = data[factor_cols].agg(', '.join, axis=1)
            params['data'] = data
            n_facet_cols = len(data[factor_cols[-1]].dtype.categories)
            factor_cols = [factor_col]
        else:
            n_facet_cols = vod(plot_meta,'factor_columns',1)
    
    plot_fn = get_plot_fn(plot)
            
    # Trim down parameters list if needed
    aspec = inspect.getfullargspec(plot_fn)
    if aspec.varkw is None: params = { k:v for k,v in params.items() if k in aspec.args }
    
    # Create the plot using it's function
    if dry_run: return params

    dims = {'width': width//n_facet_cols if factor_cols else width}
    if 'aspect_ratio' in plot_meta:   dims['height'] = int(dims['width']/plot_meta['aspect_ratio'])        
    plot = plot_fn(**params).properties(**dims, **alt_properties)
    
    # Handle rest of factors via altair facet
    if factor_cols:
        n_facet_cols = vod(plot_args,'n_facet_cols',n_facet_cols) # Allow plot_args to override col nr
        plot = plot.facet(f'{factor_cols[0]}:O',columns=n_facet_cols)
    
    return plot

# %% ../nbs/02_pp.ipynb 15
# A convenience function to draw a plot straight from a dataset
def e2e_plot(pp_desc, data_file=None, full_df=None, data_meta=None, width=800, check_match=True):
    if data_file is None and full_df is None:
        raise Exception('Data must be provided either as data_file or full_df')
    if data_file is None and data_meta is None:
        raise Exception('If data provided as full_df then data_meta must also be given')
        
    if full_df is None: full_df, data_meta = read_annotated_data(data_file)
    
    matches = matching_plots(pp_desc, full_df, data_meta, details=True)
    
    if pp_desc['plot'] not in matches: 
        raise Exception(f"Plot not registered: {pp_desc['plot']}")
    
    fit, imp = matches[pp_desc['plot']]
    if  fit<0:
        raise Exception(f"Plot {pp_desc['plot']} not applicable in this situation because of flags {imp}")
        
    fdf = get_filtered_data(full_df, data_meta, **pp_desc)
    return create_plot(fdf,data_meta,width=width,**pp_desc)

# Another convenience function to simplify testing new plots
def test_new_plot(fn, pp_desc, *args, plot_meta={}, **kwargs):
    stk_plot(**{**plot_meta,'plot_name':'test'})(fn) # Register the plot under name 'test'
    pp_desc = {**pp_desc, 'plot': 'test'}
    res = e2e_plot(pp_desc,*args,**kwargs)
    stk_deregister('test') # And de-register it again
    return res
