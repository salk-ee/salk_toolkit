# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_pp.ipynb.

# %% auto 0
__all__ = ['registry', 'registry_meta', 'stk_plot_defaults', 'n_a', 'priority_weights', 'cont_transform_options',
           'special_columns', 'internal_columns', 'get_cat_num_vals', 'stk_plot', 'stk_deregister', 'get_plot_fn',
           'get_plot_meta', 'get_all_plots', 'calculate_priority', 'matching_plots', 'get_filtered_data',
           'translate_df', 'create_plot', 'impute_factor_cols', 'e2e_plot', 'test_new_plot']

# %% ../nbs/02_pp.ipynb 3
import json, os
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import datetime as dt
import scipy.stats as sps

from typing import List, Tuple, Dict, Union, Optional

import altair as alt

from salk_toolkit.utils import *
from salk_toolkit.io import load_parquet_with_metadata, extract_column_meta, group_columns_dict, list_aliases, read_annotated_data, read_json

# %% ../nbs/02_pp.ipynb 6
# Augment each draw with bootstrap data from across whole population to make sure there are at least <threshold> samples
def augment_draws(data, factors=None, n_draws=None, threshold=50):
    if n_draws == None: n_draws = data.draw.max()+1
    
    if factors: # Run recursively on each factor separately and concatenate results
        if data[ ['draw']+factors ].value_counts().min() >= threshold: return data # This takes care of large datasets fast
        return data.groupby(factors,observed=False).apply(augment_draws,n_draws=n_draws,threshold=threshold).reset_index(drop=True) # Slow-ish, but only needed on small data now
    
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
# Get the numerical values to map categories to
def get_cat_num_vals(res_meta,pp_desc):
    nvals = res_meta.get('num_values',range(len(res_meta['categories'])))
    if 'num_values' in pp_desc: nvals = pp_desc['num_values'] 
    return nvals

# %% ../nbs/02_pp.ipynb 9
registry = {}
registry_meta = {}

# %% ../nbs/02_pp.ipynb 11
stk_plot_defaults = { 'data_format': 'longform' }

# Decorator for registering a plot type with metadata
def stk_plot(plot_name, **r_kwargs):
    
    def decorator(gfunc):
        # In theory, we could do transformations in wrapper
        # In practice, it would only obfuscate already complicated code
        #def wrapper(*args,**kwargs) :
        #    return gfunc(*args,**kwargs)

        # Register the function
        registry[plot_name] = gfunc
        registry_meta[plot_name] = { 'name': plot_name, **stk_plot_defaults, **r_kwargs }
        
        return gfunc
    
    return decorator

def stk_deregister(plot_name):
    del registry[plot_name]
    del registry_meta[plot_name]

def get_plot_fn(plot_name):
    return registry[plot_name]

def get_plot_meta(plot_name):
    return registry_meta[plot_name].copy()

def get_all_plots():
    return sorted(list(registry.keys()))

# %% ../nbs/02_pp.ipynb 12
# First is weight if not matching, second if match
# This is very much a placeholder right now
n_a = -1000000
priority_weights = {
    'draws': [n_a, 50],
    'nonnegative': [n_a, 50],
    'hidden': [n_a, 0],
    
    'ordered': [n_a, 100],
    'likert': [n_a, 200],
    'required_meta': [n_a, 500],
}

def calculate_priority(plot_meta, match):
    priority, reasons = plot_meta.get('priority',0), []

    facet_metas = match['facet_metas']
    if plot_meta.get('no_question_facet'):
        facet_metas = [ f for f in facet_metas if f['name'] not in ['question',match['res_col']]]

    # Plots with raw data assume numerical values so remove them as options
    if match['categorical'] and plot_meta.get('data_format')=='raw': return n_a, ['raw_data']

    if len(facet_metas)<plot_meta.get('n_facets',(0,0))[0]: 
        return n_a, ['n_facets'] # Not enough factors
    else: # Prioritize plots that have the right number of factors
        priority += 10*abs(len(facet_metas)-plot_meta.get('n_facets',(0,0))[1])

    for k in ['draws','nonnegative','hidden']:
        if plot_meta.get(k):
            val = priority_weights[k][1 if match.get(k) else 0]
            if val < 0: reasons.append(k)
            priority += val

    for i, d in enumerate(plot_meta.get('requires',[])):
        md = facet_metas[i]
        for k, v in d.items():
            if v!='pass': val = priority_weights[k][1 if md.get(k)==v else 0]
            else: val = priority_weights['required_meta'][1 if k in md else 0] # Use these weights for things plots require from metadata

            if k == 'ordered' and md.get('continuous'): val = priority_weights[k][1] # Continuous is turned into ordered categoricals for facets
            if val < 0: reasons.append(k)
            priority += val
                                     
    return priority, reasons


# Get a list of plot types matching required spec
def matching_plots(pp_desc, df, data_meta, details=False, list_hidden=False):
    col_meta = extract_column_meta(data_meta)
    
    rc = pp_desc['res_col']
    rcm = col_meta[rc]

    # Determine if values are non-negative
    cols = [c for c in rcm['columns'] if c in df.columns] if 'columns' in rcm else [rc]
    nonneg = ('categories' in rcm) or df[cols].min(axis=None)>=0
    if pp_desc.get('convert_res')=='continuous' and ('categories' in rcm):
        nonneg = min([ v for v in get_cat_num_vals(rcm,pp_desc) if v is not None ])>=0

    match = {
        'draws': ('draw' in df.columns),
        'nonnegative': nonneg,
        'hidden': list_hidden,

        'res_col': rc,
        'categorical': ('categories' in rcm) and pp_desc.get('convert_res')!='continuous',
        'facet_metas': [ {'name':cn, **col_meta[cn]} for cn in pp_desc['factor_cols']]
    }
    
    res = [ ( pn, *calculate_priority(get_plot_meta(pn),match)) for pn in registry.keys() ]
    
    if details: return { n: (p, i) for (n, p, i) in res } # Return dict with priorities and failure reasons
    else: return [ n for (n,p,i) in sorted(res,key=lambda t: t[1], reverse=True) if p >= 0 ] # Return list of possibilities in decreasing order of fit

# %% ../nbs/02_pp.ipynb 17
cont_transform_options = ['center','zscore','softmax','softmax-ratio']

# %% ../nbs/02_pp.ipynb 18
# Get the categories that are in use
def get_cats(col, cats=None):
    if cats is None or len(set(col.dtype.categories)-set(cats))>0: cats = col.dtype.categories
    return [ c for c in cats if c in col.unique() ]

def transform_cont(data, transform):
    if not transform: return data, '.1f'
    elif transform == 'center': return data - data.mean(skipna=True,axis=0), '.1f'
    elif transform == 'zscore': return sps.zscore(data,nan_policy='omit',axis=0), '.2f'
    elif transform == 'softmax': return np.exp(data)/(np.exp(np.array(data)).sum(axis=1)[:,None]), '.1%'
    elif transform == 'softmax-ratio': return data.shape[1]*np.exp(data)/(np.exp(np.array(data)).sum(axis=1)[:,None]), '.1f' 
    else: raise Exception(f"Unknown transform '{transform}'")

# %% ../nbs/02_pp.ipynb 19
special_columns = ['id','weight','draw','training_subsample', '__index_level_0__']

# Get all data required for a given graph
# Only return columns and rows that are needed
# This can handle either a pandas DataFrame or a polars LazyDataFrame (to allow for loading only needed data)
# NB: LazyDataFrame support is not maintained and is likely to be at least somewhat broken. Keeping code in in case it ever becomes relevant
def get_filtered_data(full_df, data_meta, pp_desc, columns=[]):

    plot_meta = get_plot_meta(pp_desc['plot'])
    
    # Figure out which columns we actually need
    meta_cols = ['weight', 'training_subsample', '__index_level_0__'] + (['draw'] if plot_meta.get('draws') else []) + columns
    cols = [ pp_desc['res_col'] ]  + pp_desc.get('factor_cols',[]) + list(pp_desc.get('filter',{}).keys())
    cols += [ c for c in meta_cols if c in full_df.columns and c not in cols ]

    # Remove draws_data if calcualted_draws is disabled       
    if not pp_desc.get('calculated_draws',True):
        data_meta = data_meta.copy()
        del data_meta['draws_data']
    
    # If any aliases are used, cconvert them to column names according to the data_meta
    gc_dict = group_columns_dict(data_meta)
    c_meta = extract_column_meta(data_meta)
    
    # Dict to remap (short) category names to longer descriptions in tooltips
    label_dict = {}
    
    cols = [ c for c in np.unique(list_aliases(cols,gc_dict)) if c in full_df.columns ]
    
    #print("C",cols)
    
    lazy = isinstance(full_df,pl.LazyFrame)
    if lazy: pl.enable_string_cache() # Needed for categories to be comparable to strings
    
    df = full_df.select(cols) if lazy else full_df[cols]

    # Replace draw with the draws used in modelling. Groups are handled separately below
    # NB! Has to happen before filtering or the draws are computed for wrong size df
    # Workaround would be to compute for a dummy df and merge on indices later
    if 'draw' in df.columns and pp_desc['res_col'] in data_meta.get('draws_data',{}):
        uid, ndraws = data_meta['draws_data'][pp_desc['res_col']]
        df = deterministic_draws(df, ndraws, uid, n_total = data_meta['total_size'] )
    n_points = len(df) # This is used later for draws
    
    # Filter using demographics dict. This is very clever but hard to read. See:
    filter_dict = pp_desc.get('filter',{})
    inds = True if lazy else np.full(len(df),True) 
    for k, v in filter_dict.items():
        
        # Range filters have form [None,start,end]
        is_range = isinstance(v,list) and v[0] is None and len(v)==3

        # Handle continuous variables separately
        if  is_range and (not isinstance(v[1],str) or c_meta[k].get('continuous') or c_meta[k].get('datetime')): # Only special case where we actually need a range
            if lazy: inds = (((pl.col(k)>=v[1]) & (pl.col(k)<=v[2]))) & inds
            else: inds = (((df[k]>=v[1]) & (df[k]<=v[2]))) & inds
            continue # NB! this approach does not work for ordered categoricals with polars LazyDataFrame, hence handling that separately below
        
        # Filter by list of values:
        if is_range: # Range of values over ordered categorical
            if c_meta[k].get('categories','infer')=='infer': raise Exception(f'Ordering unknown for column {k}')
            cats = list(c_meta[k]['categories'])
            if set(v[1:]) & set(cats) != set(v[1:]): raise Exception(f'Column {k} values {v} not found in {cats}')
            bi, ei = cats.index(v[1]), cats.index(v[2])
            flst = cats[bi:ei+1] # 
        elif isinstance(v,list): flst = v # List indicates a set of values
        elif 'groups' in c_meta[k] and v in c_meta[k]['groups']:
            flst = c_meta[k]['groups'][v]
        else: flst = [v] # Just filter on single value    
            
        inds =  (pl.col(k).is_in(flst) & ~pl.col(k).is_null() if lazy else df[k].isin(flst) & ~df[k].isna()) & inds
            
    filtered_df = df.filter(inds).collect().to_pandas() if lazy else df[inds].copy()
    if lazy and '__index_level_0__' in filtered_df.columns: # Fix index, if provided. This is a hack but seems to be needed as polars does not handle index properly by default
        filtered_df.index = filtered_df['__index_level_0__'] 

    
    # If not poststratisfied
    if not pp_desc.get('poststrat',True):
        filtered_df = filtered_df.assign(weight = 1.0) # Remove weighting
        if 'training_subsample' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['training_subsample']]
    
    n_datapoints = len(filtered_df)

    # Convert ordered categorical to continuous if we can
    res_meta = c_meta[pp_desc['res_col']]
    if pp_desc.get('convert_res') == 'continuous' and res_meta.get('ordered') and res_meta.get('categories','infer') != 'infer':
        nvals = get_cat_num_vals(res_meta,pp_desc) 
        cmap = dict(zip(res_meta['categories'],nvals))
        rc = gc_dict[pp_desc['res_col']] if pp_desc['res_col'] in gc_dict else [pp_desc['res_col']]
        for col in rc:
            filtered_df[col] = pd.to_numeric(filtered_df[col].astype('object').replace(cmap))

    # Apply continuous transformation
    dfcols = gc_dict.get(pp_desc['res_col'],[pp_desc['res_col']])
    if filtered_df[dfcols[0]].dtype.name != 'category':
        filtered_df[dfcols], val_format = transform_cont(filtered_df[dfcols],transform=pp_desc.get('cont_transform'))
    else: val_format = '.1%'
    val_format = pp_desc.get('value_format',val_format)
    
    # If res_col is a group of questions
    # This might move to wrangle but currently easier to do here as we have gc_dict handy
    if pp_desc['res_col'] in gc_dict:
        value_vars = [ c for c in gc_dict[pp_desc['res_col']] if c in cols ]
        if 'draw' in filtered_df.columns: ddraws = filtered_df['draw'] # Set aside draws as series for later
        
        id_vars = ['id'] + [ c for c in cols if (c not in value_vars or c in pp_desc.get('factor_cols',[])) ] # Make sure we leave factors in - in case we are faceting over one of the questions
        filtered_df = filtered_df.reset_index(names='id').melt(id_vars=id_vars, value_vars=value_vars, var_name='question', value_name=pp_desc['res_col'])

        # Fix the draws for each question separately
        if 'draw' in filtered_df.columns:
            draw_ar = []
            for c in value_vars:
                if c in data_meta.get('draws_data',{}):
                    uid, ndraws = data_meta['draws_data'][c]
                    draws = stable_draws(n_points, ndraws, uid)
                else: draws = ddraws
                draw_ar.append(pd.DataFrame({'draw': draws, 'question': c, 'id': np.arange(n_points) }))
            filtered_df = filtered_df.drop(columns=['draw']).merge(pd.concat(draw_ar),on=['id','question'],how='left')

        if plot_meta.get('data_format') != 'raw': filtered_df.drop(columns=['id'],inplace=True)

        # Convert to proper category with correct order
        filtered_df['question'] = pd.Categorical(filtered_df['question'],value_vars)
    elif 'question' in pp_desc['factor_cols']: # Create 'question' as a dummy dimension
        filtered_df['question'] = pd.Categorical([pp_desc['res_col']]*len(filtered_df))   

    # Filter out the unused categories so plots are cleaner
    for k in filtered_df.columns:
        if k == 'id': continue
        if filtered_df[k].dtype.name == 'category':
            m_cats = c_meta[k]['categories'] if c_meta[k].get('categories','infer')!='infer' else None
            f_cats = get_cats(filtered_df[k],m_cats) if k != pp_desc['res_col'] or not c_meta[k].get('likert') else m_cats # Do not trim likert as plots need to be symmetric
            
            #vals = filtered_df[k]
            filtered_df[k] = pd.Categorical(filtered_df[k],f_cats,ordered=c_meta[k].get('ordered',False))
    
    # Aggregate the data into right shape
    pparams = wrangle_data(filtered_df, data_meta, pp_desc)
    pparams['val_format'] = val_format

    
    # Remove prefix from question names in plots
    if 'col_prefix' in c_meta[pp_desc['res_col']] and pp_desc['res_col'] in gc_dict:
        prefix = c_meta[pp_desc['res_col']]['col_prefix']
        cmap = { c: c.replace(prefix,'') for c in pparams['data']['question'].dtype.categories }
        pparams['data']['question'] = pparams['data']['question'].cat.rename_categories(cmap)
            
    
    # How many datapoints the plot is based on. This is useful metainfo to display sometimes
    pparams['n_datapoints'] = n_datapoints
    
    if lazy: pl.disable_string_cache()
    
    return pparams

# %% ../nbs/02_pp.ipynb 21
def discretize_continuous(col, col_meta={}):

    if 'bin_breaks' in col_meta and 'bin_labels' in col_meta:
        cut = pd.cut(col, bins = col_meta['bin_breaks'], labels = col_meta['bin_labels'])
        cut = pd.Categorical(cut.astype(str), map(str,cut.dtype.categories), True) 
    else:
        breaks = col_meta.get('bin_breaks',5)
        if isinstance(breaks,int): 
            breaks = np.unique(np.quantile(col.dropna(), np.linspace(0,1,breaks+1)))
        cut = cut_nice(col, breaks, format=col_meta.get('value_format','.1f'))
    return cut

# Helper function that handles reformating data for create_plot
def wrangle_data(raw_df, data_meta, pp_desc):
    
    plot_meta = get_plot_meta(pp_desc['plot'])
    col_meta = extract_column_meta(data_meta)

    res_col, factor_cols = pp_desc.get('res_col'), pp_desc.get('factor_cols')
    
    draws, continuous, data_format = (plot_meta.get(n, False) for n in ['draws','continuous','data_format'])
    
    gb_dims = (['draw'] if draws else []) + (['id'] if 'id' in raw_df.columns else []) + (factor_cols if factor_cols else [])
    if res_col in gb_dims: gb_dims.remove(res_col) # Remove res_col from groupby dimensions in this function
    
    if 'weight' not in raw_df.columns: raw_df = raw_df.assign(weight=1.0) # This also works for empty df-s
    else: raw_df.loc[:,'weight'] = raw_df['weight'].fillna(1.0)

    if draws and 'draw' in raw_df.columns and 'augment_to' in pp_desc: # Should we try to bootstrap the data to always have augment_to points. Note this is relatively slow
        raw_df = augment_draws(raw_df,gb_dims[1:],threshold=pp_desc['augment_to'])
        
    pparams = { 'value_col': 'value' }
    data = None

    # Ensure all rv columns other than value are categorical
    for c in factor_cols:
        #if c in special_columns: continue # bypass some columns added above
        if raw_df[c].dtype.name != 'category' and c!=pp_desc['res_col']:
            if col_meta.get(c,{}).get('continuous') or not isinstance(raw_df[c],str):
                raw_df[c] = discretize_continuous(raw_df[c],col_meta.get(c,{}))
            else: # Just assume it's categorical by any other name
                raw_df[c] = pd.Categorical(raw_df[c])
    
    if data_format=='raw':
        pparams['value_col'] = res_col
        if plot_meta.get('sample'):
            data = gb_in(raw_df[gb_dims+[res_col]],gb_dims).sample(plot_meta['sample'],replace=True)
        else: data = raw_df[gb_dims+[res_col]]
        
    elif data_format=='longform':
        rc_meta = col_meta.get(res_col,{})
        if raw_df[res_col].dtype == 'category':  #'categories' in rc_meta: # categorical
            pparams['cat_col'] = res_col 
            pparams['value_col'] = 'percent'
            
            # Aggregate the data
            data = raw_df.groupby(gb_dims+[res_col],observed=False)['weight'].sum()
            if plot_meta.get('agg_fn')!='sum': data /= gb_in(raw_df,gb_dims)['weight'].sum()
            data = data.rename(pparams['value_col']).dropna().reset_index()
            
        else: # Continuous
            agg_fn = pp_desc.get('agg_fn','mean') # We may want to try median vs mean or plot sd-s or whatever
            agg_fn = plot_meta.get('agg_fn',agg_fn) # Some plots mandate this value (election model for instance)
            if len(gb_dims)>0: data = getattr(gb_in(raw_df,gb_dims)[res_col],agg_fn)().dropna().reset_index() 
            else: data = pd.DataFrame({res_col: [getattr(raw_df[res_col],agg_fn)()]}) # Single value data frame
            pparams['value_col'] = res_col
            
        if plot_meta.get('group_sizes'):
            data = data.merge(gb_in(raw_df,gb_dims).size().rename('group_size').reset_index(),on=gb_dims,how='left')
    else:
        raise Exception("Unknown data_format")
        

            
    pparams['data'] = data
    return pparams

# %% ../nbs/02_pp.ipynb 22
# Create a color scale
ordered_gradient = ["#c30d24", "#f3a583", "#94c6da", "#1770ab"]
def meta_color_scale(scale : Dict, column=None, translate=None):
    cats = column.dtype.categories if column.dtype.name=='category' else None
    if scale is None and column is not None and column.dtype.name=='category' and column.dtype.ordered:
        scale = dict(zip(cats,gradient_to_discrete_color_scale(ordered_gradient, len(cats))))
    if translate and cats is not None:
        remap = dict(zip(cats,[ translate(c) for c in cats ]))
        scale = { (remap[k] if k in remap else k) : v for k,v in scale.items() } if scale else scale
        cats = [ remap[c] for c in cats ]
    return to_alt_scale(scale,cats)

# %% ../nbs/02_pp.ipynb 23
internal_columns = ['draw','weight','group_size'] 

def translate_df(df, translate):
    df.columns = [ (translate(c) if c not in internal_columns else c) for c in df.columns ]
    for c in df.columns:
        if df[c].dtype.name == 'category':
            cats = df[c].dtype.categories
            remap = dict(zip(cats,[ translate(c) for c in cats ]))
            df[c] = df[c].cat.rename_categories(remap)
    return df

# %% ../nbs/02_pp.ipynb 24
def create_tooltip(pparams,tc_meta):
    
    data, tfn = pparams['data'], pparams['translate']
    
    label_dict = {}
    
    # Determine the columns we need tooltips for:
    tcols = [ f['col'] for f in pparams['facets'] if f['col'] in data.columns ]
            
    # Find labels mappings for regular columns
    for cn in tcols:
        if cn in tc_meta and 'labels' in tc_meta[cn]: label_dict[cn] = tc_meta[cn]['labels']
    
    # Find a mapping for multi-column questions
    question_tn = tfn('question')
    if question_tn in data.columns and any([ 'label' in tc_meta[c] for c in data[question_tn].unique() if c in tc_meta ]):
        label_dict[question_tn] = { c: tc_meta[c].get('label','') for c in data[question_tn].unique() if c in tc_meta and 'label' in tc_meta[c] }
    
    # Create the tooltips
    tooltips = [ alt.Tooltip(f"{pparams['value_col']}:Q", format=pparams['val_format']) ]
    for cn in tcols:
        if cn in label_dict:
            data[cn+'_label'] = data[cn].astype('object').replace({ k:tfn(v) for k,v in label_dict[cn].items() })
            t = alt.Tooltip(f"{cn}_label:N",title=cn)
        else:
            t = alt.Tooltip(f"{cn}:N")
        tooltips.append(t)
            
    return tooltips
    

# %% ../nbs/02_pp.ipynb 25
# Small helper function to move columns from internal to external columns
def remove_from_internal_fcols(cname, factor_cols, n_inner):
    if cname not in factor_cols[:n_inner]: return n_inner
    factor_cols.remove(cname)
    if n_inner>len(factor_cols): n_inner-=1
    factor_cols.insert(n_inner,cname)
    return n_inner

def inner_outer_factors(factor_cols, pp_desc, plot_meta):
    # Determine how many factors to use as inner facets
    in_f = pp_desc.get('internal_facet',False)
    n_min_f, n_rec_f = plot_meta.get('n_facets',(0,0))
    n_inner =  (n_rec_f if in_f else n_min_f) if isinstance(in_f,bool) else in_f
    if n_inner>len(factor_cols): n_inner = len(factor_cols)

    # If question facet as inner facet for a no_question_facet plot, just move it out
    if plot_meta.get('no_question_facet'):
        n_inner = remove_from_internal_fcols('question',factor_cols,n_inner)
        n_inner = remove_from_internal_fcols(pp_desc['res_col'],factor_cols,n_inner)
    
    return factor_cols, n_inner

# %% ../nbs/02_pp.ipynb 26
# Function that takes filtered raw data and plot information and outputs the plot
# Handles all of the data wrangling and parameter formatting
def create_plot(pparams, data_meta, pp_desc, alt_properties={}, alt_wrapper=None, dry_run=False, width=200, return_matrix_of_plots=False, translate=None):
    data = pparams['data']
    plot_meta = get_plot_meta(pp_desc['plot'])
    col_meta = extract_column_meta(data_meta)

    if 'question' in data.columns: # TODO: this should be in io.py already, probably
      col_meta['question']['colors'] = col_meta[pp_desc['res_col']].get('question_colors',None)
  
    plot_args = pp_desc.get('plot_args',{})
    pparams.update(plot_args)

        
    # Get list of factor columns (adding question and category if needed)
    factor_cols, n_inner = inner_outer_factors(pp_desc['factor_cols'], pp_desc, plot_meta)

    # Reorder categories if required
    if pp_desc.get('sort'):
        for cn in pp_desc['sort']:
            ascending = pp_desc['sort'][cn] if isinstance(pp_desc['sort'],dict) else False
            if cn not in data.columns or cn==pparams['value_col']: 
                raise Exception(f"Sort column {cn} not found")
            if plot_meta.get('sort_numeric_first_facet'): # Some plots (like likert_bars) need a more complex sort
                f0 = factor_cols[0]
                nvals = get_cat_num_vals(col_meta[f0],pp_desc) 
                cmap = dict(zip(col_meta[f0]['categories'],nvals))
                sdf = data[ [cn,f0,pparams['value_col']] ]
                sdf['sort_val'] = sdf[pparams['value_col']]*sdf[f0].astype('object').replace(cmap)
                ordervals = sdf.groupby(cn,observed=True)['sort_val'].mean()
            else:
                ordervals = data.groupby(cn,observed=True)[pparams['value_col']].mean()
            order = ordervals.sort_values(ascending=ascending).index
            data[cn] = pd.Categorical(data[cn],list(order))
    

        
    # Handle translation funcion
    if translate is None: translate = (lambda s: s)
    pparams['translate'] = translate

    # Handle internal facets (and translate as needed)
    pparams['facets'] = []
    if n_inner>0:
        for cn in factor_cols[:n_inner]:
            fd = {
                'col': translate(cn),
                'ocol': cn,
                'order': [ translate(c) for c in data[cn].dtype.categories ],
                'colors': meta_color_scale(col_meta[cn].get('colors',None), data[cn], translate=translate), 
            }
            pparams['facets'].append(fd)

        # Pass on data from facet column meta if specified by plot
        for i,d in enumerate(plot_meta.get('requires',[])):
            for k, v in d.items():
                if v=='pass': pparams[k] = col_meta[pparams['facets'][i]['ocol']].get(k)
        
        factor_cols = factor_cols[n_inner:] # Leave rest for external faceting

    # Translate the data itself
    pparams['data'] = data = translate_df(data,translate)
    pparams['value_col'] = translate(pparams['value_col'])  
    factor_cols = [ translate(c) for c in factor_cols ]
    t_col_meta = { translate(c): v for c,v in col_meta.items() }

    # Handle tooltip
    pparams['tooltip'] = create_tooltip(pparams,t_col_meta)
    
    # If we still have more than 1 factor left, merge the rest into one so we have a 2d facet
    if len(factor_cols)>1:
        n_facet_cols = len(data[factor_cols[-1]].dtype.categories)
        if not return_matrix_of_plots and len(factor_cols)>2:

            # Preserve ordering of categories we combine
            nf_order = [ ', '.join(t) for t in it.product(*[list(data[c].dtype.categories) for c in factor_cols[1:]])]
            factor_col = ', '.join(factor_cols[1:])
            jfs = data[factor_cols[1:]].agg(', '.join, axis=1)
            data.loc[:,factor_col] = pd.Categorical(jfs,nf_order)
            pparams['data'] = data
            factor_cols = [factor_cols[0], factor_col]

        if len(factor_cols)>=2:
            factor_cols = list(reversed(factor_cols))
            n_facet_cols = len(data[factor_cols[1]].dtype.categories)
    else:
        n_facet_cols = plot_meta.get('factor_columns',1)
        
    # Allow value col name to be changed. This can be useful in distinguishing different aggregation options for a column
    if 'value_name' in pp_desc: 
        pparams['data'] = pparams['data'].rename(columns={pparams['value_col']:pp_desc['value_name']})
        pparams['value_col'] = pp_desc['value_name']
    
    # Do width/height calculations
    if factor_cols: n_facet_cols = pp_desc.get('n_facet_cols',n_facet_cols) # Allow pp_desc to override col nr
    dims = {'width': width//n_facet_cols if factor_cols else width}
    if 'aspect_ratio' in plot_meta:   dims['height'] = int(dims['width']/plot_meta['aspect_ratio'])
    
    # Make plot properties available to plot function (mostly useful for as_is plots)
    pparams.update({'width':width}); pparams['alt_properties'] = alt_properties; pparams['outer_factors'] = factor_cols

    # Create the plot using it's function
    if dry_run: return pparams
    
    # Trim down parameters list if needed
    plot_fn = get_plot_fn(pp_desc['plot'])
    pparams = clean_kwargs(plot_fn,pparams)
    if alt_wrapper is None: alt_wrapper = lambda p: p
    if plot_meta.get('as_is'): # if as_is set, just return the plot as-is
        return plot_fn(**pparams)
    elif factor_cols:
        if return_matrix_of_plots: # return a 2d list of plots which can be rendeed one plot at a time
            del pparams['data']
            combs = it.product( *[data[fc].dtype.categories for fc in factor_cols ])
            #print( [ data[(data[factor_cols]==c).all(axis=1)] for c in combs ] )
            #print(list(combs))
            return list(batch([
                alt_wrapper(plot_fn(data[(data[factor_cols]==c).all(axis=1)],**pparams)
                            .properties(title='-'.join(map(str,c)),**dims, **alt_properties)
                            .configure_view(discreteHeight={'step':20}))
                for c in combs
                ], n_facet_cols))
        else: # Use faceting
            if n_facet_cols==1:
                plot = alt_wrapper(plot_fn(**pparams).properties(**dims, **alt_properties).facet(
                    row=alt.Row(f'{factor_cols[0]}:O', sort=list(data[factor_cols[0]].dtype.categories), header=alt.Header(labelOrient='top'))))
            elif n_facet_cols==len(data[factor_cols[0]].dtype.categories):
                plot = alt_wrapper(plot_fn(**pparams).properties(**dims, **alt_properties).facet(
                    column=alt.Column(f'{factor_cols[1]}:O', sort=list(data[factor_cols[1]].dtype.categories)),
                    row=alt.Row(f'{factor_cols[0]}:O', sort=list(data[factor_cols[0]].dtype.categories), header=alt.Header(labelOrient='top'))))
            else: # n_facet_cols!=1 but just one facet
                plot = alt_wrapper(plot_fn(**pparams).properties(**dims, **alt_properties).facet(f'{factor_cols[0]}:O',columns=n_facet_cols))
            plot = plot.configure_view(discreteHeight={'step':20})
    else:
        plot = alt_wrapper(plot_fn(**pparams).properties(**dims, **alt_properties).configure_view(discreteHeight={'step':20}))

        if return_matrix_of_plots: plot = [[plot]]

    return plot


# %% ../nbs/02_pp.ipynb 28
# Compute the full factor_cols list, including question and res_col as needed
def impute_factor_cols(pp_desc, col_meta, plot_meta=None):
    factor_cols = pp_desc.get('factor_cols',[]).copy()

    # Determine if res is categorical
    cat_res = 'categories' in col_meta[pp_desc['res_col']] and pp_desc.get('convert_res')!='continuous' 

    # Add res_col if we are working with a categorical input (and not converting it to continuous)
    if cat_res and pp_desc['res_col'] not in factor_cols: 
        factor_cols.insert(0,pp_desc['res_col'])

    # Determine if we have 'question' as a column
    has_q = 'columns' in col_meta[pp_desc['res_col']] # Check if res_col is a group of questions
    if len(factor_cols)<1 and not has_q: has_q = True # Create 'question' as a dummy dimension so we have at least one factor (generally required for plotting)
    
    # If we need to, add question as a factor to list
    if has_q and 'question' not in factor_cols:
        if cat_res: factor_cols.append('question') # Put it last for categorical values
        else: factor_cols.insert(0,'question') # And first for continuous values, as it then often represents the "category"

    # Pass the factor_cols through the same changes done inside plot pipeline to make more explicit what happens
    if plot_meta: factor_cols, _ = inner_outer_factors(factor_cols, pp_desc, plot_meta)

    return factor_cols

# %% ../nbs/02_pp.ipynb 29
# A convenience function to draw a plot straight from a dataset
def e2e_plot(pp_desc, data_file=None, full_df=None, data_meta=None, width=800, check_match=True, impute=True,**kwargs):
    if data_file is None and full_df is None:
        raise Exception('Data must be provided either as data_file or full_df')
    if data_file is None and data_meta is None:
        raise Exception('If data provided as full_df then data_meta must also be given')
        
    if full_df is None: 
        # No more lazy loading because a) it was very unstable b) it does not have a real use case c) it does not allow for easy virtual passes
        #if data_file.endswith('.parquet'): # Try lazy loading as it only loads what it needs from disk
        #    full_df, full_meta = load_parquet_with_metadata(data_file,lazy=lazy)
        #    dm = full_meta['data']
        #else: 
        full_df, dm = read_annotated_data(data_file)
        if data_meta is None: data_meta = dm

    pp_desc = pp_desc.copy()
    if impute: pp_desc['factor_cols'] = impute_factor_cols(pp_desc, extract_column_meta(data_meta), get_plot_meta(pp_desc['plot']))

    if check_match:
        matches = matching_plots(pp_desc, full_df, data_meta, details=True, list_hidden=True)    
        if pp_desc['plot'] not in matches: 
            raise Exception(f"Plot not registered: {pp_desc['plot']}")
        
        fit, imp = matches[pp_desc['plot']]
        if  fit<0:
            raise Exception(f"Plot {pp_desc['plot']} not applicable in this situation because of flags {imp}")
            
    pparams = get_filtered_data(full_df, data_meta, pp_desc)
    return create_plot(pparams, data_meta, pp_desc, width=width,**kwargs)

# Another convenience function to simplify testing new plots
def test_new_plot(fn, pp_desc, *args, plot_meta={}, **kwargs):
    stk_plot(**{**plot_meta,'plot_name':'test'})(fn) # Register the plot under name 'test'
    pp_desc = {**pp_desc, 'plot': 'test'}
    res = e2e_plot(pp_desc,*args,**kwargs)
    stk_deregister('test') # And de-register it again
    return res
