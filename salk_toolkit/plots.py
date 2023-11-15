# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_plots.ipynb.

# %% auto 0
__all__ = ['registry', 'registry_meta', 'priority_weights', 'register', 'get_plot_fn', 'get_plot_meta', 'calculate_priority',
           'matching_plots', 'boxplots', 'boxplots_cont', 'make_start_end', 'likert_bars', 'density', 'matrix', 'lines',
           'likert_smooth', 'likert_aggregate', 'likert_rad_pol']

# %% ../nbs/03_plots.ipynb 3
import json, os
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

import altair as alt

from salk_toolkit.utils import *
from salk_toolkit.io import extract_column_meta

# %% ../nbs/03_plots.ipynb 5
registry = {}
registry_meta = {}

# %% ../nbs/03_plots.ipynb 7
# Decorator for registering a plot type with metadata
def register(plot_name, **r_kwargs):
    
    def decorator(gfunc):
        # In theory, we could do transformations in wrapper
        # In practice, it would only obfuscate already complicated code
        #def wrapper(*args,**kwargs) :
        #    return gfunc(*args,**kwargs)

        # Register the function
        registry[plot_name] = gfunc
        registry_meta[plot_name] = r_kwargs
        
        return gfunc
    
    return decorator

def get_plot_fn(plot_name):
    return registry[plot_name]

def get_plot_meta(plot_name):
    return registry_meta[plot_name]

# %% ../nbs/03_plots.ipynb 8
# First is weight if not matching, second if match
# This is very much a placeholder right now
priority_weights = {
    'likert': [-10000, 100],
    'continuous': [-10000, 100],
    'draws': [0,50],
    'question': [0, 100],
    'ordered': [-10000,100],
    'ordered_factor':[-10000,100]
}

def calculate_priority(plot_meta, match):
    return sum([ priority_weights[k][vod(match,k) or 0] for k, v in plot_meta.items() if k in priority_weights and v ])

# Get a list of plot types matching required spec
def matching_plots(args, df, data_meta):
    
    rc = args['res_col']
    col_meta = extract_column_meta(data_meta)
    
    match = {
        'draws': ('draw' in df.columns),
        'likert': vod(col_meta[rc],'likert'),
        'question': (rc not in df.columns),
        'continuous': vod(col_meta[rc],'continuous'),
        'ordered': vod(col_meta[rc],'ordered'),
        'ordered_factor': (vod(args,'factor_cols',[])!=[]) and vod(col_meta[args['factor_cols'][0]],'ordered') and vod(vod(args,'plot_args',{}),'internal_facet'),
    }
    
    res = [ ( pn, calculate_priority(get_plot_meta(pn),match) ) for pn in registry.keys() ]
    res = [ n for (n,p) in sorted(res,key=lambda t: t[1], reverse=True) if p >= 0 ]
    
    return res

# %% ../nbs/03_plots.ipynb 14
@register('boxplots', data_format='longform', draws=True)
def boxplots(data, cat_col, value_col='value', color_scale=alt.Undefined, factor_col=None, factor_color_scale=alt.Undefined):
    cat_order = list(data[cat_col].dtype.categories)
    
    plot = alt.Chart(round(data, 3), width = 'container' \
    ).mark_boxplot(
        clip=True,
        #extent='min-max',
        outliers=False
    ).encode(
        y=alt.Y(f'{cat_col}:N', title=None, sort=cat_order),
        x=alt.X(
            f'{value_col}:Q',
            title=value_col,
            #axis=alt.Axis(format='%'),
            #scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
        
        #tooltip=[
        #    'response:N',
            #alt.Tooltip('mean(support):Q',format='.1%')
        #    ],
        **({
                'color': alt.Color(f'{cat_col}:N', scale=color_scale, legend=None)    
            } if not factor_col else {
                'yOffset':alt.YOffset(f'{factor_col}:N', title=None, sort=list(data[factor_col].dtype.categories)), 
                'color': alt.Color(f'{factor_col}:N', scale=factor_color_scale, legend=alt.Legend(orient='top'))
            }),
    )
    return plot

# Version for continous that just replaces 'question' for cat_col
@register('boxplots-cont', data_format='longform', draws=True, continuous=True, question=True)
def boxplots_cont(data, value_col='value', question_color_scale=alt.Undefined, factor_col=None, factor_color_scale=alt.Undefined):
    return boxplots(data, cat_col='question', value_col=value_col, color_scale=question_color_scale, factor_col=factor_col, factor_color_scale=factor_color_scale)

# %% ../nbs/03_plots.ipynb 16
# Make the likert bar pieces
def make_start_end(x,value_col):
    #print("######################")
    #print(x)
    scale_start=1
    x_mid = x.iloc[2:3,:]
    x_mid.loc[:,'end'] = -scale_start+x_mid[value_col]
    x_mid.loc[:,'start'] = -scale_start
    x_other = x.iloc[[0,1,3,4],:]
    x_other.loc[:,'end'] = x_other[value_col].cumsum() - x_other[0:2][value_col].sum()
    x_other.loc[:,'start'] = (x_other[value_col][::-1].cumsum()[::-1] - x_other[2:4][value_col].sum())*-1
    return pd.concat([x_other, x_mid])

@register('likert_bars',data_format='longform',question=True,draws=False,likert=True)
def likert_bars(data, cat_col, value_col='value', color_scale=alt.Undefined, factor_col=None, factor_color_scale=alt.Undefined):
    gb_cols = list(set(data.columns)-{ cat_col, value_col }) # Assume all other cols still in data will be used for factoring
    
    options_cols = list(data[cat_col].dtype.categories) # Get likert scale names
    bar_data = data.groupby(gb_cols, group_keys=False).apply(make_start_end,value_col=value_col)
    
    plot = alt.Chart(bar_data).mark_bar() \
        .encode(
            x=alt.X('start:Q', axis=alt.Axis(title=None, format = '%')),
            x2=alt.X2('end:Q'),
            y=alt.Y(f'question:N', axis=alt.Axis(title=None, offset=5, ticks=False, minExtent=60, domain=False), sort=list(data['question'].dtype.categories)),
            tooltip=[alt.Tooltip('question'), alt.Tooltip(cat_col), alt.Tooltip(f'{value_col}:Q', title=value_col, format='.1%')],
            color=alt.Color(
                f'{cat_col}:N',
                legend=alt.Legend(
                    title='Response',
                    orient='bottom',
                    ),
                scale=alt.Scale(domain=options_cols, range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab", ]),
            ),
            **({ 'yOffset':alt.YOffset(f'{factor_col}:N', title=None, sort=list(data[factor_col].dtype.categories))} if factor_col else {})
        )
    return plot

# %% ../nbs/03_plots.ipynb 18
@register('density', data_format='raw', continuous=True, sample=100, factor_columns=3)
def density(data, value_col='value',factor_col=None, factor_color_scale=alt.Undefined):
    gb_cols = list(set(data.columns)-{ value_col }) # Assume we groupby over everything except value
    plot = alt.Chart(
            data
        ).transform_density(
            value_col,
            groupby=gb_cols,
            as_=[value_col, 'density'],
            #extent=[1, 10],
        ).mark_line().encode(
            x=alt.X(f"{value_col}:Q"),
            y=alt.Y('density:Q',axis=alt.Axis(title=None, format = '%')),
            **({'color': alt.Color(f'{factor_col}:N', scale=factor_color_scale, legend=alt.Legend(orient='top'))} if factor_col else {})
        )
    return plot

# %% ../nbs/03_plots.ipynb 20
@register('matrix', data_format='longform', continuous=True, question=True)
def matrix(data, value_col='value',factor_col=None, factor_color_scale=alt.Undefined):
    
    base = alt.Chart(data).mark_rect().encode(
            x=alt.X(f'{factor_col}:N', title=None, sort=list(data[factor_col].dtype.categories)),
            y=alt.Y('question:N', title=None, sort=list(data['question'].dtype.categories)),
            color=alt.Color(value_col, scale=alt.Scale(scheme='redyellowgreen', domainMid=0),
                legend=alt.Legend(title=None)),
            tooltip=[*([factor_col] if factor_col else []), 'question', alt.Tooltip(f'{value_col}:Q', title=None, format=',.2f')],
        )

    text = base.mark_text().encode(
        text=alt.Text(value_col, format='.1f'),
        color=alt.condition(
            alt.datum[value_col]**2 > 1.5,
            alt.value('white'),
            alt.value('black')
        ),
        tooltip=[
            alt.Tooltip('question'),
            *([alt.Tooltip(factor_col)] if factor_col else []),
            alt.Tooltip(value_col, format='.2f')]
    )
    
    return base+text

# %% ../nbs/03_plots.ipynb 22
@register('lines',data_format='longform', question=False, draws=False, ordered_factor=True)
def lines(data, cat_col, value_col='value', color_scale=alt.Undefined, factor_col=None, smooth=False):
    if smooth:
        smoothing = 'basis'
        points = 'transparent'
    else:
        smoothing = 'natural'
        points = True

    plot = alt.Chart(data).mark_line(point=points, interpolate=smoothing).encode(
        alt.X(f'{factor_col}:O', title=None),
        alt.Y(f'{value_col}:Q', title=None, axis=alt.Axis(format='%')
            ),
        tooltip=[
            *([alt.Tooltip(factor_col)] if factor_col else []),
            alt.Tooltip(f'{value_col}:Q', format='.1%')],
        color=alt.Color(f'{cat_col}:N', scale=color_scale, legend=alt.Legend(orient='top'))
    )
    return plot


# %% ../nbs/03_plots.ipynb 24
@register('likert_smooth',data_format='longform', question=False, draws=False, likert=True, ordered_factor=True)
def likert_smooth(data, cat_col, value_col='value', factor_col=None):
    options_cols = list(data[cat_col].dtype.categories)
    ldict = dict(zip(options_cols, range(len(options_cols))))
    plot=alt.Chart(data
        ).mark_area(interpolate='natural').encode(
            x=alt.X(f'{factor_col}:O', title=None),
            y=alt.Y(f'{value_col}:Q', title=None, stack='normalize',
                 scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')
                 ),
            order="order:O",
            color=alt.Color(cat_col, legend=alt.Legend(orient='right', title=None),
                sort=alt.SortField("order", "descending"), scale=alt.Scale(domain=options_cols, range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab", ])
                ),
            #tooltip=[alt.Tooltip(teema, title='vastus'), 'laine',
            #    alt.Tooltip('pct:Q', title='osakaal', format='.1%')]
        ).transform_calculate(order=f"{ldict}[datum.{cat_col}]") # TODO: cat_col needs rename to be robust to weird column names
    return plot

# %% ../nbs/03_plots.ipynb 26
def likert_aggregate(x, cat_col, value_col):
    
    cc, vc = x[cat_col], x[value_col]
    cats = cc.dtype.categories
    
    #print(len(x),x.columns,x.head())
    pol = ( np.minimum(
                vc[cc.isin([cats[0], cats[1]])].sum(),
                vc[cc.isin([cats[3], cats[4]])].sum()
            ) / vc[cc !=  cats[2]].sum() )

    rad = ( vc[cc.isin([cats[0],cats[4]])].sum() /
            vc[cc != cats[2]].sum() )

    rel = vc[cc == cats[2]].sum()/vc.sum()

    return pd.Series({ 'polarisation': pol, 'radicalisation':rad, 'relevance':rel})

@register('likert_rad_pol',data_format='longform', question=False, draws=False, likert=True)
def likert_rad_pol(data, cat_col, value_col='value', factor_col=None, factor_color_scale=alt.Undefined):
    gb_cols = list(set(data.columns)-{ cat_col, value_col }) # Assume all other cols still in data will be used for factoring
    options_cols = list(data[cat_col].dtype.categories) # Get likert scale names
    likert_indices = data.groupby(gb_cols, group_keys=False).apply(likert_aggregate,cat_col=cat_col,value_col=value_col).reset_index()
    
    plot = alt.Chart(likert_indices).mark_circle().encode(
        x=alt.X('polarisation:Q'),
        y=alt.Y('radicalisation:Q'),
        size=alt.Size('relevance:Q', legend=None, scale=alt.Scale(range=[100, 500])),
        opacity=alt.value(1.0),
        stroke=alt.value('#777'),
        tooltip=[
            *([alt.Tooltip(f'{factor_col}:N')] if factor_col else []),
            alt.Tooltip('radicalisation:Q', format='.2'),
            alt.Tooltip('polarisation:Q', format='.2'),
            alt.Tooltip('relevance:Q', format='.2')
        ],
        **({'color': alt.Color(f'{factor_col}:N', scale=factor_color_scale, legend=alt.Legend(orient='top'))} if factor_col else {})
        )
    return plot
