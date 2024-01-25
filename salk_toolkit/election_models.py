# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_election_models.ipynb.

# %% auto 0
__all__ = ['dhondt', 'simulate_election', 'simulate_election_e2e', 'simulate_election_pp', 'coalition_applet', 'mandate_plot']

# %% ../nbs/04_election_models.ipynb 3
import json, os, inspect
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

import altair as alt
import scipy.stats as sps

from salk_toolkit.utils import *
from salk_toolkit.io import extract_column_meta, read_json
from salk_toolkit.plots import stk_plot, register_stk_cont_version

import streamlit as st

# %% ../nbs/04_election_models.ipynb 5
def dhondt(pvotes, n_mandates, dh_power=1.0, pmand=None):
    
    # Calculate d'Hondt values and get party indices out
    n_mandates = np.array(n_mandates)
    max_mandates = int(n_mandates.max())
    if pmand is None: pmand = np.zeros_like(pvotes) # previously handed out mandates by party - zero by default
    dhvals = pvotes[:,:,None]/(pmand[:,:,None]+np.arange(1, max_mandates+1, 1)[None,None,:])**dh_power
    sinds = np.argsort(-dhvals.reshape( (dhvals.shape[0],-1) ),axis=1) // max_mandates

    # Select the first n as compensation
    rmand = np.ones(pvotes.shape[0]) * n_mandates # This can be a vector, one per draw
    ri = ((np.arange(sinds.shape[1])[None,:]-rmand[:,None])<0)
    comp_ident = np.concatenate([np.zeros( (1,pvotes.shape[-1]) ),np.identity(pvotes.shape[-1])])
    return comp_ident[(sinds+1)*ri].sum(axis=1)

# Vectorized basic election simulation: quotas, dHondt
# Input 'support' should be of shape (draws,districts,parties)
def simulate_election(support, nmandates, threshold=0.0, ed_threshold=0.0, quotas=True, first_quota_coef=1.0, dh_power=1.0, body_size=None, **kwargs):

    # Remove parties below a national threshold
    zero_mask = (support.sum(axis=1)/(support.sum(axis=(1,2))+1e-3)[:,None])>threshold
    uzsim_t = zero_mask[:,None,:]*support
    
    # Remove parties below an electoral_district specific threshold
    zero_mask = (support/(support.sum(axis=(2))+1e-3)[:,:,None])>ed_threshold
    uzsim_t = zero_mask[:,:,:]*uzsim_t

    # Districts with quotas, then country-level compensation (Estonian system)
    if quotas:
        quotas = (support.sum(axis=-1)+1e-3)/nmandates[None,:]
        v, r = np.divmod(uzsim_t/quotas[:,:,None],1.0)
        dmandates = v+(r>=first_quota_coef)
    
        # Calculate votes and mandates for each party
        pvotes = uzsim_t.sum(axis=1)
        pmand = dmandates.sum(axis=1)

        # Calculate compensation votes using dHondt
        if body_size is None: body_size = sum(nmandates)
        remaining_mand = body_size - pmand.sum(axis=1)
        comp_mandates = dhondt(pvotes, remaining_mand, dh_power, pmand)
        
        # Return the districts + compensation results
        return np.concatenate( [dmandates,comp_mandates[:,None,:]],axis=1 )
    
    else: # Separate election in each district (Croatian system)
        
        return np.stack([ 
            dhondt(uzsim_t[:,i,:],nmandates[i],dh_power)
            for i in range(support.shape[1]) ],axis=1)

# %% ../nbs/04_election_models.ipynb 6
# Basic wrapper around simulate elections that goes from dataframe to dataframe
def simulate_election_e2e(sdf, parties, mandates_dict, ed_col='electoral_district', **kwargs):
    
    # Convert data frame to a numpy tensor for fast vectorized processing
    parties = [ p for p in parties if p in sdf.columns ]
    ed_df = sdf.groupby(['draw',ed_col])[parties].sum()
    districts = list(sdf.electoral_district.unique())
    support = ed_df.reset_index(drop=True).to_numpy().reshape( (-1,len(districts),len(parties)) )    
    nmandates = np.array([ mandates_dict[d] for d in districts ])
    
    edt = simulate_election(support, nmandates, **kwargs)
    
    if edt.shape[1]>support.shape[1]: districts = districts + ['Compensation']
    
    # Shape it back into a data frame
    eddf = pd.DataFrame( edt.reshape( (-1,) ), columns=['mandates'], dtype='int')
    eddf.loc[:, ['draw', ed_col, 'party']] = np.array(tuple(it.product( range(edt.shape[0]), districts, parties )))
    return eddf

# %% ../nbs/04_election_models.ipynb 10
def simulate_election_pp(data, mandates, electoral_system, cat_col, value_col, factor_col, cat_order, factor_order):
    # Reshape input to (draws,electoral_districts,parties)
    draws = data.draw.unique()
    pdf = data.pivot(index=['draw',factor_col], columns=cat_col, values=value_col).reset_index()
    ded = pd.DataFrame(list(it.product(draws,factor_order)),columns=['draw',factor_col])
    sdata = ded.merge(pdf,on=['draw',factor_col]).loc[:,cat_order].fillna(0).to_numpy().reshape( (len(draws),len(factor_order),len(cat_order)) )
    
    # Run the actual electoral simulation
    nmandates = np.array([ mandates[d] for d in factor_order ])
    edt = simulate_election(sdata,nmandates,**electoral_system)
    if edt.shape[1]>sdata.shape[1]: factor_order = factor_order+['Compensation']
    
    # Shape it back into a data frame
    df = pd.DataFrame( edt.reshape( (-1,) ), columns=['mandates'])
    df.loc[:, ['draw',factor_col, cat_col]] = np.array(tuple(it.product( draws, factor_order, cat_order )))
    
    return df

# %% ../nbs/04_election_models.ipynb 11
# This fits into the pp framework as: cat_col=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot('coalition_applet', data_format='longform', draws=True, requires_factor=True, agg_fn='sum', factor_meta=['mandates','electoral_system'], as_is=True)#, hidden=True)
def coalition_applet(data, mandates, electoral_system, cat_col, value_col='value', color_scale=alt.Undefined, cat_order=alt.Undefined, factor_col=None, factor_order=alt.Undefined, width=None, alt_properties={}, outer_factors=[], translate=None):
    
    tf = translate if translate else (lambda s: s)
    
    if outer_factors: raise Exception("This plot does not work with extra factors")
    
    sdf = simulate_election_pp(data, mandates, electoral_system, cat_col, value_col,factor_col,cat_order,factor_order)

    # Aggregate to total mandate counts
    adf = sdf.groupby(['draw',cat_col])['mandates'].sum().reset_index()
    adf = adf[adf['mandates']>0]

    parties = list(adf[cat_col].unique()) # Leave only parties that have mandates

    coalition = st.multiselect(tf('Select the coalition:'),
        cat_order,
        help=tf('Choose the parties whose coalition to model'))

    st.markdown("""___""")

    col1, col2 = st.columns((9, 9), gap='large')
    col1.markdown(tf('**Party mandate distributions**'))

    # Individual parties plot
    ddf = adf.groupby(cat_col)['mandates'].value_counts().rename('count').reset_index()
    p_plot = alt.Chart(
            ddf,
            #title=var
        ).mark_bar(opacity=0.5, stroke='black', strokeWidth=0, size=20).encode(
            alt.X('mandates:Q', title="Mandates", axis=alt.Axis(tickMinStep=1),scale=alt.Scale(domainMin=0)),
            alt.Y('count:Q', title=None, axis=None),
            alt.Row(f'{cat_col}:N', title=None),
            color=alt.Color(f'{cat_col}:N', legend=None, scale=color_scale),
            tooltip=[alt.Tooltip('mandates:Q', format=',d')]
        ).properties(height=60)
    col1.altair_chart(p_plot, use_container_width=True)

    total_mandates = sum(mandates.values())

    col2.markdown(tf('**Coalition simulation**'))
    n = col2.number_input(tf('Choose mandate cutoff:'), min_value=0, max_value=total_mandates, value=(total_mandates//2) + 1, step=1, help='...')

    if len(coalition)>0:
        # Coalition plot
        acdf = adf[adf[cat_col].isin(coalition)]
        cdf = acdf.groupby('draw')['mandates'].sum().value_counts().rename('count').reset_index()

        mi, ma = min(cdf['mandates'].min(),n), max(cdf['mandates'].max(),n)
        tick_count = (ma-mi+1) # This is the only way to enforce integer ticks as tickMinStep seems to not do it sometimes
        k_plot = alt.Chart(cdf).mark_bar(color='#ff2b2b',size=20).encode(
            x=alt.X('mandates:Q', title='Mandates', scale=alt.Scale(round=True), axis=alt.Axis(tickMinStep=1,tickCount=tick_count)),
            y=alt.Y('count:Q', title=None, stack=None, axis=None),
        ).properties(height=200,width=300)
        rule = alt.Chart(pd.DataFrame({'x': [n]})).mark_rule(color='red', size=1.25, strokeDash=[5, 2]).encode(x='x')
        col2.altair_chart((k_plot+rule).configure_view(strokeWidth=0), use_container_width=True)

        col2.write(tf("Probability of at least  **{0:.0f}** mandates: **{1:.1%}**").format(n, (cdf['mandates'] > n-1).mean()))
        #col3.write('Distributsiooni mediaan: **{:d}**'.format(int((d_dist[koalitsioon].sum(1)).median())))
        #m, l, h = hdi(sim_data['riigikogu'][koalitsioon], 0.9)
        #col2.write('Distributsiooni mediaan on **{:.0f}** mandaati. 90% tõenäosusega jääb mandaatide arv **{:.0f}** ning **{:.0f}** vahele.'.format(m, l, h))
        
    return None

register_stk_cont_version('coalition_applet')

# %% ../nbs/04_election_models.ipynb 12
# This fits into the pp framework as: cat_col=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot('mandate_plot', data_format='longform', draws=True, requires_factor=True, agg_fn='sum', factor_meta=['mandates','electoral_system'], as_is=True)#, hidden=True)
def mandate_plot(data, mandates, electoral_system, cat_col, value_col='value', color_scale=alt.Undefined, cat_order=alt.Undefined, factor_col=None, factor_order=alt.Undefined, width=None, alt_properties={}, outer_factors=[]):
    
    if outer_factors: raise Exception("This plot does not work with extra factors")
    
    df = simulate_election_pp(data, mandates, electoral_system, cat_col,value_col,factor_col,cat_order,factor_order)
    
    # Shape it into % values for each vote count
    maxv = df['mandates'].max()
    tv = np.arange(1,maxv+1,dtype='int')[None,:]
    dfv = df['mandates'].to_numpy()[:,None]
    dfm = pd.DataFrame((dfv>=tv).astype('int'),columns=tv[0], index=df.index)
    dfm['draw'],dfm[cat_col], dfm[factor_col] = df['draw'], df[cat_col], df[factor_col]
    res = dfm.groupby([cat_col,factor_col],observed=True)[tv[0]].mean().reset_index().melt(id_vars=[cat_col,factor_col],
                                                                                var_name='mandates',value_name='percent')
    # Remove parties who have no chance of even one elector
    eliminate = (res.groupby(cat_col,observed=True)[value_col].sum() < 0.2)
    el_cols = [i for i,v in eliminate.items() if v]
    res = res[~res[cat_col].isin(el_cols)]
    cat_order = list(eliminate[~eliminate].index)
    
    f_width = max(50,width/len(cat_order))

    plot = alt.Chart(data=res).mark_bar().encode(
        x=alt.X('mandates',title=None),
        y=alt.Y(value_col,title=None,axis=alt.Axis(format='%')),
        color=alt.Color(f'{cat_col}:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip(cat_col, title='party'),
            alt.Tooltip(factor_col),
            alt.Tooltip('mandates'),
            alt.Tooltip(value_col, format='.1%', title='probability'),
            ]
    ).properties(
        width=f_width,
        height=f_width//2,
        **alt_properties
        #title="Ringkonna- ja kompensatsioonimandaatide tõenäolised jaotused"
    ).facet(
        #header=alt.Header(labelAngle=-90),
        row=alt.X(
            f'{factor_col}:N',
            sort=factor_order+['Compensation'],
            title=None,
            header=alt.Header(labelOrient='top')
            ),
        column=alt.Y(
            f'{cat_col}:N',
            sort=cat_order,
            title=None,
            header=alt.Header(labelFontWeight='bold')
            ),
    )
    return plot

register_stk_cont_version('mandate_plot')
