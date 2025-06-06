"""Tools for election modeling, including plots spcifically designed for that"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_election_models.ipynb.

# %% auto 0
__all__ = ['dhondt', 'simulate_election', 'vec_smallest_k', 'cz_system', 'simulate_election_e2e', 'simulate_election_pp',
           'mandate_plot', 'coalition_applet']

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
from salk_toolkit.plots import stk_plot

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
def simulate_election(support, nmandates, threshold=0.0, ed_threshold=0.0, quotas=True, first_quota_coef=1.0, dh_power=1.0, body_size=None, special=None, **kwargs):
    if special=='cz': 
        return cz_system(support, nmandates, threshold=threshold, body_size=body_size, **kwargs)

    # Remove parties below a national threshold
    zero_mask = (support.sum(axis=1)/(support.sum(axis=(1,2))+1e-3)[:,None])>threshold
    uzsim_t = zero_mask[:,None,:]*support
    
    # Remove parties below an electoral_district specific threshold
    zero_mask = (support/(support.sum(axis=(2))+1e-3)[:,:,None])>ed_threshold
    uzsim_t = zero_mask[:,:,:]*uzsim_t

    # Districts with quotas, then country-level compensation (Estonian system)
    if quotas:
        quotas = (support.sum(axis=-1)+1e-3)/(nmandates[None,:])
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
# Input a tensor t and a tensor kv of one less dimension
# Output a tensor of same shape as t with k ones marking the smallest values in t over the last axis
def vec_smallest_k(t, kv):

    # Create a vector with k ones followed by zeros
    rmand = np.ones(t.shape[:-1]) * kv 
    ri = ((np.arange(t.shape[-1])[None,:]-rmand[...,None])<0) 

    # Function that maps 0 to 0, and i+1 to the i-th unit vector
    comp_ident = np.concatenate([np.zeros( (1,t.shape[-1]) ),np.identity(t.shape[-1])])

    # Marginalize that function over the newly created dimension
    return comp_ident[(np.argsort(t,axis=-1)+1)*ri].sum(axis=-2)


# %% ../nbs/04_election_models.ipynb 8
# Czech electoral system based on https://pspen.psp.cz/chamber-members/plenary/elections/#electoralsystem
def cz_system(support, nmandates, threshold=0.0, body_size=None, **kwargs):

    # Remove parties below a national threshold
    zero_mask = (support.sum(axis=1)/(support.sum(axis=(1,2))+1e-3)[:,None])>threshold
    uzsim_t = zero_mask[:,None,:]*support
    
    # Districts with quotas, then country-level compensation
    # Imperialis quotas i.e. with divisor (n_mandates + 2)
    quotas = (support.sum(axis=-1)+1e-3)/(nmandates[None,:] + 2)
    dmandates, r = np.divmod(uzsim_t/quotas[:,:,None],1.0)

    # Deal with excess allocations
    excess = np.maximum(0,(dmandates.sum(axis=-1)-nmandates[None,:]))
    rp = r+(dmandates==0) # Increase residuals to remove mandates only from those that got any
    excess_dist = vec_smallest_k(rp,excess) 
    dmandates -= excess_dist

    # Second level votes
    slvotes = ((r + excess_dist)*quotas[:,:,None]).sum(axis=1) # Margin over e_d
    remaining_mand = body_size - dmandates.sum(axis=(1,2)) # Margin over e_d and party

    # Second level quota
    slquotas = (slvotes.sum(axis=-1)+1e-3)/(remaining_mand+1)
    slmandates, r = np.divmod(slvotes/slquotas[:,None],1.0)

    # Assign all seats using highest remainders
    missing = np.maximum(0,remaining_mand-slmandates.sum(axis=-1))
    slmandates += vec_smallest_k(-r,missing)

    # Checksums to make sure all mandates get allocated
    #print(list(dmandates.sum(axis=(1,2)) + slmandates.sum(axis=1)))
    
    # Return the districts + compensation results
    return np.concatenate( [dmandates,slmandates[:,None,:]],axis=1 )

# %% ../nbs/04_election_models.ipynb 9
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

# %% ../nbs/04_election_models.ipynb 13
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

# %% ../nbs/04_election_models.ipynb 14
# This fits into the pp framework as: f0['col']=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot('mandate_plot', data_format='longform', draws=True, requires_factor=True, agg_fn='sum', n_facets=(2,2), requires=[{},{'mandates':'pass','electoral_system':'pass'}], as_is=True, priority=-500)#, hidden=True)
def mandate_plot(data, mandates, electoral_system, value_col='value', facets=[], width=None, alt_properties={}, outer_factors=[], translate=None, sim_done=False):
    f0, f1 = facets[0], facets[1]
    tf = translate if translate else (lambda s: s)
    
    if outer_factors: raise Exception("This plot does not work with extra factors")
    
    if not sim_done:
        mandates = { tf(k):v for k,v in mandates.items() }
        df = simulate_election_pp(data, mandates, electoral_system, f0['col'], value_col, f1['col'], f0['order'], f1['order'])
    else:
        df = data

    df[f1['col']] = df[f1['col']].replace({'Compensation':tf('Compensation')})
    
    # Shape it into % values for each vote count
    maxv = df['mandates'].max()
    tv = np.arange(1,maxv+1,dtype='int')[None,:]
    dfv = df['mandates'].to_numpy()[:,None]
    dfm = pd.DataFrame((dfv>=tv).astype('int'),columns=tv[0], index=df.index)
    dfm['draw'],dfm[f0['col']], dfm[f1['col']] = df['draw'], df[f0['col']], df[f1['col']]
    res = dfm.groupby([f0['col'],f1['col']],observed=True)[tv[0]].mean().reset_index().melt(id_vars=[f0['col'],f1['col']],
                                                                                var_name='mandates',value_name='percent')
    
    # Remove parties who have no chance of even one elector
    eliminate = (res.groupby(f0['col'],observed=True)['percent'].sum() < 0.2)
    el_cols = [i for i,v in eliminate.items() if v]
    res = res[~res[f0['col']].isin(el_cols)]
    cat_order = list(eliminate[~eliminate].index)
    
    f_width = max(50,width/len(cat_order))

    plot = alt.Chart(data=res).mark_bar().encode(
        x=alt.X('mandates',title=None),
        y=alt.Y('percent',title=None,axis=alt.Axis(format='%')),
        color=alt.Color(f'{f0["col"]}:N', scale=f0["colors"], legend=None),
        tooltip=[
            alt.Tooltip(f0['col'], title='party'),
            alt.Tooltip(f1['col']),
            alt.Tooltip('mandates'),
            alt.Tooltip('percent', format='.1%', title='probability'),
            ]
    ).properties(
        width=f_width,
        height=f_width//2,
        **alt_properties
        #title="Ringkonna- ja kompensatsioonimandaatide tõenäolised jaotused"
    ).facet(
        #header=alt.Header(labelAngle=-90),
        row=alt.X(
            f'{f1["col"]}:N',
            sort=f1["order"]+[tf('Compensation')],
            title=None,
            header=alt.Header(labelOrient='top')
            ),
        column=alt.Y(
            f'{f0["col"]}:N',
            sort=cat_order,
            title=None,
            header=alt.Header(labelFontWeight='bold')
            ),
    )
    return plot

# %% ../nbs/04_election_models.ipynb 17
# This fits into the pp framework as: f0['col']=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot('coalition_applet', data_format='longform', draws=True, requires_factor=True, agg_fn='sum', args={'initial_coalition':'list'},
                requires=[{},{'mandates':'pass','electoral_system':'pass'}], as_is=True, n_facets=(2,2), priority=-1000)#, hidden=True)
def coalition_applet(data, mandates, electoral_system, value_col='value', facets=[], width=None, alt_properties={}, 
                        outer_factors=[], translate=None, initial_coalition=[], sim_done=False):
    
    f0, f1 = facets[0], facets[1]
    tf = translate if translate else (lambda s: s)
    
    if outer_factors: raise Exception("This plot does not work with extra factors")

    mandates = { tf(k):v for k,v in mandates.items() }
    
    if not sim_done:
        sdf = simulate_election_pp(data, mandates, electoral_system, f0['col'], value_col, f1['col'], f0['order'], f1['order'])
    else:
        sdf = data

    # Aggregate to total mandate counts
    odf = sdf.groupby(['draw',f0['col']])['mandates'].sum().reset_index()
    odf['over_t'] = (odf['mandates']>0) 
    adf = odf[odf['mandates']>0]

    parties = list(adf[f0['col']].unique()) # Leave only parties that have mandates

    coalition = st.multiselect(tf('Select the coalition:'),
        f0["order"], default=initial_coalition,
        help=tf('Choose the parties whose coalition to model'))

    st.markdown("""___""")

    col1, col2 = st.columns((9, 9), gap='large')
    col1.markdown(tf('**Party mandate distributions**'))

    # Individual parties plot
    ddf = (adf.groupby(f0['col'])['mandates'].value_counts()/odf.groupby(f0['col']).size()).rename('percent').reset_index()
    ddf = ddf.merge(odf.groupby(f0['col'])['mandates'].median().rename(tf('median')),left_on=f0['col'],right_index=True)
    ddf = ddf.merge(odf.groupby(f0['col'])['over_t'].mean().rename(tf('over_threshold')),left_on=f0['col'],right_index=True)

    p_plot = alt.Chart(
            ddf,
            #title=var
        ).mark_rect(opacity=0.8, stroke='black', strokeWidth=0).transform_calculate(
            x1='datum.mandates - 0.45',
            x2='datum.mandates + 0.45'
        ).encode(
            alt.X('x1:Q', title=tf("mandates"), axis=alt.Axis(tickMinStep=1),scale=alt.Scale(domainMin=0)), alt.X2('x2:Q'),
            alt.Y('percent:Q', title=None, axis=None),
            alt.Row(f'{f0["col"]}:N', title=None),
            color=alt.Color(f'{f0["col"]}:N', legend=None, scale=f0["colors"]),
            tooltip=[
                alt.Tooltip('mandates:Q',title=tf('mandates'), format=',d'),
                alt.Tooltip('percent:Q',title=tf('percent'),format='.1%'),
                alt.Tooltip(tf('median'),format=',d'),
                alt.Tooltip(tf('over_threshold'),format='.1%'),
                ]
        ).properties(height=60)
    col1.altair_chart(p_plot, use_container_width=True)

    total_mandates = sum(mandates.values())

    col2.markdown(tf('**Coalition simulation**'))
    n = col2.number_input(tf('Choose mandate cutoff:'), min_value=0, max_value=total_mandates, value=(total_mandates//2) + 1, step=1, help='...')

    if len(coalition)>0:
        # Coalition plot
        acdf = adf[adf[f0['col']].isin(coalition)]
        cdf = acdf.groupby('draw')['mandates'].sum().value_counts().rename('count').reset_index()

        mi, ma = min(cdf['mandates'].min(),n), max(cdf['mandates'].max(),n)
        tick_count = (ma-mi+1) # This is the only way to enforce integer ticks as tickMinStep seems to not do it sometimes
        k_plot = alt.Chart(cdf).mark_rect(color='#ff2b2b').transform_calculate(
            x1='datum.mandates - 0.45',
            x2='datum.mandates + 0.45'
        ).encode(
            x=alt.X('x1:Q', title=tf('mandates'), axis=alt.Axis(tickMinStep=1,tickCount=tick_count), scale=alt.Scale(domain=[mi,ma])), x2=alt.X2('x2:Q'),
            y=alt.Y('count:Q', title=None, stack=None, axis=None),
        ).properties(height=200,width=300)
        rule = alt.Chart(pd.DataFrame({'x': [n]})).mark_rule(color='silver', size=1.25, strokeDash=[5, 2]).encode(x='x')
        col2.altair_chart((k_plot+rule).configure_view(strokeWidth=0), use_container_width=True)

        col2.write(tf("Probability of at least  **{0:.0f}** mandates: **{1:.1%}**").format(n, cdf[cdf['mandates'] >= n]['count'].sum()/cdf['count'].sum()))
        #col3.write('Distributsiooni mediaan: **{:d}**'.format(int((d_dist[koalitsioon].sum(1)).median())))
        #m, l, h = hdi(sim_data['riigikogu'][koalitsioon], 0.9)
        #col2.write('Distributsiooni mediaan on **{:.0f}** mandaati. 90% tõenäosusega jääb mandaatide arv **{:.0f}** ning **{:.0f}** vahele.'.format(m, l, h))
        
    return None
