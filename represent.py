from re import S
import streamlit as st
from streamlit_dimensions import st_dimensions
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

profile = False
if profile:
    from streamlit_profiler import Profiler
    p = Profiler(); p.start()

st.set_page_config(
    layout="wide",
    page_title="SALK Explorer",
    #page_icon="./s-model.png",
    initial_sidebar_state="expanded",
)

def get_plot_width(str):
    return 800

info = st.empty()

with st.spinner("Loading libraries.."):
    import pandas as pd
    import numpy as np
    import scipy as sp
    import json, io, gzip, os, sys
    from collections import defaultdict

    import arviz as az
    import altair as alt
    import itertools as it
    import warnings
    import contextlib

    from pandas.api.types import is_numeric_dtype
    from streamlit_js import st_js, st_js_blocking

    from salk_toolkit.io import load_parquet_with_metadata, read_json, extract_column_meta, read_and_process_data
    from salk_toolkit.pp import *
    from salk_toolkit.utils import *
    from salk_toolkit.dashboard import draw_plot_matrix, facet_ui, filter_ui, get_plot_width, default_translate, stss_safety

    tqdm = lambda x: x # So we can freely copy-paste from notebooks

    # Disable altair schema validations by setting debug_mode = False
    # This speeds plots up considerably as altair performs an excessive amount of these validation for some reason
    debug_m = alt.utils.schemapi.debug_mode(False); debug_m.__enter__()

# Turn off annoying warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

path = sys.argv[1] if len(sys.argv)>1 else '.'

ncol,wcol = 'N', 'weight'

@st.cache_resource(show_spinner=False)
def load_file(pdict):
    df, dm = read_and_process_data(pdict,return_meta=True)
    return df, dm

mst = st.container()

dd_e = st.expander('Data & Population sections from model',expanded=True)
ddict = eval(dd_e.text_area('Data desc','{}'))

if 'data' not in ddict or 'population' not in ddict:
    st.write("Please copy the 'data' and 'population' sections of the model")
    st.stop()

ddict['data']['file'] = os.path.join(path,ddict['data']['file'])
ddict['population']['file'] = os.path.join(path,ddict['population']['file'])
df, dm = load_file(ddict['data'])
cdf, cm = load_file(ddict['population'])

common_cols = [c for c in cdf.columns if c in df.columns]

# Get marginal distributions
def get_dists(dim,div_dim=[]):
    gbs = df.groupby(div_dim).size() if div_dim else len(df)
    dd = (df.groupby(dim+div_dim).size()/gbs).rename('d').reset_index()
    dd['cat'] = 'raw'
    wd = (df.groupby(dim+div_dim)[wcol].sum()/gb_in(df,div_dim)[wcol].sum()).rename('d').reset_index() 
    wd['cat'] = 'weighted'
    cd = (cdf.groupby(dim)[ncol].sum()/cdf[ncol].sum()).rename('d').reset_index()
    cd['cat'] = 'census'
    dd.index = dd[dim]; wd.index=wd[dim]; cd.index=cd[dim]
    return dd,wd,cd

# Calculate K-L divergences
def kl_divergence(dim):
    dd, wd, cd = get_dists(dim)    
    #if isinstance(dim,list):
    s_v = list(set(df[dim].drop_duplicates().itertuples(index=False)) & set(cdf[dim].drop_duplicates().itertuples(index=False)))
    #else: s_v = list(set(df[dim].unique()) & set(cdf[dim].unique())) 
    klr = sp.stats.entropy(dd.loc[s_v,'d'],cd.loc[s_v,'d'])
    klw = sp.stats.entropy(wd.loc[s_v,'d'],cd.loc[s_v,'d'])
    return klr, klw


with st.sidebar:
    page = st.selectbox('Page',['Overview','Margins'])

    if page=='Margins':
        dim = [st.selectbox('Dimension',common_cols)]
        dim2 = st.selectbox('Second dimension',['None']+common_cols)
        if dim2!='None':  dim += [dim2]

        split_dim = st.selectbox('Split dimension',['None']+list(df.columns))
        sdl = [split_dim] if split_dim!='None' else []

if page=='Overview':

    c1,c2,c3 = mst.columns(3)

    table = []
    kl_overall = kl_divergence(common_cols)
    table.append(('Full', kl_overall[0], kl_overall[1]))
    for c in common_cols:
        kl_c = kl_divergence([c])
        table.append((c, kl_c[0], kl_c[1]))
    sdf = pd.DataFrame(table,columns=['Dimension','Raw KL', 'Weighted KL'])
    c1.dataframe(sdf.sort_values('Raw KL'),hide_index=True)

    t2 = []
    for op in it.product(common_cols,common_cols):
        p = list(op) if op[0]!=op[1] else [op[0]]
        t2.append( op + kl_divergence(p) )

    pdf = pd.DataFrame(t2,columns=['d1','d2','Raw KL', 'Weighted KL'])

    chart = alt.Chart(pdf).mark_rect(size=10).encode(
        x=alt.X(f'd1:N',title=None),
        y=alt.Y(f'd2:N',title=None),
        color=alt.Color('Raw KL:Q',legend=None),
        tooltip = [alt.Tooltip('d1',title='Dimension 1'),alt.Tooltip('d2',title='Dimension 2'),alt.Tooltip('Raw KL',format='.4f'),alt.Tooltip('Weighted KL',format='.4f')]
    ).properties(width=400,height=400)
    c2.altair_chart(chart)

        


elif page=='Margins':

    dd, wd, cd = get_dists(dim,sdl)
    if split_dim!='None':
        cats = df[split_dim].unique() 
        cvals = np.repeat(cats,len(cd))
        cd = pd.concat([cd]*len(cats))
        cd[split_dim] = cvals

    ddf = pd.concat([dd,wd,cd])

    if len(dim)>1: # Side-by-side Heatmaps
        chart = alt.Chart(ddf).mark_rect(size=10).encode(
            x=alt.X(f'{dim[1]}:N',title=None),
            y=alt.Y(f'{dim[0]}:N',title=None),
            color=alt.Color('d:Q',legend=None),
            column=alt.Column(f'cat:N',sort=['raw','weighted','census'],title=None),
            **({ 'row':alt.Column(f'{split_dim}:N',title=None) } if split_dim!='None' else {}),
            tooltip = [alt.Tooltip(dim[0],title='Value 1'),alt.Tooltip(dim[1],title='Value 2'),alt.Tooltip('d',title='Probability',format='.1%')]
        ).properties(width=300)

    else: # Simple barplot
        st.write(split_dim)
        chart = alt.Chart(ddf).mark_bar().encode(
            y=alt.Y('cat:N',sort=['raw','weighted','census'],title=None,axis=alt.Axis(labels=False)),
            x=alt.X('d:Q',title=None),
            color=alt.Color('cat:N',sort=['raw','weighted','census'],title='Source'),
            row=alt.Row(f'{dim[0]}:N',title=None,header=alt.Header(labelAngle=0,labelAlign='left')),
            **({ 'column': alt.Column(f'{split_dim}:N') } if split_dim!='None' else {}),
            tooltip = [alt.Tooltip('cat',title='Source'),alt.Tooltip(dim[0],title='Value'),alt.Tooltip('d',title='Probability',format='.1%')]
        ) 

    mst.altair_chart(chart)

info.empty()

debug_m.__exit__(None,None,None)

if profile:
    p.stop()