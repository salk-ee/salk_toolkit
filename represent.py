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


if len(sys.argv)<3:
    print("Expecting two arguments: <data meta> <census meta>")
    sys.exit()

ncol,wcol = 'N', 'weight'

@st.cache_resource(show_spinner=False)
def load_file(pdict):
    df, dm = read_and_process_data(pdict,return_meta=True)
    return df, dm

c1, c2 = st.expander('Data description (filter & processing)').columns(2)
ddict = json.loads(c1.text_area('Data desc','{}'))
ddict['file'] = sys.argv[1]
df, dm = load_file(ddict)
cdict = json.loads(c2.text_area('Census desc','{}'))
cdict['file'] = sys.argv[2]
cdf, cm = load_file(cdict)

common_cols = [c for c in cdf.columns if c in df.columns]

c1,c2 = st.columns(2)
dim = c1.selectbox('Dimension',common_cols)

dim2 = c2.selectbox('Second Dimension',['None']+common_cols)
if dim2!='None':
    dim = [dim,dim2]


dd = (df.groupby(dim).size()/len(df)).rename('d').reset_index()
dd['cat'] = 'raw'
wd = (df.groupby(dim)[wcol].sum()/df[wcol].sum()).rename('d').reset_index() 
wd['cat'] = 'weighted'
cd = (cdf.groupby(dim)[ncol].sum()/cdf[ncol].sum()).rename('d').reset_index()
cd['cat'] = 'census'
dd.index = dd[dim]; wd.index=wd[dim]; cd.index=cd[dim]
ddf = pd.concat([dd,wd,cd])

# Calculate K-L divergences
if isinstance(dim,list):
    s_v = list(set(df[dim].drop_duplicates().itertuples(index=False)) & set(cdf[dim].drop_duplicates().itertuples(index=False)))
else: s_v = list(set(df[dim].unique()) & set(cdf[dim].unique())) 
klr = sp.stats.entropy(dd.loc[s_v,'d'],cd.loc[s_v,'d'])
klw = sp.stats.entropy(wd.loc[s_v,'d'],cd.loc[s_v,'d'])

if isinstance(dim,list): # Side-by-side Heatmaps
    chart = alt.Chart(ddf).mark_rect(size=10).encode(
        x=alt.X(f'{dim[1]}:N',title=None),
        y=alt.Y(f'{dim[0]}:N',title=None),
        color=alt.Color('d:Q',legend=None),
        column=alt.Row(f'cat:N',sort=['raw','weighted','census'],title=None)
    ).properties(width=300,height=500)

else: # Simple barplot
    
    chart = alt.Chart(ddf).mark_bar().encode(
        y=alt.Y('cat:N',sort=['raw','weighted','census'],title=None,axis=alt.Axis(labels=False)),
        x=alt.X('d:Q',title=None),
        color=alt.Color('cat:N',sort=['raw','weighted','census']),
        row=alt.Row(f'{dim}:N',title=None,header=alt.Header(labelAngle=0,labelAlign='left'))
    )

st.altair_chart(chart)
st.write(f"K-L distances: {klr:.2} raw, {klw:.2} weighted")

info.empty()

debug_m.__exit__(None,None,None)

if profile:
    p.stop()
