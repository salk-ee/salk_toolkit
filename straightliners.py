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
    page_title="Straightliners tool",
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

    from salk_toolkit.io import *
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


#@st.cache_resource(show_spinner=False)
def load_file(pname):
    df, dm = read_annotated_data(pname, ignore_excluded=True)
    return df, dm

fname = sys.argv[1]

df, meta = load_file(fname)

c_meta = extract_column_meta(meta)

st.title("Straightliners by blocks")

ndf = pd.DataFrame(index=df['original_inds'])
ndf['ind'] = ndf.index 
ndf['score'] = 0
for gn, g in c_meta.items():
    # only look at groups >=3
    if 'columns' not in g or len(g['columns'])<3 not in g: continue
    cols = g['columns']

    cats = df[cols[0]].dtype.categories if 'categories' in g else df[cols[0]].unique()
    if 'categories' not in g and len(cats)>100: continue  # Ignore actually continuous values

    cname = f'{gn} ({len(cols)})'
    ndf[cname] = ''
    n_total = 0
    for i,c in enumerate(cats):
        vals = (df[cols]==c).all(axis=1)
        penalty = (np.log(len(cols)) *  # no. questions in block is the main effect
                    (0.8 if c in g.get('nonresponses',[]) else 1) * # reduce to 0.8 if coded as a nonresponse ('I don't know')
                    (1+0.01*i)) # Small effect for category order to keep same answers close together
        ndf['score'] += vals*penalty
        ndf.loc[vals,cname] = c
        n_total += vals.sum()
    if n_total==0: ndf.drop(columns=[cname],inplace=True)

ndf.sort_values('score',ascending=False, inplace=True)
ndf['score'] = ndf['score'].round(1) 

dcols = list(ndf.columns)
ndf['excluded'] = False
ndf['exclusion_reason'] = ''

for ind,reason in meta.get('excluded',[]):
    ndf.loc[ind,'excluded'] = True
    ndf.loc[ind,'exclusion_reason'] = reason

ndf = st.data_editor(ndf, disabled=dcols, hide_index=True)

meta['excluded'] = [ [i,r] for i,r in ndf.loc[ndf['excluded'],'exclusion_reason'].items() ]

with st.expander('Excluded block (to copy to meta)'):

    res_str = '"excluded": [\n' + ''.join([ f'  [{i},"{r}"],\n' for i,r in ndf.loc[ndf['excluded'],'exclusion_reason'].items() ]) +']'
    st.code(res_str)#json.dumps(meta['excluded']))#,indent=2))

# Rewriting json fucks up formatting. So don't do that
# if fname.endswith('.json'):
#     with open(fname,'w') as jf:
#         json.dump(meta,jf,indent=2)

info.empty()

debug_m.__exit__(None,None,None)

if profile:
    p.stop()