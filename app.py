import streamlit as st

profile = False
if profile:
    from streamlit_profiler import Profiler
    p = Profiler(); p.start()

st.set_page_config(
    layout="wide",
    page_title="SALK Magic Model",
    #page_icon="./s-model.png",
    initial_sidebar_state="expanded",
)

info = st.empty()

with st.spinner("Loading libraries.."):
    import pandas as pd
    import numpy as np
    import json, io, gzip, os
    from collections import defaultdict

    import arviz as az
    import altair as alt
    import itertools as it
    import warnings

    from pandas.api.types import is_numeric_dtype

    from salk_toolkit.io import load_parquet_with_metadata, read_json
    from salk_toolkit.pp import *
    from salk_toolkit.plots import matching_plots

    tqdm = lambda x: x # So we can freely copy-paste from notebooks

# Turn off annoying warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

global_data_metafile = './data/master_meta.json'

if global_data_metafile:
    global_data_meta = read_json(global_data_metafile,replace_const=True)

#info.info("Libraries loaded.")
#path = '../salk_internal_package/samples/'
path = './samples/'

input_file_choices = [ f for f in os.listdir(path) if f[-8:]=='.parquet' ]

input_files = st.sidebar.multiselect('Select files:',input_file_choices)

########################################################################
#                                                                      #
#                       LOAD PYMC SAMPLE DATA                          #
#                                                                      #
########################################################################

@st.cache_resource(show_spinner=False)
def load_file(input_file):
    full_df, meta = load_parquet_with_metadata(path+input_file)
    return { 'data': full_df, 'data_meta': meta['data'], 'model_meta': meta['model'] }

if len(input_files)==0:
    st.markdown("""Please choose an input file from the sidebar""")
    st.stop()
else:
    loaded = { ifile:load_file(ifile) for ifile in input_files }
    first_file = loaded[input_files[0]]
    data_meta = first_file['data_meta'] if global_data_meta is None else global_data_meta
    first_data = first_file['data']

########################################################################
#                                                                      #
#                            Sidebar UI                                #
#                                                                      #
########################################################################

from salk_toolkit.utils import vod
def get_dimensions(data_meta, observations=True, whitelist=None):
    res = []
    for g in data_meta['structure']:
        if vod(g,'hidden'): continue
        if 'scale' in g and observations:
            res.append(g['name'])
        else:
            cols = [ (c if isinstance(c,str) else c[0]) for c in g['columns']]
            if whitelist is not None: cols = [ c for c in cols if c in whitelist ]
            res += cols
    return res

args = {}

with st.sidebar: #.expander("Select dimensions"):
    f_info = st.empty()
    st.markdown("""___""")

    obs_dims = get_dimensions(data_meta,True,first_data.columns)
    all_dims = get_dimensions(data_meta,False,first_data.columns)

    obs_name = st.selectbox('Observation', obs_dims)
    args['res_col'] = obs_name

    facet_dims = all_dims
    if len(input_files)>1: facet_dims = ['input_file'] + facet_dims

    facet_dim = st.sidebar.selectbox('Facet:', ['None'] + facet_dims)

    args['factor_cols'] = [facet_dim] if facet_dim != 'None' else []

    if facet_dim != 'None':
        second_dim = st.sidebar.selectbox('Facet 2:', ['None'] + all_dims)
        if second_dim != 'None':  args['factor_cols'] = [facet_dim, second_dim]

    args['plot_args'] = { 'internal_facet': st.checkbox('Internal facet?',True) }

    args['plot'] = st.selectbox('Plot type',matching_plots(args, first_data, data_meta))

    with st.sidebar.expander('Filters'):
        filter_vals = { col: list(first_data[col].unique()) for col in all_dims if col in first_data.columns }
        filters = {}

        for cn in all_dims:
            col = first_data[cn]
            if col.dtype.name=='category' and not col.dtype.ordered:
                filters[cn] = st.selectbox(cn, ['All'] + list(col.dtype.categories))
            elif col.dtype.name=='category':
                cats = col.dtype.categories
                filters[cn] = st.select_slider(cn,cats,value=(cats[0],cats[-1]))
            elif col.dtype!='bool':
                mima = (col.min(),col.max())
                filters[cn] = st.slider(cn,*mima,value=mima)

        args['filter'] = { k:v for k,v in filters.items() if v != 'All'}

        #print(args['filter'])


    x='''
    gcols = group_columns_dict(data_meta)
    for g in data_meta['structure']:
        with st.sidebar.expander(g['name']):
            for cn in gcols[g['name']]:
                if cn not in all_dims: continue
                col = first_data[cn]
                if col.dtype.name=='category' and not col.dtype.ordered:
                    filters[cn] = st.selectbox(cn, ['All'] + list(col.dtype.categories))
                elif col.dtype.name=='category':
                    cats = col.dtype.categories
                    filters[cn] = st.select_slider(cn,cats,value=(cats[0],cats[-1]))
                elif col.dtype!='bool':
                    mima = (col.min(),col.max())
                    filters[cn] = st.slider(cn,*mima,value=mima)
    '''


#left, middle, right = st.columns([2, 5, 2])
#tab = middle.radio('Tabs',['Main'],horizontal=True,label_visibility='hidden')
st.markdown("""___""")

########################################################################
#                                                                      #
#                              GRAPHS                                  #
#                                                                      #
########################################################################


# Create columns, one per input file
if len(input_files)>1 and facet_dim != 'input_file':
    cols = st.columns(len(input_files))
else: cols = [st]

if facet_dim == 'input_file':
    with st.spinner('Filtering data...'):
        dfs = []
        for ifile in input_files:
            df = get_filtered_data(loaded[ifile]['data'], data_meta, **args)
            df['input_file'] = ifile
            dfs.append(df)
        fdf = pd.concat(dfs)
        fdf['input_file'] = pd.Categorical(fdf['input_file'],input_files)
        plot = create_plot(fdf,data_meta,alt_properties={'width':800},**args)

    st.altair_chart(plot,use_container_width=True)

else:
    # Iterate over input files
    for i, ifile in enumerate(input_files):

        # Heading:
        cols[i].header(os.path.splitext(ifile.replace('_',' '))[0])

        data_meta = loaded[ifile]['data_meta'] if global_data_meta is None else global_data_meta

        with st.spinner('Filtering data...'):
            fdf = get_filtered_data(loaded[ifile]['data'], data_meta, **args)
            plot = create_plot(fdf,data_meta,alt_properties={'width':800},**args)

        cols[i].altair_chart(plot,
            use_container_width=(len(input_files)>1)
            )

        with st.expander('Data Meta'):
            st.json(loaded[ifile]['data_meta'])

        with st.expander('Model Meta'):
            st.json(loaded[ifile]['model_meta'])


st.markdown("""***""")
st.caption('Andmed & teostus: **SALK 2023**')
info.empty()

if profile:
    p.stop()
