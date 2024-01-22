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
    import polars as pl
    import numpy as np
    import json, io, gzip, os, sys
    from collections import defaultdict

    import arviz as az
    import altair as alt
    import itertools as it
    import warnings
    import contextlib

    from pandas.api.types import is_numeric_dtype

    from salk_toolkit.io import load_parquet_with_metadata, read_json, extract_column_meta
    from salk_toolkit.pp import *
    from salk_toolkit.plots import matching_plots, get_plot_meta
    from salk_toolkit.utils import vod
    from salk_toolkit.dashboard import draw_plot_matrix, facet_ui, filter_ui, get_plot_width

    tqdm = lambda x: x # So we can freely copy-paste from notebooks

    # Disable altair schema validations by setting debug_mode = False
    # This speeds plots up considerably as altair performs an excessive amount of these validation for some reason
    dm = alt.utils.schemapi.debug_mode(False); dm.__enter__()

# Turn off annoying warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

# If none, uses the meta of the first data source
global_data_metafile = sys.argv[1] if len(sys.argv)>1 else None

if global_data_metafile:
    global_data_meta = read_json(global_data_metafile,replace_const=True)
else: global_data_meta = None

#info.info("Libraries loaded.")
#path = '../salk_internal_package/samples/'
path = './samples/'

input_file_choices = sorted([ f for f in os.listdir(path) if f[-8:]=='.parquet' ])

input_files = st.sidebar.multiselect('Select files:',input_file_choices)

########################################################################
#                                                                      #
#                       LOAD PYMC SAMPLE DATA                          #
#                                                                      #
########################################################################

@st.cache_resource(show_spinner=False)
def load_file(input_file, lazy=False):
    full_df, meta = load_parquet_with_metadata(path+input_file,lazy=lazy)
    n = full_df.select(pl.count()).collect()[0,0] if lazy else len(full_df)
    if meta is None: meta = {}
    return { 'data': full_df, 'data_n': n, 'data_meta': vod(meta,'data',None), 'model_meta': vod(meta,'model') }

if len(input_files)==0:
    st.markdown("""Please choose an input file from the sidebar""")
    st.stop()
else:
    loaded = { ifile:load_file(ifile,(i!=0)) for i,ifile in enumerate(input_files) }
    first_file = loaded[input_files[0]]
    first_data_meta = first_file['data_meta'] if global_data_meta is None else global_data_meta
    first_data = first_file['data']

########################################################################
#                                                                      #
#                            Sidebar UI                                #
#                                                                      #
########################################################################

from pandas.api.types import is_datetime64_any_dtype as is_datetime

if global_data_meta: st.sidebar.info('⚠️ External meta loaded.')

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

c_meta = extract_column_meta(first_data_meta)

with st.sidebar: #.expander("Select dimensions"):
    f_info = st.empty()
    st.markdown("""___""")

    args['poststrat'] = st.toggle('Post-stratified?', True)
    if args['poststrat']: del args['poststrat'] # True is default, so clean the dict from it in that case

    show_grouped = st.toggle('Show grouped facets', True)

    if st.toggle('Convert to continuous', False):
        args['convert_res'] = 'continuous'

    obs_dims = get_dimensions(first_data_meta, show_grouped, first_data.columns)
    obs_dims = [c for c in obs_dims if c not in first_data.columns or not is_datetime(first_data[c])]
    all_dims = get_dimensions(first_data_meta, False, first_data.columns)

    obs_name = st.selectbox('Observation', obs_dims)
    args['res_col'] = obs_name

    all_dims = vod(c_meta[obs_name],'modifiers', []) + all_dims

    facet_dims = all_dims
    if len(input_files)>1: facet_dims = ['input_file'] + facet_dims

    args['factor_cols'] = facet_ui(facet_dims,two=True)

    args['internal_facet'] = st.toggle('Internal facet?',True)

    args['plot'] = st.selectbox('Plot type',matching_plots(args, first_data, first_data_meta))

    plot_args = {}
    for k, v in vod(get_plot_meta(args['plot']),'args',{}).items():
        if v=='bool':
            plot_args[k] = st.toggle(k)
        elif isinstance(v, list):
            plot_args[k] = st.selectbox(k,v)

    args['plot_args'] = plot_args

    #detailed = st.toggle('Fine-grained filter', False)    
    #args['filter'] = filter_ui(first_data,first_data_meta,
    #                    dims=all_dims,detailed=detailed)
    args['filter'] = {}


    with st.expander('Plot desc'):
        st.json(args)



# Check if any facet dims match observation dim or each other
if len(set(args['factor_cols']+[obs_name])) != len(args['factor_cols'])+1:
    st.markdown("""Please choose facets different from observation dimension""")
    st.stop()


#left, middle, right = st.columns([2, 5, 2])
#tab = middle.radio('Tabs',['Main'],horizontal=True,label_visibility='hidden')
st.markdown("""___""")

########################################################################
#                                                                      #
#                              GRAPHS                                  #
#                                                                      #
########################################################################

# Workaround for geoplot - in that case draw multiple plots instead of a facet
matrix_form = (args['plot'] == 'geoplot')

# Create columns, one per input file
facet_dim = args['factor_cols'][0] if len(args['factor_cols'])>0 else None
if len(input_files)>1 and facet_dim != 'input_file':
    cols = st.columns(len(input_files))
else: cols = [contextlib.suppress()]

if facet_dim == 'input_file':
    #with st.spinner('Filtering data...'):
    
    # This is a bit hacky because of the lazy data frames
    dfs = []
    for ifile in input_files:
        df, fargs = loaded[ifile]['data'], args.copy()
        fargs['filter'] = { k:v for k,v in fargs['filter'].items() if k in df.columns }
        fargs['factor_cols'] = [ f for f in fargs['factor_cols'] if f!='input_file' ]
        pparams = get_filtered_data(df, first_data_meta, fargs)
        dfs.append(pparams['data'])

    fdf = pd.concat(dfs)

    # Fix categories to match the first file in case there are discrepancies (like only one being ordered)
    for c in fdf.columns:
        if fdf[c].dtype.name!='category' and dfs[0][c].dtype.name=='category':
            fdf[c] = pd.Categorical(fdf[c],dtype=dfs[0][c].dtype)

    fdf['input_file'] = pd.Categorical(
        [ v for i,f in enumerate(input_files) for v in [f]*len(dfs[i]) ],input_files)

    pparams['data'] = fdf
    plot = create_plot(pparams,first_data_meta,args,
                        width=get_plot_width('full'),
                        return_matrix_of_plots=matrix_form)

    draw_plot_matrix(plot,matrix_form=matrix_form)
    #st.altair_chart(plot)#,use_container_width=True)

else:
    # Iterate over input files
    for i, ifile in enumerate(input_files):
        with cols[i]:
            
            # Heading:
            st.header(os.path.splitext(ifile.replace('_',' '))[0])

            data_meta = loaded[ifile]['data_meta'] if global_data_meta is None else global_data_meta
            if data_meta is None: data_meta = first_data_meta

            if (args['res_col'] in first_data.columns   # I.e. it is a column, not a group
                and args['res_col'] not in loaded[ifile]['data'].columns):
                st.write(f"'{args['res_col']}' not present")
                continue

            #with st.spinner('Filtering data...'):
            fargs = args.copy()
            fargs['filter'] = { k:v for k,v in args['filter'].items() if k in loaded[ifile]['data'].columns }
            pparams = get_filtered_data(loaded[ifile]['data'], data_meta, fargs)
            plot = create_plot(pparams,data_meta,fargs,
                                width=get_plot_width(f'{i}_{ifile}'),
                                return_matrix_of_plots=matrix_form)

            #n_questions = pparams['data']['question'].nunique() if 'question' in pparams['data'] else 1
            #st.write('Based on %.1f%% of data' % (100*pparams['n_datapoints']/(len(loaded[ifile]['data_n'])*n_questions)))
            st.write('Based on %.1f%% of data' % (100*pparams['n_datapoints']/loaded[ifile]['data_n']))
            #st.altair_chart(plot)#, use_container_width=(len(input_files)>1))
            draw_plot_matrix(plot,matrix_form=matrix_form)

            with st.expander('Data Meta'):
                st.json(loaded[ifile]['data_meta'])

            with st.expander('Model Meta'):
                st.json(loaded[ifile]['model_meta'])


info.empty()

dm.__exit__(None,None,None)

if profile:
    p.stop()
