from re import S
import streamlit as st
from salk_toolkit.io import read_annotated_data
from streamlit_dimensions import st_dimensions
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# If true, the profiler will be shown
profile = False
memprofile = False

if memprofile:
    import tracemalloc
    tracemalloc.start()

if profile:
    from wfork_streamlit_profiler import Profiler
    p = Profiler(); p.start()

st.set_page_config(
    layout="wide",
    page_title="SALK Explorer",
    #page_icon="./s-model.png",
    initial_sidebar_state="expanded",
)

info = st.empty()

with st.spinner("Loading libraries.."):
    import pandas as pd
    import polars as pl
    import numpy as np
    import json, io, gzip, os, sys, psutil
    from collections import defaultdict

    import arviz as az
    import altair as alt
    import itertools as it
    import warnings
    import contextlib

    from pandas.api.types import is_numeric_dtype
    from streamlit_js import st_js, st_js_blocking

    from salk_toolkit.io import read_json, extract_column_meta, read_annotated_data_lazy
    from salk_toolkit.pp import *
    from salk_toolkit.utils import *
    from salk_toolkit.dashboard import draw_plot_matrix, plot_matrix_html, facet_ui, filter_ui, get_plot_width, default_translate, stss_safety
    from copy import deepcopy

    tqdm = lambda x: x # So we can freely copy-paste from notebooks

    # Disable altair schema validations by setting debug_mode = False
    # This speeds plots up considerably as altair performs an excessive amount of these validation for some reason
    dm = alt.utils.schemapi.debug_mode(False); dm.__enter__()

    # Disable altair max rows
    alt.data_transformers.disable_max_rows()

# Override the st_dimensions based version that can cause refresh loops
def get_plot_width(str,ncols=1):
    return min(800,1200/ncols)


if 'ls_loaded' not in st.session_state:
    try:
        ls_state = json.loads(st_js_blocking(f'return localStorage.getItem("session_state")') or '{}')
    except json.JSONDecodeError as e:
        ls_state = {}
    
    for k, v in ls_state.items(): st.session_state[k] = v
    st.session_state['ls_loaded'] = True

# Turn off annoying warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)


cl_args = sys.argv[1:] if len(sys.argv)>1 else []
if len(cl_args)>0 and cl_args[0].endswith('.json'):
    global_data_meta = read_json(cl_args[0],replace_const=True)
    cl_args = cl_args[1:]
else: global_data_meta = None

translate = default_translate
path = './samples/' 
paths = defaultdict( lambda: path )

# Add command line inputs as default input files
default_inputs = []
for fname in cl_args:
    path, fname = os.path.split(fname)
    if fname in paths: # Duplicate file name: include path
        p1, p2 = os.path.split(path)
        path, fname = p1, os.path.join(p2,fname)
    paths[fname] = (path or '.')+'/' 
    default_inputs.append(fname)

if not path: path = './'
else: path += '/'

input_file_choices = default_inputs + sorted([ f for f in os.listdir(path) if f[-8:]=='.parquet' ])

input_files = st.sidebar.multiselect('Select files:',input_file_choices,default_inputs)

########################################################################
#                                                                      #
#                       LOAD PYMC SAMPLE DATA                          #
#                                                                      #
########################################################################


@st.cache_resource(show_spinner=False)
def load_file(input_file):
    pl.enable_string_cache()
    ifile = paths[input_file]+input_file
    ldf, dmeta, mmeta = read_annotated_data_lazy(ifile, return_model_meta=True)
    columns = ldf.collect_schema().names()        
    n0 = ldf.select(pl.len()).collect().item()
    n = dmeta.get('total_size', n0) # N0 is count of rows which is a fallback for older versions
    if dmeta is None: dmeta = {}
    return { 'data': ldf, 'total_size': n, 'data_meta': dmeta, 'model_meta': mmeta, 'columns': columns }

if len(input_files)==0:
    st.markdown("""Please choose an input file from the sidebar""")
    st.stop()
else:
    loaded = { ifile:load_file(ifile) for i,ifile in enumerate(input_files) }
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

def get_dimensions(data_meta, present_cols, observations=True):
    c_meta = extract_column_meta(data_meta)
    res = []
    for g in data_meta['structure']:
        if g.get('hidden'): continue
        if ('scale' in g and observations and
            (set(c_meta[g['name']]['columns']) & set(present_cols))):
            res.append(g['name'])
        else:
            cols = [ c for c in c_meta[g['name']]['columns']]
            cols = [ c for c in cols if c in present_cols ]
            res += cols
    return res

args = {}

# If we have an override, add the column block to the data meta
raw_first_data_meta = deepcopy(first_data_meta)
c_meta = extract_column_meta(first_data_meta)
if st.session_state.get('override'):
    try:
        pp = eval(st.session_state['override'])
        c_meta, _ = update_data_meta_with_pp_desc(first_data_meta, pp)
    except Exception as e:
        st.error(f"Error parsing override: {e}")

with st.sidebar: #.expander("Select dimensions"):

    f_info = st.empty()
    st.markdown("""___""")

    # Reset button - has to be high up in case something fails to load
    if st.sidebar.button('Reset choices'): 
        st_js_blocking('localStorage.removeItem("session_state")')
        st.session_state.clear()

    draw = st.toggle('Draw plots',True)

    show_grouped = st.toggle('Show grouped facets', True)

    if st.toggle('Convert to continuous', False):
        args['convert_res'] = 'continuous'

    schema = first_data.collect_schema()
    all_cols = list(schema.names())
    
    obs_dims = get_dimensions(first_data_meta, all_cols, show_grouped)
    obs_dims = [c for c in obs_dims if c not in first_data or not schema[c].is_temporal()]
    all_dims = get_dimensions(first_data_meta, all_cols, False)

    # Deduplicate them - this bypasses some issues sometimes
    obs_dims = list(dict.fromkeys(obs_dims))
    all_dims = list(dict.fromkeys(all_dims))

    stss_safety('observation',obs_dims)
    obs_name = st.selectbox('Observation', obs_dims, key='observation')
    args['res_col'] = obs_name

    res_cont = not c_meta[args['res_col']].get('categories') or args.get('convert_res') == 'continuous'

    all_dims = c_meta[obs_name].get('modifiers', []) + all_dims

    facet_dims = all_dims
    if len(input_files)>1: facet_dims = ['input_file'] + facet_dims

    args['factor_cols'] = facet_ui(facet_dims,two=True)

    # Check if any facet dims match observation dim or each other
    if len(set(args['factor_cols']+[obs_name])) != len(args['factor_cols'])+1:
        st.markdown("""Please choose facets different from observation dimension""")
        st.stop()

    args['internal_facet'] = st.toggle('Internal facet?',True,key='internal')
    sort = st.toggle('Sort facets',False,key='sort')

    # Make all dimensions explicit
    args['factor_cols'] = impute_factor_cols(args, c_meta)

    # Plot type
    matching = matching_plots(args, first_data, first_data_meta)
    plot_list = ['default'] + sorted(matching)
    if 'plot_type' in st.session_state:
        if st.session_state['plot_type'] not in matching: st.session_state['plot_type']='default'
        pt_ind = plot_list.index(st.session_state['plot_type'])
    else: pt_ind = 0
    args['plot'] = st.session_state['plot_type'] = st.selectbox(
        'Plot type', plot_list, index=pt_ind, format_func=lambda s: f'{matching[0]} (default)' if s == 'default' else s)
    if args['plot'] == 'default': args['plot'] = matching[0]

    plot_meta = get_plot_meta(args['plot'])

    # Plot arguments
    plot_args = {} # 'n_facet_cols':2 }
    for k, t in plot_meta.get('args',{}).items():
        vt, dv = t if isinstance(t, tuple) else (t, None)
        if vt=='bool':
            plot_args[k] = st.toggle(k,key=k,value=dv)
        elif vt=='int':
            plot_args[k] = st.number_input(k,key=k,value=(dv or 0))
        elif isinstance(vt, list):
            stss_safety(k,vt)
            ind = (vt.index(dv) if dv in vt else 0)
            plot_args[k] = st.selectbox(k,vt,key=k,index=ind)

    args['plot_args'] = {**args.get('plot_args',{}),**plot_args}

    with st.expander('Advanced'):
        args['poststrat'] = st.toggle('Post-stratified?', True,key='poststrat')
        if args['poststrat']: del args['poststrat'] # True is default, so clean the dict from it in that case

        detailed = st.toggle('Fine-grained filter', False,key='fine_grained')    

        if res_cont: # Extra settings for continuous data 
            cont_transform = st.selectbox('Transform', ['None'] + cont_transform_options, key='transform')
            if cont_transform != 'None': args['cont_transform'] = cont_transform
            agg_fn = st.selectbox('Aggregation', ['mean', 'median', 'sum'], key='aggregation')
            if agg_fn!='mean': args['agg_fn'] = agg_fn

        sortable = args['factor_cols']
        if plot_meta.get('sort_numeric_first_facet'): sortable = sortable[1:]
        if sort and len(sortable)>0:
            stss_safety('sortby',sortable)
            sort_facet = st.selectbox('Sort by', sortable, 0 ,key='sortby')
            ascending = st.toggle('Ascending', False,key='sort_ascending')
            args['sort'] = {sort_facet: ascending}

        qpos = st.selectbox('Question position',['Auto',1,2,3],key='q_pos')
        if 'question' in args['factor_cols'] and qpos!='Auto':
            args['factor_cols'] = [ c for c in args['factor_cols'] if c != 'question']
            args['factor_cols'].insert(qpos-1,'question')

        override = st.text_area('Override keys','{}',key='override')
        if override: args.update(eval(override))
    
    args['filter'] = filter_ui(first_data,first_data_meta,
                                dims=all_dims,detailed=detailed)

    # Export options
    with st.expander('Export'):
        width = None
        # Toggle export options because generating them is slow
        export = st.toggle('Show export options',value=False)
        if export:
            custom_width = st.toggle('Custom width',value=False)
            if custom_width:
                width = st.slider('Width',value=800,min_value=100,max_value=1200,step=50)
            st.subheader('Export')
            export_ct = st.container()
    
    #print(list(st.session_state.keys()))

    #print(f"localStorage.setItem('args','{json.dumps(args)}');")
    st_js(f"localStorage.setItem('session_state','{json.dumps(dict(st.session_state)).replace('\'','\\\'')}');")

    # Make all dimensions explicit now that plot is selected (as that can affect the factor columns)
    args['factor_cols'] = impute_factor_cols(args, c_meta, plot_meta)

    import pprint
    with st.expander('Plot desc'):
        st.code(pprint.pformat(args,indent=0,width=30))


#left, middle, right = st.columns([2, 5, 2])
#tab = middle.radio('Tabs',['Main'],horizontal=True,label_visibility='hidden')
st.markdown("""___""")

########################################################################
#                                                                      #
#                              GRAPHS                                  #
#                                                                      #
########################################################################

# Workaround for geoplot - in that case draw multiple plots instead of a facet
matrix_form = False#(args['plot'] == 'geoplot')

# Determine if one of the facets is input_file
input_files_facet = 'input_file' in args.get('factor_cols',[])

# Create columns, one per input file
if not input_files_facet:
    cols = st.columns(len(input_files))
else: cols = [contextlib.suppress()]

if not draw:
    st.text("Plot drawing disabled for refresh speed")
elif input_files_facet:
    #with st.spinner('Filtering data...'):
    
    # This is a bit hacky because of previous use of the lazy data frames
    dfs = []
    for ifile in input_files:
        df, fargs = loaded[ifile]['data'], args.copy()
        fargs['filter'] = { k:v for k,v in fargs['filter'].items() if k in loaded[ifile]['columns'] }
        fargs['factor_cols'] = [ f for f in fargs['factor_cols'] if f!='input_file' ]
        pparams = pp_transform_data(df, raw_first_data_meta, fargs)
        dfs.append(pparams['data'])

    fdf = pd.concat(dfs)

    # Fix categories to match the first file in case there are discrepancies (like only one being ordered)
    for c in fdf.columns:
        if fdf[c].dtype.name!='category' and dfs[0][c].dtype.name=='category':
            fdf[c] = pd.Categorical(fdf[c],dtype=dfs[0][c].dtype)

    fdf['input_file'] = pd.Categorical(
        [ v for i,f in enumerate(input_files) for v in [f]*len(dfs[i]) ],input_files)

    pparams['data'] = fdf
    plot = create_plot(pparams,args,
                       translate=translate,
                       width=(width or get_plot_width('full',1)),
                       return_matrix_of_plots=matrix_form)

    draw_plot_matrix(plot)
    #st.altair_chart(plot)#,use_container_width=True)

else:
    # Iterate over input files
    for i, ifile in enumerate(input_files):
        with cols[i]:
            
            # Heading:
            st.header(os.path.splitext(ifile.replace('_',' '))[0])

            data_meta = loaded[ifile]['data_meta'] if global_data_meta is None else global_data_meta
            if data_meta is None: data_meta = raw_first_data_meta

            if (args['res_col'] in all_cols   # I.e. it is a column, not a group
                and args['res_col'] not in loaded[ifile]['columns']):
                st.write(f"'{args['res_col']}' not present")
                continue

            #with st.spinner('Filtering data...'):
            fargs = args.copy()
            fargs['filter'] = { k:v for k,v in args['filter'].items() if k in loaded[ifile]['columns'] }
            pparams = pp_transform_data(loaded[ifile]['data'], data_meta, fargs)
            cur_width = width or get_plot_width(f'{i}_{ifile}',len(input_files))
            plot = create_plot(pparams,fargs,
                               translate=translate,
                               width=cur_width,
                               return_matrix_of_plots=matrix_form)

            # Add export buttons for first data source
            if export and i==0:
                name = f'{args['res_col']}_{"_".join(args["factor_cols"]) if args["factor_cols"] else "all"}'
                c1,c2 = export_ct.columns(2)
                c1.download_button('HTML', plot_matrix_html(plot, uid=name, width=cur_width, responsive=not custom_width), f'{name}.html')
                c2.download_button('Data CSV', pparams['data'].to_csv().encode("utf-8"), f'{name}.csv')

            #n_questions = pparams['data']['question'].nunique() if 'question' in pparams['data'] else 1
            #st.write('Based on %.1f%% of data' % (100*pparams['n_datapoints']/(len(loaded[ifile]['data_n'])*n_questions)))
            st.write('Based on %.1f%% of data' % (100*pparams['filtered_size']/loaded[ifile]['total_size']))
            #st.altair_chart(plot)#, use_container_width=(len(input_files)>1))
            draw_plot_matrix(plot)

            with st.expander('Data Meta'):
                st.json(loaded[ifile]['data_meta'])

            mdl = loaded[ifile]['model_meta'].copy()

            if 'sequence' in mdl:
                steps = { m['name']: m for m in mdl['sequence'] }
                del mdl['sequence']
            else: steps = {}
            steps['main_model'] = mdl
            with st.expander('Model'):
                step_name = st.selectbox('Show:', list(steps.keys()), len(steps)-1,key='mdlshow_'+ifile)
                st.json(steps[step_name])

info.empty()

st.sidebar.write("Mem: %.1f" % (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

dm.__exit__(None,None,None)

if memprofile:
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('filename')
    memres = str(tracemalloc.get_traced_memory())+'\n'
    memres += "[ Top 10 ]\n"
    for stat in top_stats[:10]:
        memres += str(stat)+'\n'
    #print(memres)
    st.code(memres)
    tracemalloc.stop()

if profile:
    p.stop()
