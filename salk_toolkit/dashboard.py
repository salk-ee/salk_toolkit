# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_dashboard.ipynb.

# %% auto 0
__all__ = ['get_plot_width', 'open_fn', 'exists_fn', 'read_annotated_data_cached', 'load_json', 'load_json_cached', 'save_json',
           'alias_file', 'default_translate', 'SalkDashboardBuilder', 'UserAuthenticationManager', 'draw_plot_matrix',
           'st_plot', 'facet_ui', 'filter_ui', 'translate_with_dict', 'log_missing_translations',
           'clean_missing_translations', 'add_missing_to_dict']

# %% ../nbs/05_dashboard.ipynb 3
import json, os, csv, re, time
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

import altair as alt
import s3fs, polib
import __main__ # to get name of py file

from pandas.api.types import is_numeric_dtype

from salk_toolkit.utils import *
from salk_toolkit.io import *
from salk_toolkit.pp import e2e_plot

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_dimensions import st_dimensions
import streamlit_authenticator as stauth

# %% ../nbs/05_dashboard.ipynb 4
def get_plot_width(key):
    wobj = st_dimensions(key=key) or { 'width': 800 } # Can return none so handle that
    return min(800,int(0.85*wobj['width'])) # Needs to be adjusted down  to leave margin for plots

# %% ../nbs/05_dashboard.ipynb 5
# Open either a local or an s3 file
def open_fn(fname, *args, s3_fs=None, **kwargs):
    if fname[:3] == 's3:':
        if s3_fs is None: s3_fs = s3fs.S3FileSystem(anon=False)
        return s3_fs.open(fname,*args,**kwargs)
    else:
        return open(fname,*args,**kwargs)
    
def exists_fn(fname, *args, s3_fs=None, **kwargs):
    if fname[:3] == 's3:':
        if s3_fs is None: s3_fs = s3fs.S3FileSystem(anon=False)
        return s3_fs.exists(fname,*args,**kwargs)
    else:
        return os.path.exists(fname,*args,**kwargs)

# %% ../nbs/05_dashboard.ipynb 6
# ttl=None - never expire. Makes sense for potentially big data files
@st.cache_resource(show_spinner=False,ttl=None)
def read_annotated_data_cached(data_source,**kwargs):
    return read_annotated_data(data_source,**kwargs)

# Load json uncached - useful for admin pages
def load_json(fname, _s3_fs=None, **kwargs):
    with open_fn(fname,'r',s3_fs=_s3_fs,encoding='utf8') as jf:
        return json.load(jf)

# This is cached very short term (1 minute) to avoid downloading it on every page change
# while still allowing users to be added / changed relatively responsively
@st.cache_resource(show_spinner=False,ttl=60)
def load_json_cached(fname, _s3_fs=None, **kwargs):
    return load_json(fname,_s3_fs,**kwargs)

# For saving json back 
def save_json(d, fname, _s3_fs=None, **kwargs):
    with open_fn(fname,'w',s3_fs=_s3_fs,encoding='utf8') as jf:
        json.dump(d,jf,indent=2,ensure_ascii=False)
        
def alias_file(fname, file_map):
    if fname[:3]!='s3:' and fname in file_map and not os.path.exists(fname):
        #print(f"Redirecting {fname} to {file_map[fname]}")
        return file_map[fname]
    else: return fname

# %% ../nbs/05_dashboard.ipynb 7
def log_event(event, username, path, s3_fs=None):
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%d-%m-%Y, %H:%M:%S')
    with open_fn(path,'a',s3_fs=s3_fs) as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event, username])

# %% ../nbs/05_dashboard.ipynb 8
# wrap the first parameter of streamlit function with self.translate
# has to be a separate function instead of in a for loop for scoping reasons
def wrap_st_with_translate(fn,self):
    func = getattr(st,fn)
    setattr(self, fn, lambda s, *args, **kwargs: func(self.tf(s),*args,**kwargs) )

# %% ../nbs/05_dashboard.ipynb 9
def default_translate(s,**kwargs):
    return (s[0].upper() + s[1:]).replace('_',' ') if isinstance(s,str) else s

# %% ../nbs/05_dashboard.ipynb 10
def po_template_updater(pot_file = None):
    if pot_file is None:
        bname = os.path.splitext(os.path.basename(__main__.__file__))[0]
        pot_file = f'locale/{bname}.pot'

    if os.path.exists(pot_file):
        po  = polib.pofile(pot_file)
        td = { entry.msgid for entry in po }
    else:
        po = polib.POFile()
        po.metadata = {
            'Project-Id-Version': '1.0',
            'Report-Msgid-Bugs-To': 'tarmo@salk.com',
            'MIME-Version': '1.0',
            'Content-Type': 'text/plain; charset=utf-8',
            'Content-Transfer-Encoding': '8bit',
        }
        td = set()

    def translate(s,**kwargs):
        if isinstance(s,str) and s not in td:
            po.append(polib.POEntry(msgid=s,msgstr=default_translate(s), 
                                    **{'msgctxt': kwargs.get('context'), 'comment': kwargs.get('comment')}))
            po.save(pot_file)
            td.add(s)
        return s
    
    return translate

def translate_fn_from_po(po_file):
    po = polib.pofile(po_file)
    td = { entry.msgid: entry.msgstr for entry in po }
    return lambda s, **kwargs: td.get(s,s)

def load_translate(translate):

    if translate is None: return default_translate
    elif callable(translate): return translate
    elif isinstance(translate,dict): return lambda s, **kwargs: translate.get(s,s)
    elif isinstance(translate,str):
        if os.path.exists(translate):
            ext = os.path.splitext(translate)[1]
            if ext == '.po' or ext == '.pot':
                return translate_fn_from_po(translate)
            elif ext == '.json':
                td = load_json_cached(translate)
                return lambda s, **kwargs: td.get(s,s)
            else:
                raise ValueError(f"Unknown translation file type: {ext}")
        elif len(translate)==2: # country code
            bname = os.path.splitext(os.path.basename(__main__.__file__))[0]
            return translate_fn_from_po(f'locale/{translate}/{bname}.po')
        else:
            raise ValueError(f"Translation file not found: {translate}")


# %% ../nbs/05_dashboard.ipynb 11
# Main dashboard wrapper - WIP
class SalkDashboardBuilder:

    def __init__(self, data_source, auth_conf, logfile, groups=['guest','user','admin'], public=False, translate=None):
        
        # Allow deployment.json to redirect files from local to s3 if local missing (i.e. in deployment scenario)
        if os.path.exists('./deployment.json'):
            dep_meta = load_json_cached('./deployment.json')
            self.filemap = vod(dep_meta,'files',{})
            data_source = alias_file(data_source,self.filemap)
            auth_conf = alias_file(auth_conf,self.filemap)
        else: self.filemap = {}
        
        self.log_path = alias_file(logfile, self.filemap)
        self.s3fs = s3fs.S3FileSystem(anon=False) # Initialize s3 access. Key in secrets.toml
        self.data_source = data_source
        self.public = public
        self.pages = []
        self.sb_info = st.sidebar.empty()
        self.info = st.empty()
        
        # Set up translation
        pot_updater = po_template_updater()
        translate = load_translate(translate)
        self.tf = lambda s,**kwargs: translate(pot_updater(s,**kwargs))
        
        self.p_widths = {}
        
        # Set up authentication
        with st.spinner(self.tf("Setting up authentication...",context='ui')):
            self.uam = UserAuthenticationManager(auth_conf, groups, s3_fs=self.s3fs, info=self.info, logger=self.log_event, 
                                                translate_func=lambda t: self.tf(t,context='ui'))

        if not public:
            self.uam.login_screen()
            
        # Wrap some streamlit functions with translate
        wrap_list = ['write','markdown','title','header','subheader','caption','text','divider',
                     'button','download_button','link_button','checkbox','toggle','radio','selectbox',
                     'multiselect','slider','select_slider','text_input','number_input','text_area',
                     'date_input','time_input','file_uploader','camera_input','color_picker', 'popover']
        for fn in wrap_list:
            wrap_st_with_translate(fn,self)

    @property
    def user(self):
        return self.uam.user
    
    def log_event(self, event, username=None):
        log_event(event, username or st.session_state['username'], self.log_path, s3_fs=self.s3fs)

    # pos_id is for plot_width to work in columns
    def plot(self, pp_desc, pos_id='main', **kwargs):
        # Find or reuse width
        width = self.p_widths[pos_id] if pos_id in self.p_widths else get_plot_width(pos_id)
        self.p_widths[pos_id] = width
        
        # Draw plot
        st_plot(pp_desc,
                width=width, translate=lambda s: self.tf(s,context='data'),
                full_df=self.df,data_meta=self.meta,**kwargs)
        
    def filter_ui(self, dims, detailed=False, raw=False, force_choice=False):
        return filter_ui(self.df, self.meta, dims=dims, detailed=detailed, raw=raw, translate=self.tf, force_choice=force_choice)
    
    def facet_ui(self, dims, two=False, raw=False, force_choice=False):
        return facet_ui(dims, two=two, raw=raw, translate=self.tf,force_choice=force_choice)

    def page(self, name, **kwargs):
        def decorator(pfunc):
            groups = vod(kwargs,'groups')
            if (groups is None or # Page is available to all
                vod(self.user,'group')=='admin' or # Admin sees all
                vod(self.user,'group','guests') in groups): # group is whitelisted
                self.pages.append( (name,pfunc,kwargs) )
        return decorator

    def build(self):    
        # If login failed and is required, don't go any further
        if not self.public and not st.session_state["authentication_status"]: return
    
        # Add user settings page if logged in
        if self.user:  self.pages.append( ('Settings',user_settings_page,{'icon': 'sliders'}) )
        
        # Add admin page for admins
        if vod(self.user,'group')=='admin':  self.pages.append( ('Administration', admin_page,{'icon': 'terminal'}) )
        
        # Draw the menu listing pages
        pnames = [t[0] for t in self.pages]
        with st.sidebar:
            
            if self.user:
                self.sb_info.info(self.tf('Logged in as **%s**',context='ui') % self.user["name"])
                self.uam.auth.logout(self.tf('Log out',context='ui'), 'sidebar')
            
            t_pnames = [ self.tf(pn,context='ui') for pn in pnames]
            menu_choice = option_menu("Pages",
                t_pnames,
                icons=[vod(t[2],'icon') for t in self.pages],
                styles={
                    "container": {"padding": "5!important"}, #, "background-color": "#fafafa"},
                    #"icon": {"color": "red", "font-size": "15px"},
                    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#red"},
                    "menu-title": {"display":"none"}
                })
            
        # Find the page
        pname, pfunc, meta = self.pages[t_pnames.index(menu_choice)]
        
        # Load data
        self.data_source = vod(meta,'data_source',self.data_source)
        with st.spinner(self.tf("Loading data...",context='ui')):
            self.df, self.meta = read_annotated_data_cached(alias_file(self.data_source,self.filemap))
        
        # Render the chosen page
        self.subheader(pname)
        pfunc(**clean_kwargs(pfunc,{'sdb':self}))
        
    # Add enter and exit so it can be used as a context
    def __enter__(self):
        return self
    
    # Render everything once we exit the with block
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.build()

# %% ../nbs/05_dashboard.ipynb 14
class UserAuthenticationManager():
    
    def __init__(self,auth_conf_file,groups,s3_fs,info,logger,translate_func):
        self.groups, self.s3fs, self.info  = groups, s3_fs, info
        
        config = load_json_cached(auth_conf_file, _s3_fs = self.s3fs)
        self.conf, self.conf_file = config, auth_conf_file
        self.auth = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )
        self.users = config['credentials']['usernames']
        self.user = {} # Filled on login
        self.log_event = logger
        self.tf = translate_func
        
        # Mark that we should log the next login
        if 'log_event' not in st.session_state: st.session_state['log_event'] = True
                          
    def require_admin(self):
        if not self.admin: raise Exception("This action requires administrator privileges")
    
    def load_uncached_conf(self):
        self.conf = load_json(self.conf_file, _s3_fs = self.s3fs)
        self.users = self.conf['credentials']['usernames']
    
    def login_screen(self):
        tf = self.tf
        _, _, username = self.auth.login('sidebar', fields={'Form name':tf('Login page'), 'Username':tf('Username'), 'Password':tf('Password'), 'Log in':tf('Log in')})
        
        if st.session_state["authentication_status"] is False:
            st.error(tf('Username/password is incorrect'))
            self.log_event('login-fail', username=username)
        if st.session_state["authentication_status"] is None:
            st.warning(tf('Please enter your username and password'))
            st.session_state['log_event'] = True 
        elif st.session_state["authentication_status"]:
            self.user = {'name': st.session_state['name'], 
                         'username': username,
                         **self.users[username] }
            
            #check if signing in has been logged - if not, log it and flip the flag
            if st.session_state['log_event']:
                self.log_event('login-success')
                st.session_state['log_event'] = False
        
        self.admin = (vod(self.user,'group') == 'admin')
        
    def update_conf(self):
        with open_fn(self.conf_file,'w',s3_fs=self.s3fs) as jf:
            json.dump(self.conf,jf)
        time.sleep(3) # Give some time for messages to display etc
        st.rerun() # Force a rerun to reload the new file
            
    def add_user(self, username, password, user_data):
        self.require_admin()
        if username not in self.users:
            user_data['password'] = stauth.Hasher([password]).generate()[0]
            self.users[username] = user_data
            self.info.success(f'User {username} successfully added.')
            self.log_event(f'add-user: {username}')
            self.update_conf()
            return True
        else:
            self.info.error(f'User **{username}** already exists.')
            return False
        
    def change_user(self, username, user_data):
        
        # Change username
        if 'username' in user_data and username != user_data['username']:
            self.users[user_data['username']] = self.users[username]
            del self.users[username]
            username = user_data['username']
            del user_data['username']
        
        # Handle password change
        if vod(user_data,'password'):
            user_data['password'] = stauth.Hasher([user_data['password']]).generate()[0]
        else: del user_data['password']
        
        # Update everything else
        self.users[username].update(user_data)
        self.log_event(f'change-user: {username}')
        self.info.success(f'User **{username}** changed.')
        self.update_conf()
        
    def delete_user(self, username): 
        self.require_admin()
        del self.users[username]
        self.info.warning(f'User **{username}** deleted.')
        self.log_event(f'delete-user: {username}')
        self.update_conf()

    def list_users(self):
        self.require_admin()
        return [ censor_dict({'username': k, **v},['password']) for k,v in self.users.items() ]


# %% ../nbs/05_dashboard.ipynb 16
# Password reset
def user_settings_page(sdb):
    if not sdb.user: return
    try:
        tf = lambda s: sdb.tf(s,context='ui')
        if sdb.uam.auth.reset_password(st.session_state["username"], 
                                       fields={'Form name':tf('Reset password'), 'Current password':tf('Current password'), 
                                               'New password':tf('New password'), 'Repeat password': tf('Repeat password'), 
                                               'Reset':tf('Reset')}):
            sdb.uam.update_conf()
            st.success(tf('Password modified successfully'))
    except Exception as e:
        st.error(e)

# %% ../nbs/05_dashboard.ipynb 17
# Helper function to highlight log rows
def highlight_cells(val):
    if 'fail' in val:
        color = 'red'
    #elif 'add' in val:
    elif any(s in val for s in ['delete', 'add', 'change']):
        color = 'blue'
    elif 'success' in val:
        color='green'
    else:
        color = ''
    return 'color: {}'.format(color)

# %% ../nbs/05_dashboard.ipynb 18
# Admin page to manage users

def admin_page(sdb):
    sdb.uam.require_admin()
    sdb.uam.load_uncached_conf() # so all admin updates would immediately be visible
    
    menu_choice = option_menu(None,[ 'Log management', 'List users', 'Add user', 'Change user', 'Delete user' ], 
                              icons=['card-list','people-fill','person-fill-add','person-lines-fill','person-fill-dash'], orientation='horizontal')
    st.write(" ")

    if menu_choice=='Log management':
        log_data=pd.read_csv(alias_file(sdb.log_path,sdb.filemap),names=['timestamp','event','username'])
        st.dataframe(log_data.sort_index(ascending=False
            ).style.map(highlight_cells, subset=['event']), width=1200) #use_container_width=True
        
    elif menu_choice=='List users':
        # Read log to get last login:
        log_data = pd.read_csv(alias_file(sdb.log_path,sdb.filemap),names=['timestamp','event','username'])
        log_data = log_data[log_data['event']=='login-success']
        log_data['timestamp'] = pd.to_datetime(log_data['timestamp'], utc=True, format='%d-%m-%Y, %H:%M:%S')
        
        # Add last login to users
        users = sdb.uam.list_users()
        for u in users:
            last_login = log_data[log_data['username'] == u['username']].timestamp.max()
            if pd.notnull(last_login):
                u['last_login'] = last_login.strftime('%d-%b-%Y')
                
        # Display the data
        st.dataframe(users, use_container_width=True)

    elif menu_choice=='Add user':
        with st.form("add_user_form"):
            st.subheader("Add user:")
            st.markdown("""---""")
            col1,col2 = st.columns((1,2))
            user_data = {}
            with col1:
                user_data['group'] = st.radio("Group:", sdb.uam.groups)
            with col2:
                username = st.text_input("Username:")
                password = st.text_input("Password:", type='password')
                user_data['name'] = st.text_input("Name:")
                st.markdown("""---""")
                user_data['email'] = st.text_input("E-mail:")
                user_data['organization'] = st.text_input("Organization:")
            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not '' in [username, password, user_data['email']]:
                    sdb.uam.add_user(username, password, user_data)
                else:
                    sdb.info.error('Must specify username, password and email.')

    elif menu_choice=='Change user':
        username=st.selectbox('Edit user', list(sdb.uam.users.keys()))
        
        user_data = sdb.uam.users[username].copy()
        #st.write(user_data)
        group_index = sdb.uam.groups.index(user_data['group'])

        with st.form("edit_user_form"):
            st.subheader("Edit user data:")
            st.markdown("""---""")
            col1,col2 = st.columns((1,2))
            with col1:
                user_data['username'] = st.text_input("Username:", value=username, disabled=True)
                user_data['group'] = st.radio("Group:", sdb.uam.groups, index=group_index) #, disabled=True)
            with col2:
                #new_user = st.text_input("Kasutaja:", value=username, disabled=True)
                user_data['name'] = st.text_input("Name:", value=user_data['name'])
                user_data['password'] = st.text_input("Password:", type='password')
                st.markdown("""---""")
                user_data['email'] = st.text_input("E-mail:", value=user_data['email'])
                user_data['organization'] = st.text_input("Organization:", value=vod(user_data,'organization',''))
                
            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                sdb.uam.change_user(username,user_data)
                
    elif menu_choice=='Delete user':
        with st.form("delete_user_form"):
            st.subheader('Delete user:')
            username = st.selectbox('Select username:', list(sdb.uam.users.keys()))
            check = st.checkbox('Deletion is FINAL and cannot be undone!')
            st.markdown("""___""")
            submitted = st.form_submit_button("Delete")
            if submitted:
                if not check:
                    sdb.info.warning(f'Tick the checkbox in order to delete user **{username}**.')
                elif username == sdb.user['username']:
                    sdb.info.error('Cannot delete the current user.')
                else:
                    sdb.uam.delete_user(username)


# %% ../nbs/05_dashboard.ipynb 22
# This is a horrible workaround to get faceting to work with altair geoplots that do not play well with streamlit
# See https://github.com/altair-viz/altair/issues/2369 -> https://github.com/vega/vega-lite/issues/3729

# Draw a matrix of plots using separate plots and st columns
def draw_plot_matrix(pmat,matrix_form = False):
    if not pmat: return # Do nothing if get None passed to it
    if not matrix_form: pmat = [[pmat]]
    cols = st.columns(len(pmat[0]))
    for j,c in enumerate(cols):
        for i, row in enumerate(pmat):
            if j>=len(pmat[i]): continue
            c.altair_chart(pmat[i][j])

# Draw the plot described by pp_desc 
def st_plot(pp_desc,**kwargs):
    matrix_form = (pp_desc['plot'] == 'geoplot')
    plots = e2e_plot(pp_desc, return_matrix_of_plots=matrix_form, **kwargs)
    draw_plot_matrix(plots, matrix_form=matrix_form)

# %% ../nbs/05_dashboard.ipynb 23
def facet_ui(dims, two=False, raw=False, translate=None, force_choice=False):
    # Set up translation
    tfc = translate if translate else (lambda s,**kwargs: s)
    tf = lambda s: tfc(s,context='data')
    
    tdims = [ tf(d) for d in dims ]
    r_map = dict(zip(tdims,dims))
    
    none = tf('None')
    stc = st.sidebar if not raw else st
    facet_dim = stc.selectbox(tfc('Facet:',context='ui'), tdims if force_choice else [none] + tdims )
    fcols = [facet_dim] if facet_dim != none else []
    if two and facet_dim != none:
        second_dim = stc.selectbox(tfc('Facet 2:',context='ui'), tdims if force_choice else [none] + tdims)
        if second_dim not in [none,facet_dim]:  fcols = [facet_dim, second_dim]
        
    return [ r_map[d] for d in fcols ]

# %% ../nbs/05_dashboard.ipynb 24
# Function that creates reset functions for multiselects in filter
def ms_reset(cn, all_vals):
    def reset_ms():
        st.session_state[f"{cn}_multiselect"] = all_vals
    return reset_ms

# %% ../nbs/05_dashboard.ipynb 25
# User interface that outputs a filter for the pp_desc
def filter_ui(data, dmeta=None, dims=None, detailed=False, raw=False, translate=None, force_choice=False,):
    
    tfc = translate if translate else (lambda s,**kwargs: s)
    tf = lambda s: tfc(s,context='data')
    
    if dims is None:
        dims = [c for c in data.columns if c not in ['draw', 'weight', 'training_subsample'] ]  
    
    if dmeta is not None:
        dims = list_aliases(dims, group_columns_dict(dmeta)) # Replace aliases like 'demographics'
        c_meta = extract_column_meta(dmeta) # mainly for groups defined in meta
    else: c_meta = defaultdict(lambda: {})
    
    if not force_choice: f_info = st.sidebar.container()
    
    stc = st.sidebar.expander(tfc('Filters',context='ui')) if not raw else st
    
    # Different selector for different category types
    # Also - make sure filter is clean and only applies when it is changed from the default 'all' value
    # This has considerable speed and efficiency implications
    filters = {}
    for cn in dims:
        col = data[cn]
        if col.dtype.name=='category':
            if len(col.dtype.categories)==1: continue
            
            # Do some prep for translations
            r_map = dict(zip([tf(c) for c in col.dtype.categories],col.dtype.categories))
            all_vals = list(r_map.keys()) # translated categories
            grp_names = vod(c_meta[cn],'groups',{}).keys()
            r_map.update(dict(zip([tf(c) for c in grp_names],grp_names)))
            
        if detailed and col.dtype.name=='category': # Multiselect
            filters[cn] = stc.multiselect(tf(cn), all_vals, all_vals, key=f"{cn}_multiselect")
            if set(filters[cn]) == set(all_vals): del filters[cn]
            else: 
                stc.button(tf("Reset"),key=f"{cn}_reset",on_click=ms_reset(cn,all_vals))
                filters[cn] = [ r_map[c] for c in filters[cn] ]
        elif col.dtype.name=='category' and not col.dtype.ordered: # Unordered categorical - selectbox
            choices = [gt for gt,g in r_map.items() if g in grp_names] + all_vals
            if not force_choice: choices = [tf('All')] + choices
            filters[cn] = stc.selectbox(tf(cn),choices)
            if filters[cn] == tf('All'): del filters[cn]
            else: filters[cn] = r_map[filters[cn]]
        # Use [None,<start>,<end>] for ranges, both categorical and continuous to distinguish them from list of values
        elif col.dtype.name=='category': # Ordered categorical - slider
            f_res = stc.select_slider(tf(cn),all_vals,value=(all_vals[0],all_vals[-1]))
            if f_res != (all_vals[0],all_vals[-1]): 
                filters[cn] = [None]+[r_map[f_res[0]],r_map[f_res[1]]]
        elif is_numeric_dtype(col) and col.dtype!='bool': # Continuous
            mima = (col.min(),col.max())
            if mima[0]==mima[1]: continue
            f_res = stc.slider(tf(cn),*mima,value=mima)
            if f_res != mima: filters[cn] = [None] + list(f_res)
            
    if filters and not force_choice: f_info.warning('⚠️ ' + tfc('Filters active',context='ui') + ' ⚠️')
            
    return filters


# %% ../nbs/05_dashboard.ipynb 27
# Use dict here as dicts are ordered as of Python 3.7 and preserving order groups things together better

def translate_with_dict(d):
    return (lambda s: d[s] if isinstance(s,str) and s in d and d[s] is not None else s)

def log_missing_translations(tf, nonchanged_dict):
    def ntf(s):
        ns = tf(s)
        if ns==s: nonchanged_dict[s]=None
        return ns
    return ntf

def clean_missing_translations(nonchanged_dict, tdict={}):
    # Filter out numbers that come in from data sometimes
    return { s:v for s,v in nonchanged_dict.items() if s not in tdict and isinstance(s,str) and not re.fullmatch('[\.\d]+',s) }

def add_missing_to_dict(missing_dict, tdict):
    return {**tdict, **{ s:s for s in missing_dict}}
