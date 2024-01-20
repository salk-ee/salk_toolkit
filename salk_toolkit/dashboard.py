# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_dashboard.ipynb.

# %% auto 0
__all__ = ['draw_plot_matrix', 'st_plot', 'get_plot_width', 'open_fn', 'read_annotated_data_cached', 'load_json',
           'load_json_cached', 'save_json', 'alias_file', 'SalkDashboardBuilder', 'UserAuthenticationManager']

# %% ../nbs/05_dashboard.ipynb 3
import json, os, inspect
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import datetime as dt

from typing import List, Tuple, Dict, Union, Optional

import altair as alt
import s3fs

from salk_toolkit.utils import *
from salk_toolkit.io import *
from salk_toolkit.pp import e2e_plot

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_dimensions import st_dimensions
import streamlit_authenticator as stauth

# %% ../nbs/05_dashboard.ipynb 4
# This is a horrible workaround to get faceting to work with altair geoplots that do not play well with streamlit
# See https://github.com/altair-viz/altair/issues/2369 -> https://github.com/vega/vega-lite/issues/3729

# Draw a matrix of plots using separate plots and st columns
def draw_plot_matrix(pmat,matrix_form = False):
    if not matrix_form: pmat = [[pmat]]
    cols = st.columns(len(pmat[0]))
    for j,c in enumerate(cols):
        for i, row in enumerate(pmat):
            c.altair_chart(pmat[i][j])

# Draw the plot described by pp_desc 
def st_plot(pp_desc,**kwargs):
    matrix_form = (pp_desc['plot'] == 'geoplot')
    plots = e2e_plot(pp_desc, return_matrix_of_plots=matrix_form, **kwargs)
    draw_plot_matrix(plots, matrix_form=matrix_form)

# %% ../nbs/05_dashboard.ipynb 5
def get_plot_width(key):
    wobj = st_dimensions(key=key) or { 'width': 900 }# Can return none so handle that
    return int(0.85*wobj['width']) # Needs to be adjusted down  to leave margin for plots

# %% ../nbs/05_dashboard.ipynb 6
# Open either a local or an s3 file
def open_fn(fname, *args, s3_fs=None, **kwargs):
    if fname[:3] == 's3:':
        if s3_fs is None: s3_fs = s3fs.S3FileSystem(anon=False)
        return s3_fs.open(fname,*args,**kwargs)
    else:
        return open(fname,*args,**kwargs)

# %% ../nbs/05_dashboard.ipynb 7
# ttl=None - never expire. Makes sense for potentially big data files
@st.cache_resource(show_spinner=False,ttl=None)
def read_annotated_data_cached(data_source,**kwargs):
    return read_annotated_data(data_source,**kwargs)

# Load json uncached - useful for admin pages
def load_json(fname, _s3_fs=None, **kwargs):
    with open_fn(fname,'r',s3_fs=_s3_fs) as jf:
        return json.load(jf)

# This is cached very short term (1 minute) to avoid downloading it on every page change
# while still allowing users to be added / changed relatively responsively
@st.cache_resource(show_spinner=False,ttl=60)
def load_json_cached(fname, _s3_fs=None, **kwargs):
    return load_json(fname,_s3_fs,**kwargs)

# For saving json back 
def save_json(d, fname, _s3_fs=None, **kwargs):
    with open_fn(fname,'w',s3_fs=_s3_fs) as jf:
        json.dump(d,jf)
        
def alias_file(fname, file_map):
    if fname[:3]!='s3:' and fname in file_map and not os.path.exists(fname):
        #print(f"Redirecting {fname} to {file_map[fname]}")
        return file_map[fname]
    else: return fname

# %% ../nbs/05_dashboard.ipynb 8
# Main dashboard wrapper - WIP
class SalkDashboardBuilder:

    def __init__(self, data_source, auth_conf, groups=['guest','user','admin'], public=False):
        
        # Allow deployment.json to redirect files from local to s3 if local missing (i.e. in deployment scenario)
        if os.path.exists('deployment.json'):
            dep_meta = load_json_cached('deployment.json')
            filemap = vod(dep_meta,'files',{})
            data_source = alias_file(data_source,filemap)
            auth_conf = alias_file(auth_conf,filemap)
        
        self.s3fs = s3fs.S3FileSystem(anon=False) # Initialize s3 access. Key in secrets.toml
        self.data_source = data_source
        self.public = public
        self.pages = []
        self.sb_info = st.sidebar.empty()
        self.info = st.empty()
        
        # Set up authentication
        with st.spinner("Setting up authentication..."):
            self.uam = UserAuthenticationManager(auth_conf, groups, s3_fs=self.s3fs, info=self.info)

        if not public:
            self.uam.login_screen()

    @property
    def user(self):
        return self.uam.user

    # pos_id is for plot_width to work in columns
    def plot(self, pp_desc, pos_id=None, **kwargs):
        st_plot(pp_desc,
                width=min(get_plot_width(pos_id or 'full'),800),
                full_df=self.df,data_meta=self.meta,**kwargs)

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
        if vod(self.user,'group')=='admin':  self.pages.append( ('Administration',admin_page,{'icon': 'terminal'}) )
        
        # Draw the menu listing pages
        pnames = [t[0] for t in self.pages]
        with st.sidebar:
            
            if st.session_state["authentication_status"]:
                self.sb_info.info(f'Logged in as **{self.user["name"]}**')
                self.uam.auth.logout('Logout', 'sidebar')
            
            menu_choice = option_menu("Pages",
                pnames, icons=[vod(t[2],'icon') for t in self.pages],
                styles={
                    "container": {"padding": "5!important"}, #, "background-color": "#fafafa"},
                    #"icon": {"color": "red", "font-size": "15px"},
                    "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#red"},
                    "menu-title": {"display":"none"}
                })
            
        # Find the page
        pname, pfunc, meta = self.pages[pnames.index(menu_choice)]
        
        # Load data
        self.data_source = vod(meta,'data_source',self.data_source)
        with st.spinner("Loading data..."):
            self.df, self.meta = read_annotated_data_cached(self.data_source)
        
        # Render the chosen page
        st.title(pname)
        pfunc(**clean_kwargs(pfunc,{'sdb':self}))
        
    # Add enter and exit so it can be used as a context
    def __enter__(self):
        return self
    
    # Render everything once we exit the with block
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.build()
    
        

# %% ../nbs/05_dashboard.ipynb 11
class UserAuthenticationManager():
    
    def __init__(self,auth_conf_file,groups,s3_fs,info):
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
                          
    def require_admin(self):
        if not self.admin: raise Exception("This action requires administrator privileges")
    
    def load_uncached_conf(self):
        self.conf = load_json(self.conf_file, _s3_fs = self.s3fs)
        self.users = self.conf['credentials']['usernames']
    
    def login_screen(self):
        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        if st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
        self.auth.login('Login', 'main')
        
        if st.session_state["authentication_status"]:
            uname = st.session_state['username']
            self.user = {'name': st.session_state['name'], 
                         'username': uname,
                         **self.users[uname] }
        
        self.admin = (vod(self.user,'group') == 'admin')
        
    def update_conf(self):
        with open_fn(self.conf_file,'w',s3_fs=self.s3fs) as jf:
            json.dump(self.conf,jf)
        st.rerun() # Force a rerun to reload the new file
            
    def add_user(self, username, password, user_data):
        self.require_admin()
        if username not in self.users:
            user_data['password'] = stauth.Hasher([password]).generate()[0]
            self.users[username] = user_data
            self.update_conf()
            self.info.success(f'User {username} successfully added.')
            #log_event('add-user: ' + user, st.session_state['username'])
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
        self.update_conf()
        
        self.info.success(f'User **{username}** changed.')
        
    def delete_user(self, username):        
        self.require_admin()
        del self.users[username]
        self.update_conf()
        self.info.warning(f'User **{username}** deleted.')
        #log_event('delete-user: ' + user, st.session_state['username'])

    def list_users(self):
        self.require_admin()
        return [ censor_dict({'username': k, **v},['password']) for k,v in self.users.items() ]


# %% ../nbs/05_dashboard.ipynb 13
# Password reset
def user_settings_page(sdb):
    if not sdb.user: return
    try:
        if sdb.uam.auth.reset_password(st.session_state["username"], 'Reset password'):
            sdb.uam.update_conf()
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)

# %% ../nbs/05_dashboard.ipynb 14
# Admin page to manage users

def admin_page(sdb):
    sdb.uam.require_admin()
    sdb.uam.load_uncached_conf() # so all admin updates would immediately be visible
    
    userlist, adduser, changeuser, deleteuser = st.tabs([
        #'Log management',
        'List users',
        'Add user',
        'Change user',
        'Delete user'
    ])
    st.write(" ")

    # TODO: logging
    x='''with log_management:
        if st.button('View logfile', disabled=restricted):
            log_data=pd.read_csv(log_file)
            st.dataframe(log_data.sort_index(ascending=False
                ).style.applymap(highlight_cells, subset=['status']), width=1200) #use_container_width=True
        if st.button('Reset logfile', disabled=True): #disabled=restricted
            create_log(log_file)

        #if st.button('Read S3'):
        #    st.write(s3_read_new('salk-test/users.json'))'''

    with userlist:
        #st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        st.dataframe(sdb.uam.list_users(), use_container_width=True)

    with adduser:
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
            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not '' in [username, password, user_data['email']]:
                    sdb.uam.add_user(username, password, user_data)
                else:
                    sdb.info.error('Must specify username, password and email.')

    with changeuser:
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
            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                sdb.uam.change_user(username,user_data)
                
    with deleteuser:
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

