"""Dashboard Framework
-------------------

Everything that used to live in `05_dashboard.ipynb` now lives here.  The
module exposes:

- helper utilities for loading data lazily, caching plots, and wiring Streamlit
  state (translations, filters, impersonation, admin tools)
- authentication plumbing (legacy auth + Frontegg), logging, and metrics hooks
- the `SalkDashboardBuilder` façade that pages, rendering, and CLI tools depend
  on
"""

# Wrapping streamlit objects with translation is funky from a type-system perspective
# For now, disabling the two type checks this affects the most
# pyright: reportCallIssue=false
# pyright: reportAttributeAccessIssue=false

__all__ = [
    "get_plot_width",
    "open_fn",
    "exists_fn",
    "read_parquet_with_data_meta_lazy_cached",
    "load_json",
    "load_json_cached",
    "save_json",
    "alias_file",
    "log_event",
    "default_translate",
    "SalkDashboardBuilder",
    "sqlite_client",
    "StreamlitAuthenticationManager",
    "frontegg_client",
    "FronteggAuthenticationManager",
    "admin_page",
    "draw_plot_matrix",
    "st_plot",
    "plot_cache",
    "stss_safety",
    "facet_ui",
    "filter_ui",
    "translate_with_dict",
    "log_missing_translations",
    "clean_missing_translations",
    "add_missing_to_dict",
    "translate_pot",
    "plot_matrix_html",
]

import json
import os
import csv
import re
import time
import inspect
import psutil
from collections import defaultdict
from contextlib import AbstractContextManager
from abc import abstractmethod
from copy import deepcopy

import pandas as pd
import polars as pl
import datetime as dt


import s3fs  # type: ignore[import-untyped]
import polib
import __main__  # to get name of py file


from salk_toolkit import utils
from salk_toolkit.utils import plot_matrix_html
from salk_toolkit.io import (
    read_parquet_with_metadata,
    fix_df_with_meta,
    extract_column_meta,
    group_columns_dict,
    list_aliases,
)
from salk_toolkit.pp import e2e_plot
from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, FilterSpec

import streamlit as st
from streamlit_option_menu import option_menu  # type: ignore[import-untyped]
from streamlit_dimensions import st_dimensions  # type: ignore[import-untyped]
import streamlit_authenticator as stauth  # type: ignore[import-untyped]
import libsql_client  # type: ignore[import-untyped]
from typing import Callable, Any, cast, IO, ContextManager, Protocol

# Type alias for JSON data
JsonDict = dict[str, Any]


class TranslationObject(Protocol):
    """Protocol for objects that provide translation functionality."""

    def tf(self, s: str, **kwargs: object) -> str:
        """Translate a string."""
        ...


def get_plot_width(key: str) -> int:
    """Get the plot width based on Streamlit dimensions.

    Args:
        key: Key used to track dimensions in Streamlit session state.

    Returns:
        Plot width in pixels, capped at 800px and adjusted to 80% of container width.
    """
    wobj = st_dimensions(key=key) or {"width": 800}  # Can return none so handle that
    return min(800, int(0.8 * wobj["width"]))  # Needs to be adjusted down  to leave margin for plots


def _update_manifest(fname: str) -> None:
    """Update the manifest JSON file with a new file entry.

    Args:
        fname: Filename to add to the manifest.
    """
    # Create manifests directory if it doesn't exist
    os.makedirs("manifests", exist_ok=True)

    # Get name of main python file without extension
    main_name = os.path.splitext(os.path.basename(__main__.__file__))[0]

    # Create manifest json file
    manifest_path = os.path.join("manifests", f"{main_name}.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if "files" not in manifest:
            manifest["files"] = []
    else:
        manifest = {
            "id": main_name,
            "app": main_name + ".py",
            "requirements": "requirements.txt",
            "files": ["deployment.json"],  # TODO: This system should be rethought
        }

    if fname not in manifest["files"]:
        manifest["files"].append(fname)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)


# Open either a local or an s3 file
def open_fn(
    fname: str, *args: object, s3_fs: s3fs.S3FileSystem | None = None, **kwargs: object
) -> ContextManager[IO[bytes]]:
    """Open a file from local filesystem or S3.

    Args:
        fname: File path (local or S3 URI starting with 's3:').
        *args: Additional positional arguments passed to open().
        s3_fs: Optional S3 filesystem instance (created if needed for S3 paths).
        **kwargs: Additional keyword arguments passed to open().

    Returns:
        File-like object opened from local filesystem or S3.
    """
    # _update_manifest(fname) # Actually these tend to be the files we want to be in S3
    if fname[:3] == "s3:":
        if s3_fs is None:
            s3_fs = s3fs.S3FileSystem(anon=False)
        return cast(ContextManager[IO[bytes]], s3_fs.open(fname, *args, **kwargs))
    else:
        return cast(ContextManager[IO[bytes]], open(fname, *args, **kwargs))


def exists_fn(fname: str, *args: object, s3_fs: s3fs.S3FileSystem | None = None, **kwargs: object) -> bool:
    """Check if a file exists in local filesystem or S3.

    Args:
        fname: File path (local or S3 URI starting with 's3:').
        *args: Additional positional arguments (unused).
        s3_fs: Optional S3 filesystem instance (created if needed for S3 paths).
        **kwargs: Additional keyword arguments passed to exists().

    Returns:
        True if file exists, False otherwise.
    """
    if fname[:3] == "s3:":
        if s3_fs is None:
            s3_fs = s3fs.S3FileSystem(anon=False)
        return s3_fs.exists(fname, *args, **kwargs)
    else:
        return os.path.exists(fname, *args, **kwargs)


# ttl=None - never expire. Makes sense for potentially big data files
@st.cache_resource(show_spinner=False, ttl=None)
def read_parquet_with_data_meta_lazy_cached(data_source: str, **kwargs: object) -> tuple[object, DataMeta]:
    """Load parquet file with metadata using lazy Polars loading (cached).

    Args:
        data_source: Path to parquet file.
        **kwargs: Additional arguments passed to read_parquet_with_metadata.

    Returns:
        Tuple of (lazy Polars DataFrame, DataMeta).
    """
    print(f"Reading lazy data from {data_source}")
    df, full_meta = read_parquet_with_metadata(data_source, lazy=True, **kwargs)
    assert full_meta is not None, "Expected metadata to be present"
    data_meta = full_meta.data
    return df, data_meta


# Load json uncached - useful for admin pages


def load_json(fname: str, _s3_fs: s3fs.S3FileSystem | None = None, **kwargs: object) -> JsonDict:
    """Load JSON file from local filesystem or S3 (uncached).

    Args:
        fname: Path to JSON file (local or S3 URI).
        _s3_fs: Optional S3 filesystem instance.
        **kwargs: Additional arguments passed to json.load().

    Returns:
        Parsed JSON object (typically a dict).
    """
    with open_fn(fname, "r", s3_fs=_s3_fs, encoding="utf8") as jf:
        result: JsonDict = json.load(jf)
        return result


# This is cached very short term (1 minute) to avoid downloading it on every page change
# while still allowing users to be added / changed relatively responsively


@st.cache_data(show_spinner=False, ttl=60)
def load_json_cached(fname: str, _s3_fs: s3fs.S3FileSystem | None = None, **kwargs: object) -> JsonDict:
    """Load JSON file from local filesystem or S3 (cached for 60 seconds).

    Args:
        fname: Path to JSON file (local or S3 URI).
        _s3_fs: Optional S3 filesystem instance.
        **kwargs: Additional arguments passed to json.load().

    Returns:
        Parsed JSON object (typically a dict).
    """
    return load_json(fname, _s3_fs, **kwargs)


# For saving json back


def save_json(d: object, fname: str, _s3_fs: s3fs.S3FileSystem | None = None, **kwargs: object) -> None:
    """Save JSON object to local filesystem or S3.

    Args:
        d: JSON-serializable object to save.
        fname: Path to output file (local or S3 URI).
        _s3_fs: Optional S3 filesystem instance.
        **kwargs: Additional arguments passed to json.dump().
    """
    with open_fn(fname, "w", s3_fs=_s3_fs, encoding="utf8") as jf:
        json.dump(d, jf, indent=2, ensure_ascii=False)


def alias_file(fname: str, file_map: dict[str, str]) -> str:
    """Resolve file path using alias mapping if local file doesn't exist.

    Args:
        fname: Original file path.
        file_map: Dictionary mapping original paths to aliased paths.

    Returns:
        Original path if file exists or no alias, otherwise aliased path.
    """
    if fname[:3] != "s3:" and fname in file_map and not os.path.exists(fname):
        # print(f"Redirecting {fname} to {file_map[fname]}")
        return file_map[fname]
    else:
        return fname


def log_event(event: str, uid: str, path: str, s3_fs: s3fs.S3FileSystem | None = None) -> None:
    """Log an event to a CSV file (local or S3).

    Args:
        event: Event description string.
        uid: User identifier.
        path: Path to log file (local or S3 URI).
        s3_fs: Optional S3 filesystem instance.
    """
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%d-%m-%Y, %H:%M:%S")

    if not exists_fn(path, s3_fs=s3_fs):  # If file not present, create it
        print(f"Log file {path} not found, creating it")
        with open_fn(path, "w", s3_fs=s3_fs):
            pass  # Just create the file

    # Just append the row to the file
    with open_fn(path, "a", s3_fs=s3_fs) as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event, uid])


# --------------------------------------------------------
#          SETUP GUIDE
# --------------------------------------------------------
# - User conf
#   - Cookie key matters; generate a robust random secret
# - Logfile: touch a local log file so development logs do not pollute deploy logs
# - Files: if `deploy.json` targets are missing, notify the user.
#   If files are absent in S3, copy them over and expose a flag to update.
# - Translations: keep translation sources (.pot/.po) in the repo.

st_wrap_list = [
    "write",
    "markdown",
    "title",
    "header",
    "subheader",
    "caption",
    "text",
    "divider",
    "button",
    "download_button",
    "link_button",
    "checkbox",
    "toggle",
    "radio",
    "selectbox",
    "multiselect",
    "slider",
    "select_slider",
    "text_input",
    "number_input",
    "text_area",
    "date_input",
    "time_input",
    "file_uploader",
    "camera_input",
    "color_picker",
    "popover",
    "spinner",
    "info",
    "error",
    "warning",
    "success",
    "pills",
]

# def debugf(f,*args,**kwargs):
#     print(f.__name__,args,kwargs)
#     return f(*args,**kwargs)

# Some keyword arguments can be translated


def _transform_kws(kws: dict[str, object], tfo: TranslationObject) -> dict[str, object]:
    """Transform keyword arguments for Streamlit functions with translation.

    Args:
        kws: Dictionary of keyword arguments to transform.
        tfo: Translation function object with tf method.

    Returns:
        Modified keyword arguments dictionary with translated values.
    """
    if "context" in kws:
        del kws["context"]
    if "format_func" in kws:
        ff = cast(Callable[[str], str], kws["format_func"])
        kws["format_func"] = lambda s: tfo.tf(ff(s))
    if "placeholder" in kws:
        kws["placeholder"] = tfo.tf(kws["placeholder"])
    return kws


# wrap the first parameter of streamlit function with self._translate
# has to be a separate function instead of in a for loop for scoping reasons


def _wrap_st_with_translate(base: object, fd: str | dict[str, object], tfo: TranslationObject) -> Callable[..., object]:
    """Wrap a Streamlit function to automatically translate its first argument.

    Args:
        base: Base object (typically st or st.sidebar) containing the function.
        fd: Function descriptor (string name or dict with 'name' and 'args').
        tfo: Translation function object with tf method.

    Returns:
        Wrapped function that translates arguments before calling the original.
    """
    if isinstance(fd, str):
        fd = {"name": fd, "args": ["str"]}
    func = getattr(base, fd["name"])

    # If format_func is a parameter, overwrite it with the translate function
    kw_defaults = {"format_func": tfo.tf} if "format_func" in inspect.signature(func).parameters else {}

    tfs = {
        "str": lambda c: (lambda s: tfo.tf(s, context=c)),
        "list": lambda c: (lambda item_list: [tfo.tf(s, context=c) for s in item_list]),
    }
    return (
        lambda *args, **kwargs: func(  # debugf(func,
            *[tfs[tt](kwargs.get("context"))(args[i]) for i, tt in enumerate(fd["args"])],
            *args[len(fd["args"]) :],
            **{**kw_defaults, **_transform_kws(kwargs, tfo)},
        )
    )


# A class that wraps another context manager


class ContextManagerWrapper(AbstractContextManager):
    """Wrapper for context managers to enable composition."""

    def __init__(self, obj: AbstractContextManager) -> None:
        """Initialize wrapper with a context manager object.

        Args:
            obj: Context manager to wrap.
        """
        self.obj = obj

    def __enter__(self) -> object:
        """Enter the wrapped context manager."""
        return self.obj.__enter__()

    def __exit__(self, *args: object) -> None:
        """Exit the wrapped context manager."""
        self.obj.__exit__(*args)


def _wrap_all_st_functions(base: object, tfo: TranslationObject, to: object | None = None) -> object:
    """Wrap all Streamlit functions in st_wrap_list with translation support.

    Args:
        base: Base object (typically st or st.sidebar) to wrap functions from.
        tfo: Translation function object with tf method.
        to: Target object to attach wrapped functions to (defaults to base).

    Returns:
        Object with all wrapped Streamlit functions attached.
    """
    if to is None:
        to = ContextManagerWrapper(base)

    for fd in st_wrap_list:
        fn = fd["name"] if isinstance(fd, dict) else fd
        if not hasattr(st, fn):
            continue
        setattr(to, fn, _wrap_st_with_translate(base, fd, tfo))

    # Container creators need to be wrapped recursively
    setattr(
        to,
        "tabs",
        lambda *args, **kwargs: tuple(
            _wrap_all_st_functions(c, tfo)
            for c in cast(Any, _wrap_st_with_translate(base, {"name": "tabs", "args": ["list"]}, tfo)(*args, **kwargs))
        ),
    )
    setattr(
        to,
        "columns",
        lambda *args, **kwargs: tuple(_wrap_all_st_functions(c, tfo) for c in base.columns(*args, **kwargs)),
    )

    setattr(
        to,
        "expander",
        lambda *args, **kwargs: _wrap_all_st_functions(
            _wrap_st_with_translate(base, "expander", tfo)(*args, **kwargs), tfo
        ),
    )
    setattr(
        to,
        "container",
        lambda *args, **kwargs: _wrap_all_st_functions(base.container(*args, **kwargs), tfo),
    )

    return to


def default_translate(s: str, **kwargs: object) -> str:
    """Default translation function that capitalizes and replaces underscores.

    Args:
        s: String to _translate.
        **kwargs: Additional arguments (unused).

    Returns:
        Translated string (capitalized with underscores replaced by spaces).
    """
    return (s[0].upper() + s[1:]).replace("_", " ") if isinstance(s, str) and len(s) > 0 else s


# A function that automatically updates the pot file with untranslated strings
def _po_template_updater(pot_file: str | None = None) -> Callable[[str], str]:
    """Create a translation function that auto-updates .pot files.

    Args:
        pot_file: Path to .pot template file (auto-detected if None).

    Returns:
        Translation function that adds new strings to the .pot file.
    """
    if pot_file is None:
        bname = os.path.splitext(os.path.basename(__main__.__file__))[0]
        pot_file = f"locale/{bname}.pot"

    if os.path.exists(pot_file):
        po = polib.pofile(pot_file)
        tdc = defaultdict(set)
        for entry in po:
            context = entry.msgctxt or ""
            tdc[context].add(entry.msgid)
    else:
        po = polib.POFile()
        po.metadata = {
            "Project-Id-Version": "1.0",
            "Report-Msgid-Bugs-To": "tarmo@salk.com",
            "MIME-Version": "1.0",
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Transfer-Encoding": "8bit",
        }
        tdc = defaultdict(set)
    _update_manifest(pot_file)

    def _translate(s: str, **kwargs: object) -> str:
        """Translate a string using the current translation function.

        Args:
            s: String to _translate.
            **kwargs: Additional arguments passed to translation function.

        Returns:
            Translated string.
        """
        ctx = kwargs.get("context") or ""
        if isinstance(s, str) and s not in tdc[ctx]:
            po.append(
                polib.POEntry(
                    msgid=s,
                    msgstr=default_translate(s),
                    **{
                        "msgctxt": kwargs.get("context"),
                        "comment": kwargs.get("comment"),
                    },
                )
            )
            po.save(pot_file)
            tdc[ctx].add(s)
        return s

    return _translate


def _translate_fn_from_po(po_file: str) -> Callable[[str], str]:
    """Load translations from a .po file and return a translation function.

    Args:
        po_file: Path to .po file containing translations.

    Returns:
        Translation function that maps msgid to msgstr.
    """
    po = polib.pofile(po_file)
    td = {entry.msgid: entry.msgstr for entry in po}
    return lambda s, **kwargs: td.get(s, s)


def _load_translate(
    translate: str | dict[str, str] | Callable[[str], str] | None,
    cc_translations: dict[str, Callable[[str], str] | None] | None = None,
) -> Callable[[str], str]:
    """Load a translation function from various sources.

    Args:
        _translate: Translation source (function, dict, file path, or language code).
        cc_translations: Dictionary mapping language codes to translation functions.

    Returns:
        Translation function that takes a string and returns translated string.

    Raises:
        ValueError: If translation source is invalid or file not found.
    """
    if cc_translations is None:
        cc_translations = {}
    if translate is None:
        return default_translate
    elif callable(translate):
        return translate
    elif isinstance(translate, dict):
        return lambda s, **kwargs: translate.get(s, s)
    elif isinstance(translate, str):
        if translate not in cc_translations or (translate in cc_translations and cc_translations[translate] is None):
            return default_translate
        if os.path.exists(translate):
            ext = os.path.splitext(translate)[1]
            if ext == ".po" or ext == ".pot":
                return _translate_fn_from_po(translate)
            elif ext == ".json":
                td = load_json_cached(translate)
                return lambda s, **kwargs: td.get(s, s)
            else:
                raise ValueError(f"Unknown translation file type: {ext}")
        elif translate in cc_translations:  # country code
            fn = cc_translations[translate]
            assert fn is not None, f"Translation function for {translate} is None"
            return fn
        else:
            raise ValueError(f"Translation file not found: {translate}")


@st.cache_resource(show_spinner=False, ttl=3600)
def _load_po_translations() -> dict[str, Callable[[str], str] | None]:
    """Load all .po translation files from locale directory (cached for 1 hour).

    Returns:
        Dictionary mapping language codes to translation functions (or None for default).
    """
    # Get base filename from __main__
    bname = os.path.splitext(os.path.basename(__main__.__file__))[0]

    # Find all locale subdirectories
    translations: dict[str, Callable[[str], str] | None] = {"en": None}  # English is the default
    if os.path.exists("locale"):
        for country_code in os.listdir("locale"):
            po_path = f"locale/{country_code}/{bname}.po"
            if os.path.exists(po_path):
                _update_manifest(po_path)
                translations[country_code] = _translate_fn_from_po(po_path)

    return translations


# Main dashboard wrapper - WIP
# --------------------------------------------------------
#          OTHER SHARED PARTS
# --------------------------------------------------------


class SalkDashboardBuilder:
    """Main dashboard builder class that orchestrates Streamlit dashboard creation.

    Handles authentication, translation, data loading, plot rendering, and page management.
    """

    def __init__(
        self,
        data_source: str,
        auth_conf: str | None = None,
        logfile: str | None = None,
        groups: list[str] | None = None,
        org_whitelist: list[str] | None = None,
        public: bool = False,
        default_lang: str = "en",
        plot_caching: bool = True,
        header_fn: Callable[..., dict[str, object]] | None = None,
        footer_fn: Callable[..., dict[str, object]] | None = None,
    ) -> None:
        """Initialize the dashboard builder.

        Args:
            data_source: Path to parquet data file.
            auth_conf: Path to authentication configuration file.
            logfile: Path to log file for events.
            groups: List of user groups (default: ["guest", "user", "admin"]).
            org_whitelist: Optional list of allowed organization names.
            public: Whether dashboard is publicly accessible without auth.
            default_lang: Default language code (default: "en").
            plot_caching: Whether to cache plots for performance.
            header_fn: Optional function to render header (receives sdb, returns shared dict).
            footer_fn: Optional function to render footer (receives sdb, shared dict).
        """
        if groups is None:
            groups = ["guest", "user", "admin"]
        # Allow deployment.json to redirect files from local to s3 if local missing (i.e. in deployment scenario)
        if os.path.exists("./deployment.json"):
            dep_meta = load_json_cached("./deployment.json")
            self.filemap = dep_meta.get("files", {})
            # data_source = alias_file(data_source,self.filemap)
            auth_conf = alias_file(auth_conf, self.filemap) if auth_conf else None  # Only needed for old login
        else:
            self.filemap = {}

        self.log_path = alias_file(logfile, self.filemap) if logfile else "log.csv"
        self.s3fs = s3fs.S3FileSystem(anon=False)  # Initialize s3 access. Key in secrets.toml
        self.data_source = data_source
        self.public = public
        self.pages = []
        self.sb_info = st.sidebar.empty()
        self.info = st.empty()
        self.plot_caching = plot_caching
        self.header_fn, self.footer_fn = (
            header_fn,
            footer_fn,
        )  # Header and footer functions

        # Current page name
        self.page_name = None

        # Set up translation
        self.pot_updater = _po_template_updater()
        self.cc_translations = _load_po_translations()
        self.default_lang = default_lang
        login_lang_choice = st.sidebar.empty()

        # print("LANG", st.session_state.get('lang'),
        #       st.session_state.get('chosen_lang'), st.session_state.get('login_lang'))

        # If only one language is available, set it as the one in use
        if len(self.cc_translations) == 1:
            st.session_state["lang"] = next(iter(self.cc_translations.keys()))

        # Don't ask for language in public dashboards
        if not public and not st.secrets.get("auth", {}).get("use_oauth"):
            # This for language select during login page, which is unnecessar
            # Set language from session state if present
            if st.session_state.get("lang"):
                self.set_translate(st.session_state.get("lang"))
            else:  # Alternatively (if on login page) - show the choice at the top of the sidebar
                # This is messy because streamlit is ... not great at this kind of thing
                opts = [self.default_lang] + [lang for lang in self.cc_translations.keys() if lang != self.default_lang]

                # chosen_lang is a temporary variable to store the chosen language on the login page
                clang = st.session_state.get("chosen_lang") or self.default_lang
                self.set_translate(clang)

                def _set_login_lang() -> None:
                    st.session_state["chosen_lang"] = st.session_state["login_lang"]

                ind = opts.index(st.session_state.get("login_lang", self.default_lang))  # FIXES lang not updating
                lang = login_lang_choice.selectbox(
                    self.tf("Language:", context="ui"),
                    opts,
                    index=ind,
                    on_change=_set_login_lang,
                    key="login_lang",
                )
                if lang != clang:
                    self.set_translate(lang)
        else:
            self.set_translate(self.default_lang)

        self.p_widths = {}

        # Set up authentication
        with st.spinner(self.tf("Setting up authentication...", context="ui")):
            if st.secrets.get("auth", {}).get("use_oauth"):
                self.uam = FronteggAuthenticationManager(
                    groups,
                    org_whitelist=org_whitelist,
                    info=self.info,
                    logger=self.log_event,
                    languages=self.cc_translations,
                    translate_func=lambda t: self.tf(t, context="ui"),
                )
            else:
                self.uam = StreamlitAuthenticationManager(
                    auth_conf,
                    groups,
                    org_whitelist=org_whitelist,
                    s3_fs=self.s3fs,
                    info=self.info,
                    logger=self.log_event,
                    languages=self.cc_translations,
                    translate_func=lambda t: self.tf(t, context="ui"),
                )

        if not public:
            self.uam.login_screen()

        # TODO: language handling is overengineered. Remove the complexity
        if self.authenticated and isinstance(self.uam, StreamlitAuthenticationManager):
            login_lang_choice.empty()

            # If user has chosen a language on the login page
            if st.session_state.get("chosen_lang"):
                self.set_translate(st.session_state["chosen_lang"], remember=True)  # Make it persistent
                st.session_state["chosen_lang"] = None  # Only do this once, at login

                # If the value differs from the user's current language, update it
                if st.session_state["lang"] != self.user["lang"]:
                    self.uam.users[st.session_state["username"]]["lang"] = st.session_state["lang"]
                    self.uam.update_user(st.session_state["username"])
            # Load language from user's profile if present
            elif (
                not st.session_state.get("lang") and self.user.get("lang") and self.user["lang"] in self.cc_translations
            ):
                self.set_translate(self.user["lang"], remember=True)
        else:
            self.set_translate(self.user.get("lang"), remember=True)

        _wrap_all_st_functions(st, self, to=self)
        self.sidebar = cast(AbstractContextManager, _wrap_all_st_functions(st.sidebar, self))

    def set_translate(self, lang: str | None, remember: bool = False) -> None:
        """Set the translation language for the dashboard.

        Args:
            lang: Language code to use (falls back to default_lang if invalid).
            remember: Whether to persist language choice in session state.
        """
        if lang is None or lang not in self.cc_translations:
            lang = self.default_lang
        translate = _load_translate(lang, self.cc_translations)
        self.tf = lambda s, **kwargs: translate(self.pot_updater(s, **kwargs))
        if remember:
            st.session_state["lang"] = lang

    # Get the pandas dataframe with given columns

    def get_df(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Get pandas DataFrame with specified columns from lazy Polars DataFrame.

        Args:
            columns: List of column names to select (None for all columns).

        Returns:
            Pandas DataFrame with metadata applied.
        """
        if columns is None:
            q = self.ldf
        else:
            ldf = cast(pl.LazyFrame, self.ldf)
            q = ldf.select(columns)
        ldf_q = cast(pl.LazyFrame, q)
        return fix_df_with_meta(ldf_q.collect().to_pandas(), self.meta)

    # For backwards compatibility - this is very inefficient
    @property
    def df(self) -> pd.DataFrame:
        """Get full DataFrame (backwards compatibility, inefficient).

        Returns:
            Full pandas DataFrame with all columns.

        Note:
            This is inefficient. Use get_df([columns]) instead to get only needed columns.
        """
        utils.warn("sdb.df is very inefficient. Use sdb.get_df([columns]) instead to get only the columns you need")
        return self.get_df()

    # Try to keep uam abstracted away
    @property
    def authenticated(self) -> bool:
        """Check if user is authenticated.

        Returns:
            True if user is authenticated, False otherwise.
        """
        return self.uam.authenticated

    @property
    def admin(self) -> bool:
        """Check if current user is an administrator.

        Returns:
            True if user is an admin, False otherwise.
        """
        return self.uam.admin

    @property
    def user(self) -> dict[str, object]:
        """Get current user data.

        Returns:
            Dictionary with user information (uid, name, group, etc.).
        """
        return self.uam.user

    def log_event(self, event: str, uid: str | None = None) -> None:
        """Log an event to the dashboard log file.

        Args:
            event: Event description string.
            uid: User identifier (defaults to current user's uid).
        """
        log_event(event, uid or self.user["uid"], self.log_path, s3_fs=self.s3fs)

    # pos_id is for plot_width to work in columns
    def plot(
        self, pp_desc: dict[str, object], pos_id: str = "main", width: int | None = None, **kwargs: object
    ) -> None:
        """Render a plot using the plot pipeline.

        Args:
            pp_desc: Plot descriptor dictionary.
            pos_id: Position identifier for width tracking (default: "main").
            width: Optional plot width in pixels (auto-calculated if None).
            **kwargs: Additional arguments passed to st_plot.
        """
        if width is None:  # Find or reuse auto-width
            width = self.p_widths[pos_id] if pos_id in self.p_widths else get_plot_width(pos_id)
            self.p_widths[pos_id] = width

        # If multiple data sources are used, make sure we key it in for caching purposes
        pp_desc["data"] = self.data_source

        # Draw plot
        st_plot(
            pp_desc,
            width=width,
            translate=lambda s: self.tf(s, context="data"),
            plot_cache=plot_cache() if self.plot_caching else None,
            full_df=self.ldf,
            data_meta=self.meta,
            **kwargs,
        )

    def filter_ui(
        self,
        dims: list[str],
        flt: FilterSpec | None = None,
        detailed: bool = False,
        raw: bool = False,
        force_choice: bool = False,
        key: str = "",
    ) -> FilterSpec:
        """Display filter UI and return filter specification.

        Args:
            dims: List of dimension names to create filters for.
            flt: Initial filter values dictionary.
            detailed: Whether to show detailed filter options.
            raw: Whether to use raw column names (skip aliases).
            force_choice: Whether to require user to make a selection.
            key: Optional key for Streamlit widget state.

        Returns:
            Filter specification dictionary.
        """
        if flt is None:
            flt = {}
        return filter_ui(
            self.ldf,
            self.meta,
            uid=f"{key}_{self.page_name}",
            dims=dims,
            flt=flt,
            detailed=detailed,
            raw=raw,
            translate=self.tf,
            force_choice=force_choice,
        )

    def facet_ui(
        self,
        dims: list[str],
        two: bool = False,
        raw: bool = False,
        force_choice: bool = False,
        label: str = "Facet",
        key: str = "",
    ) -> list[str]:
        """Display facet selection UI and return selected dimensions.

        Args:
            dims: List of available dimension names.
            two: Whether to allow selecting two facets.
            raw: Whether to use raw column names (skip aliases).
            force_choice: Whether to require user to make a selection.
            label: Label for the facet selector.
            key: Optional key for Streamlit widget state.

        Returns:
            List of selected facet dimension names.
        """
        return facet_ui(
            dims,
            two=two,
            raw=raw,
            uid=f"{key}_{self.page_name}",
            translate=self.tf,
            force_choice=force_choice,
            label=label,
        )

    def page(self, name: str, **kwargs: object) -> Callable[[Callable[..., dict[str, object] | None]], None]:
        """Decorator to register a page function.

        Args:
            name: Page name displayed in navigation.
            **kwargs: Additional page metadata (e.g., groups, icon, data_source).

        Returns:
            Decorator function that registers the page.
        """

        def _decorator(pfunc: Callable[..., dict[str, object] | None]) -> None:
            # If we have a whitelist of organizations, and the user is not in it, don't show any pages
            # This is the second line of defense as whitelist is also checked in build()
            if self.uam.org_whitelist and self.user.get("organization") not in self.uam.org_whitelist:
                return

            needed_groups = kwargs.get("groups")
            # Check if user can access this page
            can_access = False
            if needed_groups is None:  # Page is available to all
                can_access = True
            elif self.admin:  # Admin sees all
                can_access = True
            else:
                # At this point needed_groups is not None
                needed_groups_list = list(needed_groups) if needed_groups else []
                if "guest" in needed_groups_list:  # some views might be open to all
                    can_access = True
                elif (
                    len(set(self.user.get("groups", [])) & set(needed_groups_list)) > 0
                ):  # one of the groups is whitelisted
                    can_access = True

            if can_access:
                self.pages.append((name, pfunc, kwargs))

        return _decorator

    def build(self) -> None:
        """Build and render the dashboard with navigation and selected page."""
        # This is to avoid a bug of the option menu not showing up on reload
        # I don't get how this row fixes the issue, but it does
        # https://github.com/victoryhb/streamlit-option-menu/issues/68
        # This is a quirk of the old login and should be removed with it
        if (
            isinstance(self.uam, StreamlitAuthenticationManager)
            and st.session_state.get("authentication_status")
            and st.session_state["logout"] is None
        ):
            st.session_state["logout"] = True
            st.rerun()

        # If login failed and is required, don't go any further
        if not self.public and not self.authenticated:
            return

        # Logged in: add info about thtat + log out option
        if self.authenticated:
            with st.sidebar:
                self.sb_info.info(self.tf("Logged in as **%s**", context="ui") % self.user["name"])
                self.uam.logout_button(self.tf("Log out", context="ui"), "sidebar")

        # If we have a whitelist of organizations, and the user is not in it, don't show the page
        if self.uam.org_whitelist and self.user.get("organization") not in self.uam.org_whitelist:
            st.header("You are not authorized to access this dashboard!")
            return

        # Add user settings page if logged in
        if self.authenticated:
            self.pages.append(("Settings", _user_settings_page, {"icon": "sliders"}))

        # Add admin page for admins
        if self.admin:
            self.pages.append(("Administration", admin_page, {"icon": "terminal"}))

        # Draw the menu listing pages
        pnames = [t[0] for t in self.pages]
        with st.sidebar:
            t_pnames = [self.tf(pn, context="ui") for pn in pnames]
            if len(t_pnames) == 1:
                menu_choice = t_pnames[0]
            else:
                menu_choice = option_menu(
                    "Pages",
                    t_pnames,
                    icons=[t[2].get("icon") for t in self.pages],
                    styles={
                        "container": {"padding": "5!important"},  # , "background-color": "#fafafa"},
                        # "icon": {"color": "red", "font-size": "15px"},
                        "nav-link": {
                            "font-size": "12px",
                            "text-align": "left",
                            "margin": "0px",
                            "--hover-color": "#eee",
                        },
                        "nav-link-selected": {"background-color": "#red"},
                        "menu-title": {"display": "none"},
                    },
                )

        # Find the page
        pname, pfunc, meta = self.pages[t_pnames.index(menu_choice)]
        self.page_name = pname

        # Load data
        self.data_source = meta.get("data_source", self.data_source)
        with st.spinner(self.tf("Loading data...", context="ui")):
            _update_manifest(self.data_source)

            # Download the data if it's not already locally present
            # This is done because lazy loading over s3 is very painfully slow as data files are big
            if not os.path.exists(self.data_source):
                print(f"Downloading {self.filemap[self.data_source]} to {self.data_source}")
                self.s3fs.download(self.filemap[self.data_source], self.data_source)

            self.ldf, self.meta = read_parquet_with_data_meta_lazy_cached(self.data_source)
            # self.df = self.ldf.collect().to_pandas() # Backwards compatibility

        # Render the chosen page
        self.subheader(pname, context="ui")

        shared = utils.call_kwsafe(self.header_fn, sdb=self) if self.header_fn else {}
        pres = utils.call_kwsafe(pfunc, sdb=self, shared=shared)
        if pres:
            shared.update(pres)
        if self.footer_fn:
            utils.call_kwsafe(self.footer_fn, sdb=self, shared=shared)

        if self.admin:
            with self.sidebar:
                st.write("Mem: %.1fMb" % (psutil.Process(os.getpid()).memory_info().rss / 1024**2))
                if self.plot_caching:
                    pcache = plot_cache()
                    st.write("Plot cache: %d items (%.1fMb)" % (len(pcache), utils.get_size(pcache) / 1024**2))

                with st.expander("Impersonate (Admin)"):
                    org_list = (self.uam.org_whitelist or []) + (
                        [self.user.get("organization")] if self.user.get("organization") else []
                    )
                    org = st.selectbox(
                        "Organization",
                        org_list,
                        index=org_list.index(self.user.get("organization")),
                    )

                    group = st.selectbox("Group", self.uam.groups, index=self.uam.groups.index("user"))

                    langs = list(self.cc_translations.keys()) + (
                        [self.user.get("lang")] if self.user.get("lang") not in self.cc_translations else []
                    )
                    language = st.selectbox("Language", langs, index=langs.index(self.user["lang"]))
                    if st.button("Impersonate"):
                        st.success("Starting impersonation")
                        if language != self.user["lang"]:
                            self.set_translate(language, remember=True)
                        self.uam.impersonate({"organization": org, "group": group, "lang": language})
                    st.text("Browser refresh clears the impersonation")

    # Add enter and exit so it can be used as a context
    def __enter__(self) -> "SalkDashboardBuilder":
        """Enter context manager."""
        return self

    # Render everything once we exit the with block
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_tb: object | None
    ) -> None:
        """Exit context manager and build dashboard."""
        self.build()


class UserAuthenticationManager:
    """Base class for user authentication managers."""

    def __init__(
        self,
        groups: list[str],
        info: object,
        org_whitelist: list[str] | None,
        logger: Callable[[str, str | None], None],
        languages: dict[str, Callable[[str], str] | None],
        translate_func: Callable[[str], str],
    ) -> None:
        """Initialize authentication manager.

        Args:
            groups: List of user groups.
            info: Streamlit info container for messages.
            org_whitelist: Optional list of allowed organization names.
            logger: Function to log events (event, uid).
            languages: Dictionary mapping language codes to translation functions.
            translate_func: Function to translate UI strings.
        """
        self.groups, self.info = groups, info
        self.org_whitelist = org_whitelist
        self.languages = languages
        self.log_event = logger
        self.tf = translate_func
        self.passwordless = False

        # Mark that we should log the next login
        if "log_event" not in st.session_state:
            st.session_state["log_event"] = True

    @property
    @abstractmethod
    def authenticated(self) -> bool:
        """Check if user is authenticated."""
        pass

    @property
    def admin(self) -> bool:
        """Check if current user is an administrator."""
        return self.authenticated and (self.user.get("group") == "admin")

    def require_admin(self) -> None:
        """Raise exception if current user is not an administrator."""
        if not self.admin:
            raise Exception("This action requires administrator privileges")

    @abstractmethod
    def uam_user(self) -> dict[str, object]:
        """Get user data from authentication system."""
        pass

    @property
    def user(self) -> dict[str, object]:
        """Get current user data (with impersonation support)."""
        base = self.uam_user().copy()
        if st.session_state.get("impersonate_user"):
            base.update(st.session_state["impersonate_user"])
        return base

    @abstractmethod
    def login_screen(self) -> None:
        """Display login screen."""
        pass

    @abstractmethod
    def logout_button(self, text: str, location: str = "sidebar") -> None:
        """Display logout button.

        Args:
            text: Button text.
            location: Where to display button (default: "sidebar").
        """
        pass

    @abstractmethod
    def add_user(self, user_data: dict[str, object]) -> bool:
        """Add a new user.

        Args:
            user_data: Dictionary with user information.

        Returns:
            True if user was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def change_user(self, uid: str, user_data: dict[str, object]) -> None:
        """Update user information.

        Args:
            uid: User identifier.
            user_data: Dictionary with updated user information.
        """
        pass

    @abstractmethod
    def delete_user(self, uid: str) -> None:
        """Delete a user.

        Args:
            uid: User identifier to delete.
        """
        pass

    @abstractmethod
    def list_users(self) -> dict[str, dict[str, object]]:
        """List all users.

        Returns:
            Dictionary mapping user IDs to user data (passwords censored).
        """
        pass

    @abstractmethod
    def update_user(self, uid: str) -> None:
        """Update user in persistent storage.

        Args:
            uid: User identifier.
        """
        pass

    def impersonate(self, user_data: dict[str, object]) -> None:
        """Impersonate another user (admin only).

        Args:
            user_data: Dictionary with user data to impersonate.
        """
        self.require_admin()
        st.session_state["impersonate_user"] = user_data
        st.rerun()


# TODO
# - centralize the db connection, getting url and token from env
# - move other conf (cookie token) to streamlit env variables
# - create a database with username as id and migrate the auth_conf on the web


@st.cache_resource
def sqlite_client(url: str, token: str) -> libsql_client.ClientSync:
    """Create a cached SQLite client connection.

    Args:
        url: Database URL.
        token: Authentication token.

    Returns:
        SQLite client object.
    """
    print(f"User database from {url}")
    return libsql_client.create_client_sync(url=url, auth_token=token)


# --------------------------------------------------------
#          AUTHENTICATION
# --------------------------------------------------------


class StreamlitAuthenticationManager(UserAuthenticationManager):
    """Authentication manager using Streamlit Authenticator and JSON/SQLite storage."""

    def __init__(
        self,
        auth_conf_file: str | None,
        groups: list[str],
        org_whitelist: list[str] | None,
        s3_fs: object,
        info: object,
        logger: Callable[[str, str | None], None],
        languages: dict[str, Callable[[str], str] | None],
        translate_func: Callable[[str], str],
    ) -> None:
        """Initialize Streamlit authentication manager.

        Args:
            auth_conf_file: Path to authentication configuration JSON file.
            groups: List of user groups.
            org_whitelist: Optional list of allowed organization names.
            s3_fs: S3 filesystem instance for remote file access.
            info: Streamlit info container for messages.
            logger: Function to log events (event, uid).
            languages: Dictionary mapping language codes to translation functions.
            translate_func: Function to translate UI strings.
        """
        super().__init__(groups, info, org_whitelist, logger, languages, translate_func)
        self.s3fs = s3_fs
        self.stuser: JsonDict = {}
        self.client: libsql_client.ClientSync | None = None
        self.conf_file = auth_conf_file
        self.conf: JsonDict = {}
        self.load_conf()
        self.passwordless = False
        config = self.conf
        self.auth = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            [],  # config['preauthorized'] - not using preauthorization
        )

    @property
    def authenticated(self) -> bool:
        """Check if user is authenticated via Streamlit Authenticator."""
        return bool(st.session_state.get("authentication_status") and self.stuser)

    # This is abstracted into .user property with impersonation built-in
    def uam_user(self) -> dict[str, object]:
        """Get user data from Streamlit Authenticator session."""
        if self.stuser and self.stuser.get("username"):
            return {
                "uid": self.stuser["username"],
                "name": self.stuser["name"],
                "username": self.stuser["username"],
                "group": self.stuser["group"],
                "organization": self.stuser["organization"],
                "lang": self.stuser["lang"],
            }
        else:
            return {}

    def logout_button(self, text: str, location: str = "sidebar") -> None:
        """Display logout button using Streamlit Authenticator.

        Args:
            text: Button text.
            location: Where to display button (default: "sidebar").
        """
        self.auth.logout(text, location)

    def load_conf(self, cached: bool = True) -> None:
        """Load authentication configuration from file.

        Args:
            cached: Whether to use cached version of config file.
        """
        if cached:
            self.conf = load_json_cached(self.conf_file, _s3_fs=self.s3fs)
        else:
            self.conf = load_json(self.conf_file, _s3_fs=self.s3fs)

        if "libsql" in self.conf:
            url, token = self.conf["libsql"]["url"], self.conf["libsql"]["token"]
            self.client = sqlite_client(url=url, token=token)
            ures = self.client.execute("SELECT * FROM users")
            self.conf["credentials"]["usernames"] = {u["username"]: dict(zip(ures.columns, u)) for u in ures.rows}

        if self.org_whitelist is not None:
            for ud in self.conf["credentials"]["usernames"].values():
                ud["whitelisted"] = ud.get("organization") in self.org_whitelist or ud.get("group") == "admin"

            if not self.admin:
                self.conf["credentials"]["usernames"] = {
                    un: ud for un, ud in self.conf["credentials"]["usernames"].items() if ud.get("whitelisted")
                }

        self.users = self.conf["credentials"]["usernames"]

    def login_screen(self) -> None:
        """Display login screen using Streamlit Authenticator."""
        tf: Callable[[str], str] = self.tf
        _, _, username = self.auth.login(
            "sidebar",
            fields={
                "Form name": tf("Login page"),
                "Username": tf("Username"),
                "Password": tf("Password"),
                "Log in": tf("Log in"),
            },
        )

        if st.session_state["authentication_status"] is False:
            st.error(tf("Username/password is incorrect"))
            self.log_event("login-fail", uid=username)
        if st.session_state["authentication_status"] is None:
            st.warning(tf("Please enter your username and password"))
            st.session_state["log_event"] = True
        elif st.session_state["authentication_status"]:
            self.stuser = {
                "name": st.session_state["name"],
                "username": username,
                **self.users[username],
            }

            # check if signing in has been logged - if not, log it and flip the flag
            if st.session_state["log_event"]:
                self.log_event("login-success")
                st.session_state["log_event"] = False

    def update_conf(self, username: str) -> None:
        """Update authentication configuration file with user changes.

        Args:
            username: Username to update or delete.
        """
        # Read full conf file (can have more users, as load_conf filters them)
        full_conf = load_json(self.conf_file, _s3_fs=self.s3fs)

        full_u = full_conf["credentials"]["usernames"]
        cur_u = self.users

        # Update the user's entry
        if username not in cur_u and username in full_u:
            del full_u[username]  # Delete
        else:
            full_u[username] = cur_u[username]  # Update

        with open_fn(self.conf_file, "w", s3_fs=self.s3fs) as jf:
            json.dump(full_conf, jf)
        time.sleep(3)  # Give some time for messages to display etc
        st.rerun()  # Force a rerun to reload the new file

    def update_user(self, username: str) -> None:
        """Update user in persistent storage (JSON or SQLite).

        Args:
            username: Username to update.
        """
        if "libsql" in self.conf:
            assert self.client is not None, "libsql client should be initialized"
            user_data = self.users[username]
            self.client.execute(
                (
                    "UPDATE users SET name = ?, email = ?, organization = ?, "
                    '"group" = ?, password = ?, lang = ? WHERE username = ?'
                ),
                [
                    user_data["name"],
                    user_data["email"],
                    user_data["organization"],
                    user_data["group"],
                    user_data["password"],
                    user_data["lang"],
                    username,
                ],
            )
        else:
            self.update_conf(username)

    def add_user(self, user_data: dict[str, object]) -> bool:
        """Add a new user to the authentication system.

        Args:
            user_data: Dictionary with user information (username, password, etc.).

        Returns:
            True if user was added successfully, False if username already exists.
        """
        self.require_admin()
        username = user_data["username"]
        if username not in self.users:
            password = user_data.get("password")
            user_data["password"] = stauth.Hasher([password]).generate()[0]
            self.users[username] = user_data
            self.info.success(f"User {username} successfully added.")
            self.log_event(f"add-user: {username}")

            if "libsql" in self.conf:
                assert self.client is not None, "libsql client should be initialized"
                self.client.execute(
                    (
                        "INSERT INTO users (username, name, email, organization, "
                        '"group", password) VALUES (?, ?, ?, ?, ?, ?)'
                    ),
                    [
                        username,
                        user_data["name"],
                        user_data["email"],
                        user_data["organization"],
                        user_data["group"],
                        user_data["password"],
                    ],
                )
            else:
                self.update_conf(username)
            return True
        else:
            self.info.error(f"User **{username}** already exists.")
            return False

    def change_user(self, username: str, user_data: dict[str, object]) -> None:
        """Update user information (including username change).

        Args:
            username: Current username.
            user_data: Dictionary with updated user information.
        """
        # Change username
        if "username" in user_data and username != user_data["username"]:
            self.users[user_data["username"]] = self.users[username]
            del self.users[username]
            username = str(user_data["username"])
            del user_data["username"]

        # Handle password change
        if user_data.get("password"):
            user_data["password"] = stauth.Hasher([user_data["password"]]).generate()[0]
        else:
            user_data["password"] = self.users[username]["password"]

        # Update everything else
        self.users[username].update(user_data)
        self.log_event(f"change-user: {username}")
        self.info.success(f"User **{username}** changed.")
        self.update_user(username)

    def delete_user(self, username: str) -> None:
        """Delete a user from the authentication system.

        Args:
            username: Username to delete.
        """
        self.require_admin()
        del self.users[username]
        self.info.warning(f"User **{username}** deleted.")
        self.log_event(f"delete-user: {username}")

        if "libsql" in self.conf:
            assert self.client is not None, "libsql client should be initialized"
            self.client.execute("DELETE FROM users WHERE username = ?", [username])
        else:
            self.update_conf(username)

    def list_users(self) -> dict[str, dict[str, object]]:
        """List all users (passwords censored).

        Returns:
            Dictionary mapping usernames to user data (passwords removed).
        """
        self.require_admin()
        self.load_conf(cached=False)  # so all admin updates would immediately be visible
        return {k: utils.censor_dict({"uid": k, **v}, ["password"]) for k, v in self.users.items()}


@st.cache_resource
def frontegg_client() -> object:
    """Create a cached Frontegg HTTP client.

    Returns:
        Frontegg HttpClient instance.
    """
    from frontegg.common.clients import HttpClient  # type: ignore[import-untyped]

    base_url = "https://api.frontegg.com/audits"
    auth = st.secrets["auth"]
    return HttpClient(client_id=auth["client_id"], api_key=auth["client_secret"], base_url=base_url)


class FronteggAuthenticationManager(UserAuthenticationManager):
    """Authentication manager using Frontegg OAuth."""

    def __init__(
        self,
        groups: list[str],
        info: object,
        org_whitelist: list[str] | None,
        logger: Callable[[str, str | None], None],
        languages: dict[str, Callable[[str], str] | None],
        translate_func: Callable[[str], str],
    ) -> None:
        """Initialize Frontegg authentication manager.

        Args:
            groups: List of user groups.
            info: Streamlit info container for messages.
            org_whitelist: Optional list of allowed organization names.
            logger: Function to log events (event, uid).
            languages: Dictionary mapping language codes to translation functions.
            translate_func: Function to translate UI strings.
        """
        super().__init__(groups, info, org_whitelist, logger, languages, translate_func)
        self.client = frontegg_client()
        self.passwordless = True

    @property
    def authenticated(self) -> bool:
        """Check if user is authenticated via Frontegg OAuth."""
        logged_in = st.user["is_logged_in"]
        return bool(logged_in)

    def reform_user(self, user: dict[str, object]) -> dict[str, object]:
        """Reformat Frontegg user data to internal format.

        Args:
            user: Raw user data from Frontegg API.

        Returns:
            Reformatted user dictionary.
        """
        meta_raw = user.get("metadata") or {}
        if isinstance(meta_raw, str):
            meta_raw = json.loads(meta_raw)
        meta = cast(dict[str, Any], meta_raw)
        return {
            "uid": user["email"],
            "name": user.get("name", ""),
            "email": user["email"],
            #'username': st.user['cognito:username'],
            "group": meta.get("group", "guest"),
            "organization": meta.get("organization"),
            "lang": meta.get("lang"),
        }

    # This is abstracted into .user property with impersonation built-in
    def uam_user(self) -> dict[str, object]:
        """Get user data from Frontegg OAuth session.

        Note: As st.user is not updated unless you log out, and is not writable
        """
        # We need the hacky workaround to allow changing user info (like lang) during session
        # Normally, one would just do an oauth refresh on user change, but streamlit does not support that (yet?)

        if not st.user["is_logged_in"]:
            return {}
        elif "OAUser" not in st.session_state:
            st.session_state["OAUser"] = self.reform_user(st.user)
        user = st.session_state.get("OAUser")
        assert isinstance(user, dict), "Expected OAUser to be a dict"
        return user

    # Use the silent login profile to just refresh the user info if prompt config is present

    def refresh_user(self) -> None:
        """Refresh user information by triggering OAuth re-authentication."""
        # If prompr config present, assume default conf is silent login
        # In that case, logout + silent login can be used to refresh the user info
        if "prompt" in st.secrets["auth"]:
            st.logout()
        else:
            print("User refresh needs [auth.prompt] segment in secrets.toml")

    def login_screen(self) -> None:
        """Display Frontegg OAuth login screen."""
        if not self.authenticated:
            st.login()
            st.session_state["OA_fresh"] = True  # just logged in, so no need to refresh
            if st.user["is_logged_in"] and not st.session_state.get("OAUser"):
                st.session_state["OAUser"] = self.reform_user(st.user)
            elif not st.user["is_logged_in"] and "OAUser" in st.session_state:
                del st.session_state["OAUser"]
        elif (
            not st.session_state.get("OA_fresh")
            and isinstance(st.user.get("iat"), (int, float))
            and time.time() - float(st.user["iat"]) > 60
        ):
            # IF authenticated, but token not refreshed this session and is at least 60 sec old
            # This is to ensure that settings changes are also visible if logging in on another device
            # Also good for keeping users logged in for a while as it refreshes the access token
            print("Refreshing user info")
            self.refresh_user()
        else:
            st.session_state["OA_fresh"] = True

        # Record the login event to the log file
        if self.authenticated and "login_recorded" not in st.session_state:
            self.log_event("login-success")
            st.session_state["login_recorded"] = True  # Only log once per session, even if user logs in multiple times

    def logout_button(self, text: str, location: str = "sidebar") -> None:
        """Display logout button that triggers OAuth re-login.

        Args:
            text: Button text.
            location: Where to display button (default: "sidebar").
        """
        if "prompt" in st.secrets["auth"] and st.button(text):
            st.login("prompt")
        # st.write(self.reform_user(st.user)) # Debug: show sdb.user

    def add_user(self, user_data: dict[str, object]) -> bool:
        """Add a new user via Frontegg API.

        Args:
            user_data: Dictionary with user information.

        Returns:
            True if user was added successfully.
        """
        self.require_admin()
        res = self.client.post(
            url="identity/resources/users/v1/",
            data={
                "email": user_data["email"].strip(),
                "verified": True,
                "name": user_data["name"].strip(),
                "metadata": json.dumps(
                    {
                        "group": user_data["group"],
                        "organization": user_data["organization"],
                        "lang": user_data["lang"],
                    }
                ),
                "roleIds": [],
            },
            headers={"frontegg-tenant-id": st.user["tenantId"]},
        ).json()  # Add to same tenant as admin

        if res.get("errors"):
            raise Exception(str(res["errors"]))
        self.info.info(f"User **{res['email']}** added.")
        self.log_event(f"add-user: {res['email']}")
        return True

    def change_user(self, uid: str, user_data: dict[str, object]) -> None:
        """Update user information via Frontegg API.

        Args:
            uid: User email/identifier.
            user_data: Dictionary with updated user information.
        """
        if uid != self.user["uid"]:
            self.require_admin()
        uinfo = self.client.get(f"identity/resources/users/v1/email?email={uid}").json()
        if "email" in user_data and user_data["email"] != uinfo["email"]:
            eres = self.client.put(
                url=f"identity/resources/users/v1/{uinfo['id']}/email",
                data={"email": user_data["email"].strip()},
            )
            if eres.get("errors"):
                raise Exception(str(eres["errors"]))

        # Change everything else:
        res = self.client.put(
            url=f"identity/resources/users/v1/{uinfo['id']}",
            data={
                "name": user_data["name"],
                "metadata": json.dumps(
                    {
                        "group": user_data["group"],
                        "organization": user_data["organization"],
                        "lang": user_data["lang"],
                    }
                ),
            },
        ).json()

        if res.get("errors"):
            raise Exception(str(res["errors"]))
        self.info.info(f"User **{uid}** updated.")
        self.log_event(f"change-user: {uid}")

        if uid == self.user["uid"]:
            self.refresh_user()
            # st.session_state['OAUser'] = self.reform_user(res)

    def delete_user(self, uid: str) -> None:
        """Delete a user via Frontegg API.

        Args:
            uid: User email/identifier to delete.
        """
        self.require_admin()
        uinfo = self.client.get(f"identity/resources/users/v1/email?email={uid}").json()
        self.client.delete(f"identity/resources/users/v1/{uinfo['id']}")

        # if res.get('error'): raise Exception(res['error'])
        self.info.info(f"User **{uid}** deleted.")
        self.log_event(f"delete-user: {uid}")

    def list_users(self) -> dict[str, dict[str, object]]:
        """List all users from Frontegg API (passwords censored).

        Returns:
            Dictionary mapping user emails to user data (passwords removed).

        Note:
            This endpoint is paginated, so we may need to cycle over all pages.
        """
        self.require_admin()
        # TODO: this endpoint is paginated, so we may need to cycle over all pages here
        res = self.client.get(
            "identity/resources/users/v1/?_limit=200",
            headers={"frontegg-tenant-id": "7779b9fb-f279-4cd3-8f61-e751a0d06145"},
        )
        return {i["email"]: utils.censor_dict(self.reform_user(i), []) for i in res.json()["items"]}


# Password reset
def _user_settings_page(sdb: SalkDashboardBuilder) -> None:
    """Display user settings page with language selection and password reset.

    Args:
        sdb: Dashboard builder instance.
    """
    if not sdb.user:
        return

    cur_lang = st.session_state.get("lang")
    opts = [cur_lang] + [lang for lang in sdb.cc_translations.keys() if lang != cur_lang]
    lang = st.selectbox(sdb.tf("Language:", context="ui"), opts)

    if sdb.button("Save"):
        if lang != cur_lang:
            sdb.set_translate(lang, remember=True)

        if lang != sdb.user["lang"]:
            user = sdb.user.copy()
            user["lang"] = lang
            sdb.uam.change_user(sdb.user["uid"], user)
        st.rerun()

    if isinstance(sdb.uam, StreamlitAuthenticationManager):
        try:
            tf = lambda s: sdb.tf(s, context="ui")

            if sdb.uam.auth.reset_password(
                st.session_state["username"],
                fields={
                    "Form name": tf("Reset password"),
                    "Current password": tf("Current password"),
                    "New password": tf("New password"),
                    "Repeat password": tf("Repeat password"),
                    "Reset": tf("Reset"),
                },
            ):
                sdb.uam.update_user(st.session_state["username"])
                st.success(tf("Password modified successfully"))
        except Exception as e:
            st.error(e)


# Helper function to highlight log rows
def _highlight_cells(val: str) -> str:
    """Return CSS color style for log event cells based on event type.

    Args:
        val: Event string to analyze.

    Returns:
        CSS color style string (e.g., "color: red").
    """
    if "fail" in val:
        color = "red"
    # elif 'add' in val:
    elif any(s in val for s in ["delete", "add", "change"]):
        color = "blue"
    elif "success" in val:
        color = "green"
    else:
        color = ""
    return "color: {}".format(color)


# Admin page to manage users

# --------------------------------------------------------
#          ADMIN PAGES
# --------------------------------------------------------


def admin_page(sdb: SalkDashboardBuilder) -> None:
    """Display admin page for user management and log viewing.

    Args:
        sdb: Dashboard builder instance.
    """
    sdb.uam.require_admin()

    menu_choice = option_menu(
        None,
        ["Log management", "List users", "Add user", "Change user", "Delete user"],
        icons=[
            "card-list",
            "people-fill",
            "person-fill-add",
            "person-lines-fill",
            "person-fill-dash",
        ],
        orientation="horizontal",
    )
    st.write(" ")

    all_users = sdb.uam.list_users()

    if menu_choice == "Log management":
        log_data = pd.read_csv(alias_file(sdb.log_path, sdb.filemap), names=["timestamp", "event", "uid"])
        st.dataframe(
            log_data.sort_index(ascending=False).style.map(_highlight_cells, subset=["event"]),
            width=1200,
        )  # use_container_width=True

    elif menu_choice == "List users":
        # Read log to get last login:
        log_data = pd.read_csv(alias_file(sdb.log_path, sdb.filemap), names=["timestamp", "event", "uid"])
        log_data = log_data[log_data["event"] == "login-success"]
        log_data["timestamp"] = pd.to_datetime(log_data["timestamp"], utc=True, format="%d-%m-%Y, %H:%M:%S")

        # Add last login to users
        users = list(all_users.values())
        for u in users:
            last_login = log_data[log_data["uid"] == u["uid"]].timestamp.max()
            u["last_login"] = last_login if pd.notnull(last_login) else None

        if sdb.uam.org_whitelist is not None:
            for ud in users:
                ud["whitelisted"] = ud.get("organization") in sdb.uam.org_whitelist or ud.get("group") == "admin"

        users = pd.DataFrame(users)
        users["last_login"] = pd.to_datetime(users["last_login"])
        users = users.sort_values(by=["whitelisted", "last_login"], ascending=False)

        # Display the data
        st.dataframe(
            users,
            use_container_width=True,
            column_config={"last_login": st.column_config.DatetimeColumn("last_login", format="D MMM YYYY, HH:mm ")},
        )

    elif menu_choice == "Add user":
        with st.form("add_user_form"):
            st.subheader("Add user:")
            st.markdown("""---""")
            col1, col2 = st.columns((1, 2))
            user_data = {}
            with col1:
                user_data["group"] = st.radio("Group:", sdb.uam.groups)
            with col2:
                if not sdb.uam.passwordless:
                    username = st.text_input("Username:")
                    password = st.text_input("Password:", type="password")

                user_data["name"] = st.text_input("Name:")
                st.markdown("""---""")
                user_data["email"] = st.text_input("E-mail:")
                user_data["organization"] = st.text_input("Organization:")
                user_data["lang"] = st.selectbox("Language:", list(sdb.cc_translations.keys()) + [None], index=0)
            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not sdb.uam.passwordless:
                    if username in all_users:
                        sdb.info.error(f"User **{username}** already exists.")
                    elif "" not in [username, password, user_data["email"]]:
                        user_data["username"] = username
                        user_data["password"] = password
                        sdb.uam.add_user(user_data)
                    else:
                        sdb.info.error("Must specify username, password and email.")
                else:
                    if user_data["email"] in all_users:
                        sdb.info.error(f"User **{user_data['email']}** already exists.")
                    elif "" in [user_data["email"]]:
                        sdb.info.error("Must specify email.")
                    else:
                        sdb.uam.add_user(user_data)

    elif menu_choice == "Change user":
        uid = st.selectbox("Edit user", list(all_users.keys()))

        user_data = all_users[uid].copy()
        # st.write(user_data)
        group_index = sdb.uam.groups.index(user_data.get("group", "guest"))

        with st.form("edit_user_form"):
            st.subheader("Edit user data:")
            st.markdown("""---""")
            col1, col2 = st.columns((1, 2))
            with col1:
                if not sdb.uam.passwordless:
                    user_data["username"] = st.text_input("Username:", value=user_data["username"], disabled=True)
                user_data["group"] = st.radio("Group:", sdb.uam.groups, index=group_index)  # , disabled=True)
            with col2:
                # new_user = st.text_input("Kasutaja:", value=username, disabled=True)
                user_data["name"] = st.text_input("Name:", value=user_data["name"])
                if not sdb.uam.passwordless:
                    user_data["password"] = st.text_input("Password:", type="password")
                st.markdown("""---""")
                user_data["email"] = st.text_input("E-mail:", value=user_data["email"])
                user_data["organization"] = st.text_input("Organization:", value=user_data.get("organization", ""))
                cur_lang = user_data.get("lang", None)
                l_opts = [cur_lang] + [lang for lang in list(sdb.cc_translations.keys()) + [None] if lang != cur_lang]
                user_data["lang"] = st.selectbox("Language:", l_opts)
                # NB! it is known changing language here for current user does not lead to a change.
                # It's not worth the extra code overhead to make it work.

            st.markdown("""---""")
            submitted = st.form_submit_button("Submit")
            if submitted:
                sdb.uam.change_user(uid, user_data)

    elif menu_choice == "Delete user":
        with st.form("delete_user_form"):
            st.subheader("Delete user:")
            uid = st.selectbox("Select username:", list(all_users.keys()))
            check = st.checkbox("Deletion is FINAL and cannot be undone!")
            st.markdown("""___""")
            submitted = st.form_submit_button("Delete")
            if submitted:
                if not check:
                    sdb.warning(f"Tick the checkbox in order to delete user **{uid}**.")
                elif uid == sdb.uam.user["uid"]:
                    sdb.error("Cannot delete the current user.")
                else:
                    sdb.uam.delete_user(uid)


# This is a horrible workaround to get faceting to work with altair geoplots that do not play well with streamlit
# See https://github.com/altair-viz/altair/issues/2369 -> https://github.com/vega/vega-lite/issues/3729


def draw_plot_matrix(pmat: list[list[object]] | object | None) -> None:
    """Draw a matrix of Altair plots in Streamlit columns.

    Args:
        pmat: Matrix of plots (list of lists) or single plot, or None.
    """
    if not pmat:
        return  # Do nothing if get None passed to it
    if not isinstance(pmat, list):
        pmat, ucw = [[pmat]], False
    else:
        ucw = True  # If we are drawing more than one plot, we want to use the container width
    cols = st.columns(len(pmat[0])) if len(pmat[0]) > 1 else [st]
    for j, c in enumerate(cols):
        for i, row in enumerate(pmat):
            if j >= len(pmat[i]):
                continue
            # print(pmat[i][j].to_json()) # to debug json
            c.altair_chart(pmat[i][j], use_container_width=ucw)  # ,theme=None)


def st_plot(pp_desc: dict[str, object], **kwargs: object) -> None:
    """Draw a plot using the end-to-end plot pipeline.

    Args:
        pp_desc: Plot descriptor dictionary.
        **kwargs: Additional arguments passed to e2e_plot.
    """
    plots = e2e_plot(pp_desc, **kwargs)
    draw_plot_matrix(plots)


# Create a global plot cache


@st.cache_resource(show_spinner=False, ttl=None)
def plot_cache() -> object:
    """Create a global plot cache (cached resource).

    Returns:
        Dictionary cache object for storing plots.
    """
    return utils.dict_cache(size=100)


def stss_safety(key: str, opts: list[object]) -> None:
    """Streamlit session state safety - check and clear session state if it has an unfit value.

    Clear session state key if its value is not in the allowed options.

    Args:
        key: Session state key to check.
        opts: List of allowed values.
    """
    if key in st.session_state and st.session_state[key] not in opts:
        del st.session_state[key]


def facet_ui(
    dims: list[str],
    two: bool = False,
    uid: str = "base",
    raw: bool = False,
    translate: Callable[[str], str] | None = None,
    force_choice: bool = False,
    label: str = "Facet",
) -> list[str]:
    """Display facet selection UI and return selected dimensions.

    Args:
        dims: List of available dimension names.
        two: Whether to allow selecting two facets.
        uid: Unique identifier for widget state.
        raw: Whether to use raw column names (skip aliases).
        _translate: Optional translation function.
        force_choice: Whether to require user to make a selection.
        label: Label for the facet selector.

    Returns:
        List of selected facet dimension names.
    """
    # Set up translation
    tfc = translate if translate else (lambda s, **kwargs: s)
    tf = lambda s: tfc(s, context="data")

    tdims = [tf(d) for d in dims]
    r_map = dict(zip(tdims, dims))

    none = tf("None")
    stc = st.sidebar if not raw else st

    stss_safety(f"facet1_{uid}", tdims)
    facet_dim = stc.selectbox(
        tfc(label + ":", context="ui"),
        tdims if force_choice else [none] + tdims,
        key=f"facet1_{uid}",
    )
    fcols = [facet_dim] if facet_dim != none else []
    if two and facet_dim != none:
        stss_safety(f"facet2_{uid}", tdims)
        second_dim = stc.selectbox(
            tfc(label + " 2:", context="ui"),
            tdims if force_choice else [none] + tdims,
            key=f"facet2_{uid}",
        )
        if second_dim not in [none, facet_dim]:
            fcols = [facet_dim, second_dim]

    return [r_map[d] for d in fcols]


def _ms_reset(cn: str, all_vals: list[object], uid: str) -> Callable[[], None]:
    """Create a reset function for a multiselect filter.

    Args:
        cn: Column name.
        all_vals: List of all possible values to reset to.
        uid: Unique identifier for widget state.

    Returns:
        Function that resets the multiselect to all values.
    """

    def _reset_ms() -> None:
        st.session_state[f"filter_{uid}_{cn}_multiselect"] = all_vals

    return _reset_ms


@st.cache_data(
    show_spinner=False,
    hash_funcs={DataMeta: lambda x: hash(x.model_dump_json()) if x is not None else None},
)
def _get_filter_limits(
    _ldf: pl.LazyFrame | pd.DataFrame,
    dims: list[str] | None,
    dmeta: DataMeta | None,
    uid: str,
) -> dict[str, dict[str, object]]:
    """Get filter limits (min/max for continuous, categories for categorical) for dimensions.

    Args:
        _ldf: LazyFrame or DataFrame to analyze.
        dims: List of dimension names to get limits for (None for all).
        dmeta: Data metadata dictionary.
        uid: Unique identifier for caching.

    Returns:
        Dictionary mapping dimension names to their filter limits.
    """
    ldf = _ldf

    if not isinstance(ldf, pl.LazyFrame):
        ldf = pl.DataFrame(ldf).lazy()

    c_meta = extract_column_meta(dmeta)
    gcols = group_columns_dict(dmeta)
    schema = ldf.collect_schema()

    if dims is None:
        dims = schema.names()
    else:
        dims = [c for c in dims if c in schema.names() or c in gcols]

    limits = {}
    for d in dims:
        if d in gcols:  # Block group
            meta_d = c_meta[d]
            prefix = meta_d.col_prefix or ""
            limits[d] = {
                "categories": [c.removeprefix(prefix) for c in gcols[d]],
                "group": True,
            }
        elif d in c_meta:  # Individual column
            meta_d = c_meta[d]
            if meta_d.continuous and schema[d].is_numeric():
                if meta_d.val_range:
                    val_range = cast(list[int | float], list(meta_d.val_range))
                    limits[d] = {
                        "min": val_range[0],
                        "max": val_range[1],
                    }
                else:
                    limits[d] = ldf.select([pl.min(d).alias("min"), pl.max(d).alias("max")]).collect().to_dicts()[0]
                limits[d]["continuous"] = True
            elif meta_d.categories:
                if meta_d.categories == "infer":
                    if schema[d].is_numeric():
                        utils.warn(
                            f"Column {d} is numeric but marked as categorical. "
                            "Skipping in filter as inferring categories is not possible."
                        )
                        continue
                    else:
                        categories = ldf.select(pl.all()).unique(d).collect().to_series().sort().to_list()
                        limits[d] = {"categories": categories}
                else:
                    limits[d] = {"categories": meta_d.categories}

                limits[d]["ordered"] = meta_d.ordered
            else:
                utils.warn(f"Skipping {d}: {meta_d} in filter")
    return limits


def filter_ui(
    data: pl.LazyFrame | pd.DataFrame,
    dmeta: DataMeta | None = None,
    dims: list[str] | None = None,
    flt: FilterSpec | None = None,
    uid: str = "base",
    detailed: bool = False,
    raw: bool = False,
    translate: Callable[[str], str] | None = None,
    force_choice: bool = False,
    grouped: bool = False,
    obs_dim: str | None = None,
) -> FilterSpec:
    """Display filter UI and return filter specification for the pp_desc.

    Args:
        data: LazyFrame or DataFrame to filter.
        dmeta: Data metadata dictionary.
        dims: List of dimension names to create filters for (None for all).
        flt: Initial filter values dictionary.
        uid: Unique identifier for widget state.
        detailed: Whether to show detailed filter options.
        raw: Whether to use raw column names (skip aliases).
        _translate: Optional translation function.
        force_choice: Whether to require user to make a selection.
        grouped: Whether to group filters by metadata groups.
        obs_dim: Optional observation dimension name.

    Returns:
        Filter specification dictionary.
    """
    if flt is None:
        flt = {}
    tfc = translate if translate else (lambda s, **kwargs: s)

    tf = lambda s: tfc(s, context="data")

    limits = _get_filter_limits(data, dims, dmeta, uid)
    dims = list(limits.keys())

    if dmeta is not None:
        gcols = group_columns_dict(dmeta)
        if not grouped:
            dims = list_aliases(dims, gcols)  # Replace aliases like 'demographics'
        c_meta = extract_column_meta(dmeta)  # mainly for groups defined in meta
    else:
        c_meta = defaultdict(lambda: GroupOrColumnMeta())

    if not force_choice:
        f_info = st.sidebar.container()

    gstc = st.sidebar.expander(tfc("Filters", context="ui")) if not raw else st
    stss = st.session_state

    if grouped:
        gdims = {gn: [d for d in [gn] + gdims if d in dims] for gn, gdims in gcols.items()}
        gdims = [(gstc.expander(gn, expanded=(gn == "main")), gd) for gn, gd in gdims.items() if len(gd) > 0]
    else:
        gdims = [(gstc, dims)]

    # Different selector for different category types
    # Also - make sure filter is clean and only applies when it is changed from the default 'all' value
    # This has considerable speed and efficiency implications
    filters = deepcopy(flt)
    for stc, dims in gdims:
        for cn in dims:
            # If filter on this dimension already set, skip
            if cn in filters:
                continue

            # Shared prep for all cateogoricals
            if limits[cn].get("categories"):
                cats = cast(list[object], limits[cn]["categories"])

                if cn in flt:  # Already a filter set
                    cflt: list[object] = flt[cn]  # type: ignore[assignment]
                    if not isinstance(cflt, list):
                        cflt = [cflt]  # Single value
                    if cflt[0] is None:
                        miv, mav = cflt[1:]
                        if not {miv, mav} <= set(cats):
                            raise ValueError(f"Invalid filter for {cn}: {cflt}")
                        cflt = cats[cats.index(miv) : cats.index(mav) + 1]
                    cats = cflt  # Set the list of options to the current filter

                if len(cats) == 1:
                    continue

                # Do some prep for translations
                r_map = dict(zip([tf(c) for c in cats], cats))
                all_vals = list(r_map.keys())  # translated categories
                grp_groups = c_meta[cn].groups or {}
                grp_names = list(grp_groups.keys()) if isinstance(grp_groups, dict) else []
                r_map.update(dict(zip([tf(c) for c in grp_names], grp_names)))

            # Multiselect
            if (detailed or cn in gcols) and limits[cn].get("categories"):
                key = f"filter_{uid}_{cn}_multiselect"
                if key in stss and not set(stss[key]) <= set(all_vals):
                    del stss[key]
                filters[cn] = stc.multiselect(tf(cn), all_vals, all_vals, key=key)
                if set(filters[cn]) == set(all_vals):
                    del filters[cn]
                else:
                    stc.button(
                        tf("Reset"),
                        key=f"filter_{uid}_{cn}_ms_reset",
                        on_click=_ms_reset(cn, all_vals, uid),
                    )
                    filters[cn] = [r_map[c] for c in cast(list[str], filters[cn])]

            # Unordered categorical - selectbox
            elif limits[cn].get("categories") and not limits[cn].get("ordered"):
                choices = [gt for gt, g in r_map.items() if g in grp_names] + all_vals
                if not force_choice:
                    choices = [tf("All")] + choices
                stss_safety(f"filter_{cn}_sel", choices)
                key = f"filter_{uid}_{cn}_sel"
                if key in stss and stss[key] not in all_vals:
                    del stss[key]
                filters[cn] = stc.selectbox(tf(cn), choices, key=key)
                if filters[cn] == tf("All"):
                    del filters[cn]
                else:
                    filters[cn] = r_map[filters[cn]]

            # Ordered categorical - slider
            # Use [None,<start>,<end>] for ranges, both categorical and continuous
            # to distinguish them from list of values
            elif limits[cn].get("categories") and limits[cn].get("ordered"):  # Ordered categorical - slider
                key = f"filter_{uid}_{cn}_ocat"
                if key in stss and not set(stss[key]) <= set(all_vals):
                    del stss[key]
                f_res = stc.select_slider(tf(cn), all_vals, value=(all_vals[0], all_vals[-1]), key=key)
                if f_res != (all_vals[0], all_vals[-1]):
                    miv, mav = r_map[f_res[0]], r_map[f_res[1]]
                    if cn in flt:
                        filters[cn] = cats[
                            cats.index(miv) : cats.index(mav) + 1
                        ]  # As cats itself might already be a subset
                    else:
                        filters[cn] = [None] + [
                            miv,
                            mav,
                        ]  # Just use the range syntax for better legibility

            # Numeric values - slider
            elif limits[cn].get("continuous"):  # Continuous
                mima = limits[cn]["min"], limits[cn]["max"]
                if mima[0] == mima[1]:
                    continue
                f_res = stc.slider(tf(cn), *mima, value=mima, key=f"filter_{uid}_{cn}_cont")
                if f_res[0] > mima[0] or f_res[1] < mima[1]:
                    filters[cn] = (
                        [None] + [f_res[0] if f_res[0] > mima[0] else None] + [f_res[1] if f_res[1] < mima[1] else None]
                    )

    # Only leave the question group filter on if that is the current observation
    if obs_dim:
        filters = {k: v for k, v in filters.items() if not limits[k].get("group") or k == obs_dim}

    if filters != flt and not force_choice:
        f_info.warning("⚠️ " + tfc("Filters active", context="ui") + " ⚠️")

    return filters


# --------------------------------------------------------
#          TRANSLATION TOOLS
# --------------------------------------------------------

# Use dict here as dicts are ordered as of Python 3.7 and preserving order groups things together better


def translate_with_dict(d: dict[str, str]) -> Callable[[str], str]:
    """Create a translation function from a dictionary.

    Args:
        d: Dictionary mapping source strings to translated strings.

    Returns:
        Translation function that looks up strings in the dictionary.
    """
    return lambda s: d[s] if isinstance(s, str) and s in d and d[s] is not None else s


def log_missing_translations(tf: Callable[[str], str], nonchanged_dict: dict[str, object]) -> Callable[[str], str]:
    """Wrap a translation function to log untranslated strings.

    Args:
        tf: Translation function to wrap.
        nonchanged_dict: Dictionary to populate with untranslated strings.

    Returns:
        Wrapped translation function that logs missing translations.
    """

    def _ntf(s: str) -> str:
        ns = tf(s)
        if ns == s:
            nonchanged_dict[s] = None
        return ns

    return _ntf


def clean_missing_translations(
    nonchanged_dict: dict[str, object], tdict: dict[str, str] | None = None
) -> dict[str, object]:
    """Filter missing translations to remove numbers and already-translated strings.

    Args:
        nonchanged_dict: Dictionary of untranslated strings.
        tdict: Dictionary of existing translations to exclude.

    Returns:
        Filtered dictionary of missing translations.
    """
    if tdict is None:
        tdict = {}
    # Filter out numbers that come in from data sometimes
    return {
        s: v
        for s, v in nonchanged_dict.items()
        if s not in tdict and isinstance(s, str) and not re.fullmatch(r"[.\d]+", s)
    }


def add_missing_to_dict(missing_dict: dict[str, object], tdict: dict[str, str]) -> dict[str, str]:
    """Add missing translations to dictionary with identity mapping.

    Args:
        missing_dict: Dictionary of missing translation keys.
        tdict: Existing translation dictionary.

    Returns:
        Updated translation dictionary with missing keys added as identity mappings.
    """
    return {**tdict, **{s: s for s in missing_dict}}


def translate_pot(template: str, dest: str, tfunc: Callable[[str], str], sources: list[str] | None = None) -> None:
    """Translate a .pot file using a custom translation function.

    Args:
        template: Path to source .pot template file.
        dest: Path where translated .po file should be written.
        tfunc: Translation function that takes source string and returns translated string.
        sources: Optional list of existing .po files to use as translation sources.
    """
    if sources is None:
        sources = []
    pot = polib.pofile(template)

    if os.path.exists(dest):
        po = polib.pofile(dest)
    else:
        po = polib.POFile()
        po.metadata = pot.metadata

    todo = {(entry.msgctx, entry.msgid) for entry in pot}
    existing = {(entry.msgctx, entry.msgid) for entry in po}

    from tqdm import tqdm

    # Go through sources and add translations found there to the pot
    if sources:
        n_existing = len(existing)
        for source in tqdm(sources, desc="Checking existing translations"):
            spo = polib.pofile(source)
            for entry in spo:
                if (entry.msgctx, entry.msgid) not in todo:
                    continue
                if (entry.msgctx, entry.msgid) in existing:
                    continue

                if entry.msgstr:
                    entry.msgstr = tfunc(entry.msgstr)
                po.append(entry)
                existing.add((entry.msgctx, entry.msgid))

        n_found = len(existing) - n_existing
        if n_found:
            print(f"Found {n_found} translations in {sources}")

    n = len(pot) - len(existing)
    progress = tqdm(pot, desc="Translating", total=n)

    for entry in pot:
        if (entry.msgctx, entry.msgid) in existing:
            continue

        if not entry.msgstr:
            continue
        entry.msgstr = tfunc(entry.msgstr)
        po.append(entry)

        progress.update(1)

    progress.close()

    po.save(dest)
