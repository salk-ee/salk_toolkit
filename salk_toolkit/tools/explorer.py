"""Interactive data exploration tool built on Streamlit and the plot pipeline.

This module provides a web-based interface for exploring annotated survey data,
allowing users to interactively create plots, filter data, and adjust plot parameters.
"""

import warnings

import streamlit as st

warnings.simplefilter(action="ignore", category=FutureWarning)

# If true, the profiler will be shown
profile = False
memprofile = False

if memprofile:
    import tracemalloc

    tracemalloc.start()

if profile:
    from wfork_streamlit_profiler import Profiler  # type: ignore[import-untyped]

    p = Profiler()
    p.start()

st.set_page_config(
    layout="wide",
    page_title="SALK Explorer",
    # page_icon="./s-model.png",
    initial_sidebar_state="expanded",
)

info = st.empty()

try:
    s = st.secrets.get("sip", {})
except FileNotFoundError:
    has_secrets = False
else:
    has_secrets = True

# Allow explorer to be deployed online with access restrictions similar to other dashboards
if has_secrets and st.secrets.get("sip", {}).get("input_files"):  # st.secrets.get('auth',{}).get('use_oauth'):
    import s3fs  # type: ignore[import-untyped]

    from salk_toolkit.dashboard import FronteggAuthenticationManager, log_event

    groups = ["user", "admin"]
    org_whitelist = st.secrets["sip"]["org_whitelist"]

    # Set up logging
    s3fs = s3fs.S3FileSystem(anon=False)
    uam = None
    log_path = st.secrets["sip"]["log_path"]

    def logger(event: str, uid: str | None = None) -> None:
        """Log an event to S3."""
        user_uid = uam.user.get("uid", "anonymous") if uam and uam.user else "anonymous"
        log_event(event, uid or user_uid, log_path, s3_fs=s3fs)

    uam = FronteggAuthenticationManager(
        groups,
        org_whitelist=org_whitelist,
        info=info,
        logger=logger,
        languages={},
        translate_func=lambda t: t,
    )

    uam.login_screen()
    if not uam.authenticated:
        st.stop()  # Wait for login redirect to happen
    elif uam.user.get("organization") not in org_whitelist:  # Logged in but not authorized
        st.header("You are not authorized to access this dashboard!")
        st.stop()
    # else:
    #     with st.sidebar:
    #         uam.logout_button('Logout','sidebar')

    # Dummy SDB for admin panel
    sdb = type(
        "SDB-lite",
        (),
        {
            "uam": uam,
            "log_path": log_path,
            "s3fs": s3fs,
            "filemap": {},
            "cc_translations": {},
            "tf": lambda t: t,
        },
    )()

else:
    sdb = None

with st.spinner("Loading libraries.."):
    import base64
    import contextlib
    import json
    import os
    import sys
    import warnings
    from altair.utils._importers import import_vl_convert
    from collections import defaultdict
    from copy import deepcopy
    from typing import TypeVar

    import altair as alt
    import pandas as pd
    import polars as pl
    import psutil
    import streamlit.components.v1 as components
    from streamlit_js import st_js, st_js_blocking  # type: ignore[import-untyped]

    from salk_toolkit.dashboard import (
        default_translate,
        draw_plot_matrix,
        facet_ui,
        filter_ui,
        stss_safety,
    )
    from salk_toolkit.io import (
        extract_column_meta,
        read_json,
        read_parquet_with_metadata,
    )
    from salk_toolkit.pp import (
        _update_data_meta_with_pp_desc,
        cont_transform_options,
        create_plot,
        get_plot_meta,
        impute_factor_cols,
        matching_plots,
        pp_transform_data,
    )
    from salk_toolkit.utils import apply_standard_chart_config, plot_matrix_html, replace_constants
    from salk_toolkit.validation import DataMeta, PlotDescriptor, soft_validate

    T = TypeVar("T")
    tqdm = lambda x: x  # So we can freely copy-paste from notebooks

    # Disable altair schema validations by setting debug_mode = False
    # This speeds plots up considerably as altair performs an excessive amount of these validation for some reason
    dm = alt.utils.schemapi.debug_mode(False)
    dm.__enter__()

    # Disable altair max rows
    alt.data_transformers.disable_max_rows()


def get_plot_width(width_str: str, ncols: int = 1) -> int:
    """Override the st_dimensions based version that can cause refresh loops.

    Calculate plot width based on number of columns.
    """
    return min(800, int(1200 / ncols))


def chart_to_url_with_config(chart: alt.Chart | alt.LayerChart | alt.FacetChart | object) -> str:
    """Convert an Altair chart to a Vega editor URL with standard configuration.

    This ensures the Vega editor shows the same styling and configuration
    as the Streamlit display and HTML exports.

    Args:
        chart: Altair chart object (Chart, LayerChart, FacetChart, etc.).

    Returns:
        URL string for opening the chart in the Vega editor.
    """
    vlc = import_vl_convert()

    # Convert chart to dict and apply standard configuration
    chart_dict = json.loads(chart.to_json())  # type: ignore[attr-defined]
    chart_dict = apply_standard_chart_config(chart_dict)

    # Use vl-convert to build the URL with our configured spec
    # This avoids validation issues and matches Altair's encoding
    return vlc.vegalite_to_url(chart_dict, fullscreen=False)


if "ls_loaded" not in st.session_state:
    try:
        ls_state = json.loads(st_js_blocking('return localStorage.getItem("session_state")') or "{}")
    except (json.JSONDecodeError, IndexError, RuntimeError):
        ls_state = {}

    for k, v in ls_state.items():
        st.session_state[k] = v
    st.session_state["ls_loaded"] = True

# Turn off annoying warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=pd.errors.PerformanceWarning)

translate = default_translate
path = "./"
paths = defaultdict(lambda: path)

if has_secrets and st.secrets.get("sip", {}).get("input_files"):
    global_data_meta = None
    input_file_choices = st.secrets["sip"]["input_files"]
    default_inputs = input_file_choices.copy()
else:
    cl_args = sys.argv[1:] if len(sys.argv) > 1 else []
    if cl_args and cl_args[0].endswith(".json") and os.path.isfile(cl_args[0]):
        global_data_meta = soft_validate(replace_constants(read_json(cl_args[0])), DataMeta)
        cl_args = cl_args[1:]
    else:
        global_data_meta = None

    # Add command line inputs as default input files
    default_inputs = []
    for raw_arg in cl_args:
        if raw_arg.startswith("-"):
            continue
        if not os.path.isfile(raw_arg):
            continue
        path, fname = os.path.split(raw_arg)
        if fname == "." or fname == "..":
            path, fname = fname, ""
        if not fname:
            continue
        if fname in paths:  # Duplicate file name: include path
            p1, p2 = os.path.split(path)
            path, fname = p1, os.path.join(p2, fname)
        paths[fname] = (path or ".") + "/"
        default_inputs.append(fname)

    if not path:
        path = "./"
    else:
        path += "/"

    input_file_choices = default_inputs + sorted([f for f in os.listdir(path) if f[-8:] == ".parquet"])

if len(input_file_choices) > 1:
    input_files = st.sidebar.multiselect("Select files:", input_file_choices, default_inputs)
    st.sidebar.markdown("""___""")
else:
    input_files = input_file_choices  # Just show that one file

if global_data_meta:
    st.sidebar.info("⚠️ External meta loaded.")

########################################################################
#                                                                      #
#                       LOAD PYMC SAMPLE DATA                          #
#                                                                      #
########################################################################


@st.cache_resource(show_spinner=False)
def load_file(input_file: str) -> dict[str, object]:
    """Load a parquet file with metadata."""
    pl.enable_string_cache()
    ifile = paths[input_file] + input_file
    ldf, full_meta = read_parquet_with_metadata(ifile, lazy=True)
    if full_meta is None:
        raise ValueError(f"Parquet file {ifile} has no metadata")
    data_meta = full_meta.data
    mmeta = full_meta.model
    columns = ldf.collect_schema().names()
    n0 = ldf.select(pl.len()).collect().item()
    n = data_meta.total_size or n0  # fallback to row count
    return {
        "data": ldf,
        "total_size": n,
        "data_meta": data_meta,
        "model_meta": mmeta,
        "columns": columns,
    }


if len(input_files) == 0:
    st.markdown("""Please choose an input file from the sidebar""")
    st.stop()
else:
    loaded = {ifile: load_file(ifile) for i, ifile in enumerate(input_files)}
    first_file = loaded[input_files[0]]
    first_data_meta = first_file["data_meta"] if global_data_meta is None else global_data_meta
    first_data = first_file["data"]

########################################################################
#                                                                      #
#                            Sidebar UI                                #
#                                                                      #
########################################################################


def get_dimensions(data_meta: DataMeta, present_cols: list[str], observations: bool = True) -> list[str]:
    """Extract dimension names from data metadata."""
    c_meta = extract_column_meta(data_meta)
    res = []
    if data_meta.structure is None:
        return res

    for block in data_meta.structure.values():
        if block.hidden:
            continue
        group_meta = c_meta[block.name]
        has_columns = group_meta.columns or []
        if observations and block.scale is not None and has_columns and (set(has_columns) & set(present_cols)):
            res.append(block.name)
        else:
            cols = list(has_columns)
            cols = [c for c in cols if c in present_cols]
            res += cols
    return res


args = {}

# If we have an override, add the column block to the data meta
raw_first_data_meta = deepcopy(first_data_meta)
c_meta = extract_column_meta(first_data_meta)
if st.session_state.get("override"):
    try:
        pp = eval(st.session_state["override"])
        # Only validate and update if override has meaningful content
        # _update_data_meta_with_pp_desc only uses res_meta and col_meta, but PlotDescriptor
        # requires 'plot' and 'res_col', so we need to provide defaults if missing
        if isinstance(pp, dict) and pp and ("res_meta" in pp or "col_meta" in pp):
            # For partial overrides used for data meta updates, provide minimal defaults for required fields
            if "plot" not in pp:
                pp["plot"] = "default"
            if "res_col" not in pp:
                pp["res_col"] = ""  # Temporary placeholder, will be set later in the UI
            pp = soft_validate(pp, PlotDescriptor)
            c_meta, _ = _update_data_meta_with_pp_desc(first_data_meta, pp)
    except Exception as e:
        st.error(f"Error parsing override: {e}")

with st.sidebar:  # .expander("Select dimensions"):
    f_info = st.empty()

    # Reset button - has to be high up in case something fails to load
    if st.sidebar.button("Reset choices"):
        st_js_blocking('localStorage.removeItem("session_state")')
        st.session_state.clear()

    draw = st.toggle("Draw plots", True, key="draw")

    show_grouped = st.toggle("Show grouped facets", True, key="show_grouped")

    if st.toggle("Convert to continuous", False, key="convert_res"):
        args["convert_res"] = "continuous"

    schema = getattr(first_data, "collect_schema", lambda: None)()
    if schema is None:
        st.error("Cannot collect schema from data")
        st.stop()
    all_cols = list(schema.names())

    obs_dims = get_dimensions(first_data_meta, all_cols, show_grouped)
    obs_dims = [c for c in obs_dims if c not in all_cols or not schema[c].is_temporal()]
    all_dims = get_dimensions(first_data_meta, all_cols, False)
    q_groups = list(set(obs_dims) - set(all_dims))

    # Deduplicate them - this bypasses some issues sometimes
    obs_dims = list(dict.fromkeys(obs_dims))
    all_dims = list(dict.fromkeys(all_dims))

    stss_safety("observation", obs_dims)
    obs_name = st.selectbox("Observation", obs_dims, key="observation")
    args["res_col"] = obs_name

    res_meta = c_meta[args["res_col"]]
    res_cont = (not res_meta.categories) or args.get("convert_res") == "continuous"

    modifiers = c_meta[obs_name].modifiers
    all_dims = list(modifiers) + all_dims

    facet_dims = all_dims
    if len(input_files) > 1:
        facet_dims = ["input_file"] + facet_dims

    args["factor_cols"] = facet_ui(facet_dims, two=True)

    # Check if any facet dims match observation dim or each other
    if len(set(args["factor_cols"] + [obs_name])) != len(args["factor_cols"]) + 1:
        st.markdown("""Please choose facets different from observation dimension""")
        st.stop()

    args["internal_facet"] = st.toggle("Internal facet?", True, key="internal")
    sort = st.toggle("Sort facets", False, key="sort")

    # Make all dimensions explicit
    args["factor_cols"] = impute_factor_cols(args, c_meta)

    # Plot type
    matching = matching_plots(args, first_data, first_data_meta)
    plot_list = ["default"] + sorted(matching)
    if "plot_type" in st.session_state:
        if st.session_state["plot_type"] not in matching:
            st.session_state["plot_type"] = "default"
        pt_ind = plot_list.index(st.session_state["plot_type"])
    else:
        pt_ind = 0
    args["plot"] = st.session_state["plot_type"] = st.selectbox(
        "Plot type",
        plot_list,
        index=pt_ind,
        format_func=lambda s: f"{matching[0]} (default)" if s == "default" else s,
    )
    if args["plot"] == "default":
        args["plot"] = matching[0]

    plot_meta = get_plot_meta(args["plot"])
    assert plot_meta is not None, f"Plot '{args['plot']}' not found in registry"
    plot_meta_dict = plot_meta.model_dump()

    # Plot arguments
    plot_args = {}  # 'n_facet_cols':2 }
    for k, t in plot_meta_dict.get("args", {}).items():
        vt, dv = t if isinstance(t, tuple) else (t, None)
        if vt == "bool":
            plot_args[k] = st.toggle(k, key=k, value=dv)
        elif vt == "int":
            plot_args[k] = st.number_input(k, key=k, value=(dv or 0))
        elif isinstance(vt, list):
            stss_safety(k, vt)
            ind = vt.index(dv) if dv in vt else 0
            plot_args[k] = st.selectbox(k, vt, key=k, index=ind)

    args["plot_args"] = {**args.get("plot_args", {}), **plot_args}

    with st.expander("Advanced"):
        detailed = st.toggle("Fine-grained filter", False, key="fine_grained")

        if res_cont:  # Extra settings for continuous data
            cont_transform = st.selectbox("Transform", ["None"] + cont_transform_options, key="transform")
            if cont_transform != "None":
                args["cont_transform"] = cont_transform
            agg_fn = st.selectbox("Aggregation", ["mean", "median", "sum"], key="aggregation")
            if agg_fn != "mean":
                args["agg_fn"] = agg_fn

        sortable = args["factor_cols"]
        if plot_meta_dict.get("sort_numeric_first_facet"):
            sortable = sortable[1:]
        if sort and len(sortable) > 0:
            stss_safety("sortby", sortable)
            sort_facet = st.selectbox("Sort by", sortable, 0, key="sortby")
            ascending = st.toggle("Ascending", False, key="sort_ascending")
            args["sort"] = {sort_facet: ascending}

        qpos = st.selectbox("Question position", ["Auto", 1, 2, 3], key="q_pos")
        if "question" in args["factor_cols"] and qpos != "Auto":
            args["factor_cols"] = [c for c in args["factor_cols"] if c != "question"]
            args["factor_cols"].insert(int(qpos) - 1, "question")

        override = st.text_area("Override keys", "{}", key="override")
        if override:
            args.update(eval(override))

    args["filter"] = filter_ui(
        first_data,
        first_data_meta,
        grouped=True,
        dims=q_groups + all_dims,
        obs_dim=obs_name,
        detailed=detailed,
        flt=args.get("filter", {}),
    )

    # Export options
    with st.expander("Export"):
        width = None
        height = None
        # Toggle export options because generating them is slow
        export = st.toggle("Show export options", value=False)
        export_ct = None
        if export:
            custom_size = st.toggle("Custom size", value=False)
            if custom_size:
                width = st.slider("Width", value=800, min_value=100, max_value=1200, step=50)
                height = st.slider("Height", value=600, min_value=100, max_value=2500, step=50)
            st.subheader("Export")
            export_ct = st.container()

    # print(list(st.session_state.keys()))

    # print(f"localStorage.setItem('args','{json.dumps(args)}');")
    st_js(f"localStorage.setItem('session_state','{json.dumps(dict(st.session_state)).replace("'", "\\'")}');")

    # Make all dimensions explicit now that plot is selected (as that can affect the factor columns)
    args["factor_cols"] = impute_factor_cols(args, c_meta, plot_meta)

    import pprint

    with st.expander("Plot desc"):
        st.code(pprint.pformat(args, indent=0, width=30))


if sdb and getattr(getattr(sdb, "uam", None), "admin", False):
    with st.sidebar:
        apanel = st.checkbox("Admin panel", value=False, key="admin_panel")
    if apanel:
        from salk_toolkit.dashboard import admin_page

        admin_page(sdb)
        st.stop()

# left, middle, right = st.columns([2, 5, 2])
# tab = middle.radio('Tabs',['Main'],horizontal=True,label_visibility='hidden')
# st.markdown("""___""")

########################################################################
#                                                                      #
#                              GRAPHS                                  #
#                                                                      #
########################################################################

# Workaround for geoplot - in that case draw multiple plots instead of a facet
matrix_form = False  # (args['plot'] == 'geoplot')

# Determine if one of the facets is input_file
input_files_facet = "input_file" in args.get("factor_cols", [])

# Create columns, one per input file
if not input_files_facet:
    cols = st.columns(len(input_files))
else:
    cols = [contextlib.suppress()]

if not draw:
    st.text("Plot drawing disabled for refresh speed")
elif input_files_facet:
    # with st.spinner('Filtering data...'):

    # This is a bit hacky because of previous use of the lazy data frames
    dfs = []
    for ifile in input_files:
        df, fargs = loaded[ifile]["data"], args.copy()
        ifile_cols = list(loaded[ifile]["columns"])  # type: ignore[arg-type]
        fargs["filter"] = {k: v for k, v in fargs["filter"].items() if k in ifile_cols or k in q_groups}
        fargs["factor_cols"] = [f for f in fargs["factor_cols"] if f != "input_file"]
        ppd = soft_validate(fargs, PlotDescriptor)
        pi = pp_transform_data(df, raw_first_data_meta, ppd)
        dfs.append(pi.data)

    fdf = pd.concat(dfs)

    # Fix categories to match the first file in case there are discrepancies (like only one being ordered)
    for c in fdf.columns:
        if fdf[c].dtype.name != "category" and dfs[0][c].dtype.name == "category":
            fdf[c] = pd.Categorical(fdf[c], dtype=dfs[0][c].dtype)

    fdf["input_file"] = pd.Categorical([v for i, f in enumerate(input_files) for v in [f] * len(dfs[i])], input_files)

    pi.data = fdf
    plot = create_plot(
        pi,
        soft_validate(args, PlotDescriptor),
        translate=translate,
        width=(width or get_plot_width("full", 1)),
        height=height,
        return_matrix_of_plots=matrix_form,
    )

    draw_plot_matrix(plot)
    # st.altair_chart(plot)#,use_container_width=True)

else:
    # Iterate over input files
    for i, ifile in enumerate(input_files):
        with cols[i]:
            # Heading:
            st.header(os.path.splitext(ifile.replace("_", " "))[0])

            data_meta = loaded[ifile]["data_meta"] if global_data_meta is None else global_data_meta
            if data_meta is None:
                data_meta = raw_first_data_meta

            ifile_cols = list(loaded[ifile]["columns"])  # type: ignore[arg-type]
            if (
                args["res_col"] in all_cols  # I.e. it is a column, not a group
                and args["res_col"] not in ifile_cols
            ):
                st.write(f"'{args['res_col']}' not present")
                continue

            # with st.spinner('Filtering data...'):
            fargs = args.copy()
            fargs["filter"] = {k: v for k, v in args["filter"].items() if k in ifile_cols or k in q_groups}
            ppd = soft_validate(fargs, PlotDescriptor)
            pi = pp_transform_data(loaded[ifile]["data"], data_meta, ppd)
            cur_width = width or get_plot_width(f"{i}_{ifile}", len(input_files))
            plot = create_plot(
                pi,
                ppd,
                translate=translate,
                width=cur_width,
                height=height,
                return_matrix_of_plots=matrix_form,
            )

            # n_questions = pi['data']['question'].nunique() if 'question' in pi['data'] else 1
            # st.write('Based on %.1f%% of data' %
            #   (100*pi['n_datapoints']/(len(loaded[ifile]['data_n'])*n_questions)))
            total_size = loaded[ifile]["total_size"]
            denominator = float(total_size) if total_size is not None else 1.0
            st.write("Based on %.1f%% of data" % (100 * pi.filtered_size / denominator))

            if i == 0 and st.session_state.get("custom_spec"):
                custom_spec = deepcopy(st.session_state["custom_spec"])
                autosize = custom_spec.get("autosize", {})
                autosize_type = autosize.get("type") if isinstance(autosize, dict) else None

                # Render specs with autosize="none" as HTML to preserve exact width
                if autosize_type == "none":
                    if autosize and "config" in custom_spec and isinstance(custom_spec.get("config"), dict):
                        custom_spec["config"]["autosize"] = dict(autosize)
                    spec_html = plot_matrix_html(
                        custom_spec, uid="custom_preview", width=None, responsive=False, apply_config=False
                    )
                    if spec_html:
                        components.html(spec_html, height=600, scrolling=True)
                else:
                    # Inline datasets for st.vega_lite_chart (doesn't support named datasets)
                    if "datasets" in custom_spec and "data" in custom_spec:
                        data_ref = custom_spec["data"]
                        if isinstance(data_ref, dict) and "name" in data_ref:
                            dataset_name = data_ref["name"]
                            if dataset_name in custom_spec["datasets"]:
                                custom_spec["data"] = {"values": custom_spec["datasets"][dataset_name]}
                                del custom_spec["datasets"]
                    st.vega_lite_chart(custom_spec, width="stretch")
                if st.button("Clear Override"):
                    del st.session_state["custom_spec"]
                    st.rerun()
            else:
                draw_plot_matrix(plot)

            if i == 0 and export and export_ct:
                export_ct.empty()
                with export_ct:
                    custom_spec = st.session_state.get("custom_spec")
                    chart_obj = custom_spec if custom_spec else plot

                    if custom_spec:
                        spec_str = json.dumps(custom_spec)
                        vlc = import_vl_convert()
                        edit_url = vlc.vegalite_to_url(custom_spec, fullscreen=False)
                        st.link_button("Vega Editor", edit_url, width="stretch")
                    else:
                        st.link_button("Vega Editor", chart_to_url_with_config(plot), width="stretch")

                    @st.dialog("Import Vega-Lite Spec")
                    def _import_modal() -> None:
                        spec = st.text_area("Paste JSON Spec", height=300)
                        if st.button("Apply"):
                            try:
                                st.session_state["custom_spec"] = json.loads(spec)
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON")

                    if st.button("Import Spec", width="stretch"):
                        _import_modal()

                    name = f"{args['res_col']}_{'_'.join(args['factor_cols']) if args['factor_cols'] else 'all'}"
                    chart_source = (
                        deepcopy(st.session_state["custom_spec"]) if st.session_state.get("custom_spec") else plot
                    )
                    apply_cfg = not bool(st.session_state.get("custom_spec"))

                    responsive = not custom_size
                    export_width = cur_width
                    if isinstance(chart_source, dict):
                        autosize = chart_source.get("autosize", {})
                        if isinstance(autosize, dict) and autosize.get("type") == "none":
                            responsive = False
                            export_width = None

                    st.download_button(
                        "HTML",
                        plot_matrix_html(
                            chart_source, uid=name, width=export_width, responsive=responsive, apply_config=apply_cfg
                        ),
                        f"{name}.html",
                        width="stretch",
                    )
                    st.download_button(
                        "Data CSV",
                        pi.data.to_csv().encode("utf-8"),
                        f"{name}.csv",
                        width="stretch",
                    )

                    @st.dialog("iframe Code")
                    def show_iframe_modal() -> None:
                        """Display iframe embed code in a modal dialog."""
                        content = plot_matrix_html(
                            chart_source, uid=name, width=export_width, responsive=responsive, apply_config=apply_cfg
                        )
                        if content is None:
                            st.error("Failed to generate HTML content")
                            return
                        encoded_html = base64.b64encode(content.encode("utf-8")).decode("utf-8")
                        iframe_code = (
                            f'<iframe src="data:text/html;base64,{encoded_html}" width="700" '
                            'height="525" frameborder="0" allowfullscreen style="aspect-ratio: 4/3;">'
                            "</iframe>"
                        )
                        st.code(iframe_code, language="html")

                    if st.button("iframe", width="stretch"):
                        show_iframe_modal()

            print(type(loaded[ifile]["data_meta"]))

            with st.expander("Data Meta"):
                st.write("This data is cleaned of extra fields that were not parsed")
                st.json(loaded[ifile]["data_meta"].model_dump(mode="json"))

            model_meta = loaded[ifile]["model_meta"]
            mdl_raw = getattr(model_meta, "copy", lambda: model_meta)() if hasattr(model_meta, "copy") else model_meta
            mdl: dict[str, object] = dict(mdl_raw) if isinstance(mdl_raw, dict) else {}

            if "sequence" in mdl:
                seq = mdl["sequence"]
                steps = {m["name"]: m for m in seq} if isinstance(seq, list) else {}  # type: ignore[index]
                del mdl["sequence"]
            else:
                steps = {}
            steps["main_model"] = mdl
            with st.expander("Model"):
                step_name = st.selectbox("Show:", list(steps.keys()), len(steps) - 1, key="mdlshow_" + ifile)
                st.json(steps[step_name])

info.empty()

st.sidebar.write("Mem: %.1f" % (psutil.Process(os.getpid()).memory_info().rss / 1024**2))

dm.__exit__(None, None, None)

if memprofile:
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("filename")
    memres = str(tracemalloc.get_traced_memory()) + "\n"
    memres += "[ Top 10 ]\n"
    for stat in top_stats[:10]:
        memres += str(stat) + "\n"
    # print(memres)
    st.code(memres)
    tracemalloc.stop()

if profile:
    p.stop()
