"""Plot Implementations
----------------------

This file collects all registry-backed plot builders that used to live in
`03_plots.ipynb`.  Each function focuses on:

- transforming the pre-wrangled data in ways specific to that visual
- configuring colours, tooltips, legends, and layout defaults expected by the
  dashboard and explorer apps
- registering itself via `@stk_plot(...)` so `pp.py` can auto-discover it

Section comments mirror the old notebook headings so it is easy to navigate by
plot family (columns, likert bars, density/violin, matrices, geo, etc.).
"""

__all__ = [
    "estimate_legend_columns_horiz_naive",
    "estimate_legend_columns_horiz",
    "boxplot_vals",
    "boxplot_manual",
    "maxdiff_manual",
    "columns",
    "stacked_columns",
    "diff_columns",
    "massplot",
    "make_start_end",
    "likert_bars",
    "kde_bw",
    "kde_1d",
    "density",
    "violin",
    "cluster_based_reorder",
    "matrix",
    "corr_matrix",
    "lines",
    "draws_to_hdis",
    "lines_hdi",
    "area_smooth",
    "likert_aggregate",
    "likert_rad_pol",
    "barbell",
    "geoplot",
    "geobest",
    "fd_mangle",
    "facet_dist",
    "ordered_population",
    "marimekko",
]

import itertools as it
import math
from typing import Any, Callable, Dict, Mapping, Sequence

import altair as alt
import arviz as az
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps
from KDEpy import FFTKDE  # type: ignore[import-untyped]
from KDEpy.bw_selection import silvermans_rule  # type: ignore[import-untyped]
from matplotlib import font_manager
from PIL import ImageFont
from scipy.cluster import hierarchy

from salk_toolkit import utils as utils
from salk_toolkit.pp import AltairChart, stk_plot


# --------------------------------------------------------
#          LEGEND UTILITIES
# --------------------------------------------------------


# Helper function to clean unnecessary index columns created by groupbyfrom a dataframe
def clean_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop auto-added `level_*` columns that pandas often emits after groupby."""

    df.drop(columns=[c for c in df.columns if c.startswith("level_")], inplace=True)
    return df


# Find a sensible approximation to the font used in vega/altair
font = font_manager.FontProperties(family="sans-serif", weight="regular")
font_file = font_manager.findfont(font)
legend_font = ImageFont.truetype(font_file, 10)


# Legends are not wrapped, nor is there a good way of doing accurately it in vega/altair
# This attempts to estimate a reasonable value for columns which induces wrapping
def estimate_legend_columns_horiz_naive(cats: Sequence[str], width: int) -> int:
    """Heuristic that balances legend columns purely by string length."""

    max_str_len = max(map(len, cats))
    n_cols = max(1, width // (15 + 5 * max_str_len))
    # distribute them roughly equally to avoid last row being awkwardly shorter
    n_rows = int(math.ceil(len(cats) / n_cols))
    return int(math.ceil(len(cats) / n_rows))


# More sophisticated version that looks at lengths of individual strings across multiple rows
# ToDo: it should max over each column separately not just look at max(sum(row)). This is close enough though.


def estimate_legend_columns_horiz(
    cats: Sequence[str],
    width: int,
    extra_text: Sequence[str] | None = None,
) -> int:
    """Estimate legend columns while accounting for varying label lengths."""

    max_cols, restart = len(cats), True
    if extra_text:
        width -= int(max(map(legend_font.getlength, extra_text)))
    lens = list(map(lambda s: 25 + legend_font.getlength(s), cats))
    while restart:
        restart, rl, cc = False, 0, 0
        for length in lens:
            if cc >= max_cols:  # Start a new row
                rl, cc = length, 1
            elif rl + length > width:  # Exceed width - restart
                max_cols = cc
                # Start from beginning every thime columns number changes
                # This is because what ends up in second+ rows depends on length of first
                restart = True
            else:  # Just append to existing row
                rl += length
                cc += 1

    # For very long labels just accept we can't do anything
    max_cols = max(max_cols, 1)

    # distribute them roughly equally to avoid last row being awkwardly shorter
    n_rows = int(math.ceil(len(cats) / max_cols))
    return int(math.ceil(len(cats) / n_rows))


# Regular boxplot with quantiles and Tukey whiskers
def boxplot_vals(s: pd.Series, extent: float = 1.5, delta: float = 1e-4) -> pd.DataFrame:
    """Return Tukey-style summary statistics for a series."""

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    if q3 - q1 > 2 * delta:
        delta = 0.0  # only inflate when we need to
    return pd.DataFrame(
        {
            "min": s.min(),
            "q1": q1,
            "q1p": q1 - delta,
            "mean": s.mean(),
            "q2 (median)": s.median(),
            "q3": q3,
            "q3p": q3 + delta,
            "max": s.max(),
            # Tukey values
            "tmin": s[s > q1 - extent * (q3 - q1)].min(),
            "tmax": s[s < q3 + extent * (q3 - q1)].max(),
        },
        index=["row"],
    )


# --------------------------------------------------------
#          PLOTS
# --------------------------------------------------------
# Data options:
#  - data_format: either 'longform' (default) or 'raw' for raw data
#  - n_facets: tuple describing how many internal facets the plot needs: (minimum, recommended)
#  - draws: if the plot requires draws to be present (such as boxplots)
#  - no_question_facet: plots where question does not make sense as an internal facet (e.g. density plots)
#  - requires: list of dicts describing what each facet expects
#      * 'likert': True requires a symmetric category with neutral in the middle
#      * 'ordered': True requires ordered categories
#      * <any kw>: 'pass' forwards that value from column metadata (used by e.g. geoplot)
#  - args: dict of extra plot kwargs exposed via `plot_args`
#  - priority: influences how likely this plot becomes the default
#  - group_size: requests pp to add a column describing the size of each group
#  - agg_fn: locks the aggregation function for continuous inputs (usually `sum` for election modelling)
#  - nonnegative: expects the value column to be non-negative for the plot to work properly
#
@stk_plot(
    "boxplots",
    data_format="longform",
    draws=True,
    n_facets=(1, 2),
    priority=50,
    group_sizes=True,
)
def boxplot_manual(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
) -> AltairChart:
    """Manual boxplot implementation using Tukey whiskers."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None

    if (
        val_format[-1] == "%"
    ):  # Boxplots being a compound plot, this workaround is needed for axis & tooltips to be proper
        data[value_col] *= 100
        val_format = val_format[:-1] + "f"

    minv, maxv = data[value_col].min(), data[value_col].max()
    if val_format[-1] == "%":
        minv = 0.0
    if minv == maxv:
        minv, maxv = minv - 0.01, maxv + 0.01

    f_cols = outer_factors + [f["col"] for f in facets[:2] if f is not None]
    df = data.groupby(f_cols, observed=True)[value_col].apply(boxplot_vals, delta=(maxv - minv) / 400).reset_index()
    clean_levels(df)  # For test consistency

    shared = {
        "y": alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
        **({"yOffset": alt.YOffset(field=f1["col"], type="nominal", title=None, sort=f1["order"])} if f1 else {}),
        "tooltip": [
            alt.Tooltip(
                field=vn,
                type="quantitative",
                format=val_format,
                title=f"{vn[0].upper() + vn[1:]} of {value_col}",
            )
            for vn in ["min", "q1", "mean", "q2 (median)", "q3", "max"]
        ]
        + tooltip[1:],
    }

    root = alt.Chart(df).encode(**shared)
    size = 12

    # Compose each layer individually
    lower_plot = root.mark_rule().encode(
        x=alt.X(
            "tmin:Q",
            axis=alt.Axis(title=value_col, format=val_format),
            scale=alt.Scale(domain=[minv, maxv]),
        ),
        x2=alt.X2("q1:Q"),
    )

    middle_plot = root.mark_bar(size=size).encode(
        x=alt.X("q1p:Q"),
        x2=alt.X2("q3p:Q"),
        **(
            {"color": alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None)}
            if not f1
            else {
                "color": alt.Color(
                    field=f1["col"],
                    type="nominal",
                    scale=f1["colors"],
                    legend=alt.Legend(
                        orient="top",
                        columns=estimate_legend_columns_horiz(f1["order"], width),
                    ),
                )
            }
        ),
    )

    upper_plot = root.mark_rule().encode(x=alt.X("q3:Q"), x2=alt.X2("tmax:Q"))

    middle_tick = root.mark_tick(color="white", size=size).encode(
        x="mean:Q",
    )

    return lower_plot + middle_plot + upper_plot + middle_tick


# Also create a raw version for the same plot
stk_plot("boxplots-raw", data_format="raw", n_facets=(1, 2), priority=0)(boxplot_manual)


@stk_plot(
    "maxdiff",
    data_format="longform",
    transform_fn="ordered-topbot1",
    agg_fn="posneg_mean",
    draws=True,
    n_facets=(1, 2),
    group_sizes=True,
)
def maxdiff_manual(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
) -> AltairChart:
    """Render MaxDiff results as categorical columns."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    r"""
    Only meant to be used with highest_lowest_ranked custom_row_transform.
    """
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None
    reverse_val_col = next(
        filter(
            lambda s: value_col.lower() in s.lower() and "reverse" in s.lower(),
            data.columns,
        )
    )

    if (
        val_format[-1] == "%"
    ):  # Boxplots being a compound plot, this workaround is needed for axis & tooltips to be proper
        data[value_col] *= 100
        if reverse_val_col in data.columns:
            data[reverse_val_col] *= 100
        val_format = val_format[:-1] + "f"

    minv, maxv = data[value_col].min(), data[value_col].max()
    if val_format[-1] == "%":
        minv = 0.0
    if minv == maxv:
        minv, maxv = minv - 0.01, maxv + 0.01

    f_cols = outer_factors + [f["col"] for f in facets[:2] if f is not None]
    # data.to_csv("tmp_data.csv",index=False)
    df = data.groupby(f_cols, observed=True)[value_col].apply(boxplot_vals, delta=(maxv - minv) / 400).reset_index()
    df_reverse = (
        data.groupby(f_cols, observed=True)[reverse_val_col]
        .apply(boxplot_vals, delta=(maxv - minv) / 400)
        .reset_index()
    )
    df_reverse["mean"] = -df_reverse["mean"]
    df_reverse["kind"] = "Least important"

    df["kind"] = "Most important"

    shared = {
        "y": alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
        **({"yOffset": alt.YOffset(field=f1["col"], type="nominal", title=None, sort=f1["order"])} if f1 else {}),
        "tooltip": [
            alt.Tooltip(
                field=vn,
                type="quantitative",
                format=val_format,
                title=f"{vn[0].upper() + vn[1:]} of {value_col}",
            )
            for vn in ["mean"]
        ]
        + tooltip[1:],
    }

    df = pd.concat(
        [df[f_cols + ["kind", "mean"]], df_reverse[f_cols + ["kind", "mean"]]],
        ignore_index=True,
        sort=False,
    )
    clean_levels(df)  # For test consistency
    root = alt.Chart(df).encode(**shared)
    size = 12
    red, blue = utils.redblue_gradient[1], utils.redblue_gradient[-2]

    return root.mark_bar(size=size).encode(
        x=alt.X("mean"),
        # color = 'kind'
        color=alt.Color(
            field="kind",
            type="nominal",
            legend=None,
            scale=alt.Scale(domain=["Most important", "Least important"], range=[blue, red]),
        )
        if f1 is None or f1["colors"] is None
        else alt.Color(field=f1["col"], type="nominal", scale=f1["colors"], legend=None),
        opacity=alt.Opacity(
            field="kind",
            type="nominal",
            legend=None,
            scale=alt.Scale(domain=["Most important", "Least important"], range=[1.0, 0.8]),
        )
        if f1 is not None
        else alt.Opacity(value=1.0),
    )


@stk_plot("columns", data_format="longform", draws=False, n_facets=(1, 2))
def columns(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Simple column chart for categorical comparisons."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None
    plot = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                field=value_col,
                type="quantitative",
                title=value_col,
                axis=alt.Axis(format=val_format),
            ),
            y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
            tooltip=tooltip,
            **(
                {"color": alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None)}
                if not f1
                else {
                    "yOffset": alt.YOffset(field=f1["col"], type="nominal", title=None, sort=f1["order"]),
                    "color": alt.Color(
                        field=f1["col"],
                        type="nominal",
                        scale=f1["colors"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1["order"], width),
                        ),
                    ),
                }
            ),
        )
    )
    return plot


@stk_plot(
    "stacked_columns",
    data_format="longform",
    draws=False,
    nonnegative=True,
    n_facets=(2, 2),
    agg_fn="sum",
    args={"normalized": "bool"},
)
def stacked_columns(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    filtered_size: int = 1,
    val_format: str = "%",
    width: int = 800,
    normalized: bool = False,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Stacked columns with optional normalization."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1]

    data[value_col] = data[value_col] / filtered_size

    ldict = dict(zip(f1["order"], range(len(f1["order"]))))
    data["f_order"] = data[f1["col"]].astype("object").replace(ldict).astype("int")

    plot = (
        alt.Chart(round(data, 3), width="container")
        .mark_bar()
        .encode(
            x=alt.X(
                field=value_col,
                type="quantitative",
                title=value_col,
                axis=alt.Axis(format=val_format),
                **({"stack": "normalize"} if normalized else {}),
                # scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
            y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
            tooltip=tooltip,
            **(
                {"color": alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None)}
                if len(facets) <= 1
                else {
                    "order": alt.Order("f_order:O"),
                    "color": alt.Color(
                        field=f1["col"],
                        type="nominal",
                        scale=f1["colors"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1["order"], width),
                        ),
                    ),
                }
            ),
        )
    )
    return plot


@stk_plot(
    "diff_columns",
    data_format="longform",
    draws=False,
    n_facets=(2, 2),
    args={"sort_descending": "bool"},
)
def diff_columns(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    sort_descending: bool = False,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Difference columns chart (two categories per row with delta)."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1]

    ind_cols = list(set(data.columns) - {value_col, f1["col"]})
    factors = [c for c in f1["order"] if c in data[f1["col"]].unique()]  # sort factors for deterministic order

    idf = data.set_index(ind_cols)
    diff = (idf[idf[f1["col"]] == factors[1]][value_col] - idf[idf[f1["col"]] == factors[0]][value_col]).reset_index()

    if sort_descending:
        f0["order"] = list(diff.sort_values(value_col, ascending=False)[f0["col"]])

    plot = (
        alt.Chart(round(diff, 3), width="container")
        .mark_bar()
        .encode(
            y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
            x=alt.X(
                field=value_col,
                type="quantitative",
                title=f"{factors[1]} - {factors[0]}",
                axis=alt.Axis(format=val_format, title=f"{factors[0]} <> {factors[1]}"),
                # scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
            tooltip=[
                alt.Tooltip(field=f0["col"], type="nominal"),
                alt.Tooltip(
                    field=value_col,
                    type="quantitative",
                    format=val_format,
                    title=f"{value_col} difference",
                ),
            ],
            color=alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None),
        )
    )
    return plot


# The idea was to also visualize the size of each cluster. Currently not very useful, may need to be rethought
@stk_plot(
    "massplot",
    data_format="longform",
    draws=False,
    group_sizes=True,
    n_facets=(1, 2),
    hidden=True,
)
def massplot(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    filtered_size: int = 1,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Mass plot showing distributions vs. categorical facets."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None

    data["group_size"] = data["group_size"] / filtered_size  # .round(2)

    plot = (
        alt.Chart(round(data, 3), width="container")
        .mark_circle()
        .encode(
            y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
            x=alt.X(
                field=value_col,
                type="quantitative",
                title=value_col,
                axis=alt.Axis(format=val_format),
                # scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
            size=alt.Size("group_size:Q", legend=None, scale=alt.Scale(range=[100, 500])),
            opacity=alt.value(1.0),
            stroke=alt.value("#777"),
            tooltip=tooltip + [alt.Tooltip("group_size:N", format=".1%", title="Group size")],
            **(
                {"color": alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None)}
                if not f1
                else {
                    "yOffset": alt.YOffset(field=f1["col"], type="nominal", title=None, sort=f1["order"]),
                    "color": alt.Color(
                        field=f1["col"],
                        type="nominal",
                        scale=f1["colors"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1["order"], width),
                        ),
                    ),
                }
            ),
        )
    )
    return plot


# Make the likert bar pieces
def make_start_end(
    x: pd.DataFrame,
    value_col: str,
    cat_col: str,
    cat_order: Sequence[str],
    neutral: Sequence[int],
    n_negative: int,
) -> pd.DataFrame:
    """Compute start/end positions for split likert bars."""

    if len(x) != len(cat_order):
        shared = x.to_dict(orient="records")[0]
        # Fill in missing rows with value zero so they would just be skipped
        mdf = pd.DataFrame({cat_col: pd.Categorical(cat_order, cat_order, ordered=True)})
        x = pd.merge(mdf, x, on=cat_col, how="left").fillna({**shared, value_col: 0})
    x = x.sort_values(by=cat_col)

    if len(neutral) == 0:  # No neutrals
        x_other = x.copy()
        x_mids = []
    else:  # Handle neutrals
        mids = neutral
        nonmid = [i for i in range(len(x)) if i not in mids]

        scale_start = -1.0
        x_mid = x.iloc[mids, :].copy()
        x_other = x.iloc[nonmid, :].copy()

        # Compute the positions of the neutrals
        x_mid.loc[:, "end"] = scale_start + x_mid[value_col].cumsum()
        x_mid.loc[:, "start"] = x_mid.loc[:, "end"] - x_mid[value_col]
        x_mids = [x_mid]

    o_mid = n_negative
    x_other.loc[:, "end"] = x_other[value_col].cumsum() - x_other[:o_mid][value_col].sum()
    x_other.loc[:, "start"] = (x_other[value_col][::-1].cumsum()[::-1] - x_other[o_mid:][value_col].sum()) * -1
    res = pd.concat([x_other] + x_mids).dropna(subset=[value_col])  # drop any na rows added in the filling in step
    return res


@stk_plot(
    "likert_bars",
    data_format="longform",
    draws=False,
    requires=[{"likert": True}],
    n_facets=(1, 3),
    sort_numeric_first_facet=True,
    priority=50,
)
def likert_bars(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
    width: int = 800,
) -> AltairChart:
    """Display likert responses as diverging stacked bars."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    # First facet is likert, second is labeled question, third is offset.
    # Second is better for question which usually goes last, hence reorder
    if len(facets) == 1:  # Create a dummy second facet
        facets.append({"col": "question", "order": [facets[0]["col"]], "colors": alt.Undefined})
        data["question"] = facets[0]["col"]
    if len(facets) >= 3:
        f0, f1, f2 = facets[0], facets[2], facets[1]
    elif len(facets) == 2:
        f0, f1, f2 = facets[0], facets[1], None

    # Split the categories into negative, neutral, positive same way that colors were allocated
    neg, neutral, pos = utils.split_to_neg_neutral_pos(f0["order"], f0.get("neutrals", []))
    ninds = [f0["order"].index(c) for c in neutral]

    gb_cols = outer_factors + [
        f["col"] for f in facets[1:]
    ]  # There can be other extra cols (like labels) that should be ignored
    list(data[f0["col"]].dtype.categories)  # Get likert scale names
    bar_data = data.groupby(gb_cols, group_keys=False, observed=False)[data.columns].apply(
        make_start_end,
        value_col=value_col,
        cat_col=f0["col"],
        cat_order=f0["order"],
        include_groups=False,
        neutral=ninds,
        n_negative=len(neg),
    )

    plot = (
        alt.Chart(bar_data)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", axis=alt.Axis(title=None, format="%")),
            x2=alt.X2("end:Q"),
            y=alt.Y(
                field=f1["col"],
                type="nominal",
                axis=alt.Axis(title=None, offset=5, ticks=False, minExtent=60, domain=False),
                sort=f1["order"],
            ),
            tooltip=tooltip,
            color=alt.Color(
                field=f0["col"],
                type="nominal",
                legend=alt.Legend(
                    title=None,  # f0["col"],
                    orient="bottom",
                    columns=estimate_legend_columns_horiz(f0["order"], width, extra_text=f1["order"]),
                ),
                scale=f0["colors"],
            ),
            **({"yOffset": alt.YOffset(field=f2["col"], type="nominal", title=None, sort=f2["order"])} if f2 else {}),
        )
    )
    return plot


# Calculate the bandwidth for KDE
def kde_bw(ar: np.ndarray) -> float:
    """Lower-bound Silverman's rule to keep categorical densities stable."""

    return max(silvermans_rule(ar) or 0.0, 0.75 * utils.min_diff(ar[:, 0]))


# Calculate KDE ourselves using a fast libary. This gets around having to do sampling which is unstable


def kde_1d(
    vc: pd.DataFrame,
    value_col: str,
    ls: Sequence[float],
    scale: bool = False,
    bw: float | None = None,
) -> pd.DataFrame:
    """Evaluate a 1D Gaussian KDE over ``ls`` for a single series."""

    ar = vc.to_numpy()
    if bw is None:
        bw = kde_bw(ar)  # This can be problematic in small segments, so best calculated globally
    y = FFTKDE(kernel="gaussian", bw=bw).fit(ar).evaluate(ls)
    if scale:
        y *= len(vc)
    return pd.DataFrame({"density": y, value_col: ls})


@stk_plot(
    "density",
    factor_columns=3,
    draws=True,
    aspect_ratio=(1.0 / 1.0),
    n_facets=(0, 1),
    args={"stacked": "bool", "bw": "float"},
    no_question_facet=True,
)
def density(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
    stacked: bool = False,
    bw: float | None = None,
    width: int = 800,
) -> AltairChart:
    """Stacked (or overlapped) density plot for continuous responses."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0 = facets[0] if len(facets) > 0 else None
    gb_cols = [
        c for c in outer_factors + [f["col"] for f in facets] if c is not None
    ]  # There can be other extra cols (like labels) that should be ignored

    # Filter out extreme outliers (one thousandth on each side).
    # Because at 100k+, these get very extreme even for normal distributions
    lims = list(data[value_col].quantile([0.005, 0.995]))
    data = data[(data[value_col] >= lims[0]) & (data[value_col] <= lims[1])]

    ls = np.linspace(data[value_col].min() - 1e-10, data[value_col].max() + 1e-10, 101)
    if bw is None:
        bw = kde_bw(data[[value_col]].sample(10000, replace=True).to_numpy())  # Can get slow for large data otherwise
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[value_col],
        fn=kde_1d,
        value_col=value_col,
        ls=ls,
        scale=stacked,
        bw=bw,
    ).reset_index()
    clean_levels(ndata)

    if f0:
        selection = alt.selection_point(fields=[f0["col"]], bind="legend")

    if stacked:
        if f0:
            ldict = dict(zip(f0["order"], reversed(range(len(f0["order"])))))
            ndata.loc[:, "order"] = ndata[f0["col"]].astype("object").replace(ldict).astype("int")

        ndata["density"] /= len(data)
        plot = (
            alt.Chart(ndata)
            .mark_area(interpolate="natural")
            .encode(
                x=alt.X(field=value_col, type="quantitative"),
                y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%"), stack="zero"),
                tooltip=tooltip[1:],
                **(
                    {
                        "fill": alt.Fill(
                            field=f0["col"],
                            type="nominal",
                            scale=f0["colors"],
                            legend=alt.Legend(
                                orient="top",
                                columns=estimate_legend_columns_horiz(f0["order"], width),
                            ),
                        ),
                        "order": alt.Order("order:O"),
                        "opacity": alt.condition(selection, alt.value(1), alt.value(0.15)),
                    }
                    if f0
                    else {}
                ),
            )
        )
    else:
        plot = (
            alt.Chart(ndata)
            .mark_line()
            .encode(
                x=alt.X(field=value_col, type="quantitative"),
                y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%")),
                tooltip=tooltip[1:],
                **(
                    {
                        "color": alt.Color(
                            field=f0["col"],
                            type="nominal",
                            scale=f0["colors"],
                            legend=alt.Legend(
                                orient="top",
                                columns=estimate_legend_columns_horiz(f0["order"], width),
                            ),
                        ),
                        "order": alt.Order("order:O"),
                        "opacity": alt.condition(selection, alt.value(1), alt.value(0.15)),
                    }
                    if f0
                    else {}
                ),
            )
        )

    if f0:
        plot = plot.add_params(selection)

    return plot


# Also create a raw version for the same plot
stk_plot(
    "density-raw",
    data_format="raw",
    factor_columns=3,
    aspect_ratio=(1.0 / 1.0),
    n_facets=(0, 1),
    args={"stacked": "bool", "bw": "float"},
    no_question_facet=True,
    priority=0,
)(density)


@stk_plot("violin", n_facets=(1, 2), draws=True, as_is=True, args={"bw": "float"})
def violin(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
    bw: float | None = None,
    width: int = 800,
) -> AltairChart:
    """Violin plot drawing densities per facet."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None
    gb_cols = outer_factors + [
        f["col"] for f in facets
    ]  # There can be other extra cols (like labels) that should be ignored

    ls = np.linspace(data[value_col].min() - 1e-10, data[value_col].max() + 1e-10, 101)
    if bw is None:
        bw = kde_bw(data[[value_col]].sample(10000, replace=True).to_numpy())
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[value_col],
        fn=kde_1d,
        value_col=value_col,
        ls=ls,
        scale=True,
        bw=bw,
    ).reset_index()
    clean_levels(ndata)

    if f1:
        ldict = dict(zip(f1["order"], reversed(range(len(f1["order"])))))
        ndata.loc[:, "order"] = ndata[f1["col"]].astype("object").replace(ldict).astype("int")

    ndata["density"] /= len(data)
    plot = (
        alt.Chart(ndata)
        .mark_area(interpolate="natural")
        .encode(
            x=alt.X(field=value_col, type="quantitative"),
            y=alt.Y(
                "density:Q",
                axis=alt.Axis(title=None, labels=False, values=[0], grid=False),
                stack="center",
            ),
            row=alt.Row(
                field=f0["col"],
                type="nominal",
                header=alt.Header(orient="top", title=None),
                spacing=5,
                sort=f0["order"],
            ),
            tooltip=tooltip[1:],
            # color=alt.Color(f'{question_col}:N'),
            **(
                {
                    "color": alt.Color(
                        field=f1["col"],
                        type="nominal",
                        scale=f1["colors"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1["order"], width),
                        ),
                    ),
                    "order": alt.Order("order:O"),
                }
                if f1
                else {"color": alt.Color(field=f0["col"], type="nominal", scale=f0["colors"], legend=None)}
            ),
        )
        .properties(width=width, height=70)
    )

    return plot


# Also create a raw version for the same plot
stk_plot("violin-raw", data_format="raw", n_facets=(1, 2), as_is=True, args={"bw": "float"})(violin)


# Cluster-based reordering
def cluster_based_reorder(X: np.ndarray) -> np.ndarray:
    """Return leaf order from hierarchical clustering for nicer matrices."""

    pdist = sp.spatial.distance.pdist(X)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(hierarchy.ward(pdist), pdist))


@stk_plot(
    "matrix",
    data_format="longform",
    aspect_ratio=(1 / 0.8),
    n_facets=(2, 2),
    args={"reorder": "bool", "log_colors": "bool"},
    priority=55,
)
def matrix(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    reorder: bool = False,
    log_colors: bool = False,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Heatmap-style matrix plot (optionally reorder rows/cols via clustering)."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1]

    fcols = [c for c in data.columns if c not in [value_col, f0["col"]]]
    if len(fcols) == 1 and reorder:  # Reordering only works if no external facets
        X = data.pivot(columns=f1["col"], index=f0["col"]).to_numpy()
        f0["order"] = np.array(f0["order"])[cluster_based_reorder(X)]
        f1["order"] = np.array(f1["order"])[cluster_based_reorder(X.T)]

    if log_colors:
        data["val_log"] = np.log(data[value_col])
        data["val_log"] -= data["val_log"].min()  # Keep it all positive
        scale_v = "val_log"
    else:
        scale_v = value_col

    # Find max absolute value to keep color scale symmetric
    mi, ma = data[scale_v].min(), data[scale_v].max()
    dmax = float(max(-mi, ma))

    if mi < 0:
        scale, smid, swidth = (
            {
                "scheme": "redyellowgreen",
                "domainMid": 0,
                "domainMin": -dmax,
                "domainMax": dmax,
            },
            0,
            2 * dmax,
        )
    else:
        scale, smid, swidth = (
            {"scheme": "yellowgreen", "domainMin": 0, "domainMax": dmax},
            0,
            2 * dmax,
        )  # dmax/2, dmax

    # Draw colored boxes
    plot = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(field=f1["col"], type="nominal", title=None, sort=f1["order"]),
            y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
            color=alt.Color(
                field=scale_v,
                type="quantitative",
                scale=alt.Scale(**scale),
                legend=(alt.Legend(title=None) if not log_colors else None),
            ),
            tooltip=tooltip,
        )
    )

    # Add in numerical values
    if len(f1["order"]) < 20:  # only if we have less than 20 columns
        threshold = round((0.25 * swidth) ** 2, 3)  # Rounding needed for tests to be stable
        text = plot.mark_text().encode(
            text=alt.Text(field=value_col, type="quantitative", format=val_format),
            color=alt.condition(
                (alt.datum[scale_v] - smid) ** 2 > threshold,
                alt.value("white"),
                alt.value("black"),
            ),
            tooltip=tooltip,
        )
        plot += text

    return plot


@stk_plot("corr_matrix", data_format="raw", aspect_ratio=(1 / 0.8), n_facets=(1, 1))
def corr_matrix(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    reorder: bool = False,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Correlation matrix for raw grouped data."""

    facets = facets or []
    tooltip = list(tooltip or [])
    if "id" not in data.columns:
        raise Exception("Corr_matrix only works for groups of continuous variables")

    # id is required to match the rows for correllations
    cm = (
        data.pivot_table(index="id", columns=facets[0]["col"], values=value_col, observed=False)
        .corr()
        .reset_index(names="index")
    )
    cm_long = cm.melt(
        id_vars=["index"],
        value_vars=cm.columns,
        var_name=facets[0]["col"],
        value_name=value_col,
    )

    order = facets[0]["order"]
    lower_tri = cm_long["index"].map(lambda x: order.index(x)).astype(int) > cm_long[facets[0]["col"]].map(
        lambda x: order.index(x)
    ).astype(int)
    cm_long = cm_long[lower_tri]

    return matrix(
        cm_long,
        value_col=value_col,
        facets=[
            {"col": "index", "order": facets[0]["order"]},
            {"col": facets[0]["col"], "order": facets[0]["order"]},
        ],
        val_format=val_format,
        tooltip=[
            alt.Tooltip(field=value_col, type="quantitative"),
            alt.Tooltip("index:N"),
            alt.Tooltip(field=facets[0]["col"], type="nominal"),
        ],
    )


# Convert a categorical fx facet into a continous value and axis if the categories are numeric.
def cat_to_cont_axis(
    data: pd.DataFrame,
    fx: Dict[str, Any],
) -> tuple[alt.X, pd.DataFrame]:
    """Convert categorical axis to numeric when labels are numeric strings."""

    x_cont = pd.to_numeric(
        data[fx["col"]].apply(utils.unescape_vega_label), errors="coerce"
    )  # Unescape required as . gets escaped
    if x_cont.notna().all():
        data[fx["col"]] = x_cont.astype("float")
        x_axis = alt.X(
            field=fx["col"],
            type="quantitative",
            title=None,
            axis=alt.Axis(labelAngle=0, values=list(data[fx["col"]].unique())),
        )
    else:
        x_axis = alt.X(
            field=fx["col"],
            type="nominal",
            title=None,
            sort=fx["order"],
            axis=alt.Axis(labelAngle=0),
        )
    return x_axis, data


@stk_plot(
    "lines",
    data_format="longform",
    draws=False,
    requires=[{}, {"ordered": True}],
    n_facets=(2, 2),
    args={"smooth": "bool"},
    priority=10,
)
@stk_plot(
    "line",
    data_format="longform",
    draws=False,
    requires=[{"ordered": True}],
    n_facets=(1, 1),
    args={"smooth": "bool"},
    priority=10,
)
def lines(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    smooth: bool = False,
    width: int = 800,
    tooltip: Sequence[str] | None = None,
    val_format: str = ".2f",
) -> AltairChart:
    """Line chart with optional smoothing and categorical faceting."""

    facets = facets or []
    tooltip = list(tooltip or [])
    if len(facets) == 1:
        fx = facets[0]
    else:
        fy, fx = facets[0], facets[1]
    if smooth:
        smoothing = "basis"
        points = "transparent"
    else:
        smoothing = "natural"
        points = True

    # See if we should use a continous axis (if categoricals are actually numbers)
    x_axis, data = cat_to_cont_axis(data, fx)

    plot = (
        alt.Chart(data)
        .mark_line(point=points, interpolate=smoothing)
        .encode(
            x=x_axis,
            y=alt.Y(
                field=value_col,
                type="quantitative",
                title=(value_col if len(value_col) < 20 else None),
                axis=alt.Axis(format=val_format),
            ),
            tooltip=tooltip,
            **(
                {
                    "color": alt.Color(
                        field=fy["col"],
                        type="nominal",
                        scale=fy["colors"],
                        sort=fy["order"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(fy["order"], width),
                        ),
                    )
                }
                if len(facets) == 2
                else {}
            ),
        )
    )
    return plot


def draws_to_hdis(
    data: pd.DataFrame,
    vc: str,
    hdi_vals: Sequence[float],
) -> pd.DataFrame:
    """Compute HDI intervals for draw data."""

    gbc = [c for c in data.columns if c not in [vc, "draw"]]
    ldfs = []
    for hdiv in hdi_vals:
        ldf_v = (
            data.groupby(gbc, observed=False)[vc]
            .apply(lambda s: pd.Series(list(az.hdi(s.to_numpy(), hdi_prob=hdiv)), index=["lo", "hi"]))
            .reset_index()
        )
        ldf_v["hdi"] = hdiv
        ldfs.append(ldf_v)
    ldf = pd.concat(ldfs).reset_index(drop=True)
    df = ldf.pivot(index=gbc + ["hdi"], columns=ldf.columns[-3], values=vc).reset_index()
    return df


@stk_plot(
    "lines_hdi",
    data_format="longform",
    draws=True,
    requires=[{}, {"ordered": True}],
    n_facets=(2, 2),
    args={"hdi1": "float", "hdi2": "float"},
)
@stk_plot(
    "line_hdi",
    data_format="longform",
    draws=True,
    requires=[{"ordered": True}],
    n_facets=(1, 1),
    args={"hdi1": "float", "hdi2": "float"},
)
def lines_hdi(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    width: int = 800,
    tooltip: Sequence[str] | None = None,
    val_format: str = ".2f",
    hdi1: float = 0.94,
    hdi2: float = 0.5,
) -> AltairChart:
    """Line chart showing central tendency plus HDI ribbons."""

    facets = facets or []
    tooltip = list(tooltip or [])
    if len(facets) == 1:
        fx = facets[0]
    else:
        fy, fx = facets[0], facets[1]

    hdf = draws_to_hdis(data, value_col, [hdi1, hdi2])

    if len(facets) > 1:
        # Draw them in reverse order so the things that are first (i.e. most important)
        # are drawn last (i.e. on top of others). Also draw wider hdi before the narrower
        hdf.sort_values([fy["col"], "hdi"], ascending=[False, False], inplace=True)
        selection = alt.selection_point(fields=[fy["col"]], bind="legend")

    # See if we should use a continous axis (if categoricals are actually numbers)
    x_axis, hdf = cat_to_cont_axis(hdf, fx)

    plot = (
        alt.Chart(hdf)
        .mark_area(interpolate="basis")
        .encode(
            x=x_axis,
            y=alt.Y(
                "lo:Q",
                axis=alt.Axis(
                    format=val_format,
                    title=(value_col if len(value_col) < 20 else None),
                ),
                title=value_col,
            ),
            y2=alt.Y2("hi:Q"),
            tooltip=[
                alt.Tooltip("hdi:N", title="HDI", format=".0%"),
                alt.Tooltip("lo:Q", title="HDI lower", format=val_format),
                alt.Tooltip("hi:Q", title="HDI upper", format=val_format),
            ]
            + tooltip[1:],
            **(
                {
                    "fill": alt.Fill(
                        field=fy["col"],
                        type="nominal",
                        sort=fy["order"],
                        scale=fy["colors"],
                        legend=alt.Legend(symbolOpacity=1),
                    ),
                    "opacity": alt.condition(
                        selection,
                        alt.Opacity(
                            "hdi:N",
                            legend=None,
                            scale=utils.to_alt_scale({0.5: 0.75, 0.94: 0.25}),
                        ),
                        alt.value(0.1),
                    ),
                }
                if len(facets) > 1
                else {}
            ),
        )
    )

    if len(facets) > 1:
        plot = plot.add_params(selection)
    return plot


@stk_plot(
    "area_smooth",
    data_format="longform",
    requires=[{}, {"ordered": True}],
    draws=False,
    nonnegative=True,
    n_facets=(2, 2),
)
def area_smooth(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    width: int = 800,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Area chart with smoothing for cumulative comparisons."""

    facets = facets or []
    tooltip = list(tooltip or [])
    fy, fx = facets[0], facets[1]
    ldict = dict(zip(fy["order"], range(len(fy["order"]))))
    data.loc[:, "order"] = data[fy["col"]].astype("object").replace(ldict).astype("int")

    x_axis, data = cat_to_cont_axis(data, fx)

    plot = (
        alt.Chart(data)
        .mark_area(interpolate="natural")
        .encode(
            x=x_axis,
            y=alt.Y(
                field=value_col,
                type="quantitative",
                title=None,
                stack="normalize",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            order=alt.Order("order:O"),
            color=alt.Color(
                field=fy["col"],
                type="nominal",
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(fy["order"], width),
                ),
                sort=fy["order"],
                scale=fy["colors"],
            ),
            tooltip=tooltip,
        )
    )
    return plot


def likert_aggregate(
    x: pd.Series,
    cat_col: str,
    cat_order: Sequence[str],
    value_col: str,
) -> pd.Series:
    """Aggregate a single likert question by category order."""

    cc, vc = x[cat_col], x[value_col]
    cats = cat_order

    mid, odd = len(cats) // 2, len(cats) % 2

    nonmid_sum = vc[cc != cats[mid]].sum() if odd else vc.sum()

    pol = np.minimum(vc[cc.isin(cats[:mid])].sum(), vc[cc.isin(cats[mid + odd :])].sum()) / nonmid_sum

    rad = vc[cc.isin([cats[0], cats[-1]])].sum() / nonmid_sum

    rel = 1.0 - nonmid_sum / vc.sum()

    return pd.Series({"polarisation": pol, "radicalisation": rad, "relevance": rel})


@stk_plot(
    "likert_rad_pol",
    data_format="longform",
    requires=[{"likert": True}],
    args={"normalized": "bool"},
    n_facets=(1, 2),
)
def likert_rad_pol(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    normalized: bool = True,
    width: int = 800,
    outer_factors: Sequence[str] | None = None,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Radial likert plot showing positive/negative split per question."""

    facets = facets or []
    outer_factors = list(outer_factors or [])
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1] if len(facets) > 1 else None
    # gb_cols = list(set(data.columns)-{ f0["col"], value_col })
    # Assume all other cols still in data will be used for factoring
    gb_cols = outer_factors + [
        f["col"] for f in facets[1:]
    ]  # There can be other extra cols (like labels) that should be ignored
    likert_indices = utils.gb_in_apply(
        data,
        gb_cols,
        likert_aggregate,
        cat_col=f0["col"],
        cat_order=f0["order"],
        value_col=value_col,
    ).reset_index()

    if normalized and len(likert_indices) > 1:
        likert_indices.loc[:, ["polarisation", "radicalisation"]] = likert_indices[
            ["polarisation", "radicalisation"]
        ].apply(sps.zscore)

    if f1:
        selection = alt.selection_point(fields=[f1["col"]], bind="legend")

    plot = (
        alt.Chart(likert_indices)
        .mark_point()
        .encode(
            x=alt.X("polarisation:Q"),
            y=alt.Y("radicalisation:Q"),
            size=alt.Size("relevance:Q", legend=None, scale=alt.Scale(range=[100, 500])),
            # stroke=alt.value('#777'),
            tooltip=[
                alt.Tooltip("radicalisation:Q", format=".2"),
                alt.Tooltip("polarisation:Q", format=".2"),
                alt.Tooltip("relevance:Q", format=".2"),
            ]
            + tooltip[2:],
            **(
                {
                    "color": alt.Color(
                        field=f1["col"],
                        type="nominal",
                        scale=f1["colors"],
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1["order"], width),
                        ),
                    ),
                    "opacity": alt.condition(selection, alt.value(1), alt.value(0.15)),
                }
                if f1
                else {}
            ),
        )
    )
    if f1:
        plot = plot.add_params(selection)

    return plot


@stk_plot("barbell", data_format="longform", draws=False, n_facets=(2, 2))
def barbell(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    filtered_size: int = 1,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
) -> AltairChart:
    """Draw barbell-style comparison between two categories per question."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1]

    chart_base = alt.Chart(data).encode(
        x=alt.X(
            field=value_col,
            type="quantitative",
            title=None,
            axis=alt.Axis(format=val_format),
        ),
        y=alt.Y(field=f0["col"], type="nominal", title=None, sort=f0["order"]),
        tooltip=tooltip,
    )

    chart = chart_base.mark_line(color="lightgrey", size=1, opacity=1.0).encode(
        detail=alt.Detail(field=f0["col"], type="nominal")
    )
    selection = alt.selection_point(fields=[f1["col"]], bind="legend")

    chart += (
        chart_base.mark_point(size=50, opacity=1, filled=True)
        .encode(
            color=alt.Color(
                field=f1["col"],
                type="nominal",
                # legend=alt.Legend(orient='right', title=None),
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(f1["order"], width),
                ),
                scale=f1["colors"],
                sort=f1["order"],
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
        )
        .add_params(selection)
    )  # .interactive()

    return chart


@stk_plot(
    "geoplot",
    data_format="longform",
    factor_columns=2,
    n_facets=(1, 1),
    requires=[{"topo_feature": "pass"}],
    no_faceting=True,
    aspect_ratio=(4.0 / 3.0),
    no_question_facet=True,
    args={"separate_axes": "bool"},
)
def geoplot(
    data: pd.DataFrame,
    topo_feature: tuple[str, str, str],
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = ".2f",
    tooltip: Sequence[str] | None = None,
    separate_axes: bool = False,
    outer_factors: Sequence[str] | None = None,
    outer_colors: Mapping[str, Sequence[str]] | None = None,
    value_range: tuple[float, float] | None = None,
) -> AltairChart:
    """Render a choropleth map based on annotated topojson metadata."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    outer_colors = dict(outer_colors or {})
    f0 = facets[0]

    json_url, json_meta, json_col = topo_feature
    if json_meta == "geojson":
        source = alt.Data(url=json_url, format=alt.DataFormat(property="features", type="json"))
    else:
        source = alt.topo_feature(json_url, json_meta)

    # Unescape Vega labels for the column on which we merge with the geojson
    # This is a bit of a hack, but should be the only place where we need to do this due to external data
    data = data.copy()
    data[f0["col"]] = data[f0["col"]].apply(utils.unescape_vega_label)

    lmi, lma = data[value_col].min(), data[value_col].max()
    mi, ma = value_range if value_range and not separate_axes else (lmi, lma)

    # Only show maximum on legend if min and max too close together
    [lmi, lma] if (lma - lmi) / (ma - mi) > 0.5 else [lma]
    rel_range = [(lmi - mi) / (ma - mi), (lma - mi) / (ma - mi)]

    ofv = data[outer_factors[0]].iloc[0] if outer_factors else None
    # If colors provided, create a gradient based on that
    if outer_factors and outer_colors and data[outer_factors[0]].nunique() == 1 and ofv in outer_colors:
        grad = utils.gradient_from_color(outer_colors[ofv], range=rel_range)
        scale = {"domain": [lmi, lma], "range": grad}
    else:  # Blues for pos, reds for neg, redblue for both
        # If axis spans both directions
        dmax = max(-mi, ma)
        if mi < 0.0 and ma > 0.0:
            rel_range = [lmi / dmax, lma / dmax]  # Spans both sides, so scale by dmax
        elif ma < 0.0:
            rel_range = [
                -rel_range[1],
                rel_range[0],
            ]  # Use negative part i.e. red scale

        grad = utils.gradient_subrange(utils.redblue_gradient, 11, range=rel_range)
        scale = {"domain": [lmi, lma], "range": grad}

        # if mi<0 and ma>0:
        #   scale = { 'scheme':'redblue', 'domainMid':0, 'domainMin':-dmax, 'domainMax':dmax, 'rangeMax': 0.1 }
        # elif ma<0: scale = { 'scheme': 'reds', 'reverse': True }#, 'domainMin': 0, 'domainMax':dmax }
        # else: scale = { 'scheme': 'blues' }#, 'domainMin': 0, 'domainMax':dmax }

    plot = (
        alt.Chart(source)
        .mark_geoshape(stroke="white", strokeWidth=0.1)
        .transform_lookup(
            lookup=f"properties.{json_col}",
            from_=alt.LookupData(data=data, key=f0["col"], fields=list(data.columns)),
        )
        .encode(
            tooltip=tooltip,  # [alt.Tooltip(f'properties.{json_col}:N', title=f1["col"]),
            # alt.Tooltip(f'{value_col}:Q', title=value_col, format=val_format)],
            color=alt.Color(
                field=value_col,
                type="quantitative",
                scale=alt.Scale(**scale),  # To use color scale, consider switching to opacity for value
                legend=alt.Legend(
                    format=val_format,
                    title=None,
                    orient="top-left",
                    gradientThickness=6,
                    values=[lmi, lma],
                ),
            ),
        )
        .project("mercator")
    )
    return plot


@stk_plot(
    "geobest",
    data_format="longform",
    factor_columns=2,
    n_facets=(2, 2),
    requires=[{}, {"topo_feature": "pass"}],
    no_faceting=True,
    aspect_ratio=(4.0 / 3.0),
)
def geobest(
    data: pd.DataFrame,
    topo_feature: tuple[str, str, str],
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = ".2f",
    tooltip: Sequence[str] | None = None,
    width: int = 800,
) -> AltairChart:
    """Display the top-N winning regions for each facet."""

    facets = facets or []
    tooltip = list(tooltip or [])
    f0, f1 = facets[0], facets[1]

    # Same hack as geoplot - required for periods (.) in county names
    data = data.copy()
    data[f1["col"]] = data[f1["col"]].apply(utils.unescape_vega_label)

    json_url, json_meta, json_col = topo_feature
    if json_meta == "geojson":
        source = alt.Data(url=json_url, format=alt.DataFormat(property="features", type="json"))
    else:
        source = alt.topo_feature(json_url, json_meta)

    data = data.sort_values(value_col, ascending=False).drop_duplicates([f1["col"]])
    f0["colors"]

    plot = (
        alt.Chart(source)
        .mark_geoshape(stroke="white", strokeWidth=0.1)
        .transform_lookup(
            lookup=f"properties.{json_col}",
            from_=alt.LookupData(data=data, key=f1["col"], fields=list(data.columns)),
        )
        .encode(
            tooltip=tooltip,  # [alt.Tooltip(f'properties.{json_col}:N', title=f1["col"]),
            # alt.Tooltip(f'{value_col}:Q', title=value_col, format=val_format)],
            color=alt.Color(
                field=f0["col"],
                type="nominal",
                scale=f0["colors"],
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(f0["order"], width),
                ),
            ),
        )
        .project("mercator")
    )
    return plot


# Assuming ns is ordered by unique row values, find the split points
def split_ordered(cvs: np.ndarray) -> np.ndarray:
    """Split ordered category-value pairs into positive/negative halves."""

    if len(cvs.shape) == 1:
        cvs = cvs[:, None]
    unique_idxs = np.full(len(cvs), False, dtype=np.bool_)
    unique_idxs[1:] = np.any(cvs[:-1, :] != cvs[1:, :], axis=-1)
    return np.arange(len(unique_idxs))[unique_idxs]


# Split a series of weights into groups of roughly equal sum
# This algorithm is greedy and does not split values but it is fast and should be good enough for most uses


def split_even_weight(ws: np.ndarray, n: int) -> np.ndarray:
    """Split cumulative weights into ``n`` roughly equal buckets."""

    cws = np.cumsum(ws)
    cws = np.round(cws / (cws[-1] / n)).astype("int")
    return (split_ordered(cws) + 1)[:-1]


def fd_mangle(
    vc: pd.DataFrame,
    value_col: str,
    factor_col: str,
    n_points: int = 11,
) -> pd.DataFrame:
    """Prepare data for faceted density plots by slicing contiguous regions."""

    vc = vc.sort_values(value_col)

    ws = np.ones(len(vc))
    splits = split_even_weight(ws, n_points)

    ccodes, cats = vc[factor_col].factorize()

    ofreqs = np.stack(
        [
            np.bincount(g, weights=gw, minlength=len(cats)) / gw.sum()
            for g, gw in zip(np.split(ccodes, splits), np.split(ws, splits))
        ],
        axis=0,
    )

    df = pd.DataFrame(ofreqs, columns=cats)
    df["percentile"] = np.linspace(0, 1, n_points)
    return df.melt(id_vars="percentile", value_vars=cats, var_name=factor_col, value_name="density")


@stk_plot(
    "facet_dist",
    data_format="raw",
    factor_columns=3,
    aspect_ratio=(1.0 / 1.0),
    n_facets=(1, 1),
    no_question_facet=True,
)
def facet_dist(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
) -> AltairChart:
    """Facet of distributions (hist/density) across outer factors."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0 = facets[0]
    gb_cols = [
        c for c in outer_factors if c is not None
    ]  # There can be other extra cols (like labels) that should be ignored
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[value_col, f0["col"]],
        fn=fd_mangle,
        value_col=value_col,
        factor_col=f0["col"],
    ).reset_index()
    clean_levels(ndata)
    plot = (
        alt.Chart(ndata)
        .mark_area(interpolate="natural")
        .encode(
            x=alt.X("percentile:Q", axis=alt.Axis(format="%")),
            y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%"), stack="normalize"),
            tooltip=tooltip[1:],
            color=alt.Color(f"{f0['col']}:N", scale=f0["colors"], legend=alt.Legend(orient="top")),
            # order=alt.Order('order:O')
        )
    )

    return plot


# Vectorized multinomial sampling. Should be slightly faster
def vectorized_mn(prob_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample multinomial draws for each row of ``prob_matrix``."""
    s = prob_matrix.cumsum(axis=1)
    s = s / s[:, -1][:, None]
    r = rng.uniform(size=prob_matrix.shape[0])[:, None]
    return (s < r).sum(axis=1)


def linevals(
    vals: pd.DataFrame,
    value_col: str,
    n_points: int,
    dim: str,
    cats: Sequence[str],
    ccodes: Sequence[str] | None = None,
    ocols: Sequence[str] | None = None,
    boost_signal: bool = False,
    gc: bool = False,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Prepare interpolated line values for ordered population plot."""
    ws = weights if weights is not None else np.ones(len(vals))

    order = np.lexsort((vals, ccodes)) if dim and gc else np.argsort(vals)
    splits = split_even_weight(ws[order], n_points)
    aer = np.array([g.mean() for g in np.split(vals[order], splits)])
    pdf = pd.DataFrame(aer, columns=[value_col])

    if dim:
        if ccodes is None:
            raise ValueError("ccodes must be provided when dim is specified")
        # Find the frequency of each category in ccodes
        osignal = np.stack(
            [
                np.bincount(g, weights=gw, minlength=len(cats)) / gw.sum()
                for g, gw in zip(np.split(ccodes[order], splits), np.split(ws[order], splits))
            ],
            axis=0,
        )

        ref_p = osignal.mean(axis=0) + 1e-10

        signal = osignal + 1e-10  # So even if values are zero, log and vectorized_mn would work
        if (
            boost_signal
        ):  # Boost with K-L, leaving only categories that grew in probability boosted by how much they did
            klv = signal * (np.log(signal / ref_p[None, :]))
            signal = np.maximum(1e-10, klv)
            pdf["kld"] = np.sum(klv, axis=1)

        rng = utils.stable_rng(42)
        # pdf[dim] = cats[signal.apply(lambda r: rng.multinomial(1,r/r.sum()).argmax() if r.sum()>0.0 else 0,axis=1)]
        cat_inds = vectorized_mn(signal, rng)
        pdf[dim] = np.array(cats)[cat_inds]
        pdf["probability"] = osignal[np.arange(len(cat_inds)), cat_inds]

        # pdf[dim] = pdf[cats].idxmax(axis=1)
        # pdf['weight'] = np.minimum(pdf[cats].max(axis=1),pdf['matches'])

    pdf["pos"] = np.arange(0, 1, 1.0 / len(pdf))

    if ocols is not None:
        for iv in ocols.index:
            pdf[iv] = ocols[iv]

    return pdf


@stk_plot(
    "ordered_population",
    data_format="raw",
    factor_columns=3,
    aspect_ratio=(1.0 / 1.0),
    plot_args={"group_categories": "bool", "full_data": "bool"},
    n_facets=(0, 1),
    no_question_facet=True,
)
def ordered_population(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
    group_categories: bool = False,
    full_data: bool = False,
) -> AltairChart:
    """Plot ordered categorical distributions with optional grouping."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0 = facets[0] if len(facets) > 0 else None

    n_points, maxn = 200, 1000000

    # TODO: use weight if available. linevals is ready for it, just needs to be fed in.

    # Sample down to maxn points if exceeding that
    # full_data flag is mainly here for test consistency
    if not full_data and len(data) > maxn:
        data = data.sample(maxn, replace=False, random_state=42)

    data = data.sort_values(outer_factors + [value_col])  # Value col is here to ensure consistent order for tests
    vals = data[value_col].to_numpy()

    if f0 is not None:
        fcol = f0["col"]
        cat_idx, cats = pd.factorize(data[f0["col"]])
        cats = list(cats)
    else:
        fcol = None
        cat_idx, cats = None, []

    if outer_factors:
        # This is optimized to not use pandas.groupby as it makes it about 2x faster
        # which is 2+ seconds with big datasets

        # Assume data is sorted by outer_factors, split vals into groups by them
        ofids = np.stack([data[f].cat.codes.values for f in outer_factors], axis=1)
        splits = split_ordered(ofids)
        groups = np.split(vals, splits)
        cgroups = np.split(cat_idx, splits) if len(facets) >= 1 else groups

        # Perform the equivalent of groupby
        ocols = data.iloc[[0] + list(splits)][outer_factors]
        tdf = pd.concat(
            [
                linevals(
                    g,
                    value_col=value_col,
                    dim=fcol,
                    ccodes=gc,
                    cats=cats,
                    n_points=n_points,
                    ocols=ocols.iloc[i, :],
                    gc=group_categories,
                )
                for i, (g, gc) in enumerate(zip(groups, cgroups))
            ]
        )

        # tdf = data.groupby(outer_factors,observed=True).apply(
        #   linevals,value_col=value_col,dim=fcol,cats=cats,n_points=n_points,
        #   gc=group_categories,include_groups=False).reset_index()
    else:
        tdf = linevals(
            vals,
            value_col=value_col,
            dim=fcol,
            ccodes=cat_idx,
            cats=cats,
            n_points=n_points,
            gc=group_categories,
        )
        # tdf = linevals(data,value_col=value_col,cats=cats,dim=fcol,n_points=n_points,gc=group_categories)

    # if boost_signal:
    #    tdf['matches'] = np.minimum(tdf['matches'],tdf['kld']/tdf['kld'].quantile(0.75))

    base = alt.Chart(tdf).encode(
        x=alt.X(
            "pos:Q",
            title="",
            axis=alt.Axis(
                labels=False,
                ticks=False,
                # grid=False
            ),
        )
    )
    # selection = alt.selection_multi(fields=[dim], bind='legend')
    line = base.mark_circle(size=10).encode(
        y=alt.Y(f"{value_col}:Q", impute={"value": None}, title="", axis=alt.Axis(grid=True)),
        # opacity=alt.condition(selection, alt.Opacity("matches:Q",scale=None), alt.value(0.1)),
        color=alt.Color(field=f0["col"], type="nominal", sort=f0["order"], scale=f0["colors"])
        if f0 is not None
        else alt.value("red"),
        tooltip=tooltip
        + ([alt.Tooltip("probability:Q", format=".1%", title="category prob.")] if len(facets) >= 1 else []),
    )  # .add_selection(selection)

    rule = (
        alt.Chart()
        .mark_rule(color="red", strokeDash=[2, 3])
        .encode(y=alt.Y("mv:Q"))
        .transform_joinaggregate(mv=f"mean({value_col}):Q", groupby=outer_factors)
    )

    plot = alt.layer(
        rule,
        line,
        data=tdf,
    )
    return plot  # type: ignore[return-value]


@stk_plot(
    "marimekko",
    data_format="longform",
    draws=False,
    group_sizes=True,
    nonnegative=True,
    args={"separate": "bool"},
    n_facets=(2, 2),
    priority=60,
)
def marimekko(
    data: pd.DataFrame,
    value_col: str = "value",
    facets: list[Dict[str, Any]] | None = None,
    val_format: str = "%",
    width: int = 800,
    tooltip: Sequence[str] | None = None,
    outer_factors: Sequence[str] | None = None,
    separate: bool = False,
    translate: Callable[[str], str] | None = None,
) -> AltairChart:
    """Build a Marimekko (mosaic) chart showing joint distributions."""

    facets = facets or []
    tooltip = list(tooltip or [])
    outer_factors = list(outer_factors or [])
    f0, f1 = facets[0], facets[1]
    tf = translate if translate else (lambda s: s)

    xcol, ycol, ycol_scale, yorder = (
        f1["col"],
        f0["col"],
        f0["colors"],
        list(reversed(f0["order"])),
    )

    # Fill in missing values with zero
    mdf = pd.DataFrame(
        it.product(f1["order"], f0["order"], *[data[c].unique() for c in outer_factors]),
        columns=[xcol, ycol] + outer_factors,
    )
    data = mdf.merge(data, on=[xcol, ycol] + outer_factors, how="left").fillna({value_col: 0, "group_size": 1})
    data[xcol] = pd.Categorical(data[xcol], f1["order"], ordered=True)
    data[ycol] = pd.Categorical(data[ycol], yorder, ordered=True)

    data["w"] = data["group_size"] * data[value_col]
    data.sort_values([ycol, xcol], ascending=[True, False], inplace=True)

    if separate:  # Split and center each ycol group so dynamics can be better tracked for all of them
        ndata = (
            data.groupby(outer_factors + [xcol], observed=False)[[ycol, value_col, "w"]]
            .apply(lambda df: pd.DataFrame({ycol: df[ycol], "yv": df["w"] / df["w"].sum(), "w": df["w"]}))
            .reset_index()
        )
        ndata = ndata.merge(
            ndata.groupby(outer_factors + [ycol], observed=True)["yv"].max().rename("ym").reset_index(),
            on=outer_factors + [ycol],
        ).fillna({"ym": 0.0})
        ndata = (
            ndata.groupby(outer_factors + [xcol], observed=False)[[ycol, "w", "yv", "ym"]]
            .apply(
                lambda df: pd.DataFrame(
                    {
                        ycol: df[ycol],
                        "yv": df["yv"],
                        "w": df["w"].sum(),
                        "y1": (df["ym"].cumsum() - df["ym"] / 2 - df["yv"] / 2) / df["ym"].sum(),
                        "y2": (df["ym"].cumsum() - df["ym"] / 2 + df["yv"] / 2) / df["ym"].sum(),
                    }
                )
            )
            .reset_index()
        )
    else:  # Regular marimekko
        ndata = (
            data.groupby(outer_factors + [xcol], observed=False)[[ycol, value_col, "w"]]
            .apply(
                lambda df: pd.DataFrame(
                    {
                        ycol: df[ycol],
                        "w": df["w"].sum(),
                        "yv": df["w"] / df["w"].sum(),
                        "y2": df["w"].cumsum() / df["w"].sum(),
                    }
                )
            )
            .reset_index()
        )
        ndata["y1"] = ndata["y2"] - ndata["yv"]

    ndata = (
        ndata.groupby(outer_factors + [ycol], observed=False)[[xcol, "yv", "y1", "y2", "w"]]
        .apply(
            lambda df: pd.DataFrame(
                {
                    xcol: df[xcol],
                    "xv": df["w"] / df["w"].sum(),
                    "x2": df["w"].cumsum() / df["w"].sum(),
                    "yv": df["yv"],
                    "y1": df["y1"],
                    "y2": df["y2"],
                }
            )
        )
        .reset_index()
    )
    ndata["x1"] = ndata["x2"] - ndata["xv"]

    ndata["tprop"] = ndata["xv"] * ndata["yv"]  # Overall proportion

    ndata["xmid"] = (ndata["x1"] + ndata["x2"]) / 2
    ndata["text"] = ndata[xcol].astype(str)
    # ndata['text'] = list(map(lambda x: x[0]+' '+x[1],zip(ndata[xcol].astype(str),ndata['xv'].round(2).astype(str))))
    ndata.loc[ndata[ycol] != yorder[0], "text"] = ""

    # Hack an axis title to those text labels. Not pretty but it works
    ndata["text2"] = ""
    ndata.iloc[0, -1] = xcol

    clean_levels(ndata)

    # selection = alt.selection_point(fields=[yvar], bind="legend"

    STROKE = 0.25
    base = alt.Chart(ndata)
    plot = base.mark_rect(
        strokeWidth=STROKE,
        stroke="white",
        xOffset=STROKE / 2,
        x2Offset=STROKE / 2,
        yOffset=STROKE / 2,
        y2Offset=STROKE / 2,
    ).encode(
        x=alt.X(
            "x1:Q",
            axis=alt.Axis(zindex=1, format="%", grid=False, orient="top", title=None),
            scale=alt.Scale(domain=[0, 1]),
        ),
        x2=alt.X2("x2:Q"),
        y=alt.Y(
            "y1:Q",
            axis=alt.Axis(zindex=1, format="%", title="", grid=False, labels=not separate),
            scale=alt.Scale(domain=[0, 1]),
        ),
        y2=alt.Y2("y2:Q"),
        color=alt.Color(
            f"{ycol}:N",
            legend=(
                alt.Legend(
                    orient="top",
                    titleAlign="center",
                    titleOrient="left",
                    columns=estimate_legend_columns_horiz(f0["order"], width, f0["col"]),
                )
                if len(f0["order"]) <= 5
                else alt.Legend(orient="right")
            ),  # This plot needs the vertical space to be useful for 5+ cats
            # legend=alt.Legend(orient='top',columns=estimate_legend_columns_horiz(f0['order'],width)),
            # legend=alt.Legend(orient='top',titleOrient='left', symbolStrokeWidth=0), #title=f"{yvar}"),
            scale=ycol_scale,
        ),
        tooltip=[
            alt.Tooltip(field=ycol),
            alt.Tooltip(field=xcol),
            alt.Tooltip("yv:Q", title=tf("Of column"), format=".1%"),
            alt.Tooltip("tprop:Q", title=tf("Of population"), format=".1%"),
        ]
        + tooltip[3:],
        # opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
    )
    text = base.mark_text(
        baseline="top",
        align="center",
        dy=3,
        fontSize=14,
        color="#808495",  # Streamlit default theme, which we use for the app
    ).encode(
        text=alt.Text("text:N"),
        x=alt.X("xmid:Q"),
        y=alt.Y("y1:Q"),
        tooltip=[alt.Tooltip("xv:Q", title=tf("%s size") % xcol, format=".1%")],
    )

    custom_title = base.mark_text(
        align="center",
        baseline="top",  # Position text at the bottom
        fontSize=14,
        color="#808495",
        # font='"Source Sans Pro", sans-serif',
        fontWeight=200,
        dy=20,
    ).encode(
        text=alt.Text("text2:N"),
        # x=alt.datum(0),     # Center the title horizontally
        y=alt.datum(0),  # Anchor to the bottom
    )

    return plot + text + custom_title
