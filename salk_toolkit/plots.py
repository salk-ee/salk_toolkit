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
from typing import Any, Dict, Sequence, cast

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
from salk_toolkit.pp import AltairChart, PlotInput, stk_plot
from salk_toolkit.validation import FacetMeta


# --------------------------------------------------------
#          LEGEND UTILITIES
# --------------------------------------------------------


# Helper function to clean unnecessary index columns created by groupbyfrom a dataframe
def _clean_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop auto-added `level_*` columns that pandas often emits after groupby."""

    df.drop(columns=[c for c in df.columns if c.startswith("level_")], inplace=True)
    return df


# Find a sensible approximation to the font used in vega/altair
font = font_manager.FontProperties(family="sans-serif", weight="regular")
font_file = font_manager.findfont(font)
legend_font = ImageFont.truetype(font_file, 10)


def estimate_legend_columns_horiz_naive(cats: Sequence[str], width: int) -> int:
    """Legends are not wrapped, nor is there a good way of doing accurately it in vega/altair.

    This attempts to estimate a reasonable value for columns which induces wrapping.
    Heuristic that balances legend columns purely by string length.
    """

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


def boxplot_vals(s: pd.Series, extent: float = 1.5, delta: float = 1e-4) -> pd.DataFrame:
    """Regular boxplot with quantiles and Tukey whiskers.

    Return Tukey-style summary statistics for a series.
    """

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
def boxplot_manual(p: PlotInput) -> AltairChart:
    """Manual boxplot implementation using Tukey whiskers."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("boxplots requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None
    fmt = p.val_format

    if fmt.endswith("%"):
        data[p.value_col] = data[p.value_col] * 100
        fmt = fmt[:-1] + "f"

    minv = float(data[p.value_col].min())
    maxv = float(data[p.value_col].max())
    if fmt.endswith("%"):
        minv = 0.0
    if minv == maxv:
        minv -= 0.01
        maxv += 0.01

    group_cols = p.outer_factors + [facet.col for facet in p.facets[:2] if facet is not None]
    df = (
        data.groupby(group_cols, observed=True)[p.value_col]
        .apply(boxplot_vals, delta=(maxv - minv) / 400)
        .reset_index()
    )
    _clean_levels(df)

    shared = {
        "y": alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
        **({"yOffset": alt.YOffset(field=f1.col, type="nominal", title=None, sort=f1.order)} if f1 else {}),
        "tooltip": [
            alt.Tooltip(
                field=vn,
                type="quantitative",
                format=fmt,
                title=f"{vn[0].upper() + vn[1:]} of {p.value_col}",
            )
            for vn in ["min", "q1", "mean", "q2 (median)", "q3", "max"]
        ]
        + p.tooltip[1:],
    }

    root = alt.Chart(df).encode(**shared)
    size = 12

    lower_plot = root.mark_rule().encode(
        x=alt.X(
            "tmin:Q",
            axis=alt.Axis(title=p.value_col, format=fmt),
            scale=alt.Scale(domain=[minv, maxv]),
        ),
        x2=alt.X2("q1:Q"),
    )

    middle_plot = root.mark_bar(size=size).encode(
        x=alt.X("q1p:Q"),
        x2=alt.X2("q3p:Q"),
        **(
            {"color": alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None)}
            if not f1
            else {
                "color": alt.Color(
                    field=f1.col,
                    type="nominal",
                    scale=f1.colors,
                    legend=alt.Legend(
                        orient="top",
                        columns=estimate_legend_columns_horiz(f1.order, p.width),
                    ),
                )
            }
        ),
    )

    upper_plot = root.mark_rule().encode(x=alt.X("q3:Q"), x2=alt.X2("tmax:Q"))
    middle_tick = root.mark_tick(color="white", size=size).encode(x="mean:Q")

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
def maxdiff_manual(p: PlotInput) -> AltairChart:
    """Render MaxDiff results as categorical columns."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("maxdiff plot requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None
    fmt = p.val_format

    reverse_val_col = next(
        filter(
            lambda s: p.value_col.lower() in s.lower() and "reverse" in s.lower(),
            data.columns,
        )
    )

    if fmt.endswith("%"):
        data[p.value_col] *= 100
        if reverse_val_col in data.columns:
            data[reverse_val_col] *= 100
        fmt = fmt[:-1] + "f"

    minv, maxv = data[p.value_col].min(), data[p.value_col].max()
    if fmt.endswith("%"):
        minv = 0.0
    if minv == maxv:
        minv, maxv = minv - 0.01, maxv + 0.01

    f_cols = p.outer_factors + [facet.col for facet in p.facets[:2] if facet is not None]
    df = data.groupby(f_cols, observed=True)[p.value_col].apply(boxplot_vals, delta=(maxv - minv) / 400).reset_index()
    df_reverse = (
        data.groupby(f_cols, observed=True)[reverse_val_col]
        .apply(boxplot_vals, delta=(maxv - minv) / 400)
        .reset_index()
    )
    df_reverse["mean"] = -df_reverse["mean"]
    df_reverse["kind"] = "Least important"

    df["kind"] = "Most important"

    shared = {
        "y": alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
        **({"yOffset": alt.YOffset(field=f1.col, type="nominal", title=None, sort=f1.order)} if f1 else {}),
        "tooltip": [
            alt.Tooltip(
                field=vn,
                type="quantitative",
                format=fmt,
                title=f"{vn[0].upper() + vn[1:]} of {p.value_col}",
            )
            for vn in ["mean"]
        ]
        + p.tooltip[1:],
    }

    df = pd.concat(
        [df[f_cols + ["kind", "mean"]], df_reverse[f_cols + ["kind", "mean"]]],
        ignore_index=True,
        sort=False,
    )
    _clean_levels(df)  # For test consistency
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
        if f1 is None or not f1.colors
        else alt.Color(field=f1.col, type="nominal", scale=f1.colors, legend=None),
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
def columns(p: PlotInput) -> AltairChart:
    """Simple column chart for categorical comparisons."""

    data = p.data
    if not p.facets:
        raise ValueError("columns plot requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None

    plot = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                field=p.value_col,
                type="quantitative",
                title=p.value_col,
                axis=alt.Axis(format=p.val_format),
            ),
            y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
            tooltip=p.tooltip,
            **(
                {"color": alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None)}
                if not f1
                else {
                    "yOffset": alt.YOffset(field=f1.col, type="nominal", title=None, sort=f1.order),
                    "color": alt.Color(
                        field=f1.col,
                        type="nominal",
                        scale=f1.colors,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1.order, p.width),
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
def stacked_columns(p: PlotInput, normalized: bool = False) -> AltairChart:
    """Stacked columns with optional normalization."""

    data = p.data
    if len(p.facets) < 2:
        raise ValueError("stacked_columns plot requires two facet dimensions")

    f0, f1 = p.facets[0], p.facets[1]

    data[p.value_col] = data[p.value_col] / p.filtered_size

    ldict = {category: idx for idx, category in enumerate(f1.order)}
    data["f_order"] = data[f1.col].astype("object").replace(ldict).astype("int")

    plot = (
        alt.Chart(round(data, 3), width="container")
        .mark_bar()
        .encode(
            x=alt.X(
                field=p.value_col,
                type="quantitative",
                title=p.value_col,
                axis=alt.Axis(format=p.val_format),
                **({"stack": "normalize"} if normalized else {}),
                # scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
            y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
            tooltip=p.tooltip,
            **(
                {"color": alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None)}
                if len(p.facets) <= 1
                else {
                    "order": alt.Order("f_order:O"),
                    "color": alt.Color(
                        field=f1.col,
                        type="nominal",
                        scale=f1.colors,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1.order, p.width),
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
def diff_columns(p: PlotInput, sort_descending: bool = False) -> AltairChart:
    """Difference columns chart (two categories per row with delta)."""

    data = p.data
    if len(p.facets) < 2:
        raise ValueError("diff_columns requires two facet dimensions")

    f0, f1 = p.facets[0], p.facets[1]
    fmt = p.val_format

    ind_cols = list(set(data.columns) - {p.value_col, f1.col})
    factors = [c for c in f1.order if c in data[f1.col].unique()]

    idf = data.set_index(ind_cols)
    diff = (idf[idf[f1.col] == factors[1]][p.value_col] - idf[idf[f1.col] == factors[0]][p.value_col]).reset_index()

    if sort_descending:
        f0.order = list(diff.sort_values(p.value_col, ascending=False)[f0.col])

    encoded_tooltip = [
        alt.Tooltip(field=f0.col, type="nominal"),
        alt.Tooltip(
            field=p.value_col,
            type="quantitative",
            format=fmt,
            title=f"{p.value_col} difference",
        ),
    ]

    plot = (
        alt.Chart(round(diff, 3), width="container")
        .mark_bar()
        .encode(
            y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
            x=alt.X(
                field=p.value_col,
                type="quantitative",
                title=f"{factors[1]} - {factors[0]}",
                axis=alt.Axis(format=fmt, title=f"{factors[0]} <> {factors[1]}"),
            ),
            tooltip=encoded_tooltip,
            color=alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None),
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
def massplot(p: PlotInput) -> AltairChart:
    """Mass plot showing distributions vs. categorical facets."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("massplot requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None

    data["group_size"] = data["group_size"] / p.filtered_size

    plot = (
        alt.Chart(round(data, 3), width="container")
        .mark_circle()
        .encode(
            y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
            x=alt.X(
                field=p.value_col,
                type="quantitative",
                title=p.value_col,
                axis=alt.Axis(format=p.val_format),
                # scale=alt.Scale(domain=[0,30]) #see lõikab mõnedes jaotustes parema ääre ära
            ),
            size=alt.Size("group_size:Q", legend=None, scale=alt.Scale(range=[100, 500])),
            opacity=alt.value(1.0),
            stroke=alt.value("#777"),
            tooltip=p.tooltip + [alt.Tooltip("group_size:N", format=".1%", title="Group size")],
            **(
                {"color": alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None)}
                if not f1
                else {
                    "yOffset": alt.YOffset(field=f1.col, type="nominal", title=None, sort=f1.order),
                    "color": alt.Color(
                        field=f1.col,
                        type="nominal",
                        scale=f1.colors,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1.order, p.width),
                        ),
                    ),
                }
            ),
        )
    )
    return plot


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
        # Use tuple for iloc indexing to match pandas stubs
        x_mid = x.iloc[tuple([mids, slice(None)])].copy()  # type: ignore[call-overload]
        x_other = x.iloc[tuple([nonmid, slice(None)])].copy()  # type: ignore[call-overload]

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
    p: PlotInput,
) -> AltairChart:
    """Display likert responses as diverging stacked bars."""

    data = p.data.copy()
    facets = p.facets
    if not facets:
        raise ValueError("likert_bars requires at least one facet dimension")

    if len(facets) == 1:
        data["question"] = facets[0].col
        synthetic = FacetMeta(
            col="question",
            ocol="question",
            order=[facets[0].col],
            colors=alt.Undefined,
            neutrals=[],
            meta=facets[0].meta,
        )
        facets = [facets[0], synthetic]

    if len(facets) >= 3:
        f0, f1, f2 = facets[0], facets[2], facets[1]
    else:
        f0, f1, f2 = facets[0], facets[1], None

    neg, neutral, pos = utils.split_to_neg_neutral_pos(f0.order, f0.neutrals)
    ninds = [f0.order.index(c) for c in neutral]

    gb_cols = p.outer_factors + [facet.col for facet in facets[1:]]
    bar_data = data.groupby(gb_cols, group_keys=False, observed=False)[data.columns].apply(
        make_start_end,
        value_col=p.value_col,
        cat_col=f0.col,
        cat_order=f0.order,
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
                field=f1.col,
                type="nominal",
                axis=alt.Axis(title=None, offset=5, ticks=False, minExtent=60, domain=False),
                sort=f1.order,
            ),
            tooltip=p.tooltip,
            color=alt.Color(
                field=f0.col,
                type="nominal",
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    columns=estimate_legend_columns_horiz(f0.order, p.width, extra_text=f1.order),
                ),
                scale=f0.colors,
            ),
            **({"yOffset": alt.YOffset(field=f2.col, type="nominal", title=None, sort=f2.order)} if f2 else {}),
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
    continuous=True,
)
def density(
    p: PlotInput,
    stacked: bool = False,
    bw: float | None = None,
) -> AltairChart:
    """Stacked (or overlapped) density plot for continuous responses."""

    data = p.data.copy()
    f0 = p.facets[0] if p.facets else None
    gb_cols = [c for c in p.outer_factors + [f.col for f in p.facets] if c is not None]

    lims = list(data[p.value_col].quantile([0.005, 0.995]))
    data = data[(data[p.value_col] >= lims[0]) & (data[p.value_col] <= lims[1])]

    ls = np.linspace(data[p.value_col].min() - 1e-10, data[p.value_col].max() + 1e-10, 101)
    if bw is None:
        bw = kde_bw(data[[p.value_col]].sample(10000, replace=True).to_numpy())
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[p.value_col],
        fn=kde_1d,
        value_col=p.value_col,
        ls=ls,
        scale=stacked,
        bw=bw,
    ).reset_index()
    _clean_levels(ndata)

    selection = None
    if f0:
        selection = alt.selection_point(name="param_1", fields=[f0.col], bind="legend")

    if stacked:
        if f0:
            ldict = dict(zip(f0.order, reversed(range(len(f0.order)))))
            ndata.loc[:, "order"] = ndata[f0.col].astype("object").replace(ldict).astype("int")

        ndata["density"] /= len(data)
        enc_kwargs: Dict[str, Any] = {}
        if f0:
            enc_kwargs = {
                "fill": alt.Fill(
                    field=f0.col,
                    type="nominal",
                    scale=f0.colors,
                    legend=alt.Legend(
                        orient="top",
                        columns=estimate_legend_columns_horiz(f0.order, p.width),
                    ),
                ),
                "order": alt.Order("order:O"),
                "opacity": alt.condition(selection, alt.value(1), alt.value(0.15)),  # type: ignore[call-overload]
            }

        plot = (
            alt.Chart(ndata)
            .mark_area(interpolate="natural")
            .encode(
                x=alt.X(field=p.value_col, type="quantitative"),
                y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%"), stack="zero"),
                tooltip=p.tooltip[1:],
                **enc_kwargs,
            )
        )
    else:
        enc_kwargs = {}
        if f0:
            enc_kwargs = {
                "color": alt.Color(
                    field=f0.col,
                    type="nominal",
                    scale=f0.colors,
                    legend=alt.Legend(
                        orient="top",
                        columns=estimate_legend_columns_horiz(f0.order, p.width),
                    ),
                ),
                "order": alt.Order("order:O"),
                "opacity": alt.condition(selection, alt.value(1), alt.value(0.15)),  # type: ignore[call-overload]
            }

        plot = (
            alt.Chart(ndata)
            .mark_line()
            .encode(
                x=alt.X(field=p.value_col, type="quantitative"),
                y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%")),
                tooltip=p.tooltip[1:],
                **enc_kwargs,
            )
        )

    if selection is not None:
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
    p: PlotInput,
    bw: float | None = None,
) -> AltairChart:
    """Violin plot drawing densities per facet."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("violin plot requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None
    gb_cols = p.outer_factors + [f.col for f in p.facets]

    ls = np.linspace(data[p.value_col].min() - 1e-10, data[p.value_col].max() + 1e-10, 101)
    if bw is None:
        bw = kde_bw(data[[p.value_col]].sample(10000, replace=True).to_numpy())
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[p.value_col],
        fn=kde_1d,
        value_col=p.value_col,
        ls=ls,
        scale=True,
        bw=bw,
    ).reset_index()
    _clean_levels(ndata)

    if f1:
        ldict = dict(zip(f1.order, reversed(range(len(f1.order)))))
        ndata.loc[:, "order"] = ndata[f1.col].astype("object").replace(ldict).astype("int")

    ndata["density"] /= len(data)
    plot = (
        alt.Chart(ndata)
        .mark_area(interpolate="natural")
        .encode(
            x=alt.X(field=p.value_col, type="quantitative"),
            y=alt.Y(
                "density:Q",
                axis=alt.Axis(title=None, labels=False, values=[0], grid=False),
                stack="center",
            ),
            row=alt.Row(
                field=f0.col,
                type="nominal",
                header=alt.Header(orient="top", title=None),
                spacing=5,
                sort=f0.order,
            ),
            tooltip=p.tooltip[1:],
            # color=alt.Color(f'{question_col}:N'),
            **(
                {
                    "color": alt.Color(
                        field=f1.col,
                        type="nominal",
                        scale=f1.colors,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1.order, p.width),
                        ),
                    ),
                    "order": alt.Order("order:O"),
                }
                if f1
                else {"color": alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None)}
            ),
        )
        .properties(width=p.width, height=70)
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
    p: PlotInput,
    reorder: bool = False,
    log_colors: bool = False,
) -> AltairChart:
    """Heatmap-style matrix plot (optionally reorder rows/cols via clustering)."""

    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("matrix plot requires two facet dimensions")

    f0, f1 = p.facets[0], p.facets[1]
    fmt = p.val_format

    fcols = [c for c in data.columns if c not in [p.value_col, f0.col]]
    if len(fcols) == 1 and reorder:  # Reordering only works if no external facets
        X = data.pivot(columns=f1.col, index=f0.col).to_numpy()
        f0.order = np.array(f0.order)[cluster_based_reorder(X)].tolist()
        f1.order = np.array(f1.order)[cluster_based_reorder(X.T)].tolist()

    if log_colors:
        data["val_log"] = np.log(data[p.value_col])
        data["val_log"] -= data["val_log"].min()  # Keep it all positive
        scale_v = "val_log"
    else:
        scale_v = p.value_col

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
            x=alt.X(field=f1.col, type="nominal", title=None, sort=f1.order),
            y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
            color=alt.Color(
                field=scale_v,
                type="quantitative",
                scale=alt.Scale(**scale),
                legend=(alt.Legend(title=None) if not log_colors else None),
            ),
            tooltip=p.tooltip,
        )
    )

    # Add in numerical values
    if len(f1.order) < 20:  # only if we have less than 20 columns
        threshold = round((0.25 * swidth) ** 2, 3)  # Rounding needed for tests to be stable
        text = plot.mark_text().encode(
            text=alt.Text(field=p.value_col, type="quantitative", format=fmt),
            color=alt.condition(
                (alt.datum[scale_v] - smid) ** 2 > threshold,
                alt.value("white"),
                alt.value("black"),
            ),
            tooltip=p.tooltip,
        )
        plot += text

    return plot


@stk_plot(
    "corr_matrix",
    data_format="raw",
    aspect_ratio=(1 / 0.8),
    n_facets=(1, 1),
    args={"reorder": "bool"},
)
def corr_matrix(
    p: PlotInput,
    reorder: bool = False,
) -> AltairChart:
    """Correlation matrix for raw grouped data."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("corr_matrix requires at least one facet dimension")
    if "id" not in data.columns:
        raise Exception("Corr_matrix only works for groups of continuous variables")

    fmt = p.val_format

    # id is required to match the rows for correllations
    cm = (
        data.pivot_table(index="id", columns=p.facets[0].col, values=p.value_col, observed=False)
        .corr()
        .reset_index(names="index")
    )
    cm_long = cm.melt(
        id_vars=["index"],
        value_vars=cm.columns,
        var_name=p.facets[0].col,
        value_name=p.value_col,
    )

    order = p.facets[0].order
    lower_tri = cm_long["index"].map(lambda x: order.index(x)).astype(int) > cm_long[p.facets[0].col].map(
        lambda x: order.index(x)
    ).astype(int)
    cm_long = cm_long[lower_tri]

    matrix_params = p.model_copy(deep=True)
    matrix_params.data = cm_long
    matrix_params.value_col = p.value_col
    matrix_params.val_format = fmt
    matrix_params.tooltip = [
        alt.Tooltip(field=p.value_col, type="quantitative"),
        alt.Tooltip("index:N"),
        alt.Tooltip(field=p.facets[0].col, type="nominal"),
    ]
    matrix_params.facets = [
        FacetMeta(
            col="index",
            ocol="index",
            order=order,
            colors=alt.Undefined,
            neutrals=[],
            meta=p.facets[0].meta,
        ),
        FacetMeta(
            col=p.facets[0].col,
            ocol=p.facets[0].ocol,
            order=order,
            colors=p.facets[0].colors,
            neutrals=p.facets[0].neutrals,
            meta=p.facets[0].meta,
        ),
    ]

    return matrix(matrix_params, reorder=reorder, log_colors=False)  # type: ignore[return-value]


# Convert a categorical fx facet into a continous value and axis if the categories are numeric.
def _cat_to_cont_axis(
    data: pd.DataFrame,
    fx: FacetMeta | Dict[str, Any],
) -> tuple[alt.X, pd.DataFrame]:
    """Convert categorical axis to numeric when labels are numeric strings."""

    col = fx.col if isinstance(fx, FacetMeta) else fx["col"]
    order = fx.order if isinstance(fx, FacetMeta) else fx.get("order", [])
    x_cont = pd.to_numeric(
        data[col].apply(utils.unescape_vega_label),
        errors="coerce",
    )  # Unescape required as . gets escaped
    if x_cont.notna().all():
        data[col] = x_cont.astype("float")
        x_axis = alt.X(
            field=col,
            type="quantitative",
            title=None,
            axis=alt.Axis(labelAngle=0, values=list(data[col].unique())),
        )
    else:
        x_axis = alt.X(
            field=col,
            type="nominal",
            title=None,
            sort=order,
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
    p: PlotInput,
    smooth: bool = False,
) -> AltairChart:
    """Line chart with optional smoothing and categorical faceting."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("lines plot requires at least one facet dimension")

    fmt = p.val_format or ".2f"
    chart_width = p.width
    if len(p.facets) == 1:
        fx = p.facets[0]
        fy = None
    else:
        fy, fx = p.facets[0], p.facets[1]
    if smooth:
        smoothing = "basis"
        points = "transparent"
    else:
        smoothing = "natural"
        points = True

    # See if we should use a continous axis (if categoricals are actually numbers)
    x_axis, data = _cat_to_cont_axis(data, fx)

    plot = (
        alt.Chart(data)
        .mark_line(point=points, interpolate=smoothing)
        .encode(
            x=x_axis,
            y=alt.Y(
                field=p.value_col,
                type="quantitative",
                title=(p.value_col if len(p.value_col) < 20 else None),
                axis=alt.Axis(format=fmt),
            ),
            tooltip=p.tooltip,
            **(
                {
                    "color": alt.Color(
                        field=fy.col,
                        type="nominal",
                        scale=fy.colors,
                        sort=fy.order,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(fy.order, chart_width),
                        ),
                    )
                }
                if fy is not None
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
    p: PlotInput,
    hdi1: float = 0.94,
    hdi2: float = 0.5,
) -> AltairChart:
    """Line chart showing central tendency plus HDI ribbons."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("lines_hdi plot requires at least one facet dimension")

    fmt = p.val_format or ".2f"

    if len(p.facets) == 1:
        fx = p.facets[0]
        fy = None
    else:
        fy, fx = p.facets[0], p.facets[1]

    hdf = draws_to_hdis(data, p.value_col, [hdi1, hdi2])

    selection = None
    if fy is not None:
        # Draw them in reverse order so the things that are first (i.e. most important)
        # are drawn last (i.e. on top of others). Also draw wider hdi before the narrower
        hdf.sort_values([fy.col, "hdi"], ascending=[False, False], inplace=True)
        selection = alt.selection_point(name="param_5", fields=[fy.col], bind="legend")

    # See if we should use a continous axis (if categoricals are actually numbers)
    x_axis, hdf = _cat_to_cont_axis(hdf, fx)

    color_kwargs: Dict[str, Any] = {}
    if fy is not None:
        color_kwargs = {
            "fill": alt.Fill(
                field=fy.col,
                type="nominal",
                sort=fy.order,
                scale=fy.colors,
                legend=alt.Legend(symbolOpacity=1),
            ),
            "opacity": alt.condition(  # type: ignore[call-overload]
                selection,
                alt.Opacity(
                    "hdi:N",
                    legend=None,
                    scale=utils.to_alt_scale({0.5: 0.75, 0.94: 0.25}),
                ),
                alt.value(0.1),
            ),
        }

    plot = (
        alt.Chart(hdf)
        .mark_area(interpolate="basis")
        .encode(
            x=x_axis,
            y=alt.Y(
                "lo:Q",
                axis=alt.Axis(
                    format=fmt,
                    title=(p.value_col if len(p.value_col) < 20 else None),
                ),
                title=p.value_col,
            ),
            y2=alt.Y2("hi:Q"),
            tooltip=[
                alt.Tooltip("hdi:N", title="HDI", format=".0%"),
                alt.Tooltip("lo:Q", title="HDI lower", format=fmt),
                alt.Tooltip("hi:Q", title="HDI upper", format=fmt),
            ]
            + p.tooltip[1:],
            **color_kwargs,
        )
    )

    if selection is not None:
        plot = plot.add_params(selection).properties(name="view_2")
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
    p: PlotInput,
) -> AltairChart:
    """Area chart with smoothing for cumulative comparisons."""

    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("area_smooth plot requires two facet dimensions")

    fy, fx = p.facets[0], p.facets[1]
    chart_width = p.width

    ldict = dict(zip(fy.order, range(len(fy.order))))
    data.loc[:, "order"] = data[fy.col].astype("object").replace(ldict).astype("int")

    x_axis, data = _cat_to_cont_axis(data, fx)

    plot = (
        alt.Chart(data)
        .mark_area(interpolate="natural")
        .encode(
            x=x_axis,
            y=alt.Y(
                field=p.value_col,
                type="quantitative",
                title=None,
                stack="normalize",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            order=alt.Order("order:O"),
            color=alt.Color(
                field=fy.col,
                type="nominal",
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(fy.order, chart_width),
                ),
                sort=fy.order,
                scale=fy.colors,
            ),
            tooltip=p.tooltip,
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
    p: PlotInput,
    normalized: bool = True,
) -> AltairChart:
    """Radial likert plot showing positive/negative split per question."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("likert_rad_pol requires at least one facet dimension")

    f0 = p.facets[0]
    f1 = p.facets[1] if len(p.facets) > 1 else None
    chart_width = p.width

    # gb_cols = list(set(data.columns)-{ f0["col"], value_col })
    # Assume all other cols still in data will be used for factoring
    gb_cols = p.outer_factors + [
        f.col for f in p.facets[1:]
    ]  # There can be other extra cols (like labels) that should be ignored
    likert_indices = utils.gb_in_apply(
        data,
        gb_cols,
        likert_aggregate,
        cat_col=f0.col,
        cat_order=f0.order,
        value_col=p.value_col,
    ).reset_index()

    if normalized and len(likert_indices) > 1:
        likert_indices.loc[:, ["polarisation", "radicalisation"]] = likert_indices[
            ["polarisation", "radicalisation"]
        ].apply(sps.zscore)

    if f1:
        selection = alt.selection_point(name="param_3", fields=[f1.col], bind="legend")

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
            + p.tooltip[2:],
            **(
                {
                    "color": alt.Color(
                        field=f1.col,
                        type="nominal",
                        scale=f1.colors,
                        legend=alt.Legend(
                            orient="top",
                            columns=estimate_legend_columns_horiz(f1.order, chart_width),
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
    p: PlotInput,
) -> AltairChart:
    """Draw barbell-style comparison between two categories per question."""

    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("barbell plot requires two facet dimensions")

    f0, f1 = p.facets[0], p.facets[1]
    fmt = p.val_format
    chart_width = p.width

    chart_base = alt.Chart(data).encode(
        x=alt.X(
            field=p.value_col,
            type="quantitative",
            title=None,
            axis=alt.Axis(format=fmt),
        ),
        y=alt.Y(field=f0.col, type="nominal", title=None, sort=f0.order),
        tooltip=p.tooltip,
    )

    chart = chart_base.mark_line(color="lightgrey", size=1, opacity=1.0).encode(
        detail=alt.Detail(field=f0.col, type="nominal")
    )
    selection = alt.selection_point(name="param_4", fields=[f1.col], bind="legend")

    chart += (
        chart_base.mark_point(size=50, opacity=1, filled=True)
        .encode(
            color=alt.Color(
                field=f1.col,
                type="nominal",
                # legend=alt.Legend(orient='right', title=None),
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(f1.order, chart_width),
                ),
                scale=f1.colors,
                sort=f1.order,
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
        )
        .properties(name="view_3")
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
    p: PlotInput,
    topo_feature: tuple[str, str, str],
    separate_axes: bool = False,
) -> AltairChart:
    """Render a choropleth map based on annotated topojson metadata."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("geoplot requires at least one facet dimension")

    outer_colors = dict(p.outer_colors or {})
    fmt = p.val_format or ".2f"
    f0 = p.facets[0]
    vr = p.value_range

    json_url, json_meta, json_col = topo_feature
    if json_meta == "geojson":
        source = alt.Data(url=json_url, format=alt.DataFormat(property="features", type="json"))
    else:
        source = alt.topo_feature(json_url, json_meta)

    # Unescape Vega labels for the column on which we merge with the geojson
    # This is a bit of a hack, but should be the only place where we need to do this due to external data
    data = data.copy()
    data[f0.col] = data[f0.col].apply(utils.unescape_vega_label)

    lmi, lma = data[p.value_col].min(), data[p.value_col].max()
    mi, ma = vr if vr and not separate_axes else (lmi, lma)

    # Only show maximum on legend if min and max too close together
    [lmi, lma] if (lma - lmi) / (ma - mi) > 0.5 else [lma]
    rel_range = [(lmi - mi) / (ma - mi), (lma - mi) / (ma - mi)]

    outer0 = p.outer_factors[0] if p.outer_factors else None
    ofv = data[outer0].iloc[0] if outer0 else None
    # If colors provided, create a gradient based on that
    if p.outer_factors and outer_colors and data[p.outer_factors[0]].nunique() == 1 and ofv in outer_colors:
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
            from_=alt.LookupData(data=data, key=f0.col, fields=list(data.columns)),
        )
        .encode(
            tooltip=p.tooltip,  # [alt.Tooltip(f'properties.{json_col}:N', title=f1["col"]),
            # alt.Tooltip(f'{p.value_col}:Q', title=p.value_col, format=p.val_format)],
            color=alt.Color(
                field=p.value_col,
                type="quantitative",
                scale=alt.Scale(**scale),  # To use color scale, consider switching to opacity for value
                legend=alt.Legend(
                    format=fmt,
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
    p: PlotInput,
    topo_feature: tuple[str, str, str],
) -> AltairChart:
    """Display the top-N winning regions for each facet."""

    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("geobest plot requires two facet dimensions")

    f0, f1 = p.facets[0], p.facets[1]
    chart_width = p.width

    # Same hack as geoplot - required for periods (.) in county names
    data = data.copy()
    data[f1.col] = data[f1.col].apply(utils.unescape_vega_label)

    json_url, json_meta, json_col = topo_feature
    if json_meta == "geojson":
        source = alt.Data(url=json_url, format=alt.DataFormat(property="features", type="json"))
    else:
        source = alt.topo_feature(json_url, json_meta)

    data = data.sort_values(p.value_col, ascending=False).drop_duplicates([f1.col])

    plot = (
        alt.Chart(source)
        .mark_geoshape(stroke="white", strokeWidth=0.1)
        .transform_lookup(
            lookup=f"properties.{json_col}",
            from_=alt.LookupData(data=data, key=f1.col, fields=list(data.columns)),
        )
        .encode(
            tooltip=p.tooltip,  # [alt.Tooltip(f'properties.{json_col}:N', title=f1["col"]),
            # alt.Tooltip(f'{p.value_col}:Q', title=p.value_col, format=p.val_format)],
            color=alt.Color(
                field=f0.col,
                type="nominal",
                scale=f0.colors,
                legend=alt.Legend(
                    orient="top",
                    columns=estimate_legend_columns_horiz(f0.order, chart_width),
                ),
            ),
        )
        .project("mercator")
    )
    return plot


# Assuming ns is ordered by unique row values, find the split points
def _split_ordered(cvs: np.ndarray) -> np.ndarray:
    """Split ordered category-value pairs into positive/negative halves."""

    if len(cvs.shape) == 1:
        cvs = cvs[:, None]
    unique_idxs = np.full(len(cvs), False, dtype=np.bool_)
    unique_idxs[1:] = np.any(cvs[:-1, :] != cvs[1:, :], axis=-1)
    return np.arange(len(unique_idxs))[unique_idxs]


def _split_even_weight(ws: np.ndarray, n: int) -> np.ndarray:
    """Split cumulative weights into ``n`` roughly equal buckets.

    This algorithm is greedy and does not split values but it is fast and should be good enough for most uses.
    """

    cws = np.cumsum(ws)
    cws = np.round(cws / (cws[-1] / n)).astype("int")
    return (_split_ordered(cws) + 1)[:-1]


def fd_mangle(
    vc: pd.DataFrame,
    value_col: str,
    factor_col: str,
    n_points: int = 11,
) -> pd.DataFrame:
    """Prepare data for faceted density plots by slicing contiguous regions."""

    vc = vc.sort_values(value_col)

    ws = np.ones(len(vc))
    splits = _split_even_weight(ws, n_points)

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
    p: PlotInput,
) -> AltairChart:
    """Facet of distributions (hist/density) across outer factors."""

    data = p.data.copy()
    if not p.facets:
        raise ValueError("facet_dist requires at least one facet dimension")

    f0 = p.facets[0]
    gb_cols = [
        c for c in p.outer_factors if c is not None
    ]  # There can be other extra cols (like labels) that should be ignored
    ndata = utils.gb_in_apply(
        data,
        gb_cols,
        cols=[p.value_col, f0.col],
        fn=fd_mangle,
        value_col=p.value_col,
        factor_col=f0.col,
    ).reset_index()
    _clean_levels(ndata)
    plot = (
        alt.Chart(ndata)
        .mark_area(interpolate="natural")
        .encode(
            x=alt.X("percentile:Q", axis=alt.Axis(format="%")),
            y=alt.Y("density:Q", axis=alt.Axis(title=None, format="%"), stack="normalize"),
            tooltip=p.tooltip[1:],
            color=alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=alt.Legend(orient="top")),
            # order=alt.Order('order:O')
        )
    )

    return plot


# Vectorized multinomial sampling. Should be slightly faster
def _vectorized_mn(prob_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample multinomial draws for each row of ``prob_matrix``."""
    s = prob_matrix.cumsum(axis=1)
    s = s / s[:, -1][:, None]
    r = rng.uniform(size=prob_matrix.shape[0])[:, None]
    return (s < r).sum(axis=1)


def _linevals(
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
    # Cast to ndarray for indexing to match numpy stubs
    ws_arr = np.asarray(ws)
    vals_arr = np.asarray(vals)
    splits = _split_even_weight(ws_arr[order], n_points)  # type: ignore[call-overload]
    aer = np.array([g.mean() for g in np.split(vals_arr[order], splits)])  # type: ignore[call-overload]
    pdf = pd.DataFrame(aer, columns=[value_col])

    if dim:
        if ccodes is None:
            raise ValueError("ccodes must be provided when dim is specified")
        # Find the frequency of each category in ccodes
        # Cast to ndarray for indexing to match numpy stubs
        ccodes_arr = np.asarray(ccodes)
        ws_arr = np.asarray(ws)
        osignal = np.stack(
            [
                np.bincount(g, weights=gw, minlength=len(cats)) / gw.sum()
                for g, gw in zip(np.split(ccodes_arr[order], splits), np.split(ws_arr[order], splits))  # type: ignore[call-overload]
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
        cat_inds = _vectorized_mn(signal, rng)
        pdf[dim] = np.array(cats)[cat_inds]
        pdf["probability"] = osignal[np.arange(len(cat_inds)), cat_inds]

        # pdf[dim] = pdf[cats].idxmax(axis=1)
        # pdf['weight'] = np.minimum(pdf[cats].max(axis=1),pdf['matches'])

    pdf["pos"] = np.arange(0, 1, 1.0 / len(pdf))

    if ocols is not None:
        # ocols can be a pandas Series (with .index) or a dict-like object
        if isinstance(ocols, pd.Series):
            for iv in ocols.index:  # type: ignore[attr-defined]
                pdf[iv] = ocols[iv]
        else:
            for iv in ocols:  # type: ignore[call-overload]
                pdf[iv] = ocols[iv]  # type: ignore[call-overload]

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
    p: PlotInput,
    group_categories: bool = False,
    full_data: bool = False,
) -> AltairChart:
    """Plot ordered categorical distributions with optional grouping."""

    data = p.data.copy()
    outer_cols = list(p.outer_factors)
    f0 = p.facets[0] if p.facets else None

    n_points, maxn = 200, 1000000

    # TODO: use weight if available. linevals is ready for it, just needs to be fed in.

    # Sample down to maxn points if exceeding that
    # full_data flag is mainly here for test consistency
    if not full_data and len(data) > maxn:
        data = data.sample(maxn, replace=False, random_state=42)

    data = data.sort_values(outer_cols + [p.value_col])  # Value col ensures consistent order for tests
    vals = data[p.value_col].to_numpy()

    if f0 is not None:
        fcol = f0.col
        cat_idx, cats = pd.factorize(data[f0.col])
        cats = list(cats)
    else:
        fcol = None
        cat_idx, cats = None, []

    if outer_cols:
        # This is optimized to not use pandas.groupby as it makes it about 2x faster
        # which is 2+ seconds with big datasets

        # Assume data is sorted by outer_factors, split vals into groups by them
        ofids = np.stack([data[f].cat.codes.values for f in outer_cols], axis=1)  # type: ignore[call-overload]
        splits = _split_ordered(ofids)
        groups = np.split(vals, splits)  # type: ignore[call-overload]
        cgroups = np.split(cat_idx, splits) if p.facets else groups  # type: ignore[call-overload]

        # Perform the equivalent of groupby
        ocols = data.iloc[[0] + list(splits)][outer_cols]
        tdf = pd.concat(
            [
                _linevals(
                    g,
                    value_col=p.value_col,
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
        tdf = _linevals(
            vals,
            value_col=p.value_col,
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
        y=alt.Y(f"{p.value_col}:Q", impute={"value": None}, title="", axis=alt.Axis(grid=True)),
        # opacity=alt.condition(selection, alt.Opacity("matches:Q",scale=None), alt.value(0.1)),
        color=alt.Color(field=f0.col, type="nominal", sort=f0.order, scale=f0.colors)
        if f0 is not None
        else alt.value("red"),
        tooltip=p.tooltip + ([alt.Tooltip("probability:Q", format=".1%", title="category prob.")] if p.facets else []),
    )  # .add_selection(selection)

    rule = (
        alt.Chart()
        .mark_rule(color="red", strokeDash=[2, 3])
        .encode(y=alt.Y("mv:Q"))
        .transform_joinaggregate(mv=f"mean({p.value_col}):Q", groupby=outer_cols)
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
    p: PlotInput,
    separate: bool = False,
) -> AltairChart:
    """Build a Marimekko (mosaic) chart showing joint distributions."""

    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("marimekko plot requires two facet dimensions")

    outer_cols = list(p.outer_factors)
    f0, f1 = p.facets[0], p.facets[1]
    tf = p.translate or (lambda s: s)
    chart_width = p.width

    xcol, ycol, ycol_scale, yorder = (
        f1.col,
        f0.col,
        f0.colors,
        list(reversed(f0.order)),
    )

    # Fill in missing values with zero
    mdf = pd.DataFrame(
        it.product(f1.order, f0.order, *[data[c].unique() for c in outer_cols]),
        columns=[xcol, ycol] + outer_cols,
    )
    data = mdf.merge(data, on=[xcol, ycol] + outer_cols, how="left").fillna({p.value_col: 0, "group_size": 1})
    data[xcol] = pd.Categorical(data[xcol], f1.order, ordered=True)
    data[ycol] = pd.Categorical(data[ycol], yorder, ordered=True)

    data["w"] = data["group_size"] * data[p.value_col]
    data.sort_values([ycol, xcol], ascending=[True, False], inplace=True)

    if separate:  # Split and center each ycol group so dynamics can be better tracked for all of them
        ndata = (
            data.groupby(outer_cols + [xcol], observed=False)[[ycol, p.value_col, "w"]]
            .apply(lambda df: pd.DataFrame({ycol: df[ycol], "yv": df["w"] / df["w"].sum(), "w": df["w"]}))
            .reset_index()
        )
        ndata = ndata.merge(
            cast(pd.Series, ndata.groupby(outer_cols + [ycol], observed=True)["yv"].max()).rename("ym").reset_index(),
            on=outer_cols + [ycol],
        ).fillna({"ym": 0.0})
        ndata = (
            ndata.groupby(outer_cols + [xcol], observed=False)[[ycol, "w", "yv", "ym"]]
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
            data.groupby(outer_cols + [xcol], observed=False)[[ycol, p.value_col, "w"]]
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
        ndata.groupby(outer_cols + [ycol], observed=False)[[xcol, "yv", "y1", "y2", "w"]]
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

    _clean_levels(ndata)

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
            field=ycol,
            type="nominal",
            legend=(
                alt.Legend(
                    orient="top",
                    titleAlign="center",
                    titleOrient="left",
                    columns=estimate_legend_columns_horiz(f0.order, chart_width, f0.col),
                )
                if len(f0.order) <= 5
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
        + p.tooltip[3:],
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
