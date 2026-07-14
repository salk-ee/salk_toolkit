"""Tests for `create_plot_payload` (PlotPayload v1) and the `PlotInput.return_df` plot contract."""

from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import pytest

from salk_toolkit import election_models as stk_elections
from salk_toolkit import payload as stk_payload
from salk_toolkit import pp
from salk_toolkit import plots as stk_plots
from salk_toolkit.pp import FacetMeta, matching_plots, PlotInput
from salk_toolkit.validation import DataMeta, ElectoralSystem, GroupOrColumnMeta, PlotDescriptor, soft_validate


def prepared(plot_fn, cell_pi, **kwargs):
    """Build the plot's chart and return (in `.data`) the frame the payload fallback reads off it."""

    chart = plot_fn(cell_pi, **kwargs)
    frame = stk_payload._chart_frame(chart)
    assert frame is not None, "payload fallback found no frame on the chart"
    return SimpleNamespace(data=frame)


def test_payload_flag_in_plot_meta():
    """`payload=True` marks the authoritative return_df path (only non-chart plots); the rest
    are covered by the chart-extraction fallback, so the flag is not a coverage gate."""

    # coalition_applet is a streamlit widget (no chart) -> must use return_df
    assert pp.get_plot_meta("coalition_applet").payload is True
    # chart plots go through the fallback, so they carry no flag
    assert pp.get_plot_meta("columns").payload is False
    assert pp.get_plot_meta("violin").payload is False


@pytest.fixture
def registered_two_factor_plot():
    """Register a throwaway plot for the duration of a test."""

    @pp.stk_plot("__test_two_factor_plot")
    def _plot_fn(p):
        return p

    try:
        yield "__test_two_factor_plot"
    finally:
        pp._stk_deregister("__test_two_factor_plot")


@pytest.fixture
def small_pi_fixture():
    """A small synthetic PlotInput with two categorical outer-factor columns."""

    rng = np.random.default_rng(0)
    n = 12
    data = pd.DataFrame(
        {
            "score": rng.uniform(0, 1, n),
            "group.a": pd.Categorical(np.tile(["A1", "A2"], n // 2), categories=["A1", "A2"]),
            "group.b": pd.Categorical(np.repeat(["B1", "B2", "B3"], n // 3), categories=["B1", "B2", "B3"]),
        }
    )
    return PlotInput(data=data, col_meta={}, value_col="score")


@pytest.fixture
def ppd_two_factors(registered_two_factor_plot):
    """A PlotDescriptor with two outer factor columns, targeting the throwaway test plot."""

    return PlotDescriptor(
        plot=registered_two_factor_plot,
        res_col="score",
        factor_cols=["group.a", "group.b"],
    )


def test_dry_run_stashes_layout(small_pi_fixture, ppd_two_factors):
    """dry_run stashes n_facet_cols; escape_labels=False keeps dotted column names unescaped."""

    pi = pp.create_plot(small_pi_fixture, ppd_two_factors, dry_run=True, escape_labels=False)
    assert pi.n_facet_cols is not None
    joined = "".join(pi.outer_factors)
    assert "." in joined
    assert "․" not in joined


def test_dry_run_escape_labels_true_escapes_outer_factors(small_pi_fixture, ppd_two_factors):
    """Contrast case: escape_labels=True (the default) does escape outer_factors labels."""

    pi = pp.create_plot(small_pi_fixture, ppd_two_factors, dry_run=True, escape_labels=True)
    joined = "".join(pi.outer_factors)
    assert "․" in joined
    assert "." not in joined


@pytest.fixture
def ppd_columns():
    """A PlotDescriptor targeting the real `columns` plot."""

    return PlotDescriptor(plot="columns", res_col="score", factor_cols=["group.a", "group.b"])


def test_payload_shape_columns(small_pi_fixture, ppd_columns):
    """PlotPayload v1 shape: version/plot tag, column-wise cell `data`, plain-hex-or-None facet colors."""

    pl = pp.create_plot_payload(small_pi_fixture, ppd_columns)
    assert pl["payload_version"] == 1 and pl["plot"] == "columns"
    cell = pl["cells"][0][0]
    assert set(cell["data"]) == set(cell["columns"])
    n = len(cell["data"][pl["value_col"]])
    assert all(len(v) == n for v in cell["data"].values())
    for f in pl["facets"]:
        assert f["colors"] is None or all(isinstance(c, str) for c in f["colors"].values())


def test_payload_uncolored_facet_colors_stay_none(barbell_cell_pi_and_ppd):
    """A facet with no metadata colors carries None (renderer applies its own default,
    matching the spec path — dms#40 parity finding); explicit colors stay."""

    pi, ppd = barbell_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    question = next(f for f in pl["facets"] if f["col"] == "question")
    assert question["colors"] is None

    group = next(f for f in pl["facets"] if f["col"] == "group")
    assert group["colors"] == {"Group A": "#c00000", "Group B": "#00c000"}


def test_payload_fallback_covers_non_payload_plot(small_pi_fixture):
    """A plot without `payload=True` is still covered: its frame is read off the built chart."""

    marker = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

    @pp.stk_plot("__test_fallback_plot")  # note: no payload=True
    def _fn(p):
        return alt.Chart(marker).mark_point().encode(x="x:Q", y="y:Q")

    try:
        assert pp.get_plot_meta("__test_fallback_plot").payload is False
        ppd = PlotDescriptor(plot="__test_fallback_plot", res_col="score", factor_cols=["group.a", "group.b"])
        pl = pp.create_plot_payload(small_pi_fixture, ppd)
        # every cell carries the frame pulled off the chart's `.data`
        for row in pl["cells"]:
            for cell in row:
                assert cell["data"]["x"] == [1, 2, 3]
                assert cell["data"]["y"] == [4.0, 5.0, 6.0]
    finally:
        pp._stk_deregister("__test_fallback_plot")


def test_payload_no_frame_raises():
    """A plot yielding no chart/frame (e.g. a streamlit-only widget) raises `UnsupportedPayloadError`."""

    @pp.stk_plot("__test_no_frame_plot")
    def _fn(p):
        return None  # like a streamlit widget: nothing to extract

    try:
        ppd = PlotDescriptor(plot="__test_no_frame_plot", res_col="score")
        pi = PlotInput(data=pd.DataFrame({"score": [1.0, 2.0]}), col_meta={}, value_col="score")
        with pytest.raises(stk_payload.UnsupportedPayloadError):
            pp.create_plot_payload(pi, ppd)
    finally:
        pp._stk_deregister("__test_no_frame_plot")
    # Same class importable from pp for backwards compatibility
    assert pp.UnsupportedPayloadError is stk_payload.UnsupportedPayloadError


def test_payload_matches_altair_frame(small_pi_fixture, ppd_columns):
    """Consistency contract: payload rows == the frame the Altair encoder receives."""

    pi = pp.create_plot(small_pi_fixture, ppd_columns, dry_run=True, escape_labels=False)
    pl = pp.create_plot_payload(small_pi_fixture, ppd_columns)
    flat = [v for row in pl["cells"] for c in row for v in c["data"][pl["value_col"]]]
    assert sorted(flat) == sorted(pi.data[pi.value_col].tolist())


@pytest.fixture
def registered_mutating_two_factor_plot():
    """Register a plot that appends to `p.facets` in place and records what each call saw on
    entry, so a test can detect cross-cell leakage (the payload deepcopies facets per cell)."""

    seen_n_facets: list[int] = []

    @pp.stk_plot("__test_mutating_plot", payload=True)
    def _plot_fn(p):
        seen_n_facets.append(len(p.facets))
        p.facets.append(FacetMeta(col=f"mutant-{len(seen_n_facets)}", ocol=f"mutant-{len(seen_n_facets)}"))
        return p

    try:
        yield "__test_mutating_plot", seen_n_facets
    finally:
        pp._stk_deregister("__test_mutating_plot")


@pytest.fixture
def ppd_mutating_two_factors(registered_mutating_two_factor_plot):
    """Two-factor descriptor with one column inner, so the other stays outer -> multiple cells."""

    plot_name, seen_n_facets = registered_mutating_two_factor_plot
    ppd = PlotDescriptor(
        plot=plot_name,
        res_col="score",
        factor_cols=["group.a", "group.b"],
        internal_facet=1,
    )
    return ppd, seen_n_facets


def test_create_plot_payload_no_cross_cell_aliasing(small_pi_fixture, ppd_mutating_two_factors):
    """Each cell gets a fresh (deep-copied) facets container; a cell's in-place mutation does not
    leak into the next cell, and the top-level facets block reflects the untouched dry run."""

    ppd, seen_n_facets = ppd_mutating_two_factors

    dry_pi = pp.create_plot(small_pi_fixture, ppd, dry_run=True, escape_labels=False)
    assert len(dry_pi.facets) == 1
    dry_facet_cols = [f.col for f in dry_pi.facets]

    pl = pp.create_plot_payload(small_pi_fixture, ppd)

    assert len(seen_n_facets) == 3  # 3 outer categories -> 3 cells
    assert seen_n_facets == [1, 1, 1], f"cross-cell facets leakage: {seen_n_facets}"
    assert [f["col"] for f in pl["facets"]] == dry_facet_cols


# --------------------------------------------------------
#   Per-plot return_df / payload tests
# --------------------------------------------------------


def test_columns_return_df_matches_chart_frame():
    """`columns` needs no reshaping; return_df frame == chart frame."""

    rng = np.random.default_rng(1)
    n = 6
    data = pd.DataFrame(
        {
            "score": rng.uniform(0, 1, n),
            "group.a": pd.Categorical(["A1", "A2"] * (n // 2), categories=["A1", "A2"]),
        }
    )
    facets = [FacetMeta(col="group.a", ocol="group.a", order=["A1", "A2"])]
    pi = PlotInput(data=data, col_meta={}, value_col="score", facets=facets, tooltip=[])

    df = prepared(stk_plots.columns, pi).data
    assert df.equals(pi.data)

    chart = stk_plots.columns(pi)
    assert chart.data.equals(df)


@pytest.fixture
def likert_cell_pi_and_ppd():
    """Single-cell input for `likert_bars`: 5-point ordered scale with a middle neutral."""

    cats = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    data = pd.DataFrame(
        {
            "agree": pd.Categorical(cats, categories=cats, ordered=True),
            "share": [0.1, 0.2, 0.2, 0.3, 0.2],
        }
    )
    col_meta = {"agree": GroupOrColumnMeta(categories=cats, ordered=True, likert=True, neutral_middle="Neutral")}
    pi = PlotInput(data=data, col_meta=col_meta, value_col="share")
    ppd = PlotDescriptor(plot="likert_bars", res_col="share", factor_cols=["agree"], internal_facet=True)
    return pi, ppd


def test_likert_bars_return_df_matches_chart_frame(likert_cell_pi_and_ppd):
    """`likert_bars`' return_df frame is what the encoder draws, incl. `start`/`end` columns."""

    pi, ppd = likert_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1

    df = prepared(stk_plots.likert_bars, cell_pi).data
    assert {"start", "end"}.issubset(df.columns)

    chart = stk_plots.likert_bars(cell_pi)
    assert chart.data.equals(df)


def test_payload_likert_bars_smoke(likert_cell_pi_and_ppd):
    """`start`/`end` columns present; facet block carries the neutral category."""

    pi, ppd = likert_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert {"start", "end"}.issubset(cell["columns"])
    assert len(pl["facets"][0]["neutrals"]) > 0


def test_make_start_end_empty_group_returns_empty():
    """Empty phantom groups (per-cell payload filters an outer facet) return empty, not IndexError."""

    cats = ["a", "b", "c"]
    empty = pd.DataFrame({"cat": pd.Categorical([], categories=cats, ordered=True), "val": []})
    out = stk_plots.make_start_end(empty, value_col="val", cat_col="cat", cat_order=cats, neutral=[1], n_negative=1)
    assert len(out) == 0


def test_payload_likert_bars_faceted_no_crash():
    """likert_bars with an OUTER facet: per-cell filtering used to crash make_start_end (IndexError)."""

    cats = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    shares = [0.1, 0.2, 0.2, 0.3, 0.2]
    rows = [{"gender": g, "agree": c, "share": s} for g in ["Male", "Female"] for c, s in zip(cats, shares)]
    data = pd.DataFrame(rows)
    data["agree"] = pd.Categorical(data["agree"], categories=cats, ordered=True)
    data["gender"] = pd.Categorical(data["gender"], categories=["Male", "Female"])
    col_meta = {
        "agree": GroupOrColumnMeta(categories=cats, ordered=True, likert=True, neutral_middle="Neutral"),
        "gender": GroupOrColumnMeta(categories=["Male", "Female"]),
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="share")
    ppd = PlotDescriptor(plot="likert_bars", res_col="share", factor_cols=["agree", "gender"])

    pl = pp.create_plot_payload(pi, ppd)

    assert pl["outer_factors"] == ["gender"]
    assert sum(len(row) for row in pl["cells"]) == 2  # one cell per gender, neither crashes
    for row in pl["cells"]:
        for cell in row:
            assert {"start", "end"}.issubset(cell["columns"])
            # each cell carries only its own gender's rows
            assert len(set(cell["data"]["gender"])) == 1


@pytest.fixture
def matrix_cell_pi_and_ppd():
    """Single-cell input for `matrix`: 3x3 grid spanning both signs (diverging scale branch)."""

    rows, cols = ["R1", "R2", "R3"], ["C1", "C2", "C3"]
    rng = np.random.default_rng(2)
    combos = [(r, c) for r in rows for c in cols]
    data = pd.DataFrame(
        {
            "row": pd.Categorical([r for r, _ in combos], categories=rows),
            "col": pd.Categorical([c for _, c in combos], categories=cols),
            "val": rng.uniform(-1, 1, len(combos)),
        }
    )
    pi = PlotInput(data=data, col_meta={}, value_col="val")
    ppd = PlotDescriptor(plot="matrix", res_col="val", factor_cols=["row", "col"], internal_facet=True)
    return pi, ppd


def test_matrix_return_df_matches_chart_frame(matrix_cell_pi_and_ppd):
    """`matrix`'s return_df frame is exactly what the Altair encoder draws from."""

    pi, ppd = matrix_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 2

    df = prepared(stk_plots.matrix, cell_pi).data
    chart = stk_plots.matrix(cell_pi)
    assert chart.data.equals(df)


def test_payload_matrix_smoke(matrix_cell_pi_and_ppd):
    """`scale.stops` is a non-empty hex list and `domain` has 3 numbers."""

    pi, ppd = matrix_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    assert pl["scale"] is not None
    assert len(pl["scale"]["stops"]) > 0
    assert len(pl["scale"]["domain"]) == 3


@pytest.fixture
def boxplot_cell_pi_and_ppd():
    """Single-cell input for `boxplots`: one coloured facet, several values per category."""

    rng = np.random.default_rng(3)
    cats = ["Group A", "Group B", "Group C"]
    n_per = 8
    data = pd.DataFrame(
        {
            "group": pd.Categorical(np.repeat(cats, n_per), categories=cats),
            "score": rng.uniform(0, 1, len(cats) * n_per),
        }
    )
    col_meta = {
        "group": GroupOrColumnMeta(
            categories=cats,
            colors={"Group A": "#c00000", "Group B": "#00c000", "Group C": "#0000c0"},
        )
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="score")
    ppd = PlotDescriptor(plot="boxplots", res_col="score", factor_cols=["group"], internal_facet=True)
    return pi, ppd


def test_boxplots_return_df_matches_chart_frame(boxplot_cell_pi_and_ppd):
    """`boxplots`' return_df frame (Tukey stats) is what the LayerChart draws from."""

    pi, ppd = boxplot_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1

    df = prepared(stk_plots.boxplot_manual, cell_pi).data
    assert {"tmin", "q1p", "q3p", "q3", "tmax", "mean"}.issubset(df.columns)

    chart = stk_plots.boxplot_manual(cell_pi)
    assert chart.data.equals(df)


def test_payload_boxplots_smoke(boxplot_cell_pi_and_ppd):
    """Cell data carries the Tukey stat columns; facet colors arrive as plain hex."""

    pi, ppd = boxplot_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert {"tmin", "q1p", "q3p", "q3", "tmax", "mean"}.issubset(cell["columns"])
    assert pl["facets"][0]["colors"] is not None
    assert all(isinstance(c, str) for c in pl["facets"][0]["colors"].values())


_MARIMEKKO_GEOMETRY_COLS = {"xv", "yv", "x1", "x2", "y1", "y2", "xmid", "tprop", "text", "text2"}


@pytest.fixture
def marimekko_cell_pi_and_ppd():
    """Single-cell input for `marimekko`: 3x2 grid, nonnegative values, explicit `group_size`."""

    rows, cols = ["R1", "R2", "R3"], ["C1", "C2"]
    rng = np.random.default_rng(4)
    combos = [(r, c) for r in rows for c in cols]
    data = pd.DataFrame(
        {
            "row": pd.Categorical([r for r, _ in combos], categories=rows),
            "col": pd.Categorical([c for _, c in combos], categories=cols),
            "val": rng.uniform(0.1, 1, len(combos)),
            "group_size": rng.integers(10, 100, len(combos)),
        }
    )
    pi = PlotInput(data=data, col_meta={}, value_col="val")
    ppd = PlotDescriptor(plot="marimekko", res_col="val", factor_cols=["row", "col"], internal_facet=True)
    return pi, ppd


def test_marimekko_return_df_matches_chart_frame(marimekko_cell_pi_and_ppd):
    """`marimekko`'s return_df frame (cell-fill + cumsum geometry) is what the chart draws from."""

    pi, ppd = marimekko_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 2

    df = prepared(stk_plots.marimekko, cell_pi).data
    assert _MARIMEKKO_GEOMETRY_COLS.issubset(df.columns)

    chart = stk_plots.marimekko(cell_pi)
    assert chart.data.equals(df)


def test_payload_marimekko_smoke(marimekko_cell_pi_and_ppd):
    """Cell data carries the marimekko geometry columns."""

    pi, ppd = marimekko_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert _MARIMEKKO_GEOMETRY_COLS.issubset(set(cell["columns"]))


@pytest.fixture
def line_cell_pi_and_ppd():
    """Single-cell input for `line`: numeric-string x categories (quantitative-x branch)."""

    cats = ["1", "2", "3", "4"]
    data = pd.DataFrame(
        {
            "level": pd.Categorical(cats, categories=cats, ordered=True),
            "share": [0.1, 0.3, 0.4, 0.2],
        }
    )
    col_meta = {"level": GroupOrColumnMeta(categories=cats, ordered=True)}
    pi = PlotInput(data=data, col_meta=col_meta, value_col="share")
    ppd = PlotDescriptor(plot="line", res_col="share", factor_cols=["level"], internal_facet=True)
    return pi, ppd


@pytest.fixture
def lines_cell_pi_and_ppd():
    """Single-cell input for `lines`: colour facet + non-numeric x facet (nominal-x branch)."""

    groups, stages = ["G1", "G2"], ["Early", "Mid", "Late"]
    combos = [(g, s) for g in groups for s in stages]
    data = pd.DataFrame(
        {
            "group": pd.Categorical([g for g, _ in combos], categories=groups),
            "stage": pd.Categorical([s for _, s in combos], categories=stages, ordered=True),
            "share": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    col_meta = {
        "group": GroupOrColumnMeta(categories=groups),
        "stage": GroupOrColumnMeta(categories=stages, ordered=True),
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="share")
    ppd = PlotDescriptor(plot="lines", res_col="share", factor_cols=["group", "stage"], internal_facet=True)
    return pi, ppd


def test_line_return_df_matches_chart_frame(line_cell_pi_and_ppd):
    """Numeric-string x categories produce the parsed `<x>_cont` column; frames agree."""

    pi, ppd = line_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1

    df = prepared(stk_plots.lines, cell_pi).data
    assert "level_cont" in df.columns

    chart = stk_plots.lines(cell_pi)
    assert chart.data.equals(df)


def test_lines_return_df_matches_chart_frame(lines_cell_pi_and_ppd):
    """Nominal-x branch: no `_cont` column; frames agree."""

    pi, ppd = lines_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 2

    df = prepared(stk_plots.lines, cell_pi).data
    assert "stage_cont" not in df.columns

    chart = stk_plots.lines(cell_pi)
    assert chart.data.equals(df)


def test_payload_line_smoke(line_cell_pi_and_ppd):
    """Cell data carries the value column and the parsed continuous x column."""

    pi, ppd = line_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert pl["value_col"] in cell["columns"]
    assert "level_cont" in cell["columns"]


@pytest.fixture
def density_cell_pi_and_ppd():
    """Single-cell input for `density-raw`: two well-separated normal samples."""

    rng = np.random.default_rng(5)
    cats = ["G1", "G2"]
    n_per = 40
    data = pd.DataFrame(
        {
            "group": pd.Categorical(np.repeat(cats, n_per), categories=cats),
            "score": np.r_[rng.normal(0.0, 1.0, n_per), rng.normal(4.0, 1.0, n_per)],
        }
    )
    col_meta = {"group": GroupOrColumnMeta(categories=cats)}
    pi = PlotInput(data=data, col_meta=col_meta, value_col="score")
    ppd = PlotDescriptor(plot="density-raw", res_col="score", factor_cols=["group"], internal_facet=True)
    return pi, ppd


def test_density_return_df_matches_chart_frame(density_cell_pi_and_ppd):
    """`density`'s return_df KDE frame is exactly what the Altair encoder draws from."""

    pi, ppd = density_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1

    df = prepared(stk_plots.density, cell_pi).data
    assert "density" in df.columns
    assert len(df) == 2 * 101  # one 101-point KDE grid per group

    chart = stk_plots.density(cell_pi)
    assert chart.data.equals(df)


def test_density_return_df_matches_chart_frame_stacked(density_cell_pi_and_ppd):
    """Stacked mode: same kwargs produce the same frame (incl. the `order` column) on both paths."""

    pi, ppd = density_cell_pi_and_ppd
    ppd = ppd.model_copy(update={"plot_args": {"stacked": True}})
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)

    df = prepared(stk_plots.density, cell_pi, stacked=True).data
    assert "order" in df.columns

    chart = stk_plots.density(cell_pi, stacked=True)
    assert chart.data.equals(df)


def test_payload_density_smoke(density_cell_pi_and_ppd):
    """KDE columns (`density` + the value col) present in the cell data."""

    pi, ppd = density_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert "density" in cell["columns"]
    assert pl["value_col"] in cell["columns"]


@pytest.fixture
def density_faceted_pi_and_ppd(density_cell_pi_and_ppd):
    """The same density data with the group OUTER, so the grid has one cell per group."""

    pi, ppd = density_cell_pi_and_ppd
    return pi, ppd.model_copy(update={"internal_facet": False})


def test_payload_density_faceted_per_cell(density_faceted_pi_and_ppd):
    """Regression: each faceted cell must only contain its own facet category (observed=True)."""

    pi, ppd = density_faceted_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cells = [c for row in pl["cells"] for c in row]
    assert len(cells) == 2
    for cell in cells:
        groups = set(cell["data"]["group"])
        assert groups == {cell["keys"]["group"]}, f"cell {cell['keys']} leaked other groups: {groups}"


def test_altair_matrix_density_faceted_per_cell(density_faceted_pi_and_ppd):
    """Same regression on the Altair path with `return_matrix_of_plots=True`."""

    pi, ppd = density_faceted_pi_and_ppd
    charts = pp.create_plot(pi, ppd, width=400, return_matrix_of_plots=True)
    assert isinstance(charts, list)

    flat = [c for row in charts for c in row]
    assert len(flat) == 2
    for chart in flat:
        title = chart.to_dict()["title"]
        groups = set(chart.data["group"].astype(str))
        assert groups == {title}, f"cell {title!r} leaked other groups: {groups}"


def test_density_categorical_input_typed_error():
    """A categorical value column raises a clear ValueError (422 upstream), not a pandas error."""

    cats = ["Low", "Mid", "High"]
    data = pd.DataFrame({"answer": pd.Categorical(cats * 4, categories=cats, ordered=True)})
    pi = PlotInput(data=data, col_meta={}, value_col="answer")
    ppd = PlotDescriptor(plot="density-raw", res_col="answer", factor_cols=[])

    with pytest.raises(ValueError, match="continuous"):
        pp.create_plot(pi, ppd, width=400)
    with pytest.raises(ValueError, match="continuous"):
        pp.create_plot_payload(pi, ppd)


@pytest.fixture
def geoplot_cell_pi_and_ppd():
    """Single-cell input for `geoplot`: facet with `topo_feature` metadata, values spanning both signs."""

    regions = ["Harju", "Tartu", "Parnu"]
    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {
            "region": pd.Categorical(regions, categories=regions),
            "score": rng.uniform(-1, 1, len(regions)),
        }
    )
    col_meta = {
        "region": GroupOrColumnMeta(
            categories=regions,
            topo_feature=("https://example.com/estonia.topojson", "counties", "MNIMI"),
        )
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="score")
    ppd = PlotDescriptor(plot="geoplot", res_col="score", factor_cols=["region"], internal_facet=True)
    return pi, ppd


def test_geoplot_return_df_matches_chart_lookup_frame(geoplot_cell_pi_and_ppd):
    """`geoplot`'s return_df frame is the survey-side table the chart's `transform_lookup` uses."""

    pi, ppd = geoplot_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1
    topo = cell_pi.plot_args["topo_feature"]
    assert topo == ("https://example.com/estonia.topojson", "counties", "MNIMI")

    df = prepared(stk_plots.geoplot, cell_pi, topo_feature=topo).data

    chart = stk_plots.geoplot(cell_pi, topo_feature=topo)
    lookup_data = chart.transform[0]["from"].data
    assert lookup_data.equals(df)


def test_geoplot_geojson_format_omits_object(geoplot_cell_pi_and_ppd):
    """`format: geojson` must not carry an `object` key (only defined for topojson sources)."""

    pi, ppd = geoplot_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)

    chart = stk_plots.geoplot(cell_pi, topo_feature=("https://example.com/estonia.geojson", "geojson", "NAME"))
    assert stk_payload._chart_geo(chart) == {
        "url": "https://example.com/estonia.geojson",
        "format": "geojson",
        "region_key": "region",
        "name_property": "NAME",
    }


def test_payload_geoplot_smoke(geoplot_cell_pi_and_ppd):
    """`payload["geo"]` carries the wire-shape keys; `payload["scale"]` has stops + 2-number domain."""

    pi, ppd = geoplot_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    assert pl["geo"] == {
        "url": "https://example.com/estonia.topojson",
        "format": "topojson",
        "object": "counties",
        "region_key": "region",
        "name_property": "MNIMI",
    }
    assert pl["scale"] is not None
    assert len(pl["scale"]["stops"]) > 0
    assert len(pl["scale"]["domain"]) == 2


@pytest.fixture
def geobest_cell_pi_and_ppd():
    """Single-cell input for `geobest`: colored winner facet + region facet with `topo_feature`."""

    regions = ["Harju", "Tartu", "Parnu"]
    candidates = ["Alice", "Bob"]
    rng = np.random.default_rng(7)
    rows = [(c, r) for r in regions for c in candidates]
    data = pd.DataFrame(
        {
            "candidate": pd.Categorical([c for c, r in rows], categories=candidates),
            "region": pd.Categorical([r for c, r in rows], categories=regions),
            "score": rng.uniform(0, 1, len(rows)),
        }
    )
    col_meta = {
        "candidate": GroupOrColumnMeta(categories=candidates, colors={"Alice": "#FF0000", "Bob": "#0000FF"}),
        "region": GroupOrColumnMeta(
            categories=regions,
            topo_feature=("https://example.com/estonia.topojson", "counties", "MNIMI"),
        ),
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="score")
    ppd = PlotDescriptor(plot="geobest", res_col="score", factor_cols=["candidate", "region"], internal_facet=True)
    return pi, ppd


def test_geobest_return_df_matches_chart_lookup_frame(geobest_cell_pi_and_ppd):
    """`geobest`'s return_df frame (winner per region) is the chart's `transform_lookup` table."""

    pi, ppd = geobest_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 2
    topo = cell_pi.plot_args["topo_feature"]
    assert topo == ("https://example.com/estonia.topojson", "counties", "MNIMI")

    df = prepared(stk_plots.geobest, cell_pi, topo_feature=topo).data
    assert len(df) == 3  # winner-selection: one row per region
    assert set(df["region"]) == {"Harju", "Tartu", "Parnu"}

    chart = stk_plots.geobest(cell_pi, topo_feature=topo)
    lookup_data = chart.transform[0]["from"].data
    assert lookup_data.equals(df)


def test_geobest_geojson_format_omits_object(geobest_cell_pi_and_ppd):
    """`format: geojson` must not carry an `object` key (shared `_chart_geo` contract)."""

    pi, ppd = geobest_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)

    chart = stk_plots.geobest(cell_pi, topo_feature=("https://example.com/estonia.geojson", "geojson", "NAME"))
    assert stk_payload._chart_geo(chart) == {
        "url": "https://example.com/estonia.geojson",
        "format": "geojson",
        "region_key": "region",
        "name_property": "NAME",
    }


def test_payload_geobest_smoke(geobest_cell_pi_and_ppd):
    """`geo` keyed off the region facet; `scale` None (categorical map); winner colors on facets[0]."""

    pi, ppd = geobest_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    assert pl["geo"] == {
        "url": "https://example.com/estonia.topojson",
        "format": "topojson",
        "object": "counties",
        "region_key": "region",
        "name_property": "MNIMI",
    }
    assert pl["scale"] is None

    assert pl["facets"][0]["col"] == "candidate"
    assert pl["facets"][0]["colors"] == {"Alice": "#FF0000", "Bob": "#0000FF"}
    assert pl["facets"][1]["col"] == "region"

    cell = pl["cells"][0][0]
    assert len(cell["data"]["region"]) == 3
    assert set(cell["data"]["region"]) == {"Harju", "Tartu", "Parnu"}


def test_payload_geobest_null_colors_smoke(geobest_cell_pi_and_ppd):
    """An uncolored winner facet stays None; renderers own the fallback (the dms
    frontend covers this exact case — its plotDataGeobestNullColors fixture)."""

    pi, ppd = geobest_cell_pi_and_ppd
    pi.col_meta["candidate"] = GroupOrColumnMeta(categories=["Alice", "Bob"])

    pl = pp.create_plot_payload(pi, ppd)
    assert pl["facets"][0]["col"] == "candidate"
    assert pl["facets"][0]["colors"] is None


@pytest.fixture
def barbell_cell_pi_and_ppd():
    """Single-cell input for `barbell`: nominal `question` facet + colored `group` facet."""

    questions = ["Q1", "Q2", "Q3"]
    groups = ["Group A", "Group B"]
    rng = np.random.default_rng(9)
    rows = [(q, g) for q in questions for g in groups]
    data = pd.DataFrame(
        {
            "question": pd.Categorical([q for q, _ in rows], categories=questions),
            "group": pd.Categorical([g for _, g in rows], categories=groups),
            "score": rng.uniform(0, 1, len(rows)),
        }
    )
    col_meta = {
        "group": GroupOrColumnMeta(categories=groups, colors={"Group A": "#c00000", "Group B": "#00c000"}),
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="score")
    ppd = PlotDescriptor(plot="barbell", res_col="score", factor_cols=["question", "group"], internal_facet=True)
    return pi, ppd


def test_barbell_return_df_matches_chart_frame(barbell_cell_pi_and_ppd):
    """`barbell` needs no reshaping; return_df frame == the LayerChart's frame."""

    pi, ppd = barbell_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 2
    assert cell_pi.cat_col is None

    df = prepared(stk_plots.barbell, cell_pi).data
    assert df.equals(cell_pi.data)

    chart = stk_plots.barbell(cell_pi)
    assert chart.data.equals(df)


def test_payload_barbell_smoke(barbell_cell_pi_and_ppd):
    """Cell data carries the plain survey columns; facets/colors/order come from real metadata."""

    pi, ppd = barbell_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    assert pl["cat_col"] is None
    assert pl["facets"][0]["col"] == "question"
    assert pl["facets"][0]["order"] == ["Q1", "Q2", "Q3"]
    assert pl["facets"][1]["col"] == "group"
    assert pl["facets"][1]["order"] == ["Group A", "Group B"]
    assert pl["facets"][1]["colors"] == {"Group A": "#c00000", "Group B": "#00c000"}

    cell = pl["cells"][0][0]
    assert set(cell["columns"]) == {"question", "group", "score"}
    assert len(cell["data"]["score"]) == len(pi.data)


@pytest.fixture
def maxdiff_cell_pi_and_ppd():
    """Single-cell input for `maxdiff`: `topic` facet plus a `reverse_score` companion column."""

    topics = ["Economy", "Health", "Education"]
    rng = np.random.default_rng(11)
    n_per_topic = 4
    data = pd.DataFrame(
        {
            "topic": pd.Categorical(np.repeat(topics, n_per_topic), categories=topics),
            "score": rng.uniform(0, 1, len(topics) * n_per_topic),
            "reverse_score": rng.uniform(0, 1, len(topics) * n_per_topic),
        }
    )
    pi = PlotInput(data=data, col_meta={}, value_col="score")
    ppd = PlotDescriptor(plot="maxdiff", res_col="score", factor_cols=["topic"], internal_facet=True)
    return pi, ppd


def test_maxdiff_return_df_matches_chart_frame(maxdiff_cell_pi_and_ppd):
    """`maxdiff`'s return_df frame (`kind`-tagged Most/Least stack) is what the encoder charts."""

    pi, ppd = maxdiff_cell_pi_and_ppd
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)
    assert len(cell_pi.facets) == 1

    df = prepared(stk_plots.maxdiff_manual, cell_pi).data
    assert set(df["kind"].unique()) == {"Most important", "Least important"}

    chart = stk_plots.maxdiff_manual(cell_pi)
    assert chart.data.equals(df)


def test_payload_maxdiff_smoke(maxdiff_cell_pi_and_ppd):
    """Per-kind sign/stack columns present; `kind` distinguishes Most/Least important rows."""

    pi, ppd = maxdiff_cell_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    assert pl["facets"][0]["col"] == "topic"
    cell = pl["cells"][0][0]
    assert {"kind", "mean", "topic"} <= set(cell["columns"])
    assert set(cell["data"]["kind"]) == {"Most important", "Least important"}


def test_maxdiff_missing_reverse_col_raises_typed_error(maxdiff_cell_pi_and_ppd):
    """No `reverse_<col>` companion -> a clear ValueError, not a bare StopIteration."""

    pi, ppd = maxdiff_cell_pi_and_ppd
    pi.data = pi.data.drop(columns=["reverse_score"])

    with pytest.raises(ValueError, match="maxdiff scores"):
        pp.create_plot(pi, ppd, width=400)
    with pytest.raises(ValueError, match="maxdiff scores"):
        pp.create_plot_payload(pi, ppd)


def test_matching_plots_maxdiff_requires_continuous_res_col():
    """maxdiff's `continuous=True` registration excludes categorical value columns in matching_plots."""

    data_meta = soft_validate(
        {
            "structure": [
                {"name": "topic", "columns": [["topic", {"categories": ["A", "B"]}]]},
                {"name": "party", "columns": [["party", {"categories": ["X", "Y"]}]]},
                {"name": "score", "columns": [["score", {"continuous": True}]]},
            ],
            "files": [{"file": "__test__", "opts": {}, "code": "F0"}],
        },
        DataMeta,
    )
    df = pd.DataFrame(
        {
            "topic": pd.Categorical(["A", "B"], categories=["A", "B"]),
            "party": pd.Categorical(["X", "Y"], categories=["X", "Y"]),
            "score": [0.2, 0.8],
            "draw": [0, 0],  # maxdiff declares draws=True -- must be present to avoid an unrelated penalty
        }
    )

    categorical_matches = matching_plots(
        {"res_col": "party", "factor_cols": ["topic"], "plot": "maxdiff"}, df, data_meta, details=True
    )
    priority, reasons = categorical_matches["maxdiff"]
    assert priority < 0
    assert "continuous_only" in reasons

    continuous_matches = matching_plots(
        {"res_col": "score", "factor_cols": ["topic"], "plot": "maxdiff"}, df, data_meta, details=True
    )
    priority2, reasons2 = continuous_matches["maxdiff"]
    assert priority2 >= 0
    assert reasons2 == []


# --------------------------------------------------------
#   Election plots: mandate_plot / party_mandates / coalition_applet
# --------------------------------------------------------


@pytest.fixture
def election_pi_and_ppd():
    """Draws-level party support per district, with mandates/electoral_system on the district meta."""

    parties, districts = ["Alpha", "Beta"], ["North", "South"]
    rng = np.random.default_rng(13)
    rows = [(d, dist, p) for d in range(8) for dist in districts for p in parties]
    data = pd.DataFrame(
        {
            "draw": [d for d, _, _ in rows],
            "party": pd.Categorical([p for _, _, p in rows], categories=parties),
            "district": pd.Categorical([dist for _, dist, _ in rows], categories=districts),
            "support": rng.uniform(100, 1000, len(rows)),
        }
    )
    col_meta = {
        "party": GroupOrColumnMeta(categories=parties, colors={"Alpha": "#c00000", "Beta": "#0000c0"}),
        "district": GroupOrColumnMeta(
            categories=districts,
            mandates={"North": 3, "South": 4},
            # Non-default value: an all-default ElectoralSystem() serializes away entirely
            electoral_system=ElectoralSystem(threshold=0.05),
        ),
    }
    pi = PlotInput(data=data, col_meta=col_meta, value_col="support")
    ppd = PlotDescriptor(plot="mandate_plot", res_col="support", factor_cols=["party", "district"], internal_facet=True)
    return pi, ppd


def test_payload_mandate_plot_smoke(election_pi_and_ppd):
    """`mandate_plot` payload carries per-district mandate probability rows from the simulation."""

    pi, ppd = election_pi_and_ppd
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert {"party", "district", "mandates", "percent"} <= set(cell["columns"])
    assert all(0.0 <= v <= 1.0 for v in cell["data"]["percent"])
    # Compensation row synthesized by the Estonian-system simulation
    assert "Compensation" in set(cell["data"]["district"])


def test_payload_party_mandates_smoke(election_pi_and_ppd):
    """`party_mandates` payload carries per-party mandate distributions (percent/median/over_threshold)."""

    pi, ppd = election_pi_and_ppd
    ppd = ppd.model_copy(update={"plot": "party_mandates"})
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert {"party", "mandates", "percent", "median", "over_threshold"} <= set(cell["columns"])
    assert all(0.0 <= v <= 1.0 for v in cell["data"]["percent"])


def test_party_mandates_return_df_matches_chart_frame(election_pi_and_ppd):
    """`party_mandates`' return_df frame is exactly what its static Altair chart draws from."""

    pi, ppd = election_pi_and_ppd
    ppd = ppd.model_copy(update={"plot": "party_mandates"})
    cell_pi = pp.create_plot(pi, ppd, dry_run=True, escape_labels=False)
    assert isinstance(cell_pi, PlotInput)

    kwargs = dict(
        mandates=cell_pi.plot_args["mandates"],
        electoral_system=cell_pi.plot_args["electoral_system"],
        sim_done=False,
    )
    df = prepared(stk_elections.party_mandates, cell_pi, **kwargs).data
    chart = stk_elections.party_mandates(cell_pi, **kwargs)
    assert chart.data.equals(df)


def test_payload_coalition_applet_smoke(election_pi_and_ppd):
    """`coalition_applet` payload carries per-draw seat totals per party -- no streamlit involved."""

    pi, ppd = election_pi_and_ppd
    ppd = ppd.model_copy(update={"plot": "coalition_applet"})
    pl = pp.create_plot_payload(pi, ppd)

    cell = pl["cells"][0][0]
    assert {"draw", "party", "mandates", "over_t"} <= set(cell["columns"])
    assert len(set(cell["data"]["draw"])) == 8
    # Total seats allocated per draw match the mandates dict (3 + 4)
    per_draw = {}
    for d, m in zip(cell["data"]["draw"], cell["data"]["mandates"]):
        per_draw[d] = per_draw.get(d, 0) + m
    assert all(v == 7 for v in per_draw.values())
