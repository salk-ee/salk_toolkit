"""
Unit tests for plot pipeline utilities in salk_toolkit.pp.
"""

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
import altair as alt

from salk_toolkit.pp import (
    _calculate_priority as calculate_priority,
    e2e_plot,
    _transform_cont,
    FacetMeta,
    PlotInput,
    get_plot_fn,
    impute_facet_dims,
    matching_plots,
    pp_transform_data,
    create_plot,
    PlotMeta,
    registry,
    registry_meta,
    _stk_deregister as stk_deregister,
    stk_plot,
    _update_data_meta_with_pp_desc,
)
from salk_toolkit.pp.common import _question_meta_clone
from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate
from pydantic import ValidationError


def make_data_meta(meta_dict: dict[str, object]) -> DataMeta:
    """Build a DataMeta object from a bare dict for test fixtures."""
    payload = dict(meta_dict)
    if "files" not in payload:
        payload["files"] = [{"file": "__test__", "opts": {}, "code": "F0"}]
    return soft_validate(payload, DataMeta)


@pytest.fixture
def registry_guard():
    """Preserve the global plot registry around tests that register temporary plots."""
    snapshot_registry = registry.copy()
    snapshot_meta = registry_meta.copy()
    try:
        yield
    finally:
        registry.clear()
        registry.update(snapshot_registry)
        registry_meta.clear()
        registry_meta.update(snapshot_meta)


def test_update_data_meta_with_pp_desc_adds_res_meta_and_updates_columns() -> None:
    """`_update_data_meta_with_pp_desc` should add response metadata without mutating input."""
    data_meta = make_data_meta(
        {
            "structure": [
                {
                    "name": "demographics",
                    "scale": {"col_prefix": ""},
                    "columns": [
                        [
                            "gender",
                            {"categories": ["Female", "Male"], "label": "Gender (meta)"},
                        ],
                    ],
                }
            ]
        }
    )
    original_structure = deepcopy(data_meta.structure)

    pp_desc_dict = {
        "plot": "test_plot",
        "res_col": "likert_score",
        "facet_dims": ["gender"],
        "res_meta": {
            "name": "likert_question",
            "scale": {"col_prefix": "likert_"},
            "columns": [
                [
                    "likert_score",
                    {"categories": ["Low", "Medium", "High"], "label": "Likert Score"},
                ],
            ],
        },
        "col_meta": {
            "gender": {"label": "Updated Gender"},
            "likert_score": {"label": "Updated Likert"},
        },
    }
    pp_desc = soft_validate(pp_desc_dict, PlotDescriptor)

    col_meta, group_columns = _update_data_meta_with_pp_desc(data_meta, pp_desc)

    # Ensure that the original metadata was not mutated
    assert data_meta.structure == original_structure

    # Validate metadata for the newly added response block
    assert "likert_question" in col_meta
    assert col_meta["likert_question"].columns is not None
    new_res_col = col_meta["likert_question"].columns[0]
    assert new_res_col.startswith("likert_")
    assert col_meta[new_res_col].categories == ["Low", "Medium", "High"]
    assert col_meta[new_res_col].label == "Likert Score"

    # Existing column metadata should be updated with overrides from pp_desc
    assert col_meta["gender"].label == "Updated Gender"

    # Group columns dictionary should include the newly added result group
    assert group_columns["likert_question"] == [new_res_col]


def test_calculate_priority_penalizes_missing_requirements() -> None:
    """`calculate_priority` should penalize matches that miss required properties."""
    plot_meta = PlotMeta.model_validate(
        {
            "name": "test_plot",
            "priority": 5,
            "draws": True,
            "requires": [
                {"ordered": True},
            ],
        }
    )
    match = {
        "draws": False,
        "nonnegative": True,
        "hidden": False,
        "res_col": "res",
        "categorical": True,
        "facet_metas": [
            {"name": "gender", "ordered": False},
        ],
    }

    priority, reasons = calculate_priority(plot_meta, match)

    assert priority < 0
    assert "draws" in reasons
    assert "ordered" in reasons


def _make_basic_dataframe():
    return pd.DataFrame(
        {
            "res": pd.Categorical(["low", "high"], categories=["low", "high"]),
            "facet": pd.Categorical(["A", "B"], categories=["A", "B"]),
        }
    )


def _make_basic_meta() -> DataMeta:
    return make_data_meta(
        {
            "structure": [
                {
                    "name": "res",
                    "columns": [
                        ["res", {"categories": ["low", "high"], "label": "Response"}],
                    ],
                },
                {
                    "name": "facet",
                    "columns": [
                        ["facet", {"categories": ["A", "B"], "label": "Facet"}],
                    ],
                },
            ]
        }
    )


def test_matching_plots_respects_hidden_flag(registry_guard: Any) -> None:
    """Hidden plots should be omitted unless explicitly requested."""

    @stk_plot("visible_plot", priority=10)
    def _visible_plot(**_):
        return _

    @stk_plot("hidden_plot", priority=1, hidden=True)
    def _hidden_plot(**_):
        return _

    df = _make_basic_dataframe()
    data_meta = _make_basic_meta()
    pp_desc = {
        "res_col": "res",
        "facet_dims": ["facet"],
        "plot": "visible_plot",
    }

    visible_only = matching_plots(pp_desc, df, data_meta)
    assert "visible_plot" in visible_only
    assert "hidden_plot" not in visible_only

    with_hidden = matching_plots(pp_desc, df, data_meta, list_hidden=True)
    assert "visible_plot" in with_hidden
    assert "hidden_plot" in with_hidden
    assert with_hidden.index("visible_plot") < with_hidden.index("hidden_plot")

    stk_deregister("visible_plot")
    stk_deregister("hidden_plot")


def test_impute_facet_dims_handles_categorical_and_continuous_cases() -> None:
    """`impute_facet_dims` should handle categorical and continuous conversions."""
    col_meta = {
        "res_group": GroupOrColumnMeta.model_validate({"columns": ["res_variant"], "categories": ["Yes", "No"]}),
        "res_variant": GroupOrColumnMeta.model_validate({"categories": ["Yes", "No"]}),
        "region": GroupOrColumnMeta(),
        "question": GroupOrColumnMeta(),
    }

    categorical_desc = {
        "plot": "test_plot",
        "res_col": "res_group",
        "facet_dims": ["region"],
    }
    categorical_factors = impute_facet_dims(categorical_desc, col_meta)
    assert categorical_factors == ["res_group", "region", "question"]

    continuous_desc = {
        "plot": "test_plot",
        "res_col": "res_group",
        "facet_dims": ["region"],
        "convert_res": "continuous",
    }
    continuous_factors = impute_facet_dims(continuous_desc, col_meta)
    assert continuous_factors == ["question", "region"]


@pytest.mark.parametrize(
    "in_dtype",
    [pl.Float32, pl.Float64],
)
def test_transform_cont_custom_row_transform_streaming_dtype_match(in_dtype: pl.DataType) -> None:
    """Custom row transforms must declare a schema whose dtype matches the actual batch output.

    softmax-avgrank preserves its input dtype. If the probe always used float64 zeros,
    real Float32 input would produce Float32 batches while the schema declared Float64,
    causing a streaming-engine panic (`values.dtype() == &self.in_dtype`) in
    downstream group_by/agg. This test runs the full streaming path.
    """
    cols = [f"opt{i}" for i in range(4)]
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            **{c: pl.Series(rng.normal(size=200).astype(np.float64), dtype=in_dtype) for c in cols},
            "group": rng.integers(0, 3, size=200),
        }
    )

    lf, _fmt, _rng = _transform_cont(df.lazy(), cols, transform="softmax-avgrank")
    lf = lf.unpivot(index=["group"], on=cols, variable_name="q", value_name="rank")
    lf = lf.group_by(["group", "q"]).agg(pl.col("rank").mean())

    result = lf.collect(engine="streaming")
    assert result.shape == (12, 3)
    assert result["rank"].dtype in (pl.Float32, pl.Float64)


def test_transform_cont_full_streaming_pipeline(tmp_path) -> None:
    """Integration smoke test reproducing the original panic fixed by commit 0359860.

    Pipeline: scan_parquet → with_row_index → map_batches (via _transform_cont with
    ordered-topbot1) → join → unpivot → Enum cast → reverse_/res weight split →
    group_by → collect(streaming). Without the explicit schema on map_batches the
    streaming engine panics with `Option::unwrap() on a None value` in
    polars-arrow/src/array/builder.rs when initialising builders for the multi-
    column aggregation after the OPAQUE_PYTHON boundary.
    """
    cols = ["A", "B", "C", "D"]
    n_rows = 200
    rng = np.random.default_rng(0)
    parquet = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            **{c: pl.Series(rng.normal(size=n_rows).astype(np.float32), dtype=pl.Float32) for c in cols},
            "pop_group_size": pl.Series(rng.integers(1000, 9999, size=n_rows), dtype=pl.UInt32),
        }
    ).write_parquet(parquet)

    lf = pl.scan_parquet(parquet)
    lf = lf.with_columns(pl.col("pop_group_size").cast(pl.Float64).fill_null(1.0))
    lf = lf.with_row_index("id").with_columns(pl.col("id").cast(pl.Int64))
    lf, _fmt, _rng = _transform_cont(lf, cols, transform="ordered-topbot1")

    draws = pl.DataFrame(
        {"id": pl.Series(range(n_rows), dtype=pl.Int64), "draw": pl.Series([0] * n_rows, dtype=pl.Int64)}
    ).lazy()
    lf = lf.join(draws, on="id", how="left")
    lf = lf.unpivot(index=["id", "draw", "pop_group_size"], on=cols, variable_name="question", value_name="score")
    lf = lf.with_columns(pl.col("question").cast(pl.Enum(cols)))
    # Mirror _wrangle_data's weight split for top/bot transforms.
    lf = lf.with_columns(((pl.col("score") == -1) * pl.col("pop_group_size")).alias("reverse_score"))
    lf = lf.with_columns(((pl.col("score") == 1) * pl.col("pop_group_size")).alias("score"))
    lf = lf.group_by(["question", "draw"]).agg(
        [
            pl.col(["score", "pop_group_size"]).sum(),
            pl.col(["reverse_score", "pop_group_size"]).sum().name.prefix("reverse_"),
        ]
    )

    result = lf.collect(engine="streaming")
    assert result.shape == (len(cols), 6)


def test_convert_res_continuous_maps_unmapped_nonresponse_to_null() -> None:
    """Categories beyond num_values (e.g. nonresponse) become null instead of crashing the cast."""
    cats = [
        "Do not agree at all",
        "Tend to disagree",
        "Tend to agree",
        "Agree completely",
        "Don't know",
        "No answer",
    ]
    data_meta = make_data_meta(
        {
            "structure": [
                {
                    "name": "demographics",
                    "scale": {},
                    "columns": [["gender", {"categories": ["Female", "Male"]}]],
                },
                {
                    "name": "values",
                    "scale": {
                        "categories": cats,
                        "ordered": True,
                        "likert": True,
                        "num_values": [-2.0, -1.0, 1.0, 2.0],
                        "nonresponse": ["Don't know", "No answer"],
                    },
                    "columns": [["value_diversity", {"label": "Cultural diversity"}]],
                },
            ]
        }
    )
    answers = (
        ["Do not agree at all"]
        + ["Tend to disagree"] * 2
        + ["Tend to agree"] * 3
        + ["Agree completely"] * 4
        + ["Don't know"] * 2
        + ["No answer"]
    )
    df = pd.DataFrame(
        {
            "draw": [0] * 13,
            "gender": ["Female", "Male"] * 6 + ["Female"],
            "value_diversity": answers,
        }
    )
    ppd = soft_validate(
        {
            "res_col": "value_diversity",
            "factor_cols": ["gender"],
            "convert_res": "continuous",
            "plot": "boxplots",
        },
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), data_meta, ppd)

    # Nonresponse rows drop out of the aggregate: mean over substantive answers only.
    by_gender = dict(zip(pi.data["gender"], pd.to_numeric(pi.data[pi.value_col], errors="raise")))
    assert by_gender["Female"] == pytest.approx((-2 - 1 + 1 + 2 + 2) / 5)
    assert by_gender["Male"] == pytest.approx((-1 + 1 + 1 + 2 + 2) / 5)


def _threshold_meta() -> DataMeta:
    return make_data_meta(
        {
            "structure": [
                {
                    "name": "demographics",
                    "scale": {},
                    "columns": [["gender", {"categories": ["Female", "Male"]}]],
                },
                {
                    "name": "ratings",
                    "scale": {"continuous": True},
                    "columns": [["party_a"], ["party_b"]],
                },
            ]
        }
    )


def _continuous_res_meta(col_meta: dict[str, Any], col: str = "vote_prob") -> DataMeta:
    """A gender + one-response-column data meta, response column meta given by the caller."""
    return make_data_meta(
        {
            "structure": [
                {"name": "demographics", "scale": {}, "columns": [["gender", {"categories": ["Female", "Male"]}]]},
                {"name": "probs", "scale": {}, "columns": [[col, col_meta]]},
            ]
        }
    )


def test_cont_transform_threshold_is_the_share_past_the_cutoff() -> None:
    """``ge:<x>`` turns a continuous rating into "share who rated it at least x"."""
    df = pd.DataFrame(
        {
            "draw": [0] * 6,
            "gender": ["Female"] * 3 + ["Male"] * 3,
            "party_a": [0.0, 1.0, 2.0, -1.0, 0.0, 5.0],
            "party_b": [3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
        }
    )
    ppd = soft_validate(
        {"res_col": "party_a", "factor_cols": ["gender"], "cont_transform": "ge:1", "plot": "columns"},
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)
    shares = dict(zip(pi.data["gender"], pi.data[pi.value_col]))
    assert shares["Female"] == pytest.approx(2 / 3)  # 1.0 and 2.0 clear the cutoff
    assert shares["Male"] == pytest.approx(1 / 3)  # only 5.0


def test_threshold_counts_are_not_formatted_as_percentages() -> None:
    """Under ``agg_fn: sum`` the indicator is a weighted count, so the ``.1%`` share format is wrong."""
    df = pd.DataFrame(
        {"draw": [0] * 4, "gender": ["Female"] * 4, "party_a": [0.0, 1.0, 2.0, 3.0], "party_b": [0.0] * 4}
    )
    desc: dict[str, Any] = {"res_col": "party_a", "cont_transform": "ge:1", "plot": "columns", "weights": False}
    pi = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), soft_validate(desc, PlotDescriptor))
    assert pi.val_format == ".1%" and pi.data[pi.value_col].iloc[0] == pytest.approx(0.75)

    pi = pp_transform_data(
        pl.LazyFrame(df), _threshold_meta(), soft_validate({**desc, "agg_fn": "sum"}, PlotDescriptor)
    )
    assert pi.val_format == ".0f" and pi.data[pi.value_col].iloc[0] == pytest.approx(3.0)


def test_ge_minus_inf_counts_non_null_responses() -> None:
    """``ge:-inf`` is the "did they answer at all" indicator, so its weighted sum is the response count."""
    df = pd.DataFrame(
        {"draw": [0] * 4, "gender": ["Female"] * 4, "party_a": [0.0, None, 2.0, 3.0], "party_b": [0.0] * 4}
    )
    ppd = soft_validate(
        {"res_col": "party_a", "cont_transform": "ge:-inf", "agg_fn": "sum", "plot": "columns", "weights": False},
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)
    assert pi.data[pi.value_col].iloc[0] == pytest.approx(3.0)  # the null does not count
    assert pi.filtered_size == pytest.approx(4.0)  # ... but it is still in the scope


@pytest.mark.parametrize("bad", ["ge:", "ge:high", "gt:1", "le:1"])
def test_cont_transform_rejects_malformed_threshold(bad: str) -> None:
    """Only `ge` with a numeric cutoff; anything else fails validation, not silently."""
    with pytest.raises((ValueError, ValidationError)):
        soft_validate({"res_col": "party_a", "cont_transform": bad, "plot": "columns"}, PlotDescriptor)


def test_convert_res_categorical_bins_a_continuous_response() -> None:
    """``convert_res="categorical"`` is the inverse direction: bin, then take shares."""
    df = pd.DataFrame(
        {
            "draw": [0] * 8,
            "gender": ["Female"] * 8,
            "party_a": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "party_b": [0.0] * 8,
        }
    )
    ppd = soft_validate(
        {
            "res_col": "party_a",
            "convert_res": "categorical",
            "col_meta": {"party_a": {"bin_breaks": [0.5, 1.5], "bin_labels": ["low", "mid", "high"]}},
            "plot": "columns",
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)
    shares = dict(zip(pi.data["party_a"].astype(str), pi.data[pi.value_col]))
    assert shares == {
        "low": pytest.approx(0.25),
        "mid": pytest.approx(0.25),
        "high": pytest.approx(0.5),
    }


def test_binning_drops_meta_keyed_to_the_old_values() -> None:
    """Colours, groups and a neutral middle naming categories that no longer exist are worse than none."""
    df = pd.DataFrame(
        {"draw": [0] * 4, "gender": ["Female"] * 4, "party_a": [0.0, 1.0, 2.0, 3.0], "party_b": [0.0] * 4}
    )
    stale = {"colors": {"0.0": "#ff0000"}, "groups": {"low": ["0.0", "1.0"]}, "likert": True, "neutral_middle": "1.0"}
    ppd = soft_validate(
        {
            "res_col": "party_a",
            "convert_res": "categorical",
            "col_meta": {"party_a": {"bin_breaks": [1.5], "bin_labels": ["low", "high"], **stale}},
            "plot": "columns",
        },
        PlotDescriptor,
    )
    meta = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd).col_meta["party_a"]
    assert meta.categories == ["low", "high"]
    assert not meta.colors and not meta.groups and not meta.likert and meta.neutral_middle is None


def test_a_binned_response_matches_plots_as_a_categorical_one() -> None:
    """convert_res decides categorical-ness everywhere, not just where the frame is built."""
    df = pd.DataFrame(
        {"draw": [0] * 4, "gender": ["Female", "Male"] * 2, "party_a": [0.0, 1.0, 2.0, 3.0], "party_b": [0.0] * 4}
    )
    desc: dict[str, Any] = {"plot": "default", "res_col": "party_a", "facet_dims": ["gender"]}

    binned = cast(dict, matching_plots({**desc, "convert_res": "categorical"}, df, _threshold_meta(), details=True))

    # Bin labels are never negative, so a plot demanding non-negative values must not reject them
    assert all("nonnegative" not in reasons for _fit, reasons in binned.values())
    assert binned["stacked_columns"][0] >= 0  # a categorical-response plot, reachable only if both agree


def test_convert_res_categorical_rejects_what_it_cannot_bin() -> None:
    """A categorical has nothing to bin; a block would get per-column breaks and keep only the last one's labels."""
    df = pd.DataFrame({"draw": [0, 0], "gender": ["Female", "Male"], "party_a": [1.0, 2.0], "party_b": [0.0, 3.0]})
    ppd = soft_validate({"res_col": "gender", "convert_res": "categorical", "plot": "columns"}, PlotDescriptor)
    with pytest.raises(ValueError, match="needs a numeric column"):
        pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)

    ppd = soft_validate({"res_col": "ratings", "convert_res": "categorical", "plot": "columns"}, PlotDescriptor)
    with pytest.raises(ValueError, match="needs a single-column res_col"):
        pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)


@pytest.mark.parametrize(
    "values",
    [[2.0, 4.0, 6.0, 8.0], pd.Categorical(["2", "4", "6", "8"])],
    ids=["float-column", "stored-as-categorical"],
)
def test_convert_res_continuous_on_continuous_col_keeps_values(values: Any) -> None:
    """A continuous column has no categories to map, so convert_res must cast - never null the whole frame."""
    col_meta = {"continuous": True}
    df = pd.DataFrame(
        {
            "draw": [0] * 4,
            "gender": ["Female", "Male", "Female", "Male"],
            "vote_prob": values,
        }
    )
    ppd = soft_validate(
        {"res_col": "vote_prob", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), _continuous_res_meta(col_meta), ppd)

    assert len(pi.data) > 0
    by_gender = dict(zip(pi.data["gender"], pd.to_numeric(pi.data[pi.value_col], errors="raise")))
    assert by_gender["Female"] == pytest.approx(4.0)
    assert by_gender["Male"] == pytest.approx(6.0)


def test_convert_res_continuous_keeps_full_precision() -> None:
    """Values are the data itself here, not category codes, so they must not be rounded to Float32."""
    df = pd.DataFrame({"draw": [0] * 2, "gender": ["Female", "Male"], "vote_prob": [16777217.0, 1234567891.0]})
    ppd = soft_validate(
        {"res_col": "vote_prob", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), _continuous_res_meta({"continuous": True}), ppd)

    by_gender = dict(zip(pi.data["gender"], pd.to_numeric(pi.data[pi.value_col], errors="raise")))
    assert by_gender["Female"] == 16777217.0


@pytest.mark.parametrize(
    "col_meta,expected",
    [({"continuous": True, "val_range": [0.0, 10.0]}, (0.0, 10.0)), ({"continuous": True}, None)],
    ids=["declared", "undeclared"],
)
def test_convert_res_continuous_reports_declared_val_range(col_meta: dict[str, Any], expected: Any) -> None:
    """A declared val_range survives convert_res; an undeclared one stays unknown rather than becoming (0, 1)."""
    df = pd.DataFrame({"draw": [0] * 4, "gender": ["Female", "Male"] * 2, "vote_prob": [2.0, 4.0, 6.0, 8.0]})
    ppd = soft_validate(
        {"res_col": "vote_prob", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), _continuous_res_meta(col_meta), ppd)

    assert pi.val_range == expected


def test_convert_res_continuous_rejects_datetime() -> None:
    """convert_res on a plain datetime column is not a thing - fail loudly instead of casting timestamps to floats."""
    df = pd.DataFrame({"draw": [0] * 2, "gender": ["Female", "Male"], "when": pd.to_datetime(["2026-01-01"] * 2)})
    ppd = soft_validate(
        {"res_col": "when", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    with pytest.raises(Exception, match="Cannot convert datetime column when"):
        pp_transform_data(pl.LazyFrame(df), _continuous_res_meta({"datetime": True}, col="when"), ppd)


def test_convert_res_continuous_on_bucketed_datetime_maps_categories() -> None:
    """datetime is orthogonal to categorical: a parse-then-bucket column still converts through its categories."""
    df = pd.DataFrame(
        {
            "draw": [0] * 4,
            "gender": ["Female", "Male"] * 2,
            "when": pd.Categorical(["01 Jan 25", "02 Jan 25"] * 2, categories=["01 Jan 25", "02 Jan 25"], ordered=True),
        }
    )
    col_meta = {"datetime": True, "categories": ["01 Jan 25", "02 Jan 25"], "ordered": True, "num_values": [1.0, 2.0]}
    ppd = soft_validate(
        {"res_col": "when", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), _continuous_res_meta(col_meta, col="when"), ppd)

    by_gender = dict(zip(pi.data["gender"], pd.to_numeric(pi.data[pi.value_col], errors="raise")))
    assert by_gender["Female"] == pytest.approx(1.0)
    assert by_gender["Male"] == pytest.approx(2.0)


def test_convert_res_continuous_rejects_uninterpretable_column() -> None:
    """Neither categories to map nor numbers to parse: error instead of casting the whole column to null."""
    df = pd.DataFrame({"draw": [0] * 2, "gender": ["Female", "Male"], "note": ["yes", "no"]})
    ppd = soft_validate(
        {"res_col": "note", "factor_cols": ["gender"], "convert_res": "continuous", "plot": "boxplots"},
        PlotDescriptor,
    )

    with pytest.raises(Exception, match="neither categorical nor continuous"):
        pp_transform_data(pl.LazyFrame(df), _continuous_res_meta({}, col="note"), ppd)


def test_col_meta_override_contradicting_the_annotation_raises() -> None:
    """Descriptor overrides bypass model validation, so the merged result is revalidated - loudly."""
    data_meta = _continuous_res_meta({"categories": ["low", "high"], "ordered": True}, col="q")
    ppd = soft_validate({"res_col": "q", "plot": "boxplots", "col_meta": {"q": {"continuous": True}}}, PlotDescriptor)

    with pytest.raises(ValidationError, match="continuous, so it cannot have categories"):
        _update_data_meta_with_pp_desc(data_meta, ppd)


def test_question_meta_clone_is_nominal() -> None:
    """The synthetic `question` column holds column names, so it inherits no type flag from its group."""
    base = soft_validate({"datetime": True, "categories": ["01 Jan 25"], "val_range": None}, GroupOrColumnMeta)

    clone = _question_meta_clone(base, ["q1", "q2"])

    assert (clone.datetime, clone.continuous, clone.ordered) == (False, False, False)
    assert clone.categories == ["q1", "q2"]


def _battery_meta(n_topics: int, weight_col: str | None = None) -> DataMeta:
    payload: dict[str, object] = {
        "structure": [
            {
                "name": "demographics",
                "scale": {},
                "columns": [["gender", {"categories": ["Female", "Male"]}]],
            },
            {
                "name": "battery",
                "scale": {"continuous": True},
                "columns": [[f"t{i}"] for i in range(n_topics)],
            },
        ]
    }
    if weight_col:
        payload["weight_col"] = weight_col
    return make_data_meta(payload)


def test_convert_res_categorical_composes_with_cont_transform_as_literal_bins() -> None:
    """categorical + a transform = convert -> transform -> categorize (not the old silent no-op).

    Every row rates topic i exactly i, so ordered-avgrank puts each topic at rank i+1 with share
    1.0 - and with 12 topics, numeric label ordering diverges from the lexicographic one.
    """
    n = 12
    df = pd.DataFrame({"draw": [0] * 4, "gender": ["Female"] * 4, **{f"t{i}": [float(i)] * 4 for i in range(n)}})
    ppd = soft_validate(
        {
            "res_col": "battery",
            "factor_cols": ["question"],
            "convert_res": "categorical",
            "cont_transform": "ordered-avgrank",
            "plot": "columns",
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _battery_meta(n), ppd)
    # Literal bins: one category per distinct rank, ordered numerically - not "1", "10", "11", "2", ...
    cats = list(pi.data["battery"].cat.categories)
    assert cats == [str(i) for i in range(1, n + 1)]
    assert pi.col_meta["battery"].num_values == [float(i) for i in range(1, n + 1)]
    assert pi.val_format == ".1%"
    # Topic i sits at rank i+1 with share 1 (shares within each question sum to 1)
    peak = pi.data[pi.data[pi.value_col] > 0]
    assert dict(zip(peak["question"].astype(str), peak["battery"].astype(str))) == {
        f"t{i}": str(i + 1) for i in range(n)
    }
    assert peak[pi.value_col].tolist() == pytest.approx([1.0] * n)


def test_literal_categorization_is_weighted() -> None:
    """The literal path aggregates through the declared weight column, unlike a raw-format count."""
    df = pd.DataFrame(
        {
            "draw": [0] * 2,
            "gender": ["Female"] * 2,
            "w": [3.0, 1.0],
            "t0": [1.0, 2.0],  # ranked 1st by the w=3 row, 2nd by the w=1 row
            "t1": [2.0, 1.0],
        }
    )
    ppd = soft_validate(
        {
            "res_col": "battery",
            "factor_cols": ["question"],
            "convert_res": "categorical",
            "cont_transform": "ordered-avgrank",
            "plot": "columns",
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _battery_meta(2, weight_col="w"), ppd)
    t0 = pi.data[pi.data["question"].astype(str) == "t0"]
    shares = dict(zip(t0["battery"].astype(str), t0[pi.value_col]))
    assert shares["1"] == pytest.approx(0.75)  # weight 3 of 4, not row count 1 of 2
    assert shares["2"] == pytest.approx(0.25)


def test_literal_categorization_caps_distinct_values() -> None:
    """Literal bins on genuinely continuous output fail loudly instead of minting hundreds of categories."""
    df = pd.DataFrame(
        {
            "draw": [0] * 60,
            "gender": ["Female"] * 60,
            "party_a": np.linspace(0, 1, 60),
            "party_b": [0.0] * 60,
        }
    )
    ppd = soft_validate(
        {"res_col": "party_a", "convert_res": "categorical", "cont_transform": "center", "plot": "columns"},
        PlotDescriptor,
    )
    with pytest.raises(Exception, match="bin_breaks"):
        pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)


def test_post_transform_categorical_honors_bin_specs() -> None:
    """With bin_breaks set, the post-transform categorization bins instead of literal categories."""
    df = pd.DataFrame(
        {
            "draw": [0] * 4,
            "gender": ["Female"] * 4,
            "party_a": [0.0, 1.0, 2.0, 3.0],
            "party_b": [0.0] * 4,
        }
    )
    ppd = soft_validate(
        {
            "res_col": "party_a",
            "convert_res": "categorical",
            "cont_transform": "center",  # values become -1.5, -0.5, 0.5, 1.5
            "col_meta": {"party_a": {"bin_breaks": [0.0], "bin_labels": ["below mean", "above mean"]}},
            "plot": "columns",
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), _threshold_meta(), ppd)
    shares = dict(zip(pi.data["party_a"].astype(str), pi.data[pi.value_col]))
    assert shares == {"below mean": pytest.approx(0.5), "above mean": pytest.approx(0.5)}


def _rank_desc(df: pd.DataFrame, n: int, **extra: object) -> Any:
    """Run a battery through convert->transform->categorize with 1 = the row's top item."""
    ppd = soft_validate(
        {
            "res_col": "battery",
            "facet_dims": ["question"],
            "convert_res": "categorical",
            "cont_transform": "ordered-avgrank-desc",
            "plot": "columns",
            **extra,
        },
        PlotDescriptor,
    )
    return pp_transform_data(pl.LazyFrame(df), _battery_meta(n), ppd)


def test_avgrank_desc_ranks_from_the_top_even_with_nulls() -> None:
    """1 = the row's best item; a skipped item shortens the row's rank span, it must not shift it."""
    df = pd.DataFrame({"draw": [0, 0], "gender": ["Female"] * 2, "t0": [3.0, 3.0], "t1": [2.0, 2.0], "t2": [1.0, None]})
    pi = _rank_desc(df, 3)
    top = pi.data[(pi.data["question"].astype(str) == "t0") & (pi.data[pi.value_col] > 0)]
    assert list(top["battery"].astype(str)) == ["1"]  # top item is rank 1 in both rows, null or not


def test_literal_categorization_drops_nulls_from_the_denominator() -> None:
    """A null response is not a category and does not dilute the shares of the ones present."""
    df = pd.DataFrame(
        {
            "draw": [0] * 4,
            "gender": ["Female"] * 4,
            "t0": [1.0, 1.0, 1.0, 1.0],
            "t1": [2.0, 2.0, 2.0, 2.0],
            "t2": [3.0, 3.0, 3.0, None],  # one respondent skipped t2
        }
    )
    pi = _rank_desc(df, 3)
    t2 = pi.data[pi.data["question"].astype(str) == "t2"]
    assert "nan" not in set(t2["battery"].astype(str))
    assert t2[pi.value_col].sum() == pytest.approx(1.0)  # denominator is the 3 answered rows, not 4


def test_facet_sort_on_a_distribution_uses_the_category_scale() -> None:
    """Sorting a rank distribution orders by mean rank, not by the near-constant mean of shares."""
    # t0 is everyone's top item, t2 everyone's worst -> mean ranks 1, 2, 3
    df = pd.DataFrame({"draw": [0] * 3, "gender": ["Female"] * 3, "t0": [3.0] * 3, "t1": [2.0] * 3, "t2": [1.0] * 3})
    ppd = soft_validate(
        {
            "res_col": "battery",
            "facet_dims": ["question"],
            "convert_res": "categorical",
            "cont_transform": "ordered-avgrank-desc",
            "sort": {"question": True},
            "plot": "columns",
        },
        PlotDescriptor,
    )
    pi = create_plot(pp_transform_data(pl.LazyFrame(df), _battery_meta(3), ppd), ppd, dry_run=True)
    assert isinstance(pi, PlotInput)
    assert list(pi.data["question"].cat.categories) == ["t0", "t1", "t2"]


def test_post_categorize_rejects_a_plot_that_cannot_categorize() -> None:
    """On a raw-format plot nothing categorizes, so the pair would be the silent no-op it replaced."""
    df = pd.DataFrame({"draw": [0] * 2, "gender": ["Female"] * 2, "t0": [1.0, 2.0], "t1": [2.0, 1.0]})
    ppd = soft_validate(
        {
            "res_col": "battery",
            "facet_dims": ["question"],
            "convert_res": "categorical",
            "cont_transform": "ordered-avgrank",
            "plot": "boxplots-raw",
        },
        PlotDescriptor,
    )
    with pytest.raises(ValueError, match="longform"):
        pp_transform_data(pl.LazyFrame(df), _battery_meta(2), ppd)


def test_get_plot_fn_builds_chart_from_plot_input() -> None:
    """`get_plot_fn` returns the plot function, callable with a PlotInput plus plot kwargs."""

    pi = PlotInput(
        data=pd.DataFrame(
            {
                "row": ["A", "A", "B", "B"],
                "col": ["X", "Y", "X", "Y"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        facets=[
            FacetMeta(col="row", order=["A", "B"], colors=alt.Undefined),
            FacetMeta(col="col", order=["X", "Y"], colors=alt.Undefined),
        ],
        value_col="value",
        val_format=".2",
    )
    plot = get_plot_fn("matrix")(pi, log_colors=False)

    assert hasattr(plot, "to_dict")


def test_plot_descriptor_accepts_legacy_factor_cols_key():
    """Stored descriptors (dashboards, URLs) use the legacy "factor_cols" key."""
    from salk_toolkit.pp import impute_factor_cols, impute_facet_dims
    from salk_toolkit.validation import PlotDescriptor

    d = PlotDescriptor.model_validate({"plot": "columns", "res_col": "x", "factor_cols": ["age"]})
    assert d.facet_dims == ["age"]
    assert "facet_dims" in d.model_dump()
    assert impute_factor_cols is impute_facet_dims


def test_pp_transform_data_reports_total_and_filtered_weight() -> None:
    """`total_size` is the pre-filter weight, `filtered_size` the post-filter weight -
    both weight-summed (not row counts), so consumers can show "filtered to X%"."""

    data_meta = make_data_meta(
        {
            "weight_col": "w",
            "structure": [
                {
                    "name": "demographics",
                    "scale": {"col_prefix": ""},
                    "columns": [
                        ["gender", {"categories": ["Female", "Male"]}],
                        ["party", {"categories": ["X", "Y"]}],
                    ],
                }
            ],
        }
    )
    # 3 Female @ w=2 (=6) + 2 Male @ w=3 (=6) -> total weight 12, 5 rows
    df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Female", "Male", "Male"],
            "party": ["X", "Y", "X", "Y", "X"],
            "w": [2.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    ppd = soft_validate(
        {"plot": "columns", "res_col": "party", "facet_dims": [], "filter": {"gender": ["Female"]}},
        PlotDescriptor,
    )

    pi = pp_transform_data(df, data_meta, ppd)

    # Weight-based, not row-count-based (which would be 5 and 3)
    assert pi.total_size == 12.0
    assert pi.filtered_size == 6.0

    # A meta-supplied population total overrides the recomputed weight sum
    data_meta.total_size = 1000.0
    pi = pp_transform_data(df, data_meta, ppd)
    assert pi.total_size == 1000.0
    assert pi.filtered_size == 6.0  # post-filter weight is unaffected


def _weights_fixture() -> tuple[DataMeta, pd.DataFrame]:
    data_meta = make_data_meta(
        {
            "weight_col": "w",
            "structure": [
                {
                    "name": "demographics",
                    "scale": {"col_prefix": ""},
                    "columns": [
                        ["gender", {"categories": ["Female", "Male"]}],
                        ["party", {"categories": ["X", "Y"]}],
                    ],
                }
            ],
        }
    )
    # Female rows: X @ w=2, Y @ w=4, X @ w=2 -> weighted X = 0.5, unweighted X = 2/3
    df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Female", "Male", "Male"],
            "party": ["X", "Y", "X", "Y", "X"],
            "w": [2.0, 4.0, 2.0, 3.0, 3.0],
        }
    )
    return data_meta, df


def test_weights_false_uses_rows_as_is() -> None:
    """`weights: False` ignores the declared weight column entirely: shares are
    unweighted and the sizes report plain row counts - even when the annotation
    declares a population `total_size`."""

    data_meta, df = _weights_fixture()
    data_meta.total_size = 1000.0
    ppd = soft_validate(
        {"plot": "columns", "res_col": "party", "filter": {"gender": ["Female"]}, "weights": False},
        PlotDescriptor,
    )

    pi = pp_transform_data(df, data_meta, ppd)

    assert pi.total_size == 5  # row count, not 12.0 (weight sum) nor 1000.0 (declared)
    assert pi.filtered_size == 3
    shares = dict(zip(pi.data["party"], pi.data["percent"]))
    assert shares["X"] == pytest.approx(2 / 3)  # weighted would be 0.5


def test_weights_column_name_overrides_the_declared_one() -> None:
    """A descriptor can weigh by any column, not just the annotation-declared one."""

    data_meta, df = _weights_fixture()
    df = df.assign(w2=[1.0, 1.0, 3.0, 1.0, 1.0])  # Female: X 1+3=4, Y 1 -> X = 0.8
    ppd = soft_validate(
        {"plot": "columns", "res_col": "party", "filter": {"gender": ["Female"]}, "weights": "w2"},
        PlotDescriptor,
    )

    pi = pp_transform_data(df, data_meta, ppd)

    shares = dict(zip(pi.data["party"], pi.data["percent"]))
    assert shares["X"] == pytest.approx(0.8)


def test_a_declared_weight_column_missing_from_the_data_is_an_error() -> None:
    """Parquet and annotation drifting apart must fail, not quietly produce unweighted numbers."""

    data_meta, df = _weights_fixture()
    df = df.drop(columns=["w"])

    for weights in (True, "w"):  # True resolves the declared column; naming it is the same demand
        strict = soft_validate({"plot": "columns", "res_col": "party", "weights": weights}, PlotDescriptor)
        with pytest.raises(ValueError, match="'w'"):
            pp_transform_data(df, data_meta, strict)

    unweighted = soft_validate({"plot": "columns", "res_col": "party", "weights": False}, PlotDescriptor)
    assert pp_transform_data(df, data_meta, unweighted).total_size == 5  # the deliberate way to say it


def test_an_annotation_declaring_no_weight_column_is_unweighted() -> None:
    """`weights: True` is the default, so annotations that never declared a weight column must still run."""

    data_meta, df = _weights_fixture()
    data_meta.weight_col = None

    pi = pp_transform_data(df, data_meta, soft_validate({"plot": "columns", "res_col": "party"}, PlotDescriptor))
    assert pi.total_size == 5  # row count: nothing to weigh by
    assert dict(zip(pi.data["party"], pi.data["percent"]))["X"] == pytest.approx(3 / 5)


def test_weights_none_is_not_a_way_to_ask_for_unweighted() -> None:
    """`None` used to mean "declared column, silently 1.0 if absent"; that ambiguity is gone."""
    with pytest.raises(ValidationError):
        PlotDescriptor.model_validate({"plot": "columns", "res_col": "party", "weights": None})


def _topk_meta() -> DataMeta:
    return make_data_meta(
        {
            "structure": [
                {
                    "name": "ratings",
                    "scale": {"continuous": True},
                    "columns": [["a"], ["b"], ["c"], ["d"]],
                }
            ]
        }
    )


def _topk_shares(df: pl.LazyFrame, transform: str) -> dict[str, float]:
    ppd = soft_validate(
        {
            "plot": "columns",
            "res_col": "ratings",
            "factor_cols": ["question"],
            "agg_fn": "mean",
            "cont_transform": transform,
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(df, _topk_meta(), ppd)
    return {q: v for q, v in zip(pi.data["question"], pi.data[pi.value_col]) if not pd.isna(v)}


def test_ordered_top_ties_selects_everything_at_the_kth_best_value() -> None:
    """Unlike the rank-based `ordered-top2/3`, a tie at the cutoff selects more than k."""
    df = pl.LazyFrame({"a": [3.0], "b": [3.0], "c": [3.0], "d": [1.0]})

    assert _topk_shares(df, "ordered-top-ties:2") == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert sum(_topk_shares(df, "ordered-top2").values()) == pytest.approx(2)  # rank-based: exactly two
    assert _topk_shares(df, "ordered-top-ties:1") == _topk_shares(df, "ordered-top1")  # top1 is value-based


def test_top_k_ranks_a_partly_answered_row_among_its_own_answers() -> None:
    """A respondent who answered 2 of 4 columns still has a top-2, and the unanswered ones stay null."""
    df = pl.LazyFrame({"a": [3.0], "b": [None], "c": [1.0], "d": [None]}, schema={c: pl.Float64 for c in "abcd"})

    assert _topk_shares(df, "ordered-top2") == {"a": 1.0, "c": 1.0}  # not "nothing is in the top 2 of 4"
    assert _topk_shares(df, "ordered-top-ties:2") == {"a": 1.0, "c": 1.0}
    assert _topk_shares(df, "ordered-avgrank") == {"a": 2.0, "c": 1.0}  # ranked among the answered, not 1..4


@pytest.mark.parametrize(
    "bad", ["ordered-top-ties:", "ordered-top-ties:two", "ordered-top-ties:0", "ordered-mid:2", "ordered-top:2"]
)
def test_ordered_topk_rejects_malformed(bad: str) -> None:
    """Parameterized top-k needs a positive integer k on the one registered family."""
    with pytest.raises((ValueError, ValidationError)):
        soft_validate({"plot": "columns", "res_col": "ratings", "cont_transform": bad}, PlotDescriptor)


def test_res_meta_declares_a_virtual_block_over_loose_columns() -> None:
    """`res_meta` builds a block out of columns no annotation block covers —
    the answer to "my battery isn't in a block", without editing the annotation.
    The block unpivots into `question` and crosses with facets like any other."""
    meta = make_data_meta(
        {
            "structure": [
                {"name": "demographics", "scale": {}, "columns": [["gender", {"categories": ["Female", "Male"]}]]},
            ]
        }
    )
    df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male", "Male"],
            "tv": [1.0, 0.0, 1.0, 1.0],
            "radio": [0.0, 0.0, 1.0, 0.0],
            "web": [1.0, 1.0, 1.0, 1.0],
        }
    )
    ppd = soft_validate(
        {
            "plot": "columns",
            "res_col": "media",
            "factor_cols": ["question"],
            "agg_fn": "mean",
            "res_meta": {"name": "media", "scale": {"continuous": True}, "columns": [["tv"], ["radio"], ["web"]]},
        },
        PlotDescriptor,
    )

    pi = pp_transform_data(pl.LazyFrame(df), meta, ppd)
    shares = dict(zip(pi.data["question"], pi.data[pi.value_col]))
    assert shares == {"tv": 0.75, "radio": 0.25, "web": 1.0}

    ppd = soft_validate(
        {
            "plot": "columns",
            "res_col": "media",
            "factor_cols": ["question", "gender"],
            "agg_fn": "mean",
            "res_meta": {"name": "media", "scale": {"continuous": True}, "columns": [["tv"], ["radio"], ["web"]]},
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(pl.LazyFrame(df), meta, ppd)
    by_cell = {(q, g): v for q, g, v in zip(pi.data["question"], pi.data["gender"], pi.data[pi.value_col])}
    assert by_cell[("tv", "Female")] == 0.5
    assert by_cell[("tv", "Male")] == 1.0


def test_weights_expression_builds_the_weight_per_row() -> None:
    """A non-identifier `weights` string is a polars expression (the pl_filter
    contract) - a weight combined from several columns, e.g. design weight x
    turnout propensity. The declared population total is ignored: totals are
    recomputed from the actual weights."""

    data_meta, df = _weights_fixture()
    data_meta.total_size = 1000.0
    df = df.assign(turnout=[1.0, 0.5, 1.0, 1.0, 0.0])
    # Female effective weights: X 2*1=2, Y 4*0.5=2, X 2*1=2 -> X = 4/6
    ppd = soft_validate(
        {
            "plot": "columns",
            "res_col": "party",
            "filter": {"gender": ["Female"]},
            "weights": "pl.col('w') * pl.col('turnout')",
        },
        PlotDescriptor,
    )

    pi = pp_transform_data(df, data_meta, ppd)

    shares = dict(zip(pi.data["party"], pi.data["percent"]))
    assert shares["X"] == pytest.approx(4 / 6)
    assert pi.total_size == pytest.approx(2 + 2 + 2 + 3 + 0)  # expression sum, not 1000.0
    assert pi.filtered_size == pytest.approx(6.0)


def test_e2e_plot_return_input_carries_the_sizes() -> None:
    """`return_input=True` hands back the full PlotInput - the aggregate plus
    `filtered_size`/`total_size`, which under `weights: False` are the row
    counts a payload reports next to the shares. `return_data` discards them."""

    data_meta, df = _weights_fixture()
    pi = cast(
        PlotInput,
        e2e_plot(
            {"plot": "columns", "res_col": "party", "filter": {"gender": ["Female"]}, "weights": False},
            full_df=df,
            data_meta=data_meta,
            return_input=True,
        ),
    )
    assert pi.total_size == 5
    assert pi.filtered_size == 3
    assert dict(zip(pi.data["party"], pi.data["percent"]))["X"] == pytest.approx(2 / 3)


def test_e2e_plot_imputes_facets_for_a_virtual_block() -> None:
    """`impute_facet_dims` must see the res_meta virtual block - it used to read
    the raw annotation meta and KeyError on the block name."""

    meta = make_data_meta(
        {"structure": [{"name": "demographics", "scale": {}, "columns": [["gender", {"categories": ["F", "M"]}]]}]}
    )
    df = pd.DataFrame({"gender": ["F", "M"], "tv": [1.0, 0.0], "web": [1.0, 1.0]})
    pi = cast(
        PlotInput,
        e2e_plot(
            {
                "plot": "columns",
                "res_col": "media",
                "agg_fn": "mean",
                "weights": False,
                "res_meta": {"name": "media", "scale": {"continuous": True}, "columns": [["tv"], ["web"]]},
            },
            full_df=df,
            data_meta=meta,
            return_input=True,
        ),
    )
    assert dict(zip(pi.data["question"], pi.data[pi.value_col])) == {"tv": 0.5, "web": 1.0}


def test_integer_facet_quantile_binning_is_engine_independent() -> None:
    """Splitting integer ties across quantiles used to jitter them with a per-batch RNG, so which
    rows crossed a bin edge depended on the engine, the chunking and the thread count."""
    from salk_toolkit.pp.filters import _discretize_continuous

    ldf = pl.LazyFrame({"uses": np.random.default_rng(0).integers(0, 10, 3000)})
    binned, labels = _discretize_continuous(ldf, "uses", GroupOrColumnMeta(bin_breaks=5))

    counts = [binned.collect(engine=e)["uses"].value_counts().sort("uses") for e in ("in-memory", "streaming")]
    assert counts[0].equals(counts[1])
    sizes = sorted(counts[0]["count"])
    assert sizes[-1] - sizes[0] <= 2 and len(labels) == 5  # ties split evenly, up to quantile interpolation


def test_integer_facet_binning_does_not_follow_file_order() -> None:
    """Deterministic is not enough: splitting a tie group by its position in the frame invents a
    100/0 association with anything the file happens to be sorted by, e.g. wave or region."""
    from salk_toolkit.pp.filters import _discretize_continuous

    # The 40s straddle the median, and the file is grouped by gender - as survey exports are
    ldf = pl.LazyFrame({"age": [30] * 300 + [40] * 400 + [50] * 300, "gender": ["F"] * 500 + ["M"] * 500})
    binned, _ = _discretize_continuous(ldf, "age", GroupOrColumnMeta(bin_breaks=2))

    cells = {
        (str(r["age"]), r["gender"]): r["len"]
        for r in binned.collect().group_by(["age", "gender"]).len().rows(named=True)
    }
    tied_below = cells[("Bottom 50%", "M")]  # all-M side of the tie group that fell below the edge
    assert 60 <= tied_below <= 140  # 0 when the split follows row order; ~100 when it does not


def test_bin_breaks_make_an_integer_facet_exact() -> None:
    """Declared edges opt out of quantile binning, so each integer value is its own cell."""

    meta = make_data_meta(
        {"structure": [{"name": "demographics", "scale": {}, "columns": [["party", {"categories": ["X", "Y"]}]]}]}
    )
    df = pd.DataFrame({"party": ["X"] * 6 + ["Y"] * 4, "uses": [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]})
    pi = cast(
        PlotInput,
        e2e_plot(
            {
                "plot": "columns",
                "res_col": "party",
                "factor_cols": ["uses"],
                "agg_fn": "sum",
                "weights": False,
                "col_meta": {"uses": {"bin_breaks": [0, 1, 2]}},
            },
            full_df=df,
            data_meta=meta,
            return_input=True,
        ),
    )
    cells = dict(zip(zip(map(str, pi.data["uses"]), pi.data["party"]), pi.data["percent"]))
    assert len(cells) == 4  # one cell per (integer value, party), no quantile collapsing
    assert sorted(cells.values()) == [1, 3, 3, 3]


def test_unweighted_total_size_is_the_row_count_not_the_declared_weight() -> None:
    """weights: False means the total is a row count, whatever the declared column holds.

    It used to be derived by summing the synthesized unit weight over the whole
    pre-filter frame, which produces every row to add up a literal; pl.len() comes
    off the scan's metadata instead.
    """

    meta = make_data_meta(
        {
            "weight_col": "N",
            "total_size": 500.0,
            "structure": [{"name": "demographics", "scale": {}, "columns": [["party", {"categories": ["X", "Y"]}]]}],
        }
    )
    df = pd.DataFrame({"party": ["X", "X", "Y", "Y", "Y"], "N": [100.0] * 5})

    unweighted = cast(
        PlotInput,
        e2e_plot(
            {"plot": "columns", "res_col": "party", "weights": False}, full_df=df, data_meta=meta, return_input=True
        ),
    )
    assert unweighted.total_size == 5
    assert unweighted.filtered_size == 5

    # The declared weighting still reports the annotation's population total.
    weighted = cast(
        PlotInput, e2e_plot({"plot": "columns", "res_col": "party"}, full_df=df, data_meta=meta, return_input=True)
    )
    assert weighted.total_size == 500.0


def test_expression_stats_answer_cells_over_different_row_sets() -> None:
    """Each stat is a row-level expression, so masked cells ("uses i but not j")
    ride one group_by instead of one descriptor per row set."""

    meta = make_data_meta({"structure": [{"name": "d", "scale": {}, "columns": [["tv", {"categories": ["0", "1"]}]]}]})
    df = pd.DataFrame({"tv": [1, 1, 0, 0], "web": [1, 0, 1, 0], "N": [10.0, 20.0, 30.0, 40.0]})
    d = e2e_plot(
        {
            "plot": "columns",
            "res_col": "tv",
            "weights": False,
            "stats": [
                {"name": "tv_share", "expr": "(pl.col('tv') > 0)"},
                {"name": "tv_not_web", "expr": "((pl.col('tv') > 0) & ~(pl.col('web') > 0))"},
                {"name": "tv_n", "expr": "(pl.col('tv') > 0)", "agg_fn": "sum"},
            ],
        },
        full_df=df,
        data_meta=meta,
        return_data=True,
    )
    row = d.to_dict("records")[0]
    assert row["tv_share"] == pytest.approx(0.5)
    assert row["tv_not_web"] == pytest.approx(0.25)
    assert row["tv_n"] == 2


def test_expression_stats_apply_the_declared_weighting() -> None:
    """agg_fn folds the weight in per stat: mean over non-null rows, weighted sum."""

    meta = make_data_meta(
        {
            "weight_col": "N",
            "structure": [{"name": "d", "scale": {}, "columns": [["tv", {"categories": ["0", "1"]}]]}],
        }
    )
    df = pd.DataFrame({"tv": [1, 1, 0, 0], "N": [10.0, 20.0, 30.0, 40.0]})
    d = e2e_plot(
        {
            "plot": "columns",
            "res_col": "tv",
            "stats": [
                {"name": "share", "expr": "(pl.col('tv') > 0)"},
                {"name": "mass", "expr": "(pl.col('tv') > 0)", "agg_fn": "sum"},
            ],
        },
        full_df=df,
        data_meta=meta,
        return_data=True,
    )
    row = d.to_dict("records")[0]
    assert row["share"] == pytest.approx(30.0 / 100.0)
    assert row["mass"] == pytest.approx(30.0)


def test_cont_transform_validates_against_the_live_registry() -> None:
    """The point of dropping the frozen Literal: a name registered after import validates."""
    from salk_toolkit.pp.transforms import custom_row_transforms

    for builtin in ("center", "zscore", "01range", "softmax", "ordered-top1", "ordered-top3"):
        assert (
            soft_validate({"plot": "columns", "res_col": "p", "cont_transform": builtin}, PlotDescriptor).cont_transform
            == builtin
        )

    with pytest.raises(ValidationError):
        soft_validate({"plot": "columns", "res_col": "p", "cont_transform": "not-registered"}, PlotDescriptor)

    custom_row_transforms["late-registered"] = (lambda x: x, ".1%")
    try:
        assert (
            soft_validate(
                {"plot": "columns", "res_col": "p", "cont_transform": "late-registered"}, PlotDescriptor
            ).cont_transform
            == "late-registered"
        )
    finally:
        del custom_row_transforms["late-registered"]


@pytest.mark.parametrize(
    ("weights", "total_size"),
    [
        (None, 1000.0),  # declared weighting keeps the annotation's population total
        (True, 1000.0),
        ("N", 9.0),  # any other override recomputes it from the actual weights
        ("w2", 7.0),
        (False, 3.0),  # unweighted: the row count
        ("pl.col('N') * 2", 18.0),
    ],
)
def test_total_size_per_weights_mode(weights: object, total_size: float) -> None:
    """A declared population total describes the declared weighting only."""

    meta = make_data_meta(
        {
            "weight_col": "N",
            "total_size": 1000.0,
            "structure": [{"name": "d", "scale": {}, "columns": [["p", {"categories": ["x", "y"]}]]}],
        }
    )
    df = pd.DataFrame({"p": ["x", "x", "y"], "N": [2.0, 3.0, 4.0], "w2": [1.0, 1.0, 5.0]})
    desc: dict[str, Any] = {"plot": "columns", "res_col": "p"}
    if weights is not None:
        desc["weights"] = weights
    pi = cast(PlotInput, e2e_plot(desc, full_df=df, data_meta=meta, return_input=True))
    assert pi.total_size == total_size


def test_matching_plots_sees_a_virtual_block() -> None:
    """`impute` reads the descriptor-updated meta, so a res_meta block is not an unknown
    column name here either - the same fix e2e_plot got."""
    meta = make_data_meta(
        {"structure": [{"name": "demographics", "scale": {}, "columns": [["gender", {"categories": ["F", "M"]}]]}]}
    )
    df = pd.DataFrame({"gender": ["F", "M"], "tv": [1.0, 0.0], "web": [1.0, 1.0]})
    matches = matching_plots(
        {
            "plot": "columns",
            "res_col": "media",
            "agg_fn": "mean",
            "weights": False,
            "res_meta": {"name": "media", "scale": {"continuous": True}, "columns": [["tv"], ["web"]]},
        },
        df,
        meta,
        impute=True,
    )
    assert "columns" in matches


def test_stats_rejects_the_shapes_it_cannot_serve() -> None:
    """A block res_col loses the columns the expressions name, and stats output has no
    single value column to plot - both should say so rather than fail inside polars."""

    meta = make_data_meta(
        {
            "structure": [
                {"name": "d", "scale": {}, "columns": [["gender", {"categories": ["F", "M"]}]]},
                {"name": "b", "scale": {"continuous": True}, "columns": [["tv"], ["web"]]},
            ]
        }
    )
    df = pd.DataFrame({"tv": [1.0, 0.0], "web": [1.0, 1.0], "gender": ["F", "M"]})
    stats = [{"name": "x", "expr": "(pl.col('tv') > 0)"}]

    # Both the wide-aggregated block (faceted on 'question') and the melted one, which used to
    # slip past the guard and multiply every `sum` by the block width
    for facets in (["question"], ["gender"]):
        with pytest.raises(ValueError, match="block res_col"):
            e2e_plot(
                {"plot": "columns", "res_col": "b", "factor_cols": facets, "weights": False, "stats": stats},
                full_df=df,
                data_meta=meta,
                return_data=True,
                impute=False,
            )

    with pytest.raises(ValueError, match="data-only"):
        e2e_plot({"plot": "columns", "res_col": "tv", "weights": False, "stats": stats}, full_df=df, data_meta=meta)

    with pytest.raises(ValueError, match="longform"):  # raw plots would otherwise drop stats silently
        e2e_plot(
            {"plot": "boxplots-raw", "res_col": "tv", "weights": False, "stats": stats},
            full_df=df,
            data_meta=meta,
            return_data=True,
        )

    with pytest.raises(ValueError, match="collide"):
        e2e_plot(
            {
                "plot": "columns",
                "res_col": "tv",
                "factor_cols": ["question"],
                "weights": False,
                "stats": [{"name": "question", "expr": "(pl.col('tv') > 0)"}],
            },
            full_df=df,
            data_meta=meta,
            return_data=True,
        )


def test_stats_does_not_combine_with_the_single_statistic_path() -> None:
    """Each stat carries its own agg_fn, so a descriptor-level one is a contradiction, not a default."""
    stats = [{"name": "x", "expr": "(pl.col('tv') > 0)"}]
    for extra in ({"agg_fn": "sum"}, {"cont_transform": "ge:1"}):
        with pytest.raises(ValidationError, match="each stat carries its own"):
            PlotDescriptor.model_validate({"plot": "columns", "res_col": "tv", "stats": stats, **extra})

    with pytest.raises(ValidationError, match="appear more than once"):  # else polars dies on the duplicate alias
        PlotDescriptor.model_validate({"plot": "columns", "res_col": "tv", "stats": stats + stats})


def test_a_bad_expression_names_the_field_it_came_from() -> None:
    """`stats`, `weights` and `pl_filter` share one eval, so the message has to say which one failed."""
    from salk_toolkit.pp.wrangle import _eval_expr

    with pytest.raises(ValueError, match=r"stats\[x\].expr"):
        _eval_expr("pl.col('tv'", "stats[x].expr")
    with pytest.raises(ValueError, match="not a polars expression"):
        _eval_expr("'tv'", "weights")


def test_expression_stats_run_per_facet_in_one_group_by() -> None:
    """The point of stats: cells over different row sets, resolved for every facet level in one scan."""

    meta = make_data_meta(
        {
            "structure": [
                {
                    "name": "d",
                    "scale": {},
                    "columns": [["tv", {"categories": ["0", "1"]}], ["gender", {"categories": ["F", "M"]}]],
                }
            ]
        }
    )
    df = pd.DataFrame({"tv": [1, 1, 0, 0], "web": [1, 0, 1, 0], "gender": ["F", "F", "M", "M"]})
    d = e2e_plot(
        {
            "plot": "columns",
            "res_col": "tv",
            "factor_cols": ["gender"],
            "weights": False,
            "stats": [
                {"name": "tv_share", "expr": "(pl.col('tv') > 0)"},
                {"name": "tv_not_web", "expr": "((pl.col('tv') > 0) & ~(pl.col('web') > 0))", "agg_fn": "sum"},
            ],
        },
        full_df=df,
        data_meta=meta,
        return_data=True,
    )
    rows = {r["gender"]: r for r in d.to_dict("records")}
    assert rows["F"]["tv_share"] == pytest.approx(1.0) and rows["M"]["tv_share"] == pytest.approx(0.0)
    assert rows["F"]["tv_not_web"] == pytest.approx(1.0) and rows["M"]["tv_not_web"] == pytest.approx(0.0)
