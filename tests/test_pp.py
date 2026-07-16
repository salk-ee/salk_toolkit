"""
Unit tests for plot pipeline utilities in salk_toolkit.pp.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import altair as alt

from salk_toolkit.pp import (
    _calculate_priority as calculate_priority,
    _transform_cont,
    get_plot_fn,
    impute_facet_dims,
    matching_plots,
    PlotMeta,
    registry,
    registry_meta,
    _stk_deregister as stk_deregister,
    stk_plot,
    _update_data_meta_with_pp_desc,
)
from salk_toolkit.validation import DataMeta, GroupOrColumnMeta, PlotDescriptor, soft_validate


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


def test_get_plot_fn_legacy_wrapper_builds_chart() -> None:
    """`get_plot_fn` should support the legacy `get_plot_fn(name)(**pparams)` convention."""

    plot = get_plot_fn("matrix")(
        data=pd.DataFrame(
            {
                "row": ["A", "A", "B", "B"],
                "col": ["X", "Y", "X", "Y"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        facets=[
            {"col": "row", "order": ["A", "B"], "colors": alt.Undefined},
            {"col": "col", "order": ["X", "Y"], "colors": alt.Undefined},
        ],
        value_col="value",
        val_format=".2",
        log_colors=False,
    )

    assert hasattr(plot, "to_dict")


def test_plot_descriptor_accepts_legacy_factor_cols_key():
    """Stored descriptors (dashboards, URLs) use the legacy "factor_cols" key."""
    from salk_toolkit.pp import impute_factor_cols, impute_facet_dims
    from salk_toolkit.validation import PlotDescriptor

    d = PlotDescriptor.model_validate({"plot": "columns", "res_col": "x", "factor_cols": ["age"]})
    assert d.facet_dims == ["age"]
    assert "facet_dims" in d.model_dump()
    assert impute_factor_cols is impute_facet_dims
