"""
Unit tests for plot pipeline utilities in salk_toolkit.pp.
"""

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from salk_toolkit.pp import (
    _calculate_priority as calculate_priority,
    impute_factor_cols,
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
        "factor_cols": ["gender"],
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
        "factor_cols": ["facet"],
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


def test_impute_factor_cols_handles_categorical_and_continuous_cases() -> None:
    """`impute_factor_cols` should handle categorical and continuous conversions."""
    col_meta = {
        "res_group": GroupOrColumnMeta.model_validate({"columns": ["res_variant"], "categories": ["Yes", "No"]}),
        "res_variant": GroupOrColumnMeta.model_validate({"categories": ["Yes", "No"]}),
        "region": GroupOrColumnMeta(),
        "question": GroupOrColumnMeta(),
    }

    categorical_desc = {
        "plot": "test_plot",
        "res_col": "res_group",
        "factor_cols": ["region"],
    }
    categorical_factors = impute_factor_cols(categorical_desc, col_meta)
    assert categorical_factors == ["res_group", "region", "question"]

    continuous_desc = {
        "plot": "test_plot",
        "res_col": "res_group",
        "factor_cols": ["region"],
        "convert_res": "continuous",
    }
    continuous_factors = impute_factor_cols(continuous_desc, col_meta)
    assert continuous_factors == ["question", "region"]
