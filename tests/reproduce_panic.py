"""Regression test to reproduce Polars panic in pp_transform_data."""

import os

import polars as pl
import pytest

from salk_toolkit.io import read_parquet_with_metadata
from salk_toolkit.pp import pp_transform_data
from salk_toolkit.validation import PlotDescriptor, soft_validate

PARQUET = "tests/leedu_2026_full.parquet"


@pytest.fixture(scope="module")
def ldf_and_meta():
    "Define fixture"
    if not os.path.exists(PARQUET):
        pytest.skip(f"Parquet file not found: {PARQUET}")
    pl.enable_string_cache()
    return read_parquet_with_metadata(PARQUET, lazy=True)


def test_pp_transform_data_no_panic(ldf_and_meta):
    "Tries to produce Polars panic"
    ldf, full_meta = ldf_and_meta
    ppd = soft_validate(
        {
            "res_col": "maxdiff_score",
            "factor_cols": [],
            "plot": "maxdiff",
            "calculated_draws": True,
        },
        PlotDescriptor,
    )
    pi = pp_transform_data(ldf, full_meta.data, ppd)
    assert pi.data is not None
