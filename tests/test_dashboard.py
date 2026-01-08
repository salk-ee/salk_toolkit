"""
Tests for the Streamlit dashboard helper utilities.
"""

from __future__ import annotations

import csv
from pathlib import Path

import altair as alt
import pandas as pd
import polars as pl
import pytest

from salk_toolkit import dashboard
from salk_toolkit.validation import DataMeta, soft_validate


def test_alias_file_redirects_missing(tmp_path: Path) -> None:
    """Alias map should redirect missing files to a known replacement."""
    missing = tmp_path / "missing.parquet"
    alias = tmp_path / "redirect.parquet"
    file_map = {str(missing): str(alias)}

    resolved = dashboard.alias_file(str(missing), file_map)

    assert resolved == str(alias)


def test_alias_file_preserves_existing(tmp_path: Path) -> None:
    """Alias map must not overwrite an existing file path."""
    existing = tmp_path / "existing.parquet"
    existing.write_text("")
    file_map = {str(existing): str(tmp_path / "redirect.parquet")}

    resolved = dashboard.alias_file(str(existing), file_map)

    assert resolved == str(existing)


def test_log_event_appends_rows(tmp_path: Path) -> None:
    """`log_event` should append multiple rows to the CSV log."""
    log_path = tmp_path / "events.csv"

    dashboard.log_event("login-success", "user-1", str(log_path))
    dashboard.log_event("view-page", "user-2", str(log_path))

    with log_path.open() as fh:
        rows = list(csv.reader(fh))

    assert [row[1:] for row in rows] == [
        ["login-success", "user-1"],
        ["view-page", "user-2"],
    ]


def test_translate_with_dict_handles_missing() -> None:
    """Dictionary-based translator should fall back to the source string."""
    translate = dashboard.translate_with_dict({"hello": "tere"})

    assert translate("hello") == "tere"
    assert translate("unknown") == "unknown"


def test_log_missing_translations_records() -> None:
    """Missing translation logger should store unknown keys."""
    captured = {}

    wrapped = dashboard.log_missing_translations(lambda s: s, captured)
    result = wrapped("value")

    assert result == "value"
    assert captured == {"value": None}
    changed = dashboard.log_missing_translations(lambda s: s.upper(), captured)
    changed("translated")
    assert "translated" not in captured


def test_clean_missing_translations_filters_numbers() -> None:
    """Missing translation cleaner should drop numeric-like keys."""
    missing = {"keep": None, "123": None, "456.7": None}
    filtered = dashboard.clean_missing_translations(missing, tdict={"keep": "hoia"})

    assert "keep" not in filtered
    assert "123" not in filtered
    assert "456.7" not in filtered
    assert filtered == {}


def test_add_missing_to_dict_merges_existing() -> None:
    """Merge helper should retain base translations and fill gaps."""
    missing = {"world": None}
    base = {"hello": "tere"}

    merged = dashboard.add_missing_to_dict(missing, base)

    assert merged["hello"] == "tere"
    assert merged["world"] == "world"


def test_plot_matrix_html_generates_embed_block() -> None:
    """`plot_matrix_html` should emit a Vega embed block."""
    chart = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [1, 2]})).mark_line().encode(x="x", y="y").properties(width=200)

    html = dashboard.plot_matrix_html(chart, uid="test", width=400, responsive=True)

    assert html is not None
    assert "https://cdn.jsdelivr.net/npm/vega@5" in html
    assert "vegaEmbed" in html
    assert "test-0" in html

    # Test that uid with spaces is sanitized
    html_with_spaces = dashboard.plot_matrix_html(chart, uid="test plot with spaces", width=400, responsive=True)
    assert html_with_spaces is not None
    assert "test_plot_with_spaces" in html_with_spaces
    assert "test plot with spaces" not in html_with_spaces
    assert "test_plot_with_spaces-0" in html_with_spaces

    # Test that uid with special characters is sanitized
    html_with_special = dashboard.plot_matrix_html(
        chart, uid="test@plot#with$special%chars", width=400, responsive=True
    )
    assert html_with_special is not None
    assert "test_plot_with_special_chars" in html_with_special
    assert "test@plot#with$special%chars" not in html_with_special
    assert "test_plot_with_special_chars-0" in html_with_special


@pytest.fixture
def sample_meta() -> DataMeta:
    """Synthetic dashboard metadata used by filter limit tests."""
    meta = {
        "structure": [
            {
                "name": "demographics",
                "columns": [
                    ["age", {"continuous": True}],
                    ["gender", {"categories": ["male", "female"], "ordered": False}],
                ],
            },
            {
                "name": "opinions",
                "scale": {
                    "col_prefix": "op_",
                },
                "columns": [
                    [
                        "support",
                        {
                            "categories": [
                                "Strongly disagree",
                                "Disagree",
                                "Agree",
                                "Strongly agree",
                            ],
                            "ordered": True,
                        },
                    ]
                ],
            },
        ]
    }
    meta["files"] = [{"file": "__test__", "opts": {}, "code": "F0"}]
    return soft_validate(meta, DataMeta)


def test_get_filter_limits_handles_groups_and_continuous(sample_meta: DataMeta) -> None:
    """Filter limit extraction should handle grouped metadata and numeric ranges."""
    frame = pl.DataFrame(
        {
            "age": [21, 34, 42],
            "gender": ["male", "female", "female"],
            "op_support": ["Agree", "Disagree", "Strongly agree"],
        }
    ).lazy()

    limits = dashboard._get_filter_limits.__wrapped__(  # type: ignore[attr-defined]
        frame, dims=["opinions", "gender", "age"], dmeta=sample_meta, uid="demo"
    )

    assert limits["opinions"]["categories"] == ["support"]
    assert limits["opinions"]["group"] is True
    assert limits["gender"]["categories"] == ["male", "female"]
    assert limits["gender"]["ordered"] is False
    assert limits["age"]["continuous"] is True
    assert limits["age"]["min"] == 21
    assert limits["age"]["max"] == 42


def test_get_filter_limits_infer_categories_from_correct_column() -> None:
    """Categories 'infer' must extract unique values from the target column, not another column.

    This is a regression test for a bug where the code used:
        ldf.select(pl.all()).unique(d).collect().to_series().sort().to_list()

    The bug was that .to_series() returns the FIRST column by default,
    so if the first column was numeric (like 'draw'), the categories
    would be integers from that column instead of the actual string values
    from the target column.

    The fix is:
        ldf.select(pl.col(d).unique()).collect()[d].sort().to_list()
    """
    # Create metadata with categories: "infer"
    meta = {
        "structure": [
            {
                "name": "demographics",
                "columns": [
                    # citizen has categories: "infer" - bug would return values from 'draw' column
                    ["citizen", {"categories": "infer"}],
                ],
            },
        ],
        "files": [{"file": "__test__", "opts": {}, "code": "F0"}],
    }
    dmeta = soft_validate(meta, DataMeta)

    # Create a DataFrame where:
    # - First column is 'draw' (numeric) with values [1, 4]
    # - Target column 'citizen' has string values ['Estonian', 'Other']
    # The bug would return [1, 4] instead of ['Estonian', 'Other']
    frame = pl.DataFrame(
        {
            "draw": [1, 1, 4, 4],  # First column - numeric
            "citizen": ["Estonian", "Estonian", "Other", "Other"],  # Target column - strings
        }
    ).lazy()

    limits = dashboard._get_filter_limits.__wrapped__(  # type: ignore[attr-defined]
        frame, dims=["citizen"], dmeta=dmeta, uid="infer_test"
    )

    # The bug would cause this to be [1, 4] (from 'draw' column)
    # The fix correctly returns ['Estonian', 'Other'] (from 'citizen' column)
    assert "citizen" in limits
    categories = limits["citizen"]["categories"]

    # Verify we got string categories, not integers
    assert all(isinstance(c, str) for c in categories), (
        f"Expected string categories but got: {categories}. "
        "This indicates the bug where .to_series() returns the wrong column."
    )
    assert set(categories) == {"Estonian", "Other"}, f"Expected {{'Estonian', 'Other'}} but got {set(categories)}"
