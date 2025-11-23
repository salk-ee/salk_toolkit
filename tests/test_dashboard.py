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
    meta["file"] = "__test__"
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
