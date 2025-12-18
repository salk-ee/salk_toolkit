"""Tests for validation helpers in `salk_toolkit.validation`."""

from salk_toolkit.validation import DataMeta, soft_validate


def test_constant_tracking() -> None:
    """`soft_validate(..., context=...)` passes context through constant replacement."""
    raw_meta = {
        "constants": {"my_color": "#ff0000"},
        "structure": {"B1": {"name": "B1", "columns": {"C1": {"categories": ["A"], "colors": {"A": "my_color"}}}}},
    }

    # Test without tracking (default)
    meta = soft_validate(raw_meta, DataMeta)
    assert meta.structure["B1"].columns["C1"].colors["A"].as_hex() == "#f00"

    # Test with tracking
    tracker = set()
    context = {"tracker": tracker}
    meta = soft_validate(raw_meta, DataMeta, context=context)

    assert meta.structure["B1"].columns["C1"].colors["A"].as_hex() == "#f00"
    assert "my_color" in tracker
