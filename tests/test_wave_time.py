"""Tests for the auto-generated wave_time survey-date column."""

import json

import pandas as pd
import pytest

from salk_toolkit.io import read_annotated_data, read_and_process_data


def write_json(path, data):
    """Write dict as JSON."""
    with open(path, "w") as f:
        json.dump(data, f)


def make_wave(tmp_path, code, dates=None, extra_meta=None, n=4):
    """Write a small wave csv + meta; returns the meta path."""
    df = pd.DataFrame({"q": ["Yes", "No"] * (n // 2)})
    df.to_csv(tmp_path / f"{code}.csv", index=False)
    meta = {
        "file": f"{code}.csv",
        "structure": [{"name": "opinions", "columns": [["q", {"categories": ["No", "Yes"]}]]}],
        **(dates or {}),
        **(extra_meta or {}),
    }
    mpath = tmp_path / f"{code}_meta.json"
    write_json(mpath, meta)
    return str(mpath)


class TestSingleMeta:
    """Single-meta injection behavior."""

    def test_midpoint_from_start_end(self, tmp_path):
        """Date = start/end midpoint; single wave hidden."""
        m = make_wave(tmp_path, "w1", {"collection_start": "2026-01-10", "collection_end": "2026-01-20"})
        df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None
        assert list(df["wave_time"].unique()) == ["2026-01-15"]
        blk = meta.structure["waves"]
        assert blk.columns["wave_time"].categories == ["2026-01-15"]
        assert blk.columns["wave_time"].ordered
        assert blk.hidden  # single wave -> hidden from dashboards

    def test_collection_center_overrides(self, tmp_path):
        """collection_center beats the midpoint."""
        m = make_wave(
            tmp_path,
            "w1",
            {"collection_start": "2026-01-10", "collection_end": "2026-01-20", "collection_center": "2026-01-12"},
        )
        df = read_annotated_data(m)
        assert list(df["wave_time"].unique()) == ["2026-01-12"]

    def test_no_dates_no_column(self, tmp_path):
        """No dates -> no wave_time column."""
        df = read_annotated_data(make_wave(tmp_path, "w1"))
        assert "wave_time" not in df.columns

    def test_wave_time_false_disables(self, tmp_path):
        """wave_time: false suppresses injection."""
        m = make_wave(tmp_path, "w1", {"collection_start": "2026-01-10"}, {"wave_time": False})
        df = read_annotated_data(m)
        assert "wave_time" not in df.columns

    def test_user_declared_column_wins(self, tmp_path):
        """Hand-declared column suppresses injection."""
        df = pd.DataFrame({"q": ["Yes", "No"], "wave_time": ["a", "b"]})
        df.to_csv(tmp_path / "w1.csv", index=False)
        meta = {
            "file": "w1.csv",
            "collection_center": "2026-01-10",
            "structure": [{"name": "b", "columns": ["q", ["wave_time", {"categories": ["a", "b"]}]]}],
        }
        write_json(tmp_path / "w1_meta.json", meta)
        df, m = read_annotated_data(str(tmp_path / "w1_meta.json"), return_meta=True)
        assert m is not None
        assert set(df["wave_time"]) == {"a", "b"}
        assert "waves" not in m.structure


class TestCombining:
    """read_and_process_data multi-meta combining."""

    def test_two_waves_chronological(self, tmp_path):
        """Categories chronological regardless of file order."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-04-15"})
        # Later wave listed FIRST: chronological category order must not depend on file order
        df, meta = read_and_process_data({"files": [{"file": m2}, {"file": m1}]}, return_meta=True)
        cats = meta.structure["waves"].columns["wave_time"].categories
        assert cats == ["2026-01-15", "2026-04-15"]
        assert list(df["wave_time"].cat.categories) == cats
        assert df["wave_time"].cat.ordered
        assert not meta.structure["waves"].hidden
        # per-row values follow each wave's own date
        assert (df.loc[df["file_code"] == "F1", "wave_time"] == "2026-01-15").all()
        assert (df.loc[df["file_code"] == "F0", "wave_time"] == "2026-04-15").all()

    def test_same_date_shares_category(self, tmp_path):
        """Same-day waves share one category."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-01-15"})
        df = read_and_process_data({"files": [{"file": m1}, {"file": m2}]})
        assert list(df["wave_time"].cat.categories) == ["2026-01-15"]

    def test_dateless_wave_gets_na_with_warning(self, tmp_path):
        """A dateless wave beside a dated one gets NA rows plus a warning naming the file."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2")  # no dates
        with pytest.warns(UserWarning, match="No survey date"):
            df = read_and_process_data({"files": [{"file": m2}, {"file": m1}]})
        assert df["wave_time"].notna().sum() == 4 and df["wave_time"].isna().sum() == 4


class TestForeignColumns:
    """Pre-existing / user-owned columns of the same name are never trampled."""

    def _raw_col_meta(self, tmp_path, values, extra=None):
        pd.DataFrame({"q": ["Yes", "No"], "wave_time": values}).to_csv(tmp_path / "d.csv", index=False)
        meta = {
            "file": "d.csv",
            "collection_center": "2026-01-15",
            **(extra or {}),
            "structure": [{"name": "op", "columns": [["q", {"categories": ["No", "Yes"]}]]}],
        }
        write_json(tmp_path / "d_meta.json", meta)
        return str(tmp_path / "d_meta.json")

    def test_undeclared_non_date_column_does_not_crash(self, tmp_path):
        """A raw column of that name holding non-dates is left alone, not parsed as wave dates."""
        m = self._raw_col_meta(tmp_path, ["wave 1", "wave 1"])
        with pytest.warns(UserWarning):
            df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None and "waves" not in meta.structure

    def test_user_waves_block_columns_survive(self, tmp_path):
        """Injecting into a user block named 'waves' keeps its own columns and visibility."""
        pd.DataFrame({"q": ["Yes", "No"], "surf": [1, 2]}).to_csv(tmp_path / "d.csv", index=False)
        write_json(
            tmp_path / "d_meta.json",
            {
                "file": "d.csv",
                "collection_center": "2026-01-15",
                "structure": [
                    {"name": "waves", "columns": [["q", {"categories": ["No", "Yes"]}], ["surf", {"continuous": True}]]}
                ],
            },
        )
        df, meta = read_annotated_data(str(tmp_path / "d_meta.json"), return_meta=True)
        assert meta is not None
        assert {"q", "surf"} <= set(meta.structure["waves"].columns) and {"q", "surf"} <= set(df.columns)


class TestNestedMeta:
    """Parent meta over child metas."""

    def test_parent_preserves_child_dates(self, tmp_path):
        """Children's dates carry through a dateless parent."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-04-15"})
        parent = {
            "files": [{"file": m2, "code": "A"}, {"file": m1, "code": "B"}],  # reverse chronological
            "structure": [{"name": "opinions", "columns": [["q", {"categories": ["No", "Yes"]}]]}],
        }
        write_json(tmp_path / "parent_meta.json", parent)
        df, meta = read_annotated_data(str(tmp_path / "parent_meta.json"), return_meta=True)
        assert meta is not None
        assert meta.structure["waves"].columns["wave_time"].categories == ["2026-01-15", "2026-04-15"]
        assert df["wave_time"].cat.ordered
        assert set(df["wave_time"]) == {"2026-01-15", "2026-04-15"}

    def test_parent_date_fills_dateless_child(self, tmp_path):
        """Parent's date fills a dateless child."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2")  # child without dates -> parent's date applies
        parent = {
            "files": [{"file": m1, "code": "A"}, {"file": m2, "code": "B"}],
            "collection_center": "2026-04-15",
            "structure": [{"name": "opinions", "columns": [["q", {"categories": ["No", "Yes"]}]]}],
        }
        write_json(tmp_path / "parent_meta.json", parent)
        df = read_annotated_data(str(tmp_path / "parent_meta.json"))
        assert set(df["wave_time"].dropna()) == {"2026-01-15", "2026-04-15"}
