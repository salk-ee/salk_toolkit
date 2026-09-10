"""Tests for the auto-generated wave_time survey-date column."""

import json

import pandas as pd
import pytest

from salk_toolkit.io import read_annotated_data, read_and_process_data, write_parquet_with_metadata


QBLOCK = [{"name": "op", "columns": [["q", {"categories": ["No", "Yes"]}]]}]


def write_meta(path, **meta):
    """Write a meta json defaulting to a single `op` block; any key (incl. structure) overrides."""
    json.dump({"structure": QBLOCK, **meta}, open(path, "w"))
    return str(path)


def make_wave(tmp_path, code, dates=None, extra_meta=None, n=4):
    """Write a small wave csv + meta; returns the meta path."""
    pd.DataFrame({"q": ["Yes", "No"] * (n // 2)}).to_csv(tmp_path / f"{code}.csv", index=False)
    return write_meta(tmp_path / f"{code}_meta.json", file=f"{code}.csv", **(dates or {}), **(extra_meta or {}))


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
        dates = {"collection_start": "2026-01-10", "collection_end": "2026-01-20", "collection_center": "2026-01-12"}
        m = make_wave(tmp_path, "w1", dates)
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
        pd.DataFrame({"q": ["Yes", "No"], "wave_time": ["a", "b"]}).to_csv(tmp_path / "w1.csv", index=False)
        block = [{"name": "b", "columns": ["q", ["wave_time", {"categories": ["a", "b"]}]]}]
        m = write_meta(tmp_path / "w1_meta.json", file="w1.csv", collection_center="2026-01-10", structure=block)
        df, m = read_annotated_data(m, return_meta=True)
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
        df, meta = read_and_process_data(
            {"files": [{"file": m2, "code": "LATE"}, {"file": m1, "code": "EARLY"}]}, return_meta=True
        )
        cats = meta.structure["waves"].columns["wave_time"].categories
        assert cats == ["2026-01-15", "2026-04-15"]
        assert list(df["wave_time"].cat.categories) == cats
        assert df["wave_time"].cat.ordered
        assert not meta.structure["waves"].hidden
        # per-row values follow each wave's own date
        assert (df.loc[df["file_code"] == "EARLY", "wave_time"] == "2026-01-15").all()
        assert (df.loc[df["file_code"] == "LATE", "wave_time"] == "2026-04-15").all()

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
    """The column name is reserved: a raw file's own is overwritten like file_code, a declared one is kept."""

    def test_raw_column_of_that_name_is_overwritten(self, tmp_path):
        """Whatever a raw file carries under the name - junk or years - is replaced by the meta's date."""
        pd.DataFrame({"q": ["Yes", "No"], "wave_time": ["wave 1", "2027"]}).to_csv(tmp_path / "d.csv", index=False)
        m = write_meta(tmp_path / "d_meta.json", file="d.csv", collection_center="2026-01-15")
        df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None and list(df["wave_time"].unique()) == ["2026-01-15"]

    def test_user_waves_block_columns_survive(self, tmp_path):
        """Injecting into a user block named 'waves' keeps its own columns and visibility."""
        pd.DataFrame({"q": ["Yes", "No"], "surf": [1, 2]}).to_csv(tmp_path / "d.csv", index=False)
        block = [{"name": "waves", "columns": [["q", {"categories": ["No", "Yes"]}], ["surf", {"continuous": True}]]}]
        m = write_meta(tmp_path / "d_meta.json", file="d.csv", collection_center="2026-01-15", structure=block)
        df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None
        assert {"q", "surf"} <= set(meta.structure["waves"].columns) and {"q", "surf"} <= set(df.columns)
        assert not meta.structure["waves"].hidden  # the user block's own visibility survives


class TestNestedMeta:
    """Parent meta over child metas."""

    def test_parent_preserves_child_dates(self, tmp_path):
        """Children's dates carry through a dateless parent."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-04-15"})
        files = [{"file": m2, "code": "A"}, {"file": m1, "code": "B"}]  # reverse chronological
        df, meta = read_annotated_data(write_meta(tmp_path / "p_meta.json", files=files), return_meta=True)
        assert meta is not None
        assert meta.structure["waves"].columns["wave_time"].categories == ["2026-01-15", "2026-04-15"]
        assert df["wave_time"].cat.ordered
        assert set(df["wave_time"]) == {"2026-01-15", "2026-04-15"}

    def test_parent_date_fills_dateless_child(self, tmp_path):
        """Parent's date fills a dateless child."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2")  # child without dates -> parent's date applies
        files = [{"file": m1, "code": "A"}, {"file": m2, "code": "B"}]
        df = read_annotated_data(write_meta(tmp_path / "p_meta.json", files=files, collection_center="2026-04-15"))
        assert set(df["wave_time"].dropna()) == {"2026-01-15", "2026-04-15"}


class TestUserOwnedColumns:
    """A wave_time declared or owned by the annotation is never rewritten or re-hidden."""

    def test_generated_block_declaring_it_suppresses_injection(self, tmp_path):
        """A hand-written `generated: true` block owning the column must not collide with injection."""
        pd.DataFrame({"q": ["Yes", "No"], "wave_time": ["2026-01-15"] * 2}).to_csv(tmp_path / "d.csv", index=False)
        mine = {"name": "mytime", "generated": True, "columns": [["wave_time", {"categories": ["2026-01-15"]}]]}
        m = write_meta(
            tmp_path / "d_meta.json", file="d.csv", collection_center="2026-01-15", structure=[*QBLOCK, mine]
        )
        df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None
        assert "waves" not in meta.structure and "wave_time" in df.columns

    def test_user_waves_block_keeps_its_visibility(self, tmp_path):
        """Merging into a user block named 'waves' must not flip the block's own `hidden`."""
        pd.DataFrame({"q": ["Yes", "No"]}).to_csv(tmp_path / "d.csv", index=False)
        block = [{"name": "waves", "hidden": False, "columns": [["q", {"categories": ["No", "Yes"]}]]}]
        m = write_meta(tmp_path / "d_meta.json", file="d.csv", collection_center="2026-01-15", structure=block)
        _, meta = read_and_process_data({"files": [{"file": m}]}, return_meta=True)
        blk = meta.structure["waves"]
        assert not blk.generated and not blk.hidden, "a user block's visibility was overwritten"

    def test_unparseable_collection_date_errors_clearly(self, tmp_path):
        """A free-text collection date names the offending fields rather than raising from pandas."""
        m = make_wave(tmp_path, "w1", {"collection_center": "sometime in spring"})
        with pytest.raises(ValueError, match="Unparseable collection date"):
            read_annotated_data(m)

    def test_wave_time_false_on_parent_drops_it(self, tmp_path):
        """Opting a combined dataset out beats the children's own dates."""
        m1 = make_wave(tmp_path, "w1", {"collection_center": "2026-01-15"})
        m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-04-15"})
        files = [{"file": m1, "code": "A"}, {"file": m2, "code": "B"}]
        m = write_meta(tmp_path / "p_meta.json", files=files, wave_time=False)
        df, meta = read_annotated_data(m, return_meta=True)
        assert meta is not None
        assert "waves" not in meta.structure and "wave_time" not in df.columns


def test_annotated_parquet_wave_gets_its_date(tmp_path):
    """A parquet source never runs `process()`, so its embedded collection dates are filled at load."""
    pd.DataFrame({"q": ["Yes", "No", "Yes"]}).to_csv(tmp_path / "w1.csv", index=False)
    # wave_time: false stands in for a parquet written before the column existed
    dates = {"collection_start": "2026-01-10", "collection_end": "2026-01-20"}
    m = write_meta(tmp_path / "w1_meta.json", file="w1.csv", wave_time=False, **dates)
    df, meta = read_annotated_data(m, return_meta=True)
    assert meta is not None and "wave_time" not in df.columns
    pq = str(tmp_path / "w1.parquet")
    write_parquet_with_metadata(df, {"data": meta.model_copy(update={"wave_time": True}).model_dump(mode="json")}, pq)

    m2 = make_wave(tmp_path, "w2", {"collection_center": "2026-04-15"})
    out = read_and_process_data({"files": [{"file": pq, "code": "P"}, {"file": m2, "code": "J"}]})
    assert out.loc[out["file_code"] == "P", "wave_time"].notna().all()
    assert set(out["wave_time"].dropna()) == {"2026-01-15", "2026-04-15"}
