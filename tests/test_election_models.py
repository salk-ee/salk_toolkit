"""Unit tests for election modeling utilities."""

import altair as alt
import numpy as np
import pandas as pd
import pytest

from salk_toolkit.election_models import (
    cz_system,
    coalition_applet,
    dhondt,
    simulate_election,
    simulate_election_e2e,
    simulate_election_pp,
    vec_smallest_k,
)
from salk_toolkit.pp import FacetMeta, PlotInput
from salk_toolkit.validation import ColumnMeta, ElectoralSystem


def _make_plot_params(data: pd.DataFrame, mandates: dict[str, int]) -> PlotInput:
    facets = [
        FacetMeta(
            col="party",
            ocol="party",
            order=["Alpha", "Beta"],
            colors=alt.Scale(domain=["Alpha", "Beta"], range=["#c00", "#00c"]),
            neutrals=[],
            meta=ColumnMeta(categories=["Alpha", "Beta"]),
        ),
        FacetMeta(
            col="district",
            ocol="district",
            order=["North", "South"],
            colors=alt.Scale(domain=["North", "South"], range=["#090", "#990"]),
            neutrals=[],
            meta=ColumnMeta(categories=["North", "South"], mandates=mandates),
        ),
    ]
    return PlotInput(
        data=data,
        col_meta={},
        value_col="value",
        facets=facets,
        translate=lambda s: s,
        tooltip=[],
        alt_properties={},
        outer_factors=[],
        plot_args={},
    )


def test_dhondt_allocates_expected_counts() -> None:
    """d'Hondt should give more seats to higher vote getter."""
    pvotes = np.array([[100.0, 50.0]])
    seats = dhondt(pvotes, np.array([2]))
    assert seats.shape == (1, 2)
    assert seats.sum() == 2
    assert seats[0, 0] > seats[0, 1]


def test_simulate_election_quota_sum() -> None:
    """simulate_election should return expected mandate totals."""
    support = np.array([[[10.0, 5.0]]])  # draws=1, districts=1, parties=2
    parties = ["PartyA", "PartyB"]
    districts = ["District1"]
    mandates = {"District1": 2}
    result = simulate_election(support, parties, districts, mandates, ElectoralSystem(threshold=0.0))
    assert result.shape == (1, 2, 2)  # district + compensation row
    assert result.sum() == 2


def test_simulate_election_variable_thresholds() -> None:
    """simulate_election should apply per-party thresholds correctly."""
    # Setup: 3 parties with different thresholds
    # Total votes: 100
    # PartyA: 50 votes (50%), threshold 10% -> should pass (50% > 10%)
    # PartyB: 15 votes (15%), threshold 20% -> should fail (15% < 20%)
    # PartyC: 35 votes (35%), threshold 5% -> should pass (35% > 5%)
    # This setup ensures PartyB (15%) is between PartyA's threshold (10%) and PartyB's threshold (20%)
    # If code incorrectly used PartyA's threshold (10%) for all, PartyB would pass (15% > 10%)
    # But with correct per-party thresholds, PartyB should fail (15% < 20%)
    parties = ["PartyA", "PartyB", "PartyC"]
    support = np.array([[[50.0, 15.0, 35.0]]])  # draws=1, districts=1, parties=3

    # Variable thresholds: different for each party
    thresholds = {
        "PartyA": 0.10,  # 10% threshold
        "PartyB": 0.20,  # 20% threshold (PartyB will fail with 15%)
        "PartyC": 0.05,  # 5% threshold
    }
    districts = ["District1"]
    mandates = {"District1": 5}
    es = ElectoralSystem(threshold=thresholds)

    result = simulate_election(support, parties, districts, mandates, es)

    # PartyB should be excluded (below 20% threshold)
    # Only PartyA and PartyC should get mandates
    assert result.shape == (1, 2, 3)  # district + compensation row, 3 parties

    # PartyB (index 1) should have zero mandates in both district and compensation
    party_b_total = result[0, :, 1].sum()
    assert party_b_total == 0, (
        f"PartyB should be excluded from mandates (15% < 20% threshold). "
        f"Got {party_b_total} mandates. "
        f"If this fails, the code might be using only the first threshold value (10%) instead of per-party thresholds. "
        f"With 10% threshold, PartyB would incorrectly pass (15% > 10%)."
    )

    # PartyA and PartyC should get some mandates (non-zero)
    # Since PartyB is excluded, PartyA and PartyC split the 5 mandates
    party_a_total = result[0, :, 0].sum()
    party_c_total = result[0, :, 2].sum()
    assert party_a_total > 0, "PartyA should get mandates (50% > 10% threshold)"
    assert party_c_total > 0, "PartyC should get mandates (35% > 5% threshold)"

    # Total mandates should still equal nmandates (5)
    assert result.sum() == 5, f"Total mandates should equal nmandates (5), got {result.sum()}"

    # Verify PartyA and PartyC together get all 5 mandates
    assert party_a_total + party_c_total == 5, (
        f"PartyA and PartyC should together get all 5 mandates. Got PartyA: {party_a_total}, PartyC: {party_c_total}"
    )


def test_vec_smallest_k_marks_positions() -> None:
    """Mark the k smallest entries across the tensor."""
    tensor = np.array([[3, 1, 2]])
    split = np.array([2])
    marked = vec_smallest_k(tensor, split)
    assert marked.shape == tensor.shape
    assert marked.sum() == 2
    assert np.all(marked[0, 1:] >= marked[0, :1])


def test_cz_system_returns_comp_layer() -> None:
    """Czech system should output compensation layer."""
    support = np.array([[[20.0, 10.0]]])
    result = cz_system(support, np.array([2]), threshold=0.0, body_size=2)
    assert result.shape == (1, 2, 2)


def test_simulate_election_pp_accepts_electoral_system() -> None:
    """simulate_election_pp should respect provided ElectoralSystem."""
    df = pd.DataFrame(
        {
            "draw": [0, 0, 0, 0],
            "district": ["North", "North", "South", "South"],
            "party": ["Alpha", "Beta", "Alpha", "Beta"],
            "value": [100, 80, 60, 40],
        }
    )
    mandates = {"North": 1, "South": 1}
    es = ElectoralSystem()
    result = simulate_election_pp(
        df,
        mandates,
        es,
        cat_col="party",
        value_col="value",
        factor_col="district",
        cat_order=["Alpha", "Beta"],
        factor_order=["North", "South"],
    )
    assert set(result.columns) == {"draw", "district", "party", "mandates"}
    assert set(result["district"]) == {"North", "South", "Compensation"}


def test_simulate_election_e2e_returns_expected_columns() -> None:
    """End-to-end simulation should produce draw/party/mandate columns."""
    sdf = pd.DataFrame(
        {
            "draw": [0, 0],
            "electoral_district": ["North", "North"],
            "Alpha": [100, 80],
            "Beta": [70, 60],
        }
    )
    result = simulate_election_e2e(sdf, parties=["Alpha", "Beta"], mandates_dict={"North": 2})
    assert set(result.columns) == {"draw", "electoral_district", "party", "mandates"}
    assert set(result["electoral_district"]) == {"North", "Compensation"}


def test_coalition_applet_sim_done_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """coalition_applet should render without re-running simulation."""
    data = pd.DataFrame(
        {
            "draw": [0, 0, 0, 0],
            "party": ["Alpha", "Beta", "Alpha", "Beta"],
            "district": ["North", "North", "South", "South"],
            "mandates": [1, 0, 0, 1],
        }
    )
    params = _make_plot_params(data, {"North": 1, "South": 1})

    class _DummyCol:
        def markdown(self, *args, **kwargs) -> None:
            pass

        def altair_chart(self, *args, **kwargs) -> None:
            pass

        def number_input(self, *args, **kwargs) -> int:
            return kwargs.get("value", 0)

        def write(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.multiselect", lambda *a, **k: [])

    def _columns(*args, **kwargs):
        return [_DummyCol(), _DummyCol()]

    monkeypatch.setattr("streamlit.columns", _columns)

    coalition_applet(
        p=params,
        mandates={"North": 1, "South": 1},
        electoral_system=ElectoralSystem(),
        sim_done=True,
    )
