"""Election Models
------------------

This module consolidates the helper functions, data structures, and Streamlit
widgets that used to live in `04_election_models.ipynb`.  It covers:

- mandate allocation helpers (d'Hondt, seat simulations, coalition calculators)
- deterministic + bootstrap election simulations exposed via
  `simulate_election`/`simulate_election_e2e`
- Altair helpers tailored to electoral visualisations (mandate plots,
  polarisation charts, etc.)

Keeping the commentary here means the file can be edited directly alongside the
rest of the library.
"""

__all__ = [
    "dhondt",
    "simulate_election",
    "vec_smallest_k",
    "cz_system",
    "simulate_election_e2e",
    "simulate_election_pp",
    "mandate_plot",
    "coalition_applet",
]

import itertools as it
from typing import Mapping, Sequence, cast

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from salk_toolkit import utils as stk_utils
from salk_toolkit.plots import stk_plot
from salk_toolkit.pp import AltairChart, PlotInput
from salk_toolkit.validation import ElectoralSystem

# --------------------------------------------------------
#          MACHINERY
# --------------------------------------------------------

# List of party names that can never be elected because they are just aggregates
# They still count towards Threshold computation but need to be ignored afterwards
party_aggregate_cat_names = ["Other"]


def dhondt(
    pvotes: np.ndarray, n_mandates: np.ndarray | int, dh_power: float = 1.0, pmand: np.ndarray | None = None
) -> np.ndarray:
    """Calculate d'Hondt method mandate allocation.

    Args:
        pvotes: Party votes array of shape (draws, parties).
        n_mandates: Number of mandates to allocate (scalar or array).
        dh_power: Power parameter for d'Hondt divisor (default: 1.0).
        pmand: Previously allocated mandates by party (default: zeros).

    Returns:
        Array of compensation mandates allocated to each party.
    """
    # Calculate d'Hondt values and get party indices out
    n_mandates = np.array(n_mandates)
    max_mandates = int(n_mandates.max())
    if pmand is None:
        pmand = np.zeros_like(pvotes)  # previously handed out mandates by party - zero by default
    dhvals = pvotes[:, :, None] / (pmand[:, :, None] + np.arange(1, max_mandates + 1, 1)[None, None, :]) ** dh_power
    sinds = np.argsort(-dhvals.reshape((dhvals.shape[0], -1)), axis=1) // max_mandates

    # Select the first n as compensation
    rmand = np.ones(pvotes.shape[0]) * n_mandates  # This can be a vector, one per draw
    ri = (np.arange(sinds.shape[1])[None, :] - rmand[:, None]) < 0
    comp_ident = np.concatenate([np.zeros((1, pvotes.shape[-1])), np.identity(pvotes.shape[-1])])
    return comp_ident[(sinds + 1) * ri].sum(axis=1)


# Vectorized basic election simulation: quotas, dHondt
# Input 'support' should be of shape (draws,districts,parties)


def simulate_election(
    support: np.ndarray,
    parties: list[str],
    districts: list[str],
    mandates: Mapping[str, int],
    es: ElectoralSystem,
) -> np.ndarray:
    """Simulate election with quota and d'Hondt allocation.

    Args:
        support: Support array of shape (draws, districts, parties).
        parties: List of party names in the same order as the last dimension of support.
        districts: List of district names in the same order as the second dimension of support.
        mandates: Dictionary mapping district names to mandate counts.
        es: ElectoralSystem object with election parameters.

    Returns:
        Array of mandates allocated per (draw, district, party).
    """
    if not isinstance(es, ElectoralSystem):
        es = ElectoralSystem.model_validate(es)

    # Convert mandates dict to array
    nmandates = np.array([mandates[d] for d in districts])

    # Convert exclude list (party names) to indices
    exclude_list: list[int] = []
    if es.exclude:
        exclude_list = [parties.index(p) for p in es.exclude if p in parties]

    # Handle threshold: can be float or dict of party->threshold
    if isinstance(es.threshold, dict):
        # Per-party thresholds: create array of thresholds for each party
        threshold_array = np.array([es.threshold.get(p, es.threshold.get("default", 0.0)) for p in parties])
        # Apply threshold per-party: shape (parties,)
        party_votes = support.sum(axis=1)  # (draws, parties)
        total_votes = support.sum(axis=(1, 2))  # (draws,)
        zero_mask = (party_votes / (total_votes[:, None] + 1e-3)) > threshold_array[None, :]
        zero_mask = zero_mask[:, None, :]  # (draws, 1, parties)
    else:
        # Single threshold for all parties
        threshold_val = es.threshold
        zero_mask = (support.sum(axis=1) / (support.sum(axis=(1, 2)) + 1e-3)[:, None]) > threshold_val
        zero_mask = zero_mask[:, None, :]  # (draws, 1, parties)

    # Exclude the parties in the exclude list
    # usually the grouping of "Other" that might otherwise exceed the threshold
    # if they were a single entity but they are not)
    exclude_zero_mask = np.ones(support.shape[-1])
    exclude_zero_mask[exclude_list] = 0
    uzsim_t = np.ones_like(support) * exclude_zero_mask

    # Apply threshold mask
    uzsim_t = zero_mask * support * uzsim_t

    # Remove parties below an electoral_district specific threshold
    zero_mask_ed = (support / (support.sum(axis=(2)) + 1e-3)[:, :, None]) > es.ed_threshold
    uzsim_t = zero_mask_ed * uzsim_t

    # Handle special systems
    if es.special == "cz":
        # For Czech system, use single threshold value (first party's or default)
        threshold_for_cz = (
            es.threshold
            if isinstance(es.threshold, float)
            else (es.threshold.get(parties[0], es.threshold.get("default", 0.0)) if parties else 0.0)
        )
        return cz_system(support, nmandates, threshold=threshold_for_cz, body_size=es.body_size)

    # Districts with quotas, then country-level compensation (Estonian system)
    if es.quotas:
        quota_values = (support.sum(axis=-1) + 1e-3) / (nmandates[None, :])
        v, r = np.divmod(uzsim_t / quota_values[:, :, None], 1.0)
        dmandates = v + (r >= es.first_quota_coef)

        # Calculate votes and mandates for each party
        pvotes = uzsim_t.sum(axis=1)
        pmand = dmandates.sum(axis=1)

        # Calculate compensation votes using dHondt
        body_size_val = es.body_size if es.body_size is not None else sum(nmandates)
        remaining_mand = body_size_val - pmand.sum(axis=1)
        comp_mandates = dhondt(pvotes, remaining_mand, es.dh_power, pmand)

        # Return the districts + compensation results
        return np.concatenate([dmandates, comp_mandates[:, None, :]], axis=1)

    else:  # Separate election in each district (Croatian system)
        return np.stack(
            [dhondt(uzsim_t[:, i, :], nmandates[i], es.dh_power) for i in range(support.shape[1])],
            axis=1,
        )


# Input a tensor t and a tensor kv of one less dimension
# Output a tensor of same shape as t with k ones marking the smallest values in t over the last axis
def vec_smallest_k(t: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """Mark k smallest values per row with ones.

    Args:
        t: Input tensor.
        kv: Number of smallest values to mark (one per row).

    Returns:
        Tensor with ones marking k smallest values per row.
    """
    # Create a vector with k ones followed by zeros
    rmand = np.ones(t.shape[:-1]) * kv
    ri = (np.arange(t.shape[-1])[None, :] - rmand[..., None]) < 0

    # Function that maps 0 to 0, and i+1 to the i-th unit vector
    comp_ident = np.concatenate([np.zeros((1, t.shape[-1])), np.identity(t.shape[-1])])

    # Marginalize that function over the newly created dimension
    return comp_ident[(np.argsort(t, axis=-1) + 1) * ri].sum(axis=-2)


def cz_system(
    support: np.ndarray, nmandates: np.ndarray, threshold: float = 0.0, body_size: int | None = None, **kwargs: object
) -> np.ndarray:
    """Czech electoral system based on https://pspen.psp.cz/chamber-members/plenary/elections/#electoralsystem.

    Simulate Czech electoral system with Imperialis quotas and two-level allocation.

    Args:
        support: Support array of shape (draws, districts, parties).
        nmandates: Number of mandates per district (array).
        threshold: National threshold for party eligibility (default: 0.0).
        body_size: Total body size for compensation (default: sum of nmandates).
        **kwargs: Additional arguments (unused).

    Returns:
        Array of mandates allocated per (draw, district, party).
    """
    # Remove parties below a national threshold
    zero_mask = (support.sum(axis=1) / (support.sum(axis=(1, 2)) + 1e-3)[:, None]) > threshold
    uzsim_t = zero_mask[:, None, :] * support

    # Districts with quotas, then country-level compensation
    # Imperialis quotas i.e. with divisor (n_mandates + 2)
    quotas = (support.sum(axis=-1) + 1e-3) / (nmandates[None, :] + 2)
    dmandates, r = np.divmod(uzsim_t / quotas[:, :, None], 1.0)

    # Deal with excess allocations
    excess = np.maximum(0, (dmandates.sum(axis=-1) - nmandates[None, :]))
    rp = r + (dmandates == 0)  # Increase residuals to remove mandates only from those that got any
    excess_dist = vec_smallest_k(rp, excess)
    dmandates -= excess_dist

    # Second level votes
    slvotes = ((r + excess_dist) * quotas[:, :, None]).sum(axis=1)  # Margin over e_d
    remaining_mand = body_size - dmandates.sum(axis=(1, 2))  # Margin over e_d and party

    # Second level quota
    slquotas = (slvotes.sum(axis=-1) + 1e-3) / (remaining_mand + 1)
    slmandates, r = np.divmod(slvotes / slquotas[:, None], 1.0)

    # Assign all seats using highest remainders
    missing = np.maximum(0, remaining_mand - slmandates.sum(axis=-1))
    slmandates += vec_smallest_k(-r, missing)

    # Checksums to make sure all mandates get allocated
    # print(list(dmandates.sum(axis=(1,2)) + slmandates.sum(axis=1)))

    # Return the districts + compensation results
    return np.concatenate([dmandates, slmandates[:, None, :]], axis=1)


def simulate_election_e2e(
    sdf: pd.DataFrame,
    parties: list[str],
    mandates_dict: dict[str, int],
    ed_col: str = "electoral_district",
    electoral_system: ElectoralSystem | Mapping[str, object] | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """End-to-end election simulation from DataFrame to DataFrame.

    Args:
        sdf: DataFrame with draws, electoral districts, and party support columns.
        parties: List of party column names.
        mandates_dict: Dictionary mapping electoral districts to mandate counts.
        ed_col: Name of electoral district column (default: "electoral_district").
        electoral_system: ElectoralSystem object. If None, creates one from kwargs.
        **kwargs: Additional arguments used to create ElectoralSystem if not provided.

    Returns:
        DataFrame with columns [draw, electoral_district, party, mandates].
    """
    # Convert data frame to a numpy tensor for fast vectorized processing
    parties = [p for p in parties if p in sdf.columns]
    ed_df = sdf.groupby(["draw", ed_col])[parties].sum()
    districts = list(sdf[ed_col].unique())
    support = ed_df.reset_index(drop=True).to_numpy().reshape((-1, len(districts), len(parties)))

    # Create ElectoralSystem from kwargs if not provided
    if electoral_system is None:
        # Handle exclude: convert from party names if provided
        exclude = kwargs.pop("exclude", party_aggregate_cat_names)
        if exclude and isinstance(exclude, list) and all(isinstance(x, str) for x in exclude):
            kwargs["exclude"] = exclude
        else:
            kwargs.setdefault("exclude", party_aggregate_cat_names)
        electoral_system = ElectoralSystem.model_validate(kwargs)
    elif not isinstance(electoral_system, ElectoralSystem):
        electoral_system = ElectoralSystem.model_validate(electoral_system)

    edt = simulate_election(support, parties, districts, mandates_dict, electoral_system)

    if edt.shape[1] > support.shape[1]:
        districts = districts + ["Compensation"]

    # Shape it back into a data frame
    eddf = pd.DataFrame(edt.reshape((-1,)), columns=["mandates"], dtype="int")
    eddf.loc[:, ["draw", ed_col, "party"]] = np.array(tuple(it.product(range(edt.shape[0]), districts, parties)))
    return eddf


# Factor = districts, Category = parties
def simulate_election_pp(
    data: pd.DataFrame,
    mandates: Mapping[str, int],
    electoral_system: ElectoralSystem | Mapping[str, object],
    cat_col: str,
    value_col: str,
    factor_col: str,
    cat_order: list[str],
    factor_order: list[str],
) -> pd.DataFrame:
    """Simulate election using plot pipeline data format.

    Args:
        data: DataFrame with draws, factor_col, cat_col, and value_col.
        mandates: Dictionary mapping factor values to mandate counts.
        electoral_system: Dictionary with electoral system parameters.
        cat_col: Name of category column (parties).
        value_col: Name of value column (support).
        factor_col: Name of factor column (districts).
        cat_order: Ordered list of category values.
        factor_order: Ordered list of factor values.

    Returns:
        DataFrame with columns [draw, factor_col, cat_col, mandates].
    """
    # Reshape input to (draws,electoral_districts,parties)
    draws = data.draw.unique()
    pdf = data.pivot(index=["draw", factor_col], columns=cat_col, values=value_col).reset_index()
    ded = pd.DataFrame(list(it.product(draws, factor_order)), columns=["draw", factor_col])
    sdata = (
        ded.merge(pdf, on=["draw", factor_col])
        .loc[:, cat_order]
        .fillna(0)
        .to_numpy()
        .reshape((len(draws), len(factor_order), len(cat_order)))
    )

    if not isinstance(electoral_system, ElectoralSystem):
        electoral_system = ElectoralSystem.model_validate(electoral_system)

    # Run the actual electoral simulation
    edt = simulate_election(sdata, cat_order, factor_order, mandates, electoral_system)
    if edt.shape[1] > sdata.shape[1]:
        factor_order = factor_order + ["Compensation"]

    # Shape it back into a data frame
    df = pd.DataFrame(edt.reshape((-1,)), columns=["mandates"])
    df.loc[:, ["draw", factor_col, cat_col]] = np.array(tuple(it.product(draws, factor_order, cat_order)))

    return df


# This fits into the pp framework as: f0['col']=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot(
    "mandate_plot",
    data_format="longform",
    draws=True,
    requires_factor=True,
    agg_fn="sum",
    n_facets=(2, 2),
    requires=[{}, {"mandates": "pass", "electoral_system": "pass"}],
    args={
        "mandates": "pass",
        "electoral_system": "pass",
        "value_col": "pass",
        "width": "pass",
        "alt_properties": "pass",
        "sim_done": "pass",
    },
    as_is=True,
    priority=-500,
)  # , hidden=True)
def mandate_plot(
    p: PlotInput,
    mandates: Mapping[str, int] | None = None,
    electoral_system: ElectoralSystem | Mapping[str, object] | None = None,
    value_col: str | None = None,
    width: int | None = None,
    alt_properties: Mapping[str, object] | None = None,
    sim_done: bool = False,
) -> AltairChart:
    """Create a mandate distribution visualization for election results.

    Args:
        p: Plot parameters bundle provided by the plot pipeline.
        mandates: Mapping between district names and mandate counts.
        electoral_system: Dictionary describing the electoral system.
        value_col: Optional override for the value column.
        width: Optional chart width override.
        alt_properties: Optional Altair property overrides.
        sim_done: Whether simulation has already been run.

    Returns:
        Altair chart showing mandate probability distributions.
    """
    data = p.data.copy()
    if len(p.facets) < 2:
        raise ValueError("mandate_plot requires at least two facets (party and district)")
    if p.outer_factors:
        raise ValueError("mandate_plot does not support outer factors")

    f0, f1 = p.facets[0], p.facets[1]
    party_col, district_col = f0.col, f1.col
    tf = p.translate or (lambda s: s)
    value_col = value_col or p.value_col
    width = width or p.width
    alt_props: dict[str, object] = dict(p.alt_properties)
    if alt_properties:
        alt_props.update(dict(alt_properties))

    if not sim_done:
        if mandates is None:
            raise ValueError("mandates must be provided when sim_done=False")
        translated_mandates = {tf(k): v for k, v in mandates.items()}
        if electoral_system is None:
            raise ValueError("electoral_system must be provided when sim_done=False")
        if not isinstance(electoral_system, ElectoralSystem):
            electoral_system = ElectoralSystem.model_validate(electoral_system)
        df = simulate_election_pp(
            data,
            translated_mandates,
            electoral_system,
            party_col,
            value_col,
            district_col,
            f0.order,
            f1.order,
        )
    else:
        df = data

    df[f1.col] = df[f1.col].replace({"Compensation": tf("Compensation")})

    # Shape it into % values for each vote count
    maxv = df["mandates"].max()
    tv = np.arange(1, maxv + 1, dtype="int")[None, :]
    dfv = df["mandates"].to_numpy()[:, None]
    dfm = pd.DataFrame((dfv >= tv).astype("int"), columns=tv[0], index=df.index)
    dfm["draw"], dfm[f0.col], dfm[f1.col] = (
        df["draw"],
        df[f0.col],
        df[f1.col],
    )
    res = (
        dfm.groupby([f0.col, f1.col], observed=True)[tv[0]]
        .mean()
        .reset_index()
        .melt(id_vars=[f0.col, f1.col], var_name="mandates", value_name="percent")
    )

    # Remove parties who have no chance of even one elector
    eliminate = cast(pd.Series, res.groupby(f0.col, observed=True)["percent"].sum()) < 0.2
    el_cols = [i for i, v in eliminate.items() if v]
    res = res[~res[f0.col].isin(el_cols)]
    cat_order = list(eliminate[~eliminate].index)

    f_width = max(50, width / len(cat_order)) if width is not None else 50

    plot = (
        alt.Chart(data=res)
        .mark_bar()
        .encode(
            x=alt.X("mandates", title=None),
            y=alt.Y("percent", title=None, axis=alt.Axis(format="%")),
            color=alt.Color(field=f0.col, type="nominal", scale=f0.colors, legend=None),
            tooltip=[
                alt.Tooltip(field=f0.col, title="party"),
                alt.Tooltip(field=f1.col),
                alt.Tooltip("mandates"),
                alt.Tooltip("percent", format=".1%", title="probability"),
            ],
        )
        .properties(
            width=f_width,
            height=f_width // 2,
            **alt_props,
            # title="Ringkonna- ja kompensatsioonimandaatide tõenäolised jaotused"
        )
        .facet(
            # header=alt.Header(labelAngle=-90),
            row=alt.X(
                f"{f1.col}:N",
                sort=list(f1.order) + [tf("Compensation")],
                title=None,
                header=alt.Header(labelOrient="top"),
            ),
            column=alt.Y(
                f"{f0.col}:N",
                sort=cat_order,
                title=None,
                header=alt.Header(labelFontWeight="bold"),
            ),
        )
    )
    return plot  # type: ignore[return-value]


# This fits into the pp framework as: f0['col']=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot(
    "coalition_applet",
    data_format="longform",
    draws=True,
    requires_factor=True,
    agg_fn="sum",
    args={
        "mandates": "pass",
        "electoral_system": "pass",
        "value_col": "pass",
        "initial_coalition": "list",
        "sim_done": "pass",
    },
    requires=[{}, {"mandates": "pass", "electoral_system": "pass"}],
    as_is=True,
    n_facets=(2, 2),
    priority=-1000,
)  # , hidden=True)
def coalition_applet(
    p: PlotInput,
    mandates: Mapping[str, int] | None = None,
    electoral_system: ElectoralSystem | Mapping[str, object] | None = None,
    value_col: str | None = None,
    initial_coalition: Sequence[str] | None = None,
    sim_done: bool = False,
) -> None:
    """Interactive Streamlit widget for exploring coalition scenarios.

    Args:
        p: Plot parameters with data/metadata supplied by the pipeline.
        mandates: Mapping of districts to mandate counts (required if sim_done=False).
        electoral_system: Electoral system definition (ElectoralSystem or dict).
        value_col: Optional override for vote column.
        initial_coalition: Sequence of parties pre-selected in the widget.
        sim_done: Whether simulation has already been run.
    """
    tf = p.translate or (lambda s: s)
    initial_coalition = initial_coalition or []
    value_col = value_col or p.value_col

    if p.outer_factors:
        raise ValueError("coalition_applet does not support outer factors")
    if len(p.facets) < 2:
        raise ValueError("coalition_applet expects two facets (party and district)")
    f0, f1 = p.facets[0], p.facets[1]
    party_col, district_col = f0.col, f1.col

    if not sim_done:
        if mandates is None or electoral_system is None:
            raise ValueError("mandates and electoral_system must be provided when sim_done=False")
        if not isinstance(electoral_system, ElectoralSystem):
            electoral_system = ElectoralSystem.model_validate(electoral_system)
        mandates_dict = {tf(k): v for k, v in mandates.items()}
        sdf = simulate_election_pp(
            p.data,
            mandates_dict,
            electoral_system,
            party_col,
            value_col or p.value_col,
            district_col,
            f0.order,
            f1.order,
        )
    else:
        sdf = p.data
        if mandates is None:
            mandates_meta = getattr(f1.meta, "mandates", None)
            mandates_dict = dict(mandates_meta or {})
        else:
            mandates_dict = {tf(k): v for k, v in mandates.items()}

    # Aggregate to total mandate counts
    odf = sdf.groupby(["draw", party_col])["mandates"].sum().reset_index()
    odf["over_t"] = odf["mandates"] > 0
    adf = odf[odf["mandates"] > 0]

    list(adf[party_col].unique())  # Leave only parties that have mandates

    coalition = st.multiselect(
        tf("Select the coalition:"),
        f0.order,
        default=initial_coalition,
        help=tf("Choose the parties whose coalition to model"),
    )

    st.markdown("""___""")

    col1, col2 = st.columns((9, 9), gap="large")
    col1.markdown(tf("**Party mandate distributions**"))

    # Individual parties plot
    ddf = (
        (adf.groupby(party_col)["mandates"].value_counts() / odf.groupby(party_col).size())
        .rename("percent")
        .reset_index()
    )
    ddf = ddf.merge(
        cast(pd.Series, odf.groupby(party_col)["mandates"].median()).rename(str(tf("median"))),
        left_on=party_col,
        right_index=True,
    )
    ddf = ddf.merge(
        cast(pd.Series, odf.groupby(party_col)["over_t"].mean()).rename(str(tf("over_threshold"))),
        left_on=party_col,
        right_index=True,
    )

    p_plot = (
        alt.Chart(
            ddf,
            # title=var
        )
        .mark_rect(opacity=0.8, stroke="black", strokeWidth=0)
        .transform_calculate(x1="datum.mandates - 0.45", x2="datum.mandates + 0.45")
        .encode(
            alt.X(
                "x1:Q",
                title=tf("mandates"),
                axis=alt.Axis(tickMinStep=1),
                scale=alt.Scale(domainMin=0),
            ),
            alt.X2("x2:Q"),
            alt.Y("percent:Q", title=None, axis=None),
            alt.Row(f"{party_col}:N", title=None),
            color=alt.Color(f"{party_col}:N", legend=None, scale=f0.colors),
            tooltip=[
                alt.Tooltip("mandates:Q", title=tf("mandates"), format=",d"),
                alt.Tooltip("percent:Q", title=tf("percent"), format=".1%"),
                alt.Tooltip(tf("median"), format=",d"),
                alt.Tooltip(tf("over_threshold"), format=".1%"),
            ],
        )
        .properties(height=60)
    )
    col1.altair_chart(p_plot, use_container_width=True)

    total_mandates = sum(mandates_dict.values())

    col2.markdown(tf("**Coalition simulation**"))
    n = col2.number_input(
        tf("Choose mandate cutoff:"),
        min_value=0,
        max_value=total_mandates,
        value=(total_mandates // 2) + 1,
        step=1,
        help="...",
    )

    if len(coalition) > 0:
        # Coalition plot
        acdf = adf[adf[party_col].isin(coalition)]
        cdf = acdf.groupby("draw")["mandates"].sum().value_counts().rename("count").reset_index()

        mi, ma = min(cdf["mandates"].min(), n), max(cdf["mandates"].max(), n)
        tick_count = (
            ma - mi + 1
        )  # This is the only way to enforce integer ticks as tickMinStep seems to not do it sometimes
        k_plot = (
            alt.Chart(cdf)
            .mark_rect(color="#ff2b2b")
            .transform_calculate(x1="datum.mandates - 0.45", x2="datum.mandates + 0.45")
            .encode(
                x=alt.X(
                    "x1:Q",
                    title=tf("mandates"),
                    axis=alt.Axis(tickMinStep=1, tickCount=tick_count),
                    scale=alt.Scale(domain=[mi, ma]),
                ),
                x2=alt.X2("x2:Q"),
                y=alt.Y("count:Q", title=None, stack=None, axis=None),
            )
            .properties(height=200, width=300)
        )
        rule = alt.Chart(pd.DataFrame({"x": [n]})).mark_rule(color="silver", size=1.25, strokeDash=[5, 2]).encode(x="x")
        col2.altair_chart((k_plot + rule).configure_view(strokeWidth=0), use_container_width=True)

        threshold_prob = float(cdf[cdf["mandates"] >= n]["count"].sum() / cdf["count"].sum())

        col2.write(
            stk_utils.unescape_vega_label(tf("Probability of at least  **{0}** mandates: **{1:.1%}**")).format(
                n, threshold_prob
            )
        )
        # col3.write('Distributsiooni mediaan: **{:d}**'.format(int((d_dist[koalitsioon].sum(1)).median())))
        # m, l, h = hdi(sim_data['riigikogu'][koalitsioon], 0.9)
        # col2.write('Distributsiooni mediaan on **{:.0f}** mandaati. '
        #            '90% tõenäosusega jääb mandaatide arv **{:.0f}** ning **{:.0f}** vahele.'.format(m, l, h))

    return None
