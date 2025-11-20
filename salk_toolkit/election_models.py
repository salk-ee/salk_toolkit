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

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from salk_toolkit import utils as stk_utils
from salk_toolkit.plots import stk_plot

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
    nmandates: np.ndarray,
    threshold: float = 0.0,
    ed_threshold: float = 0.0,
    quotas: bool = True,
    first_quota_coef: float = 1.0,
    dh_power: float = 1.0,
    body_size: int | None = None,
    special: str | None = None,
    exclude: list[int] | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Simulate election with quota and d'Hondt allocation.

    Args:
        support: Support array of shape (draws, districts, parties).
        nmandates: Number of mandates per district (array).
        threshold: National threshold for party eligibility (default: 0.0).
        ed_threshold: Electoral district threshold (default: 0.0).
        quotas: Whether to use quota system (default: True).
        first_quota_coef: Coefficient for first quota allocation (default: 1.0).
        dh_power: Power parameter for d'Hondt divisor (default: 1.0).
        body_size: Total body size for compensation (default: sum of nmandates).
        special: Special system identifier (e.g., "cz" for Czech system).
        exclude: List of party indices to exclude from allocation.
        **kwargs: Additional arguments passed to special systems.

    Returns:
        Array of mandates allocated per (draw, district, party).
    """
    if exclude is None:
        exclude = []
    if special == "cz":
        return cz_system(support, nmandates, threshold=threshold, body_size=body_size, **kwargs)

    # Exclude the parties in the exclude list.
    # usually the grouping of "Other" that might otherwise exceed the threshold
    # if they were a single entity but they are not)
    exclude_zero_mask = np.ones(support.shape[-1])
    exclude_zero_mask[exclude] = 0
    uzsim_t = np.ones_like(support) * exclude_zero_mask

    # Remove parties below a national threshold
    zero_mask = (support.sum(axis=1) / (support.sum(axis=(1, 2)) + 1e-3)[:, None]) > threshold
    uzsim_t = zero_mask[:, None, :] * support * uzsim_t

    # Remove parties below an electoral_district specific threshold
    zero_mask = (support / (support.sum(axis=(2)) + 1e-3)[:, :, None]) > ed_threshold
    uzsim_t = zero_mask[:, :, :] * uzsim_t

    # Districts with quotas, then country-level compensation (Estonian system)
    if quotas:
        quotas = (support.sum(axis=-1) + 1e-3) / (nmandates[None, :])
        v, r = np.divmod(uzsim_t / quotas[:, :, None], 1.0)
        dmandates = v + (r >= first_quota_coef)

        # Calculate votes and mandates for each party
        pvotes = uzsim_t.sum(axis=1)
        pmand = dmandates.sum(axis=1)

        # Calculate compensation votes using dHondt
        if body_size is None:
            body_size = sum(nmandates)
        remaining_mand = body_size - pmand.sum(axis=1)
        comp_mandates = dhondt(pvotes, remaining_mand, dh_power, pmand)

        # Return the districts + compensation results
        return np.concatenate([dmandates, comp_mandates[:, None, :]], axis=1)

    else:  # Separate election in each district (Croatian system)
        return np.stack(
            [dhondt(uzsim_t[:, i, :], nmandates[i], dh_power) for i in range(support.shape[1])],
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


# Czech electoral system based on https://pspen.psp.cz/chamber-members/plenary/elections/#electoralsystem
def cz_system(
    support: np.ndarray, nmandates: np.ndarray, threshold: float = 0.0, body_size: int | None = None, **kwargs: object
) -> np.ndarray:
    """Simulate Czech electoral system with Imperialis quotas and two-level allocation.

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


# Basic wrapper around simulate elections that goes from dataframe to dataframe
def simulate_election_e2e(
    sdf: pd.DataFrame,
    parties: list[str],
    mandates_dict: dict[str, int],
    ed_col: str = "electoral_district",
    **kwargs: object,
) -> pd.DataFrame:
    """End-to-end election simulation from DataFrame to DataFrame.

    Args:
        sdf: DataFrame with draws, electoral districts, and party support columns.
        parties: List of party column names.
        mandates_dict: Dictionary mapping electoral districts to mandate counts.
        ed_col: Name of electoral district column (default: "electoral_district").
        **kwargs: Additional arguments passed to simulate_election.

    Returns:
        DataFrame with columns [draw, electoral_district, party, mandates].
    """
    # Convert data frame to a numpy tensor for fast vectorized processing
    parties = [p for p in parties if p in sdf.columns]
    ed_df = sdf.groupby(["draw", ed_col])[parties].sum()
    districts = list(sdf.electoral_district.unique())
    support = ed_df.reset_index(drop=True).to_numpy().reshape((-1, len(districts), len(parties)))
    nmandates = np.array([mandates_dict[d] for d in districts])

    # Translate exclusion list to party indices (and add the proper default)
    kwargs["exclude"] = [parties.index(p) for p in kwargs.get("exclude", party_aggregate_cat_names) if p in parties]

    edt = simulate_election(support, nmandates, **kwargs)

    if edt.shape[1] > support.shape[1]:
        districts = districts + ["Compensation"]

    # Shape it back into a data frame
    eddf = pd.DataFrame(edt.reshape((-1,)), columns=["mandates"], dtype="int")
    eddf.loc[:, ["draw", ed_col, "party"]] = np.array(tuple(it.product(range(edt.shape[0]), districts, parties)))
    return eddf


# Factor = districts, Category = parties
def simulate_election_pp(
    data: pd.DataFrame,
    mandates: dict[str, int],
    electoral_system: dict[str, object],
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

    if isinstance(electoral_system.get("threshold"), dict):
        td = electoral_system["threshold"]
        electoral_system["threshold"] = np.array([td[d] if d in td else td["default"] for d in cat_order])
        print(td, electoral_system["threshold"])

    # Translate exclusion list to party indices (and add the proper default)
    electoral_system["exclude"] = [
        cat_order.index(p) for p in electoral_system.get("exclude", party_aggregate_cat_names) if p in cat_order
    ]

    # Run the actual electoral simulation
    nmandates = np.array([mandates[d] for d in factor_order])
    edt = simulate_election(sdata, nmandates, **electoral_system)
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
    as_is=True,
    priority=-500,
)  # , hidden=True)
def mandate_plot(
    data: pd.DataFrame,
    mandates: dict[str, int],
    electoral_system: str | object,
    value_col: str = "value",
    facets: list[dict[str, object]] = [],
    width: int | None = None,
    alt_properties: dict[str, object] = {},
    outer_factors: list[str] = [],
    translate: object | None = None,
    sim_done: bool = False,
) -> alt.Chart:
    """Create a mandate distribution visualization for election results.

    Args:
        data: Election data with party support by draw and facets.
        mandates: Dictionary mapping regions to number of mandates to allocate.
        electoral_system: Name of electoral system or custom function.
        value_col: Column name containing vote values.
        facets: List of facet specifications (party, region, etc).
        width: Chart width in pixels.
        alt_properties: Additional Altair chart properties.
        outer_factors: Additional grouping factors.
        translate: Translation function for labels.
        sim_done: Whether simulation has already been run.

    Returns:
        Altair chart showing mandate probability distributions.
    """
    f0, f1 = facets[0], facets[1]
    tf = translate if translate else (lambda s: s)

    if outer_factors:
        raise Exception("This plot does not work with extra factors")

    if not sim_done:
        mandates = {tf(k): v for k, v in mandates.items()}
        df = simulate_election_pp(
            data,
            mandates,
            electoral_system,
            f0["col"],
            value_col,
            f1["col"],
            f0["order"],
            f1["order"],
        )
    else:
        df = data

    df[f1["col"]] = df[f1["col"]].replace({"Compensation": tf("Compensation")})

    # Shape it into % values for each vote count
    maxv = df["mandates"].max()
    tv = np.arange(1, maxv + 1, dtype="int")[None, :]
    dfv = df["mandates"].to_numpy()[:, None]
    dfm = pd.DataFrame((dfv >= tv).astype("int"), columns=tv[0], index=df.index)
    dfm["draw"], dfm[f0["col"]], dfm[f1["col"]] = (
        df["draw"],
        df[f0["col"]],
        df[f1["col"]],
    )
    res = (
        dfm.groupby([f0["col"], f1["col"]], observed=True)[tv[0]]
        .mean()
        .reset_index()
        .melt(id_vars=[f0["col"], f1["col"]], var_name="mandates", value_name="percent")
    )

    # Remove parties who have no chance of even one elector
    eliminate = res.groupby(f0["col"], observed=True)["percent"].sum() < 0.2
    el_cols = [i for i, v in eliminate.items() if v]
    res = res[~res[f0["col"]].isin(el_cols)]
    cat_order = list(eliminate[~eliminate].index)

    f_width = max(50, width / len(cat_order))

    plot = (
        alt.Chart(data=res)
        .mark_bar()
        .encode(
            x=alt.X("mandates", title=None),
            y=alt.Y("percent", title=None, axis=alt.Axis(format="%")),
            color=alt.Color(f"{f0['col']}:N", scale=f0["colors"], legend=None),
            tooltip=[
                alt.Tooltip(f0["col"], title="party"),
                alt.Tooltip(f1["col"]),
                alt.Tooltip("mandates"),
                alt.Tooltip("percent", format=".1%", title="probability"),
            ],
        )
        .properties(
            width=f_width,
            height=f_width // 2,
            **alt_properties,
            # title="Ringkonna- ja kompensatsioonimandaatide tõenäolised jaotused"
        )
        .facet(
            # header=alt.Header(labelAngle=-90),
            row=alt.X(
                f"{f1['col']}:N",
                sort=f1["order"] + [tf("Compensation")],
                title=None,
                header=alt.Header(labelOrient="top"),
            ),
            column=alt.Y(
                f"{f0['col']}:N",
                sort=cat_order,
                title=None,
                header=alt.Header(labelFontWeight="bold"),
            ),
        )
    )
    return plot


# This fits into the pp framework as: f0['col']=party_pref, factor=electoral_district, hence the as_is and hidden flags
@stk_plot(
    "coalition_applet",
    data_format="longform",
    draws=True,
    requires_factor=True,
    agg_fn="sum",
    args={"initial_coalition": "list"},
    requires=[{}, {"mandates": "pass", "electoral_system": "pass"}],
    as_is=True,
    n_facets=(2, 2),
    priority=-1000,
)  # , hidden=True)
def coalition_applet(
    data: pd.DataFrame,
    mandates: dict[str, int],
    electoral_system: str | object,
    value_col: str = "value",
    facets: list[dict[str, object]] = [],
    width: int | None = None,
    alt_properties: dict[str, object] = {},
    outer_factors: list[str] = [],
    translate: object | None = None,
    initial_coalition: list[str] = [],
    sim_done: bool = False,
) -> None:
    """Interactive Streamlit widget for exploring coalition scenarios.

    Args:
        data: Election data with party support by draw and facets.
        mandates: Dictionary mapping regions to number of mandates to allocate.
        electoral_system: Name of electoral system or custom function.
        value_col: Column name containing vote values.
        facets: List of facet specifications (party, region, etc).
        width: Chart width in pixels.
        alt_properties: Additional Altair chart properties.
        outer_factors: Additional grouping factors.
        translate: Translation function for labels.
        initial_coalition: Initial list of parties in coalition.
        sim_done: Whether simulation has already been run.
    """
    f0, f1 = facets[0], facets[1]
    tf = translate if translate else (lambda s: s)

    if outer_factors:
        raise Exception("This plot does not work with extra factors")

    mandates = {tf(k): v for k, v in mandates.items()}

    if not sim_done:
        sdf = simulate_election_pp(
            data,
            mandates,
            electoral_system,
            f0["col"],
            value_col,
            f1["col"],
            f0["order"],
            f1["order"],
        )
    else:
        sdf = data

    # Aggregate to total mandate counts
    odf = sdf.groupby(["draw", f0["col"]])["mandates"].sum().reset_index()
    odf["over_t"] = odf["mandates"] > 0
    adf = odf[odf["mandates"] > 0]

    list(adf[f0["col"]].unique())  # Leave only parties that have mandates

    coalition = st.multiselect(
        tf("Select the coalition:"),
        f0["order"],
        default=initial_coalition,
        help=tf("Choose the parties whose coalition to model"),
    )

    st.markdown("""___""")

    col1, col2 = st.columns((9, 9), gap="large")
    col1.markdown(tf("**Party mandate distributions**"))

    # Individual parties plot
    ddf = (
        (adf.groupby(f0["col"])["mandates"].value_counts() / odf.groupby(f0["col"]).size())
        .rename("percent")
        .reset_index()
    )
    ddf = ddf.merge(
        odf.groupby(f0["col"])["mandates"].median().rename(tf("median")),
        left_on=f0["col"],
        right_index=True,
    )
    ddf = ddf.merge(
        odf.groupby(f0["col"])["over_t"].mean().rename(tf("over_threshold")),
        left_on=f0["col"],
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
            alt.Row(f"{f0['col']}:N", title=None),
            color=alt.Color(f"{f0['col']}:N", legend=None, scale=f0["colors"]),
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

    total_mandates = sum(mandates.values())

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
        acdf = adf[adf[f0["col"]].isin(coalition)]
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
