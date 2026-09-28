"""
Cohorts page — Cohort retention, revenue retention, maturity, and reactivation analysis.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_section_label,
    render_kpi_row,
    render_science_card,
    render_missing_data,
    render_download_button,
)
from app_charts import (
    plot_retention_heatmap,
    plot_retention_decay,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import format_count, format_retention, format_nrr_index, format_percent


def render_cohorts_page() -> None:
    """Render the Cohorts page."""
    registry = get_registry()

    logo_file = registry.load_dataframe("cohorts", "matrix_logo_retention")
    nrr_file = registry.load_dataframe("cohorts", "matrix_net_revenue_retention")
    grr_file = registry.load_dataframe("cohorts", "matrix_gross_revenue_retention")
    decay = registry.load_dataframe("cohorts", "retention_decay_curve")
    scorecard = registry.load_dataframe("cohorts", "cohort_scorecard")
    acquisition_quality = registry.load_dataframe("cohorts", "cohort_acquisition_quality")
    lifecycle = registry.load_dataframe("cohorts", "customer_lifecycle_status")

    if all(x is None for x in [logo_file, nrr_file, decay, scorecard]):
        st.warning("No cohort outputs found. Run cohort_analysis.py first.")
        st.stop()

    # Page hero
    render_page_hero(
        title="Cohort Intelligence",
        description="Read acquisition quality through retention, revenue persistence, and reactivation over customer age.",
        kicker="LIFECYCLE INTELLIGENCE",
    )

    # ---- KPIs ----
    n_cohorts = None
    if scorecard is not None:
        cohort_col = "cohort_month" if "cohort_month" in scorecard.columns else "cohort"
        if cohort_col:
            n_cohorts = scorecard[cohort_col].nunique()

    # Month-3 retention from decay curve
    ret_3 = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        ret_col = "weighted_logo_retention" if "weighted_logo_retention" in decay.columns else "logo_retention"
        if age_col and ret_col:
            match = decay.loc[pd.to_numeric(decay[age_col], errors="coerce") == 3, ret_col]
            if not match.empty:
                ret_3 = float(pd.to_numeric(match, errors="coerce").iloc[0])

    # Month-6 NRR
    nrr_6 = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        nrr_col = "weighted_net_revenue_retention" if "weighted_net_revenue_retention" in decay.columns else "net_revenue_retention"
        if age_col and nrr_col:
            match = decay.loc[pd.to_numeric(decay[age_col], errors="coerce") == 6, nrr_col]
            if not match.empty:
                nrr_6 = float(pd.to_numeric(match, errors="coerce").iloc[0])

    # Observed age horizon
    max_age = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        if age_col:
            max_age = int(pd.to_numeric(decay[age_col], errors="coerce").max())

    kpis = [
        {"label": "Acquisition Cohorts", "value": format_count(n_cohorts)},
        {"label": "Month-3 Logo Retention", "value": format_retention(ret_3) if ret_3 is not None else "—"},
        {"label": "Month-6 Net Revenue Retention", "value": format_nrr_index(nrr_6) if nrr_6 is not None else "—"},
        {"label": "Observed Age Horizon", "value": format_count(max_age) if max_age is not None else "—"},
    ]
    render_kpi_row(kpis, columns=4)

    # ---- TABS ----
    tabs = st.tabs([
        "Logo Retention",
        "Revenue Retention",
        "Maturity Curve",
        "Cohort Scorecard",
        "Acquisition Quality",
    ])

    with tabs[0]:
        render_logo_retention_tab(logo_file, nrr_file)

    with tabs[1]:
        render_revenue_retention_tab(nrr_file, grr_file)

    with tabs[2]:
        render_maturity_curve_tab(decay)

    with tabs[3]:
        render_scorecard_tab(scorecard)

    with tabs[4]:
        render_acquisition_quality_tab(acquisition_quality)

    # ---- NRR SEMANTICS NOTE ----
    st.markdown("---")
    render_science_card(
        "Net Revenue Retention Index Semantics",
        "Values above 100% are possible because surviving customers can spend more than the cohort did in its acquisition month. "
        "This is a **retail revenue index**, not a SaaS contractual NRR. "
        "Do not cap values at 100% — the index correctly reflects revenue expansion among retained customers.",
    )


def render_logo_retention_tab(logo_file: pd.DataFrame, nrr_file: pd.DataFrame) -> None:
    """Render logo retention heatmap."""
    st.markdown("#### Logo Retention Cohort Matrix")

    if logo_file is not None:
        heat = prepare_retention_matrix(logo_file)
        if not heat.empty:
            cohorts = heat.index.tolist()
            ages = heat.columns.tolist()

            fig = plot_retention_heatmap(
                heat.values,
                cohorts=cohorts,
                ages=ages,
                title="Logo Retention Cohort Matrix",
                height=max(500, 220 + 24 * min(25, len(heat))),
                cmap="Blues",
                zmin=0,
                zmax=1,
            )
            render_chart_responsive(fig, "cohort_logo")

            # Legend for immature cohorts
            st.caption("🟫 Light gray cells = not yet observable at this maturity (right-censored).")
        else:
            render_missing_data("Logo retention matrix is empty.")
    else:
        render_missing_data("Logo retention matrix not available.")


def render_revenue_retention_tab(nrr_file: pd.DataFrame, grr_file: pd.DataFrame) -> None:
    """Render revenue retention heatmaps."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Net Revenue Retention Index")
        if nrr_file is not None:
            heat = prepare_retention_matrix(nrr_file)
            if not heat.empty:
                cohorts = heat.index.tolist()
                ages = heat.columns.tolist()

                fig = plot_retention_heatmap(
                    heat.values,
                    cohorts=cohorts,
                    ages=ages,
                    title="Net Revenue Retention Cohort Matrix",
                    height=max(500, 220 + 24 * min(25, len(heat))),
                    cmap="Viridis",
                    zmin=0,
                    zmax=3,  # Allow >100%
                    format_as_pct=True,
                )
                render_chart_responsive(fig, "cohort_nrr")
            else:
                render_missing_data("NRR matrix is empty.")
        else:
            render_missing_data("Net revenue retention matrix not available.")

    with col2:
        st.markdown("#### Gross Revenue Retention")
        if grr_file is not None:
            heat = prepare_retention_matrix(grr_file)
            if not heat.empty:
                cohorts = heat.index.tolist()
                ages = heat.columns.tolist()

                fig = plot_retention_heatmap(
                    heat.values,
                    cohorts=cohorts,
                    ages=ages,
                    title="Gross Revenue Retention Cohort Matrix",
                    height=max(500, 220 + 24 * min(25, len(heat))),
                    cmap="Blues",
                    zmin=0,
                    zmax=1,
                )
                render_chart_responsive(fig, "cohort_grr")
            else:
                render_missing_data("GRR matrix is empty.")
        else:
            render_missing_data("Gross revenue retention matrix not available.")


def render_maturity_curve_tab(decay: pd.DataFrame) -> None:
    """Render maturity-aware retention decay curve."""
    st.markdown("#### Maturity-Aware Retention Decay")

    if decay is not None:
        age = "age_month" if "age_month" in decay.columns else None
        logo = "weighted_logo_retention" if "weighted_logo_retention" in decay.columns else "logo_retention"
        nrr = "weighted_net_revenue_retention" if "weighted_net_revenue_retention" in decay.columns else "net_revenue_retention"
        grr = "weighted_gross_revenue_retention" if "weighted_gross_revenue_retention" in decay.columns else "gross_revenue_retention"

        if age:
            # Only include series that exist in the data
            has_logo = logo in decay.columns
            has_nrr = nrr in decay.columns
            has_grr = grr in decay.columns

            if not any([has_logo, has_nrr, has_grr]):
                render_missing_data("No retention series found in decay curve.")
                return

            fig = plot_retention_decay(
                ages=decay[age].tolist(),
                logo_retention=decay[logo].tolist() if has_logo else None,
                nrr=decay[nrr].tolist() if has_nrr else None,
                grr=decay[grr].tolist() if has_grr else None,
                title="Maturity-Aware Retention Decay",
                height=500,
            )
            render_chart_responsive(fig, "cohort_decay")

            render_science_card(
                "Why the Curve Matters",
                "The cohort matrix shows individual acquisition vintages; the maturity curve aggregates customers at the same age. "
                "This separates lifecycle decay from calendar-time effects and prevents newer cohorts from being treated as if they "
                "had already had time to mature. The curve is weighted by cohort size.",
            )
        else:
            render_missing_data("Decay curve missing age column.")
    else:
        render_missing_data("Retention decay curve not available.")


def render_scorecard_tab(scorecard: pd.DataFrame) -> None:
    """Render cohort scorecard table."""
    st.markdown("#### Cohort Scorecard")

    if scorecard is not None:
        # Add download
        render_download_button(scorecard, "cohort_scorecard.csv", "Download Scorecard (CSV)")

        st.dataframe(scorecard, use_container_width=True, hide_index=True)

        render_science_card(
            "Maturity Censoring",
            "The scorecard averages across cohorts with different maximum observable ages. "
            "Early cohorts (2009-12) are observable to age 24+; late cohorts (2011-09+) only to age 3. "
            "Averages at higher ages only include cohorts that have reached that maturity. "
            "This right-censoring is explicitly tracked in the maturity curve.",
        )
    else:
        render_missing_data("Cohort scorecard not available.")


def render_acquisition_quality_tab(acquisition_quality: pd.DataFrame) -> None:
    """Render acquisition quality metrics."""
    st.markdown("#### Cohort Acquisition Quality")

    if acquisition_quality is not None:
        render_download_button(acquisition_quality, "cohort_acquisition_quality.csv", "Download (CSV)")
        st.dataframe(acquisition_quality, use_container_width=True, hide_index=True)
    else:
        render_missing_data("Cohort acquisition quality not available.")


def prepare_retention_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare retention matrix for heatmap: index=cohort, columns=age."""
    df = df.copy()

    cohort_col = "cohort_month" if "cohort_month" in df.columns else "cohort"
    if not cohort_col or cohort_col not in df.columns:
        return pd.DataFrame()

    df = df.set_index(cohort_col)

    # Find age columns
    numeric_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str.lower().startswith("age"):
            numeric_cols.append(col)
        elif col_str.isdigit():
            numeric_cols.append(col)

    if not numeric_cols:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return pd.DataFrame()

    out = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Standardize column names to integers
    new_cols = []
    for c in out.columns:
        c_str = str(c)
        if c_str.lower().startswith("age_"):
            try:
                new_cols.append(int(c_str.replace("age_", "")))
            except Exception:
                new_cols.append(c)
        elif c_str.isdigit():
            new_cols.append(int(c_str))
        else:
            new_cols.append(c)

    out.columns = new_cols
    out = out.sort_index()

    return out


if __name__ == "__main__":
    render_cohorts_page()