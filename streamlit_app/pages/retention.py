"""
Retention & Next Purchase page for Retail Customer Intelligence.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st
from pathlib import Path

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_card,
    render_kpi_row,
    render_missing,
)
from ..ui.scope import apply_scope_to_dataframe
from ..app_data import get_registry
from ..app_formatting import format_probability, format_count
from ..ui.charts import (
    plot_missing,
    plot_histogram,
    plot_scatter,
    plot_scatter_with_quadrants,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Retention & Next Purchase page."""
    registry = get_registry()

    churn = registry.load_dataframe("churn")
    next_purchase = registry.load_dataframe("churn")  # Same file

    combined = registry.load_dataframe("customer_360")

    if churn is None and combined is not None and "churn_probability" in combined.columns:
        churn = combined
    if next_purchase is None and combined is not None and "next_purchase_probability" in combined.columns:
        next_purchase = combined

    if churn is None and next_purchase is None:
        from ..ui import render_empty_state
        render_empty_state(
            "No retention data",
            "No churn or next-purchase outputs found.",
            "Run the pipeline",
            "python scripts/churn_next_purchase.py",
        )
        st.stop()

    tabs = st.tabs(["Risk profile", "Purchase propensity", "Risk vs propensity", "Model diagnostics"])

    with tabs[0]:
        if churn is not None:
            ccol = "churn_probability" if "churn_probability" in churn.columns else None
            for c in ["churn_prob", "prob_churn"]:
                if c in churn.columns:
                    ccol = c
                    break

            if ccol:
                values = pd.to_numeric(churn[ccol], errors="coerce").dropna()

                render_kpi_row([
                    {"label": "Customers scored", "value": len(values), "formatter": "count"},
                    {"label": "Median inactivity risk", "value": values.median(), "formatter": "probability"},
                    {"label": "High risk (≥70%)", "value": (values >= 0.70).mean(), "formatter": "percent"},
                    {"label": "Very high risk (≥85%)", "value": (values >= 0.85).mean(), "formatter": "percent"},
                ])

                fig = plot_histogram(values, "Inactivity Risk", "Predicted probability of next-month inactivity")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                render_science_card(
                    "Why survival modeling is different from a churn label",
                    "A survival formulation models the timing of continued activity "
                    "and accounts for the fact that newer customers have had less "
                    "time to experience an observed lapse. This is preferable to "
                    "defining churn with one arbitrary recency threshold and calling "
                    "that label ground truth."
                )
            else:
                render_missing("Churn output found but no probability field was recognized.")
        else:
            render_missing()

    with tabs[1]:
        if next_purchase is not None:
            pcol = "next_purchase_probability" if "next_purchase_probability" in next_purchase.columns else None
            for c in ["next_purchase_probability_30d", "purchase_probability_30d"]:
                if c in next_purchase.columns:
                    pcol = c
                    break

            if pcol:
                values = pd.to_numeric(next_purchase[pcol], errors="coerce").dropna()

                render_kpi_row([
                    {"label": "Customers scored", "value": len(values), "formatter": "count"},
                    {"label": "Median probability", "value": values.median(), "formatter": "probability"},
                    {"label": "High propensity (≥70%)", "value": (values >= 0.70).mean(), "formatter": "percent"},
                    {"label": "Low propensity (<30%)", "value": (values < 0.30).mean(), "formatter": "percent"},
                ])

                fig = plot_histogram(values, "Next-purchase probability", "Predicted probability of the next purchase event")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                render_missing("Next-purchase output found but no probability field was recognized.")
        else:
            render_missing()

    with tabs[2]:
        if churn is not None and next_purchase is not None:
            ccol = "churn_probability" if "churn_probability" in churn.columns else None
            for c in ["churn_prob", "prob_churn"]:
                if c in churn.columns:
                    ccol = c
                    break

            pcol = "next_purchase_probability" if "next_purchase_probability" in next_purchase.columns else None
            for c in ["next_purchase_probability_30d", "purchase_probability_30d"]:
                if c in next_purchase.columns:
                    pcol = c
                    break

            id1 = "Customer ID" if "Customer ID" in churn.columns else None
            id2 = "Customer ID" if "Customer ID" in next_purchase.columns else None

            if ccol and pcol and id1 and id2:
                left = churn[[id1, ccol]].copy()
                right = next_purchase[[id2, pcol]].copy()
                left["Customer ID"] = pd.to_numeric(left[id1], errors="coerce").round()
                right["Customer ID"] = pd.to_numeric(right[id2], errors="coerce").round()

                merged = left[["Customer ID", ccol]].merge(
                    right[["Customer ID", pcol]],
                    on="Customer ID",
                    how="inner",
                )

                merged[ccol] = pd.to_numeric(merged[ccol], errors="coerce")
                merged[pcol] = pd.to_numeric(merged[pcol], errors="coerce")
                merged = merged.dropna(subset=[ccol, pcol])

                if len(merged) > 12000:
                    merged_plot = merged.sample(12000, random_state=42)
                else:
                    merged_plot = merged

                fig = plot_scatter_with_quadrants(
                    merged_plot,
                    x_col=ccol,
                    y_col=pcol,
                    x_threshold=0.70,
                    y_threshold=0.50,
                    title="Customer risk versus near-term purchase propensity",
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                st.caption(
                    "The most commercially interesting customers are not necessarily "
                    "the highest-risk or highest-propensity customers in isolation. "
                    "The decision layer combines these signals with value and behavioral context."
                )
            else:
                render_missing("Customer ID and probability columns could not be matched across the two outputs.")
        else:
            render_missing("Both churn and next-purchase outputs are needed for this view.")

    with tabs[3]:
        diagnostics = []

        registry = get_registry()
        output_dir = Path(registry.base_path) / "churn_next_purchase_output"
        churn_card_path = output_dir / "model_card.json"

        churn_card = {}
        if churn_card_path.exists():
            try:
                with open(churn_card_path, "r") as f:
                    churn_card = json.load(f)
            except Exception:
                pass

        for label, df in [("Churn / Next-month inactivity risk", churn), ("Next purchase", next_purchase)]:
            if df is None:
                diagnostics.append({
                    "Model": label,
                    "Status": "Not available",
                    "Rows": 0,
                    "AUC": "—",
                    "Brier": "—",
                    "Log Loss": "—",
                    "Calibration": "—",
                })
                continue

            if churn_card and "metrics" in churn_card:
                metrics = churn_card.get("metrics", {})
                diagnostics.append({
                    "Model": label,
                    "Status": "Available",
                    "Rows": len(df),
                    "AUC": f"{metrics.get('roc_auc', 0):.3f}" if metrics.get('roc_auc') else "—",
                    "Brier": f"{metrics.get('brier', 0):.4f}" if metrics.get('brier') else "—",
                    "Log Loss": f"{metrics.get('log_loss', 0):.4f}" if metrics.get('log_loss') else "—",
                    "Calibration": "Available" if "calibration" in churn_card else "—",
                })
            else:
                metric_candidates = [
                    c for c in df.columns
                    if any(
                        token in c.lower()
                        for token in [
                            "auc", "logloss", "brier", "precision", "recall",
                            "f1", "accuracy", "calibration", "rmse", "mae",
                        ]
                    )
                ]

                diagnostics.append({
                    "Model": label,
                    "Status": "Available (column-based)",
                    "Rows": len(df),
                    "Detected metrics": len(metric_candidates),
                })

        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), use_container_width=True, hide_index=True)

        render_science_card(
            "Validation principle",
            "Predictive retention and purchase models should be evaluated with "
            "time-based backtesting. A random train/test split can leak future "
            "customer behavior into the training population and make model "
            "performance look artificially strong."
        )