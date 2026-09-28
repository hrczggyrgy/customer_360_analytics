"""
Segmentation page — Behavioral segmentation analysis with PCA, profiles, and stability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_section_label,
    render_kpi_row,
    render_science_card,
    render_missing_data,
    render_status_chip,
)
from app_charts import (
    plot_segment_profile_heatmap,
    plot_pca_scatter,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import format_count, format_probability


def render_segmentation_page() -> None:
    """Render the Segmentation page."""
    registry = get_registry()

    segments = registry.load_dataframe("segmentation", "customer_segments")
    profiles = registry.load_dataframe("segmentation", "segment_profiles")
    pca = registry.load_dataframe("segmentation", "pca_coordinates")
    model_metadata = registry.load_dataframe("segmentation", "model_metadata")

    if segments is None and profiles is None and pca is None:
        st.warning("No segmentation outputs found. Run customer_segmentation.py first.")
        st.stop()

    # Page hero
    render_page_hero(
        title="Behavioral Segmentation",
        description="A multi-dimensional view of customer behavior beyond conventional RFM.",
        kicker="BEHAVIORAL INTELLIGENCE",
    )

    # ---- TOP KPIs ----
    if segments is not None:
        seg_col = "segment" if "segment" in segments.columns else "cluster"
        if seg_col:
            seg_summary = segments.groupby(seg_col, dropna=False).size().reset_index(name="customers")

            # Detect noise correctly (segment == -1 or string "noise")
            noise_mask = (
                segments[seg_col].astype(str).str.contains("noise", case=False, na=False) |
                (pd.to_numeric(segments[seg_col], errors="coerce") == -1)
            )
            noise_count = int(noise_mask.sum())
            noise_pct = noise_count / len(segments) if len(segments) > 0 else 0

            n_segments = seg_summary[seg_col].nunique()
            if noise_count > 0:
                n_segments -= 1  # Don't count noise as a segment

            kpis = [
                {"label": "Customers", "value": format_count(len(segments))},
                {"label": "Segments (excl. noise)", "value": format_count(n_segments)},
                {"label": "Largest Segment", "value": format_count(seg_summary["customers"].max())},
                {"label": "Noise / Low-Density", "value": format_probability(noise_pct)},
            ]
            render_kpi_row(kpis, columns=4)

            # Segment size chart
            st.markdown("#### Segment Size Distribution")
            seg_summary_clean = seg_summary[~noise_mask].copy()
            seg_summary_clean = seg_summary_clean.sort_values("customers", ascending=True)

            import plotly.express as px
            fig = px.bar(
                seg_summary_clean,
                x="customers",
                y=seg_col,
                orientation="h",
                labels={"customers": "Customers", seg_col: "Segment"},
                color_discrete_sequence=["#315efb"],
            )
            render_chart_responsive(fig, "seg_sizes")

    # ---- PCA MAP & BEHAVIORAL VIEWS ----
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("#### Latent Behavioral Map (PCA)")

        if pca is not None:
            pc1 = "PC1" if "PC1" in pca.columns else "pc1"
            pc2 = "PC2" if "PC2" in pca.columns else "pc2"
            ps = "segment" if "segment" in pca.columns else "cluster"

            if pc1 in pca.columns and pc2 in pca.columns:
                fig = plot_pca_scatter(
                    pca,
                    pc1_col=pc1,
                    pc2_col=pc2,
                    color_col=ps,
                    title="Customer Behavioral Space after PCA",
                    height=560,
                    noise_label="Noise",
                )
                render_chart_responsive(fig, "seg_pca")

                # PCA explained variance
                if model_metadata is not None:
                    try:
                        if isinstance(model_metadata, pd.DataFrame):
                            meta = model_metadata.to_dict("records")[0] if len(model_metadata) > 0 else {}
                        else:
                            meta = model_metadata
                        ev = meta.get("pca_explained_variance") or meta.get("explained_variance")
                        if ev:
                            st.caption(f"PCA Explained Variance: PC1={ev.get('PC1', 'N/A'):.1%}, PC2={ev.get('PC2', 'N/A'):.1%}, Cumulative={ev.get('cumulative', 'N/A'):.1%}")
                    except Exception:
                        pass
            else:
                render_missing_data("PCA output found but PC1/PC2 not detected.")
        else:
            render_missing_data("PCA coordinates not available.")

    with right:
        st.markdown("#### Behavioral Profile by Block")

        if profiles is not None:
            p = profiles.copy()
            seg = "cluster" if "cluster" in p.columns else "segment"

            # Define behavioral blocks
            block_features = {
                "Economic": ["lifetime_gross_revenue", "lifetime_net_revenue", "historical_avg_order_value",
                            "lifetime_orders", "lifetime_units", "historical_value_per_order"],
                "Cadence": ["median_interpurchase_days", "interpurchase_cv", "active_month_count",
                           "orders_last_1m", "orders_last_3m", "orders_last_6m"],
                "Assortment": ["unique_products", "product_revenue_hhi", "repeat_product_ratio",
                              "product_breadth"],
                "Pricing": ["mean_unit_price", "unit_price_cv", "premium_price_line_share",
                           "price_behavior", "avg_price"],
                "Temporal": ["hour_entropy", "weekend_order_share", "month_entropy",
                            "temporal_behavior", "seasonality_score"],
                "Lifecycle": ["tenure_days", "recency_days", "lifecycle_state",
                             "reactivation_count", "has_reactivated"],
                "Returns": ["lifetime_return_value", "lifetime_return_rate", "return_rate",
                           "return_behavior"],
            }

            # Find available features in profiles
            available_features = {}
            for block, features in block_features.items():
                avail = [f for f in features if f in p.columns]
                if avail:
                    available_features[block] = avail

            if available_features:
                selected_block = st.selectbox(
                    "Behavioral Block",
                    options=list(available_features.keys()),
                    index=0,
                )

                selected_features = available_features[selected_block]

                if seg and selected_features:
                    heat = p.set_index(seg)[selected_features].apply(pd.to_numeric, errors="coerce")
                    heat = heat.replace([np.inf, -np.inf], np.nan)

                    fig = plot_segment_profile_heatmap(
                        profiles=p,
                        segment_col=seg,
                        feature_cols=selected_features,
                        title=f"Segment Profile — {selected_block} Block",
                        height=560,
                    )
                    render_chart_responsive(fig, f"seg_heatmap_{selected_block}")

                st.caption(
                    "Feature blocks are balanced before PCA so that economic intensity, cadence, "
                    "assortment, pricing, temporal behavior, lifecycle, and returns each contribute "
                    "to the latent space rather than letting one block dominate."
                )
            else:
                # Fallback to all numeric
                exclude = {seg, "customers", "confidence_mean", "share", "size"}
                numeric = p.select_dtypes(include=["number"]).columns.tolist()
                metric_cols = [c for c in numeric if c not in exclude]

                if metric_cols:
                    selected = st.multiselect(
                        "Behavioral Dimensions",
                        metric_cols,
                        default=metric_cols[:min(8, len(metric_cols))],
                    )

                    if selected:
                        fig = plot_segment_profile_heatmap(
                            profiles=p,
                            segment_col=seg,
                            feature_cols=selected,
                            title="Segment Behavioral Profile",
                            height=560,
                        )
                        render_chart_responsive(fig, "seg_heatmap_custom")
        else:
            render_missing_data("segment_profiles.csv not available.")

    # ---- SEGMENT PROFILE TABLE ----
    if profiles is not None:
        st.markdown("#### Segment Profile Table")
        st.dataframe(profiles, use_container_width=True, hide_index=True)

    # ---- SEGMENT BUSINESS DESCRIPTION ----
    if profiles is not None and segments is not None:
        st.markdown("#### Segment Business Description")

        seg_col = "cluster" if "cluster" in profiles.columns else "segment"
        available_segments = profiles[seg_col].dropna().unique().tolist()
        available_segments = [s for s in available_segments if str(s) != "-1" and "noise" not in str(s).lower()]

        if available_segments:
            selected_segment = st.selectbox(
                "Select Segment for Description",
                options=available_segments,
                format_func=lambda x: f"Segment {x}",
            )

            if selected_segment is not None:
                render_segment_description(selected_segment, profiles, segments, seg_col)

    # ---- MODEL DETAILS ----
    if model_metadata is not None:
        with st.expander("Model Details"):
            try:
                if isinstance(model_metadata, pd.DataFrame):
                    st.dataframe(model_metadata, use_container_width=True, hide_index=True)
                else:
                    st.json(model_metadata)
            except Exception:
                st.json(model_metadata)


def render_segment_description(
    segment_id: Any,
    profiles: pd.DataFrame,
    segments: pd.DataFrame,
    seg_col: str,
) -> None:
    """Generate and render a business description for a segment."""
    seg_profile = profiles[profiles[seg_col] == segment_id]
    if seg_profile.empty:
        st.info("No profile data for this segment.")
        return

    seg_size = len(segments[segments[seg_col] == segment_id])
    total_customers = len(segments)
    seg_share = seg_size / total_customers if total_customers > 0 else 0

    # Get key metrics
    profile_dict = seg_profile.iloc[0].to_dict()

    # Build description from actual statistics
    st.markdown(f"**Segment {segment_id}** — {seg_size:,} customers ({seg_share:.1%} of base)")

    desc_parts = []

    # Economic
    econ_metrics = []
    for m in ["lifetime_net_revenue", "lifetime_gross_revenue", "historical_avg_order_value"]:
        if m in profile_dict and pd.notna(profile_dict[m]):
            econ_metrics.append(f"{m}: {profile_dict[m]:,.0f}")
    if econ_metrics:
        desc_parts.append("**Economics:** " + "; ".join(econ_metrics))

    # Cadence
    cad_metrics = []
    for m in ["median_interpurchase_days", "interpurchase_cv", "active_month_count"]:
        if m in profile_dict and pd.notna(profile_dict[m]):
            cad_metrics.append(f"{m}: {profile_dict[m]:.1f}")
    if cad_metrics:
        desc_parts.append("**Cadence:** " + "; ".join(cad_metrics))

    # Assortment
    asm_metrics = []
    for m in ["unique_products", "product_revenue_hhi", "repeat_product_ratio"]:
        if m in profile_dict and pd.notna(profile_dict[m]):
            asm_metrics.append(f"{m}: {profile_dict[m]:.2f}")
    if asm_metrics:
        desc_parts.append("**Assortment:** " + "; ".join(asm_metrics))

    # Pricing
    prc_metrics = []
    for m in ["mean_unit_price", "unit_price_cv", "premium_price_line_share"]:
        if m in profile_dict and pd.notna(profile_dict[m]):
            prc_metrics.append(f"{m}: {profile_dict[m]:.2f}")
    if prc_metrics:
        desc_parts.append("**Pricing:** " + "; ".join(prc_metrics))

    # Returns
    ret_metrics = []
    for m in ["lifetime_return_rate", "return_rate"]:
        if m in profile_dict and pd.notna(profile_dict[m]):
            ret_metrics.append(f"{m}: {profile_dict[m]:.1%}")
    if ret_metrics:
        desc_parts.append("**Returns:** " + "; ".join(ret_metrics))

    for part in desc_parts:
        st.markdown(part)

    # Interpretation
    st.markdown("---")
    st.markdown("**Interpretation:**")
    interpretations = []

    # Economic tier
    rev = profile_dict.get("lifetime_net_revenue", 0)
    if pd.notna(rev):
        if rev > 5000:
            interpretations.append("High-value segment with substantial historical spend.")
        elif rev > 1000:
            interpretations.append("Medium-value segment with moderate historical spend.")
        else:
            interpretations.append("Lower-value segment with limited historical spend.")

    # Cadence
    cadence = profile_dict.get("median_interpurchase_days", 0)
    if pd.notna(cadence):
        if cadence < 30:
            interpretations.append("High purchase frequency (frequent buyers).")
        elif cadence < 90:
            interpretations.append("Moderate purchase frequency.")
        else:
            interpretations.append("Low purchase frequency (occasional buyers).")

    # Assortment
    breadth = profile_dict.get("unique_products", 0)
    if pd.notna(breadth):
        if breadth > 20:
            interpretations.append("Broad assortment buyers (cross-category).")
        elif breadth > 5:
            interpretations.append("Moderate assortment breadth.")
        else:
            interpretations.append("Narrow assortment (focused buyers).")

    # Returns
    ret_rate = profile_dict.get("lifetime_return_rate", 0)
    if pd.notna(ret_rate):
        if ret_rate > 0.20:
            interpretations.append("High return rate — potential fit/quality issues.")
        elif ret_rate > 0.10:
            interpretations.append("Moderate return rate.")
        else:
            interpretations.append("Low return rate.")

    for interp in interpretations:
        st.markdown(f"• {interp}")

    if not interpretations:
        st.markdown("• Insufficient profile data for automated interpretation.")


if __name__ == "__main__":
    render_segmentation_page()