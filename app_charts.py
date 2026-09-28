"""
Chart utilities for the Retail Customer Intelligence application.

Provides consistent Plotly chart styling, color schemes, and reusable chart types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# DESIGN TOKENS
# =============================================================================

# Semantic color palette
COLORS = {
    # Primary brand
    "primary": "#315efb",
    "primary_soft": "#eef2ff",
    "primary_muted": "#697386",

    # Semantic states
    "success": "#218739",
    "success_soft": "#e8f5e9",
    "warning": "#a56600",
    "warning_soft": "#fff8e1",
    "danger": "#c53d32",
    "danger_soft": "#fdeaea",
    "info": "#1976d2",
    "info_soft": "#e3f2fd",

    # Neutrals
    "ink": "#172033",
    "muted": "#697386",
    "line": "#e4e7ec",
    "panel": "#ffffff",
    "panel_soft": "#f7f8fa",
    "background": "#fbfcfe",

    # Action colors (consistent across app)
    "action_protect_value": "#218739",      # green
    "action_accelerate_purchase": "#315efb", # blue
    "action_reactivate": "#a56600",          # amber
    "action_cross_sell": "#7b1fa2",          # purple
    "action_nurture": "#1976d2",             # light blue
    "action_monitor": "#697386",             # gray

    # Segment colors (categorical, colorblind-safe)
    "segment_palette": [
        "#315efb",  # blue
        "#218739",  # green
        "#a56600",  # amber
        "#c53d32",  # red
        "#7b1fa2",  # purple
        "#0097a7",  # cyan
        "#e65100",  # deep orange
        "#388e3c",  # dark green
        "#d81b60",  # pink
        "#455a64",  # blue gray
        "#f57f17",  # yellow
        "#6a1b9a",  # deep purple
    ],

    # Retention heatmap
    "retention_cmap": "Blues",
    "nrr_cmap": "Viridis",

    # CLV uncertainty
    "uncertainty_fill": "rgba(49, 94, 251, 0.15)",
    "uncertainty_line": "rgba(49, 94, 251, 0.5)",

    # Risk/propensity
    "risk_high": "#c53d32",
    "risk_medium": "#a56600",
    "risk_low": "#218739",
}

# Action color mapping
ACTION_COLORS = {
    "protect_value": COLORS["action_protect_value"],
    "accelerate_purchase": COLORS["action_accelerate_purchase"],
    "reactivate": COLORS["action_reactivate"],
    "cross_sell": COLORS["action_cross_sell"],
    "nurture": COLORS["action_nurture"],
    "monitor": COLORS["action_monitor"],
}

# =============================================================================
# BASE LAYOUT
# =============================================================================

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def base_layout(
    fig: go.Figure,
    height: int = 420,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    show_legend: bool = True,
    legend_orientation: str = "h",
) -> go.Figure:
    """Apply consistent base layout to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, ui-sans-serif, system-ui, sans-serif",
            color=COLORS["ink"],
            size=12,
        ),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(size=17, color=COLORS["ink"]),
            )
            if title
            else None
        ),
        xaxis=dict(
            title=xaxis_title,
            showgrid=True,
            gridcolor=COLORS["line"],
            zeroline=False,
            tickfont=dict(size=11, color=COLORS["muted"]),
            title_font=dict(size=12, color=COLORS["ink"]),
        ) if xaxis_title else dict(
            showgrid=True,
            gridcolor=COLORS["line"],
            zeroline=False,
            tickfont=dict(size=11, color=COLORS["muted"]),
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor=COLORS["line"],
            zeroline=False,
            tickfont=dict(size=11, color=COLORS["muted"]),
            title_font=dict(size=12, color=COLORS["ink"]),
        ) if yaxis_title else dict(
            showgrid=True,
            gridcolor=COLORS["line"],
            zeroline=False,
            tickfont=dict(size=11, color=COLORS["muted"]),
        ),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=COLORS["line"],
            borderwidth=1,
            font=dict(size=11, color=COLORS["ink"]),
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=COLORS["line"],
            font=dict(size=11, color=COLORS["ink"]),
        ),
    )
    return fig


# =============================================================================
# SEMANTIC COLOR HELPERS
# =============================================================================

def get_action_color(action: str) -> str:
    """Get consistent color for an action."""
    action_lower = action.lower().replace(" ", "_")
    return ACTION_COLORS.get(action_lower, COLORS["primary"])


def get_segment_color(segment_index: int) -> str:
    """Get consistent color for a segment by index."""
    palette = COLORS["segment_palette"]
    return palette[segment_index % len(palette)]


def get_risk_color(probability: float) -> str:
    """Get color based on risk probability."""
    if probability >= 0.70:
        return COLORS["risk_high"]
    elif probability >= 0.40:
        return COLORS["risk_medium"]
    return COLORS["risk_low"]


# =============================================================================
# CHART TYPES
# =============================================================================

def plot_concentration_curve(
    values: Sequence[float],
    title: str = "Concentration Curve",
    height: int = 420,
    value_label: str = "Value",
    reference_lines: List[float] = None,
) -> go.Figure:
    """Plot a Lorenz/Pareto concentration curve."""
    values = np.array([v for v in values if v > 0])
    values = np.sort(values)[::-1]
    cumsum = np.cumsum(values)
    total = cumsum[-1] if len(cumsum) > 0 else 1
    cumshare = cumsum / total
    rank_pct = np.arange(1, len(values) + 1) / len(values)

    # Sample for performance
    if len(values) > 1000:
        idx = np.linspace(0, len(values) - 1, 1000, dtype=int)
        rank_pct = rank_pct[idx]
        cumshare = cumshare[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rank_pct,
        y=cumshare,
        mode="lines",
        line=dict(width=2, color=COLORS["primary"]),
        name=f"Cumulative {value_label} share",
        hovertemplate=f"Customer percentile: %{{x:.0%}}<br>Cumulative {value_label} share: %{{y:.0%}}<extra></extra>",
    ))

    # Equality line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(width=1, dash="dash", color=COLORS["muted"]),
        name="Perfect equality",
        hoverinfo="skip",
    ))

    # Reference lines
    if reference_lines:
        for ref in reference_lines:
            fig.add_hline(
                y=ref,
                line_dash="dot",
                line_color=COLORS["muted"],
                annotation_text=f"{ref:.0%} of value",
                annotation_position="right",
            )

    fig.update_xaxes(tickformat=".0%", title="Customer percentile by value")
    fig.update_yaxes(tickformat=".0%", title=f"Cumulative {value_label} share")

    return base_layout(fig, height=height, title=title)


def plot_distribution_histogram(
    values: Sequence[float],
    title: str = "Distribution",
    height: int = 420,
    x_title: str = "Value",
    nbins: int = 40,
    log_x: bool = False,
    show_box: bool = True,
) -> go.Figure:
    """Plot a histogram with optional box plot marginal."""
    values = np.array([v for v in values if not np.isnan(v)])

    fig = px.histogram(
        x=values,
        nbins=nbins,
        marginal="box" if show_box else None,
        labels={"x": x_title, "y": "Count"},
        color_discrete_sequence=[COLORS["primary"]],
    )

    if log_x:
        fig.update_xaxes(type="log")

    return base_layout(fig, height=height, title=title)


def plot_uncertainty_band(
    x: Sequence[float],
    y: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    title: str = "Expected Value with Uncertainty",
    height: int = 420,
    x_title: str = "Index",
    y_title: str = "Value",
    sample_every: int = 1,
) -> go.Figure:
    """Plot expected value with uncertainty band."""
    x = np.array(x)
    y = np.array(y)
    lower = np.array(lower)
    upper = np.array(upper)

    # Sample for performance
    if sample_every > 1:
        idx = np.arange(0, len(x), sample_every)
        x, y, lower, upper = x[idx], y[idx], lower[idx], upper[idx]

    fig = go.Figure()

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor=COLORS["uncertainty_fill"],
        line=dict(width=0),
        name="Uncertainty range (p10-p90)",
        hoverinfo="skip",
    ))

    # Expected value line
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(width=2, color=COLORS["primary"]),
        name="Expected value",
        hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y:,.0f}}<extra></extra>",
    ))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    return base_layout(fig, height=height, title=title)


def plot_retention_heatmap(
    matrix: np.ndarray,
    cohorts: List[str],
    ages: List[int],
    title: str = "Retention Matrix",
    height: int = 500,
    cmap: str = "Blues",
    zmin: float = 0,
    zmax: float = 1,
    format_as_pct: bool = True,
) -> go.Figure:
    """Plot a cohort retention heatmap with proper NaN handling."""
    # Mask NaN values for immature cohorts
    mask = np.isnan(matrix)
    display_matrix = np.where(mask, np.nan, matrix)

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix,
        x=ages,
        y=cohorts,
        colorscale=cmap,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title="Retention" if format_as_pct else "Index",
            tickformat=".0%" if format_as_pct else ".1f",
        ),
        hovertemplate=(
            "Cohort: %{y}<br>"
            "Age: %{x} months<br>"
            f"Retention: %{{z:{'.0%' if format_as_pct else '.1f'}}}<extra></extra>"
        ),
        showscale=True,
    ))

    # Add pattern for NaN (immature) cells
    if np.any(mask):
        # Create a separate trace for masked cells
        fig.add_trace(go.Heatmap(
            z=np.where(mask, 1, np.nan),
            x=ages,
            y=cohorts,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(200,200,200,0.3)"]],
            showscale=False,
            hoverinfo="skip",
        ))

    fig.update_xaxes(
        title="Months since acquisition",
        tickmode="linear",
        dtick=1,
    )
    fig.update_yaxes(
        title="Acquisition cohort",
        autorange="reversed",
    )

    # Dynamic height based on number of cohorts
    dynamic_height = max(height, 220 + 24 * min(20, len(cohorts)))

    return base_layout(fig, height=dynamic_height, title=title)


def plot_retention_decay(
    ages: Sequence[int],
    logo_retention: Optional[Sequence[float]] = None,
    nrr: Optional[Sequence[float]] = None,
    grr: Optional[Sequence[float]] = None,
    title: str = "Retention Decay Curve",
    height: int = 500,
) -> go.Figure:
    """Plot maturity-aware retention decay curves."""
    fig = go.Figure()

    traces = []
    if logo_retention is not None:
        traces.append(("Logo retention", logo_retention, COLORS["primary"], "lines+markers"))
    if nrr is not None:
        traces.append(("Net revenue retention", nrr, COLORS["success"], "lines+markers"))
    if grr is not None:
        traces.append(("Gross revenue retention", grr, COLORS["info"], "lines+markers"))

    for name, values, color, mode in traces:
        fig.add_trace(go.Scatter(
            x=list(ages),
            y=list(values),
            mode=mode,
            name=name,
            line=dict(width=2, color=color),
            marker=dict(size=6),
            hovertemplate=f"Age: %{{x}} months<br>{name}: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_xaxes(title="Months since acquisition", dtick=1)
    fig.update_yaxes(title="Retention", tickformat=".0%")

    return base_layout(fig, height=height, title=title)


def plot_scatter_risk_propensity(
    churn_prob: Sequence[float],
    purchase_prob: Sequence[float],
    colors: Optional[Sequence[str]] = None,
    sizes: Optional[Sequence[float]] = None,
    title: str = "Risk vs Propensity",
    height: int = 560,
    sample_max: int = 12000,
) -> go.Figure:
    """Plot churn risk vs next-purchase propensity scatter."""
    churn_prob = np.array(churn_prob)
    purchase_prob = np.array(purchase_prob)

    # Sample for performance
    if len(churn_prob) > sample_max:
        idx = np.random.choice(len(churn_prob), sample_max, replace=False)
        churn_prob = churn_prob[idx]
        purchase_prob = purchase_prob[idx]
        if colors is not None:
            colors = np.array(colors)[idx]
        if sizes is not None:
            sizes = np.array(sizes)[idx]

    fig = go.Figure()

    if colors is not None:
        # Categorical coloring by action/segment
        unique_colors = np.unique(colors)
        for c in unique_colors:
            mask = colors == c
            fig.add_trace(go.Scatter(
                x=churn_prob[mask],
                y=purchase_prob[mask],
                mode="markers",
                marker=dict(
                    size=sizes[mask] if sizes is not None else 6,
                    color=c,
                    opacity=0.45,
                    line=dict(width=0),
                ),
                name=str(c),
                hovertemplate="Churn risk: %{x:.0%}<br>Purchase propensity: %{y:.0%}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=churn_prob,
            y=purchase_prob,
            mode="markers",
            marker=dict(size=6, color=COLORS["primary"], opacity=0.4),
            hovertemplate="Churn risk: %{x:.0%}<br>Purchase propensity: %{y:.0%}<extra></extra>",
        ))

    # Reference lines
    fig.add_vline(x=0.70, line_dash="dot", line_color=COLORS["muted"], annotation_text="High risk threshold")
    fig.add_hline(y=0.50, line_dash="dot", line_color=COLORS["muted"], annotation_text="Medium propensity")

    fig.update_xaxes(title="Churn probability (next-month inactivity risk)", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Next-purchase probability (30-day)", tickformat=".0%", range=[0, 1])

    return base_layout(fig, height=height, title=title)


def plot_segment_profile_heatmap(
    profiles: pd.DataFrame,
    segment_col: str,
    feature_cols: List[str],
    title: str = "Segment Behavioral Profile",
    height: int = 560,
    block_labels: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Plot segment profile heatmap with block annotations."""
    # Prepare data
    heat_data = profiles.set_index(segment_col)[feature_cols].apply(pd.to_numeric, errors="coerce")
    heat_data = heat_data.replace([np.inf, -np.inf], np.nan)

    fig = px.imshow(
        heat_data,
        aspect="auto",
        labels={"x": "Behavioral feature", "y": "Segment", "color": "Score"},
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )

    fig.update_xaxes(side="bottom", tickangle=45)
    fig.update_yaxes(title="Segment")

    return base_layout(fig, height=height, title=title)


def plot_action_allocation(
    action_counts: pd.DataFrame,
    action_col: str,
    count_col: str,
    title: str = "Action Allocation",
    height: int = 470,
) -> go.Figure:
    """Plot horizontal bar chart of action allocation."""
    plot_df = action_counts.sort_values(count_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=count_col,
        y=action_col,
        orientation="h",
        labels={count_col: "Customers", action_col: "Recommended action"},
        color=action_col,
        color_discrete_map=ACTION_COLORS,
    )

    fig.update_traces(hovertemplate="Action: %{y}<br>Customers: %{x:,}<extra></extra>")

    return base_layout(fig, height=height, title=title, show_legend=False)


def plot_decision_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: Optional[str] = None,
    title: str = "Decision Scatter",
    height: int = 470,
    x_log: bool = False,
    y_pct: bool = False,
    sample_max: int = 12000,
) -> go.Figure:
    """Plot decision engine scatter: value vs risk colored by action."""
    plot_df = df.copy()

    if len(plot_df) > sample_max:
        plot_df = plot_df.sample(n=sample_max, random_state=42)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=18,
        opacity=0.45,
        color_discrete_map=ACTION_COLORS,
        hover_data=[c for c in ["Customer ID", "priority_score", "decision_confidence"] if c in plot_df.columns],
    )

    if x_log:
        fig.update_xaxes(type="log", title=x_col)
    else:
        fig.update_xaxes(title=x_col)

    if y_pct:
        fig.update_yaxes(tickformat=".0%", title=y_col)
    else:
        fig.update_yaxes(title=y_col)

    return base_layout(fig, height=height, title=title)


def plot_pca_scatter(
    df: pd.DataFrame,
    pc1_col: str,
    pc2_col: str,
    color_col: Optional[str] = None,
    title: str = "Behavioral Space (PCA)",
    height: int = 560,
    sample_max: int = 12000,
    noise_label: str = "Noise",
) -> go.Figure:
    """Plot PCA scatter with noise handling."""
    plot_df = df.copy()

    if len(plot_df) > sample_max:
        plot_df = plot_df.sample(n=sample_max, random_state=42)

    # Separate noise if present
    if color_col and noise_label in plot_df[color_col].astype(str).unique():
        noise_df = plot_df[plot_df[color_col].astype(str) == noise_label]
        cluster_df = plot_df[plot_df[color_col].astype(str) != noise_label]

        fig = go.Figure()

        # Plot clusters first
        if not cluster_df.empty and color_col:
            for seg in cluster_df[color_col].unique():
                seg_df = cluster_df[cluster_df[color_col] == seg]
                fig.add_trace(go.Scatter(
                    x=seg_df[pc1_col],
                    y=seg_df[pc2_col],
                    mode="markers",
                    marker=dict(size=5, opacity=0.6, color=get_segment_color(hash(str(seg)) % 12)),
                    name=str(seg),
                    hovertemplate=f"PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>Segment: {seg}<extra></extra>",
                ))

        # Plot noise last (on top, but muted)
        if not noise_df.empty:
            fig.add_trace(go.Scatter(
                x=noise_df[pc1_col],
                y=noise_df[pc2_col],
                mode="markers",
                marker=dict(size=3, opacity=0.2, color=COLORS["muted"]),
                name=noise_label,
                hovertemplate=f"PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>{noise_label}<extra></extra>",
            ))
    else:
        fig = px.scatter(
            plot_df,
            x=pc1_col,
            y=pc2_col,
            color=color_col,
            opacity=0.55,
            color_discrete_sequence=COLORS["segment_palette"],
            hover_data=[c for c in ["Customer ID", "segment_confidence"] if c in plot_df.columns],
        )

    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")

    return base_layout(fig, height=height, title=title)


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    title: str = "Time Series",
    height: int = 460,
    secondary_y_cols: Optional[List[str]] = None,
) -> go.Figure:
    """Plot time series with optional secondary y-axis."""
    fig = go.Figure()

    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[col],
            mode="lines+markers",
            name=col.replace("_", " ").title(),
            yaxis="y2" if secondary_y_cols and col in secondary_y_cols else "y",
        ))

    if secondary_y_cols:
        fig.update_layout(
            yaxis2=dict(
                title=secondary_y_cols[0].replace("_", " ").title(),
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=value_cols[0].replace("_", " ").title() if value_cols else "Value")

    return base_layout(fig, height=height, title=title)


def plot_horizontal_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Horizontal Bar",
    height: int = 420,
    color_col: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Plot horizontal bar chart."""
    plot_df = df.sort_values(x_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        orientation="h",
        color=color_col,
        color_discrete_map=color_map,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
    )

    fig.update_traces(hovertemplate=f"{y_col}: %{{y}}<br>{x_col}: %{{x:,}}<extra></extra>")

    return base_layout(fig, height=height, title=title, show_legend=color_col is not None)


# =============================================================================
# RESPONSIVE CHART HELPER
# =============================================================================

def render_chart_responsive(fig: go.Figure, key: str = "") -> None:
    """Render a Plotly chart with responsive config."""
    import streamlit as st
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)