"""
Consistent Plotly Charts for Retail Customer Intelligence Streamlit App.

This module provides standardized chart functions with consistent design tokens,
color schemes, and styling across all pages.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# DESIGN TOKENS
# =============================================================================

CHART_COLORS = {
    "primary": "#315efb",
    "secondary": "#697386",
    "success": "#218739",
    "warning": "#a56600",
    "danger": "#e11d48",
    "muted": "#697386",
    "background": "#fbfcfe",
    "panel": "#ffffff",
    "line": "#e4e7ec",
}

ACTION_COLORS = {
    "protect_value": "#218739",
    "accelerate_purchase": "#315efb",
    "reactivate": "#a56600",
    "cross_sell": "#7c3aed",
    "nurture": "#0ea5e9",
    "monitor": "#697386",
}

SEGMENT_PALETTE = [
    "#315efb", "#218739", "#a56600", "#7c3aed",
    "#0ea5e9", "#e11d48", "#f97316", "#84cc16",
    "#ec4899", "#6366f1", "#14b8a6", "#f43f5e",
]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

SEED = 42
np.random.seed(SEED)


# =============================================================================
# BASE LAYOUT
# =============================================================================

def base_layout(
    fig: go.Figure,
    height: int = 420,
    title: Optional[str] = None,
) -> go.Figure:
    """Apply consistent base layout to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, ui-sans-serif, system-ui, sans-serif",
            color="#172033",
        ),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(size=17, color="#172033"),
            )
            if title
            else None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def plot_missing(message: str = "Required output is not available yet.") -> None:
    """Render a missing data placeholder."""
    import streamlit as st
    st.info(message)


# =============================================================================
# HISTOGRAM
# =============================================================================

def plot_histogram(
    data: np.ndarray,
    x_label: str,
    title: str,
    nbins: int = 40,
    height: int = 480,
    x_format: Optional[str] = None,
) -> go.Figure:
    """Create a consistent histogram."""
    fig = px.histogram(
        x=data,
        nbins=nbins,
        labels={x_label: x_label},
    )
    
    if x_format:
        fig.update_xaxes(tickformat=x_format)
    
    return base_layout(fig, height=height, title=title)


def plot_histogram_with_marginal(
    data: np.ndarray,
    x_label: str,
    title: str,
    nbins: int = 60,
    height: int = 500,
) -> go.Figure:
    """Create histogram with box marginal."""
    fig = px.histogram(
        x=data,
        nbins=nbins,
        marginal="box",
        labels={x_label: x_label},
    )
    fig.update_xaxes(rangemode="tozero")
    return base_layout(fig, height=height, title=title)


# =============================================================================
# BAR CHARTS
# =============================================================================

def plot_horizontal_bar(
    df,
    x_col: str,
    y_col: str,
    title: str,
    height: int = 470,
    labels: Optional[dict] = None,
    orientation: str = "h",
) -> go.Figure:
    """Create a consistent horizontal bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation=orientation,
        labels=labels or {},
    )
    return base_layout(fig, height=height, title=title)


def plot_stacked_bar(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    height: int = 470,
    labels: Optional[dict] = None,
) -> go.Figure:
    """Create a stacked/grouped bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        labels=labels or {},
    )
    return base_layout(fig, height=height, title=title)


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def plot_scatter(
    df,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: str = "",
    height: int = 560,
    labels: Optional[dict] = None,
    opacity: float = 0.45,
    max_points: int = 12000,
    x_format: Optional[str] = None,
    y_format: Optional[str] = None,
    log_x: bool = False,
) -> go.Figure:
    """Create a consistent scatter plot with optional sampling."""
    plot_df = df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=SEED)
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        opacity=opacity,
        labels=labels or {},
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    if x_format:
        fig.update_xaxes(tickformat=x_format)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return base_layout(fig, height=height, title=title)


def plot_scatter_with_quadrants(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    x_threshold: float,
    y_threshold: float,
    title: str,
    height: int = 560,
    labels: Optional[dict] = None,
) -> go.Figure:
    """Create scatter plot with quadrant reference lines."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        opacity=0.40,
        labels=labels or {},
    )
    
    fig.add_vline(x=x_threshold, line_dash="dot")
    fig.add_hline(y=y_threshold, line_dash="dot")
    
    if labels:
        if x_col in labels:
            fig.update_xaxes(title=labels[x_col])
        if y_col in labels:
            fig.update_yaxes(title=labels[y_col])
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# HEATMAPS
# =============================================================================

def plot_heatmap(
    data,
    title: str,
    height: int = 500,
    labels: Optional[dict] = None,
    color_scale: str = "Blues",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """Create a consistent heatmap."""
    fig = px.imshow(
        data,
        aspect="auto",
        labels=labels or {},
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
    )
    return base_layout(fig, height=height, title=title)


def plot_segment_heatmap(
    df,
    segment_col: str,
    metric_cols: List[str],
    title: str,
    height: int = 560,
) -> go.Figure:
    """Create a segment profile heatmap."""
    heat = df.set_index(segment_col)[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    heat = heat.replace([np.inf, -np.inf], np.nan)
    
    fig = px.imshow(
        heat,
        aspect="auto",
        labels={"x": "Behavioral feature", "y": "Segment", "color": "Score"},
    )
    return base_layout(fig, height=height, title=title)


# =============================================================================
# LINE CHARTS
# =============================================================================

def plot_line(
    df,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    height: int = 500,
    labels: Optional[dict] = None,
    y_format: Optional[str] = None,
) -> go.Figure:
    """Create a consistent line chart."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        labels=labels or {},
    )
    
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return base_layout(fig, height=height, title=title)


def plot_dual_axis_line_bar(
    df,
    x_col: str,
    bar_col: str,
    line_col: str,
    title: str,
    height: int = 460,
    bar_name: str = "Revenue",
    line_name: str = "Orders",
) -> go.Figure:
    """Create a dual-axis chart with bars and line."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[bar_col],
        name=bar_name,
        hovertemplate=f"%{{x|%Y-%m}}<br>{bar_name}: %{{y:,.2f}}<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[line_col],
        mode="lines+markers",
        name=line_name,
        yaxis="y2",
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title=line_name,
            overlaying="y",
            side="right",
            showgrid=False,
        )
    )
    
    fig.update_xaxes(title="Calendar month")
    fig.update_yaxes(title=bar_name)
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# RETENTION MATRIX
# =============================================================================

def plot_retention_matrix(
    matrix,
    title: str,
    height: int = 500,
    zmin: float = 0,
    zmax: float = 1,
    color_scale: str = "Blues",
    max_cols: int = 36,
) -> go.Figure:
    """Plot a cohort retention matrix heatmap."""
    heat = matrix.iloc[:, :max_cols]
    
    fig = px.imshow(
        heat,
        aspect="auto",
        zmin=zmin,
        zmax=zmax,
        labels={
            "x": "Months since acquisition",
            "y": "Acquisition cohort",
            "color": "Retention",
        },
        color_continuous_scale=color_scale,
    )
    
    return base_layout(fig, height=height, title=title)


def plot_decay_curve(
    decay_df,
    age_col: str,
    logo_col: Optional[str] = None,
    nrr_col: Optional[str] = None,
    title: str = "Maturity-aware retention decay",
    height: int = 500,
) -> go.Figure:
    """Plot logo retention and NRR decay curves."""
    fig = go.Figure()
    
    if logo_col and logo_col in decay_df.columns:
        fig.add_trace(go.Scatter(
            x=decay_df[age_col],
            y=decay_df[logo_col],
            mode="lines+markers",
            name="Logo retention",
        ))
    
    if nrr_col and nrr_col in decay_df.columns:
        fig.add_trace(go.Scatter(
            x=decay_df[age_col],
            y=decay_df[nrr_col],
            mode="lines+markers",
            name="Net revenue retention",
        ))
    
    fig.update_yaxes(tickformat=".0%", title="Retention")
    fig.update_xaxes(title="Months since acquisition")
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# UNCERTAINTY BANDS
# =============================================================================

def plot_uncertainty_band(
    x_vals: np.ndarray,
    expected: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
    height: int = 500,
    x_title: str = "Customers ordered by value",
    y_title: str = "Value",
    max_points: int = 1000,
) -> go.Figure:
    """Plot expected values with uncertainty ribbon."""
    # Sample if too many points
    if len(x_vals) > max_points:
        idx = np.linspace(0, len(x_vals) - 1, max_points, dtype=int)
        x_vals = x_vals[idx]
        expected = expected[idx]
        lower = lower[idx]
        upper = upper[idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=expected,
        mode="lines",
        name="Expected value",
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_vals, x_vals[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        line=dict(width=0),
        name="Uncertainty range",
        opacity=0.25,
    ))
    
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# PCA SCATTER
# =============================================================================

def plot_pca_scatter(
    pca_df,
    pc1_col: str,
    pc2_col: str,
    color_col: Optional[str] = None,
    title: str = "Customer behavioral space after PCA",
    height: int = 560,
    max_points: int = 12000,
) -> go.Figure:
    """Create a PCA scatter plot."""
    plot_df = pca_df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=SEED)
    
    fig = px.scatter(
        plot_df,
        x=pc1_col,
        y=pc2_col,
        color=color_col,
        opacity=0.55,
        labels={pc1_col: "PC1", pc2_col: "PC2"},
        hover_data=[c for c in ["Customer ID", color_col, "segment_confidence"] 
                    if c and c in plot_df.columns],
    )
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# CONCENTRATION CURVE
# =============================================================================

def plot_concentration_curve(
    values: np.ndarray,
    title: str = "How concentrated is customer value?",
    height: int = 500,
) -> go.Figure:
    """Plot a concentration/Lorenz curve."""
    values = np.array(values)
    values = values[values > 0]
    values = np.sort(values)[::-1]
    
    cumsum = np.cumsum(values)
    total = cumsum[-1] if len(cumsum) > 0 else 1
    cumshare = cumsum / total
    
    ranks = np.arange(1, len(values) + 1) / len(values)
    
    # Sample for performance
    if len(values) > 1000:
        idx = np.linspace(0, len(values) - 1, 1000, dtype=int)
        ranks = ranks[idx]
        cumshare = cumshare[idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks,
        y=cumshare,
        mode="lines",
        line=dict(width=2),
        name="Cumulative value share",
        hovertemplate=(
            "Customer percentile: %{x:.0%}<br>"
            "Cumulative share: %{y:.0%}<extra></extra>"
        ),
    ))
    
    fig.add_hline(y=0.50, line_dash="dot", annotation_text="50% of value")
    fig.update_xaxes(tickformat=".0%", title="Customer percentile")
    fig.update_yaxes(tickformat=".0%", title="Cumulative share")
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# ACTION ALLOCATION BAR
# =============================================================================

def plot_action_allocation(
    action_counts: pd.DataFrame,
    action_col: str,
    count_col: str,
    title: str = "Customer allocation",
    height: int = 470,
) -> go.Figure:
    """Plot action allocation horizontal bar chart."""
    from .app_components import ACTION_COLORS
    
    action_counts = action_counts.sort_values(count_col, ascending=True)
    
    fig = px.bar(
        action_counts,
        x=count_col,
        y=action_col,
        orientation="h",
        labels={count_col: "Customers", action_col: "Action"},
        color=action_col,
        color_discrete_map=ACTION_COLORS,
    )
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# CLV BY SEGMENT
# =============================================================================

def plot_clv_by_segment(
    segment_profile: pd.DataFrame,
    segment_col: str,
    clv_col: str,
    count_col: str = "customers",
    median_col: str = "median_clv",
    title: str = "Where is customer value concentrated?",
    height: int = 470,
) -> go.Figure:
    """Plot CLV by segment as horizontal bar chart."""
    fig = px.bar(
        segment_profile,
        x=clv_col,
        y=segment_col,
        orientation="h",
        labels={clv_col: "Total CLV", segment_col: "Segment"},
        hover_data=[count_col, median_col],
    )
    return base_layout(fig, height=height, title=title)


# =============================================================================
# DECISION SCATTER
# =============================================================================

def plot_decision_scatter(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: str,
    title: str,
    height: int = 470,
    x_log: bool = False,
    y_format: Optional[str] = None,
) -> go.Figure:
    """Create decision engine scatter plot with action coloring."""
    from .app_components import ACTION_COLORS
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=18,
        opacity=0.45,
        labels={x_col: x_col, y_col: y_col, color_col: "Action"},
        color_discrete_map=ACTION_COLORS,
    )
    
    if x_log:
        fig.update_xaxes(type="log", title=f"{x_col} (log scale)")
    else:
        fig.update_xaxes(title=x_col)
    
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# PRIORITY DISTRIBUTION
# =============================================================================

def plot_priority_distribution(
    df,
    priority_col: str,
    action_col: str,
    title: str = "Priority Score Distribution by Action",
    height: int = 470,
) -> go.Figure:
    """Plot priority score distribution by action."""
    from .app_components import ACTION_COLORS
    
    fig = go.Figure()
    
    for action in df[action_col].unique():
        subset = df[df[action_col] == action]
        fig.add_trace(go.Histogram(
            x=subset[priority_col],
            name=action,
            opacity=0.5,
            nbinsx=20,
            marker_color=ACTION_COLORS.get(action, "#697386"),
        ))
    
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Priority Score")
    fig.update_yaxes(title="Density")
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# EXPECTED VALUE BY ACTION
# =============================================================================

def plot_expected_value_by_action(
    df,
    action_col: str,
    value_col: str,
    title: str = "Average Expected Value by Recommended Action",
    height: int = 470,
) -> go.Figure:
    """Plot expected value by action."""
    ev_by_action = df.groupby(action_col)[value_col].mean().sort_values()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ev_by_action.index,
        x=ev_by_action.values,
        orientation="h",
        marker_color=[ACTION_COLORS.get(a, "#697386") for a in ev_by_action.index],
    ))
    
    fig.update_xaxes(title="Expected Value Proxy")
    
    return base_layout(fig, height=height, title=title)


# =============================================================================
# CLV VS CHURN BY ACTION
# =============================================================================

def plot_clv_vs_churn_by_action(
    df,
    clv_col: str,
    churn_col: str,
    action_col: str,
    title: str = "Customer Value vs Churn Risk by Recommended Action",
    height: int = 470,
) -> go.Figure:
    """Plot CLV vs Churn colored by action."""
    from .app_components import ACTION_COLORS
    
    fig = go.Figure()
    
    for action in df[action_col].unique():
        subset = df[df[action_col] == action]
        fig.add_trace(go.Scatter(
            x=subset[clv_col],
            y=subset[churn_col],
            mode="markers",
            marker=dict(
                size=20,
                color=ACTION_COLORS.get(action, "#697386"),
                opacity=0.5,
            ),
            name=action,
        ))
    
    fig.update_xaxes(type="log", title="CLV (log scale)")
    fig.update_yaxes(tickformat=".0%", title="Churn Probability")
    
    return base_layout(fig, height=height, title=title)