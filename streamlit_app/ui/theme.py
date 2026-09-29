"""
Theme detection and management for Retail Customer Intelligence Streamlit App.

Provides runtime theme detection and CSS generation for both light and dark modes.
"""

from __future__ import annotations

import streamlit as st
from typing import Literal

from .tokens import (
    LIGHT_TOKENS,
    DARK_TOKENS,
    SPACING,
    RADIUS,
    SHADOW,
    TYPOGRAPHY,
    LAYOUT,
    ColorTokens,
    generate_css_variables,
    STATUS_COLORS,
    ACTION_COLORS,
    SEGMENT_PALETTE,
    PLOTLY_CONFIG,
    SEED,
)


# Cache theme detection
_THEME_CACHE: Literal["light", "dark"] | None = None


def get_current_theme() -> Literal["light", "dark"]:
    """
    Detect the current Streamlit theme.
    
    Streamlit sets the theme via config.toml or user preference.
    We check the config and also try to detect from the runtime.
    """
    global _THEME_CACHE
    if _THEME_CACHE is not None:
        return _THEME_CACHE
    
    # Try to get from Streamlit config
    try:
        theme_base = st.get_option("theme.base")
        if theme_base in ("light", "dark"):
            _THEME_CACHE = theme_base
            return _THEME_CACHE
    except Exception:
        pass
    
    # Default to light
    _THEME_CACHE = "light"
    return _THEME_CACHE


def get_color_tokens() -> ColorTokens:
    """Get color tokens for the current theme."""
    theme = get_current_theme()
    return LIGHT_TOKENS if theme == "light" else DARK_TOKENS


def inject_global_css() -> None:
    """Inject global CSS with theme-aware variables."""
    tokens = get_color_tokens()
    
    css = f"""
    <style>
    :root {{
{generate_css_variables(tokens)}
    }}
    
    /* Base app styling */
    .stApp {{
        background: var(--background);
        color: var(--text-primary);
    }}
    
    .block-container {{
        max-width: {LAYOUT.container_max_width};
        padding-top: {LAYOUT.content_padding_top};
        padding-bottom: {LAYOUT.content_padding_bottom};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--surface-subtle);
        border-right: 1px solid var(--border);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary);
    }}
    
    /* Metric cards */
    [data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: {RADIUS.lg};
        padding: 12px 14px;
        box-shadow: {SHADOW.sm};
    }}
    
    [data-testid="stMetric"] label {{
        color: var(--text-secondary) !important;
        font-size: {TYPOGRAPHY.text_sm} !important;
        font-weight: {TYPOGRAPHY.weight_medium} !important;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-size: {TYPOGRAPHY.text_2xl} !important;
        font-weight: {TYPOGRAPHY.weight_bold} !important;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: {TYPOGRAPHY.text_sm} !important;
    }}
    
    /* Hero section */
    .hero {{
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-subtle) 100%);
        border: 1px solid var(--border);
        border-radius: {RADIUS.xl};
        padding: 28px 32px;
        margin-bottom: 20px;
    }}
    
    .hero h1 {{
        margin: 0 0 8px 0;
        color: var(--text-primary);
        font-size: {TYPOGRAPHY.text_4xl};
        line-height: {TYPOGRAPHY.leading_tight};
        letter-spacing: -0.03em;
        font-weight: {TYPOGRAPHY.weight_bold};
    }}
    
    .hero p {{
        margin: 0;
        color: var(--text-secondary);
        max-width: 980px;
        font-size: {TYPOGRAPHY.text_lg};
        line-height: {TYPOGRAPHY.leading_relaxed};
    }}
    
    /* Section labels */
    .section-label {{
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: {TYPOGRAPHY.text_xs};
        font-weight: {TYPOGRAPHY.weight_bold};
        letter-spacing: 0.12em;
        margin: 22px 0 8px 0;
    }}
    
    /* Science cards */
    .science-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: {RADIUS.lg};
        padding: 18px 20px;
        height: 100%;
        box-shadow: {SHADOW.sm};
    }}
    
    .science-card h4 {{
        margin: 0 0 8px 0;
        color: var(--text-primary);
        font-size: {TYPOGRAPHY.text_base};
        font-weight: {TYPOGRAPHY.weight_semibold};
    }}
    
    .science-card p {{
        margin: 0;
        color: var(--text-secondary);
        line-height: {TYPOGRAPHY.leading_normal};
        font-size: {TYPOGRAPHY.text_sm};
    }}
    
    /* Status chips */
    .status-chip {{
        display: inline-block;
        padding: 5px 10px;
        border-radius: {RADIUS.full};
        background: var(--primary-soft);
        color: var(--primary);
        font-size: {TYPOGRAPHY.text_xs};
        font-weight: {TYPOGRAPHY.weight_bold};
        margin-right: 6px;
        margin-bottom: 6px;
    }}
    
    /* Kicker text */
    .kicker {{
        color: var(--primary);
        font-weight: {TYPOGRAPHY.weight_bold};
        font-size: {TYPOGRAPHY.text_xs};
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    
    /* Customer header */
    .customer-header {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: {RADIUS.xl};
        padding: 20px 22px;
        margin-bottom: 16px;
    }}
    
    .customer-header h2 {{
        margin: 0;
        color: var(--text-primary);
        font-size: {TYPOGRAPHY.text_2xl};
        font-weight: {TYPOGRAPHY.weight_bold};
    }}
    
    .customer-header p {{
        margin: 5px 0 0 0;
        color: var(--text-secondary);
        font-size: {TYPOGRAPHY.text_base};
    }}
    
    /* DataFrames */
    div[data-testid="stDataFrame"] {{
        border-radius: {RADIUS.lg};
        overflow: hidden;
        border: 1px solid var(--border);
    }}
    
    /* Tables - improve readability */
    div[data-testid="stDataFrame"] table {{
        font-size: {TYPOGRAPHY.text_sm};
    }}
    
    div[data-testid="stDataFrame"] th {{
        background: var(--surface-subtle);
        color: var(--text-secondary);
        font-weight: {TYPOGRAPHY.weight_semibold};
        font-size: {TYPOGRAPHY.text_xs};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border);
        padding: 10px 12px;
    }}
    
    div[data-testid="stDataFrame"] td {{
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-subtle);
        padding: 8px 12px;
    }}
    
    div[data-testid="stDataFrame"] tr:last-child td {{
        border-bottom: none;
    }}
    
    div[data-testid="stDataFrame"] tr:hover td {{
        background: var(--surface-subtle);
    }}
    
    /* Buttons */
    .stButton > button {{
        border-radius: {RADIUS.md};
        font-weight: {TYPOGRAPHY.weight_medium};
        font-size: {TYPOGRAPHY.text_sm};
        padding: 8px 16px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-primary);
        transition: all 0.15s ease;
    }}
    
    .stButton > button:hover {{
        background: var(--surface-subtle);
        border-color: var(--primary);
        color: var(--primary);
    }}
    
    .stButton > button[kind="primary"] {{
        background: var(--primary);
        border-color: var(--primary);
        color: var(--text-inverse);
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: var(--primary-strong);
        border-color: var(--primary-strong);
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        border-radius: {RADIUS.md};
        border: 1px solid var(--border);
        background: var(--surface);
    }}
    
    .stSelectbox label {{
        color: var(--text-secondary) !important;
        font-size: {TYPOGRAPHY.text_sm} !important;
        font-weight: {TYPOGRAPHY.weight_medium} !important;
    }}
    
    /* Text input */
    .stTextInput > div > div > input {{
        border-radius: {RADIUS.md};
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-primary);
        font-size: {TYPOGRAPHY.text_base};
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primary-soft);
    }}
    
    .stTextInput label {{
        color: var(--text-secondary) !important;
        font-size: {TYPOGRAPHY.text_sm} !important;
        font-weight: {TYPOGRAPHY.weight_medium} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: {RADIUS.md} {RADIUS.md} 0 0;
        padding: 10px 16px;
        font-weight: {TYPOGRAPHY.weight_medium};
        font-size: {TYPOGRAPHY.text_sm};
        color: var(--text-secondary);
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary);
        background: var(--surface-subtle);
    }}
    
    .stTabs [aria-selected="true"] {{
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
        background: var(--surface) !important;
    }}
    
    /* Expander */
    .stExpander {{
        border: 1px solid var(--border);
        border-radius: {RADIUS.lg};
        background: var(--surface);
    }}
    
    .stExpander summary {{
        font-weight: {TYPOGRAPHY.weight_semibold};
        color: var(--text-primary);
        font-size: {TYPOGRAPHY.text_base};
    }}
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {{
        color: var(--primary);
    }}
    
    /* Radio */
    .stRadio label {{
        color: var(--text-secondary) !important;
        font-size: {TYPOGRAPHY.text_sm} !important;
    }}
    
    /* Caption */
    .stCaption {{
        color: var(--text-muted) !important;
        font-size: {TYPOGRAPHY.text_xs} !important;
    }}
    
    /* Info/Warning/Error/Success boxes */
    .stAlert {{
        border-radius: {RADIUS.lg};
        border: 1px solid var(--border);
    }}
    
    /* Download button */
    .stDownloadButton > button {{
        border-radius: {RADIUS.md};
        font-weight: {TYPOGRAPHY.weight_medium};
    }}
    
    /* Plotly chart container */
    .stPlotlyChart {{
        border-radius: {RADIUS.lg};
        overflow: hidden;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--surface-subtle);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}
    
    /* Focus visible for accessibility */
    *:focus-visible {{
        outline: 2px solid var(--primary);
        outline-offset: 2px;
    }}
    
    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def get_status_color(status: str) -> str:
    """Get status color from centralized map."""
    return STATUS_COLORS.get(status, "#64748b")


def get_action_color(action: str) -> str:
    """Get action color from centralized map."""
    return ACTION_COLORS.get(action.lower(), "#64748b")


def get_segment_color(index: int) -> str:
    """Get segment color from palette."""
    return SEGMENT_PALETTE[index % len(SEGMENT_PALETTE)]


# Chart theme for Plotly
def get_plotly_theme() -> dict:
    """Get Plotly theme configuration for current theme."""
    tokens = get_color_tokens()
    is_dark = get_current_theme() == "dark"
    
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {
                "family": TYPOGRAPHY.font_sans,
                "color": tokens.chart_text,
                "size": 13,
            },
            "title": {
                "font": {
                    "size": 17,
                    "color": tokens.text_primary,
                },
                "x": 0,
                "xanchor": "left",
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "font": {"size": 12, "color": tokens.text_secondary},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "xaxis": {
                "gridcolor": tokens.chart_grid,
                "linecolor": tokens.chart_axis,
                "tickcolor": tokens.chart_axis,
                "tickfont": {"color": tokens.text_secondary, "size": 11},
                "titlefont": {"color": tokens.text_secondary, "size": 12},
                "zerolinecolor": tokens.border_subtle,
            },
            "yaxis": {
                "gridcolor": tokens.chart_grid,
                "linecolor": tokens.chart_axis,
                "tickcolor": tokens.chart_axis,
                "tickfont": {"color": tokens.text_secondary, "size": 11},
                "titlefont": {"color": tokens.text_secondary, "size": 12},
                "zerolinecolor": tokens.border_subtle,
            },
            "colorway": SEGMENT_PALETTE,
            "hoverlabel": {
                "bgcolor": tokens.surface,
                "bordercolor": tokens.border,
                "font": {"color": tokens.text_primary, "size": 12, "family": TYPOGRAPHY.font_sans},
            },
            "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
        }
    }


def apply_plotly_theme(fig, height: int = 420, title: str = None):
    """Apply consistent theme to a Plotly figure."""
    tokens = get_color_tokens()
    is_dark = get_current_theme() == "dark"
    
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=TYPOGRAPHY.font_sans,
            color=tokens.chart_text,
            size=13,
        ),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(size=17, color=tokens.text_primary),
            )
            if title
            else None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=tokens.text_secondary),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=tokens.chart_grid,
            linecolor=tokens.chart_axis,
            tickcolor=tokens.chart_axis,
            tickfont=dict(color=tokens.text_secondary, size=11),
            titlefont=dict(color=tokens.text_secondary, size=12),
            zerolinecolor=tokens.border_subtle,
        ),
        yaxis=dict(
            gridcolor=tokens.chart_grid,
            linecolor=tokens.chart_axis,
            tickcolor=tokens.chart_axis,
            tickfont=dict(color=tokens.text_secondary, size=11),
            titlefont=dict(color=tokens.text_secondary, size=12),
            zerolinecolor=tokens.border_subtle,
        ),
        colorway=SEGMENT_PALETTE,
        hoverlabel=dict(
            bgcolor=tokens.surface,
            bordercolor=tokens.border,
            font=dict(color=tokens.text_primary, size=12, family=TYPOGRAPHY.font_sans),
        ),
    )
    return fig


# Export commonly used items
__all__ = [
    "get_current_theme",
    "get_color_tokens",
    "inject_global_css",
    "get_status_color",
    "get_action_color",
    "get_segment_color",
    "get_plotly_theme",
    "apply_plotly_theme",
    "STATUS_COLORS",
    "ACTION_COLORS",
    "SEGMENT_PALETTE",
    "PLOTLY_CONFIG",
    "SPACING",
    "RADIUS",
    "SHADOW",
    "TYPOGRAPHY",
    "LAYOUT",
    "SEED",
]