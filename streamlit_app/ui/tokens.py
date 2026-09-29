"""
Design Tokens for Retail Customer Intelligence Streamlit App.

Centralized design system tokens for colors, spacing, typography, and components.
Supports both light and dark themes through Streamlit's theme configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class ColorTokens:
    """Semantic color tokens that adapt to theme."""
    # Base
    background: str
    surface: str
    surface_subtle: str
    border: str
    border_subtle: str
    
    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    text_inverse: str
    
    # Brand
    primary: str
    primary_soft: str
    primary_strong: str
    
    # Semantic
    success: str
    success_soft: str
    warning: str
    warning_soft: str
    danger: str
    danger_soft: str
    info: str
    info_soft: str
    
    # Chart
    chart_grid: str
    chart_axis: str
    chart_text: str


@dataclass(frozen=True)
class SpacingTokens:
    """Consistent spacing scale."""
    xs: str = "4px"
    sm: str = "8px"
    md: str = "12px"
    lg: str = "16px"
    xl: str = "24px"
    xxl: str = "32px"
    xxxl: str = "40px"


@dataclass(frozen=True)
class RadiusTokens:
    """Border radius scale."""
    sm: str = "6px"
    md: str = "10px"
    lg: str = "14px"
    xl: str = "18px"
    full: str = "999px"


@dataclass(frozen=True)
class ShadowTokens:
    """Subtle shadow scale."""
    sm: str = "0 1px 2px rgba(0, 0, 0, 0.04)"
    md: str = "0 4px 8px rgba(0, 0, 0, 0.06)"
    lg: str = "0 8px 16px rgba(0, 0, 0, 0.08)"


@dataclass(frozen=True)
class TypographyTokens:
    """Typography scale."""
    # Font families
    font_sans: str = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_mono: str = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
    
    # Font sizes
    text_xs: str = "0.72rem"    # 11.5px
    text_sm: str = "0.82rem"    # 13px
    text_base: str = "0.92rem"  # 14.7px
    text_lg: str = "1rem"       # 16px
    text_xl: str = "1.125rem"   # 18px
    text_2xl: str = "1.375rem"  # 22px
    text_3xl: str = "1.75rem"   # 28px
    text_4xl: str = "2.25rem"   # 36px
    
    # Font weights
    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700
    
    # Line heights
    leading_tight: float = 1.2
    leading_snug: float = 1.4
    leading_normal: float = 1.55
    leading_relaxed: float = 1.6


@dataclass(frozen=True)
class LayoutTokens:
    """Layout constraints."""
    container_max_width: str = "1480px"
    sidebar_width: str = "280px"
    content_padding_top: str = "1.25rem"
    content_padding_bottom: str = "3rem"
    card_padding: str = "18px 20px"


# Light theme tokens (default)
LIGHT_TOKENS = ColorTokens(
    background="#fbfcfe",
    surface="#ffffff",
    surface_subtle="#f7f8fa",
    border="#e4e7ec",
    border_subtle="#edf0f5",
    text_primary="#172033",
    text_secondary="#3a4558",
    text_muted="#697386",
    text_inverse="#ffffff",
    primary="#315efb",
    primary_soft="#eef2ff",
    primary_strong="#1e3dc9",
    success="#218739",
    success_soft="#dcfce7",
    warning="#a56600",
    warning_soft="#fef3c7",
    danger="#e11d48",
    danger_soft="#fee2e2",
    info="#0ea5e9",
    info_soft="#e0f2fe",
    chart_grid="#e4e7ec",
    chart_axis="#697386",
    chart_text="#172033",
)

# Dark theme tokens
DARK_TOKENS = ColorTokens(
    background="#0f1419",
    surface="#1a1f2e",
    surface_subtle="#222838",
    border="#2d3548",
    border_subtle="#353d50",
    text_primary="#f1f3f5",
    text_secondary="#d4d8de",
    text_muted="#8b95a1",
    text_inverse="#0f1419",
    primary="#5c86ff",
    primary_soft="#1e2a4a",
    primary_strong="#8db0ff",
    success="#4ade80",
    success_soft="#0f2a1a",
    warning="#fbbf24",
    warning_soft="#2a2200",
    danger="#f87171",
    danger_soft="#2a0f0f",
    info="#38bdf8",
    info_soft="#0f2a3a",
    chart_grid="#2d3548",
    chart_axis="#8b95a1",
    chart_text="#f1f3f5",
)

# Shared tokens
SPACING = SpacingTokens()
RADIUS = RadiusTokens()
SHADOW = ShadowTokens()
TYPOGRAPHY = TypographyTokens()
LAYOUT = LayoutTokens()


def get_theme_tokens(theme: Literal["light", "dark"] = "light") -> ColorTokens:
    """Get color tokens for the specified theme."""
    return LIGHT_TOKENS if theme == "light" else DARK_TOKENS


def generate_css_variables(tokens: ColorTokens, prefix: str = "") -> str:
    """Generate CSS custom properties from color tokens."""
    lines = []
    for key, value in tokens.__dict__.items():
        css_key = key.replace("_", "-")
        lines.append(f"  {prefix}--{css_key}: {value};")
    return "\n".join(lines)


# Status color map (semantic, not CSS color names)
STATUS_COLORS: Dict[str, str] = {
    "ready": "#218739",
    "incomplete": "#a56600",
    "stale": "#e11d48",
    "validation_failed": "#e11d48",
    "unavailable": "#697386",
    "unknown": "#697386",
}

# Action colors (colorblind-safe palette)
ACTION_COLORS: Dict[str, str] = {
    "protect_value": "#16a34a",      # Green
    "accelerate_purchase": "#2563eb", # Blue
    "reactivate": "#d97706",          # Amber
    "cross_sell": "#7c3aed",          # Purple
    "nurture": "#0891b2",             # Cyan
    "monitor": "#64748b",             # Slate
}

# Segment palette (colorblind-safe, categorical)
SEGMENT_PALETTE = [
    "#2563eb", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#dc2626", "#ea580c", "#65a30d",
    "#db2777", "#4f46e5", "#0d9488", "#e11d48",
]

# Chart config
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

SEED = 42