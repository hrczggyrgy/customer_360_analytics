"""
Compatibility shim for legacy app_components imports.

This module re-exports from the new centralized ui package.
All new code should import from streamlit_app.ui directly.
"""

from __future__ import annotations

from .ui import (
    render_kpi_card,
    render_science_card,
    render_customer_selector,
    render_customer_header,
    render_customer_metric_row,
    render_evidence_table,
    render_missing,
    render_action_summary_table,
    render_status_chip,
    render_download_button,
    render_page_hero,
    render_section_label,
    HERO_COPY,
    ACTION_COLORS,
    SEGMENT_PALETTE,
    STATUS_COLORS,
)

__all__ = [
    "render_kpi_card",
    "render_science_card",
    "render_customer_selector",
    "render_customer_header",
    "render_customer_metric_row",
    "render_evidence_table",
    "render_missing",
    "render_action_summary_table",
    "render_status_chip",
    "render_download_button",
    "render_page_hero",
    "render_section_label",
    "HERO_COPY",
    "ACTION_COLORS",
    "SEGMENT_PALETTE",
    "STATUS_COLORS",
]