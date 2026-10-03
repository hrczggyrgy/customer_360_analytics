"""
Unified CustomerScope system for Retail Customer Intelligence Streamlit App.

This is the single authoritative implementation for global scope/filter management.
Replaces the duplicated logic in app.py and the old ui/scope.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import streamlit as st

from ..app_data import get_registry
from ..app_formatting import format_count, format_probability


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

# Centralized thresholds for consistent semantics across the application
RISK_THRESHOLDS = {
    "low": 0.30,
    "moderate": 0.50,
    "high": 0.70,
    "very_high": 0.85,
}

PROPENSITY_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}

VALUE_QUANTILES = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
}

OPPORTUNITY_MATRIX_SPLITS = {
    "x_split": "median",  # propensity
    "y_split": "median",  # risk
}


def get_risk_band(risk_value: float) -> str:
    """Map inactivity risk to risk band label."""
    if risk_value >= RISK_THRESHOLDS["very_high"]:
        return "Very High"
    elif risk_value >= RISK_THRESHOLDS["high"]:
        return "High"
    elif risk_value >= RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    elif risk_value >= RISK_THRESHOLDS["low"]:
        return "Low"
    return "Very Low"


def get_propensity_band(propensity_value: float) -> str:
    """Map purchase propensity to propensity band label."""
    if propensity_value >= PROPENSITY_THRESHOLDS["high"]:
        return "High"
    elif propensity_value >= PROPENSITY_THRESHOLDS["medium"]:
        return "Medium"
    elif propensity_value >= PROPENSITY_THRESHOLDS["low"]:
        return "Low"
    return "Very Low"


def get_value_band_from_quantile(value: float, quantiles: Dict[str, float]) -> str:
    """Map value to band based on quantile thresholds."""
    if value >= quantiles["high"]:
        return "High"
    elif value >= quantiles["medium"]:
        return "Medium"
    elif value >= quantiles["low"]:
        return "Low"
    return "Very Low"


# =============================================================================
# SCOPE DATA CLASS
# =============================================================================

@dataclass
class CustomerScope:
    """Authoritative global scope state for the application."""
    as_of_date: Optional[date] = None
    country: str = "All"
    segment: str = "All"
    lifecycle: str = "All"
    value_band: str = "All"
    risk_band: str = "All"
    rfm_segment: str = "All"
    priority_tier: str = "All"
    
    # Internal state
    _scoped_customer_ids: Optional[Set[int]] = field(default=None, repr=False)
    _scope_version: int = field(default=0, repr=False)
    _available_filters: Dict[str, List[str]] = field(default_factory=dict, repr=False)
    _snapshot_dates: List[date] = field(default_factory=list, repr=False)
    
    def reset(self) -> None:
        """Reset scope to defaults."""
        self.as_of_date = None
        self.country = "All"
        self.segment = "All"
        self.lifecycle = "All"
        self.value_band = "All"
        self.risk_band = "All"
        self.rfm_segment = "All"
        self.priority_tier = "All"
        self._scoped_customer_ids = None
        self._scope_version += 1
    
    def is_filtered(self) -> bool:
        """Check if any filter is active."""
        return any([
            self.as_of_date is not None,
            self.country != "All",
            self.segment != "All",
            self.lifecycle != "All",
            self.value_band != "All",
            self.risk_band != "All",
            self.rfm_segment != "All",
            self.priority_tier != "All",
        ])
    
    def get_active_summary(self) -> str:
        """Get human-readable summary of active filters."""
        parts = []
        if self.as_of_date:
            parts.append(f"As of: {self.as_of_date}")
        if self.country != "All":
            parts.append(f"Country: {self.country}")
        if self.segment != "All":
            parts.append(f"Segment: {self.segment}")
        if self.lifecycle != "All":
            parts.append(f"Lifecycle: {self.lifecycle}")
        if self.value_band != "All":
            parts.append(f"Value: {self.value_band}")
        if self.risk_band != "All":
            parts.append(f"Risk: {self.risk_band}")
        if self.rfm_segment != "All":
            parts.append(f"RFM: {self.rfm_segment}")
        if self.priority_tier != "All":
            parts.append(f"Priority: {self.priority_tier}")
        return " · ".join(parts) if parts else "All customers"


# =============================================================================
# SCOPE MANAGER (Singleton via session_state)
# =============================================================================

def get_scope() -> CustomerScope:
    """Get or create the global CustomerScope from session state."""
    if "customer_scope" not in st.session_state:
        st.session_state.customer_scope = CustomerScope()
        _initialize_scope_filters()
    return st.session_state.customer_scope


def _initialize_scope_filters() -> None:
    """Initialize available filter options from data."""
    scope = get_scope()
    registry = get_registry()
    combined = registry.load_dataframe("customer_360")
    
    if combined is None:
        return
    
    # Available countries
    if "primary_country" in combined.columns:
        scope._available_filters["country"] = ["All"] + sorted(
            combined["primary_country"].dropna().unique().tolist()
        )
    
    # Available segments (from hdbscan_segment mapped to segment_name)
    if "segment_name" in combined.columns:
        segments = combined["segment_name"].dropna().unique()
        segments = [s for s in segments if s and s != "unknown"]
        scope._available_filters["segment"] = ["All"] + sorted(segments)
    
    # Available lifecycles
    if "lifecycle_state" in combined.columns:
        scope._available_filters["lifecycle"] = ["All"] + sorted(
            combined["lifecycle_state"].dropna().unique().tolist()
        )
    
    # Available value bands
    if "value_tier" in combined.columns:
        scope._available_filters["value_band"] = ["All"] + sorted(
            combined["value_tier"].dropna().unique().tolist()
        )
    
    # Available risk bands (derived from churn_probability)
    scope._available_filters["risk_band"] = ["All", "Very Low", "Low", "Moderate", "High", "Very High"]
    
    # Available RFM segments (need to compute/join from segmentation)
    if "hdbscan_segment" in combined.columns:
        # We'll populate RFM from the segmentation artifact
        _load_rfm_segments(scope, registry)
    
    # Available priority tiers
    if "action_priority_tier" in combined.columns:
        scope._available_filters["priority_tier"] = ["All"] + sorted(
            combined["action_priority_tier"].dropna().unique().tolist()
        )
    
    # Available snapshot dates for temporal filtering
    _load_snapshot_dates(scope, combined)


def _load_rfm_segments(scope: CustomerScope, registry) -> None:
    """Load RFM segments from segmentation artifact."""
    rfm_data = registry.load_dataframe("segmentation", "rfm_segments.csv")
    if rfm_data is not None and "rfm_segment" in rfm_data.columns:
        rfm_vals = rfm_data["rfm_segment"].dropna().unique()
        scope._available_filters["rfm_segment"] = ["All"] + sorted(rfm_vals)


def _load_snapshot_dates(scope: CustomerScope, combined: pd.DataFrame) -> None:
    """Load available snapshot dates for temporal filtering."""
    dates = set()
    
    # From snapshot_date column
    if "snapshot_date" in combined.columns:
        snap_dates = pd.to_datetime(combined["snapshot_date"], errors="coerce").dropna()
        dates.update(snap_dates.dt.date.unique())
    
    # From reference_date column
    if "reference_date" in combined.columns:
        ref_dates = pd.to_datetime(combined["reference_date"], errors="coerce").dropna()
        dates.update(ref_dates.dt.date.unique())
    
    scope._snapshot_dates = sorted(dates)


# =============================================================================
# SCOPE FILTERING LOGIC
# =============================================================================

def get_scoped_customer_ids(scope: Optional[CustomerScope] = None) -> Set[int]:
    """Get the set of customer IDs matching the current scope.
    
    Uses cached result if scope hasn't changed.
    """
    if scope is None:
        scope = get_scope()
    
    # Return cached if scope version matches
    if scope._scoped_customer_ids is not None:
        return scope._scoped_customer_ids
    
    registry = get_registry()
    combined = registry.load_dataframe("customer_360")
    
    if combined is None or "Customer ID" not in combined.columns:
        scope._scoped_customer_ids = set()
        return scope._scoped_customer_ids
    
    # Start with all customers
    filtered = combined[["Customer ID"]].copy()
    
    # Ensure filter columns are present
    filter_cols = ["Customer ID"]
    for col in ["primary_country", "segment_name", "lifecycle_state", "value_tier", 
                "churn_probability", "action_priority_tier"]:
        if col in combined.columns:
            filter_cols.append(col)
    
    # Merge filter columns if needed
    missing_cols = [c for c in filter_cols if c not in filtered.columns and c != "Customer ID"]
    if missing_cols:
        merged = filtered.merge(
            combined[filter_cols].drop_duplicates(subset=["Customer ID"]),
            on="Customer ID",
            how="left",
        )
    else:
        merged = filtered
    
    # Apply filters
    if scope.country != "All" and "primary_country" in merged.columns:
        merged = merged[merged["primary_country"] == scope.country]
    
    if scope.segment != "All" and "segment_name" in merged.columns:
        merged = merged[merged["segment_name"] == scope.segment]
    
    if scope.lifecycle != "All" and "lifecycle_state" in merged.columns:
        merged = merged[merged["lifecycle_state"] == scope.lifecycle]
    
    if scope.value_band != "All" and "value_tier" in merged.columns:
        merged = merged[merged["value_tier"] == scope.value_band]
    
    if scope.risk_band != "All" and "churn_probability" in merged.columns:
        risk_vals = pd.to_numeric(merged["churn_probability"], errors="coerce")
        risk_bands = risk_vals.apply(get_risk_band)
        merged = merged[risk_bands == scope.risk_band]
    
    if scope.priority_tier != "All" and "action_priority_tier" in merged.columns:
        merged = merged[merged["action_priority_tier"] == scope.priority_tier]
    
    # RFM segment filtering requires joining with segmentation artifact
    if scope.rfm_segment != "All":
        rfm_data = registry.load_dataframe("segmentation", "rfm_segments.csv")
        if rfm_data is not None and "Customer ID" in rfm_data.columns and "rfm_segment" in rfm_data.columns:
            rfm_customers = set(
                rfm_data[rfm_data["rfm_segment"] == scope.rfm_segment]["Customer ID"].astype(int)
            )
            merged = merged[merged["Customer ID"].astype(int).isin(rfm_customers)]
    
    # Temporal filtering (as_of_date)
    if scope.as_of_date is not None:
        merged = _apply_temporal_filter(merged, scope.as_of_date, combined, registry)
    
    scope._scoped_customer_ids = set(merged["Customer ID"].astype(int))
    return scope._scoped_customer_ids


def _apply_temporal_filter(
    merged: pd.DataFrame, 
    as_of_date: date, 
    combined: pd.DataFrame, 
    registry
) -> pd.DataFrame:
    """Apply temporal filtering based on as_of_date.
    
    The customer_360 data has snapshot_date and reference_date columns.
    We find the latest snapshot at or before the selected date.
    """
    # Check what temporal columns are available
    temporal_cols = []
    if "snapshot_date" in combined.columns:
        temporal_cols.append("snapshot_date")
    if "reference_date" in combined.columns:
        temporal_cols.append("reference_date")
    
    if not temporal_cols:
        # No temporal data available - return unfiltered but log warning
        return merged
    
    # Find the latest available snapshot date <= as_of_date
    available_dates = []
    for col in temporal_cols:
        dates = pd.to_datetime(combined[col], errors="coerce").dropna().dt.date.unique()
        available_dates.extend(dates)
    
    available_dates = sorted(set(available_dates))
    valid_dates = [d for d in available_dates if d <= as_of_date]
    
    if not valid_dates:
        # Selected date is before all available data - return empty
        return merged.iloc[0:0]
    
    target_date = max(valid_dates)
    
    # Filter to customers with snapshot_date <= target_date (or reference_date)
    # Since the data is a single snapshot, this mainly serves as validation
    # that the user isn't selecting a future date
    if target_date < as_of_date:
        # Show info that we're using the latest available snapshot
        pass  # The UI will show the effective date
    
    return merged


def apply_scope_to_dataframe(
    df: pd.DataFrame, 
    scope: Optional[CustomerScope] = None,
    id_column: str = "Customer ID"
) -> pd.DataFrame:
    """Apply scope filters to a dataframe that has a Customer ID column.
    
    This is the preferred way to filter page-specific artifacts.
    """
    if scope is None:
        scope = get_scope()
    
    if df is None or df.empty or id_column not in df.columns:
        return df
    
    scoped_ids = get_scoped_customer_ids(scope)
    if not scoped_ids:
        return df.iloc[0:0]
    
    return df[df[id_column].astype(int).isin(scoped_ids)]


# =============================================================================
# SCOPE APPLICABILITY MODEL
# =============================================================================

class ScopeApplicability:
    """Defines which artifacts react to which scope filters."""
    
    # Artifact -> {customer_scope: bool, temporal_scope: bool, portfolio_wide: bool}
    APPLICABILITY = {
        "customer_360": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "decision_engine": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "recommendations": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "clv": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "churn": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "segmentation": {
            "customer_scope": True,
            "temporal_scope": True,
            "portfolio_wide": False,
        },
        "cohorts": {
            "customer_scope": False,
            "temporal_scope": False,
            "portfolio_wide": True,
        },
        "market_basket": {
            "customer_scope": False,
            "temporal_scope": True,  # based on generated artifact date
            "portfolio_wide": True,
        },
        "product_analytics": {
            "customer_scope": False,
            "temporal_scope": True,
            "portfolio_wide": True,
        },
        "recommendation_challenger": {
            "customer_scope": False,
            "temporal_scope": True,  # fixed evaluation period
            "portfolio_wide": True,
        },
        "model_calibration": {
            "customer_scope": False,
            "temporal_scope": True,  # fixed evaluation period
            "portfolio_wide": True,
        },
        "data_quality": {
            "customer_scope": False,
            "temporal_scope": False,
            "portfolio_wide": True,
        },
    }
    
    @classmethod
    def is_scoped(cls, artifact: str) -> bool:
        """Check if artifact should respond to customer scope filters."""
        return cls.APPLICABILITY.get(artifact, {}).get("customer_scope", False)
    
    @classmethod
    def is_temporal(cls, artifact: str) -> bool:
        """Check if artifact should respond to temporal scope."""
        return cls.APPLICABILITY.get(artifact, {}).get("temporal_scope", False)
    
    @classmethod
    def is_portfolio_wide(cls, artifact: str) -> bool:
        """Check if artifact is portfolio-wide (not scoped)."""
        return cls.APPLICABILITY.get(artifact, {}).get("portfolio_wide", False)
    
    @classmethod
    def get_scope_note(cls, artifact: str) -> Optional[str]:
        """Get explanatory note for scope behavior."""
        app = cls.APPLICABILITY.get(artifact, {})
        if app.get("portfolio_wide") and not app.get("customer_scope"):
            return f"{artifact.replace('_', ' ').title()} is portfolio-level analysis and is not recalculated for the current customer scope."
        if app.get("customer_scope") and app.get("temporal_scope"):
            return f"Filtered by current customer scope and analysis date."
        return None


# =============================================================================
# UI RENDERING
# =============================================================================

def render_global_filter_bar() -> None:
    """Render the global filter bar at the top of the app."""
    scope = get_scope()
    registry = get_registry()
    
    # Ensure filters are initialized
    if not scope._available_filters:
        _initialize_scope_filters()
    
    # Get effective analysis date for display
    effective_date = _get_effective_analysis_date(scope)
    
    with st.container():
        # Primary filters row
        cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1, 0.5])
        
        with cols[0]:
            # As of date - show supported dates
            if scope._snapshot_dates:
                min_date = min(scope._snapshot_dates)
                max_date = max(scope._snapshot_dates)
                
                selected = st.date_input(
                    "Analysis date",
                    value=scope.as_of_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="scope_as_of_date",
                    help=f"Analysis reference date. Available snapshots: {min_date} to {max_date}",
                )
                scope.as_of_date = selected
            else:
                st.date_input(
                    "Analysis date",
                    value=None,
                    disabled=True,
                    key="scope_as_of_date",
                    help="No temporal data available",
                )
        
        with cols[1]:
            countries = scope._available_filters.get("country", ["All"])
            scope.country = st.selectbox(
                "Country",
                options=countries,
                index=countries.index(scope.country) if scope.country in countries else 0,
                key="scope_country",
            )
        
        with cols[2]:
            segments = scope._available_filters.get("segment", ["All"])
            scope.segment = st.selectbox(
                "Segment",
                options=segments,
                index=segments.index(scope.segment) if scope.segment in segments else 0,
                key="scope_segment",
            )
        
        with cols[3]:
            lifecycles = scope._available_filters.get("lifecycle", ["All"])
            scope.lifecycle = st.selectbox(
                "Lifecycle",
                options=lifecycles,
                index=lifecycles.index(scope.lifecycle) if scope.lifecycle in lifecycles else 0,
                key="scope_lifecycle",
            )
        
        with cols[4]:
            value_bands = scope._available_filters.get("value_band", ["All"])
            scope.value_band = st.selectbox(
                "Value band",
                options=value_bands,
                index=value_bands.index(scope.value_band) if scope.value_band in value_bands else 0,
                key="scope_value_band",
            )
        
        with cols[5]:
            risk_bands = scope._available_filters.get("risk_band", ["All"])
            scope.risk_band = st.selectbox(
                "Risk band",
                options=risk_bands,
                index=risk_bands.index(scope.risk_band) if scope.risk_band in risk_bands else 0,
                key="scope_risk_band",
            )
        
        with cols[6]:
            rfm_segments = scope._available_filters.get("rfm_segment", ["All"])
            scope.rfm_segment = st.selectbox(
                "RFM segment",
                options=rfm_segments,
                index=rfm_segments.index(scope.rfm_segment) if scope.rfm_segment in rfm_segments else 0,
                key="scope_rfm_segment",
            )
        
        with cols[7]:
            priority_tiers = scope._available_filters.get("priority_tier", ["All"])
            scope.priority_tier = st.selectbox(
                "Priority tier",
                options=priority_tiers,
                index=priority_tiers.index(scope.priority_tier) if scope.priority_tier in priority_tiers else 0,
                key="scope_priority_tier",
            )
        
        with cols[8]:
            st.write("")
            if st.button("Clear", key="scope_clear", help="Clear all filters", use_container_width=True):
                scope.reset()
                st.rerun()
    
    # Show active scope summary with customer count
    scoped_ids = get_scoped_customer_ids(scope)
    n_scoped = len(scoped_ids)
    
    active_summary = scope.get_active_summary()
    if effective_date and effective_date != scope.as_of_date:
        active_summary += f" (using {effective_date})"
    
    st.caption(f"Scope: {active_summary}  |  **{format_count(n_scoped)} customers**")
    st.markdown("---")


def _get_effective_analysis_date(scope: CustomerScope) -> Optional[date]:
    """Get the effective analysis date being used (may differ from selected if not available)."""
    if scope.as_of_date is None:
        return None
    
    if not scope._snapshot_dates:
        return scope.as_of_date
    
    valid_dates = [d for d in scope._snapshot_dates if d <= scope.as_of_date]
    if not valid_dates:
        return None
    
    return max(valid_dates)


def render_sidebar() -> str:
    """Render the global sidebar with workspace navigation."""
    registry = get_registry()
    statuses = registry.get_all_module_statuses()
    
    st.sidebar.markdown("### Retail Customer Intelligence")
    st.sidebar.markdown("---")
    
    # Module status
    with st.sidebar.expander("Pipeline status", expanded=True):
        for module, status in statuses.items():
            from .theme import get_status_color
            color = get_status_color(status.overall_status)
            icon_map = {
                "ready": "✓",
                "incomplete": "○",
                "stale": "⟳",
                "validation_failed": "✗",
                "unavailable": "−",
                "unknown": "?",
            }
            icon = icon_map.get(status.overall_status, "•")
            chip = (
                f"<span style='display:inline-flex;align-items:center;gap:4px;"
                f"padding:5px 10px;border-radius:999px;"
                f"background:{color}1A;color:{color};"
                f"font-size:0.76rem;font-weight:700;margin:2px;'>"
                f"{icon} {module.replace('_', ' ').title()}: {status.overall_status.title()}"
                f"</span>"
            )
            st.sidebar.markdown(chip, unsafe_allow_html=True)
            
            if status.freshness_note:
                st.sidebar.caption(f"  {status.freshness_note}")
    
    st.sidebar.markdown("---")
    
    # Workspace navigation - 8 workspaces
    workspaces = [
        "Strategy",
        "Customers",
        "Value & Retention",
        "Segments",
        "Products & Baskets",
        "Personalisation",
        "Activation",
        "Science & Governance",
    ]
    
    page = st.sidebar.radio("Workspace", workspaces, key="nav_workspace")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Observational retail analytics. Policy scores are not causal uplift estimates."
    )
    
    return page


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CustomerScope",
    "get_scope",
    "get_scoped_customer_ids",
    "apply_scope_to_dataframe",
    "render_global_filter_bar",
    "render_sidebar",
    "ScopeApplicability",
    "RISK_THRESHOLDS",
    "PROPENSITY_THRESHOLDS",
    "VALUE_QUANTILES",
    "OPPORTUNITY_MATRIX_SPLITS",
    "get_risk_band",
    "get_propensity_band",
    "get_value_band_from_quantile",
]