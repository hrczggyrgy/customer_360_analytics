"""
Global scope and filter utilities for Retail Customer Intelligence Streamlit App.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..app_data import get_registry


def init_global_scope() -> None:
    """Initialize global scope state in session."""
    if "global_scope" not in st.session_state:
        st.session_state.global_scope = {
            "as_of_date": None,
            "country": "All",
            "segment": "All",
            "lifecycle": "All",
            "value_band": "All",
            "risk_band": "All",
            "rfm_segment": "All",
        }
    
    if "scope_dirty" not in st.session_state:
        st.session_state.scope_dirty = False


def apply_global_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Apply global scope filters to a dataframe with Customer ID."""
    if df is None or df.empty:
        return df
    
    scope = st.session_state.global_scope
    result = df.copy()
    
    # Need to join with customer_360 for filter columns if not present
    registry = get_registry()
    combined = registry.load_dataframe("customer_360")
    
    if combined is not None:
        filter_cols = ["Customer ID"]
        for col in ["primary_country", "segment_name", "lifecycle_state", "value_tier", "action_priority_tier"]:
            if col in combined.columns:
                filter_cols.append(col)
        
        # Merge filter columns if not already present
        missing_cols = [c for c in filter_cols if c not in result.columns and c != "Customer ID"]
        if missing_cols:
            result = result.merge(
                combined[filter_cols].drop_duplicates(subset=["Customer ID"]),
                on="Customer ID",
                how="left",
            )
    
    # Apply filters
    if scope["country"] != "All" and "primary_country" in result.columns:
        result = result[result["primary_country"] == scope["country"]]
    
    if scope["segment"] != "All" and "segment_name" in result.columns:
        result = result[result["segment_name"] == scope["segment"]]
    
    if scope["lifecycle"] != "All" and "lifecycle_state" in result.columns:
        result = result[result["lifecycle_state"] == scope["lifecycle"]]
    
    if scope["value_band"] != "All" and "value_tier" in result.columns:
        result = result[result["value_tier"] == scope["value_band"]]
    
    if scope["risk_band"] != "All" and "action_priority_tier" in result.columns:
        result = result[result["action_priority_tier"] == scope["risk_band"]]
    
    return result


__all__ = [
    "init_global_scope",
    "apply_global_scope",
]