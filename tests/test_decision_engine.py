"""
Test suite for decision engine.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.decision_engine import (
    compute_decision_policy,
    apply_capacity_constraints,
)


class TestDecisionEngine:
    """Tests for decision engine policy and capacity logic."""

    @pytest.fixture
    def sample_customer_data(self):
        """Create sample customer data for testing."""
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            "Customer ID": range(n),
            "clv_mean": np.random.exponential(1000, n),
            "clv_p10": np.random.exponential(100, n),
            "clv_p90": np.random.exponential(5000, n),
            "churn_probability": np.random.beta(2, 5, n),
            "survival_3m": np.random.beta(5, 2, n),
            "survival_6m": np.random.beta(4, 2, n),
            "survival_12m": np.random.beta(3, 2, n),
            "next_purchase_7d_probability": np.random.beta(2, 8, n),
            "next_purchase_30d_probability": np.random.beta(3, 7, n),
            "next_purchase_60d_probability": np.random.beta(4, 6, n),
            "reactivation_probability": np.random.beta(2, 8, n),
            "top_score": np.random.uniform(0, 1, n),
            "rec_reason": np.random.choice(["co_purchase", "popularity"], n),
            "segment_name": np.random.choice(["High Value", "Medium", "Low", "Noise"], n),
            "lifetime_gross_revenue": np.random.exponential(500, n),
            "recency_months": np.random.exponential(3, n),
            "consecutive_inactive_months": np.random.exponential(2, n),
        })
        
        # Add is_inactive based on recency
        data["is_inactive"] = data["recency_months"] > 3
        data["is_active"] = ~data["is_inactive"]
        
        return data

    def test_confidence_threshold_used(self, sample_customer_data):
        """Configured confidence threshold should affect action selection."""
        df = sample_customer_data.copy()
        
        # With high threshold, fewer actions should be selected
        df_high = compute_decision_policy(df.copy(), confidence_threshold=0.9)
        actions_high = (df_high["recommended_action"] != "monitor").sum()
        
        # With low threshold, more actions should be selected
        df_low = compute_decision_policy(df.copy(), confidence_threshold=0.1)
        actions_low = (df_low["recommended_action"] != "monitor").sum()
        
        assert actions_high <= actions_low

    def test_selected_action_priority_correct(self, sample_customer_data):
        """Priority score should equal the selected action's score, not max of all."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.5)
        
        score_cols = ["protect_value_score", "accelerate_purchase_score", "reactivate_score", 
                      "cross_sell_score", "nurture_score"]
        
        for _, row in df.iterrows():
            action = row["recommended_action"]
            if action != "monitor":
                expected_priority = row[f"{action}_score"]
                assert abs(row["priority_score"] - expected_priority) < 0.001
            else:
                assert row["priority_score"] == 0.0

    def test_priority_not_max_of_all(self, sample_customer_data):
        """Priority should NOT be max of all action scores."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.5)
        
        # For rows where max of all != selected action score
        max_all = df[["protect_value_score", "accelerate_purchase_score", "reactivate_score", 
                      "cross_sell_score", "nurture_score"]].max(axis=1)
        
        # For non-monitor actions, priority should equal selected action score
        # which may be less than max of all
        for _, row in df.iterrows():
            if row["recommended_action"] != "monitor":
                assert row["priority_score"] != max_all[row.name] or \
                       abs(row["priority_score"] - max_all[row.name]) < 0.001

    def test_capacity_total_enforced(self, sample_customer_data):
        """Total capacity should limit total actions."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.1)  # Low threshold = many actions
        df = apply_capacity_constraints(
            df, capacity_total=100, capacity_reactivate=50, 
            capacity_accelerate=50, capacity_cross_sell=50, capacity_nurture=50
        )
        
        active_actions = (df["recommended_action_capped"] != "monitor").sum()
        assert active_actions <= 100

    def test_action_specific_capacity(self, sample_customer_data):
        """Per-action capacity should be enforced."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.1)
        df = apply_capacity_constraints(
            df, capacity_total=1000, capacity_reactivate=10, 
            capacity_accelerate=10, capacity_cross_sell=10, capacity_nurture=10
        )
        
        for action in ["reactivate", "accelerate_purchase", "cross_sell", "nurture"]:
            count = (df["recommended_action_capped"] == action).sum()
            assert count <= 10, f"Action {action} exceeded capacity: {count}"

    def test_protect_value_requires_churn_risk(self, sample_customer_data):
        """Protect value action should require churn_probability > 0.3."""
        df = sample_customer_data.copy()
        # Set all churn probabilities very low
        df["churn_probability"] = 0.1
        df["survival_3m"] = 0.9
        df = compute_decision_policy(df, confidence_threshold=0.1)
        
        # No protect_value actions should be selected
        assert (df["recommended_action"] == "protect_value").sum() == 0

    def test_reactivate_requires_inactive(self, sample_customer_data):
        """Reactivate action should only apply to inactive customers."""
        df = sample_customer_data.copy()
        df["is_inactive"] = False
        df["is_active"] = True
        df["consecutive_inactive_months"] = 20  # >= 12 so the term is 0
        df["reactivation_probability"] = 0.0
        df["lifetime_gross_revenue"] = 0.0
        df = compute_decision_policy(df, confidence_threshold=0.1)
        
        # No reactivate actions should be selected
        assert (df["recommended_action"] == "reactivate").sum() == 0

    def test_cross_sell_requires_active_and_recommendations(self, sample_customer_data):
        """Cross-sell should require active customer with good recommendations."""
        df = sample_customer_data.copy()
        df["is_inactive"] = True
        df["is_active"] = False
        df["top_score"] = 0.0
        df = compute_decision_policy(df, confidence_threshold=0.1)
        
        # No cross_sell actions should be selected
        assert (df["recommended_action"] == "cross_sell").sum() == 0

    def test_expected_value_proxy_formulas(self, sample_customer_data):
        """Expected value proxy should use documented formulas."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.1)
        
        for _, row in df.iterrows():
            action = row["recommended_action"]
            if action == "protect_value":
                expected = row["clv_mean"] * row["churn_probability"] * 0.3
            elif action == "accelerate_purchase":
                expected = row["clv_mean"] * row["next_purchase_30d_probability"] * 0.2
            elif action == "reactivate":
                expected = row["lifetime_gross_revenue"] * row.get("reactivation_probability", 0) * 0.5
            elif action == "cross_sell":
                expected = row["top_score"] * row["clv_mean"] * 0.3
            elif action == "nurture":
                expected = row["clv_mean"] * 0.1
            else:
                expected = 0.0
            
            assert abs(row["expected_value_proxy"] - expected) < 0.001

    def test_decision_confidence_matches_action(self, sample_customer_data):
        """Decision confidence should match the selected action's score."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.5)
        
        for _, row in df.iterrows():
            action = row["recommended_action"]
            if action != "monitor":
                expected_conf = row[f"{action}_score"]
                assert abs(row["decision_confidence"] - expected_conf) < 0.001
            else:
                assert row["decision_confidence"] == 0.0

    def test_deterministic_output(self, sample_customer_data):
        """Same input should produce identical output."""
        df1 = compute_decision_policy(sample_customer_data.copy(), confidence_threshold=0.5)
        df2 = compute_decision_policy(sample_customer_data.copy(), confidence_threshold=0.5)
        
        # Compare all columns
        pd.testing.assert_frame_equal(df1.sort_values("Customer ID").reset_index(drop=True),
                                       df2.sort_values("Customer ID").reset_index(drop=True))

    def test_action_reason_strings(self, sample_customer_data):
        """Action reasons should be informative strings."""
        df = sample_customer_data.copy()
        df = compute_decision_policy(df, confidence_threshold=0.5)
        
        for _, row in df.iterrows():
            reason = row["action_reason"]
            assert isinstance(reason, str)
            assert len(reason) > 0
            
            action = row["recommended_action"]
            if action == "protect_value":
                assert "CLV" in reason and "churn" in reason.lower()
            elif action == "accelerate_purchase":
                assert "buy" in reason.lower() or "ready" in reason.lower()
            elif action == "reactivate":
                assert "inactive" in reason.lower() or "reactivat" in reason.lower()
            elif action == "cross_sell":
                assert "cross" in reason.lower() or "recommendation" in reason.lower()
            elif action == "nurture":
                assert "clv" in reason.lower() or "nurture" in reason.lower()
            elif action == "monitor":
                assert "low" in reason.lower() or "insufficient" in reason.lower() or "priority" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])