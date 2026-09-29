"""
Tests for Semantic Formatters - Contract and Edge Cases
"""
import pytest
import math
from datetime import date, datetime
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_ratio,
    format_count,
    format_date,
    format_month,
    format_duration_months,
    format_days,
    format_score,
    auto_format,
    infer_semantic_type,
    SemanticType,
    COLUMN_FORMAT_MAP,
    format_dataframe_columns,
)


class TestFormatCurrency:
    def test_standard_values(self):
        assert format_currency(0) == "£0"
        assert format_currency(100) == "£100"
        assert format_currency(1234) == "£1.2K"
        assert format_currency(1_234_567) == "£1.2M"

    def test_decimal_precision(self):
        assert format_currency(1234.56, decimals=2) == "£1.23K"
        assert format_currency(12345, decimals=2) == "£12.35K"

    def test_negative_values(self):
        assert format_currency(-1000) == "-£1.0K"
        assert format_currency(-1_500_000) == "-£1.5M"

    def test_none_and_nan(self):
        assert format_currency(None) == "—"
        assert format_currency(float("nan")) == "—"
        assert format_currency(float("inf")) == "—"


class TestFormatProbability:
    def test_standard_probabilities(self):
        assert format_probability(0.0) == "0.0%"
        assert format_probability(0.5) == "50.0%"
        assert format_probability(0.75) == "75.0%"
        assert format_probability(1.0) == "100.0%"

    def test_precision(self):
        assert format_probability(0.1234, decimals=2) == "12.34%"
        assert format_probability(0.1234, decimals=3) == "12.340%"

    def test_out_of_range(self):
        assert format_probability(1.5) == "Invalid"
        assert format_probability(-0.1) == "Invalid"

    def test_none_and_nan(self):
        assert format_probability(None) == "—"
        assert format_probability(float("nan")) == "—"


class TestFormatPercent:
    def test_probability_scale(self):
        assert format_percent(0.25) == "25.0%"
        assert format_percent(0.5) == "50.0%"

    def test_percentage_scale(self):
        assert format_percent(25) == "25.0%"
        assert format_percent(75.5) == "75.5%"

    def test_negative(self):
        assert format_percent(-10) == "-10.0%"

    def test_none_and_nan(self):
        assert format_percent(None) == "—"
        assert format_percent(float("nan")) == "—"


class TestFormatDays:
    def test_integer_days(self):
        assert format_days(45) == "45 days"
        assert format_days(1) == "1 days"

    def test_decimal_days(self):
        assert format_days(45.5, decimals=1) == "45.5 days"

    def test_none_and_nan(self):
        assert format_days(None) == "—"
        assert format_days(float("nan")) == "—"


class TestFormatScore:
    def test_standard_scores(self):
        assert format_score(0.8765) == "0.88"
        assert format_score(0.5) == "0.50"
        assert format_score(1.0) == "1.00"

    def test_precision(self):
        assert format_score(0.8765, decimals=4) == "0.8765"

    def test_none_and_nan(self):
        assert format_score(None) == "—"
        assert format_score(float("nan")) == "—"


class TestAutoFormat:
    def test_currency_columns(self):
        assert auto_format(1000, "revenue") == "£1.0K"
        assert auto_format(500, "avg_price") == "£500"

    def test_probability_columns(self):
        assert auto_format(0.75, "churn_probability") == "75.0%"
        assert auto_format(0.5, "survival_3m") == "50.0%"

    def test_percentage_columns(self):
        assert auto_format(0.25, "retention") == "25.0%"
        assert auto_format(50, "logo_retention") == "50.0%"

    def test_ratio_columns(self):
        assert auto_format(1.5, "ratio") == "1.50x"

    def test_count_columns(self):
        assert auto_format(1000, "orders") == "1,000"

    def test_score_columns(self):
        assert auto_format(0.87, "score") == "0.87"

    def test_date_columns(self):
        assert auto_format(date(2023, 1, 15), "first_purchase_date") == "2023-01-15"

    def test_month_columns(self):
        assert auto_format(date(2023, 1, 15), "cohort_month") == "2023-01"

    def test_duration_months_columns(self):
        assert auto_format(12.5, "recency_months") == "12.5 mo"

    def test_duration_days_columns(self):
        assert auto_format(45, "expected_days_to_next_purchase") == "45 days"

    def test_unknown_fallback_numeric(self):
        assert auto_format(123, "unknown_col") == "123"
        assert auto_format(12.34, "unknown_col") == "12.34"

    def test_unknown_fallback_string(self):
        assert auto_format("hello", "unknown_col") == "hello"


class TestInferSemanticType:
    def test_explicit_map_currency(self):
        assert infer_semantic_type("clv") == SemanticType.CURRENCY
        assert infer_semantic_type("revenue") == SemanticType.CURRENCY

    def test_explicit_map_probability(self):
        assert infer_semantic_type("churn_probability") == SemanticType.PROBABILITY
        assert infer_semantic_type("prob_churn") == SemanticType.PROBABILITY

    def test_explicit_map_percentage(self):
        assert infer_semantic_type("retention") == SemanticType.PERCENTAGE
        assert infer_semantic_type("repeat_customer_rate") == SemanticType.PERCENTAGE

    def test_explicit_map_ratio(self):
        assert infer_semantic_type("nrr") == SemanticType.RATIO
        assert infer_semantic_type("lift") == SemanticType.RATIO

    def test_explicit_map_count(self):
        assert infer_semantic_type("orders") == SemanticType.COUNT
        assert infer_semantic_type("customers") == SemanticType.COUNT

    def test_explicit_map_duration_months(self):
        assert infer_semantic_type("recency_months") == SemanticType.DURATION_MONTHS

    def test_explicit_map_duration_days(self):
        assert infer_semantic_type("expected_days_to_next_purchase") == SemanticType.DURATION_DAYS

    def test_explicit_map_score(self):
        assert infer_semantic_type("score") == SemanticType.SCORE
        assert infer_semantic_type("priority_score") == SemanticType.SCORE

    def test_explicit_map_date(self):
        assert infer_semantic_type("first_purchase_date") == SemanticType.DATE

    def test_explicit_map_month(self):
        assert infer_semantic_type("cohort_month") == SemanticType.MONTH

    def test_decision_confidence_is_score(self):
        assert infer_semantic_type("decision_confidence") == SemanticType.SCORE


class TestColumnFormatMap:
    def test_decision_confidence_in_map(self):
        assert "decision_confidence" in COLUMN_FORMAT_MAP
        assert COLUMN_FORMAT_MAP["decision_confidence"] == SemanticType.SCORE

    def test_expected_days_to_next_purchase_in_map(self):
        assert "expected_days_to_next_purchase" in COLUMN_FORMAT_MAP
        assert COLUMN_FORMAT_MAP["expected_days_to_next_purchase"] == SemanticType.DURATION_DAYS


class TestFormatDataFrameColumns:
    def test_formats_currency_columns(self):
        df = pd.DataFrame({"revenue": [1000, 2000, 3000]})
        result = format_dataframe_columns(df, ["revenue"])
        assert result["revenue"].tolist() == ["£1.0K", "£2.0K", "£3.0K"]

    def test_formats_probability_columns(self):
        df = pd.DataFrame({"churn_probability": [0.25, 0.75]})
        result = format_dataframe_columns(df, ["churn_probability"])
        assert result["churn_probability"].tolist() == ["25.0%", "75.0%"]

    def test_formats_percentage_columns(self):
        df = pd.DataFrame({"retention": [0.5, 0.8]})
        result = format_dataframe_columns(df, ["retention"])
        assert result["retention"].tolist() == ["50.0%", "80.0%"]

    def test_formats_duration_days(self):
        df = pd.DataFrame({"expected_days_to_next_purchase": [30, 60]})
        result = format_dataframe_columns(df, ["expected_days_to_next_purchase"])
        assert result["expected_days_to_next_purchase"].tolist() == ["30 days", "60 days"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = format_dataframe_columns(df)
        assert result.empty

    def test_preserves_unformatted_columns(self):
        df = pd.DataFrame({"revenue": [1000], "name": ["A"], "id": [1]})
        result = format_dataframe_columns(df, ["revenue"])
        assert result["name"].tolist() == ["A"]
        assert result["id"].tolist() == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])