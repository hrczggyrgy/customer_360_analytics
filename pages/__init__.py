"""
Pages package for the Retail Customer Intelligence application.
"""

from pages.executive import render_executive_page
from pages.customer_360 import render_customer_360_page
from pages.segmentation import render_segmentation_page
from pages.cohorts import render_cohorts_page
from pages.predictive import render_predictive_page
from pages.retention import render_retention_page
from pages.recommendations import render_recommendations_page
from pages.decision_engine import render_decision_page
from pages.methodology import render_methodology_page

__all__ = [
    "render_executive_page",
    "render_customer_360_page",
    "render_segmentation_page",
    "render_cohorts_page",
    "render_predictive_page",
    "render_retention_page",
    "render_recommendations_page",
    "render_decision_page",
    "render_methodology_page",
]