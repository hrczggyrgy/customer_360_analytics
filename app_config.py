"""
Configuration loader for the Retail Customer Intelligence application.

Loads project configuration from config/project.yaml and provides
type-safe access to paths, model parameters, and dashboard settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class AppConfig:
    """Application configuration loaded from config/project.yaml."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config" / "project.yaml"

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def data_raw_online_retail_ii(self) -> Path:
        return Path(self._config["data"]["raw"]["online_retail_ii"])

    @property
    def data_processed_canonical_transactions(self) -> Path:
        return Path(self._config["data"]["processed"]["canonical_transactions"])

    @property
    def data_processed_customer_month(self) -> Path:
        return Path(self._config["data"]["processed"]["customer_month"])

    @property
    def output_dirs(self) -> Dict[str, Path]:
        return {k: Path(v) for k, v in self._config["outputs"].items()}

    @property
    def model_seed(self) -> int:
        return self._config["model"]["seed"]

    @property
    def model_segmentation(self) -> Dict[str, Any]:
        return self._config["model"]["segmentation"]

    @property
    def model_clv(self) -> Dict[str, Any]:
        return self._config["model"]["clv"]

    @property
    def model_churn(self) -> Dict[str, Any]:
        return self._config["model"]["churn"]

    @property
    def model_reactivation(self) -> Dict[str, Any]:
        return self._config["model"]["reactivation"]

    @property
    def model_product_analytics(self) -> Dict[str, Any]:
        return self._config["model"]["product_analytics"]

    @property
    def model_recommendations(self) -> Dict[str, Any]:
        return self._config["model"]["recommendations"]

    @property
    def model_decision_engine(self) -> Dict[str, Any]:
        return self._config["model"]["decision_engine"]

    @property
    def dashboard_port(self) -> int:
        return self._config["dashboard"]["port"]

    @property
    def dashboard_headless(self) -> bool:
        return self._config["dashboard"]["headless"]

    @property
    def dashboard_theme(self) -> Dict[str, str]:
        return self._config["dashboard"]["theme"]

    @property
    def logging_level(self) -> str:
        return self._config["logging"]["level"]

    @property
    def logging_format(self) -> str:
        return self._config["logging"]["format"]


# Singleton instance
_config_instance: Optional[AppConfig] = None


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """Get the application configuration singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig(config_path)
    return _config_instance


def reset_config() -> None:
    """Reset the configuration singleton (mainly for testing)."""
    global _config_instance
    _config_instance = None


# Environment-aware project root
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent


def get_output_dir(module: str) -> Path:
    """Get the output directory for a module from config."""
    config = get_config()
    return config.output_dirs.get(module, get_project_root() / f"{module}_output")


# Module name mappings for the app
MODULE_KEYS = [
    "data_quality",
    "customer_360",
    "segmentation",
    "cohorts",
    "clv",
    "churn",
    "reactivation",
    "product_analytics",
    "recommendations",
    "decision_engine",
]

# Human-readable module labels
MODULE_LABELS = {
    "data_quality": "Data Quality",
    "customer_360": "Customer 360",
    "segmentation": "Segmentation",
    "cohorts": "Cohorts",
    "clv": "CLV / Predictive Value",
    "churn": "Churn / Survival",
    "reactivation": "Reactivation",
    "product_analytics": "Product Analytics",
    "recommendations": "Recommendations",
    "decision_engine": "Decision Engine",
}