"""
Configuration loader for Retail Customer Intelligence Streamlit App.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# Module-level cache
_CONFIG: Optional[Dict[str, Any]] = None
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        # streamlit_app/ is one level down from project root
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


def get_config() -> Dict[str, Any]:
    """Load and return the project configuration."""
    global _CONFIG
    if _CONFIG is None:
        config_path = get_project_root() / "config" / "project.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            _CONFIG = yaml.safe_load(f)
        
        # Resolve relative paths to absolute paths
        _CONFIG = _resolve_paths(_CONFIG, get_project_root())
    
    return _CONFIG


def _resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Recursively resolve relative paths to absolute paths."""
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if isinstance(value, str) and _looks_like_path(value):
                # Resolve relative to project root
                path = Path(value)
                if not path.is_absolute():
                    result[key] = str(project_root / path)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = _resolve_paths(value, project_root)
            elif isinstance(value, list):
                result[key] = [_resolve_paths(item, project_root) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result
    elif isinstance(config, list):
        return [_resolve_paths(item, project_root) if isinstance(item, dict) else item for item in config]
    else:
        return config


def _looks_like_path(value: str) -> bool:
    """Heuristic to detect if a string value is a file path."""
    if not isinstance(value, str):
        return False
    # Check for common path patterns
    path_indicators = [".", "/", "\\", "output", "data_", "csv", "parquet", "json", "xlsx"]
    return any(indicator in value for indicator in path_indicators)


def get_output_dir(module: str) -> Path:
    """Get the output directory for a module."""
    config = get_config()
    rel_path = config.get("outputs", {}).get(module)
    if not rel_path:
        raise ValueError(f"No output directory configured for module: {module}")
    return Path(rel_path)


def get_raw_input_path() -> Path:
    """Get the raw input data path."""
    config = get_config()
    rel_path = config.get("data", {}).get("raw", {}).get("online_retail_ii")
    if not rel_path:
        raise ValueError("No raw input path configured")
    return Path(rel_path)


def get_canonical_transactions_path() -> Path:
    """Get the canonical transactions path."""
    config = get_config()
    rel_path = config.get("data", {}).get("processed", {}).get("canonical_transactions")
    if not rel_path:
        raise ValueError("No canonical transactions path configured")
    return Path(rel_path)


def get_customer_month_path() -> Path:
    """Get the customer-month panel path."""
    config = get_config()
    rel_path = config.get("data", {}).get("processed", {}).get("customer_month")
    if not rel_path:
        raise ValueError("No customer month path configured")
    return Path(rel_path)


def get_model_param(category: str, param: str, default: Any = None) -> Any:
    """Get a model parameter from config."""
    config = get_config()
    return config.get("model", {}).get(category, {}).get(param, default)