"""
Configuration Management Utilities

This module provides utilities for loading and managing configuration files
for the TABULA-R2 project, with support for YAML configuration files and
environment variable overrides.

Functions:
    load_config: Load YAML configuration file
    merge_configs: Merge multiple configuration dictionaries
    get_config_value: Get configuration value with default fallback
"""

import yaml
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(
    config_path: Union[str, Path], validate_schema: bool = True
) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (Union[str, Path]): Path to YAML configuration file
        validate_schema (bool): Whether to validate configuration schema

    Returns:
        Dict[str, Any]: Loaded configuration dictionary

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")

    if config is None:
        config = {}

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    if validate_schema:
        validate_config_schema(config)

    logger.debug(f"Successfully loaded configuration from {config_path}")
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configurations override earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Dict[str, Any]: Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        merged = deep_merge(merged, config)

    return merged


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base (Dict[str, Any]): Base dictionary
        update (Dict[str, Any]): Dictionary with updates

    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    Environment variables should be prefixed with 'TABULA_' and use '_' to separate nested keys.

    Example: TABULA_OWID_MAX_DATASET_SIZE_MB=50

    Args:
        config (Dict[str, Any]): Base configuration

    Returns:
        Dict[str, Any]: Configuration with environment overrides applied
    """
    prefix = "TABULA_"
    result = config.copy()

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Convert environment variable name to config path
        config_path = env_key[len(prefix) :].lower().split("_")

        # Navigate to the correct nested position
        current = result
        for key in config_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value (with type conversion)
        final_key = config_path[-1]
        current[final_key] = convert_env_value(env_value)

        logger.debug(f"Applied environment override: {env_key} = {env_value}")

    return result


def convert_env_value(value: str) -> Union[str, int, float, bool]:
    """
    Convert environment variable string to appropriate type.

    Args:
        value (str): Environment variable value

    Returns:
        Union[str, int, float, bool]: Converted value
    """
    # Boolean conversion
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False

    # Numeric conversion
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass

    # Return as string
    return value


def validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Validate configuration schema for TABULA-R2 project.

    Args:
        config (Dict[str, Any]): Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["owid", "cleaning", "quality_control", "metadata", "storage"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate storage configuration
    storage = config.get("storage", {})
    if "base_directory" not in storage:
        raise ValueError("storage.base_directory is required")

    # Validate metadata configuration
    metadata = config.get("metadata", {})
    if "llm_model" not in metadata:
        raise ValueError("metadata.llm_model is required")

    # Validate numeric ranges
    cleaning = config.get("cleaning", {})
    if cleaning.get("max_missing_ratio", 0) > 1.0:
        raise ValueError("cleaning.max_missing_ratio must be <= 1.0")

    logger.debug("Configuration schema validation passed")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation path.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        key_path (str): Dot-separated path to configuration value
        default (Any): Default value if key not found

    Returns:
        Any: Configuration value or default

    Example:
        >>> get_config_value(config, "owid.max_dataset_size_mb", 50)
    """
    keys = key_path.split(".")
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.

    Args:
        config (Dict[str, Any]): Configuration to save
        output_path (Union[str, Path]): Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)

    logger.debug(f"Configuration saved to {output_path}")


# Default configuration for TABULA-R2 project
DEFAULT_CONFIG = {
    "owid": {"root_path": "owid-datasets", "max_dataset_size_mb": 50},
    "cleaning": {
        "max_rows_per_table": 1000,
        "min_numeric_columns": 1,
        "max_missing_ratio": 0.9,
        "remove_duplicates": True,
        "standardize_names": True,
        "missing_value_strategy": {"numeric": "keep_na", "categorical": "keep_na"},
    },
    "quality_control": {
        "min_data_completeness": 0.3,
        "required_columns_min": 3,
        "min_rows": 5,
        "max_entity_cardinality": 300,
    },
    "metadata": {
        "llm_model": "llama3",
        "max_retries": 2,
        "require_llm": False,
        "domain_categories": [
            "economics",
            "demographics",
            "health",
            "education",
            "environment",
            "technology",
            "general",
        ],
    },
    "storage": {"base_directory": "data/tables"},
    "pipeline": {"mode": "tables_with_index", "index_filename": "my_test_index.json"},
}
