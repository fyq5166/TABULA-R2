"""
Utility Module

This module provides common utility functions for the TABULA-R2 project,
including configuration management, logging, and I/O operations.

Functions:
    load_config: Load YAML configuration files
    setup_logging: Configure project logging
    ensure_dir: Create directories if they don't exist
"""

# Import utility functions
from .config import (
    load_config,
    merge_configs,
    get_config_value,
    save_config,
    DEFAULT_CONFIG,
)
from .io_utils import (
    ensure_dir,
    safe_save_json,
    safe_load_json,
    create_table_directory,
    save_table_files,
    load_table_metadata,
    load_table_data,
    scan_existing_tables,
    get_table_summary,
    create_global_index,
    cleanup_backup_files,
)
from .logging_utils import (
    setup_logging,
    get_logger,
    log_function_call,
    log_processing_step,
    log_data_summary,
    create_session_log_file,
    setup_dev_logging,
    setup_prod_logging,
)

__all__ = [
    # Configuration utilities
    "load_config",
    "merge_configs",
    "get_config_value",
    "save_config",
    "DEFAULT_CONFIG",
    # I/O utilities
    "ensure_dir",
    "safe_save_json",
    "safe_load_json",
    "create_table_directory",
    "save_table_files",
    "load_table_metadata",
    "load_table_data",
    "scan_existing_tables",
    "get_table_summary",
    "create_global_index",
    "cleanup_backup_files",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_processing_step",
    "log_data_summary",
    "create_session_log_file",
    "setup_dev_logging",
    "setup_prod_logging",
]
