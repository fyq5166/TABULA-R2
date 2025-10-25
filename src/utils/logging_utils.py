"""
Logging Utilities

Simple and effective logging utilities for the TABULA-R2 project.
Supports both file and console output with appropriate formatting.

Functions:
    setup_logging: Configure project logging
    get_logger: Get logger instance for specific module
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    verbose: bool = False,
    plain_file: bool = False,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> None:
    """
    Configure project logging settings.

    Args:
        log_level (str): Default logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file (Optional[Union[str, Path]]): Log file path, None means no file output
        console_output (bool): Whether to output to console
        verbose (bool): Whether to enable verbose mode
        console_level (Optional[str]): Console-specific log level, defaults to log_level
        file_level (Optional[str]): File-specific log level, defaults to log_level
    """
    # Convert log levels
    console_numeric_level = getattr(
        logging, (console_level or log_level).upper(), logging.INFO
    )
    file_numeric_level = getattr(
        logging, (file_level or log_level).upper(), logging.INFO
    )

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root log level to the most permissive level
    root_logger.setLevel(min(console_numeric_level, file_numeric_level))

    # Create formatter
    if verbose:
        # Verbose format: includes filename and line number
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # Simple format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        # Create log directory
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_numeric_level)

        # File logs format
        if plain_file:
            file_formatter = logging.Formatter("%(message)s")
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log configuration info
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Logging system configured - Level: {log_level}, File: {log_file or 'None'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for specific module.

    Args:
        name (str): Module name, typically use __name__

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call information (for debugging).

    Args:
        func_name (str): Function name
        **kwargs: Function arguments
    """
    logger = logging.getLogger("function_call")
    args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling function: {func_name}({args_str})")


def log_processing_step(step_name: str, status: str = "started", **details) -> None:
    """
    Log data processing steps.

    Args:
        step_name (str): Step name
        status (str): Status ("started", "completed", "failed")
        **details: Additional details
    """
    logger = logging.getLogger("processing")

    if details:
        details_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
        logger.debug(f"[{step_name}] {status} | {details_str}")
    else:
        logger.debug(f"[{step_name}] {status}")


def log_data_summary(
    df_name: str, row_count: int, col_count: int, missing_ratio: float = None
) -> None:
    """
    Log dataset summary information.

    Args:
        df_name (str): Dataset name
        row_count (int): Number of rows
        col_count (int): Number of columns
        missing_ratio (float): Missing value ratio
    """
    logger = logging.getLogger("data_summary")

    if missing_ratio is not None:
        logger.debug(
            f"{df_name}: {row_count} rows x {col_count} cols, missing: {missing_ratio:.1%}"
        )
    else:
        logger.debug(f"{df_name}: {row_count} rows x {col_count} cols")


def create_session_log_file(
    base_dir: str = "logs", *, plain_name: bool = False
) -> Path:
    """
    Create timestamped session log file.

    Args:
        base_dir (str): Log directory

    Returns:
        Path: Generated log file path
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if plain_name:
        log_file = log_dir / f"{timestamp}.log"
    else:
        log_file = log_dir / f"tabula_session_{timestamp}.log"
    return log_file


# Convenience function: Quick development environment logging setup
def setup_dev_logging(verbose: bool = False) -> None:
    """
    Quick setup for development environment logging (console + session file).

    Args:
        verbose (bool): Whether to enable verbose mode
    """
    session_log = create_session_log_file()
    setup_logging(
        log_level="DEBUG" if verbose else "INFO",
        log_file=session_log,
        console_output=True,
        verbose=verbose,
    )


# Convenience function: Quick production environment logging setup
def setup_prod_logging(log_file: Optional[str] = None) -> None:
    """
    Quick setup for production environment logging (file priority, minimal console output).

    Args:
        log_file (Optional[str]): Log file path
    """
    if log_file is None:
        log_file = "logs/tabula_production.log"

    setup_logging(
        log_level="INFO", log_file=log_file, console_output=True, verbose=False
    )


def setup_session_logging(
    session_name: str = "tabula", verbose: bool = False, *, plain_name: bool = False
) -> str:
    """
    Setup logging for a processing session with unique session ID.

    Args:
        session_name (str): Base name for the session
        verbose (bool): Whether to enable verbose logging

    Returns:
        str: Unique session ID
    """
    # Generate unique session ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{session_name}_{timestamp}"

    # Create session log file
    base = create_session_log_file("logs", plain_name=plain_name)
    if plain_name:
        log_file = base  # already YYYYMMDD_HHMMSS.log
    else:
        log_file = base.parent / f"session_{session_id}.log"

    # Setup logging
    setup_logging(
        log_level="DEBUG" if verbose else "INFO",
        log_file=log_file,
        console_output=True,
        verbose=verbose,
    )

    # Log session start
    logger = get_logger(__name__)
    logger.debug(f"Started session: {session_id}")
    logger.debug(f"Session log file: {log_file}")

    return session_id
