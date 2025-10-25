"""
I/O Utilities

This module provides input/output utility functions for the TABULA-R2 project,
with focus on table-based storage operations and file management.

Functions:
    ensure_dir: Create directories if they don't exist
    safe_save_json: Safely save JSON data with backup
    safe_load_json: Safely load JSON data with error handling
    create_table_directory: Create standardized table directory structure
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(directory: Path, create_parents: bool = True) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (Path): Directory path to create
        create_parents (bool): Whether to create parent directories

    Returns:
        Path: The created directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=create_parents, exist_ok=True)
    return directory


def safe_save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """
    Safely save JSON data to file.

    Args:
        data (Dict[str, Any]): Data to save
        file_path (Path): Target file path
        indent (int): JSON indentation level

    Raises:
        IOError: If file operations fail
    """
    file_path = Path(file_path)

    # Create parent directory if it doesn't exist
    ensure_dir(file_path.parent)

    # Write JSON data
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise IOError(f"Failed to save JSON: {e}")


def safe_load_json(file_path: Path, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Safely load JSON data from file with error handling.

    Args:
        file_path (Path): JSON file path to load
        default (Optional[Dict]): Default value if loading fails

    Returns:
        Dict[str, Any]: Loaded JSON data or default value
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"JSON file not found: {file_path}")
        return default or {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.debug(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return default or {}


def create_table_directory(base_dir: Path, table_id: str) -> Path:
    """
    Create standardized table directory structure.

    Args:
        base_dir (Path): Base directory for tables
        table_id (str): Table identifier (e.g., "table_0001")

    Returns:
        Path: Created table directory path

    Example:
        Creates: data/tables/table_0001/
    """
    table_dir = base_dir / table_id
    ensure_dir(table_dir)

    logger.debug(f"Created table directory: {table_dir}")
    return table_dir


def save_table_files(
    table_dir: Path,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    metadata: Dict[str, Any],
    quality_report: Dict[str, Any],
) -> None:
    """
    Save all files for a table in the standardized structure.

    Args:
        table_dir (Path): Table directory path
        raw_df (pd.DataFrame): Raw data DataFrame
        processed_df (pd.DataFrame): Processed data DataFrame
        metadata (Dict[str, Any]): Table metadata
        quality_report (Dict[str, Any]): Quality assessment report
    """
    # Save CSV files
    raw_df.to_csv(table_dir / "raw.csv", index=False)
    processed_df.to_csv(table_dir / "table.csv", index=False)

    # Save JSON files
    safe_save_json(metadata, table_dir / "meta.json")
    safe_save_json(quality_report, table_dir / "quality_report.json")

    logger.debug(f"Saved all files for table in {table_dir}")


def load_table_metadata(table_dir: Path) -> Dict[str, Any]:
    """
    Load metadata from a table directory.

    Args:
        table_dir (Path): Table directory path

    Returns:
        Dict[str, Any]: Table metadata
    """
    meta_path = table_dir / "meta.json"
    return safe_load_json(meta_path, default={})


def load_table_data(
    table_dir: Path, data_type: str = "processed"
) -> Optional[pd.DataFrame]:
    """
    Load table data from a table directory.

    Args:
        table_dir (Path): Table directory path
        data_type (str): Type of data to load ("raw" or "processed")

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if file doesn't exist
    """
    if data_type == "raw":
        csv_path = table_dir / "raw.csv"
    elif data_type == "processed":
        csv_path = table_dir / "table.csv"
    else:
        raise ValueError(
            f"Invalid data_type: {data_type}. Must be 'raw' or 'processed'"
        )

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        logger.debug(f"Successfully loaded {data_type} data from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {csv_path}: {e}")
        return None


def scan_existing_tables(base_dir: Path) -> List[str]:
    """
    Scan for existing table directories.

    Args:
        base_dir (Path): Base directory containing table directories

    Returns:
        List[str]: List of existing table IDs
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return []

    table_ids = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("table_"):
            table_ids.append(item.name)

    table_ids.sort()  # Sort to maintain order
    return table_ids


def get_table_summary(table_dir: Path) -> Dict[str, Any]:
    """
    Get summary information about a table directory.

    Args:
        table_dir (Path): Table directory path

    Returns:
        Dict[str, Any]: Summary information
    """
    table_dir = Path(table_dir)

    summary = {"table_id": table_dir.name, "exists": table_dir.exists(), "files": {}}

    if table_dir.exists():
        # Check for expected files
        expected_files = ["raw.csv", "table.csv", "meta.json", "quality_report.json"]
        for filename in expected_files:
            file_path = table_dir / filename
            summary["files"][filename] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
            }

        # Get metadata summary
        metadata = load_table_metadata(table_dir)
        summary["title"] = metadata.get("title", "Unknown")
        summary["domain"] = metadata.get("llm_classification", {}).get(
            "category", "Unknown"
        )

        # Get data summary
        processed_df = load_table_data(table_dir, "processed")
        if processed_df is not None:
            summary["row_count"] = len(processed_df)
            summary["column_count"] = len(processed_df.columns)

    return summary


def create_global_index(
    base_dir: Path, output_file: str = "tables_index.json"
) -> Dict[str, Any]:
    """
    Create global index of all tables.

    Args:
        base_dir (Path): Base directory containing table directories
        output_file (str): Output filename for index

    Returns:
        Dict[str, Any]: Global index data
    """
    base_dir = Path(base_dir)
    table_ids = scan_existing_tables(base_dir)

    index_data = {
        "total_tables": len(table_ids),
        "created_at": pd.Timestamp.now().isoformat(),
        "tables": {},
    }

    for table_id in table_ids:
        table_dir = base_dir / table_id
        summary = get_table_summary(table_dir)
        index_data["tables"][table_id] = summary

    # Save index file
    index_path = base_dir / output_file
    safe_save_json(index_data, index_path)

    logger.debug(f"Created global index with {len(table_ids)} tables: {index_path}")
    return index_data


def cleanup_backup_files(directory: Path, max_backups: int = 5) -> None:
    """
    Clean up old backup files, keeping only the most recent ones.

    Args:
        directory (Path): Directory to clean up
        max_backups (int): Maximum number of backup files to keep
    """
    directory = Path(directory)

    if not directory.exists():
        return

    # Find all backup files
    backup_files = list(directory.glob("*.backup"))

    if len(backup_files) <= max_backups:
        return

    # Sort by modification time (oldest first)
    backup_files.sort(key=lambda x: x.stat().st_mtime)

    # Remove oldest backup files
    files_to_remove = backup_files[:-max_backups]
    for backup_file in files_to_remove:
        backup_file.unlink()
        logger.debug(f"Removed old backup: {backup_file}")

        logger.debug(
            f"Cleaned up {len(files_to_remove)} old backup files in {directory}"
        )


def save_table_data(
    table_dir: Path,
    table_id: str,
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    metadata: Dict[str, Any],
    cleaning_summary: Dict[str, Any],
    original_name: str,
) -> bool:
    """
    Save complete table data with clean metadata structure.

    Args:
        table_dir (Path): Table directory path
        table_id (str): Table identifier
        raw_data (pd.DataFrame): Raw data DataFrame (before cleaning)
        processed_data (pd.DataFrame): Processed data DataFrame (after cleaning)
        metadata (Dict[str, Any]): Table metadata
        cleaning_summary (Dict[str, Any]): Cleaning process summary
        original_name (str): Original dataset name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Save CSV files
        raw_data.to_csv(table_dir / "raw.csv", index=False)
        processed_data.to_csv(table_dir / "table.csv", index=False)

        # Create clean metadata with table_id first
        clean_metadata = {"table_id": table_id}

        # Add original metadata (excluding any existing table_id)
        for key, value in metadata.items():
            if key != "table_id":
                clean_metadata[key] = value

        # Save metadata and cleaning summary
        safe_save_json(clean_metadata, table_dir / "meta.json")
        safe_save_json(cleaning_summary, table_dir / "quality_report.json")

        logger.debug(f"Successfully saved all data for {table_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to save table data for {table_id}: {e}")
        return False


def update_global_index(base_dir: Path, new_tables: List[Dict[str, Any]]) -> bool:
    """
    Update the global index with new table information.

    Args:
        base_dir (Path): Base directory containing tables
        new_tables (List[Dict[str, Any]]): List of new table information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        index_path = base_dir / "tables_index.json"

        # Load existing index or create new one
        if index_path.exists():
            index_data = safe_load_json(index_path, default={})
        else:
            index_data = {
                "total_tables": 0,
                "created_at": pd.Timestamp.now().isoformat(),
                "tables": {},
            }

        # Add new tables to index
        for table_info in new_tables:
            table_id = table_info["table_id"]

            # Create table entry
            table_entry = {
                "table_id": table_id,
                "original_name": table_info["original_name"],
                "title": table_info["metadata"].get("title", "Unknown"),
                "row_count": table_info["metadata"].get("row_count", 0),
                "column_count": table_info["metadata"].get("column_count", 0),
                "complexity_level": table_info["metadata"].get(
                    "complexity_level", "unknown"
                ),
                "cleaning_success": table_info["cleaning_summary"].get(
                    "success", False
                ),
                "added_at": pd.Timestamp.now().isoformat(),
            }

            index_data["tables"][table_id] = table_entry

        # Update summary statistics
        index_data["total_tables"] = len(index_data["tables"])
        index_data["last_updated"] = pd.Timestamp.now().isoformat()

        # Calculate basic statistics
        total_tables = len(index_data["tables"])
        avg_rows = (
            sum(meta.get("row_count", 0) for meta in index_data["tables"].values())
            / total_tables
            if total_tables > 0
            else 0
        )
        avg_columns = (
            sum(meta.get("column_count", 0) for meta in index_data["tables"].values())
            / total_tables
            if total_tables > 0
            else 0
        )

        index_data["statistics"] = {
            "total_tables": total_tables,
            "avg_rows": avg_rows,
            "avg_columns": avg_columns,
        }

        # Save updated index
        safe_save_json(index_data, index_path)

        logger.debug(f"Updated global index with {len(new_tables)} new tables")
        return True

    except Exception as e:
        logger.error(f"Failed to update global index: {e}")
        return False


def get_processed_dataset_names(base_dir: Path) -> List[str]:
    """
    Get list of dataset names that have already been processed.

    Args:
        base_dir (Path): Base directory containing table directories

    Returns:
        List[str]: List of processed dataset names (titles)
    """
    processed_names = []
    table_ids = scan_existing_tables(base_dir)

    for table_id in table_ids:
        table_dir = base_dir / table_id
        meta_path = table_dir / "meta.json"

        if meta_path.exists():
            metadata = safe_load_json(meta_path, default={})
            title = metadata.get("title", "")
            if title:
                processed_names.append(title)

    # Sort alphabetically to match processing order
    processed_names.sort()
    return processed_names


# Question Generation Support Functions


def load_table_data_by_id(
    table_id: str, base_dir: str = "data/tables", data_type: str = "processed"
) -> Optional[pd.DataFrame]:
    """
    Load table data by table ID for question generation.

    Args:
        table_id (str): Table identifier (e.g., "table_0001")
        base_dir (str): Base directory containing tables
        data_type (str): Type of data to load ("raw" or "processed")

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if file doesn't exist
    """
    table_dir = Path(base_dir) / table_id
    return load_table_data(table_dir, data_type)


def load_table_metadata_by_id(
    table_id: str, base_dir: str = "data/tables"
) -> Dict[str, Any]:
    """
    Load table metadata by table ID for question generation.

    Args:
        table_id (str): Table identifier (e.g., "table_0001")
        base_dir (str): Base directory containing tables

    Returns:
        Dict[str, Any]: Table metadata
    """
    table_dir = Path(base_dir) / table_id
    return load_table_metadata(table_dir)
