"""
Table Cleaner

This module provides functionality for cleaning and standardizing OWID datasets.
Handles data type conversion, missing values, and quality control.

Classes:
    TableCleaner: Main class for dataset cleaning and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..utils.logging_utils import get_logger, log_processing_step, log_data_summary


class TableCleaner:
    """
    Table Cleaner for preparing OWID datasets for reasoning tasks.

    This class performs comprehensive data cleaning including:
    - Missing value handling
    - Data type standardization
    - Outlier detection and handling
    - Column name standardization
    - Row/column filtering based on quality criteria
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Table Cleaner.

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Extract cleaning configuration
        self.max_rows_per_table = config["cleaning"]["max_rows_per_table"]
        # Remove max_columns constraint per new requirements
        self.max_columns = None
        self.min_numeric_columns = config["cleaning"]["min_numeric_columns"]
        self.max_missing_ratio = config["cleaning"]["max_missing_ratio"]
        self.remove_duplicates = config["cleaning"]["remove_duplicates"]
        self.standardize_names = config["cleaning"]["standardize_names"]
        self.missing_strategy = config["cleaning"]["missing_value_strategy"]

        # Quality control settings
        self.min_data_completeness = config["quality_control"]["min_data_completeness"]
        self.required_columns_min = config["quality_control"]["required_columns_min"]
        self.min_rows = config["quality_control"]["min_rows"]
        self.max_entity_cardinality = config["quality_control"][
            "max_entity_cardinality"
        ]

        self.logger.debug(
            "TableCleaner initialized with quality-focused cleaning strategy"
        )

    def clean_dataset(
        self, df: pd.DataFrame, dataset_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Perform comprehensive data cleaning on a dataset.

        Args:
            df (pd.DataFrame): Raw dataset to clean
            dataset_name (str): Name of the dataset for logging

        Returns:
            Optional[pd.DataFrame]: Cleaned dataset or None if cleaning failed
        """
        log_processing_step(
            "data_cleaning",
            "started",
            dataset=dataset_name,
            original_rows=len(df),
            original_columns=len(df.columns),
        )

        try:
            # Step 1: Initial validation and basic cleaning
            df_clean = self._initial_cleaning(df.copy())
            if df_clean is None:
                return None

            # Step 2: Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            if df_clean is None:
                return None

            # Step 3: Standardize data types and column names
            df_clean = self._standardize_data_types(df_clean)
            df_clean = self._standardize_column_names(df_clean)

            # Step 4: Remove duplicates if configured
            if self.remove_duplicates:
                df_clean = self._remove_duplicates(df_clean)

            # Step 5: Filter by size constraints
            df_clean = self._apply_size_constraints(df_clean)
            if df_clean is None:
                return None

            # Step 6: Final quality validation
            if not self._validate_final_quality(df_clean):
                self.logger.warning(
                    f"Dataset {dataset_name} failed final quality validation"
                )
                return None

            # Log cleaning results
            log_processing_step(
                "data_cleaning",
                "completed",
                dataset=dataset_name,
                final_rows=len(df_clean),
                final_columns=len(df_clean.columns),
                rows_removed=len(df) - len(df_clean),
                columns_removed=len(df.columns) - len(df_clean.columns),
            )

            log_data_summary(
                f"{dataset_name} (cleaned)", len(df_clean), len(df_clean.columns)
            )

            return df_clean

        except Exception as e:
            self.logger.error(f"Error cleaning dataset {dataset_name}: {e}")
            return None

    def _initial_cleaning(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Perform initial cleaning steps."""
        original_shape = df.shape

        # Remove completely empty rows and columns
        df = df.dropna(how="all")  # Remove rows with all NaN
        df = df.dropna(axis=1, how="all")  # Remove columns with all NaN

        # Remove unnamed columns (often created by CSV parsing errors)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Check if dataset still has minimum required structure
        if len(df) < self.min_rows or len(df.columns) < self.required_columns_min:
            self.logger.warning(f"Dataset too small after initial cleaning: {df.shape}")
            return None

        self.logger.debug(f"Initial cleaning: {original_shape} -> {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Handle missing values based on configuration strategy."""

        # Check overall missing ratio before processing
        total_missing_ratio = df.isnull().sum().sum() / df.size
        if total_missing_ratio > self.max_missing_ratio:
            self.logger.warning(
                f"Dataset has too many missing values: {total_missing_ratio:.2%}"
            )
            return None

        # Apply missing value strategy
        if (
            self.missing_strategy["numeric"] == "keep_na"
            and self.missing_strategy["categorical"] == "keep_na"
        ):
            # Keep all missing values as NA - only drop rows missing critical identifier columns
            self.logger.debug("Keeping missing values as NA")

            # Only drop rows that have missing values in critical identifier columns
            identifier_cols = [
                col for col in ["Entity", "Country", "Region"] if col in df.columns
            ]

            if identifier_cols:
                original_rows = len(df)
                # Only drop if ALL identifier columns are missing
                df = df.dropna(subset=identifier_cols, how="all")
                rows_dropped = original_rows - len(df)
                if rows_dropped > 0:
                    self.logger.debug(
                        f"Dropped {rows_dropped} rows with all identifier columns missing"
                    )

        else:
            # Legacy handling for other strategies (mean, median, mode, drop)
            for col in df.columns:
                if col in ["Entity", "Country", "Region", "Year", "Date", "Time"]:
                    continue

                is_numeric = df[col].dtype in ["int64", "float64", "int32", "float32"]

                if is_numeric and self.missing_strategy["numeric"] == "mean":
                    mean_val = df[col].mean()
                    if not pd.isna(mean_val):
                        df[col] = df[col].fillna(mean_val)

                elif is_numeric and self.missing_strategy["numeric"] == "median":
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        df[col] = df[col].fillna(median_val)

                elif not is_numeric and self.missing_strategy["categorical"] == "mode":
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])

            # Drop rows with missing identifier values (for non-keep_na strategies)
            identifier_cols = [
                col
                for col in ["Entity", "Country", "Region", "Year", "Date", "Time"]
                if col in df.columns
            ]
            if identifier_cols:
                original_rows = len(df)
                df = df.dropna(subset=identifier_cols)
                rows_dropped = original_rows - len(df)
                if rows_dropped > 0:
                    self.logger.debug(
                        f"Dropped {rows_dropped} rows with missing identifier values"
                    )

        # Verify we still have sufficient data (but be more lenient with keep_na strategy)
        min_rows_threshold = (
            max(1, self.min_rows // 2)
            if self.missing_strategy["numeric"] == "keep_na"
            else self.min_rows
        )
        if len(df) < min_rows_threshold:
            self.logger.warning(
                f"Too few rows remaining after missing value handling: {len(df)}"
            )
            return None

        return df

    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types for consistency."""

        for col in df.columns:
            # Skip Entity and other identifier columns
            if col in ["Entity", "Country", "Region"]:
                continue

            # Try to convert potential numeric columns
            if df[col].dtype == "object":
                # Try to convert to numeric
                try:
                    # Remove common non-numeric characters
                    cleaned_values = df[col].astype(str).str.replace(",", "")
                    cleaned_values = cleaned_values.str.replace("$", "")
                    cleaned_values = cleaned_values.str.replace("%", "")

                    # Convert to numeric
                    numeric_series = pd.to_numeric(cleaned_values, errors="coerce")

                    # If most values converted successfully, use numeric version
                    if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                        df[col] = numeric_series
                        self.logger.debug(f"Converted column '{col}' to numeric")

                except Exception as e:
                    self.logger.debug(
                        f"Could not convert column '{col}' to numeric: {e}"
                    )
                    continue

        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names if configured."""
        if not self.standardize_names:
            return df

        # Create mapping for standardized names
        new_columns = {}
        for col in df.columns:
            # Remove extra whitespace and standardize separators
            new_name = col.strip()
            new_name = " ".join(new_name.split())  # Normalize whitespace
            new_columns[col] = new_name

        df = df.rename(columns=new_columns)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_rows = len(df)
        df = df.drop_duplicates()
        rows_removed = original_rows - len(df)

        if rows_removed > 0:
            self.logger.debug(f"Removed {rows_removed} duplicate rows")

        return df

    def _apply_size_constraints(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply size constraints to the dataset."""

        # Limit number of rows if too large
        if len(df) > self.max_rows_per_table:
            self.logger.debug(
                f"Dataset too large ({len(df)} rows), sampling {self.max_rows_per_table} rows"
            )
            df = df.head(self.max_rows_per_table)

        # Do not limit number of columns anymore

        # Check Entity cardinality if Entity column exists
        if "Entity" in df.columns:
            entity_count = df["Entity"].nunique()
            if entity_count > self.max_entity_cardinality:
                self.logger.warning(
                    f"Too many entities ({entity_count}), may affect reasoning quality"
                )
                # Keep top entities by data count
                top_entities = (
                    df["Entity"].value_counts().head(self.max_entity_cardinality).index
                )
                df = df[df["Entity"].isin(top_entities)]
                self.logger.debug(f"Filtered to top {len(top_entities)} entities")

        return df

    def _validate_final_quality(self, df: pd.DataFrame) -> bool:
        """Validate final data quality meets requirements."""

        # Check minimum size requirements
        if len(df) < self.min_rows:
            self.logger.debug(f"Final dataset too small: {len(df)} rows")
            return False

        if len(df.columns) < self.required_columns_min:
            self.logger.debug(f"Final dataset has too few columns: {len(df.columns)}")
            return False

        # For keep_na strategy, be more lenient with completeness requirements
        if self.missing_strategy["numeric"] == "keep_na":
            # Only require that we have some non-null data and critical columns
            if "Entity" in df.columns:
                entity_completeness = df["Entity"].notna().sum() / len(df)
                if (
                    entity_completeness < 0.5
                ):  # At least 50% of entities should be present
                    self.logger.debug(
                        f"Too few valid entities: {entity_completeness:.2%}"
                    )
                    return False

            # Check that we have at least some non-null data in numeric columns
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in ["int64", "float64", "int32", "float32"]
            ]
            if numeric_cols:
                total_numeric_values = sum(
                    df[col].notna().sum() for col in numeric_cols
                )
                if total_numeric_values == 0:
                    self.logger.debug("No valid numeric data found")
                    return False
        else:
            # Original completeness check for non-keep_na strategies
            completeness = 1 - (df.isnull().sum().sum() / df.size)
            if completeness < self.min_data_completeness:
                self.logger.debug(
                    f"Final dataset completeness too low: {completeness:.2%}"
                )
                return False

        # Check for minimum numeric columns
        numeric_columns = self._count_numeric_columns(df)
        if numeric_columns < self.min_numeric_columns:
            self.logger.debug(
                f"Final dataset has too few numeric columns: {numeric_columns}"
            )
            return False

        return True

    def _count_numeric_columns(self, df: pd.DataFrame) -> int:
        """Count numeric columns in the dataset."""
        numeric_count = 0
        for col in df.columns:
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                numeric_count += 1
        return numeric_count

    def get_cleaning_summary(
        self, original_df: pd.DataFrame, cleaned_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Generate a summary of the cleaning process.

        Args:
            original_df (pd.DataFrame): Original dataset
            cleaned_df (Optional[pd.DataFrame]): Cleaned dataset

        Returns:
            Dict[str, Any]: Cleaning summary statistics
        """
        if cleaned_df is None:
            return {
                "success": False,
                "original_shape": original_df.shape,
                "final_shape": (0, 0),
                "rows_removed": len(original_df),
                "columns_removed": len(original_df.columns),
                "data_completeness": 0.0,
            }

        return {
            "success": True,
            "original_shape": original_df.shape,
            "final_shape": cleaned_df.shape,
            "rows_removed": len(original_df) - len(cleaned_df),
            "columns_removed": len(original_df.columns) - len(cleaned_df.columns),
            "data_completeness": 1
            - (cleaned_df.isnull().sum().sum() / cleaned_df.size),
            "numeric_columns": self._count_numeric_columns(cleaned_df),
        }
