"""
OWID Dataset Processor

This module handles the processing of Our World in Data datasets for the TABULA-R2 project.
It includes functionality for scanning, validating, loading, cleaning, and extracting metadata from OWID datasets.

Classes:
    OWIDProcessor: Main processor class for handling OWID datasets
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import os

from ..utils.logging_utils import get_logger, log_processing_step, log_data_summary
from ..utils.config import get_config_value
from .table_cleaner import TableCleaner
from .llm_domain_classifier import LLMDomainClassifier
from .metadata_extractor import MetadataExtractor


class OWIDProcessor:
    """
    OWID Dataset Processor for TABULA-R2 project.

    This class handles the complete processing pipeline for OWID datasets,
    including scanning, validation, loading, cleaning, and metadata extraction.

    Processing Flow:
    1. scan_owid_datasets() - Find processable datasets
    2. validate_dataset_structure() - Check structure validity
    3. load_raw_data() - Load raw data and datapackage info
    4. clean_data() - Clean and prepare data for analysis
    5. extract_metadata() - Extract metadata from cleaned data
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OWID Processor.

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Extract key configuration values
        self.owid_root = Path(config["owid"]["root_path"])
        self.max_dataset_size_mb = config["owid"]["max_dataset_size_mb"]

        # Quality control settings
        self.max_rows_per_table = config["cleaning"]["max_rows_per_table"]
        self.max_missing_ratio = config["cleaning"]["max_missing_ratio"]
        self.min_rows = config["quality_control"]["min_rows"]
        self.required_columns_min = config["quality_control"]["required_columns_min"]

        # Initialize components
        self.table_cleaner = TableCleaner(config)
        self.metadata_extractor = MetadataExtractor(config)
        self.domain_classifier = LLMDomainClassifier(config=config)

        self.logger.debug(f"OWIDProcessor initialized with root path: {self.owid_root}")

    def process_single_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Process a single dataset through the complete pipeline.

        Args:
            dataset_name (str): Name of the dataset to process

        Returns:
            Optional[Dict[str, Any]]: Complete processing result or None if failed
        """
        dataset_path = self.owid_root / dataset_name

        log_processing_step("complete_processing", "started", dataset=dataset_name)

        try:
            # Step 1: Validate structure
            if not self.validate_dataset_structure(dataset_path):
                self.logger.warning(
                    f"Dataset {dataset_name} failed structure validation"
                )
                return None

            # Step 2: Load raw data
            raw_df, datapackage_info = self.load_raw_data(dataset_path)
            if raw_df is None:
                self.logger.warning(f"Dataset {dataset_name} failed to load")
                return None

            # Step 3: Clean data (NEW STEP)
            cleaned_df = self.table_cleaner.clean_dataset(raw_df, dataset_name)
            if cleaned_df is None:
                self.logger.warning(f"Dataset {dataset_name} failed cleaning")
                return None

            # Step 4: Classify domain using LLM
            try:
                description = (
                    datapackage_info.get("description")
                    if isinstance(datapackage_info, dict)
                    else None
                )
                domain = self.domain_classifier.classify_table(
                    cleaned_df, dataset_name, description
                )
            except Exception as exc:
                self.logger.error(
                    f"Dataset {dataset_name} aborted during domain classification: {exc}"
                )
                if get_config_value(self.config, "metadata.require_llm", False):
                    raise
                domain = None

            if not domain:
                self.logger.warning(
                    f"Dataset {dataset_name} failed domain classification"
                )
                # Default to general domain to match legacy processor behavior
                domain = "general"

            # Step 5: Extract metadata using the dedicated extractor
            metadata = self.metadata_extractor.extract_metadata(
                cleaned_df, datapackage_info, dataset_name
            )

            # Add domain to metadata
            metadata["domain"] = domain

            # Step 6: Generate complete result
            result = {
                "dataset_name": dataset_name,
                "metadata": metadata,
                "raw_data": raw_df,  # Store actual raw data
                "cleaned_data": cleaned_df,
                "cleaning_summary": self.table_cleaner.get_cleaning_summary(
                    raw_df, cleaned_df
                ),
            }

            log_processing_step(
                "complete_processing", "completed", dataset=dataset_name
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing dataset {dataset_name}: {e}")
            return None

    def scan_owid_datasets(self) -> List[str]:
        """
        Scan the OWID datasets directory and return a list of processable datasets.

        Returns:
            List[str]: List of dataset names that meet processing criteria
        """
        log_processing_step(
            "dataset_scanning", "started", root_path=str(self.owid_root)
        )

        if not self.owid_root.exists():
            self.logger.error(f"OWID root path does not exist: {self.owid_root}")
            return []

        valid_datasets = []
        total_datasets = 0

        for item in self.owid_root.iterdir():
            if not item.is_dir():
                continue

            total_datasets += 1
            dataset_name = item.name

            # Check if dataset meets basic requirements
            if self._is_dataset_valid_for_scanning(item):
                valid_datasets.append(dataset_name)
                self.logger.debug(f"Valid dataset found: {dataset_name}")
            else:
                self.logger.debug(f"Skipped dataset: {dataset_name}")

        # Sort datasets alphabetically to ensure consistent ordering
        valid_datasets.sort()

        log_processing_step(
            "dataset_scanning",
            "completed",
            total_found=total_datasets,
            valid_datasets=len(valid_datasets),
        )

        return valid_datasets

    def _is_dataset_valid_for_scanning(self, dataset_path: Path) -> bool:
        """
        Check if a dataset directory meets basic scanning criteria.

        Args:
            dataset_path (Path): Path to dataset directory

        Returns:
            bool: True if dataset meets basic criteria
        """
        # Check for required files
        datapackage_file = dataset_path / "datapackage.json"
        if not datapackage_file.exists():
            return False

        # Check for CSV files
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            return False

        # Check dataset size
        try:
            total_size = sum(
                f.stat().st_size for f in dataset_path.iterdir() if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            if size_mb > self.max_dataset_size_mb:
                self.logger.debug(
                    f"Dataset too large: {size_mb:.1f}MB > {self.max_dataset_size_mb}MB"
                )
                return False
        except Exception as e:
            self.logger.warning(f"Could not check dataset size: {e}")
            return False

        return True

    def validate_dataset_structure(self, dataset_path: Path) -> bool:
        """
        Validate that a dataset has the required structure for processing.

        Args:
            dataset_path (Path): Path to the dataset directory

        Returns:
            bool: True if dataset structure is valid
        """
        log_processing_step(
            "structure_validation", "started", dataset=dataset_path.name
        )

        try:
            # Check datapackage.json
            datapackage_file = dataset_path / "datapackage.json"
            if not datapackage_file.exists():
                self.logger.warning(f"Missing datapackage.json in {dataset_path.name}")
                return False

            # Try to load and parse datapackage.json
            with open(datapackage_file, "r", encoding="utf-8") as f:
                datapackage = json.load(f)

            if not isinstance(datapackage, dict):
                self.logger.warning(
                    f"Invalid datapackage.json format in {dataset_path.name}"
                )
                return False

            # Check for CSV files
            csv_files = list(dataset_path.glob("*.csv"))
            if not csv_files:
                self.logger.warning(f"No CSV files found in {dataset_path.name}")
                return False

            # Try to read the main CSV file
            main_csv = csv_files[0]  # Use the first CSV file
            try:
                # Quick read to check structure
                df_sample = pd.read_csv(main_csv, nrows=5)

                # Check minimum columns
                if len(df_sample.columns) < self.required_columns_min:
                    self.logger.warning(
                        f"Insufficient columns in {dataset_path.name}: {len(df_sample.columns)}"
                    )
                    return False

                log_processing_step(
                    "structure_validation",
                    "completed",
                    dataset=dataset_path.name,
                    csv_file=main_csv.name,
                    columns=len(df_sample.columns),
                )
                return True

            except Exception as e:
                self.logger.warning(f"Cannot read CSV file {main_csv.name}: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error validating dataset {dataset_path.name}: {e}")
            return False

    def load_raw_data(
        self, dataset_path: Path
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Load raw data and metadata from an OWID dataset.

        Args:
            dataset_path (Path): Path to the dataset directory

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[Dict]]: DataFrame and datapackage info, or (None, None) if failed
        """
        log_processing_step("data_loading", "started", dataset=dataset_path.name)

        try:
            # Load datapackage.json
            datapackage_file = dataset_path / "datapackage.json"
            with open(datapackage_file, "r", encoding="utf-8") as f:
                datapackage_info = json.load(f)

            # Find and load the main CSV file
            csv_files = list(dataset_path.glob("*.csv"))
            if not csv_files:
                self.logger.error(f"No CSV files found in {dataset_path.name}")
                return None, None

            # Use the first CSV file (OWID datasets typically have one main CSV)
            main_csv = csv_files[0]

            # Load CSV with error handling
            try:
                df = pd.read_csv(main_csv, encoding="utf-8")
            except UnicodeDecodeError:
                # Try alternative encodings
                for encoding in ["latin-1", "iso-8859-1", "cp1252"]:
                    try:
                        df = pd.read_csv(main_csv, encoding=encoding)
                        self.logger.debug(
                            f"Loaded {main_csv.name} with {encoding} encoding"
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError(
                        "Could not decode CSV with any common encoding"
                    )

            # Basic data cleaning
            df = df.dropna(how="all")  # Remove completely empty rows
            df = df.loc[
                :, ~df.columns.str.contains("^Unnamed")
            ]  # Remove unnamed columns

            log_data_summary(f"{dataset_path.name} (raw)", len(df), len(df.columns))
            log_processing_step(
                "data_loading",
                "completed",
                dataset=dataset_path.name,
                rows=len(df),
                columns=len(df.columns),
            )

            return df, datapackage_info

        except Exception as e:
            self.logger.error(f"Error loading raw data for {dataset_path.name}: {e}")
            return None, None
