"""
Observation Space Generator for QnA Evaluation

This module generates different observation spaces (obs) for presenting table data
to LLMs during QnA evaluation. It supports various modes like full table,
header + 5 rows, header + 1 row, etc.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..utils.config import get_config_value


@dataclass
class ObsConfig:
    """Configuration for observation space generation."""

    mode: str  # "full_table", "header_5", "header_1"
    max_rows_per_table: int = 1000
    sample_strategy: str = "head_tail_balanced"  # "head_tail_balanced", "random_seeded"


class ObservationSpaceGenerator:
    """Generates observation spaces for table data presentation."""

    def __init__(self, config: Dict[str, Any], data_root: Path = Path("data/tables")):
        """
        Initialize the observation space generator.

        Args:
            config: Configuration dictionary containing obs settings
            data_root: Root directory containing table data
        """
        self.config = config
        self.data_root = data_root

        # Extract obs configuration
        obs_config = get_config_value(config, "qna.obs", {})
        self.obs_config = ObsConfig(
            mode=get_config_value(obs_config, "mode", "header_5"),
            max_rows_per_table=get_config_value(obs_config, "max_rows_per_table", 1000),
            sample_strategy="head",  # Always use head for consistent behavior
        )
        self.include_description: bool = bool(
            get_config_value(obs_config, "include_description", True)
        )

    def generate_obs_for_question(
        self, question_record: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate observation space description for a question (single or multi-table).

        Args:
            question_record: Question record with table_refs, question, answer

        Returns:
            Dictionary with:
                - visible_columns_desc: (kept for backward compatibility; not used in prompt)
                - visible_rows_desc: Description of visible sample rows (may contain multiple tables)
                - table_stats_desc: Table statistics summary (single or concatenated per-table)
                - table_description: Optional concise description(s) from meta.json
        """
        table_refs = question_record.get("table_refs", [])
        if not table_refs:
            return {
                "visible_columns_desc": "No tables referenced",
                "visible_rows_desc": "No data available",
                "table_stats_desc": "No table statistics",
                "table_structures": {},
            }

        # Normalize to list
        refs: List[str] = table_refs if isinstance(table_refs, list) else [table_refs]
        if len(refs) == 1:
            table_id = refs[0]
            table_path = self.data_root / table_id / "table.csv"
            if not table_path.exists():
                return {
                    "visible_columns_desc": f"Table {table_id} not found",
                    "visible_rows_desc": "No data available",
                    "table_stats_desc": "Table not accessible",
                    "table_structures": {},
                }
            try:
                df = pd.read_csv(table_path)
                obs = self._generate_obs_for_dataframe(df, table_id)
                if self.include_description:
                    desc = self._load_table_description(table_id, df)
                    if desc:
                        obs["table_description"] = desc

                # Add table structure information
                obs["table_structures"] = {
                    table_id: self._generate_table_structure(df, table_id)
                }
                return obs
            except Exception as e:
                return {
                    "visible_columns_desc": f"Error loading {table_id}: {str(e)}",
                    "visible_rows_desc": "Data loading failed",
                    "table_stats_desc": "Unable to generate statistics",
                    "table_structures": {},
                }

        # Multi-table: aggregate per-table
        rows_blocks: List[str] = []
        stats_lines: List[str] = []
        desc_lines: List[str] = []
        table_structures = {}
        for table_id in refs:
            table_path = self.data_root / table_id / "table.csv"
            if not table_path.exists():
                stats_lines.append(f"Table {table_id}: not found")
                continue
            try:
                df = pd.read_csv(table_path)
                single = self._generate_obs_for_dataframe(df, table_id)
                rows_blocks.append(f"[Table {table_id}]\n{single['visible_rows_desc']}")
                stats_lines.append(single["table_stats_desc"])
                if self.include_description:
                    desc = self._load_table_description(table_id, df)
                    if desc:
                        desc_lines.append(desc)
                # Add table structure information
                table_structures[table_id] = self._generate_table_structure(
                    df, table_id
                )
            except Exception as e:
                stats_lines.append(f"Table {table_id}: error loading ({e})")
                continue

        combined: Dict[str, Any] = {
            "visible_columns_desc": "",
            "visible_rows_desc": (
                "\n\n".join(rows_blocks) if rows_blocks else "No data available"
            ),
            "table_stats_desc": (
                "\n".join(stats_lines) if stats_lines else "No table statistics"
            ),
            "table_structures": table_structures,
        }
        if desc_lines:
            combined["table_description"] = " | ".join(desc_lines)
        return combined

    def _generate_obs_for_dataframe(
        self, df: pd.DataFrame, table_id: str
    ) -> Dict[str, str]:
        """Generate observation space for a single dataframe."""

        # Generate column description (all columns are visible)
        visible_columns_desc = f"Available columns: {', '.join(df.columns.tolist())}"

        # Generate sample rows based on mode
        if self.obs_config.mode == "full_table":
            # Show up to max_rows_per_table
            max_rows = min(len(df), self.obs_config.max_rows_per_table)
            sample_df = df.head(max_rows)
            visible_rows_desc = f"Table data (showing {len(sample_df)} of {len(df)} rows):\n{sample_df.to_string(index=False, max_cols=None)}"
        elif self.obs_config.mode == "header_5":
            sample_df = df.head(5)
            visible_rows_desc = f"Sample data (first 5 rows of {len(df)}):\n{sample_df.to_string(index=False, max_cols=None)}"
        elif self.obs_config.mode == "header_1":
            sample_df = df.head(1)
            visible_rows_desc = f"Sample data (first row of {len(df)}):\n{sample_df.to_string(index=False, max_cols=None)}"
        else:
            visible_rows_desc = f"Unknown obs mode: {self.obs_config.mode}"

        # Generate table statistics
        table_stats_desc = self._generate_table_stats(df, table_id)

        return {
            "visible_columns_desc": visible_columns_desc,  # Keep for backward compatibility
            "visible_rows_desc": visible_rows_desc,
            "table_stats_desc": table_stats_desc,
        }

    def _sample_rows(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample rows based on the configured strategy."""
        if len(df) <= n:
            return df

        if self.obs_config.sample_strategy == "head_tail_balanced":
            # Take from head and tail
            if n == 1:
                return df.head(1)
            elif n == 2:
                return pd.concat([df.head(1), df.tail(1)])
            else:
                head_n = n // 2
                tail_n = n - head_n
                return pd.concat([df.head(head_n), df.tail(tail_n)])
        elif self.obs_config.sample_strategy == "random_seeded":
            # Random sample with fixed seed for reproducibility
            return df.sample(n=n, random_state=42)
        else:
            # Default to head
            return df.head(n)

    def _generate_table_stats(self, df: pd.DataFrame, table_id: str) -> str:
        """Generate concise table statistics summary (no table_id prefix)."""
        stats = [f"rows: {df.shape[0]}", f"cols: {df.shape[1]}"]

        # Add basic info about data types
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats.append(f"Numeric columns: {len(numeric_cols)}")

        text_cols = df.select_dtypes(include=["object"]).columns
        if len(text_cols) > 0:
            stats.append(f"Text columns: {len(text_cols)}")

        return " | ".join(stats)

    def _load_table_description(
        self, table_id: str, df: Optional[pd.DataFrame] = None
    ) -> str:
        """Load a concise table description from meta.json if available.

        Falls back to a compact autogenerated line if meta is missing.
        """
        meta_path = self.data_root / table_id / "meta.json"
        title = None
        description = None
        try:
            if meta_path.exists():
                import json

                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                title = meta.get("title")
                description = meta.get("description")
        except Exception:
            pass

        # Try to infer time span
        time_span = None
        if df is not None and "Year" in df.columns:
            try:
                year_min = int(pd.to_numeric(df["Year"], errors="coerce").min())
                year_max = int(pd.to_numeric(df["Year"], errors="coerce").max())
                time_span = f"{year_min}–{year_max}"
            except Exception:
                time_span = None

        pieces: List[str] = []
        if title:
            pieces.append(title)
        if time_span:
            pieces.append(time_span)
        concise = (
            " (".join([pieces[0], ", ".join(pieces[1:]) + ")"])
            if len(pieces) > 1
            else (pieces[0] if pieces else None)
        )

        if concise:
            return f"{table_id}: {concise}"

        # Fallback: minimal autogenerated description
        n_rows = len(df) if df is not None else "?"
        n_cols = len(df.columns) if df is not None else "?"
        return f"{table_id}: {n_rows} rows, {n_cols} columns"

    def _generate_table_structure(
        self, df: pd.DataFrame, table_id: str
    ) -> Dict[str, Any]:
        """Generate detailed table structure information."""
        # Get column names and data types
        columns = list(df.columns)
        dtypes = [str(df[col].dtype) for col in columns]

        # Get sample values for each column (first non-null value)
        sample_values = {}
        for col in columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample_values[col] = str(non_null_values.iloc[0])
            else:
                sample_values[col] = "No data"

        return {
            "columns": columns,
            "dtypes": dtypes,
            "sample_values": sample_values,
            "row_count": len(df),
            "null_counts": {col: int(df[col].isnull().sum()) for col in columns},
        }
