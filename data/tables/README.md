# Tables Directory

This directory contains the full table corpus used by TABULA-R².

Notes:
- Large data files (e.g., `.csv`, `.json`) are intentionally ignored by Git to keep the repository lightweight.
- Each table directory typically includes:
  - `table.csv`
  - `meta.json`
  - `quality_report.json`

If you need the full dataset, please refer to the project documentation or dataset release instructions.
# data/tables Directory

## Overview
This directory stores the canonical, reproducible table bundles produced by the data processing pipeline (`scripts/data_processing.py`). Each OWID-derived dataset is materialized as a self-contained unit under `table_XXXX/`. A catalog index `table_index.json` summarizes the collection and supports downstream discovery and analysis.

## Table Bundle Layout (`table_XXXX/`)
| File | Description |
|------|-------------|
| `table.csv` | Cleaned dataset (column headers normalized, missing values preserved unless filtered by policy). |
| `raw.csv` | Optional snapshot of the original CSV prior to cleaning (emitted when the pipeline is configured to save raw inputs). |
| `meta.json` | Metadata produced by the Metadata Extractor: human-readable descriptors and structured attributes for downstream consumers. |
| `quality_report.json` | Cleaning diagnostics emitted by the Table Cleaner: shapes, removals, and completeness metrics. |

Typical bundle example (`table_0001`):
```text
table_0001/
├─ table.csv
├─ raw.csv
├─ meta.json
└─ quality_report.json
```

## Metadata Schema (meta.json)
Key fields (non-exhaustive):
- `table_id` (string): Bundle identifier, e.g., `table_0001`.
- `title` (string): Human-readable dataset title.
- `description` (string): Dataset description (may be derived from the OWID datapackage).
- `columns` (string[]): Ordered list of column names present in `table.csv`.
- `row_count` (int): Number of rows in the cleaned table.
- `column_count` (int): Number of columns in the cleaned table.
- `numeric_columns` (string[]): Subset of `columns` identified as numeric.
- `has_time_series` (bool): Whether the table contains a time axis (e.g., `Year`).
- `domain` (string): Canonical domain classification (e.g., `health`, `environment`).
- `complexity_level` (string): `low` | `medium` | `high`, based on structural heuristics.
- `suitable_for_reasoning` (bool): Whether the table is appropriate for reasoning tasks.

Example excerpt:
```json
{
  "table_id": "table_0001",
  "title": "20th century deaths in US - CDC",
  "row_count": 94,
  "column_count": 50,
  "numeric_columns": ["Year", "Heart disease - Deaths", "Cancers - Deaths"],
  "has_time_series": true,
  "domain": "health",
  "complexity_level": "high",
  "suitable_for_reasoning": true
}
```

## Cleaning Report Schema (quality_report.json)
Key fields:
- `original_shape` ([int, int]): Rows × columns before cleaning.
- `final_shape` ([int, int]): Rows × columns after cleaning.
- `rows_removed` (int): Number of rows removed by cleaning policies.
- `columns_removed` (int): Number of columns removed or pruned.
- `data_completeness` (float): Completeness ratio in `[0, 1]` for the final table.
- `numeric_columns` (int): Count of columns considered numeric after cleaning.

Example excerpt:
```json
{
  "original_shape": [94, 84],
  "final_shape": [94, 50],
  "rows_removed": 0,
  "columns_removed": 34,
  "data_completeness": 0.58,
  "numeric_columns": 49
}
```

## Catalog Index (`table_index.json`)
The catalog index is rebuilt by the pipeline when configured to generate indices. It summarizes global statistics and provides per-domain groupings for fast lookup.

### Top-level fields
- `total_tables` (int): Number of bundles detected under `data/tables/`.
- `last_updated` (ISO 8601 string): Timestamp when the index was generated.
- `statistics` (object):
  - `avg_rows`, `avg_columns` (float): Mean dimensions across all tables.
  - `domain_distribution` (object): Table counts per domain (e.g., `{ "health": 59, "economics": 16 }`).
  - `complexity_distribution` (object): Totals for `low`, `medium`, `high` levels.

### `by_domain` section
Maps each domain to its constituent tables. Each entry provides:
- `table_id` (string): Bundle identifier, e.g., `table_0001`.
- `title` (string): Dataset title.
- `has_time_series` (bool)
- `row_count`, `column_count` (int)

Example excerpt:
```json
"health": [
  {
    "table_id": "table_0001",
    "title": "20th century deaths in US - CDC",
    "has_time_series": true,
    "row_count": 94,
    "column_count": 50
  }
]
```

## Topic Grouping Outputs
When the pipeline is configured with a topic output file (see `configs/data_processing.yaml`, `pipeline.topic_output_file`), the LLM-based grouper writes a lightweight JSON summary that mirrors the domain layout but retains only `table_id`, `title`, and `description` for each table. This design preserves essential context for question design while keeping topic archives compact.

## Downstream Usage Note (Observation Space)
All columns are treated as visible in downstream prompts. Row visibility depends on the observation-space mode configured in `configs/qna.yaml` (`qna.obs.mode`): `header_5`, `header_1`, or `full_table` (capped by `qna.obs.max_rows_per_table`). This ensures consistent, human-auditable context presentation for LLM evaluation.

## Rebuilding the Index
To regenerate the catalog index as part of a processing run, configure the appropriate pipeline mode in `configs/data_processing.yaml` and invoke the script, e.g.:
```bash
uv run python -m scripts.data_processing --config configs/data_processing.yaml
```

## Logging
The pipeline uses structured logging with mirrored output to both terminal and file. Log files are named using a datestamp convention (`YYYYMMDD_HHMMSS.log`) and stored under the configured logs directory.

## Related Configuration
- `configs/data_processing.yaml` — pipeline controls for ingestion, cleaning, metadata extraction, and storage.
- `configs/README.md` — consolidated documentation for all configuration files and key fields.
