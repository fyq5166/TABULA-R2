"""Index builder utilities for processed tables."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..utils.io_utils import scan_existing_tables, safe_load_json, safe_save_json
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_domain_index(
    base_dir: Path, output_file: str = "my_test_index.json"
) -> Dict[str, Any]:
    """Create a simplified, domain-organized index for processed tables."""
    base_dir = Path(base_dir)
    table_ids = scan_existing_tables(base_dir)

    logger.info("Scanning %s for processed tables", base_dir)

    domain_index: Dict[str, list] = {}
    all_tables_metadata: Dict[str, Dict[str, Any]] = {}

    for table_id in table_ids:
        meta_path = base_dir / table_id / "meta.json"
        if not meta_path.exists():
            logger.warning("meta.json missing for %s; skipping", table_id)
            continue

        metadata = safe_load_json(meta_path, default={})
        table_info = {
            "table_id": table_id,
            "title": metadata.get("title", "Unknown"),
            "has_time_series": metadata.get("has_time_series", False),
            "row_count": metadata.get("row_count", 0),
            "column_count": metadata.get("column_count", 0),
        }

        domain = metadata.get("domain", "unknown")
        domain_index.setdefault(domain, []).append(table_info)

        all_tables_metadata[table_id] = {
            "domain": domain,
            "complexity_level": metadata.get("complexity_level", "unknown"),
            "row_count": table_info["row_count"],
            "column_count": table_info["column_count"],
        }

    for tables in domain_index.values():
        tables.sort(key=lambda x: x["title"])

    total_tables = len(table_ids)
    avg_rows = (
        sum(meta.get("row_count", 0) for meta in all_tables_metadata.values())
        / total_tables
        if total_tables
        else 0
    )
    avg_columns = (
        sum(meta.get("column_count", 0) for meta in all_tables_metadata.values())
        / total_tables
        if total_tables
        else 0
    )

    domain_distribution = {
        domain: len(tables) for domain, tables in domain_index.items()
    }
    for domain, count in sorted(domain_distribution.items(), key=lambda item: item[0]):
        logger.info("Domain '%s': %d table(s)", domain, count)

    complexity_distribution: Dict[str, int] = {}
    for meta in all_tables_metadata.values():
        complexity = meta.get("complexity_level", "unknown")
        complexity_distribution[complexity] = (
            complexity_distribution.get(complexity, 0) + 1
        )

    index_data: Dict[str, Any] = {
        "total_tables": total_tables,
        "last_updated": pd.Timestamp.now().isoformat(),
        "statistics": {
            "total_tables": total_tables,
            "avg_rows": avg_rows,
            "avg_columns": avg_columns,
            "domain_distribution": domain_distribution,
            "complexity_distribution": complexity_distribution,
        },
        "by_domain": domain_index,
    }

    index_path = base_dir / output_file
    safe_save_json(index_data, index_path)
    logger.info(
        "Domain index saved to %s (tables=%d, domains=%d)",
        index_path,
        total_tables,
        len(domain_index),
    )

    return index_data
