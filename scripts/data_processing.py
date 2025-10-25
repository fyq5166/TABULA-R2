from pathlib import Path
import argparse
import time
import sys
from collections import defaultdict
from typing import Any, Dict

# Add src to sys.path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import (
    create_session_log_file,
    setup_logging,
    get_logger,
)
from src.utils.io_utils import (
    save_table_data,
    scan_existing_tables,
    get_processed_dataset_names,
    safe_save_json,
    safe_load_json,
)
from src.data_processing.index_builder import create_domain_index
from src.data_processing.owid_processor import OWIDProcessor
from src.data_processing.table_grouper import TableGrouper
from src.utils.llm_client import LLMClient


def main():
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the OWID data processing pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of OWID datasets processed this run (overrides pipeline.limit in YAML).",
    )
    parser.add_argument(
        "--topic-output",
        type=str,
        default=None,
        help="Path to write LLM-generated topic groupings for newly processed tables (overrides pipeline.topic_output_file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for table bundles (overrides storage.base_directory / pipeline.output_dir).",
    )
    args = parser.parse_args()

    # Setup logging: write to file and mirror to terminal (plain filename format)
    session_log = create_session_log_file(base_dir="logs", plain_name=True)
    setup_logging(
        log_level="INFO",
        log_file=str(session_log),
        console_output=True,
        verbose=False,
        plain_file=True,
    )
    logger = get_logger(__name__)

    logger.debug("======================================================")
    logger.debug("      UNIFIED DATA PROCESSING PIPELINE STARTING  ")
    logger.debug("======================================================")

    try:
        # Load configuration
        config = load_config("configs/data_processing.yaml")

        pipeline_cfg = config.get("pipeline", {})
        mode = pipeline_cfg.get("mode", "tables_with_index")
        index_filename = pipeline_cfg.get("index_filename", "my_test_index.json")

        valid_modes = {"tables_only", "tables_with_index", "index_only", "topics_only"}
        if mode not in valid_modes:
            logger.error(
                f"Invalid pipeline.mode '{mode}'. Expected one of {sorted(valid_modes)}"
            )
            sys.exit(1)

        process_tables = mode in {"tables_only", "tables_with_index"}
        build_index = mode in {"tables_with_index", "index_only"}
        group_only = mode == "topics_only"

        limit_cfg = pipeline_cfg.get("limit")
        output_file_cfg = pipeline_cfg.get("topic_output_file")
        output_dir_cfg = pipeline_cfg.get("output_dir")

        limit = args.limit if args.limit is not None else limit_cfg
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                logger.warning(
                    f"Unable to interpret limit '{limit}' as integer; processing all datasets instead."
                )
                limit = None

        topic_output = args.topic_output if args.topic_output else output_file_cfg
        if group_only and not topic_output:
            logger.error(
                "topics_only mode requires --topic-output or pipeline.topic_output_file in the config."
            )
            sys.exit(1)
        output_dir = args.output_dir if args.output_dir else output_dir_cfg

        if output_dir:
            override_path = Path(output_dir).resolve()
            override_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Overriding storage.base_directory -> {override_path}")
            config["storage"]["base_directory"] = str(override_path)

        tables_dir = Path(config["storage"]["base_directory"])

        newly_processed_tables = []
        table_grouper = None

        if process_tables:
            processor = OWIDProcessor(config=config)
            if topic_output:
                llm_client = LLMClient(model_name=config["metadata"]["llm_model"])
                table_grouper = TableGrouper(llm_client=llm_client)

            all_datasets = processor.scan_owid_datasets()
            processed_datasets = get_processed_dataset_names(tables_dir)
            datasets_to_process = [
                d for d in all_datasets if d not in processed_datasets
            ]

            if limit is not None:
                datasets_to_process = datasets_to_process[:limit]

            if datasets_to_process:
                logger.debug(
                    f"Preparing to process {len(datasets_to_process)} dataset(s)"
                )
                start_time = time.time()

                for i, dataset_name in enumerate(datasets_to_process):
                    logger.info(
                        f"Processing table {i+1}/{len(datasets_to_process)}: {dataset_name}"
                    )
                    result = processor.process_single_dataset(dataset_name)

                    if result:
                        existing_tables_count = len(scan_existing_tables(tables_dir))
                        table_id = f"table_{existing_tables_count + 1:04d}"
                        table_path = tables_dir / table_id

                        table_path.mkdir(parents=True, exist_ok=True)

                        save_args = {
                            "raw_data": result["raw_data"],
                            "processed_data": result["cleaned_data"],
                            "metadata": result["metadata"],
                            "cleaning_summary": result["cleaning_summary"],
                            "original_name": result["dataset_name"],
                        }

                        save_table_data(
                            table_dir=table_path, table_id=table_id, **save_args
                        )
                        logger.debug(
                            "Saved dataset '%s' as %s (Domain: %s)",
                            dataset_name,
                            table_id,
                            result.get("metadata", {}).get("domain", "N/A"),
                        )

                        full_metadata = result["metadata"]
                        full_metadata["table_id"] = table_id
                        newly_processed_tables.append(full_metadata)
                    else:
                        logger.warning(f"Processing failed for {dataset_name}")

                processing_time = time.time() - start_time
                logger.debug(
                    f"Dataset processing finished in {processing_time:.2f} seconds"
                )
            else:
                logger.debug("No new datasets to process.")
        else:
            logger.debug(
                "Pipeline mode set to '%s'; skipping dataset processing.", mode
            )

        def _summarize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "table_id": meta.get("table_id"),
                "title": meta.get("title", meta.get("table_id", "unknown")),
                "description": meta.get("description", ""),
            }

        if process_tables and topic_output and newly_processed_tables:
            logger.debug(
                f"Creating topic-grouped index for {len(newly_processed_tables)} new table(s)"
            )
            start_index_time = time.time()

            titles_by_domain = defaultdict(list)
            table_info_map = defaultdict(list)
            for item in newly_processed_tables:
                title = item.get("title", f"untitled_{item.get('table_id')}")
                domain = item.get("domain", "unknown")
                titles_by_domain[domain].append(title)
                table_info_map[title].append(_summarize_metadata(item))

            final_grouped_index = defaultdict(list)
            for domain, titles in titles_by_domain.items():
                if not titles:
                    continue
                logger.debug(f"Grouping {len(titles)} titles for domain: '{domain}'")
                grouped_topics = table_grouper.group_titles_by_topic(titles, domain)

                for topic_group in grouped_topics:
                    group_titles = topic_group.get("titles", [])
                    tables_in_group = []
                    for title in group_titles:
                        if table_info_map.get(title):
                            tables_in_group.append(table_info_map[title].pop(0))
                    if tables_in_group:
                        final_grouped_index[domain].append(
                            {
                                "topic_group_name": topic_group.get(
                                    "group_name", "Untitled"
                                ),
                                "tables": tables_in_group,
                            }
                        )

            output_path = Path(topic_output)
            safe_save_json(dict(final_grouped_index), output_path)
            logger.debug(
                f"Topic-grouped index with full metadata saved to {output_path}"
            )

            index_time = time.time() - start_index_time
            logger.debug(f"Topic grouping finished in {index_time:.2f} seconds")
        elif process_tables and topic_output:
            logger.debug(
                "No new tables were processed, skipping topic grouping output."
            )

        if group_only:
            if table_grouper is None:
                llm_client = LLMClient(model_name=config["metadata"]["llm_model"])
                table_grouper = TableGrouper(llm_client=llm_client)

            domain_titles = defaultdict(list)
            table_info_map = defaultdict(list)

            for table_id in scan_existing_tables(tables_dir):
                meta_path = tables_dir / table_id / "meta.json"
                if not meta_path.exists():
                    continue
                metadata = safe_load_json(meta_path, default={})
                metadata["table_id"] = table_id
                summary = _summarize_metadata(metadata)
                title = summary["title"]
                domain = metadata.get("domain", "unknown")
                domain_titles[domain].append(title)
                table_info_map[title].append(summary)

            if not domain_titles:
                logger.warning(
                    "No tables found under %s; skipping topic grouping.", tables_dir
                )
            else:
                logger.debug(
                    "Grouping existing tables across %d domains", len(domain_titles)
                )
                final_grouped_index = defaultdict(list)

                for domain, titles in domain_titles.items():
                    logger.debug(
                        "Grouping %d titles for domain: '%s'", len(titles), domain
                    )
                    grouped_topics = table_grouper.group_titles_by_topic(titles, domain)

                    for topic_group in grouped_topics:
                        tables_in_group = []
                        for title in topic_group.get("titles", []):
                            if table_info_map[title]:
                                tables_in_group.append(table_info_map[title].pop(0))
                        if tables_in_group:
                            final_grouped_index[domain].append(
                                {
                                    "topic_group_name": topic_group.get(
                                        "group_name", "Untitled"
                                    ),
                                    "tables": tables_in_group,
                                }
                            )

                output_path = Path(topic_output)
                safe_save_json(dict(final_grouped_index), output_path)
                logger.debug(
                    "Topic grouping (existing tables) saved to %s", output_path
                )

        if build_index:
            index_data = create_domain_index(tables_dir, index_filename)
            logger.info(
                f"Domain index written to {tables_dir / index_filename} ({index_data['total_tables']} tables)"
            )

        logger.debug("======================================================")
        logger.debug("  DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY  ")
        logger.debug("======================================================")

    except Exception as e:
        logger.critical(
            f"A critical error occurred in the pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
