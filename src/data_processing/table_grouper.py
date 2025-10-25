"""
Table Grouper

This module provides functionality for grouping tables by topic using LLM-based analysis.
It has been refactored into an object-oriented structure for better maintainability.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..prompts import prompt_manager
from ..utils.io_utils import safe_load_json, safe_save_json
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TableGrouper:
    """
    Groups table titles by topic using an LLM.

    This class encapsulates the logic for prompt generation, LLM interaction,
    and post-processing of results to create meaningful table groups.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the TableGrouper.

        Args:
            llm_client (LLMClient): A pre-configured client for LLM interaction.
        """
        self.llm_client = llm_client
        if not self.llm_client.is_available:
            raise RuntimeError(
                "TableGrouper requires an available LLM client. Ensure Ollama is running and reachable."
            )
        self.domain_examples = {
            "health": "Mortality_Stats_and_Trends, Child_Health_and_Development, Nutrition_and_Dietary_Health",
            "economics": "Economic_Inequality_and_Governance, Labor_Markets_and_Employment, Energy_Economics",
            "environment": "Carbon_Emissions_and_Climate, Air_Pollution_and_Health_Impact, Industrial_Emissions",
            "demographics": "Population_Growth_and_Urbanization, Marriage_and_Family_Structure",
            "education": "Educational_Assessment_and_Outcomes, Literacy_and_Skills_Development",
        }

    def group_titles_by_topic(
        self, titles: List[str], domain: str
    ) -> List[Dict[str, Any]]:
        """
        Use an LLM to group table titles by topic.

        Args:
            titles (List[str]): List of table titles to group.
            domain (str): Domain name for context.

        Returns:
            List[Dict[str, Any]]: List of topic groups with titles and group names.
        """
        context = {
            "domain": domain,
            "examples": self.domain_examples.get(
                domain, self.domain_examples["health"]
            ),
            "titles": titles,
        }
        prompt = prompt_manager.get_prompt("topic_grouping", context)

        max_retries = 3
        for attempt in range(max_retries):
            logger.debug(
                f"LLM grouping attempt {attempt + 1}/{max_retries} for domain '{domain}' ({len(titles)} titles)"
            )

            response_text = self.llm_client.generate(prompt)

            if response_text:
                try:
                    start_idx = response_text.find("[")
                    end_idx = response_text.rfind("]") + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = response_text[start_idx:end_idx]
                        raw_groups = json.loads(json_str)

                        if isinstance(raw_groups, list) and raw_groups:
                            processed_groups = self._post_process_groups(
                                raw_groups, titles
                            )
                            final_groups = self._merge_similar_groups(processed_groups)

                            logger.info(
                                f"Successfully grouped {len(titles)} titles into {len(final_groups)} groups for domain '{domain}'"
                            )
                            return final_groups

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse LLM JSON response on attempt {attempt + 1}: {e}"
                    )
                    continue

        raise RuntimeError(
            f"LLM grouping failed after {max_retries} attempts for domain '{domain}'."
        )

    def _post_process_groups(
        self, raw_groups: List[Dict], all_titles: List[str]
    ) -> List[Dict[str, Any]]:
        """Process raw groups from LLM, match titles, and clean names."""
        processed_groups = []
        used_titles = set()

        for group in raw_groups:
            if (
                not isinstance(group, dict)
                or "group_name" not in group
                or "titles" not in group
            ):
                continue

            group_name = self._clean_group_name(group["group_name"])
            llm_titles = group.get("titles", [])

            if not isinstance(llm_titles, list):
                continue

            matched_titles = []
            for llm_title in llm_titles:
                if not isinstance(llm_title, str):
                    continue

                matched_title = self._find_best_matching_title(llm_title, all_titles)
                if matched_title and matched_title not in used_titles:
                    matched_titles.append(matched_title)
                    used_titles.add(matched_title)

            if matched_titles:
                processed_groups.append(
                    {"group_name": group_name, "titles": matched_titles}
                )

        # Add remaining unmatched titles as individual groups
        remaining_titles = [t for t in all_titles if t not in used_titles]
        for title in remaining_titles:
            group_name = self._clean_group_name(title)
            processed_groups.append({"group_name": group_name, "titles": [title]})

        return processed_groups

    def _find_best_matching_title(
        self, target_title: str, actual_titles: List[str]
    ) -> Optional[str]:
        """Find the best matching actual title using multiple strategies."""
        target_lower = target_title.lower().strip()

        for actual_title in actual_titles:
            if target_lower == actual_title.lower().strip():
                return actual_title

        for actual_title in actual_titles:
            actual_lower = actual_title.lower()
            if target_lower in actual_lower or actual_lower in target_lower:
                return actual_title

        best_match = max(
            actual_titles,
            key=lambda title: SequenceMatcher(
                None, target_lower, title.lower()
            ).ratio(),
            default=None,
        )

        if (
            best_match
            and SequenceMatcher(None, target_lower, best_match.lower()).ratio() >= 0.7
        ):
            return best_match

        return None

    def _clean_group_name(self, name: str) -> str:
        """Clean and standardize group names with Title_Case format."""
        cleaned = re.sub(r"[^\w\s-]", "", name).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)

        words = [word.capitalize() for word in cleaned.split("_") if word]
        cleaned = "_".join(words)

        if len(cleaned) > 60:
            cleaned = cleaned[:60].rsplit("_", 1)[0]

        return cleaned or "Untitled_Group"

    def _merge_similar_groups(
        self, groups: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge groups with similar names or overlapping content."""
        if not groups:
            return []

        merged = True
        while merged:
            merged = False
            merged_groups = []
            used_indices = set()

            for i in range(len(groups)):
                if i in used_indices:
                    continue

                current_group = groups[i]
                current_titles = set(current_group["titles"])

                for j in range(i + 1, len(groups)):
                    if j in used_indices:
                        continue

                    other_group = groups[j]
                    other_titles = set(other_group["titles"])

                    # Merge if names are similar or titles overlap significantly
                    name_similarity = SequenceMatcher(
                        None, current_group["group_name"], other_group["group_name"]
                    ).ratio()
                    title_overlap = len(
                        current_titles.intersection(other_titles)
                    ) / len(current_titles.union(other_titles))

                    if name_similarity > 0.7 or title_overlap > 0.5:
                        # Merge these two groups
                        new_name = (
                            current_group["group_name"]
                            if len(current_group["group_name"])
                            >= len(other_group["group_name"])
                            else other_group["group_name"]
                        )
                        all_titles = sorted(list(current_titles.union(other_titles)))

                        current_group = {"group_name": new_name, "titles": all_titles}
                        current_titles = set(all_titles)

                        used_indices.add(j)
                        merged = True

                merged_groups.append(current_group)
                used_indices.add(i)

            groups = merged_groups

        # Filter out single-table groups if possible, unless they are all single
        multi_table_groups = [g for g in groups if len(g["titles"]) > 1]
        if multi_table_groups:
            return multi_table_groups
        return groups


# Standalone functions that now use the TableGrouper class
# This maintains the external API while benefiting from the new structure.


def group_titles_by_topic_llm(
    titles: List[str], domain: str, config: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to group table titles by topic. Wraps the TableGrouper class.

    Args:
        titles (List[str]): List of table titles to group.
        domain (str): Domain name for context.
        config (Optional[Dict]): Optional config for LLMClient.

    Returns:
        List[Dict[str, Any]]: List of topic groups.
    """
    if config is None:
        config = {"llm": {"model_name": "llama3.2"}}  # Default config

    llm_client = LLMClient(
        model_name=config.get("llm", {}).get("model_name", "llama3.2")
    )
    grouper = TableGrouper(llm_client)
    return grouper.group_titles_by_topic(titles, domain)


def create_topic_grouped_index(
    tables_dir: Path,
    config: Optional[Dict] = None,
    output_file_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a comprehensive, topic-grouped index of all tables.

    This function scans a directory of processed tables, groups them by domain,
    and then uses an LLM to further group them by topic within each domain.
    The final index is saved to a JSON file.

    Args:
        tables_dir (Path): The directory containing the processed table folders.
        config (Optional[Dict]): Application configuration.
        output_file_override (Optional[str]): If provided, save the index to this full path instead of the default.

    Returns:
        Dict[str, Any]: The generated topic-grouped index.
    """
    if config is None:
        from ..utils.config import load_config

        config = load_config()

    logger.info(f"Starting topic-grouped index creation for directory: {tables_dir}")

    index_path = tables_dir / "tables_index.json"
    if not index_path.exists():
        logger.error(f"Index file not found at {index_path}")
        return {}

    index_data = safe_load_json(index_path, {})
    domain_tables = index_data.get("domains", {})

    grouped_index = {}
    for domain, tables in domain_tables.items():
        logger.info(f"Grouping {len(tables)} tables for domain: {domain}")
        titles = [table_info["title"] for table_info in tables]

        # Use the standalone wrapper function
        topic_groups = group_titles_by_topic_llm(titles, domain, config)

        # Map titles back to full table info
        title_to_info = {info["title"]: info for info in tables}

        final_groups = []
        for group in topic_groups:
            group_tables = [
                title_to_info[title]
                for title in group["titles"]
                if title in title_to_info
            ]
            if group_tables:
                final_groups.append(
                    {"group_name": group["group_name"], "tables": group_tables}
                )

        grouped_index[domain] = final_groups

    # Determine output path
    if output_file_override:
        output_path = Path(output_file_override)
    else:
        output_path = tables_dir / "topic_grouped_index.json"

    # Save the final index
    safe_save_json(grouped_index, output_path)
    logger.info(f"Topic-grouped index successfully saved to: {output_path}")

    return grouped_index
