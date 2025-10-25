"""
LLM-based Domain Classifier using Ollama

This module uses local LLM models via Ollama to classify tables into domain categories
based on their descriptions and titles.
"""

import json
import requests
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import pandas as pd

from ..utils.logging_utils import get_logger
from ..utils.config import get_config_value
from ..utils.llm_client import LLMClient
from ..prompts.prompt_manager import prompt_manager


@dataclass
class ClassificationResult:
    """Domain classification result."""

    primary_domain: str
    reasoning: str = ""


class LLMDomainClassifier:
    """
    Classifies table domains using a local LLM.
    Refactored for better configuration management and prompt injection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM Domain Classifier.

        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.logger = get_logger(__name__)
        self.config = config

        # Get LLM configuration from the correct section
        model_name = get_config_value(config, "metadata.llm_model", "llama3")
        ollama_url = get_config_value(
            config, "llm.ollama_url", "http://localhost:11434"
        )  # Assuming this might be separate

        self.max_retries = get_config_value(config, "metadata.max_retries", 3)

        # Initialize LLM client with the correct model name string
        self.llm_client = LLMClient(model_name=model_name, url=ollama_url)

        self.domain_categories = get_config_value(
            config, "metadata.domain_categories", []
        )
        if not self.domain_categories:
            self.logger.warning("No domain categories found in configuration.")

        # Build alias mapping so synonyms collapse to the canonical labels expected downstream
        self.domain_aliases = self._build_domain_aliases(self.domain_categories)
        # Legacy processor treated uncategorized tables as general; make that explicit
        self.default_domain = "general"
        self.require_llm = get_config_value(config, "metadata.require_llm", False)

        # Keyword heuristics cover the legacy datasets when the LLM client is unavailable
        self.keyword_domains = self._build_keyword_domains()

    def classify_table(
        self, df: pd.DataFrame, table_name: str, description: Optional[str] = None
    ) -> Optional[str]:
        """
        Classify a table into a domain category.

        Args:
            df (pd.DataFrame): Table data
            table_name (str): Table name

        Returns:
            Optional[str]: Classified domain or None if classification failed
        """
        title = table_name
        description_text = description if description else df.to_markdown()

        # Build classification prompt
        prompt = self._build_classification_prompt(title, description_text)

        domain = None

        if self.llm_client.is_available:
            try:
                generation_options = {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "num_predict": 150,
                }
                response = self.llm_client.generate(prompt, options=generation_options)

                if response is None:
                    raise RuntimeError("LLM generation failed (received None response)")

                result = self._parse_classification_response(response)
                domain = result.primary_domain
                self.logger.debug("LLM classified '%s' as '%s'", title, domain)

            except Exception as e:
                self.logger.error(f"Classification failed for '{title}': {e}")
                if self.require_llm:
                    raise
        elif self.require_llm:
            self.logger.error("LLM classification required but client unavailable")
            raise RuntimeError("LLM classification required but client unavailable")

        if not domain or domain == "general":
            domain = self._heuristic_domain(title, df)
            if domain:
                self.logger.debug("Heuristic classified '%s' as '%s'", title, domain)
            elif self.require_llm:
                raise RuntimeError(
                    "LLM classification required but no domain identified"
                )

        return self._normalize_domain(domain)

    def _build_classification_prompt(self, title: str, description: str) -> str:
        """
        Build classification prompt for the LLM.

        Args:
            title (str): Table title
            description (str): Table description

        Returns:
            str: Formatted prompt
        """
        domains_str = ", ".join(self.domain_categories)

        prompt = f"""You are a data classification expert. Classify the following dataset into one of these domains:

Domains: {domains_str}

Dataset Information:
Title: {title}
Description: {description}

Instructions:
1. Choose the MOST appropriate domain from the list above

Respond in this exact JSON format:
{{
    "domain": "domain_name"
}}

Response:"""

        return prompt

    def _parse_classification_response(self, response: str) -> ClassificationResult:
        """
        Parse LLM response to extract classification result.

        Args:
            response (str): Raw LLM response

        Returns:
            ClassificationResult: Parsed classification
        """
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON block
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)

                domain = data.get("domain", "general").lower()

                # Validate domain
                if domain not in self.domain_categories:
                    self.logger.warning(f"Invalid domain '{domain}', using 'general'")
                    domain = "general"

                return ClassificationResult(
                    primary_domain=domain, reasoning=data.get("reasoning", "")
                )

        except Exception as e:
            self.logger.warning(f"Failed to parse classification response: {e}")

        # Fallback classification
        return ClassificationResult(
            primary_domain="general",
            reasoning="Failed to parse classification response",
        )

    def _normalize_domain(self, candidate: Optional[str]) -> Optional[str]:
        """Map a candidate domain to the canonical label expected downstream."""
        if not candidate:
            return self.default_domain

        candidate_lower = candidate.lower()
        if candidate_lower in self.domain_aliases:
            return self.domain_aliases[candidate_lower]

        if candidate_lower in self.domain_categories:
            return candidate_lower

        self.logger.debug(
            "Unknown domain '%s', defaulting to %s", candidate, self.default_domain
        )
        return self.default_domain

    def _build_domain_aliases(self, domain_categories: List[str]) -> Dict[str, str]:
        """Create a mapping from common aliases to canonical domain labels."""
        aliases = {domain.lower(): domain for domain in domain_categories}

        # Legacy processors stored pluralised forms; keep lightweight synonyms for safety
        synonym_map = {
            "economic": "economics",
            "demographic": "demographics",
            "environmental": "environment",
            "tech": "technology",
            # Social and political content collapsed into the general bucket for legacy parity
            "social": "general",
            "social_science": "general",
            "political": "general",
            "governance": "general",
        }
        for alias, target in synonym_map.items():
            if target in domain_categories:
                aliases[alias] = target

        aliases.setdefault("general", "general")
        return aliases

    def _build_keyword_domains(self) -> Dict[str, List[str]]:
        """Keyword lists for heuristic domain classification."""
        return {
            "health": [
                "health",
                "disease",
                "mortality",
                "life expectancy",
                "hospital",
                "covid",
                "hiv",
                "obesity",
                "vaccine",
                "malaria",
                "diabetes",
                "heart",
                "cancer",
                "cause of death",
            ],
            "economics": [
                "econom",
                "gdp",
                "income",
                "consumption",
                "trade",
                "inflation",
                "employment",
                "unemployment",
                "wage",
                "finance",
                "industry",
                "productivity",
                "market",
                "gini",
            ],
            "environment": [
                "climate",
                "emission",
                "carbon",
                "co2",
                "pollution",
                "environment",
                "energy",
                "forest",
                "biodiversity",
                "fossil",
                "temperature",
                "water",
                "agriculture",
            ],
            "demographics": [
                "population",
                "fertility",
                "birth",
                "death rate",
                "migration",
                "demograph",
                "age structure",
                "life expectancy",
            ],
            "education": [
                "education",
                "literacy",
                "school",
                "student",
                "teacher",
                "learning",
                "enrol",
            ],
            "technology": [
                "technology",
                "internet",
                "ict",
                "digital",
                "innovation",
                "patent",
            ],
            "general": [
                "inequality",
                "poverty",
                "wellbeing",
                "happiness",
                "crime",
                "violence",
                "social",
                "politic",
                "governance",
                "democracy",
                "election",
                "military",
                "conflict",
                "war",
            ],
        }

    def _heuristic_domain(self, title: str, df: pd.DataFrame) -> Optional[str]:
        """Fallback keyword-based classifier when LLM output is unavailable."""
        searchable_text = f"{title} {' '.join(df.columns)}".lower()

        for domain, keywords in self.keyword_domains.items():
            if any(keyword in searchable_text for keyword in keywords):
                return (
                    domain
                    if domain in self.domain_categories
                    else self.domain_aliases.get(domain)
                )

        return None

    def classify_batch(
        self,
        df_list: List[pd.DataFrame],
        table_names: List[str],
        descriptions: Optional[List[Optional[str]]] = None,
    ) -> List[Optional[str]]:
        """
        Classify multiple tables in batch.

        Args:
            df_list (List[pd.DataFrame]): List of table data
            table_names (List[str]): List of table names

        Returns:
            List[Optional[str]]: List of classified domains or None if classification failed
        """
        results = []

        for i, (df, table_name) in enumerate(zip(df_list, table_names)):
            self.logger.debug(f"Classifying table {i+1}/{len(df_list)}")
            description = None
            if descriptions and i < len(descriptions):
                description = descriptions[i]
            result = self.classify_table(df, table_name, description)
            results.append(result)

        return results

    def get_domain_distribution(self, results: List[Optional[str]]) -> Dict[str, int]:
        """
        Get distribution of domains across classification results.

        Args:
            results (List[Optional[str]]): List of classified domains or None

        Returns:
            Dict[str, int]: Domain distribution
        """
        distribution = {domain: 0 for domain in self.domain_categories}

        for result in results:
            if result:
                distribution[result] += 1

        return distribution
