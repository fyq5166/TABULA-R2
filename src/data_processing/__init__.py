"""
Data Processing Module

This module handles the ingestion, cleaning, and preprocessing of OWID datasets.
It includes functionality for:
- OWID dataset processing and validation
- Data cleaning and standardization
- Metadata extraction using LLM classification
- Quality control and filtering
"""

# Import implemented classes
from .owid_processor import OWIDProcessor
from .table_cleaner import TableCleaner
from .metadata_extractor import MetadataExtractor
from .llm_domain_classifier import LLMDomainClassifier, ClassificationResult
from .table_grouper import group_titles_by_topic_llm, create_topic_grouped_index

__all__ = [
    "OWIDProcessor",
    "TableCleaner",
    "MetadataExtractor",
    "LLMDomainClassifier",
    "ClassificationResult",
    "group_titles_by_topic_llm",
    "create_topic_grouped_index",
]
