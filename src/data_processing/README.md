# data_processing Module

## Overview

The data processing module provides a comprehensive pipeline for transforming raw Our World in Data (OWID) datasets into structured, reasoning-ready table bundles. It implements automated discovery, validation, cleaning, domain classification, and metadata extraction to support downstream table reasoning evaluation frameworks.

**Core Functionality:**
- Automated OWID dataset discovery and validation
- Data quality enforcement through configurable cleaning policies
- LLM-based domain classification with heuristic fallbacks
- Structural metadata extraction for complexity assessment
- Domain-organized index generation for efficient discovery
- Topic clustering for thematic organization

**Design Principles:**
- **Reproducibility**: Deterministic outputs for identical inputs
- **Modularity**: Independent components with clear interfaces
- **Robustness**: Graceful degradation when external services unavailable
- **Extensibility**: Easy addition of new cleaning rules and classifiers

## Architecture & Key Components

### Data Flow
```
Raw OWID Datasets → Validation → Cleaning → Domain Classification → Metadata Extraction → Index Generation → Table Bundles
```

### Core Components

#### **Data Processing Pipeline**
- [`owid_processor.py`](owid_processor.py): Orchestrates the complete processing pipeline for individual datasets. Implements the main workflow: scan → validate → load → clean → classify → metadata extraction.
- [`table_cleaner.py`](table_cleaner.py): Applies configurable data quality rules including missing-value policies, size constraints, deduplication, and column normalization.
- [`index_builder.py`](index_builder.py): Generates domain-organized catalog indices (`table_index.json`) with global statistics and per-domain groupings.

#### **Classification & Metadata**
- [`llm_domain_classifier.py`](llm_domain_classifier.py): Maps tables to canonical domains using local LLM inference. Implements fallback to heuristic classification when LLM unavailable.
- [`metadata_extractor.py`](metadata_extractor.py): Computes structural metadata including dimensions, complexity tiers, reasoning affordances, and temporal characteristics.
- [`table_grouper.py`](table_grouper.py): Groups tables into thematic clusters using LLM-based topic analysis. Produces lightweight summaries for question design.

### Entry Points
- [`scripts/data_processing.py`](../../scripts/data_processing.py): Main CLI interface for pipeline execution
- [`configs/data_processing.yaml`](../../configs/data_processing.yaml): Centralized configuration management

## Configuration

The module reads configuration from [`configs/data_processing.yaml`](../../configs/data_processing.yaml) with the following key sections:

### **OWID Settings**
- `owid.source_path`: Root directory containing OWID datasets
- `owid.size_gate`: Minimum dataset size threshold for processing

### **Cleaning Policies**
- `cleaning.max_rows`: Maximum rows per table (default: 1000)
- `cleaning.max_columns`: Maximum columns per table (default: 50)
- `cleaning.min_numeric_ratio`: Minimum ratio of numeric columns
- `cleaning.missing_value_policy`: Handling strategy for missing values

### **Quality Control**
- `quality_control.min_completeness`: Minimum data completeness threshold
- `quality_control.min_rows`: Minimum rows after cleaning
- `quality_control.min_columns`: Minimum columns after cleaning

### **Metadata & Classification**
- `metadata.llm_model`: LLM model for domain classification
- `metadata.require_llm`: Whether LLM classification is mandatory
- `metadata.domain_categories`: Canonical domain list
- `metadata.retries`: Number of retry attempts for LLM calls

### **Storage & Pipeline**
- `storage.base_directory`: Output directory for table bundles
- `pipeline.mode`: Execution mode (full, index_only, etc.)
- `pipeline.limit`: Optional processing limit for testing

## Logging

The module implements structured logging with dual output (file + console):

### **Log File Format**
- **Location**: `logs/YYYYMMDD_HHMMSS.log`
- **Content**: Hierarchical logging with timestamps, module names, and context
- **Levels**: DEBUG, INFO, WARNING, ERROR

### **Console Output**
- **Mirror**: All log messages mirrored to terminal for real-time monitoring
- **Format**: Human-readable with color coding for different log levels

### **Key Log Events**
- Dataset discovery and validation results
- Cleaning statistics and quality metrics
- Domain classification outcomes
- Index generation progress
- Error conditions and recovery actions

## Public API & Contracts

### **Core Interfaces**

#### **OWIDProcessor**
```python
def process_single_dataset(dataset_name: str) -> dict | None:
    """
    Process a single OWID dataset through the complete pipeline.
    
    Returns:
        dict: {
            'raw_data': DataFrame,
            'cleaned_data': DataFrame, 
            'metadata': dict,
            'cleaning_summary': dict
        } or None if processing fails
    """
```

#### **TableCleaner**
```python
def clean_dataset(df: DataFrame) -> DataFrame | None:
    """
    Apply cleaning policies to a dataset.
    
    Returns:
        DataFrame: Cleaned dataset or None if quality thresholds not met
    """
```

#### **LLMDomainClassifier**
```python
def classify_table(title: str, description: str) -> str:
    """
    Classify table domain using LLM or heuristics.
    
    Returns:
        str: Canonical domain label
    """
```

#### **MetadataExtractor**
```python
def extract_metadata(df: DataFrame, domain: str) -> dict:
    """
    Extract structural metadata from cleaned dataset.
    
    Returns:
        dict: Metadata including complexity tier and reasoning flags
    """
```

### **Error Handling**
- **LLM Unavailable**: Graceful fallback to heuristic classification
- **Cleaning Failures**: Log warnings and skip problematic datasets
- **Validation Errors**: Detailed error messages with recovery suggestions
- **Index Generation**: Partial success handling with error reporting

## Data Inputs/Outputs

### **Input Format**
- **Source**: OWID-compliant datasets in CSV format
- **Structure**: Standard OWID packaging with datapackage.json metadata
- **Encoding**: UTF-8 with automatic fallback detection

### **Output Structure**
Each processed dataset generates a self-contained bundle:

```
data/tables/table_XXXX/
├── table.csv           # Cleaned dataset
├── raw.csv            # Original snapshot (optional)
├── meta.json          # Extracted metadata
└── quality_report.json # Cleaning statistics
```

### **Metadata Schema** (meta.json)
- `table_id`: Unique identifier
- `title`: Human-readable dataset title
- `description`: Dataset description
- `columns`: List of column names
- `row_count`, `column_count`: Dimensions
- `numeric_columns`: Numeric column list
- `has_time_series`: Temporal data flag
- `domain`: Canonical domain classification
- `complexity_level`: Difficulty tier (low/medium/high)
- `suitable_for_reasoning`: Reasoning task suitability

### **Quality Report Schema** (quality_report.json)
- `original_shape`: Pre-cleaning dimensions
- `final_shape`: Post-cleaning dimensions
- `rows_removed`, `columns_removed`: Cleaning statistics
- `data_completeness`: Completeness ratio
- `numeric_columns`: Numeric column count

## Extensibility Guidelines

### **Adding New Cleaning Rules**
1. Extend `TableCleaner` class with new methods
2. Update configuration schema in `data_processing.yaml`
3. Add validation logic in `_apply_size_constraints()`
4. Test with representative datasets

### **Custom Domain Classifiers**
1. Implement new classifier in `llm_domain_classifier.py`
2. Update `_build_keyword_domains()` with domain mappings
3. Add fallback logic for LLM unavailability
4. Update domain categories in configuration

### **Metadata Enhancement**
1. Extend `MetadataExtractor` with new heuristics
2. Update complexity calculation in `_calculate_complexity_level()`
3. Add new reasoning affordance detection
4. Update output schema documentation

### **Pipeline Stages**
1. Add new processing steps in `owid_processor.py`
2. Maintain backward compatibility with existing contracts
3. Update configuration schema
4. Add comprehensive logging and error handling

## Performance & Scalability

### **Memory Management**
- Streaming processing for large datasets
- Configurable row/column limits
- Efficient DataFrame operations

### **Concurrency**
- Sequential processing to ensure reproducibility
- LLM call batching for efficiency
- Progress tracking for long-running operations

### **Storage Optimization**
- Compressed output formats
- Optional raw data archiving
- Efficient index generation

## Related Documentation

- [`configs/README.md`](../../configs/README.md) — Configuration reference and parameter descriptions
- [`data/tables/README.md`](../../data/tables/README.md) — Output schema and bundle structure
- [`scripts/README.md`](../../scripts/README.md) — CLI usage and pipeline modes
- [`experiments/PLAN.md`](../../experiments/PLAN.md) — Experimental design and dataset requirements
