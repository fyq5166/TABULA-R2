## Module: Utils

### Overview
The `src/utils` module provides shared infrastructure utilities used across the TABULA-R2 project, including configuration management, I/O operations, logging, and LLM client functionality. This module serves as the foundation for other modules, providing standardized interfaces and error handling.

**Responsibilities:**
- Configuration loading and validation from YAML files
- File I/O operations with error handling and backup mechanisms
- Structured logging with console and file output
- LLM client for Ollama API communication
- Environment variable overrides and defaults

**Non-responsibilities:**
- Business logic implementation (handled by specific modules)
- Data processing operations (handled by `src/data_processing`)
- Evaluation logic (handled by `src/evaluation`)

### Architecture & Key Components
- [`config.py`](config.py): Loads and validates YAML configuration files with environment variable overrides. Guarantees consistent configuration access across modules.
- [`io_utils.py`](io_utils.py): Handles file I/O operations with error handling and backup mechanisms. Provides safe JSON/CSV operations and directory management.
- [`llm_client.py`](llm_client.py): Manages communication with Ollama API servers. Handles availability checks, generation requests, and error recovery.
- [`logging_utils.py`](logging_utils.py): Configures project-wide logging with console and file output. Provides structured logging for different modules and operations.

### Configuration
- [`configs/data_processing.yaml`](../../configs/data_processing.yaml)
  - `pipeline.mode`: {tables_only, tables_with_index, index_only, topics_only} - Processing mode
  - `storage.base_directory`: Output root for processed tables
  - `metadata.llm_model`: Model used for topic grouping
  - `metadata.llm_url`: LLM server URL for metadata extraction
- [`configs/qna.yaml`](../../configs/qna.yaml)
  - `qna.llm.url`: Ollama server URL (default: http://localhost:11434)
  - `qna.llm.model`: Default LLM model name
  - `qna.llm.timeout_s`: Request timeout in seconds
  - `qna.logging.level`: Logging level (INFO, DEBUG, etc.)

### Logging
- This module uses [`logging_utils.py`](logging_utils.py) for structured logging configuration.
- Logs are emitted to timestamped files under `logs/` and mirrored to the terminal.
- Logging levels: INFO for user-facing progress; DEBUG for detailed diagnostics.
- Supports separate console and file log levels for different output requirements.

### Public API & Contracts
- [`load_config()`](config.py#L23) `(config_path: Union[str, Path], validate_schema: bool = True) -> Dict[str, Any]`
  - Returns validated configuration dictionary. Raises `ConfigError` on invalid schema or file not found.
- [`get_config_value()`](config.py#L45) `(config: Dict[str, Any], key_path: str, default: Any = None) -> Any`
  - Returns nested configuration value with default fallback. Supports dot notation (e.g., "qna.llm.model").
- [`safe_save_json()`](io_utils.py#L45) `(data: Dict[str, Any], file_path: Union[str, Path], backup: bool = True) -> None`
  - Saves JSON data with atomic write and optional backup. Raises `IOError` on write failure.
- [`safe_load_json()`](io_utils.py#L65) `(file_path: Union[str, Path], default: Any = None) -> Any`
  - Loads JSON data with error handling. Returns default value on file not found or parse error.
- [`LLMClient.generate()`](llm_client.py#L45) `(prompt: str, options: Optional[Dict[str, Any]] = None) -> Optional[str]`
  - Returns generated text or None on failure. Raises `TimeoutError` on timeout, `ConnectionError` on server unavailable.
- [`setup_logging()`](logging_utils.py#L19) `(log_level: str = "INFO", log_file: Optional[Union[str, Path]] = None, console_output: bool = True) -> None`
  - Configures project-wide logging. Sets up both console and file handlers with appropriate formatting.

### Data Inputs/Outputs
- **Inputs**: YAML configuration files from `configs/` directory with environment variable overrides
- **Outputs**: 
  - Processed table bundles in `data/tables/table_XXXX/` with CSV, JSON, and metadata files
  - Log files in `logs/` with timestamped filenames
  - LLM responses from Ollama API endpoints
- **Configuration Schema**: Validates required sections (pipeline, storage, metadata, qna) with type checking

### Entry Points
- **CLI Usage**: 
```bash
# Configuration loading in scripts
python -c "from src.utils.config import load_config; config = load_config('configs/qna.yaml')"
```
- **Python API**: 
```python
from src.utils import load_config, get_config_value, safe_save_json, LLMClient, setup_logging

# Load configuration
config = load_config("configs/qna.yaml")
model_name = get_config_value(config, "qna.llm.model", "llama3")

# Setup logging
setup_logging(log_level="INFO", log_file="logs/app.log")

# Use LLM client
client = LLMClient(model_name="llama3", url="http://localhost:11434")
response = client.generate("Hello, world!")

# Safe file operations
safe_save_json({"key": "value"}, "output.json")
```

### Extensibility Guidelines
- **Adding New Configuration Sections**: Extend `validate_config_schema()` in `config.py` and document defaults in `DEFAULT_CONFIG`.
- **Custom I/O Operations**: Extend `io_utils.py` with new file types while maintaining error handling patterns from `safe_save_json`/`safe_load_json`.
- **Additional LLM Backends**: Subclass `LLMClient` or create new client classes while maintaining the same `generate()` interface.
- **Custom Logging**: Extend `logging_utils.py` with new formatters or handlers (e.g., JSON logs, structured metrics).
- **Coding Style**: Use descriptive names, early returns, minimal try/except blocks, no broad exception swallowing, document non-obvious rationale only.

### Related Documentation
- [`src/evaluation/README.md`](../evaluation/README.md) for evaluation framework usage
- [`src/data_processing/README.md`](../data_processing/README.md) for data processing pipeline
- [`scripts/README.md`](../../scripts/README.md) for CLI usage
- [`configs/README.md`](../../configs/README.md) for configuration details
