# scripts Module

## Module Scope
- **Purpose:** Entry-point scripts for executing the OWID data pipeline and Q&A evaluation workflows.

## Scripts & Workflows

### [`data_processing.py`](data_processing.py)
- **Purpose:** Config-driven OWID pipeline runner (process tables, create domain index, optional topic grouping).
- **Usage:** `uv run python scripts/data_processing.py [--limit N] [--output-dir PATH] [--topic-output JSON]`
- **Workflow:**
  1. Load [`configs/data_processing.yaml`](../configs/data_processing.yaml) and interpret `pipeline.mode`.
  2. If mode includes tables (`tables_only`, `tables_with_index`):
     - Discover unprocessed OWID datasets under `owid.root_path`.
     - For each dataset (respecting `--limit`): validate structure, load CSV, clean, classify with LLM fallback, extract metadata, persist `table_XXXX/` bundle.
     - Emit progress logs for each table.
  3. If topic grouping output is requested (`--topic-output` and tables were processed):
     - Group newly processed tables by domain/topic via TableGrouper.
     - Write a JSON file containing `topic_group_name` plus tables summarized with `table_id`, `title`, and `description`.
  4. If mode includes indexing (`tables_with_index`, `index_only`):
     - Rebuild domain-organized index (e.g., `my_test_index.json`) via `create_domain_index`.
  5. If mode is `topics_only`:
     - Load all existing table metadata and group titles per domain via TableGrouper.
     - Emit the consolidated topic grouping JSON (table summaries include `table_id`, `title`, `description`) to the path supplied via `--topic-output`.
- **Notes:**
  - Default values for `--limit`, `--topic-output`, and `--output-dir` can be pre-set via `pipeline.limit`, `pipeline.topic_output_file`, and `pipeline.output_dir` in [`../configs/data_processing.yaml`](../configs/data_processing.yaml); CLI flags override these.
  - `topics_only` mode requires `--topic-output` (or the config equivalent) and an available Ollama instance; otherwise the script aborts.

### [`qna_runner.py`](qna_runner.py)
- **Purpose:** Advanced Q&A evaluation runner with multi-round LLM interactions, DSL execution, batching, and continue functionality.
- **Usage:** `uv run python scripts/qna_runner.py [OPTIONS] --config configs/qna.yaml`
- **Key Features:**
  - **Multi-round LLM interactions** with conversation history
  - **DSL plan execution** with validation and error handling
  - **Batching support** for large-scale experiments
  - **Continue functionality** for resuming interrupted runs
  - **Stubborn behavior detection** and termination
  - **LLM validation** of answers
  - **Comprehensive logging** and debugging support

#### **Command Line Options:**
```bash
# Basic usage
uv run python scripts/qna_runner.py --config configs/qna.yaml

# Continue from specific question
uv run python scripts/qna_runner.py --continue --continue-from 217 --config configs/qna.yaml

# Continue all batches
uv run python scripts/qna_runner.py --continue-batches --config configs/qna.yaml

# Override configuration
uv run python scripts/qna_runner.py --config configs/qna.yaml --continue --continue-from 100
```

#### **Core Workflow:**
1. **Configuration Loading**: Load Q&A configuration from YAML file
2. **Question Selection**: Select questions based on mode (single_table, multi_table, distractor, custom)
3. **Batching Logic**: 
   - If `batch_size > 0`: Split questions into batches
   - If `continue=true`: Read previous batch_info and resume from specific question
   - If `continue_batches=true`: Run all batches sequentially
4. **Multi-round Processing**: For each question:
   - Generate observation space (table structure and sample data)
   - Build prompts with few-shot examples
   - Execute multi-round LLM conversation
   - Parse PLAN/END blocks from LLM responses
   - Execute DSL plans with validation
   - Handle stubborn behavior and repeated plans
   - Validate answers with LLM validator
5. **Output Generation**: Save results to JSONL files and generate batch_info

#### **Advanced Features:**

##### **Batching and Continue:**
- **Batch Size**: Set `qna.selection.batch_size` to enable batching
- **Continue Mode**: Resume from specific question using `--continue --continue-from <question_id>`
- **Batch Info**: Automatically generates `batch_info_<run_id>.txt` for continue functionality
- **Progress Tracking**: Accurate progress display based on original question order

##### **Stubborn Behavior Handling:**
- **Repeated Plan Detection**: Automatically use results from repeated successful plans
- **Stubborn Threshold**: Force different approach when failed plans are repeated
- **Termination**: Stop processing when stubborn behavior exceeds threshold
- **Error Recovery**: Comprehensive error handling with specific error codes

##### **Multi-round LLM Interactions:**
- **Conversation History**: Maintain context across multiple turns
- **Plan Execution**: Execute DSL plans and provide results to LLM
- **END Block Detection**: Recognize when LLM provides final answer
- **Timeout Handling**: Per-call and per-question time limits

##### **Validation and Quality Control:**
- **LLM Validator**: Use separate LLM to validate answers
- **Answer Type Checking**: Ensure answer format matches expected type
- **Error Code Classification**: Detailed error tracking and analysis
- **Performance Metrics**: Execution time, turn count, success rate tracking

#### **Output Files:**
- **Question Results**: `experiments/results/YYYYMMDD_HHMMSS/question_XXX.jsonl`
- **Batch Info**: `experiments/results/YYYYMMDD_HHMMSS/batch_info_<run_id>.txt`
- **Prompt Dumps**: `experiments/results/YYYYMMDD_HHMMSS/prompt_dumps/` (when `dump_prompt=true`)
- **Logs**: `logs/YYYYMMDD_HHMMSS.log`

#### **Configuration Parameters:**
- **Selection**: Mode, batch size, continue settings
- **LLM**: Provider, URL, temperature, timeout settings
- **Run**: Time limits, turn limits, repeat counts
- **Validation**: Validator model and timeout settings
- **Prompt**: Few-shot examples, CoT inclusion, debug options

#### **Error Handling:**
- **LLM Timeouts**: Automatic retry with exponential backoff
- **Plan Validation**: Strict DSL syntax checking before execution
- **Execution Errors**: Safe execution environment with limits
- **Stubborn Behavior**: Detection and termination of repeated failures
- **Continue Errors**: Graceful handling of missing batch_info files

#### **Performance Optimization:**
- **Parallel Processing**: Efficient question processing
- **Memory Management**: Streaming for large datasets
- **Timeout Controls**: Per-question and per-call limits
- **Progress Tracking**: Real-time progress updates
- **Logging Levels**: Separate console and file logging

#### **Related Documentation:**
- [`configs/README.md`](../configs/README.md) — Configuration parameters
- [`experiments/results/README.md`](../experiments/results/README.md) — Output file formats
- [`logs/README.md`](../logs/README.md) — Log file formats and analysis
- [`src/evaluation/README.md`](../src/evaluation/README.md) — Core evaluation components

### [`data_analysis/analyze_results.py`](data_analysis/analyze_results.py)
- **Purpose:** Comprehensive analysis of Q&A experiment results with statistical analysis and visualization
- **Usage:** `uv run python scripts/data_analysis/analyze_results.py --config configs/analysis.yaml`
- **Features:**
  - Success rate analysis by various dimensions (reasoning type, domain, complexity)
  - Error type distribution and detailed analysis
  - LLM output round statistics and patterns
  - Metadata-based analysis and cross-relationships
  - Consistency analysis across multiple experiment runs
  - Comprehensive visualization generation (charts, plots, heatmaps)
- **Output:** Analysis reports (JSON), statistical charts (PNG), and detailed summaries

### [`data_analysis/table_statistics.py`](data_analysis/table_statistics.py)
- **Purpose:** Analyze and visualize table data characteristics and structure
- **Usage:** `uv run python scripts/data_analysis/table_statistics.py`
- **Features:**
  - Table structure analysis (column/row count distributions)
  - Domain distribution analysis across all tables
  - Complexity level analysis with proper ordering (low, medium, high)
  - Data quality metrics and cleaning statistics
  - Numeric ratio analysis by domain
- **Output:** Table structure charts, domain distributions, statistical reports (JSON)

### [`data_analysis/question_statistics.py`](data_analysis/question_statistics.py)
- **Purpose:** Analyze and visualize question data characteristics and patterns
- **Usage:** `uv run python scripts/data_analysis/question_statistics.py`
- **Features:**
  - Question type distribution (single vs multi-table questions)
  - Reasoning type analysis and distribution
  - Domain distribution across question types
  - Complexity analysis with standardized categorization
  - Answer type distribution and patterns
  - Cross-relationship analysis between question characteristics
- **Output:** Question type charts, complexity analysis, statistical reports (JSON)

#### **Data Analysis Workflow:**
1. **Experiment Results Analysis**: Use `analyze_results.py` to analyze Q&A experiment outcomes
2. **Table Characteristics**: Use `table_statistics.py` to understand dataset structure
3. **Question Patterns**: Use `question_statistics.py` to analyze question complexity and distribution
4. **Comparative Analysis**: Combine results for comprehensive dataset understanding

#### **Data Analysis Configuration:**
- Create analysis configs in `configs/analysis.yaml` for `analyze_results.py`
- Table and question statistics scripts run with default parameters
- All scripts generate outputs in `experiments/analysis/` subdirectories
- Statistical reports include calculation standards and methodology
