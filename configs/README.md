# configs

Central catalog of YAML configuration files that drive pipeline behavior.

## Files

- `data_processing.yaml`
  - Purpose: Control OWID ingestion, cleaning, metadata extraction, storage, and pipeline mode.
  - Highlights:
    - `owid`: raw dataset source path and size gate.
    - `cleaning`: row cap, numeric column minimum, missing-value policy, deduplication, column-name normalization.
    - `quality_control`: minimal completeness (legacy), minimal rows/columns, entity cardinality cap.
    - `metadata`: LLM tag, retries, fail-on-LLM-unavailable, domain category list.
    - `storage`: output bundle root directory.
    - `pipeline`: which steps to run; index filename; optional `limit`, `topic_output_file`, `output_dir`.
  - Logging:
    - Logs mirror to both file and console. Log file names follow `YYYYMMDD_HHMMSS.log`.

- `qna.yaml`
  - Purpose: Drive Q&A evaluation runs (chat-first, multi-round, DSL-enabled, repeats supported, batching and continue functionality).
  - Key sections and fields:
    - General
      - `qna.questions_path`: root of curated questions.
      - `qna.model`: model name entry used by runner (when `llm.provider=ollama`, ensure it is installed locally via `ollama pull`).
      - `qna.limit`: optional cap on number of questions per run.
      - `qna.seed`: random seed for reproducibility.
    - LLM
      - `qna.llm.provider`: `ollama` | `vllm` | `together` (non-ollama providers are unchanged in behavior).
      - `qna.llm.url`: base URL for the LLM server (e.g., `http://localhost:11434` for Ollama).
      - `qna.llm.temperature`, `qna.llm.top_p`, `qna.llm.max_tokens`: decoding options.
      - Note: per-call timeout is configured under `qna.run.llm_timeout_s`.
    - Selection
      - `qna.selection.mode`: `all` | `single_table` | `multi_table` | `distractor` | `custom`.
      - `qna.selection.custom_ids`: when `mode=custom`, accepts either explicit IDs or ID ranges, e.g., `["1-3", "4"]`.
      - `qna.selection.batch_size`: number of questions per batch (enables batching when > 0).
      - `qna.selection.continue`: enable continue mode from a specific question.
      - `qna.selection.continue_from`: question number to continue from (1-based, e.g., 217).
      - `qna.selection.continue_batches`: whether to continue to next batch after current batch completes.
    - Observation Space
      - `qna.obs.mode`: `full_table` | `header_5` | `header_1` (all columns are visible; rows follow the selected mode).
      - `qna.obs.max_rows_per_table`: maximum rows shown in `full_table` mode (e.g., `500`).
    - Prompt
      - `qna.prompt.few_shot_k`: `0 | 1 | 3`.
      - `qna.prompt.include_cot`: include CoT reasoning in examples.
      - `qna.prompt.dump_prompt`: debug flag to dump rendered messages to per-run files.
    - Run
      - `qna.run.time_limit_s`: per-question wall-clock budget (overrun → `TIME OUT【run】`).
      - `qna.run.llm_timeout_s`: per-call timeout (overrun → `TIME OUT【llmcall】`).
      - `qna.run.max_turns`: multi-round cap.
      - `qna.run.history_limit`: `None` | `"all"` | integer.
      - `qna.run.repeats`: number of repeated runs per question (results aggregated into a single JSONL per question).
      - `qna.run.repeated_plan_threshold`: if same PLAN executed N times, use result directly (default: 5).
      - `qna.run.stubborn_threshold`: if same failed PLAN repeated N times, force different approach (default: 3).
      - `qna.run.stubborn_termination_threshold`: if same failed PLAN repeated N times, force termination (default: 5).
    - Validator
      - `qna.validator.model`, `qna.validator.timeout_s`, `qna.validator.url`: LLM-based validator settings (results logged and written to JSONL).
    - Result & Logging
      - `qna.result.result_dir`: per-run timestamped directory; one JSONL per question containing all `runs` (repeats).
      - `qna.result.log_dir`: logs folder; log file name is `YYYYMMDD_HHMMSS.log` and mirrored to the console.

## Batching and Continue Functionality

The Q&A runner supports advanced batching and continue functionality for large-scale experiments:

### Batching
- **Batch Size**: Set `qna.selection.batch_size` to enable batching (e.g., `10` for 10 questions per batch).
- **Batch Info**: Each run generates a `batch_info_<run_id>.txt` file documenting batch structure and question order.
- **Continue Batches**: Set `qna.selection.continue_batches=true` to run all batches sequentially.

### Continue Mode
- **Continue from Question**: Set `qna.selection.continue=true` and `qna.selection.continue_from=<question_number>` to resume from a specific question.
- **Batch Info Reading**: The system reads the most recent `batch_info_*.txt` file to determine the shuffled question order.
- **Progress Tracking**: Displays accurate progress based on the original question order and continue position.

### Example Usage
```bash
# Run first batch only
uv run python scripts/qna_runner.py --config configs/qna.yaml

# Continue from question 217
uv run python scripts/qna_runner.py --continue --continue-from 217 --config configs/qna.yaml

# Continue all batches
uv run python scripts/qna_runner.py --continue-batches --config configs/qna.yaml
```

## Advanced Run Controls

The Q&A runner includes sophisticated controls for handling LLM behavior and plan execution:

### Repeated Plan Detection
- **repeated_plan_threshold**: When the same PLAN is executed multiple times successfully, use the result directly instead of re-executing.
- **Use Case**: Prevents redundant execution of identical plans that have already been validated.

### Stubborn Behavior Handling
- **stubborn_threshold**: When a failed PLAN is repeated multiple times, force the LLM to try a different approach.
- **stubborn_termination_threshold**: When a failed PLAN is repeated too many times, terminate the question with an error.
- **Use Case**: Prevents infinite loops when the LLM gets stuck on incorrect plans.

### Error Recovery
- **LLM Timeout Handling**: Automatic retry with exponential backoff for LLM server timeouts.
- **Plan Validation**: Strict validation of DSL plans before execution to catch syntax errors early.
- **Execution Sandbox**: Safe execution environment with timeout and memory limits.

## Model installation (Ollama)

When `qna.llm.provider=ollama`, ensure the model is installed locally. Example:

```bash
ollama pull llama3:latest
```

For convenience, you can batch pull representative models used in the experiments plan.

## Analysis Configuration

### `analysis.yaml`
- **Purpose**: Comprehensive analysis configuration for Q&A experiment results with statistical analysis and visualization
- **Usage**: `uv run python scripts/data_analysis/analyze_results.py --config configs/analysis.yaml`
- **Key sections:**
  - **Input Configuration**:
    - `input.result_dirs`: List of experiment result directories to analyze
    - `input.question_folder`: Path to question metadata for additional analysis
  - **Output Configuration**:
    - `output.output_dir`: Directory for analysis results and visualizations
    - `output.analysis_name`: Name for the analysis run
  - **Analysis Switches**:
    - `statistics.basic_stats`: Enable basic success rate and error statistics
    - `statistics.error_analysis`: Enable detailed error type analysis
    - `statistics.llm_rounds`: Enable LLM output round analysis
    - `statistics.metadata_analysis`: Enable analysis by reasoning type, domain, complexity
    - `statistics.consistency_analysis`: Enable consistency analysis across multiple runs
  - **Visualization Switches**:
    - `visualizations.results_distribution`: Generate result distribution charts
    - `visualizations.success_by_metadata`: Generate success rate charts by metadata
    - `visualizations.error_analysis`: Generate error analysis charts
    - `visualizations.llm_rounds`: Generate LLM round statistics charts
    - `visualizations.metadata_charts`: Generate metadata-based visualizations
- **Output**: JSON analysis reports, PNG visualization charts, statistical summaries

## Comparison Experiments

### Comparison Experiment Design
The comparison experiments are designed to systematically evaluate different aspects of the Q&A system:

#### **Model Comparison**
- **Baseline Models**: `codellama:7b-instruct`, `llama3.2:latest`, `gemma:7b`, `qwen:7b`
- **Purpose**: Compare performance across different LLM architectures and training approaches
- **Configuration**: Each model uses identical parameters except for the model name

#### **Few-shot Learning Analysis**
- **Zero-shot**: `llama3_fewshot0.yaml` - No few-shot examples provided
- **One-shot**: `llama3_fewshot1.yaml` - Single few-shot example provided
- **Purpose**: Evaluate the impact of few-shot learning on performance

#### **Chain-of-Thought (CoT) Analysis**
- **With CoT**: `llama3_cot.yaml` - Include chain-of-thought reasoning in examples
- **Purpose**: Assess the impact of explicit reasoning steps on model performance

#### **Observation Mode Analysis**
- **Header-1**: `llama3_header1.yaml` - Show only column headers and first row
- **Full Table**: `llama3_fulltable.yaml` - Show complete table data
- **Purpose**: Evaluate the impact of data visibility on reasoning ability

### Configuration Files
All comparison experiment configurations are located in `configs/comparison_experiments/`:
- `codellama_baseline.yaml`, `llama32_baseline.yaml`, `gemma_baseline.yaml`, `qwen_baseline.yaml`
- `llama3_fewshot0.yaml`, `llama3_fewshot1.yaml`
- `llama3_cot.yaml`
- `llama3_header1.yaml`, `llama3_fulltable.yaml`

### Question Selection
- **Sample Size**: 50 questions (25 single-table, 25 multi-table)
- **Selection Criteria**: Balanced by reasoning type, domain, and complexity
- **Reasoning Types**: arithmetic_aggregation, conditional_reasoning, entity_alignment, proxy_inference
- **Domains**: health, environment, economics, demographics, education
- **Complexity**: Low (≤6.0), Medium (6.1-21.5), High (>21.5)

## Comparison Analysis

### Analysis Configuration Files
Located in `configs/comparison_analysis/`, these configurations analyze the results from comparison experiments:

#### **Model Comparison Analysis**
- `codellama_baseline_analysis.yaml`, `llama32_baseline_analysis.yaml`
- `gemma_baseline_analysis.yaml`, `qwen_baseline_analysis.yaml`
- **Purpose**: Analyze performance differences between baseline models

#### **Parameter Variation Analysis**
- `llama3_fewshot0_analysis.yaml`, `llama3_fewshot1_analysis.yaml`
- `llama3_cot_analysis.yaml`
- `llama3_header1_analysis.yaml`, `llama3_fulltable_analysis.yaml`
- **Purpose**: Analyze the impact of different parameters on performance

#### **Sample Baseline Analysis**
- `llama_baseline_analysis.yaml`
- **Purpose**: Analyze baseline performance on the selected sample questions

### Analysis Features
- **Success Rate Comparison**: Compare success rates across different models and parameters
- **Error Pattern Analysis**: Identify common failure modes and error types
- **Performance Metrics**: Execution time, turn count, and efficiency analysis
- **Statistical Significance**: Determine if observed differences are statistically significant
- **Visualization**: Generate comparative charts and heatmaps


## Conventions

- Paths are relative to the repo root unless noted.
- Use absolute paths on CLI when convenient.
- Keep YAML comments concise but specific; explain intent and typical ranges.
- Analysis configurations include detailed calculation standards and methodology.
- Comparison experiments follow controlled variable principles for scientific rigor.

