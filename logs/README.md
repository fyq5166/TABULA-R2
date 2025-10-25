# logs Directory

## Overview
This directory contains structured log files from both data processing and Q&A evaluation runs. Logs provide comprehensive debugging information, performance metrics, and execution traces for analysis and troubleshooting.

## Log File Naming
- **Format**: `YYYYMMDD_HHMMSS.log`
- **Example**: `20251023_213843.log`
- **Timestamp**: Based on run start time

## Log Levels and Output

### Console Output (Terminal)
- **Level**: INFO and above (INFO, WARNING, ERROR)
- **Content**: User-facing progress information
- **Excludes**: DEBUG level details (full LLM responses, detailed execution traces)
- **Purpose**: Real-time monitoring without overwhelming output

### File Output (Log Files)
- **Level**: DEBUG and above (DEBUG, INFO, WARNING, ERROR)
- **Content**: Complete execution traces including full LLM responses
- **Purpose**: Comprehensive debugging and analysis

## Log Structure

### Header Section
```
Logging system configured - Level: DEBUG, File: logs/YYYYMMDD_HHMMSS.log
======================================================
      UNIFIED Q-A RUNNER & EVALUATOR STARTING  
======================================================
- started_at: 2025-10-23 21:38:43
- run_id: 20251023_213843(AS current style)
```

### Configuration Summary
```
CONFIG SUMMARY
- model: ollama
- validator: llama3
- run.time_limit_s: 300
- run.llm_timeout_s: 30
- run.max_turns: 10
- obs.mode: header_5
- selection.mode: multi_table
- selection.custom_ids: []
- selection.batch_size: 10
- selection.continue: False
```

### Question Processing
```
QUESTION PROGRESS
- progress: 1/75
- question_id: question_274
- question: What is the total number of deaths in 1950?
- table_refs: ['table_0001']
```

### LLM Interaction (DEBUG Level)
```
LLM call (round 1):
        prompt_length: 2048
        model: llama3
        temperature: 1.0
        max_tokens: 1024
        call_time_s: 2.345
        returned: 
I'll analyze the table to find the total number of deaths in 1950.

PLAN
load(table_0001)
filter_all(Year == 1950)
agg({sum: "Deaths"})
END PLAN
```

### Plan Execution
```
PLAN EXECUTION
        plan: load(table_0001) filter_all(Year == 1950) agg({sum: "Deaths"})
        execution_time_s: 0.123
        result: {'rows': 1, 'preview': [['746438']]}
        status: SUCCESS
```

### Validation Results
```
VALIDATION
- golden_answer: 746438
- llm_answer: 746438
- validator_model: llama3
- validator_decision: True
- validation_time_s: 0.522
- total_elapsed_s: 103.343
```

### Error Conditions
```
ERROR CONDITIONS
- error_code: EXEC_ERROR
- error_message: Column 'Deaths' not found in table
- status: ERROR
- retry_count: 1
```

## Log Content Categories

### 1. **System Initialization**
- Logging configuration
- Model availability checks
- Configuration loading
- Question selection

### 2. **Question Processing**
- Progress tracking
- Question metadata
- Table loading
- Observation space generation

### 3. **LLM Interactions** (DEBUG only in files)
- Complete prompt content
- Full LLM responses
- Token counts and timing
- Model parameters

### 4. **Plan Execution**
- DSL plan parsing
- Execution results
- Error handling
- Performance metrics

### 5. **Validation**
- Golden answer comparison
- LLM validator decisions
- Answer type checking
- Accuracy metrics

### 6. **Error Handling**
- Error codes and messages
- Retry attempts
- Timeout conditions
- Recovery actions

### 7. **Performance Metrics**
- Execution times
- Memory usage
- Token consumption
- Throughput statistics

## Data Processing Logs

For data processing runs, logs include:

### Pipeline Stages
```
[INFO] Starting data processing pipeline
[INFO] Scanning OWID datasets: 150 found
[INFO] Processing dataset: mortality_data
[DEBUG] Cleaning dataset: original_shape=(1000, 50), final_shape=(800, 45)
[INFO] Domain classification: health
[INFO] Metadata extraction complete
```

### Quality Control
```
[INFO] Quality control checks:
- Completeness: 0.85
- Numeric columns: 42/45
- Time series detected: true
- Complexity level: medium
```

### Error Handling
```
[WARNING] Dataset processing failed: Invalid column format
[ERROR] LLM classification timeout: Using heuristic fallback
[INFO] Recovery: Skipping problematic dataset
```

## Log Analysis Tools

### Quick Statistics
```bash
# Count total questions processed
grep -c "QUESTION PROGRESS" logs/YYYYMMDD_HHMMSS.log

# Count successful runs
grep -c "status: SUCCESS" logs/YYYYMMDD_HHMMSS.log

# Count error types
grep -o "error_code: [A-Z_]*" logs/YYYYMMDD_HHMMSS.log | sort | uniq -c
```

### Performance Analysis
```bash
# Extract execution times
grep "execution_time_s:" logs/YYYYMMDD_HHMMSS.log | awk '{print $2}' | sort -n

# Find timeout conditions
grep "TIME OUT" logs/YYYYMMDD_HHMMSS.log

# Analyze LLM response times
grep "call_time_s:" logs/YYYYMMDD_HHMMSS.log | awk '{print $2}' | sort -n
```

### Error Analysis
```bash
# Count validation failures
grep -c "validator_decision: False" logs/YYYYMMDD_HHMMSS.log

# Find stubborn behavior
grep -c "stubborn_behavior: true" logs/YYYYMMDD_HHMMSS.log

# Analyze error distribution
grep "error_code:" logs/YYYYMMDD_HHMMSS.log | cut -d: -f2 | sort | uniq -c
```

## Log Rotation and Management

### File Size Considerations
- Log files can grow large (50MB+ for full runs)
- DEBUG level includes complete LLM responses
- Consider compression for long-term storage

### Cleanup Recommendations
```bash
# Compress old logs
gzip logs/*.log

# Remove logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete
```

## Related Documentation

- [`experiments/results/README.md`](../experiments/results/README.md) — Output file formats
- [`configs/README.md`](../configs/README.md) — Configuration parameters
- [`src/utils/logging_utils.py`](../src/utils/logging_utils.py) — Logging implementation

## Troubleshooting

### Common Log Patterns
- **High timeout rates**: Check LLM server performance
- **Frequent EXEC_ERROR**: Review DSL plan generation
- **Validation failures**: Analyze answer format mismatches
- **Stubborn behavior**: Consider prompt engineering improvements

### Debug Information
- Full LLM responses available in log files (DEBUG level)
- Complete execution traces for plan debugging
- Detailed error messages with context
- Performance metrics for optimization
