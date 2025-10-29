# TABULA-R²
**TABular Universal Language Analytics for Reasoning and Research**

A comprehensive benchmark for evaluating large language models on tabular reasoning tasks, with support for local model deployment and multi-table reasoning scenarios.

## Overview

TABULA-R² provides an end-to-end pipeline for generating, validating, and evaluating tabular reasoning questions across single- and multi-table settings. We have published a comprehensive benchmark dataset with rich metadata including reasoning steps for chain-of-thought analysis, complexity scores, and domain classifications. The framework enables reproducible benchmarking for locally deployed LLMs with comprehensive analysis and visualization tools.

## Key Features

- 🎯 **Multi-table reasoning benchmark** with domain-balanced question design
- 🤖 **Local LLM support** (Ollama, OpenAI-compatible APIs) 
- 📊 **Comprehensive analysis tools** with statistical visualization
- ⚙️ **Configurable experiments** with batching and continue functionality
- 🔧 **Modular architecture** for easy extension and customization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fyq5166/TABULA-R2.git
cd TABULA-R2

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run Q&A experiments
python scripts/qna_runner.py --config configs/qna.yaml

# Analyze results
python scripts/data_analysis/analyze_results.py --config configs/analysis.yaml

# Generate table statistics
python scripts/data_analysis/table_statistics.py

# Generate question statistics  
python scripts/data_analysis/question_statistics.py
```

## Project Structure

```
TABULA-R²/
├── src/                    # Core modules
│   ├── evaluation/         # Evaluation framework
│   ├── data_processing/    # Data processing pipeline
│   ├── prompts/           # Prompt templates and examples
│   └── utils/             # Utility functions
├── scripts/               # Execution scripts
│   ├── qna_runner.py      # Main Q&A experiment runner
│   ├── data_processing.py # Data processing pipeline
│   └── data_analysis/     # Analysis and visualization tools
├── configs/               # Configuration files
│   ├── qna.yaml           # Q&A experiment configuration
│   ├── analysis.yaml      # Analysis configuration
│   ├── comparison_experiments/ # Model comparison configs
│   └── comparison_analysis/   # Analysis configs for comparisons
├── data/                  # Data and questions
│   ├── questions/         # Question repository
│   └── tables/           # Table data and metadata
└── experiments/          # Experiment results and analysis
    └── results/          # Experiment outputs
```

## Core Components

### Data Processing Pipeline
- **OWID Dataset Processing**: Automated ingestion and cleaning of Our World in Data datasets
- **Table Cleaning**: Standardized data cleaning with quality control
- **Metadata Extraction**: Automated domain classification and complexity assessment

### Evaluation Framework
- **DSL Executor**: Domain-specific language for table operations
- **Plan Parser**: Robust parsing of LLM-generated execution plans
- **LLM Validator**: Automated answer validation and scoring
- **Guidance System**: Context-aware error handling and suggestions

### Analysis Tools
- **Statistical Analysis**: Comprehensive performance metrics and error analysis
- **Visualization**: Interactive charts and plots for result interpretation
- **Comparison Framework**: Controlled experiments across models and configurations

## Configuration

The project uses YAML configuration files for flexible experiment design:

- **`configs/qna.yaml`**: Main experiment configuration
- **`configs/analysis.yaml`**: Analysis and visualization settings
- **`configs/comparison_experiments/`**: Model comparison configurations
- **`configs/comparison_analysis/`**: Analysis configurations for comparisons

## Supported Models

- **Local Models**: Ollama (Llama, Qwen, CodeLlama, Gemma)
- **Cloud APIs**: OpenAI GPT-4, Claude, Gemini
- **Custom Models**: Any OpenAI-compatible API endpoint

## Documentation

- [Configuration Guide](configs/README.md) - Detailed configuration options
- [Scripts Documentation](scripts/README.md) - Script usage and examples
- [Evaluation Framework](src/evaluation/README.md) - Core evaluation components
- [Data Processing](src/data_processing/README.md) - Data processing pipeline
- [Utils Documentation](src/utils/README.md) - Utility functions and helpers
- [Data Questions](data/questions/README.md) - Question repository structure and metadata
- [Data Tables](data/tables/) - Table data structure and quality metrics
- [Experiment Results](experiments/results/) - Result file formats and analysis

## Results and Benchmarks

### Published Benchmark

We have published a comprehensive benchmark dataset with three complementary subsets:

- **Single-table questions** (`data/questions/single_table/`): Questions requiring reasoning over individual tables with arithmetic aggregation, conditional reasoning, and filtering operations
- **Multi-table questions** (`data/questions/multi_table/`): Questions requiring cross-table joins and entity alignment across thematic cohorts
- **Distractor-bank questions** (`data/questions/distractor_bank/`): Robustness variants with irrelevant or misleading tables to test model resilience

The benchmark includes rich metadata for each question:
- **Reasoning steps** for chain-of-thought analysis
- **Complexity scores** and domain classifications
- **Answer types** (numerical, categorical, boolean, text)
- **Table references** and relationship mappings

### Data Source and Quality

Our benchmark is built on [Our World in Data (OWID)](https://ourworldindata.org/), a comprehensive database of global development indicators. This ensures:

- **Real-world relevance**: Questions based on actual global development data
- **Large-scale tables**: High-quality datasets with substantial row counts
- **Diverse domains**: Economics, health, education, environment, and technology
- **Temporal coverage**: Historical data spanning multiple decades

### Local Model Deployment

We provide comprehensive evaluation results using locally deployed models via [Ollama](https://ollama.ai/):

- **[Llama 3](https://ollama.ai/library/llama3)** - Meta's flagship model for general reasoning
- **[Llama 3.2](https://ollama.ai/library/llama3.2)** - Latest iteration with improved performance
- **[CodeLlama](https://ollama.ai/library/codellama)** - Specialized for code and structured reasoning
- **[Qwen](https://ollama.ai/library/qwen)** - Alibaba's multilingual model
- **[Gemma](https://ollama.ai/library/gemma)** - Google's efficient open model

### Evaluation Metrics

The framework provides comprehensive analysis across multiple dimensions:

- **Success Rate Analysis**: Overall accuracy and result distribution across question types
- **DSL Execution Analysis**: Step-by-step execution success rates and error pattern analysis
- **Error Pattern Analysis**: Detailed breakdown of failure modes including NULL answers, execution errors, and syntax issues
- **Metadata-based Analysis**: Performance correlation with reasoning type, domain, complexity, and table number
- **Consistency Analysis**: Multi-run consistency evaluation for reliability assessment
- **LLM Round Analysis**: Feedback round distribution and interaction pattern analysis


## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues and pull requests.

## Citation

If you use TABULA-R² in your research, please cite:

```bibtex
@misc{tabula-r2-2025,
  title={TABULA-R²: TABular Universal Language Analytics for Reasoning and Research},
  author={Yeqiao Fu},
  year={2025},
  url={https://github.com/fyq5166/TABULA-R2}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted under the guidance of Prof. Philipp Koehn at Johns Hopkins University. 

---

**TABULA-R²** - Advancing tabular reasoning through comprehensive benchmarking and analysis.