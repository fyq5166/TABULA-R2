# src/prompts Directory

## Overview
This directory contains the prompt templates and few-shot examples used by the Q&A evaluation framework. It provides structured, reusable prompt components for LLM interactions, including system messages, user messages, few-shot examples, and DSL specifications. The templates use Jinja2 templating for dynamic content generation.

## Directory Layout
```
src/prompts/
├── prompt_manager.py          # Main prompt management and template rendering
├── templates/
│   ├── common/                # Shared template components
│   │   ├── dsl_action_space.md    # DSL command reference and examples
│   │   ├── dsl_spec.md            # DSL syntax specification
│   │   └── output_format.md       # Output format guidelines
│   ├── qna_execution/         # Q&A evaluation prompts
│   │   ├── system_message.md      # System-level instructions
│   │   ├── user_message.md        # User-level question context
│   │   ├── history_section.md     # Previous steps display
│   │   ├── general_prompt.md       # General prompt template
│   │   └── few_shot/              # Few-shot examples
│   │       ├── index.json         # Example metadata and indexing
│   │       └── bank/              # Example categories
│   │           ├── aggregation/   # Aggregation examples (ex_0001-0004)
│   │           ├── comparison/     # Comparison examples (ex_0004-0006)
│   │           ├── filtering_selection/ # Filtering examples (ex_0010-0012)
│   │           ├── relational_join/     # Join examples (ex_0007-0009)
│   │           └── unanswerable/       # Unanswerable examples (ex_0013-0015)
│   └── topic_grouping/         # Topic grouping prompts
│       ├── system_message.md      # System instructions for topic grouping
│       └── user_message_template.md # User message template
```

## Template System

### Jinja2 Templating
All templates use Jinja2 templating for dynamic content generation:
- `{% if condition %}` - Conditional content inclusion
- `{{ variable }}` - Variable substitution
- `{% include 'path' %}` - Template inclusion

### Common Template Variables
- `question` - The question text
- `table_refs` - List of table identifiers
- `visible_columns_desc` - Column descriptions
- `visible_rows_desc` - Sample data rows
- `table_stats_desc` - Table statistics
- `table_structures` - Detailed table structure
- `history` - Previous conversation steps
- `include_cot` - Whether to include chain-of-thought reasoning

## Q&A Execution Templates

### System Message (`system_message.md`)
Core instructions for the LLM, including:
- Task description and rules
- DSL specification and syntax rules
- Output format requirements
- Critical formatting rules (no prefixes, markdown, etc.)

### User Message (`user_message.md`)
Question context and data presentation:
- Question text and table references
- Table structure information
- Sample data rows
- Previous conversation history
- Specific instructions for different question types

### Few-Shot Examples
Structured examples organized by reasoning type:

| Category | Examples | Reasoning Type | Difficulty |
|----------|----------|----------------|------------|
| `aggregation/` | ex_0001-0004 | Aggregation operations | 1-2 |
| `comparison/` | ex_0004-0006 | Entity/entity comparison | 1-2 |
| `filtering_selection/` | ex_0010-0012 | Data filtering and selection | 1 |
| `relational_join/` | ex_0007-0009 | Multi-table joins | 1-2 |
| `unanswerable/` | ex_0013-0015 | Unanswerable questions | 1 |

### Example Structure
Each example follows this format:
```markdown
### Example: ex_agg_0001
Question:
[Question text]

{% if include_cot %}
Reasoning:
[Step-by-step reasoning]

{% endif %}
---- Output (emit exactly one block) ----
PLAN
[DSL commands]
END PLAN
```

## DSL Components

### DSL Action Space (`common/dsl_action_space.md`)
Complete reference for DSL commands:
- `load(table_id)` - Load table
- `select(col1, col2, ...)` - Select columns
- `filter(expr)` - Filter rows
- `groupby().agg({fn: col})` - Aggregation
- `join(left, right, on=cols)` - Table joins
- `derive(new_col = expr)` - Calculated columns

### DSL Specification (`common/dsl_spec.md`)
Detailed syntax rules and examples for each command.

### Output Format (`common/output_format.md`)
Guidelines for LLM output formatting and block structure.

## Topic Grouping Templates

### System Message (`topic_grouping/system_message.md`)
Instructions for grouping related topics in metadata extraction.

### User Message Template (`topic_grouping/user_message_template.md`)
Template for presenting data to the topic grouping LLM.

## Usage Examples

### Basic Template Rendering
```python
from src.prompts import prompt_manager

# Load and render a template
template = prompt_manager.load_template("qna_execution/system_message")
rendered = template.render(
    question="What is the total GDP?",
    table_refs=["table_001"],
    include_cot=True
)
```

### Few-Shot Example Selection
```python
# Get examples by reasoning type
examples = prompt_manager.get_few_shot_examples(
    reasoning_type="aggregation",
    max_examples=3,
    include_cot=True
)
```

### Template Variables
```python
# Common variables for Q&A execution
context = {
    "question": "What is the population of USA?",
    "table_refs": ["table_001"],
    "visible_columns_desc": "Available columns: Country, Year, Population",
    "visible_rows_desc": "Sample data:\nCountry | Year | Population\nUSA | 2020 | 331M",
    "table_stats_desc": "rows: 1000 | cols: 3",
    "history": [],
    "include_cot": False
}
```

## Template Categories

### Q&A Execution Templates
- **Purpose**: Generate prompts for table reasoning tasks
- **Key Features**: DSL integration, few-shot examples, history tracking
- **Usage**: Main evaluation pipeline

### Topic Grouping Templates  
- **Purpose**: Group related topics in metadata extraction
- **Key Features**: Topic clustering, semantic grouping
- **Usage**: Data processing pipeline

### Common Components
- **Purpose**: Shared template components and DSL references
- **Key Features**: Reusable components, DSL documentation
- **Usage**: Included in other templates

## File Naming Conventions
- Template files: `*.md` (Markdown format)
- Example files: `ex_XXXX.md` (zero-padded IDs)
- Index files: `index.json` (metadata and organization)
- Python modules: `*.py` (template management)

## Related Documentation
- [`src/evaluation/README.md`](../evaluation/README.md) for evaluation framework usage
- [`scripts/README.md`](../../scripts/README.md) for CLI usage
- [`configs/README.md`](../../configs/README.md) for configuration options
