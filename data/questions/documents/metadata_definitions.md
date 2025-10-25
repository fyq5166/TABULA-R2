# Question Generation Metadata Definitions

## Field Specifications

### Required Fields

| Field Name | Type | Description | Constraints |
|------------|------|-------------|-------------|
| `question_id` | String | Unique identifier for the question | Format: "question_XXX" where XXX is 3-digit number |
| `reasoning_type` | String | Type of reasoning required | See Reasoning Types below |
| `domain` | String | Academic domain of the question | Auto-extracted from tables_index.json |
| `table_number` | Integer | Number of tables involved in the question | Positive integer, typically 1 for single-table tasks |
| `table_refs` | Array[String] | List of table IDs referenced | Format: ["table_XXXX"] where XXXX is 4-digit number |
| `question` | String | The main question text | Clear, specific question in English |
| `answer` | Mixed | Expected correct answer | Type depends on answer_type, keep concise |
| `reasoning_steps` | Array[String] | Step-by-step reasoning process | Ordered list of logical steps |
| `complexity_metrics` | Object | Quantitative complexity measures | See Complexity Metrics below |
| `answer_type` | String | Type of expected answer | See Answer Types below |
| `answerable` | Boolean | Whether the question can be answered from the data | true/false - false indicates insufficient data |
| `requires_calculation` | Boolean | Whether numerical calculation is needed | true/false |
| `has_distractor` | Boolean | Whether the question contains distractor tables | true/false |
| `distractor_type` | String | Type of distractor tables present | See Distractor Types below |

## Enumerated Field Options

### Reasoning Types
- **`arithmetic_aggregation`**: Questions requiring mathematical operations (sum, average, count, etc.)
- **`conditional_reasoning`**: Questions with conditional logic (if-then, filtering, comparisons)
- **`entity_alignment`**: Questions requiring matching or comparing entities across datasets
- **`proxy_inference`**: Questions requiring inference of unmeasured concepts from measured data to derive a single, specific answer (e.g., determining "best performing" entity based on multiple indicators)

### Domain Options
Domain is automatically extracted from `data/tables/tables_index.json`:
- **`health`**: Medical, public health, mortality, disease data
- **`economics`**: Economic indicators, inequality, trade, finance
- **`environment`**: Climate, pollution, emissions, environmental data
- **`demographics`**: Population, birth/death rates, migration, social statistics
- **`education`**: Educational attainment, literacy, learning outcomes
- **`technology`**: Technological adoption, innovation metrics
- **`general`**: Cross-domain or administrative data

### Answer Types
- **`numerical`**: Numeric answer (integer or decimal) - Most reliable for evaluation
- **`categorical`**: Single category/label from predefined set - Reliable for evaluation
- **`boolean`**: True/false answer - Most reliable for evaluation
- **`list`**: Multiple items (countries, years, categories) - Moderately reliable (order matters)
- **`text`**: Free-form text response - **Use sparingly, keep very concise (1-2 words max)**
- **`NA`**: Question cannot be answered due to insufficient data - Used when answerable=false

### Distractor Types
- **`none`**: No distractor tables (original questions)
- **`irrelevant`**: Completely unrelated tables (e.g., weather data for health questions)
- **`relevant`**: Related but unnecessary tables (e.g., additional health metrics not needed for the specific question)
- **`misleading`**: Tables that appear relevant but lead to incorrect conclusions

## Complexity Metrics Object

| Metric | Type | Description | Calculation |
|--------|------|-------------|-------------|
| `rows_involved` | Integer | Number of table rows examined | Count of rows that must be read |
| `columns_involved` | Integer | Number of table columns examined | Count of columns that must be accessed |
| `steps_count` | Integer | Number of reasoning steps | Count of items in reasoning_steps array |
| `complexity_score` | Float | Weighted complexity score | Formula: `(rows_involved × 0.1) + (columns_involved × 0.2) + (steps_count × 0.5)` |

### Complexity Score Rationale
- **Steps (0.5 weight)**: Highest cognitive load, most important for difficulty
- **Columns (0.2 weight)**: Data navigation and understanding burden  
- **Rows (0.1 weight)**: Data volume impact, least cognitively demanding

## Proxy Inference Guidelines

### Definition
Proxy inference questions require deriving a specific, measurable conclusion about an unmeasured concept using available data indicators. The answer must be:
1. **Single and specific**: One country, one year, one value, or yes/no
2. **Inferrable**: Not directly stated in the data but derivable through analysis
3. **Objective**: Based on clear criteria that can be applied to the data
4. **Concise**: Categorical, numerical, or boolean answers preferred

### Examples of Good Proxy Inference
- "Which country had the best economic performance in 2010?" (using GDP, unemployment, inflation data)
- "In which year did public health improve most significantly?" (using mortality rate changes)
- "Did inequality decrease faster in developed or developing countries?" (using Gini coefficient trends)

### Examples to Avoid
- Open-ended explanations requiring lengthy text responses
- Questions with multiple valid interpretations
- Subjective assessments without clear criteria

## Answer Format Guidelines

### Numerical Answers
- Use appropriate precision (integers for counts, decimals for rates/percentages)
- Example: `746438` or `48.49`
- **Evaluation**: Highly reliable with tolerance for rounding

### List Answers
- Use array format with consistent item types
- Example: `["United States", "China"]` or `[1990, 1995, 2000]`
- **Evaluation**: Consider order-insensitive matching for most cases

### Text Answers
- **Keep extremely concise** (1-2 words maximum)
- Example: `"United States"`, `"improving"`, `"significant"`
- **Evaluation Concern**: Even short text may require careful matching

### Boolean Answers
- Use `true` or `false` (lowercase)
- **Evaluation**: Most reliable format

### Categorical Answers
- Use exact string matching expected categories
- Example: `"high"`, `"developing"`, `"significant"`
- **Evaluation**: Reliable with exact string matching

## Answer Evaluation Considerations

### High Reliability (Recommended for automated evaluation)
- `numerical`: Direct numeric comparison with tolerance
- `boolean`: Exact matching
- `categorical`: Exact string matching

### Medium Reliability
- `list`: Requires consideration of order sensitivity and partial matches

### Low Reliability (Use sparingly)
- `text`: Even when concise, requires careful string matching

## Quality Standards

### Question Requirements
1. **Specificity**: Questions must be answerable with the provided data
2. **Clarity**: Unambiguous language and clear expectations
3. **Relevance**: Questions should test meaningful analytical capabilities
4. **Diversity**: Vary complexity, domains, and reasoning types
5. **Evaluability**: Prefer answer types that can be reliably evaluated
6. **Conciseness**: Keep answers brief and specific

### Reasoning Steps Requirements
1. **Completeness**: Cover all necessary steps to reach the answer
2. **Logical Order**: Steps should follow logical sequence
3. **Specificity**: Reference exact column names and operations
4. **Clarity**: Each step should be clearly understandable

### Answer Type Selection Guidelines
1. **For arithmetic operations**: Use `numerical`
2. **For filtering/selection**: Use `list` or `categorical`
3. **For yes/no questions**: Use `boolean`
4. **For inference**: Use `categorical` or `numerical` (avoid lengthy `text`)

## Distractor Table Guidelines

### Design Principles
- **Maintain Answer Integrity**: Distractor tables should not change the correct answer
- **Test Robustness**: Evaluate agent's ability to identify relevant data and ignore distractions
- **Realistic Scenarios**: Simulate real-world data analysis situations with multiple data sources
- **Clear Reasoning**: Include explicit steps to ignore or filter out distractor information

### Implementation Guidelines
- **Table Selection**: Choose distractor tables that are plausible but not directly relevant
- **Reasoning Steps**: Include explicit steps to identify and ignore distractor tables
- **Complexity Adjustment**: Account for additional tables in complexity calculations
- **Answer Verification**: Ensure distractor tables do not affect the correct answer

### Distractor Type Selection
1. **For testing domain knowledge**: Use `irrelevant` (completely unrelated domains)
2. **For testing data selection**: Use `relevant` (related but unnecessary data)
3. **For testing critical thinking**: Use `misleading` (appears relevant but leads to wrong conclusions)
4. **For baseline comparison**: Use `none` (original questions without distractors) 