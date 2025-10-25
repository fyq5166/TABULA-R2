## Question
{{ question }}

## Data Available
{% if table_refs %}
**Available tables:** {{ table_refs | join(', ') }}
{% endif %}
{% if table_description %}
{{ table_description }}
{% endif %}

{{ table_stats_desc }}

{{ visible_rows_desc }}

{% if table_structures %}
## Table Structure Information
{% for table_id, table_info in table_structures.items() %}
**{{ table_id }}:**
- Columns: {{ table_info.columns | join(', ') }}
- Data types: {{ table_info.dtypes | join(', ') }}
- Sample values: {{ table_info.sample_values }}
{% endfor %}
{% endif %}

{% if history %}
## Previous Steps
**Note: The results of your DSL operations are shown below. You can see the actual data and use it to make decisions.**
{% include 'qna_execution/history_section.md' %}
{% endif %}

{% if exemplars_block %}
## Examples (for format and reasoning style)
The following examples illustrate how to think step-by-step (internally) and how to format the final output.
Each example shows optional Reasoning (do not output this) followed by the exact output block (PLAN or END).

{{ exemplars_block }}
{% endif %}

{% if is_final_round %}
## FINAL ROUND WARNING
**THIS IS YOUR LAST CHANCE!** You must output exactly one END block with your final answer. Do not output a PLAN block. You must provide your final answer now.

END
<your final answer>
END
{% else %}
## IMPORTANT: When to use END vs PLAN
- Use **PLAN** when you need to analyze or process data
- Use **END** when you have the final answer and want to provide it
- If you already have the answer from previous analysis, output END with your answer
- Don't keep analyzing if you already know the result

## END Block Output Format
**The END block is where you provide your final answer. The answer can be:**
- **String**: A text response (e.g., "The GDP growth rate is 3.2%")
- **Integer**: A numerical value (e.g., 42)
- **List**: Multiple items (e.g., "Afghanistan, Angola, Burundi")
- **Paragraph**: A detailed explanation or analysis

**Format:**
```
END
<your final answer here>
END
```

## CRITICAL: PLAN Block Must Contain ONLY DSL Commands
**If you output a PLAN block, it MUST contain ONLY executable DSL commands:**
- `load(table_XXXX)`
- `filter_all(condition)`
- `select("column1", "column2")`
- `derive(new_col = "col1" + "col2")`
- `order("column", ascending=False)`
- `limit(5)`
- `groupby().agg({sum: "column"})`

**DO NOT include in PLAN blocks:**
- Natural language descriptions
- Explanations like "This plan aims to..."
- Reasoning like "I will analyze..."
- Text like "The varying estimates suggest..."
- Any non-DSL content

**WRONG:** `PLAN This plan aims to analyze the data... END PLAN`
**CORRECT:** `PLAN load(table_0100) filter_all(Entity == "US") select("Year") END PLAN`

## CRITICAL: For List Questions
- **If the question asks for a list** (e.g., "which countries", "what are all the..."), you MUST return ALL matching items
- **Do NOT use limit()** unless specifically asked for a limited number
- **Your END block should contain the complete list** of all matching items
- **Example**: If asked "which countries have X < 20", return ALL countries that meet the condition
{% endif %}

CRITICAL FORMATTING REQUIREMENTS:
- Start your response with EXACTLY "PLAN" or "END"
- NO prefixes: "Here is...", "## PLAN", "**PLAN**", "This is...", "My...", etc.
- NO explanations before the block
- NO markdown formatting
- NO code block markers (```)
- NO step numbers or labels

PLEASE ONLY CONTAIN BLOCK SIGNAL `PLAN` & `END` and INFORMATION INSIDE BLOCK, NO INFORMATION SHOULD BE OUTPUT OUTSIDE THE BLOCK