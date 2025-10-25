from __future__ import annotations


def guidance_for(error_code: str, failed_plan_count: int = 0) -> str:
    mapping = {
        "LLM_TIMEOUT": "Your last response timed out. Respond faster. Output exactly one block only. No markdown.",
        "LLM_UNAVAILABLE": "Previous attempt failed to connect. Retry now. Output exactly one block only (PLAN or END). No markdown.",
        "FORMAT_NO_BLOCK": "No valid block found. Start directly with PLAN or END. Do not use **PLAN**, ##PLAN, or any prefixes. Output exactly: PLAN\\n<commands>\\nEND PLAN or END\\n<answer>\\nEND",
        "FORMAT_MULTIPLE_BLOCKS": "You output more than one block. Produce exactly one block per turn (PLAN or END). No markdown or explanations.",
        "PLAN_SYNTAX": 'Your PLAN has DSL syntax errors. Common issues: 1) Use exact column names with quotes: "Column Name" 2) Use derive() for calculations: derive(new_col = "A" + "B") or derive(new_col = mean("A")) 3) Use filter_all() for multiple conditions: filter_all(Year >= 2000; Entity == "US") 4) For aggregations, use derive(new_col = mean("Column")) or groupby().agg({mean: "Column"}) 5) Use END PLAN not END 6) Check parentheses are balanced 7) NO natural language or comments in PLAN blocks 8) For groupby with expressions, first derive a new column. Fix and resend.',
        "EXEC_ERROR": 'Execution failed. Common fixes: 1) Use \'derive(new_col = "A" + "B")\' instead of \'select("A" + "B")\' 2) Use exact column names with spaces 3) Check table name is correct 4) For aggregations, use groupby().agg({mean: "column"}) NOT derive(col = mean("column")) 5) Use proper column names from table structure 6) Use descending=False instead of descending=True 7) For groupby expressions, first derive a new column. Fix and resend.',
        "COLUMN_NAME_ERROR": "Column name not found. Use the exact column names from the table. Copy them exactly as shown in the table structure. Check for typos and ensure proper quoting. Available columns: [list first 5 columns]",
    }

    # Handle stubborn behavior for all error types
    if failed_plan_count >= 3:
        if error_code == "PLAN_SYNTAX":
            return f"STUBBORN BEHAVIOR DETECTED: You've made the same DSL syntax error {failed_plan_count} times. You MUST try a completely different approach. Consider: 1) Use simpler DSL operations 2) Check exact column names 3) Use basic select/filter only 4) Try END block instead of PLAN. Do NOT repeat the same PLAN."
        elif error_code == "EXEC_ERROR":
            return f"STUBBORN BEHAVIOR DETECTED: You've tried the same failed PLAN {failed_plan_count} times. You MUST try a completely different approach. Consider: 1) Simpler operations 2) Different column names 3) Basic select/filter only 4) Check table names. Do NOT repeat the same PLAN."
        elif error_code == "FORMAT_NO_BLOCK":
            return f"STUBBORN BEHAVIOR DETECTED: You've failed to output proper blocks {failed_plan_count} times. You MUST try a completely different approach. Consider: 1) Use simple END block with direct answer 2) Avoid complex PLAN blocks 3) Check your output format. Do NOT repeat the same approach."
        else:
            return f"STUBBORN BEHAVIOR DETECTED: You've made the same error {failed_plan_count} times. You MUST try a completely different approach. Consider: 1) Use simpler operations 2) Try END block instead of PLAN 3) Use basic select/filter only. Do NOT repeat the same approach."

    if error_code.startswith("VALIDATION_"):
        return "Validation failed. Use only provided tables/columns and allowed operators; respect resource limits. Fix and resend."
    return mapping.get(
        error_code,
        "General error. Produce exactly one valid PLAN or END block. No markdown.",
    )
