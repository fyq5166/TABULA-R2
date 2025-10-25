from __future__ import annotations

import re
from typing import Tuple


def parse_plan_or_end(text: str, plan_executed: bool = False) -> Tuple[str, str]:
    """Return (kind, content) where kind in {PLAN, END} or (ERROR, code).

    Args:
        text: The LLM response text to parse
        plan_executed: Whether the current PLAN has already been executed in this conversation
    """
    if not text or not text.strip():
        return ("ERROR", "FORMAT_NO_BLOCK")

    # First, define END patterns for detection
    end_patterns = [
        # **END** variants (most common)
        r"\*\*END\*\*\s*\n([\s\S]+?)(?:\n\*\*|$)",  # **END** followed by content (at least 1 char)
        r"\*\*END\*\*\s+([^\n]+)(?:\n|$)",  # **END** followed by single line
        r"\*\*END\*\*\s*$",  # **END** with no content (return empty string)
        # ##END variants
        r"##\s+END\s*\n([\s\S]*?)(?:\n|$)",  # ## END followed by content
        r"##\s+END\s+([^\n]+)(?:\n|$)",  # ## END followed by single line
        # END with explanatory prefixes (high priority)
        r"(?:^|\n)here\s+is\s+the\s+answer:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Here is the answer: END
        r"(?:^|\n)the\s+answer\s+is:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # The answer is: END
        r"(?:^|\n)my\s+answer:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # My answer: END
        r"(?:^|\n)final\s+answer:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Final answer: END
        r"(?:^|\n)the\s+result\s+is:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # The result is: END
        r"(?:^|\n)answer:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Answer: END
        r"(?:^|\n)result:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Result: END
        # END with more complex explanatory prefixes
        r"(?:^|\n)based\s+on\s+.*?:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Based on...: END
        r"(?:^|\n)after\s+.*?:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # After...: END
        r"(?:^|\n)following\s+.*?:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Following...: END
        r"(?:^|\n)according\s+to\s+.*?:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # According to...: END
        # Standard END blocks (but avoid matching "END PLAN")
        r"(?:^|\n)END\s*\n([\s\S]*?)\nEND(?:\n|$)",  # Standard END block (not END PLAN)
        r"(?:^|\n)END\s*\n([\s\S]*?)$",  # END at end of text (not END PLAN)
        r"(?:^|\n)END\s+([^\n]+)(?:\n|$)",  # END followed by single line (not END PLAN)
        # END with other prefixes
        r"(?:^|\n)therefore\s+END\s*\n([\s\S]*?)(?:\n|$)",  # Therefore END
        r"(?:^|\n)thus\s+END\s*\n([\s\S]*?)(?:\n|$)",  # Thus END
        r"(?:^|\n)hence\s+END\s*\n([\s\S]*?)(?:\n|$)",  # Hence END
        r"(?:^|\n)conclusion:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Conclusion: END
        r"(?:^|\n)summary:\s*END\s*\n([\s\S]*?)(?:\n|$)",  # Summary: END
    ]

    # SPECIAL HANDLING: If both PLAN and END are present, decide based on execution status
    has_plan = "PLAN" in text or "**PLAN**" in text or "##PLAN" in text
    has_end = "END" in text or "**END**" in text or "##END" in text

    if has_plan and has_end:
        if plan_executed:
            # PLAN has been executed, extract END as the answer
            # This means the model is providing the final answer based on executed results
            for pattern in end_patterns:
                if pattern == r"\*\*END\*\*\s*$":
                    # Special case: empty END block
                    if re.search(pattern, text, flags=re.IGNORECASE):
                        return ("END", "")  # Return empty string for empty END block
                else:
                    end_matches = re.findall(pattern, text, flags=re.IGNORECASE)
                    if end_matches:
                        end_content = end_matches[0].strip() if end_matches[0] else ""
                        if end_content != "PLAN":
                            return ("END", end_content)
        else:
            # PLAN has not been executed, prioritize PLAN execution
            # Extract PLAN content and return it, ignore END for now
            # Collect all PLAN blocks from different formats
            all_plan_content = []

            # First, extract nested **PLAN** with PLAN...END PLAN content (highest priority)
            nested_plan_matches = re.findall(
                r"\*\*PLAN\*\*\s*\nPLAN\s*\n([\s\S]*?)\nEND PLAN", text
            )
            all_plan_content.extend(nested_plan_matches)

            # Then extract standard **PLAN** format (but avoid duplicates from nested)
            plan_matches = re.findall(r"\*\*PLAN\*\*\s*\n([\s\S]*?)\n\*\*END\*\*", text)
            for match in plan_matches:
                # Only add if not already captured by nested format
                if not any(match.strip() in nested for nested in nested_plan_matches):
                    all_plan_content.append(match)

            # Finally extract standard PLAN format (but avoid duplicates)
            standard_plan_matches = re.findall(r"PLAN\n([\s\S]*?)\nEND PLAN", text)
            for match in standard_plan_matches:
                # Only add if not already captured by nested format
                if not any(match.strip() in nested for nested in nested_plan_matches):
                    all_plan_content.append(match)

            if all_plan_content:
                # Merge all PLAN blocks
                merged_content = "\n".join(
                    [match.strip() for match in all_plan_content if match.strip()]
                )
                return ("PLAN", merged_content)

    # SPECIAL HANDLING: Check for END blocks anywhere in the text (when not both present)
    # This handles cases where model outputs END in the middle of a PLAN
    # But avoid matching "END PLAN" - only match standalone END blocks
    for pattern in end_patterns:
        if pattern == r"\*\*END\*\*\s*$":
            # Special case: empty END block
            if re.search(pattern, text, flags=re.IGNORECASE):
                return ("END", "")  # Return empty string for empty END block
        else:
            end_matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if end_matches:
                # Found END block, extract the content
                end_content = end_matches[0].strip() if end_matches[0] else ""
                if end_content != "PLAN":  # Avoid extracting "PLAN" from "END PLAN"
                    return ("END", end_content)

    # SPECIAL HANDLING: Clean common LLM output prefixes that interfere with parsing
    # This is a workaround for models that add explanatory prefixes despite explicit instructions
    cleaned_text = text.strip()

    # Remove common prefixes that models add despite being told not to
    # Order matters: more specific patterns first, then general patterns
    prefix_patterns = [
        # Specific common patterns first
        r"^here\s+is\s+the\s+solution:\s*",  # Here is the solution:
        r"^here\s+is\s+the\s+output:\s*",  # Here is the output:
        r"^here\s+is\s+the\s+response:\s*",  # Here is the response:
        r"^here\s+is\s+the\s+corrected\s+plan:\s*",  # Here is the corrected plan:
        r"^here\s+is\s+my\s+response:\s*",  # Here is my response:
        r"^here\s+is\s+my\s+attempt:\s*",  # Here is my attempt:
        r"^here\s+is\s+the\s+revised\s+plan:\s*",  # Here is the revised plan:
        r"^here\s+is\s+the\s+plan:\s*",  # Here is the plan:
        r"^here\s+is\s+my\s+plan:\s*",  # Here is my plan:
        r"^here\s+is\s+my\s+solution:\s*",  # Here is my solution:
        # Code block combinations
        r"^here\s+is\s+.*?```\s*",  # Here is... ```
        r"^this\s+is\s+.*?```\s*",  # This is... ```
        r"^my\s+.*?```\s*",  # My... ```
        # General patterns
        r"^here\s+is\s+.*?:\s*",  # Here is anything:
        r"^this\s+is\s+.*?:\s*",  # This is anything:
        r"^my\s+.*?:\s*",  # My anything:
        r"^the\s+.*?:\s*",  # The anything:
        # Markdown and formatting prefixes - PRIORITY: Handle **PLAN** and ##PLAN
        r"^\*\*PLAN\*\*\s*",  # **PLAN** -> (empty, need to add PLAN back)
        r"^##\s+PLAN\s*",  # ## PLAN -> (empty, need to add PLAN back)
        r"^\*\*.*?\*\*\s*",  # **anything**
        r"^##?\s+.*?\s*",  # # or ## anything
        r"^step\s+\d+.*?:\s*",  # Step X: anything
        r"^\d+\.\s*",  # 1. anything
        # Code block markers
        r"^```\s*",  # ```
        r"^`\s*",  # `
    ]

    # Track if we removed **PLAN** or ##PLAN prefixes
    had_plan_prefix = False
    for pattern in prefix_patterns:
        if pattern in [r"^\*\*PLAN\*\*\s*", r"^##\s+PLAN\s*"]:
            if re.match(pattern, cleaned_text, flags=re.IGNORECASE):
                had_plan_prefix = True
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    # If we removed **PLAN** or ##PLAN, add PLAN back and END PLAN
    if had_plan_prefix and not cleaned_text.startswith("PLAN"):
        cleaned_text = "PLAN\n" + cleaned_text + "\nEND PLAN"

    # Additional aggressive cleaning for stubborn prefixes
    # Remove any remaining explanatory prefixes
    cleaned_text = re.sub(
        r"^(Here is|This is|The|My|I will|Let me|I need).*?:\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    # Remove any remaining markdown formatting
    cleaned_text = re.sub(r"^[#*]+.*?:\s*", "", cleaned_text)
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
    cleaned_text = re.sub(r"^`\s*", "", cleaned_text)

    # Remove any remaining step numbers or labels
    cleaned_text = re.sub(
        r"^(Step \d+|Step\d+|Step:)\s*", "", cleaned_text, flags=re.IGNORECASE
    )
    cleaned_text = re.sub(r"^\d+\.\s*", "", cleaned_text)

    # Remove any remaining explanatory text before PLAN/END (but be more careful)
    # Only remove if there's actual explanatory text, not just formatting
    if not cleaned_text.startswith(("PLAN", "END")):
        cleaned_text = re.sub(
            r"^[^PEND]*?(?=PLAN|END)", "", cleaned_text, flags=re.IGNORECASE
        )

    # Also clean any leading/trailing whitespace after prefix removal
    cleaned_text = cleaned_text.strip()

    # Now parse the cleaned text
    plan_matches = re.findall(r"PLAN\n([\s\S]*?)\nEND PLAN", cleaned_text)
    end_matches = re.findall(r"(?:^|\n)END\n([\s\S]*?)\nEND(?:\n|$)", cleaned_text)

    # Handle incomplete PLAN format (PLAN without END PLAN)
    if len(plan_matches) == 0 and len(end_matches) == 0:
        # Check for incomplete PLAN format: "PLAN command1 command2 ..."
        incomplete_plan_match = re.search(
            r"PLAN\s+([\s\S]+?)(?:\s*#.*)?$", cleaned_text, re.MULTILINE
        )
        if incomplete_plan_match:
            plan_content = incomplete_plan_match.group(1).strip()
            # Remove any trailing comments
            plan_content = re.sub(r"\s*#.*$", "", plan_content, flags=re.MULTILINE)
            if plan_content:
                return ("PLAN", plan_content)

    total = len(plan_matches) + len(end_matches)
    if total == 0:
        return ("ERROR", "FORMAT_NO_BLOCK")
    if total > 1:
        return ("ERROR", "FORMAT_MULTIPLE_BLOCKS")
    if len(plan_matches) == 1:
        return ("PLAN", plan_matches[0].strip())
    return ("END", end_matches[0].strip())
