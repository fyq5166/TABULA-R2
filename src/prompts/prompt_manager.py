"""
Prompt Manager for the Research Question Generation System

This module serves as a centralized hub for loading and rendering prompt
templates using a Jinja2-based templating system. It separates prompt content
from application logic, allowing for easier management and iteration.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

# Define the base path for templates relative to this file's location
TEMPLATE_BASE_DIR = Path(__file__).parent / "templates"


class PromptManager:
    """
    Centralized prompt management system using Jinja2 templates.
    """

    def __init__(self, template_dir: Path = TEMPLATE_BASE_DIR):
        """
        Initialize the prompt manager.

        Args:
            template_dir (Path): The base directory where prompt templates are stored.
        """
        if not template_dir.is_dir():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        self.env = Environment(
            loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
        )
        # TODO: Load reasoning definitions from a file instead of hardcoding
        self.reasoning_definitions = self._load_reasoning_definitions()

    def _load_reasoning_definitions(self):
        # Placeholder for loading from a config file in the future
        return {
            "proxy_inference": {
                "description": "Inferring a relationship between two variables where one acts as a proxy for a harder-to-measure concept.",
                "examples": "Example for Proxy Inference...",
            },
            "arithmetic_aggregation": {
                "description": "Performing calculations like sum, average, min, max, or ratio across one or more tables.",
                "examples": "Example for Arithmetic Aggregation...",
            },
        }

    def get_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """
        Generates a complete prompt by rendering a user message template for a specific task.

        Args:
            task (str): The name of the task, corresponding to a subfolder in the templates directory (e.g., 'question_generation').
            context (Dict[str, Any]): A dictionary of values to render into the template.

        Returns:
            str: The fully rendered prompt.
        """
        template_path = f"{task}/user_message_template.md"
        try:
            template = self.env.get_template(template_path)
            # Add shared context like reasoning definitions
            full_context = {
                **context,
                "reasoning_definitions": self.reasoning_definitions,
            }
            return template.render(full_context)
        except TemplateNotFound:
            raise ValueError(
                f"Template for task '{task}' not found at '{template_path}'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to render template for task '{task}': {e}")

    def get_system_message(self, task: str) -> str:
        """
        Retrieves the system message for a given task.

        Args:
            task (str): The name of the task.

        Returns:
            str: The content of the system message.
        """
        template_path = f"{task}/system_message.md"
        try:
            # Jinja environment can also be used to just load files
            return self.env.get_template(template_path).render()
        except TemplateNotFound:
            raise ValueError(
                f"System message for task '{task}' not found at '{template_path}'"
            )

    def get_qna_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generates a complete QnA execution prompt using the general template.
        DEPRECATED: Use get_chat_messages() instead for better model understanding.

        Args:
            context (Dict[str, Any]): Context including:
                - question_id: str
                - table_refs: List[str]
                - question: str
                - answer: str
                - answer_type: str
                - visible_columns_desc: str
                - visible_rows_desc: str
                - table_stats_desc: str
                - exemplars_block: str (optional, for few-shot)

        Returns:
            str: The fully rendered prompt for QnA execution.
        """
        template_path = "qna_execution/general_prompt.md"
        try:
            template = self.env.get_template(template_path)
            # Add system message and optional exemplars
            system_message = self.get_system_message("qna_execution")
            exemplars_block = context.get("exemplars_block", "")
            full_context = {
                **context,
                "system_message": system_message,
                "exemplars_block": exemplars_block,
                "reasoning_definitions": self.reasoning_definitions,
            }
            return template.render(full_context)
        except TemplateNotFound:
            raise ValueError(f"QnA template not found at '{template_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to render QnA template: {e}")

    def get_chat_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return chat-format messages: system + user."""
        system_message = self.get_system_message("qna_execution")
        # Render user message template directly to keep roles separated
        user_template = self.env.get_template("qna_execution/user_message.md")
        exemplars_block = context.get("exemplars_block", "")
        user_content = user_template.render(
            {
                **context,
                "exemplars_block": exemplars_block,
                "reasoning_definitions": self.reasoning_definitions,
            }
        )
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def build_exemplars(
        self,
        *,
        few_shot_k: int = 0,
        include_cot: bool = True,
        is_multi_table: bool = False,
        seed: int = 42,
    ) -> str:
        """Build exemplars from bank with arbitrary K and optional CoT."""
        if few_shot_k <= 0:
            return ""
        try:
            import json, random

            index_tpl = self.env.get_template("qna_execution/few_shot/index.json")
            index = json.loads(index_tpl.render())
            # filter by tables kind
            filtered = [
                it
                for it in index
                if (
                    it.get("tables") == ("multi" if is_multi_table else "single")
                    or it.get("tables") == "either"
                )
            ]
            if not filtered:
                filtered = index
            rnd = random.Random(seed)
            rnd.shuffle(filtered)
            chosen = filtered[:few_shot_k]
            blocks: List[str] = []
            for item in chosen:
                tpl = self.env.get_template(item["path"])
                text = tpl.render({"include_cot": include_cot})
                # Remove any leading example headings like '### Example: ex_xxxx'
                cleaned_lines = []
                for line in text.splitlines():
                    if line.strip().startswith("### Example:"):
                        continue
                    cleaned_lines.append(line)
                blocks.append("\n".join(cleaned_lines).strip())
            return "\n\n".join(blocks)
        except Exception:
            return ""


# Global instance for easy access
prompt_manager = PromptManager()

__all__ = ["PromptManager", "prompt_manager"]
