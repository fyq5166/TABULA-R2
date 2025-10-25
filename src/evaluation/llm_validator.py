"""
LLM-based validator for evaluating model responses.

Design:
- Uses a configurable LLM model as a validator (default: mock-always-true).
- Configurable timeout per validation.
- Strict prompt: output MUST be exactly `True` or `False` with no extra text.
- Decision rule in prompt: case-insensitive exact match after trimming whitespace.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from typing import Optional, Dict, Any

import requests

# Mocks removed from validator path; always use HTTP when model != 'mock-always-true'


@dataclass
class ValidationResult:
    """Result of a validation call."""

    decision: Optional[bool]  # True/False/None (None = invalid response)
    time_s: float
    retried: bool = False
    error: Optional[str] = None


PROMPT_TEMPLATE = (
    "System: You are a strict, deterministic binary validator.\n"
    "Task: Given a question, a gold answer (reference), and a model answer, decide if the model answer is correct.\n"
    "All answers are provided as strings.\n"
    "\n"
    "Decision guidelines (apply consistently):\n"
    "- Trim surrounding whitespace on both answers.\n"
    "- Case-insensitive.\n"
    "- Ignore trailing punctuation marks (., !, ?).\n"
    "- Treat boolean synonyms as equivalent (yes/true vs no/false).\n"
    "- Treat common numeric expressions as equivalent when clearly the same value in context (e.g., '3' vs 'three').\n"
    "- For comma-separated lists: compare as sets (order-insensitive), split by ',', trim items, ignore case and duplicates.\n"
    "- If the model answer contradicts the gold answer, is unrelated, or does not address the question, output False.\n"
    "- Prefer precision over generosity; if uncertain, output False.\n"
    "\n"
    "Output format requirement:\n"
    "- Output MUST be exactly one token: True or False (capitalized exactly). No extra text.\n"
    "\n"
    "Question: {question}\n"
    "Gold Answer: {gold}\n"
    "Model Answer: {model_answer}\n"
    "\n"
    "Output (must be exactly 'True' or 'False'):"
)

RETRY_PROMPT_TEMPLATE = (
    "System: You are a strict, deterministic binary validator.\n"
    "Task: Given a question, a gold answer (reference), and a model answer, decide if the model answer is correct.\n"
    "All answers are provided as strings.\n"
    "\n"
    "Apply the same rules as before (whitespace/case normalization, trailing punctuation ignored, boolean synonyms, numeric equivalence when obvious, list as sets).\n"
    "\n"
    "Examples (negative cases that must be False):\n"
    "- Q: Largest planet? | Gold: Jupiter | Model: Saturn -> False\n"
    "- Q: Is 5 greater than 7? | Gold: false | Model: true -> False\n"
    "- Q: List top 2 countries | Gold: United States, China | Model: china, united states -> False\n"
    "- Q: What is the capital of the moon? | Gold: This question cannot be answered based on the given tables. | Model: Question cannot be answered -> True\n"
    "\n"
    "CRITICAL STABILITY REQUIREMENT:\n"
    "- If your output is not exactly one token 'True' or 'False', it will be considered incorrect.\n"
    "- Do NOT output any other text.\n"
    "\n"
    "Question: {question}\n"
    "Gold Answer: {gold}\n"
    "Model Answer: {model_answer}\n"
    "\n"
    "Output (must be exactly 'True' or 'False'):"
)


class LLMValidator:
    """LLM-based validator for evaluating model responses."""

    def __init__(
        self,
        model: str = "mock-always-true",
        timeout_s: float = 2.0,
        url: str = "http://localhost:11434",
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.url = url

        # Mocks removed; keep attribute for compatibility
        self.client = None

    def _call_ollama_generate(
        self, prompt: str, options: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Call Ollama API for validation."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        try:
            resp = requests.post(
                f"{self.url}/api/generate", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
            obj = resp.json()
            return obj.get("response", "").strip()
        except requests.Timeout:
            raise TimeoutError("TIME OUT【llmcall】")
        except requests.RequestException:
            return None
        except json.JSONDecodeError:
            return None

    def _judge(self, prompt: str, opts: Dict[str, Any]) -> Optional[bool]:
        """Parse LLM response to boolean decision."""
        if self.model == "mock-always-true":
            return True
        out = self._call_ollama_generate(prompt, options=opts)

        if out is None:
            return None
        token = out.split()[0] if out else ""
        if token == "True":
            return True
        if token == "False":
            return False
        return None

    def validate(
        self,
        question: str,
        gold_answer: str,
        model_answer: str,
        allow_retry: bool = True,
    ) -> ValidationResult:
        """Validate a model answer against gold answer."""
        start_time = time.perf_counter()

        try:
            base_prompt = PROMPT_TEMPLATE.format(
                question=question, gold=gold_answer, model_answer=model_answer
            )
            base_opts = {"temperature": 0.0, "top_p": 0.9, "num_predict": 4}
            res = self._judge(base_prompt, base_opts)

            if res is not None or not allow_retry:
                elapsed = time.perf_counter() - start_time
                return ValidationResult(decision=res, time_s=elapsed, retried=False)

            # Retry with stricter prompt
            retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                question=question, gold=gold_answer, model_answer=model_answer
            )
            retry_opts = {"temperature": 0.0, "top_p": 0.5, "num_predict": 2}
            res2 = self._judge(retry_prompt, retry_opts)
            elapsed = time.perf_counter() - start_time
            return ValidationResult(decision=res2, time_s=elapsed, retried=True)

        except TimeoutError as e:
            elapsed = time.perf_counter() - start_time
            return ValidationResult(decision=None, time_s=elapsed, error=str(e))
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return ValidationResult(decision=None, time_s=elapsed, error=str(e))
