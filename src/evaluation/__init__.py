"""
Evaluation Module

This module provides functionality for evaluating LLM performance
on complex reasoning tasks over tabular data.

Main components:
- LLM response validation and scoring
- Question selection and filtering
- Observation space generation
- Plan parsing and DSL execution
- Mock agents for testing
- Guidance and error handling
"""

# Import implemented classes
from .llm_validator import LLMValidator
from .question_selector import SelectionConfig, collect_question_records
from .observation_space import ObservationSpaceGenerator, ObsConfig
from .plan_parser import parse_plan_or_end
from .dsl_executor import execute_plan, parse_plan, DSLCommand
from .guidance import guidance_for

__all__ = [
    "LLMValidator",
    "SelectionConfig",
    "collect_question_records",
    "ObservationSpaceGenerator",
    "ObsConfig",
    "parse_plan_or_end",
    "execute_plan",
    "parse_plan",
    "DSLCommand",
    "guidance_for",
]
