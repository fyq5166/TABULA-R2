"""
Prompts Module Initialization

This file initializes the prompts module and exports the centralized `prompt_manager`
instance for easy access throughout the application.

The new template-based system is managed by the `PromptManager` class, which
handles loading and rendering of Jinja2 templates from the `templates` directory.
"""

from .prompt_manager import PromptManager, prompt_manager

__all__ = ["PromptManager", "prompt_manager"]
