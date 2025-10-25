"""
LLM Client for Ollama API

This module provides a standardized client for interacting with the Ollama API.
It handles requests, responses, streaming, and error handling in a centralized way.
"""

import json
import requests
import logging
from typing import Dict, Any, Optional, Generator, List

from .logging_utils import get_logger


class LLMClient:
    """
    A client for making requests to a local Ollama server.

    This client simplifies interactions with the Ollama API by providing
    methods for standard generation and error handling.
    """

    def __init__(
        self,
        model_name: str,
        url: str = "http://localhost:11434",
        backend: str = "ollama",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model_name (str): The default model to use for requests.
            url (str): The base URL for the Ollama server.
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.base_url = url.rstrip("/")
        self.backend = backend  # 'ollama' | 'openai_compat'
        self.api_key = api_key
        # Endpoints differ by backend
        if self.backend == "openai_compat":
            # vLLM OpenAI-compatible chat completions endpoint
            self.chat_api_url = f"{self.base_url}/v1/chat/completions"
            self.api_url = f"{self.base_url}/v1/completions"
        else:
            self.api_url = f"{self.base_url}/api/generate"
            self.chat_api_url = f"{self.base_url}/api/chat"
        self.is_available = self._test_connection()

        if not self.is_available:
            self.logger.error(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running and the model '{self.model_name}' is available."
            )
            # We don't raise an exception here to allow for initialization
            # even if the server is temporarily down.

    def _test_connection(self) -> bool:
        """Test the connection to the Ollama server and check for model availability."""
        try:
            if self.backend == "openai_compat":
                # For vLLM/OpenAI compatible, just try a lightweight GET or assume available
                resp = requests.get(f"{self.base_url}", timeout=3)
                resp.raise_for_status()
                return True
            else:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(self.model_name in name for name in model_names):
                    self.logger.debug(
                        f"Successfully connected to Ollama; model '{self.model_name}' is available."
                    )
                    return True
                self.logger.warning(
                    f"Connected to Ollama, but model '{self.model_name}' not found. Available models: {model_names}"
                )
                return False
        except requests.RequestException as e:
            self.logger.error(
                f"Failed to connect to LLM backend at {self.base_url}: {e}"
            )
            return False

    def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt (str): The prompt to send to the model.
            options (Optional[Dict[str, Any]]): Ollama generation options (e.g., temperature).

        Returns:
            Optional[str]: The generated text response, or None if an error occurred.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        try:
            self.logger.debug(f"Sending request to LLM: {prompt[:100]}...")
            response = requests.post(
                self.api_url, json=payload, timeout=timeout_s or 60
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "").strip()
            self.logger.debug(f"Received LLM response: {response_text[:100]}...")
            if response_text:
                summary = response_text.replace("\n", " ")[:120]
                self.logger.info(f"LLM reply (truncated): {summary}...")
            return response_text

        except requests.Timeout:
            self.logger.error("Request to Ollama timed out.")
            raise TimeoutError("TIME OUT【llmcall】")
        except requests.RequestException as e:
            self.logger.error(f"An error occurred during request to Ollama: {e}")
            return None
        except json.JSONDecodeError:
            self.logger.error("Failed to decode JSON response from Ollama.")
            return None

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate a response using chat format with system/user message separation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Valid roles: 'system', 'user', 'assistant'
            options: Optional Ollama generation options (e.g., temperature).
            timeout_s: Optional timeout in seconds.

        Returns:
            Optional[str]: The generated text response, or None if an error occurred.
        """
        # Validate messages format
        if not messages:
            self.logger.error("Messages list cannot be empty")
            return None

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                self.logger.error(f"Invalid message format at index {i}: {msg}")
                return None
            if msg["role"] not in ["system", "user", "assistant"]:
                self.logger.error(
                    f"Invalid role '{msg['role']}' at index {i}. Valid roles: system, user, assistant"
                )
                return None

        if self.backend == "openai_compat":
            # OpenAI compatible schema
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": options.get("temperature", 0.0) if options else 0.0,
                "top_p": options.get("top_p", 1.0) if options else 1.0,
                "max_tokens": options.get("num_predict", 512) if options else 512,
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
            }
            if options:
                payload["options"] = options
            headers = None

        try:
            # Log the conversation structure (without full content for privacy)
            roles = [msg["role"] for msg in messages]
            self.logger.debug(
                f"Sending chat request with {len(messages)} messages: {roles}"
            )

            response = requests.post(
                self.chat_api_url,
                json=payload,
                headers=headers,
                timeout=timeout_s or 60,
            )
            response.raise_for_status()

            result = response.json()
            if self.backend == "openai_compat":
                # OpenAI format
                choices = result.get("choices", [])
                response_text = (
                    choices[0].get("message", {}).get("content", "") if choices else ""
                ).strip()
            else:
                response_text = result.get("message", {}).get("content", "").strip()

            self.logger.debug(f"Received chat response: {response_text[:100]}...")
            if response_text:
                # Log complete response for debugging, but keep terminal output concise
                self.logger.debug(f"LLM complete response:\n{response_text}")
                summary = response_text.replace("\n", " ")[:120]
                self.logger.info(f"LLM chat reply (truncated): {summary}...")
            return response_text

        except requests.Timeout:
            self.logger.error("Chat request to Ollama timed out.")
            raise TimeoutError("TIME OUT【llmcall】")
        except requests.RequestException as e:
            self.logger.error(f"An error occurred during chat request to Ollama: {e}")
            return None
        except json.JSONDecodeError:
            self.logger.error("Failed to decode JSON response from Ollama chat API.")
            return None
        except KeyError as e:
            self.logger.error(f"Unexpected response format from Ollama chat API: {e}")
            return None
