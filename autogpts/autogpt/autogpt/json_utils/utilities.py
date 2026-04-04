"""Utilities for the json_fixes package."""
import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    """Extract a JSON dict from an LLM response string.

    Handles:
    - Bare JSON
    - JSON wrapped in ```json ... ``` or ``` ... ``` code blocks
    - Thinking-model preamble before the JSON object
    - ast.literal_eval fallback for Python-dict-style responses
    """
    if not response_content:
        return {}

    text = response_content.strip()

    # Strip <think>...</think> reasoning blocks emitted by reasoning models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip code fences (```json ... ``` or ``` ... ```)
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)

    # Find the outermost JSON object (skip any preamble/thinking text)
    brace_match = re.search(r"\{", text)
    if brace_match:
        start = brace_match.start()
        # Walk to find the matching closing brace
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: try full text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: Python literal eval (original behaviour)
    try:
        return ast.literal_eval(text)
    except BaseException as e:
        logger.info(f"All JSON extraction methods failed: {e}")
        logger.debug(f"Invalid response content: {response_content}")
        return {}
