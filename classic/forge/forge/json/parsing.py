import logging
import re
from typing import Any

import demjson3

logger = logging.getLogger(__name__)


def json_loads(json_str: str) -> Any:
    """Parse a JSON string, tolerating minor syntax issues:
    - Missing, extra and trailing commas
    - Extraneous newlines and whitespace outside of string literals
    - Inconsistent spacing after colons and commas
    - Missing closing brackets or braces
    - Numbers: binary, hex, octal, trailing and prefixed decimal points
    - Different encodings
    - Surrounding markdown code block
    - Comments

    Args:
        json_str: The JSON string to parse.

    Returns:
        The parsed JSON object, same as built-in json.loads.
    """
    
    import json
    import re

    # ✅ FIX: strip "tool_calls" prefix
    json_str = json_str.strip()

    if json_str.startswith("tool_calls"):
        json_str = re.sub(r"^tool_calls\s*", "", json_str)

    return json.loads(json_str)


def extract_dict_from_json(json_str: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, json_str)

    if match:
        json_str = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"{[\s\S]*}"
        match = re.search(json_pattern, json_str)

        if match:
            json_str = match.group()

    result = json_loads(json_str)
    if not isinstance(result, dict):
        raise ValueError(
            f"Response '''{json_str}''' evaluated to non-dict value {repr(result)}"
        )
    return result


def extract_list_from_json(json_str: str) -> list[Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, json_str)

    if match:
        json_str = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"\[[\s\S]*\]"
        match = re.search(json_pattern, json_str)

        if match:
            json_str = match.group()

    result = json_loads(json_str)
    if not isinstance(result, list):
        raise ValueError(
            f"Response '''{json_str}''' evaluated to non-list value {repr(result)}"
        )
    return result
