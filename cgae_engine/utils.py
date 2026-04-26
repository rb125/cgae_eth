"""Shared utilities for the CGAE engine."""

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code block wrapping.

    Returns the cleaned JSON string or None if no JSON found.
    """
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from text (tolerant of markdown wrapping)."""
    cleaned = extract_json(text)
    if cleaned is None:
        return None
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
