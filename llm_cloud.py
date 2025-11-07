import os, json, re, time
from typing import Dict, Any, Optional
import requests

# --- Common system prompt (tight JSON schema) ---
SYSTEM = (
    "You convert user commands about a 2D world into a strict JSON:\n"
    '{ "src": "robot" | "object", '
    '"src_color": null | "red" | "blue" | "green" | "yellow", '
    '"src_shape": null | "square" | "circle" | "triangle", '
    '"relation": "next to" | "left of" | "right of" | "above" | "below" | "close to", '
    '"dst_color": "red" | "blue" | "green" | "yellow", '
    '"dst_shape": "square" | "circle" | "triangle" }\n'
    "Return ONLY JSON. No prose."
)
# This system prompt is sent to the LLM to force it to output *only* a strict JSON structure.
# The model should never produce natural text, just an object like:
# { "src":"object", "src_color":"red", "src_shape":"square", ... }


USER_TMPL = 'Command: "{cmd}"\nReturn ONLY JSON.'
# Template for the user’s message. Example:
# Command: "move the robot next to the blue circle"
# Return ONLY JSON.


# ----------------------------------------------------------------------
# Helper to extract a JSON object from LLM output
# ----------------------------------------------------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts and validates the JSON object returned by the LLM.
    - Finds the first {...} block via regex (to ignore stray text).
    - Parses JSON.
    - Validates required keys and value ranges.
    """
    # Find a JSON object substring using a permissive regex
    m = re.search(r"\{.*\}", text, re.S)
    blob = m.group(0) if m else text

    # Try to load as JSON
    data = json.loads(blob)

    # --- minimal validation (assert ensures correct schema) ---
    assert data["relation"] in ["next to", "left of", "right of", "above", "below", "close to"]
    assert data["src"] in ["robot", "object"]
    assert data["dst_color"] in ["red", "blue", "green", "yellow"]
    assert data["dst_shape"] in ["square", "circle", "triangle"]

    return data


# ----------------------------------------------------------------------
# OpenAI-specific cloud parser
# ----------------------------------------------------------------------

OPENAI_BASE = "https://api.openai.com/v1/responses"  # Responses API endpoint

def parse_openai(cmd: str, model: str, api_key: Optional[str] = None,
                 max_retries: int = 4, timeout_s: int = 60) -> Dict[str, Any]:
    """
    Send the command to the OpenAI 'Responses' API and return parsed JSON.

    Args:
      cmd         : natural language command ("put the red circle above the green square")
      model       : model name, e.g. "gpt-4.1-mini"
      api_key     : OpenAI API key (or read from env OPENAI_API_KEY)
      max_retries : how many times to retry on transient errors
      timeout_s   : per-request timeout in seconds

    Returns:
      Dict with keys like {src, src_color, src_shape, relation, dst_color, dst_shape}
    """
    # Get API key from argument or environment variable
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --llm_api_key")

    # Build the request payload for the Responses API
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER_TMPL.format(cmd=cmd)}
        ],
        "temperature": 0.0  # deterministic behavior
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Simple retry with exponential backoff for rate limit or server errors
    # ------------------------------------------------------------------
    for attempt in range(max_retries):
        r = requests.post(OPENAI_BASE, headers=headers, data=json.dumps(payload), timeout=timeout_s)

        # Retry on transient 429 / 5xx
        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            # Sleep: exponential backoff with slight jitter
            time.sleep(min(8, 2 ** attempt) + 0.1 * attempt)
            continue

        # If non-retriable or final attempt, raise on HTTP error
        r.raise_for_status()

        # Parse JSON response
        data = r.json()

        # The Responses API may return output in different formats:
        # - `output_text`: direct text result
        # - `output`: list of content blocks
        text = data.get("output_text")
        if not text:
            # Try to extract text from blocks
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                # Last resort: stringify the raw JSON
                text = json.dumps(data)

        # Extract and validate the JSON object from model text
        return _extract_json(text)

    # If loop exits (e.g., max_retries exhausted), raise the last HTTP error
    r.raise_for_status()


# ----------------------------------------------------------------------
# Unified entry point for future multi-provider support
# ----------------------------------------------------------------------

def parse_cloud(provider: str, cmd: str, model: str,
                api_key: Optional[str] = None, endpoint: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Unified entry for external LLM providers.
    Currently supports only 'openai', but can be extended.

    Example call:
      parse_cloud("openai", "move red circle next to blue square",
                  model="gpt-4.1-mini", api_key="...")

    Returns:
      A dictionary structured as:
        {
          "src": "object",
          "src_color": "red",
          "src_shape": "circle",
          "relation": "next to",
          "dst_color": "blue",
          "dst_shape": "square"
        }
    """
    provider = (provider or "openai").lower()

    if provider != "openai":
        raise ValueError("Only 'openai' provider is implemented in llm_cloud.py right now.")

    return parse_openai(cmd, model, api_key=api_key)
