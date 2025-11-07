# llm_cloud.py
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
USER_TMPL = 'Command: "{cmd}"\nReturn ONLY JSON.'

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    blob = m.group(0) if m else text
    data = json.loads(blob)
    # minimal validation
    assert data["relation"] in ["next to","left of","right of","above","below","close to"]
    assert data["src"] in ["robot","object"]
    assert data["dst_color"] in ["red","blue","green","yellow"]
    assert data["dst_shape"] in ["square","circle","triangle"]
    return data

OPENAI_BASE = "https://api.openai.com/v1/responses"  # Responses API

def parse_openai(cmd: str, model: str, api_key: Optional[str] = None,
                 max_retries: int = 4, timeout_s: int = 60) -> Dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --llm_api_key")

    payload = {
        "model": model,
        "input": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":USER_TMPL.format(cmd=cmd)}
        ],
        "temperature": 0.0
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # simple retry/backoff for 429/5xx
    for attempt in range(max_retries):
        r = requests.post(OPENAI_BASE, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            # exponential backoff with small jitter
            time.sleep(min(8, 2 ** attempt) + 0.1 * attempt)
            continue
        r.raise_for_status()

        data = r.json()
        # Responses API may expose either 'output_text' or the raw blocks
        text = data.get("output_text")
        if not text:
            # fallback to blocks: output[0].content[0].text
            try:
                text = data["output"][0]["content"][0]["text"]
            except Exception:
                # final fallback: stringify json (will raise in _extract_json if not valid)
                text = json.dumps(data)

        return _extract_json(text)

    # If we got here and didn't return, last call raised; raise for clarity
    r.raise_for_status()  # will throw the last error

def parse_cloud(provider: str, cmd: str, model: str,
                api_key: Optional[str] = None, endpoint: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Unified entry used by main.py. For now, we support 'openai'.
    You can extend with 'mistral' or 'google' later.
    """
    provider = (provider or "openai").lower()
    if provider != "openai":
        raise ValueError("Only 'openai' provider is implemented in llm_cloud.py right now.")
    return parse_openai(cmd, model, api_key=api_key)
