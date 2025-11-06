# llm_cloud.py
import os, json, re, requests
from typing import Dict, Any, Optional

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
    return data



OPENAI_BASE = "https://api.openai.com/v1/responses"  # Responses API

def parse_openai(cmd: str, model: str, api_key: str = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --llm_api_key")
    SYSTEM = (
        "You convert user commands about a 2D world into a strict JSON:\n"
        '{ "src": "robot" | "object", '
        '"src_color": null | "red" | "blue" | "green" | "yellow", '
        '"src_shape": null | "square" | "circle" | "triangle", '
        '"relation": "next to" | "left of" | "right of" | "above" | "below" | "close to", '
        '"dst_color": "red" | "blue" | "green" | "yellow", '
        '"dst_shape": "square" | "circle" | "triangle" }\n'
        "Return ONLY JSON."
    )
    USER = f'Command: "{cmd}"\nReturn ONLY JSON.'
    payload = {
        "model": model,
        "input": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":USER}
        ],
        "temperature": 0.0
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_BASE, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["output"][0]["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)

# -------------------------
# Unified dispatcher
# -------------------------
def parse_cloud(provider: str, cmd: str, model: str,
                api_key: Optional[str] = None, endpoint: Optional[str]=None) -> Dict[str, Any]:
    provider = provider.lower()
    if provider == "openai":
        return parse_openai(cmd, model, api_key=api_key)
    elif provider == "mistral":
        return parse_mistral(cmd, model, api_key=api_key, endpoint=endpoint)
    elif provider in ["google","gemini","gemma"]:
        return parse_gemini(cmd, model, api_key=api_key)
    else:
        raise ValueError("Unknown provider. Use one of: openai, mistral, google")
