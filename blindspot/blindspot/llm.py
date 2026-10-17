"""Tiny LLM client: free cloud models through any OpenAI-compatible API, with
a disk cache so the demo runs offline and gives identical answers every time.

Settings come from environment variables or a .env file in the project folder
(never commit .env):

    LLM_PROVIDER=groq            # or "openrouter"
    GROQ_API_KEY=...
    OPENROUTER_API_KEY=...
    LLM_MODEL=...                # optional; overrides the provider default
    LLM_OFFLINE=1                # optional; use the cache only, never call out

RULE: only public text (advisories) and aggregate numbers go to the model.
Never send transactions or customer IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional; plain env vars work too
    pass

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}
CACHE = Path("llm_cache")


def _settings() -> tuple[str, dict, str]:
    name = os.getenv("LLM_PROVIDER", "groq").lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got {name!r}")
    p = PROVIDERS[name]
    return name, p, os.getenv("LLM_MODEL", p["model"])


def _key(model: str, messages: list[dict], json_mode: bool) -> str:
    blob = json.dumps({"model": model, "messages": messages, "json": json_mode}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def chat(messages: list[dict], json_mode: bool = False, temperature: float = 0.0) -> str:
    """Send messages, return the reply text. Cached by model + messages."""
    name, p, model = _settings()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{_key(model, messages, json_mode)}.json"
    if path.exists():
        return json.loads(path.read_text())["reply"]
    if os.getenv("LLM_OFFLINE") == "1":
        raise RuntimeError(f"LLM_OFFLINE=1 and no cached reply ({path}). Run once online first.")

    api_key = os.getenv(p["key"])
    if not api_key:
        raise RuntimeError(f"Set {p['key']} in your .env file (provider: {name}).")
    from openai import OpenAI  # imported here so the rest of Blindspot works without it
    client = OpenAI(base_url=p["base_url"], api_key=api_key)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=temperature, **kwargs)
    reply = resp.choices[0].message.content or ""
    path.write_text(json.dumps({"provider": name, "model": model, "messages": messages,
                                "reply": reply}, indent=2))
    return reply


def chat_json(messages: list[dict]) -> dict:
    """Like chat(), but parses a JSON object out of the reply."""
    reply = chat(messages, json_mode=True)
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", reply, re.S)   # models sometimes wrap JSON in prose
        if not m:
            raise ValueError(f"model did not return JSON: {reply[:200]}")
        return json.loads(m.group(0))