"""Minimal client for a local LLM served by Ollama (default: Gemma 3 4B). Standard library only.

    reply = chat([{"role": "user", "content": "..."}], schema={...})   # -> dict parsed from the model's JSON

Env vars: OLLAMA_URL (default http://localhost:11434), BRAIN_MODEL (default gemma3:4b).
"""
import json
import os
import urllib.request

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("BRAIN_MODEL", "gemma3:4b")


def chat(messages, schema=None, timeout=60.0):
    """One chat call. `schema` (JSON schema) forces the reply into that shape; temperature 0 = repeatable."""
    messages = [{"role": m["role"], "content": m["content"]} for m in messages]   # drop our own bookkeeping keys
    body = {"model": MODEL, "messages": messages, "stream": False, "format": schema or "json",
            "options": {"temperature": 0}, "keep_alive": "30m"}
    req = urllib.request.Request(f"{URL}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(json.loads(r.read())["message"]["content"])
