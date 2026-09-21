#!/usr/bin/env python3
"""Minimal, secret-safe smoke test for EpicLLMRouter chat completions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("EPICLLMROUTER_BASE_URL", "https://api.epicllmrouter.top/v1")
DEFAULT_MODELS = ("deepseek-v4-flash", "qwen3.6-35b-a3b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help="Optional key file. Defaults to the EPICLLMROUTER_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model ID to test; repeat for multiple models. Defaults to the two verified models.",
    )
    parser.add_argument("--prompt", default="Reply with exactly: API_OK")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def chat_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def probe(url: str, api_key: str, model: str, prompt: str, timeout: int) -> dict[str, Any]:
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128,
            "thinking": {"type": "disabled"},
        },
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError:
        return {"model": model, "ok": False, "status": response.status_code, "error": "non-JSON response"}
    if not response.ok:
        error = payload.get("error", payload)
        if isinstance(error, dict):
            error = error.get("message", error)
        return {"model": model, "ok": False, "status": response.status_code, "error": str(error)[:500]}
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    return {
        "model": model,
        "response_model": payload.get("model"),
        "ok": bool(message.get("content")),
        "status": response.status_code,
        "content": str(message.get("content") or "")[:500],
    }


def main() -> int:
    args = parse_args()
    if args.key_file is not None:
        api_key = args.key_file.expanduser().read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get("EPICLLMROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "No API key. Set EPICLLMROUTER_API_KEY or pass --key-file pointing at a key file."
        )
    models = args.models or list(DEFAULT_MODELS)
    results = [probe(chat_url(args.base_url), api_key, model, args.prompt, args.timeout) for model in models]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
