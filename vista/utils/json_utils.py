from __future__ import annotations

import ast
import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_fenced_or_raw(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def parse_json_like(text: str, default: Any = None) -> Any:
    payload = extract_fenced_or_raw(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    # A model may prepend an explanation before an otherwise valid JSON object.
    # Decode every possible object/array start and keep the longest complete
    # fragment so that an inner ``module`` object cannot shadow its outer update.
    decoder = json.JSONDecoder()
    fragments: list[tuple[int, int, Any]] = []
    for start, char in enumerate(payload):
        if char not in "[{":
            continue
        try:
            parsed, end = decoder.raw_decode(payload, start)
        except json.JSONDecodeError:
            continue
        fragments.append((end - start, -start, parsed))
    if fragments:
        return max(fragments, key=lambda item: (item[0], item[1]))[2]

    try:
        return ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return default


def parse_python_list(text: str) -> list[str]:
    parsed = parse_json_like(text, default=None)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def find_scores(data: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "score":
                continue
            if isinstance(value, dict) and "score" in value:
                try:
                    scores[str(key)] = float(value["score"])
                except (TypeError, ValueError):
                    pass
            elif isinstance(value, (int, float)):
                scores[str(key)] = float(value)
            elif isinstance(value, dict):
                scores.update(find_scores(value))
    return scores
