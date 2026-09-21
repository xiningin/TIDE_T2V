from __future__ import annotations

import re
from typing import Any


WORDS_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "during",
    "explicitly",
    "in",
    "is",
    "of",
    "one",
    "or",
    "source",
    "the",
    "that",
    "to",
}
UNBOUNDED_TRIGGER_RE = re.compile(r"\b(?:always apply|every prompt|all prompts|any prompt)\b", re.IGNORECASE)


def trigger_words(value: str) -> set[str]:
    return {word for word in WORDS_RE.findall(value.lower()) if word not in STOP_WORDS}


def trigger_jaccard(left: str, right: str) -> float:
    left_words = trigger_words(left)
    right_words = trigger_words(right)
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 1.0


def validate_trigger_candidate(
    rules: dict[str, Any],
    rule_id: str,
    trigger: str,
    *,
    change_rationale: str | None = None,
) -> dict[str, Any]:
    """Validate one learned trigger without changing its frozen semantic axis."""
    errors: list[str] = []
    policy = rules.get("trigger_policy") if isinstance(rules.get("trigger_policy"), dict) else {}
    modules = [item for item in rules.get("modules", []) if isinstance(item, dict)]
    current = next((item for item in modules if item.get("rule_id") == rule_id), None)
    candidate = str(trigger).strip()
    if current is None:
        errors.append(f"unknown rule_id: {rule_id}")
        old_trigger = ""
    else:
        old_trigger = str(current.get("trigger", ""))
    changed = bool(current is not None and candidate != old_trigger)
    if not candidate:
        errors.append("trigger is empty")
    max_chars = int(policy.get("max_trigger_chars", 320))
    if len(candidate) > max_chars:
        errors.append(f"trigger exceeds {max_chars} characters")
    if changed and policy.get("trigger_trainable") is not True:
        errors.append("trigger_policy does not allow learned trigger changes")
    if changed and change_rationale is not None and not str(change_rationale).strip():
        errors.append("learned trigger change requires direct routing evidence")

    global_ids = {str(item) for item in policy.get("global_rule_ids", [])}
    if rule_id not in global_ids and UNBOUNDED_TRIGGER_RE.search(candidate):
        errors.append("conditional trigger cannot become an unbounded global trigger")

    threshold = float(policy.get("max_conditional_trigger_jaccard", 0.35))
    overlaps: list[dict[str, Any]] = []
    if rule_id not in global_ids:
        for other in modules:
            other_id = str(other.get("rule_id", ""))
            if not other_id or other_id == rule_id or other_id in global_ids:
                continue
            score = trigger_jaccard(candidate, str(other.get("trigger", "")))
            overlaps.append({"rule_id": other_id, "jaccard": round(score, 4)})
            if score >= threshold:
                errors.append(
                    f"trigger overlaps {other_id} at {score:.3f}, exceeding the {threshold:.3f} limit"
                )

    return {
        "valid": not errors,
        "changed": changed,
        "max_conditional_trigger_jaccard": max((item["jaccard"] for item in overlaps), default=0.0),
        "overlaps": overlaps,
        "errors": errors,
    }
