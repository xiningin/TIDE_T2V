#!/usr/bin/env python3
"""Validate the rewrite Skill v1.1 conservative trigger-learning contract."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parents[1]
AXIS_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
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


def _words(value: str) -> set[str]:
    return {word for word in WORDS_RE.findall(value.lower()) if word not in STOP_WORDS}


def validate(rules: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if rules.get("skill_version") != "1.1":
        errors.append("skill_version must be 1.1")
    if int(rules.get("revision", 0)) < 0:
        errors.append("revision must be non-negative")
    policy = rules.get("trigger_policy")
    if not isinstance(policy, dict):
        errors.append("missing trigger_policy")
        policy = {}
    if policy.get("version") != "1.1":
        errors.append("trigger_policy.version must be 1.1")
    if policy.get("trigger_trainable") is not True:
        errors.append("trigger_policy.trigger_trainable must be true")
    if policy.get("update_mode") != "conservative_evidence_gated":
        errors.append("trigger_policy.update_mode must be conservative_evidence_gated")
    if policy.get("trigger_axis_frozen") is not True:
        errors.append("trigger_policy.trigger_axis_frozen must be true")
    if policy.get("capability_scope_frozen") is not True:
        errors.append("trigger_policy.capability_scope_frozen must be true")
    if int(policy.get("min_batch_change_support", 0)) < 2:
        errors.append("trigger_policy.min_batch_change_support must be >= 2")
    if int(policy.get("max_trigger_chars", 0)) <= 0:
        errors.append("trigger_policy.max_trigger_chars must be positive")
    similarity_limit = float(policy.get("max_conditional_trigger_jaccard", 0.0))
    if not 0.0 < similarity_limit < 1.0:
        errors.append("trigger_policy.max_conditional_trigger_jaccard must be between 0 and 1")
    if policy.get("unique_axis_required") is not True:
        errors.append("trigger_policy.unique_axis_required must be true")
    if policy.get("multi_activation_allowed") is not True:
        errors.append("trigger_policy.multi_activation_allowed must be true")

    modules = rules.get("modules")
    if not isinstance(modules, list) or not modules:
        errors.append("rules.json requires a non-empty modules list")
        modules = []
    rule_ids: set[str] = set()
    axes: dict[str, str] = {}
    triggers: dict[str, str] = {}
    for module in modules:
        if not isinstance(module, dict):
            errors.append("every module must be an object")
            continue
        rule_id = str(module.get("rule_id", "")).strip()
        trigger = str(module.get("trigger", "")).strip()
        axis = str(module.get("trigger_axis", "")).strip()
        if not rule_id or rule_id in rule_ids:
            errors.append(f"missing or duplicate rule_id: {rule_id!r}")
        rule_ids.add(rule_id)
        if not trigger:
            errors.append(f"{rule_id}: missing trigger")
        max_chars = int(policy.get("max_trigger_chars", 320))
        if len(trigger) > max_chars:
            errors.append(f"{rule_id}: trigger exceeds {max_chars} characters")
        if not AXIS_RE.fullmatch(axis):
            errors.append(f"{rule_id}: invalid trigger_axis {axis!r}")
        if axis in axes:
            errors.append(f"trigger_axis {axis!r} is shared by {axes[axis]} and {rule_id}")
        axes[axis] = rule_id
        triggers[rule_id] = trigger

    global_ids = {str(item) for item in policy.get("global_rule_ids", [])}
    if not global_ids.issubset(rule_ids):
        errors.append("trigger_policy.global_rule_ids contains an unknown rule")
    for axis, rule_id in axes.items():
        if rule_id in global_ids and not axis.startswith("global."):
            errors.append(f"{rule_id}: global rule must use a global.* axis")
        if rule_id not in global_ids and axis.startswith("global."):
            errors.append(f"{rule_id}: conditional rule cannot use a global.* axis")

    similarities: list[dict[str, Any]] = []
    conditional_ids = sorted(rule_ids - global_ids)
    for left, right in combinations(conditional_ids, 2):
        left_words = _words(triggers[left])
        right_words = _words(triggers[right])
        union = left_words | right_words
        score = len(left_words & right_words) / len(union) if union else 1.0
        similarities.append({"left": left, "right": right, "jaccard": round(score, 4)})
        if score >= similarity_limit:
            errors.append(f"conditional triggers are too lexically similar: {left} vs {right} ({score:.3f})")

    return {
        "valid": not errors,
        "skill_version": rules.get("skill_version"),
        "revision": rules.get("revision"),
        "rule_count": len(rule_ids),
        "global_rule_count": len(global_ids),
        "conditional_rule_count": len(conditional_ids),
        "max_conditional_trigger_jaccard": max((item["jaccard"] for item in similarities), default=0.0),
        "max_allowed_conditional_trigger_jaccard": similarity_limit,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=SKILL_DIR / "rules.json")
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    report = validate(rules)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
