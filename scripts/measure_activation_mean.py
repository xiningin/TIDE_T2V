#!/usr/bin/env python3
"""Measure Full-skill activation counts for the random-trigger ablation.

The random-trigger arm needs a target count calibrated from an unchanged Full
skill run. This utility scans trajectory JSON files and reports the mean number
of activated rules (including global rules), plus a rounded integer suitable for
``skill_learning.random_activation_count``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _counts(value: object) -> list[int]:
    found: list[int] = []
    if isinstance(value, dict):
        activated = value.get("activated_rule_ids")
        if isinstance(activated, list) and activated and all(isinstance(x, (str, int)) for x in activated):
            found.append(len(activated))
        for child in value.values():
            found.extend(_counts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_counts(child))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Full-skill output directory to scan")
    args = parser.parse_args()
    root = Path(args.root)
    counts: list[int] = []
    for path in sorted(root.rglob("trajectory.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Prefer the initial frozen-skill rewrite, one count per prompt. A
        # trajectory can also contain later copies of the same activation list.
        initial = value.get("initial_skill_rewrite") if isinstance(value, dict) else None
        activated = initial.get("activated_rule_ids") if isinstance(initial, dict) else None
        if isinstance(activated, list):
            counts.append(len(activated))
        else:
            counts.extend(_counts(value)[:1])
    if not counts:
        raise SystemExit(f"No activated_rule_ids found under {root}")
    mean = sum(counts) / len(counts)
    print(json.dumps({
        "trajectory_count": len(counts),
        "activation_counts": {str(k): counts.count(k) for k in sorted(set(counts))},
        "mean_activation_count": mean,
        "random_activation_count": int(round(mean)),
        "min": min(counts),
        "max": max(counts),
    }, indent=2))


if __name__ == "__main__":
    main()
