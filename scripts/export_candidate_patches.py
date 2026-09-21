#!/usr/bin/env python3
"""Export accepted, validated candidate patches from a corpus run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect candidate skill patches from completed TIDE items.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    run_dir.relative_to(root.resolve())
    output = Path(args.output).resolve() if args.output else run_dir / "candidate_patches.json"
    output.relative_to(root.resolve())
    candidates: list[dict[str, Any]] = []
    rejected_count = 0
    for status_path in sorted((run_dir / "items").glob("*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            continue
        for learned in status.get("learned", []):
            episode = learned.get("episode", {})
            patch = learned.get("patch", {})
            if episode.get("status") == "accepted" and patch.get("status") == "candidate":
                candidates.append(
                    {
                        "item_id": status.get("item_id"),
                        "prompt_sha256": status.get("prompt_sha256"),
                        "episode_id": episode.get("episode_id"),
                        "patch": patch,
                    }
                )
            else:
                rejected_count += 1
    by_rule: dict[str, int] = {}
    for candidate in candidates:
        rule_id = str(candidate["patch"].get("edit", {}).get("rule_id", ""))
        by_rule[rule_id] = by_rule.get(rule_id, 0) + 1
    result = {
        "schema_version": "1.0",
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "noncandidate_episode_count": rejected_count,
        "candidate_counts_by_rule": by_rule,
        "candidates": candidates,
        "promotion_note": "Candidates remain unpromoted until an independent held-out report passes.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "candidate_count": len(candidates), "by_rule": by_rule}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
