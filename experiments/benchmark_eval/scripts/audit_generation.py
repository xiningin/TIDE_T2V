#!/usr/bin/env python3
"""Audit prepared videos and receipts without invoking any evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generation_common import atomic_json, read_json, read_jsonl, sha256_file, sha256_text  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", type=Path, required=True)
    result.add_argument("--report", type=Path)
    result.add_argument("--max-errors", type=int, default=200)
    return result


def _errors(job: dict[str, Any]) -> list[str]:
    target = Path(str(job["output_path"]))
    receipt_path = Path(str(job["receipt_path"]))
    if not target.is_file():
        return ["missing_video"]
    if target.stat().st_size <= 0:
        return ["empty_video"]
    if not receipt_path.is_file():
        return ["missing_receipt"]
    try:
        receipt = read_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["invalid_receipt_json"]

    problems: list[str] = []
    expected = {
        "schema_version": "1.0",
        "status": "generated",
        "dry_run": False,
        "job_id": job["job_id"],
        "job_ordinal": job["job_ordinal"],
        "arm": job["arm"],
        "group": job["group"],
        "benchmark": job["benchmark"],
        "prompt_set": job["prompt_set"],
        "prompt_index": job["prompt_index"],
        "sample_index": job["sample_index"],
        "seed": job["seed"],
        "raw_prompt": job["raw_prompt"],
        "positive_prompt": job["positive_prompt"],
        "conditional_negative_prompt": job["conditional_negative_prompt"],
        "final_negative_prompt": job["final_negative_prompt"],
        "positive_prompt_sha256": sha256_text(str(job["positive_prompt"])),
        "negative_prompt_sha256": sha256_text(str(job["final_negative_prompt"])),
        "method_signature": job["method_signature"],
        "generator_signature": job["generator_signature"],
        "video_path": str(target),
        "video_bytes": target.stat().st_size,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            problems.append(f"receipt_{field}_mismatch")
    if receipt.get("video_sha256") != sha256_file(target):
        problems.append("video_sha256_mismatch")
    return problems


def main() -> None:
    args = parser().parse_args()
    if args.max_errors < 0:
        raise ValueError("max-errors must be non-negative")
    experiment_path = args.experiment.expanduser().resolve()
    experiment = read_json(experiment_path)
    jobs = read_jsonl(Path(str(experiment["jobs"])).resolve())
    if len(jobs) != int(experiment["task_count"]):
        raise ValueError("experiment task_count does not match jobs.jsonl")

    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    complete = 0
    for job in jobs:
        problems = _errors(job)
        if not problems:
            complete += 1
            continue
        counts.update(problems)
        if len(failures) < args.max_errors:
            failures.append(
                {
                    "job_id": job["job_id"],
                    "output_path": job["output_path"],
                    "receipt_path": job["receipt_path"],
                    "errors": problems,
                }
            )

    report = {
        "schema_version": "1.0",
        "arm": experiment["arm"],
        "status": "complete" if complete == len(jobs) else "incomplete",
        "expected": len(jobs),
        "complete": complete,
        "incomplete": len(jobs) - complete,
        "error_counts": dict(sorted(counts.items())),
        "reported_errors": failures,
        "errors_truncated": max(0, len(jobs) - complete - len(failures)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else experiment_path.parent / "generation_audit.json"
    )
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
