#!/usr/bin/env python3
"""Audit paired VBench-2.0 manifests, receipts, hashes, and video formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.utils.config import load_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    return streams[0]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = _parser().parse_args()
    experiment_path = args.experiment.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    output = Path(str(experiment["output"])).resolve()
    arms = [str(value) for value in experiment["arms"]]
    t2v_config_path = Path(str(experiment["t2v_config"])).resolve()
    t2v = load_config(t2v_config_path)["t2v"]
    width, height = (
        int(value) for value in str(t2v["size"]).lower().replace("x", "*").split("*")
    )
    expected_frames = int(t2v["frame_num"])
    expected_fps = float(t2v["fps"])
    expected_config_sha = str(experiment["t2v_config_sha256"])
    jobs = []
    for path in sorted((output / "jobs").glob("shard_*.jsonl")):
        jobs.extend(read_jsonl(path))
    receipts = []
    for path in sorted((output / "receipts").glob("shard_*.jsonl")):
        receipts.extend(read_jsonl(path))

    job_by_id = {str(row.get("job_id")): row for row in jobs}
    receipt_by_id = {str(row.get("job_id")): row for row in receipts}
    if len(job_by_id) != len(jobs):
        raise ValueError("duplicate job id across shard files")
    if len(receipt_by_id) != len(receipts):
        raise ValueError("duplicate receipt id across shard files")

    pair_fields = (
        "dimension",
        "prompt_id",
        "source_line",
        "canonical_prompt",
        "sample_index",
        "seed",
    )
    manifests: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, Any] = {}
    errors: list[str] = []
    if sha256_file(t2v_config_path) != expected_config_sha:
        errors.append("T2V config hash differs from the prepared experiment")
    if len(jobs) != int(experiment["job_count"]):
        errors.append(
            f"job count mismatch: experiment={experiment['job_count']} shards={len(jobs)}"
        )
    for arm in arms:
        manifest_path = output / arm / "manifest.jsonl"
        manifest = read_jsonl(manifest_path)
        manifests[arm] = manifest
        format_counts: Counter[str] = Counter()
        total_bytes = 0
        verified = 0
        missing = 0
        failed = 0
        for row in manifest:
            job_id = str(row["job_id"])
            job = job_by_id.get(job_id)
            receipt = receipt_by_id.get(job_id)
            path = Path(str(row["video_path"]))
            if job is None:
                errors.append(f"{arm}: manifest job absent from shards: {job_id}")
                missing += 1
                continue
            expected_job_values = {
                "arm": row.get("arm"),
                "dimension": row.get("dimension"),
                "prompt_id": row.get("prompt_id"),
                "sample_index": row.get("sample_index"),
                "seed": row.get("seed"),
                "output_path": row.get("video_path"),
                "final_negative_prompt": row.get("final_negative_prompt"),
            }
            mismatched = [
                field
                for field, expected in expected_job_values.items()
                if job.get(field) != expected
            ]
            if mismatched:
                errors.append(
                    f"{arm}: job/manifest mismatch for {job_id}: {mismatched}"
                )
                failed += 1
                continue
            if receipt is None:
                errors.append(f"{arm}: missing receipt: {job_id}")
                missing += 1
                continue
            if receipt.get("valid") is not True:
                errors.append(
                    f"{arm}: failed receipt: {job_id}: {receipt.get('error', '')}"
                )
                failed += 1
                continue
            if receipt.get("video_path") != str(path):
                errors.append(f"{arm}: receipt path mismatch: {job_id}")
                failed += 1
                continue
            if receipt.get("final_negative_prompt") != row.get("final_negative_prompt"):
                errors.append(f"{arm}: final negative prompt mismatch: {job_id}")
                failed += 1
                continue
            if receipt.get("t2v_config_sha256") != expected_config_sha:
                errors.append(f"{arm}: receipt T2V config hash mismatch: {job_id}")
                failed += 1
                continue
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"{arm}: missing or empty video: {path}")
                missing += 1
                continue
            digest = sha256_file(path)
            if digest != receipt.get("video_sha256"):
                errors.append(f"{arm}: video hash mismatch: {path}")
                failed += 1
                continue
            try:
                stream = probe(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{arm}: ffprobe failed for {path}: {exc}")
                failed += 1
                continue
            format_counts[
                f"{stream.get('width')}x{stream.get('height')}|"
                f"fps={stream.get('avg_frame_rate')}|frames={stream.get('nb_frames')}"
            ] += 1
            actual_width = int(stream.get("width", 0))
            actual_height = int(stream.get("height", 0))
            try:
                actual_fps = float(Fraction(str(stream.get("avg_frame_rate", "0"))))
            except (ValueError, ZeroDivisionError):
                actual_fps = 0.0
            try:
                actual_frames = int(stream.get("nb_frames", 0))
            except (TypeError, ValueError):
                actual_frames = 0
            format_errors = []
            if (actual_width, actual_height) != (width, height):
                format_errors.append(
                    f"resolution={actual_width}x{actual_height}, expected={width}x{height}"
                )
            if abs(actual_fps - expected_fps) > 1e-6:
                format_errors.append(f"fps={actual_fps}, expected={expected_fps}")
            if actual_frames != expected_frames:
                format_errors.append(
                    f"frames={actual_frames}, expected={expected_frames}"
                )
            if format_errors:
                errors.append(
                    f"{arm}: invalid video format for {path}: {format_errors}"
                )
                failed += 1
                continue
            verified += 1
            total_bytes += path.stat().st_size
        arm_summaries[arm] = {
            "expected": len(manifest),
            "verified": verified,
            "missing": missing,
            "failed": failed,
            "total_bytes": total_bytes,
            "format_counts": dict(sorted(format_counts.items())),
            "manifest": str(manifest_path),
            "expected_format": {
                "width": width,
                "height": height,
                "fps": expected_fps,
                "frames": expected_frames,
            },
        }

    reference = sorted(
        manifests[arms[0]], key=lambda row: tuple(str(row.get(x)) for x in pair_fields)
    )
    for arm in arms[1:]:
        candidate = sorted(
            manifests[arm], key=lambda row: tuple(str(row.get(x)) for x in pair_fields)
        )
        left = [tuple(row.get(field) for field in pair_fields) for row in reference]
        right = [tuple(row.get(field) for field in pair_fields) for row in candidate]
        if left != right:
            errors.append(f"{arm}: manifest is not seed-paired with {arms[0]}")

    manifest_ids = {
        str(row["job_id"]) for values in manifests.values() for row in values
    }
    extra_jobs = sorted(set(job_by_id).difference(manifest_ids))
    extra_receipts = sorted(set(receipt_by_id).difference(manifest_ids))
    if extra_jobs:
        errors.append(f"{len(extra_jobs)} shard jobs have no manifest row")
    if extra_receipts:
        errors.append(f"{len(extra_receipts)} receipts have no manifest row")

    status = "complete" if not errors else "incomplete"
    summary = {
        "status": status,
        "experiment": str(experiment_path),
        "paired": not any("not seed-paired" in item for item in errors),
        "paired_fields": list(pair_fields),
        "job_count": len(jobs),
        "receipt_count": len(receipts),
        "arms": arm_summaries,
        "errors": errors,
    }
    summary_path = (
        args.summary.resolve() if args.summary else output / "audit_summary.json"
    )
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
