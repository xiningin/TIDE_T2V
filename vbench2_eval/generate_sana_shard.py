#!/usr/bin/env python3
"""Generate one prepared VBench-2.0 job shard with a persistent SANA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.clients.sana_client import SanaDiffusersT2VClient  # noqa: E402
from vista.utils.config import load_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--gpu", required=True, help="Physical GPU id visible to the SANA child."
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def valid_existing_receipt(
    receipt: dict[str, Any], job: dict[str, Any], config_sha256: str
) -> bool:
    path = Path(str(job["output_path"]))
    if (
        receipt.get("status") not in {"generated", "resumed"}
        or receipt.get("job_id") != job.get("job_id")
        or receipt.get("video_path") != str(path)
        or receipt.get("t2v_config_sha256") != config_sha256
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        return False
    return receipt.get("video_sha256") == sha256_file(path)


def main() -> None:
    args = _parser().parse_args()
    jobs = read_jsonl(args.jobs.resolve())
    if not jobs:
        raise ValueError(f"empty job shard: {args.jobs}")
    ids = [str(job.get("job_id", "")) for job in jobs]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("job ids must be present and unique inside a shard")

    experiment = json.loads(args.experiment.resolve().read_text(encoding="utf-8"))
    config_path = Path(str(experiment["t2v_config"])).resolve()
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha != experiment.get("t2v_config_sha256"):
        raise RuntimeError(
            "T2V config changed after preparation; rerun prepare before generation"
        )

    prior_rows = read_jsonl(args.results) if args.results.is_file() else []
    prior = {str(row.get("job_id")): row for row in prior_rows}
    completed: dict[str, dict[str, Any]] = {}
    if args.resume:
        for job in jobs:
            receipt = prior.get(str(job["job_id"]))
            if receipt and valid_existing_receipt(receipt, job, actual_config_sha):
                completed[str(job["job_id"])] = receipt

    config = load_config(config_path)["t2v"]
    config["cuda_visible_devices"] = str(args.gpu)
    client: SanaDiffusersT2VClient | None = None
    failures = 0
    try:
        for index, job in enumerate(jobs, start=1):
            job_id = str(job["job_id"])
            target = Path(str(job["output_path"]))
            if job_id in completed:
                print(f"[{index}/{len(jobs)}] resume {job_id}: {target}", flush=True)
                continue

            started = time.perf_counter()
            try:
                if args.resume and target.is_file() and target.stat().st_size > 0:
                    receipt = {
                        "job_id": job_id,
                        "arm": job["arm"],
                        "dimension": job["dimension"],
                        "prompt_id": job["prompt_id"],
                        "sample_index": job["sample_index"],
                        "seed": job["seed"],
                        "status": "resumed",
                        "valid": True,
                        "video_path": str(target),
                        "video_bytes": target.stat().st_size,
                        "video_sha256": sha256_file(target),
                        "elapsed_seconds": 0.0,
                        "final_negative_prompt": job.get("final_negative_prompt", ""),
                        "gpu": str(args.gpu),
                        "t2v_config_sha256": actual_config_sha,
                    }
                else:
                    if client is None:
                        client = SanaDiffusersT2VClient(config)
                    generation = client.generate(
                        prompt=str(job["prompt"]),
                        output_dir=target.parent,
                        candidate_id=target.stem,
                        seed=int(job["seed"]),
                        negative_prompt=str(job.get("conditional_negative_prompt", "")),
                    )
                    generated_path = Path(generation.video_path)
                    if generated_path.resolve() != target.resolve():
                        raise RuntimeError(
                            f"client output mismatch: expected {target}, got {generated_path}"
                        )
                    if not target.is_file() or target.stat().st_size <= 0:
                        raise RuntimeError(
                            f"generation did not create a non-empty file: {target}"
                        )
                    call = client.calls[-1]
                    final_negative = str(call["negative_prompt"])
                    expected_negative = str(
                        job.get("final_negative_prompt", final_negative)
                    )
                    if final_negative != expected_negative:
                        raise RuntimeError(
                            "negative prompt mismatch between preparation and SANA client"
                        )
                    receipt = {
                        "job_id": job_id,
                        "arm": job["arm"],
                        "dimension": job["dimension"],
                        "prompt_id": job["prompt_id"],
                        "sample_index": job["sample_index"],
                        "seed": job["seed"],
                        "status": "generated",
                        "valid": True,
                        "video_path": str(target),
                        "video_bytes": target.stat().st_size,
                        "video_sha256": sha256_file(target),
                        "elapsed_seconds": generation.elapsed_seconds,
                        "worker_load_seconds": call["worker_load_seconds"],
                        "final_negative_prompt": final_negative,
                        "gpu": str(args.gpu),
                        "dry_run": generation.dry_run,
                        "t2v_config_sha256": actual_config_sha,
                    }
                completed[job_id] = receipt
                write_jsonl(
                    args.results,
                    [completed[item] for item in ids if item in completed],
                )
                print(
                    f"[{index}/{len(jobs)}] {receipt['status']} {job_id}: {target}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                completed[job_id] = {
                    "job_id": job_id,
                    "arm": job.get("arm"),
                    "dimension": job.get("dimension"),
                    "prompt_id": job.get("prompt_id"),
                    "sample_index": job.get("sample_index"),
                    "seed": job.get("seed"),
                    "status": "failed",
                    "valid": False,
                    "video_path": str(target),
                    "elapsed_seconds": time.perf_counter() - started,
                    "final_negative_prompt": job.get("final_negative_prompt", ""),
                    "gpu": str(args.gpu),
                    "t2v_config_sha256": actual_config_sha,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                write_jsonl(
                    args.results,
                    [completed[item] for item in ids if item in completed],
                )
                print(
                    f"[{index}/{len(jobs)}] failed {job_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
    finally:
        if client is not None:
            client.close()

    summary = {"jobs": len(jobs), "complete": len(jobs) - failures, "failed": failures}
    print(json.dumps(summary, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
