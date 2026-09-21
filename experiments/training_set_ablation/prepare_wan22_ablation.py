#!/usr/bin/env python3
"""Prepare Wan2.2 generation experiments from trigger-ablation prompt jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WAN_REPO = Path("/tmp/Wan2.2_repo")
CHECKPOINT = Path(os.environ.get("WAN22_CKPT_DIR", "/path/to/Wan2.2-TI2V-5B"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    source = Path(args.source_dir).resolve()
    out = Path(args.output_dir).resolve()
    jobs_in = source / "prompts_495.jsonl"
    if not jobs_in.is_file():
        raise FileNotFoundError(jobs_in)
    if not WAN_REPO.is_dir():
        raise FileNotFoundError(WAN_REPO)
    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(CHECKPOINT)
    rows = [json.loads(line) for line in jobs_in.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 495:
        raise ValueError(f"expected 495 prompt jobs, found {len(rows)}")
    arm = out.name
    generator = {
        "backend": "wan_ti2v",
        "wan_repo": str(WAN_REPO),
        "checkpoint": str(CHECKPOINT),
        "width": 832,
        "height": 480,
        "frames": 81,
        "fps": 16,
        "steps": 50,
        "guidance_scale": 5.0,
        "sample_shift": 3.0,
    }
    method_sig = text_sha({"arm": arm, "source_jobs_sha256": sha256(jobs_in), "skill_revision": rows[0].get("skill_revision"), "skill_sha256": rows[0].get("skill_sha256")})
    generator_sig = text_sha({"generator": generator, "wan_repo_head": (WAN_REPO / ".git" / "HEAD").read_text(encoding="utf-8").strip()})
    registry = {
        "schema_version": "1.0",
        "fixed_negative_prompt": "chaotic sequence, deformed limbs, heavy motion blur, jump cuts, jerky movements, temporal artifacts, jitter, ghosting",
        "generator": generator,
        "arm": arm,
    }
    out.mkdir(parents=True, exist_ok=True)
    registry_path = out / "generation_registry.json"
    write_json(registry_path, registry)
    jobs: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        filename = str(row["benchmark_filename"])
        video = out / "videos" / "sana_ablation_vbench_core" / arm / filename
        receipt = out / "receipts" / "sana_ablation_vbench_core" / arm / Path(filename).with_suffix(".json")
        job = dict(row)
        job.update({
            "group": "sana_ablation_vbench_core",
            "arm": arm,
            "model": "Wan2.2-TI2V-5B",
            "method": arm,
            "benchmark": "vbench",
            "job_id": f"sana_ablation_vbench_core:{arm}:{row['prompt_set']}:{row['prompt_index']}:{row['sample_index']}",
            "job_ordinal": ordinal,
            "output_path": str(video.resolve()),
            "receipt_path": str(receipt.resolve()),
            "method_signature": method_sig,
            "generator_signature": generator_sig,
            "generator": "wan22",
            "prompt_method": arm,
            "rewrite_metadata": {
                "skill_revision": row.get("skill_revision"),
                "skill_sha256": row.get("skill_sha256"),
                "source_prompt_sha256": row.get("source_prompt_sha256"),
                "activation_mode": "all" if arm == "no_tigger" else "random",
                "random_activation_count": 3 if arm == "random_trigger" else None,
                "random_activation_seed": 42,
            },
        })
        jobs.append(job)
    jobs_path = out / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs), encoding="utf-8")
    experiment = {
        "schema_version": "1.0",
        "arm": arm,
        "ready": True,
        "groups": ["sana_ablation_vbench_core"],
        "task_count": len(jobs),
        "generator": "wan22",
        "prompt_method": arm,
        "experiment_dir": str(out),
        "method_signature": method_sig,
        "generator_signature": generator_sig,
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "jobs": str(jobs_path),
        "videos_root": str(out / "videos"),
        "receipts_root": str(out / "receipts"),
        "generator_config": generator,
        "fixed_negative_prompt": registry["fixed_negative_prompt"],
        "skill_runtime": {"source_prompt_dir": str(source), "wan_repo_head": (WAN_REPO / ".git" / "HEAD").read_text(encoding="utf-8").strip()},
        "videos_generated": False,
    }
    write_json(out / "experiment.json", experiment)
    print(json.dumps({"arm": arm, "jobs": len(jobs), "experiment": str(out / 'experiment.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
