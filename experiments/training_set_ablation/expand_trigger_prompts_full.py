#!/usr/bin/env python3
"""Expand trigger-ablation prompt rewrites to the complete paired VBench suite.

The existing ablation runs contain the 165-prompt core subset.  This script
reuses those cached rewrites, rewrites only the remaining canonical prompts
with the selected ablation Skill, and then rebuilds a 431 x 3 jobs manifest in
the same order and with the same seeds as the main video-prompting-skill arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN_UNIQUE = ROOT / "experiments/cross_method_results/video-prompting-skill/prompts_431_unique.jsonl"
MAIN_JOBS = ROOT / "experiments/cross_method_results/video-prompting-skill/jobs.jsonl"
CONFIG = ROOT / "config/deepseek_qwen36_35ba3b_sana_posneg_train.yaml"
ROUTER_KEY_ENV = "EPICLLMROUTER_API_KEY"
ROUTER_KEY_FILE_ENV = "EPICLLMROUTER_API_KEY_FILE"
FIXED_NEGATIVE = (
    "chaotic sequence, deformed limbs, heavy motion blur, jump cuts, "
    "jerky movements, temporal artifacts, jitter, ghosting"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    tmp.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rewrite_one(row: dict[str, Any], config: dict[str, Any], skill_dir: str, rules_file: str,
                activation_mode: str, random_count: int | None, random_seed: int) -> dict[str, Any]:
    from vista.clients.mllm_client import MLLMClient
    from vista.core.prompt_rewriter import PromptRewriter

    client = MLLMClient(config)
    rewriter = PromptRewriter(
        client,
        skill_dir=skill_dir,
        rules_file=rules_file,
        activation_mode=activation_mode,
        random_activation_count=random_count,
        random_activation_seed=random_seed,
    )
    result = rewriter.rewrite(row["raw_prompt"])
    return {
        **row,
        "positive_prompt": result.rewritten_prompt,
        "conditional_negative_prompt": result.negative_prompt,
        "activated_rule_ids": result.activated_rule_ids,
        "skill_revision": result.skill_revision,
        "skill_sha256": result.skill_sha256,
        "rewrite_dry_run": result.dry_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="Actual video experiment dir (no_tigger or random_trigger).")
    ap.add_argument("--cache-dir", required=True, help="Prompt rewrite cache dir (usually no_trigger or random_trigger).")
    ap.add_argument("--skill-dir", required=True)
    ap.add_argument("--rules-file", required=True)
    ap.add_argument("--activation-mode", choices=("normal", "all", "random"), required=True)
    ap.add_argument("--random-activation-count", type=int, default=None)
    ap.add_argument("--random-activation-seed", type=int, default=42)
    ap.add_argument(
        "--drop-conditional-negative",
        action="store_true",
        help="Do not pass per-rule conditional negative terms to the generator; use only FIXED_NEGATIVE.",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.activation_mode == "random" and args.random_activation_count is None:
        raise ValueError("random mode requires --random-activation-count")
    for p in (MAIN_UNIQUE, MAIN_JOBS, CONFIG, Path(args.skill_dir), Path(args.rules_file)):
        if not p.exists():
            raise FileNotFoundError(p)

    sys.path.insert(0, str(ROOT))
    from vista.utils.config import load_config

    if not os.environ.get(ROUTER_KEY_ENV, "").strip():
        key_file = os.environ.get(ROUTER_KEY_FILE_ENV, "").strip()
        if not (key_file and Path(key_file).is_file()):
            raise RuntimeError(
                f"Set {ROUTER_KEY_ENV} (or point {ROUTER_KEY_FILE_ENV} at a key file) before running."
            )
        os.environ[ROUTER_KEY_ENV] = Path(key_file).read_text(encoding="utf-8").strip()
    config = dict(load_config(CONFIG)["text_mllm"])
    config["timeout_seconds"] = min(int(config.get("timeout_seconds", 600)), 180)
    config["max_retries"] = min(int(config.get("max_retries", 5)), 2)

    unique = load_jsonl(MAIN_UNIQUE)
    source_jobs = load_jsonl(MAIN_JOBS)
    if len(unique) != 431 or len(source_jobs) != 1293:
        raise ValueError(f"expected 431 unique/1293 jobs, found {len(unique)}/{len(source_jobs)}")
    if len({r["raw_prompt"] for r in unique}) != 431:
        raise ValueError("main unique prompt list contains duplicates")

    out = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_path = cache_dir / "rewrite_records.json"
    cache: dict[str, Any] = {"schema_version": "1.0", "records": {}}
    if cache_path.is_file():
        old = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(old, dict) and isinstance(old.get("records"), dict):
            cache = old

    pending = [r for r in unique if sha256_text(r["raw_prompt"]) not in cache["records"]]
    print(f"{out.name}: total={len(unique)} cached={len(unique)-len(pending)} pending={len(pending)}", flush=True)
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                rewrite_one,
                row,
                config,
                str(Path(args.skill_dir).resolve()),
                str(Path(args.rules_file).resolve()),
                args.activation_mode,
                args.random_activation_count,
                args.random_activation_seed,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"raw_prompt": row["raw_prompt"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(pending)}] FAILED {row['prompt_index']}: {exc}", file=sys.stderr, flush=True)
                continue
            cache["records"][sha256_text(row["raw_prompt"])] = result
            atomic_json(cache_path, cache)
            print(f"[{index}/{len(pending)}] {row['prompt_index']}: {row['raw_prompt'][:70]}", flush=True)
    if failures:
        atomic_json(out / "rewrite_failures.json", {"failures": failures})
        raise RuntimeError(f"{len(failures)} rewrites failed; rerun to resume")

    records = [cache["records"][sha256_text(r["raw_prompt"])] for r in unique]
    atomic_jsonl(out / "prompts_431_unique.jsonl", records)

    # Keep signatures from the already-generated core jobs so their receipts
    # remain valid when the manifest is extended.
    old_jobs_path = out / "jobs.jsonl"
    old_jobs = load_jsonl(old_jobs_path) if old_jobs_path.is_file() else []
    method_sig = old_jobs[0].get("method_signature") if old_jobs else sha256_text(f"{out.name}:trigger-ablation")
    generator_sig = old_jobs[0].get("generator_signature") if old_jobs else sha256_text("wan22-ti2v-5B")
    arm = out.name
    rec_by_prompt = {r["raw_prompt"]: r for r in records}
    jobs: list[dict[str, Any]] = []
    for ordinal, source in enumerate(source_jobs):
        rec = rec_by_prompt[source["raw_prompt"]]
        sample_index = int(source["sample_index"])
        filename = str(source["benchmark_filename"])
        video = out / "videos" / "sana_ablation_vbench_core" / arm / filename
        receipt = out / "receipts" / "sana_ablation_vbench_core" / arm / Path(filename).with_suffix(".json")
        conditional_negative = "" if args.drop_conditional_negative else rec.get("conditional_negative_prompt", "")
        jobs.append({
            "group": "sana_ablation_vbench_core",
            "arm": arm,
            "model": "Wan2.2-TI2V-5B",
            "method": arm,
            "benchmark": "vbench",
            "prompt_set": source["prompt_set"],
            "dimensions": source["dimensions"],
            "prompt_index": source["prompt_index"],
            "raw_prompt": source["raw_prompt"],
            "positive_prompt": rec["positive_prompt"],
            "conditional_negative_prompt": conditional_negative,
            "final_negative_prompt": FIXED_NEGATIVE if args.drop_conditional_negative else ", ".join(x for x in (FIXED_NEGATIVE, conditional_negative) if x),
            "sample_index": sample_index,
            "seed": source["seed"],
            "benchmark_filename": filename,
            "source_prompt_sha256": sha256_text(source["raw_prompt"]),
            "activated_rule_ids": rec.get("activated_rule_ids", []),
            "skill_revision": rec.get("skill_revision"),
            "skill_sha256": rec.get("skill_sha256"),
            "output_relpath": f"videos/sana_ablation_vbench_core/{arm}/{filename}",
            "job_id": f"sana_ablation_vbench_core:{arm}:{source['prompt_set']}:{source['prompt_index']}:{sample_index}",
            "job_ordinal": ordinal,
            "output_path": str(video.resolve()),
            "receipt_path": str(receipt.resolve()),
            "method_signature": method_sig,
            "generator_signature": generator_sig,
            "generator": "wan22",
            "prompt_method": arm,
            "rewrite_metadata": {
                "skill_revision": rec.get("skill_revision"),
                "skill_sha256": rec.get("skill_sha256"),
                "source_prompt_sha256": sha256_text(source["raw_prompt"]),
                "activation_mode": args.activation_mode,
                "random_activation_count": args.random_activation_count,
                "random_activation_seed": args.random_activation_seed,
            },
        })
    atomic_jsonl(out / "jobs.jsonl", jobs)
    registry_path = out / "generation_registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        # New ablation arms do not have a pre-existing registry.  Materialize
        # the same Wan2.2 defaults used by the benchmark generator so the
        # manifest is immediately consumable by generate_videos.py.
        registry = {
            "schema_version": "1.0",
            "fixed_negative_prompt": FIXED_NEGATIVE,
            "generator": {
                "backend": "wan_ti2v",
                "wan_repo": "/tmp/Wan2.2_repo",
                "checkpoint": os.environ.get("WAN22_CKPT_DIR", "/path/to/Wan2.2-TI2V-5B"),
                "width": 832, "height": 480, "frames": 81, "fps": 16,
                "steps": 50, "guidance_scale": 5.0, "sample_shift": 3.0,
            },
            "arm": arm,
        }
        atomic_json(registry_path, registry)
    generator = registry.get("generator", {
        "backend": "wan_ti2v", "wan_repo": "/tmp/Wan2.2_repo",
        "checkpoint": os.environ.get("WAN22_CKPT_DIR", "/path/to/Wan2.2-TI2V-5B"),
        "width": 832, "height": 480, "frames": 81, "fps": 16,
        "steps": 50, "guidance_scale": 5.0, "sample_shift": 3.0,
    })
    atomic_json(out / "experiment.json", {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": arm, "ready": True, "groups": ["sana_ablation_vbench_core"],
        "task_count": len(jobs), "generator": "wan22", "prompt_method": arm,
        "experiment_dir": str(out), "method_signature": method_sig,
        "generator_signature": generator_sig, "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path), "jobs": str(out / "jobs.jsonl"),
        "videos_root": str(out / "videos"), "receipts_root": str(out / "receipts"),
        "generator_config": generator, "fixed_negative_prompt": FIXED_NEGATIVE,
        "source_prompt_count": len(unique), "generation_job_count": len(jobs),
        "vbench_seeds": [3407, 7859, 12491], "dimensions": sorted({d for r in unique for d in r.get("dimensions", [])}),
        "skill_dir": str(Path(args.skill_dir).resolve()), "rules_file": str(Path(args.rules_file).resolve()),
        "activation_mode": args.activation_mode, "random_activation_count": args.random_activation_count,
        "random_activation_seed": args.random_activation_seed,
        "drop_conditional_negative": args.drop_conditional_negative, "videos_generated": False,
    })
    print(json.dumps({"arm": arm, "rewrites": len(records), "jobs": len(jobs), "new_videos": len(jobs)-len(old_jobs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
