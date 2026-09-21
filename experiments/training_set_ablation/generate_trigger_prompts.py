#!/usr/bin/env python3
"""Generate VBench-core rewrites for the trigger ablation Skills.

The script uses the same PromptRewriter/DeepSeek-V4-Flash path as deployment,
but keeps one rewrite per canonical prompt and expands it to the three paired
VBench seeds.  It is resumable: completed source-prompt hashes are reused.
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
CONFIG = ROOT / "config" / "deepseek_qwen36_35ba3b_sana_posneg_train.yaml"
PROMPT_ROOT = ROOT / "experiments" / "benchmark_eval" / "data" / "vbench" / "prompts"
ROUTER_KEY_ENV = "EPICLLMROUTER_API_KEY"
ROUTER_KEY_FILE_ENV = "EPICLLMROUTER_API_KEY_FILE"
FIXED_NEGATIVE = (
    "chaotic sequence, deformed limbs, heavy motion blur, jump cuts, "
    "jerky movements, temporal artifacts, jitter, ghosting"
)
SEEDS = (3407, 7859, 12491)
PROMPT_SETS = (
    ("vbench_subject", "subject_consistency.txt", ("subject_consistency", "motion_smoothness", "dynamic_degree"), 72),
    ("vbench_overall", "overall_consistency.txt", ("overall_consistency", "aesthetic_quality", "imaging_quality"), 93),
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_router_key() -> None:
    """Populate EPICLLMROUTER_API_KEY from the env or an optional key file."""
    if os.environ.get(ROUTER_KEY_ENV, "").strip():
        return
    key_file = os.environ.get(ROUTER_KEY_FILE_ENV, "").strip()
    if key_file and Path(key_file).is_file():
        os.environ[ROUTER_KEY_ENV] = Path(key_file).read_text(encoding="utf-8").strip()
        return
    raise RuntimeError(
        f"Set {ROUTER_KEY_ENV} (or point {ROUTER_KEY_FILE_ENV} at a key file) before running."
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def load_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_set, filename, dimensions, expected in PROMPT_SETS:
        path = PROMPT_ROOT / filename
        prompts = [" ".join(line.strip().split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) != expected:
            raise ValueError(f"{path}: expected {expected}, found {len(prompts)}")
        for index, raw_prompt in enumerate(prompts, start=1):
            rows.append({
                "prompt_set": prompt_set,
                "prompt_index": index,
                "raw_prompt": raw_prompt,
                "dimensions": list(dimensions),
                "benchmark_filename": f"{raw_prompt}.mp4",
                "source_prompt_sha256": sha256_text(raw_prompt),
            })
    if len(rows) != 165 or len({r["raw_prompt"] for r in rows}) != 165:
        raise ValueError("VBench core source set must contain 165 unique prompts")
    return rows


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--rules-file", required=True)
    parser.add_argument("--activation-mode", choices=("all", "random"), required=True)
    parser.add_argument("--random-activation-count", type=int, default=None)
    parser.add_argument("--random-activation-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.activation_mode == "random" and args.random_activation_count is None:
        raise ValueError("random mode requires --random-activation-count")
    for path in (CONFIG, Path(args.skill_dir), Path(args.rules_file)):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(ROOT))
    from vista.utils.config import load_config

    _load_router_key()
    config = dict(load_config(CONFIG)["text_mllm"])
    config["timeout_seconds"] = min(int(config.get("timeout_seconds", 600)), 180)
    config["max_retries"] = min(int(config.get("max_retries", 5)), 2)
    sources = load_sources()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "rewrite_records.json"
    cache: dict[str, Any] = {"schema_version": "1.0", "records": {}}
    if cache_path.exists():
        old = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(old, dict) and isinstance(old.get("records"), dict):
            cache = old
    pending = [r for r in sources if r["source_prompt_sha256"] not in cache["records"]]
    print(f"{out.name}: source={len(sources)} cached={len(sources)-len(pending)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(rewrite_one, row, config, str(Path(args.skill_dir).resolve()), str(Path(args.rules_file).resolve()),
                        args.activation_mode, args.random_activation_count, args.random_activation_seed): row
            for row in pending
        }
        failures: list[dict[str, str]] = []
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep the batch resumable
                failures.append({"raw_prompt": row["raw_prompt"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(pending)}] FAILED {row['prompt_set']}:{row['prompt_index']}: {exc}", file=sys.stderr, flush=True)
                continue
            cache["records"][row["source_prompt_sha256"]] = result
            atomic_json(cache_path, cache)
            print(f"[{index}/{len(pending)}] {row['prompt_set']}:{row['prompt_index']}", flush=True)
    if failures:
        atomic_json(out / "rewrite_failures.json", {"failures": failures})
        raise RuntimeError(f"{len(failures)} prompt rewrites failed; rerun to resume")

    unique = [cache["records"][r["source_prompt_sha256"]] for r in sources]
    atomic_jsonl(out / "prompts_165_unique.jsonl", unique)
    jobs: list[dict[str, Any]] = []
    for row in unique:
        for sample_index, seed in enumerate(SEEDS):
            jobs.append({
                "group": "sana_ablation_vbench_core",
                "arm": out.name,
                "model": "SANA-Video_2B",
                "method": out.name,
                "benchmark": "vbench",
                "prompt_set": row["prompt_set"],
                "dimensions": row["dimensions"],
                "prompt_index": row["prompt_index"],
                "raw_prompt": row["raw_prompt"],
                "positive_prompt": row["positive_prompt"],
                "conditional_negative_prompt": row["conditional_negative_prompt"],
                "final_negative_prompt": ", ".join(x for x in (FIXED_NEGATIVE, row["conditional_negative_prompt"]) if x),
                "sample_index": sample_index,
                "seed": seed,
                "benchmark_filename": f"{row['raw_prompt']}-{sample_index}.mp4",
                "source_prompt_sha256": row["source_prompt_sha256"],
                "activated_rule_ids": row["activated_rule_ids"],
                "skill_revision": row["skill_revision"],
                "skill_sha256": row["skill_sha256"],
                "output_relpath": f"videos/sana_ablation_vbench_core/{out.name}/{row['raw_prompt']}-{sample_index}.mp4",
            })
    atomic_jsonl(out / "prompts_495.jsonl", jobs)
    atomic_jsonl(out / "jobs.jsonl", jobs)
    atomic_json(out / "experiment.json", {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": out.name,
        "method": "TIDE final ablation Skill + DeepSeek V4 Flash",
        "activation_mode": args.activation_mode,
        "random_activation_count": args.random_activation_count,
        "random_activation_seed": args.random_activation_seed,
        "source_prompt_count": len(unique),
        "generation_job_count": len(jobs),
        "vbench_seeds": list(SEEDS),
        "dimensions": sorted({d for row in unique for d in row["dimensions"]}),
        "skill_dir": str(Path(args.skill_dir).resolve()),
        "rules_file": str(Path(args.rules_file).resolve()),
        "skill_sha256": unique[0]["skill_sha256"],
        "rules_sha256": sha256_file(Path(args.rules_file)),
        "text_model": config.get("model"),
        "config": str(CONFIG),
    })
    print(f"completed {out.name}: {len(unique)} unique rewrites, {len(jobs)} VBench jobs", flush=True)


if __name__ == "__main__":
    main()
