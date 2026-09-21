#!/usr/bin/env python3
"""Resumable TIDE wave controller for eight SANA GPUs and an API video judge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promote_skill_patch import promote  # noqa: E402
from vista.core.waves import (  # noqa: E402
    collect_candidates,
    is_promotion_prompt,
    prompt_rows,
    representative_candidate,
    skill_digest,
)


SANA_GPUS = ("0", "1", "2", "3", "4", "5", "6", "7")
ROUTER_KEY_ENV = "EPICLLMROUTER_API_KEY"
ROUTER_KEY_FILE_ENV = "EPICLLMROUTER_API_KEY_FILE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate/gate/promote/train waves until the train corpus is exhausted.")
    parser.add_argument("--prompts", default="training_prompts/evalcrafter_prompt700.txt")
    parser.add_argument("--config", default="config/deepseek_qwen37plus_sana_posneg_train.yaml")
    parser.add_argument("--output-root", default="outputs/evalcrafter_posneg_waves")
    parser.add_argument("--bootstrap-wave", default="outputs/evalcrafter_wave000")
    parser.add_argument("--bootstrap-consumed", type=int, default=24)
    parser.add_argument("--wave-size", type=int, default=24)
    parser.add_argument("--heldout-count", type=int, default=3)
    parser.add_argument("--promotion-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="psista-evalcrafter-v1")
    parser.add_argument("--max-waves", type=int, default=None, help="Stop after this many newly trained waves.")
    parser.add_argument("--router-key-file", default=None, help="Optional key file; EPICLLMROUTER_API_KEY is used when omitted.")
    parser.add_argument("--skip-bootstrap-gate", action="store_true", help="Train wave 1 directly under the current skill.")
    parser.add_argument("--all-dry-run", action="store_true")
    parser.add_argument("--activation-mode", choices=("normal", "all", "random"), default=None)
    parser.add_argument("--random-activation-count", type=int, default=None)
    parser.add_argument("--random-activation-seed", type=int, default=None)
    parser.add_argument("--freeze-negative-prompt", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Command failed with status {completed.returncode}; see {log_path}")


def completed_wave(wave_dir: Path, expected: int) -> bool:
    total = 0
    for shard in range(len(SANA_GPUS)):
        manifest = load_json(wave_dir / f"shard_{shard}" / "manifest.json", {})
        total += int((manifest.get("counts") or {}).get("completed", 0))
    return total == expected


def write_wave_prompt_file(wave_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = wave_dir / "prompts.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row["prompt"] for row in rows) + "\n", encoding="utf-8")
    return path


def annotate_global_indices(wave_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_hash = {row["prompt_sha256"]: row["corpus_index"] for row in rows}
    for status_path in wave_dir.glob("shard_*/items/*/status.json"):
        status = load_json(status_path, {})
        global_index = by_hash.get(status.get("prompt_sha256"))
        if global_index is not None and status.get("global_corpus_index") != global_index:
            status["global_corpus_index"] = global_index
            write_json(status_path, status)


def train_wave(wave_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, env: dict[str, str]) -> None:
    prompt_file = write_wave_prompt_file(wave_dir, rows)
    revision, digest = skill_digest(ROOT / "skills" / "rewrite-t2v-prompt")
    frozen = wave_dir / "frozen_skill"
    if not frozen.exists():
        shutil.copytree(ROOT / "skills" / "rewrite-t2v-prompt", frozen)
    write_json(
        wave_dir / "wave_manifest.json",
        {
            "skill_revision": revision,
            "skill_sha256": digest,
            "prompt_rows": rows,
            "sana_gpus": SANA_GPUS,
            "vision_backend": "api",
            "state": "training",
        },
    )
    processes: list[tuple[subprocess.Popen[Any], Any, Path]] = []
    per_shard = (len(rows) + len(SANA_GPUS) - 1) // len(SANA_GPUS)
    for shard, t2v_gpu in enumerate(SANA_GPUS):
        start = shard * per_shard
        limit = min(per_shard, max(0, len(rows) - start))
        if not limit:
            continue
        log_path = wave_dir / "logs" / f"shard_{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_training_corpus.py"),
            "--prompts", str(prompt_file.relative_to(ROOT)),
            "--config", args.config,
            "--skill-dir", str(frozen.relative_to(ROOT)),
            "--split", "all",
            "--start-index", str(start),
            "--limit", str(limit),
            "--resume",
            "--output-dir", str((wave_dir / f"shard_{shard}").relative_to(ROOT)),
            "--t2v-gpu", t2v_gpu,
        ]
        if args.all_dry_run:
            command.append("--all-dry-run")
        if args.activation_mode is not None:
            command.extend(["--activation-mode", args.activation_mode])
        if args.random_activation_count is not None:
            command.extend(["--random-activation-count", str(args.random_activation_count)])
        if args.random_activation_seed is not None:
            command.extend(["--random-activation-seed", str(args.random_activation_seed)])
        if args.freeze_negative_prompt:
            command.append("--freeze-negative-prompt")
        processes.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle, log_path))
    failures = []
    for process, handle, log_path in processes:
        status = process.wait()
        handle.close()
        if status:
            failures.append((status, str(log_path)))
    annotate_global_indices(wave_dir, rows)
    if failures:
        raise RuntimeError(f"Wave shard failures: {failures}")
    if not completed_wave(wave_dir, len(rows)):
        raise RuntimeError(f"Wave did not complete all {len(rows)} prompts")
    manifest = load_json(wave_dir / "wave_manifest.json", {})
    manifest["state"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(wave_dir / "wave_manifest.json", manifest)


def aggregate_and_gate(
    wave_dir: Path,
    wave_number: int,
    promotion_rows: list[dict[str, Any]],
    excluded_indices: set[int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    gate_dir = wave_dir / "promotion"
    gate_dir.mkdir(parents=True, exist_ok=True)
    final_report = gate_dir / "heldout_report.json"
    promotion_record = gate_dir / "promotion_result.json"
    if promotion_record.exists():
        return load_json(promotion_record, {})
    candidates = collect_candidates(wave_dir, excluded_indices)
    rules = load_json(ROOT / "skills" / "rewrite-t2v-prompt" / "rules.json", {})
    selected = representative_candidate(candidates, rules)
    write_json(gate_dir / "candidates.json", {"candidate_count": len(candidates), "candidates": candidates})
    if selected is None:
        result = {"promoted": False, "reason": "no candidate patch", "wave": wave_number}
        write_json(promotion_record, result)
        return result
    selected_path = gate_dir / "selected_patch.json"
    write_json(selected_path, selected)
    eligible = [row for row in promotion_rows if row["corpus_index"] not in excluded_indices]
    start = (wave_number * args.heldout_count) % len(eligible)
    heldout = [eligible[(start + offset) % len(eligible)] for offset in range(args.heldout_count)]
    heldout_path = gate_dir / "heldout_prompts.json"
    write_json(heldout_path, heldout)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_skill_heldout_gate.py"),
        "--patch", str(selected_path),
        "--prompts", str(heldout_path),
        "--config", args.config,
        "--output-dir", str(gate_dir.relative_to(ROOT)),
        "--t2v-gpu", "0",
        "--resume",
    ]
    if args.all_dry_run:
        command.append("--all-dry-run")
    run_logged(command, gate_dir / "gate.log", env)
    report = load_json(final_report, {})
    result: dict[str, Any] = {"promoted": False, "wave": wave_number, "heldout_report": report}
    if report.get("passed") is True and report.get("guard_noninferior") is True:
        promoted = promote(
            selected["patch"],
            report,
            ROOT / "skills" / "rewrite-t2v-prompt",
            freeze_negative_prompt=args.freeze_negative_prompt,
        )
        result.update(promoted)
    else:
        result["reason"] = "held-out gate did not pass"
    write_json(promotion_record, result)
    return result


def main() -> None:
    args = parse_args()
    if args.wave_size <= 0 or args.wave_size % len(SANA_GPUS):
        raise ValueError("wave-size must be positive and divisible by eight")
    output_root = (ROOT / args.output_root).resolve()
    output_root.relative_to(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not args.all_dry_run:
        if args.router_key_file:
            env[ROUTER_KEY_ENV] = Path(args.router_key_file).expanduser().read_text(encoding="utf-8").strip()
        elif os.environ.get(ROUTER_KEY_ENV, "").strip():
            env[ROUTER_KEY_ENV] = os.environ[ROUTER_KEY_ENV].strip()
        elif os.environ.get(ROUTER_KEY_FILE_ENV, "").strip():
            env[ROUTER_KEY_ENV] = Path(os.environ[ROUTER_KEY_FILE_ENV]).read_text(encoding="utf-8").strip()
        else:
            raise RuntimeError(
                f"Set {ROUTER_KEY_ENV} (or pass --router-key-file) before a non-dry-run wave controller run."
            )
    # Each child launches exactly one SANA worker on its explicit GPU. Vision evaluation
    # is remote and must never consume a local CUDA device.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    rows = prompt_rows((ROOT / args.prompts).resolve())
    promotion_rows = [row for row in rows if is_promotion_prompt(row, args.promotion_fraction, args.split_seed)]
    promotion_indices = {row["corpus_index"] for row in promotion_rows}
    bootstrap_indices = set(range(args.bootstrap_consumed))
    train_rows = [row for row in rows if row["corpus_index"] not in promotion_indices and row["corpus_index"] not in bootstrap_indices]
    state_path = output_root / "controller_state.json"
    state = load_json(state_path, {"next_wave": 1, "train_offset": 0, "trained_new_waves": 0})
    previous_wave = (ROOT / args.bootstrap_wave).resolve() if state["next_wave"] == 1 else output_root / f"wave_{state['next_wave'] - 1:03d}"
    while state["train_offset"] < len(train_rows):
        wave_number = int(state["next_wave"])
        if not (wave_number == 1 and args.skip_bootstrap_gate):
            aggregate_and_gate(
                previous_wave,
                wave_number - 1,
                promotion_rows,
                (promotion_indices & bootstrap_indices) if wave_number == 1 else set(),
                args,
                env,
            )
        batch = train_rows[state["train_offset"] : state["train_offset"] + args.wave_size]
        wave_dir = output_root / f"wave_{wave_number:03d}"
        train_wave(wave_dir, batch, args, env)
        state.update(
            next_wave=wave_number + 1,
            train_offset=state["train_offset"] + len(batch),
            trained_new_waves=state["trained_new_waves"] + 1,
            last_completed_wave=wave_number,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(state_path, state)
        previous_wave = wave_dir
        if args.max_waves is not None and state["trained_new_waves"] >= args.max_waves:
            break
    write_json(output_root / "controller_summary.json", {**state, "remaining_train_prompts": len(train_rows) - state["train_offset"]})


if __name__ == "__main__":
    main()
