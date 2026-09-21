#!/usr/bin/env python3
"""Keep a resumable TIDE batch controller running until its corpus completes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--initial-rules", required=True)
    parser.add_argument("--sana-gpus", default="6,7")
    parser.add_argument("--wave-size", type=int, default=8)
    parser.add_argument("--rule-update-size", type=int, default=4, choices=(4,))
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=900.0)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=0,
        help="Stop after this many no-progress failures; 0 retries until the corpus completes.",
    )
    parser.add_argument("--all-dry-run", action="store_true")
    parser.add_argument("--activation-mode", choices=("normal", "all", "random"), default=None)
    parser.add_argument("--random-activation-count", type=int, default=None)
    parser.add_argument("--random-activation-seed", type=int, default=None)
    parser.add_argument("--freeze-negative-prompt", action="store_true")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def repo_path(value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def prompt_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def controller_progress(output_root: Path) -> int:
    state = load_json(output_root / "controller_state.json", {})
    if not isinstance(state, dict):
        return 0
    try:
        return max(0, int(state.get("train_offset", 0)))
    except (TypeError, ValueError):
        return 0


def training_complete(output_root: Path, total_prompts: int) -> bool:
    if controller_progress(output_root) < total_prompts:
        return False
    state = load_json(output_root / "controller_state.json", {})
    if isinstance(state, dict) and "finalization_complete" in state:
        return bool(state.get("finalization_complete"))
    return True


def retry_delay(consecutive_failures: int, *, base_seconds: float, max_seconds: float) -> float:
    exponent = max(0, consecutive_failures - 1)
    return min(max_seconds, base_seconds * (2**exponent))


def controller_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_batch_skill_controller.py"),
        "--prompts",
        args.prompts,
        "--config",
        args.config,
        "--output-root",
        args.output_root,
        "--initial-rules",
        args.initial_rules,
        "--sana-gpus",
        args.sana_gpus,
        "--wave-size",
        str(args.wave_size),
        "--rule-update-size",
        str(args.rule_update_size),
    ]
    if args.max_waves is not None:
        command.extend(["--max-waves", str(args.max_waves)])
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
    return command


def main() -> int:
    args = parse_args()
    if args.wave_size <= 0:
        raise ValueError("wave-size must be positive")
    if args.retry_delay_seconds < 0 or args.max_retry_delay_seconds < args.retry_delay_seconds:
        raise ValueError("retry delays must satisfy 0 <= retry-delay <= max-retry-delay")
    if args.max_consecutive_failures < 0:
        raise ValueError("max-consecutive-failures must be non-negative")

    prompts_path = repo_path(args.prompts)
    repo_path(args.config)
    repo_path(args.initial_rules)
    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total_prompts = prompt_count(prompts_path)
    if total_prompts == 0:
        raise ValueError("prompt corpus is empty")

    lock_path = output_root / ".controller_supervisor.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"A supervisor already owns {output_root}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    state_path = output_root / "supervisor_state.json"
    controller_log_path = output_root / "controller.log"
    supervisor_started = utc_now()
    attempt = 0
    consecutive_failures = 0
    previous_progress = controller_progress(output_root)
    child: subprocess.Popen[Any] | None = None
    stopping = False

    def write_state(status: str, **extra: Any) -> None:
        state = {
            "schema_version": "1.0",
            "status": status,
            "supervisor_pid": os.getpid(),
            "supervisor_started_at": supervisor_started.isoformat(),
            "updated_at": utc_now().isoformat(),
            "attempt": attempt,
            "consecutive_failures": consecutive_failures,
            "train_offset": controller_progress(output_root),
            "total_prompts": total_prompts,
            "controller_log": str(controller_log_path),
            **extra,
        }
        write_json(state_path, state)

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        print(f"[{utc_now().isoformat()}] supervisor received signal={signum}", flush=True)
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        while not stopping:
            if training_complete(output_root, total_prompts):
                write_state("completed", completed_at=utc_now().isoformat())
                print(f"[{utc_now().isoformat()}] corpus complete prompts={total_prompts}", flush=True)
                return 0

            attempt += 1
            started_at = utc_now()
            command = controller_command(args)
            write_state("running", controller_started_at=started_at.isoformat(), command=command)
            print(
                f"[{started_at.isoformat()}] starting controller attempt={attempt} "
                f"progress={previous_progress}/{total_prompts}",
                flush=True,
            )
            with controller_log_path.open("a", encoding="utf-8") as controller_log:
                controller_log.write(
                    f"\n[{started_at.isoformat()}] supervisor starting controller attempt={attempt}\n"
                )
                controller_log.flush()
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=controller_log,
                    stderr=subprocess.STDOUT,
                )
                write_state(
                    "running",
                    controller_started_at=started_at.isoformat(),
                    controller_pid=child.pid,
                    command=command,
                )
                returncode = child.wait()
                child = None

            finished_at = utc_now()
            progress = controller_progress(output_root)
            if training_complete(output_root, total_prompts):
                write_state(
                    "completed",
                    completed_at=finished_at.isoformat(),
                    controller_returncode=returncode,
                )
                print(f"[{finished_at.isoformat()}] corpus complete prompts={total_prompts}", flush=True)
                return 0
            if stopping:
                break

            if progress > previous_progress:
                consecutive_failures = 1
            else:
                consecutive_failures += 1
            previous_progress = progress
            if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                write_state(
                    "failed",
                    controller_returncode=returncode,
                    failure_reason="max_consecutive_failures_reached",
                )
                return returncode or 1

            delay = retry_delay(
                consecutive_failures,
                base_seconds=args.retry_delay_seconds,
                max_seconds=args.max_retry_delay_seconds,
            )
            retry_at = finished_at + timedelta(seconds=delay)
            write_state(
                "retry_wait",
                controller_finished_at=finished_at.isoformat(),
                controller_returncode=returncode,
                retry_delay_seconds=delay,
                next_retry_at=retry_at.isoformat(),
            )
            print(
                f"[{finished_at.isoformat()}] controller exited returncode={returncode} "
                f"progress={progress}/{total_prompts}; retrying in {delay:.1f}s",
                flush=True,
            )
            deadline = time.monotonic() + delay
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if stopping:
            write_state("stopped", stopped_at=utc_now().isoformat())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
