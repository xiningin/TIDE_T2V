#!/usr/bin/env python3
"""Run resumable TIDE search/distillation over a fixed prompt corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# API key is resolved from the environment; an optional key file may be passed
# with --router-key-file for setups that keep the key outside the shell env.
ROUTER_KEY_ENV = "EPICLLMROUTER_API_KEY"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.core.agent import VistaAgent  # noqa: E402
from vista.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TIDE Skill on a resumable prompt corpus.")
    parser.add_argument("--prompts", default="training_prompts/evalcrafter_prompt700.txt")
    parser.add_argument("--config", default="config/deepseek_qwen_sana_train.yaml")
    parser.add_argument("--output-dir", default="outputs/evalcrafter_training")
    parser.add_argument("--skill-dir", default=None, help="Frozen rewrite skill directory for this shard.")
    parser.add_argument("--split", choices=("train", "promotion", "all"), default="all")
    parser.add_argument("--promotion-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="psista-evalcrafter-v1")
    parser.add_argument("--start-index", type=int, default=0, help="Index within the selected split.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--data-inspection-retries",
        type=int,
        default=2,
        help=(
            "Regenerate a prompt with fresh T2V seeds this many times when the "
            "vision provider rejects generated frames with data_inspection_failed."
        ),
    )
    parser.add_argument("--t2v-dry-run", action="store_true")
    parser.add_argument("--all-dry-run", action="store_true", help="Disable all external/model inference for a framework smoke test.")
    parser.add_argument(
        "--router-key-file",
        "--deepseek-key-file",
        dest="router_key_file",
        default=None,
        help="Optional EpicLLMRouter key file; if omitted the EPICLLMROUTER_API_KEY environment variable is used.",
    )
    parser.add_argument("--vision-gpu", default=None, help="Override local Qwen CUDA device, restricted operationally to GPU 0-7.")
    parser.add_argument("--t2v-gpu", default=None, help="Override SANA CUDA device, restricted operationally to GPU 0-7.")
    parser.add_argument(
        "--activation-mode",
        choices=("normal", "all", "random"),
        default=None,
        help="Prompt-routing ablation mode.",
    )
    parser.add_argument(
        "--random-activation-count",
        type=int,
        default=None,
        help="Total active rules for random mode, including global rules.",
    )
    parser.add_argument("--random-activation-seed", type=int, default=None)
    parser.add_argument(
        "--freeze-negative-prompt",
        action="store_true",
        help="Keep module negative_terms fixed during Skill distillation.",
    )
    parser.add_argument("--vision-api-base-url", default=None, help="Use an OpenAI-compatible Qwen multimodal API instead of local Qwen.")
    parser.add_argument("--vision-api-model", default=None)
    parser.add_argument("--vision-api-key-env", default="EPICLLMROUTER_API_KEY")
    return parser.parse_args()


def _inside_repo(value: str) -> Path:
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        prompt = " ".join(raw.strip().split())
        if not prompt:
            continue
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        rows.append({"corpus_index": len(rows), "line_number": line_number, "prompt": prompt, "prompt_sha256": digest})
    return rows


def _partition(rows: list[dict[str, Any]], split: str, fraction: float, seed: str) -> list[dict[str, Any]]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("promotion-fraction must be in [0, 1)")
    if split == "all":
        return rows
    cutoff = int(fraction * 10_000)
    selected = []
    for row in rows:
        bucket = int(hashlib.sha256(f"{seed}\0{row['prompt']}".encode()).hexdigest()[:8], 16) % 10_000
        is_promotion = bucket < cutoff
        if (split == "promotion") == is_promotion:
            selected.append(row)
    return selected


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _is_data_inspection_failure(value: object) -> bool:
    """Return whether a provider rejected generated visual input during inspection."""
    normalized = str(value).replace("_", "").lower()
    return "datainspectionfailed" in normalized


def _attempt_run_dir(item_dir: Path, attempt_index: int) -> Path:
    """Keep rejected generations auditable instead of overwriting their evidence."""
    if attempt_index == 0:
        return item_dir / "run"
    return item_dir / f"run_retry_{attempt_index:02d}"


def _prior_inspection_attempts(old: dict[str, Any] | None) -> tuple[int, list[dict[str, Any]]]:
    """Resume retry numbering and retain the error that triggered controller recovery."""
    if not isinstance(old, dict):
        return 0, []
    history = [dict(item) for item in old.get("attempt_errors", []) if isinstance(item, dict)]
    error = str(old.get("error", ""))
    if old.get("state") not in {"failed", "retrying"} or not _is_data_inspection_failure(error):
        return 0, history
    attempt_index = max(0, int(old.get("attempt_index", 0)))
    if not history or str(history[-1].get("message", "")) != error:
        history.append(
            {
                "attempt_index": attempt_index,
                "type": "vision_data_inspection_failed",
                "message": error,
            }
        )
    return attempt_index + 1, history


def _resume_completed_search_distillation(
    agent: VistaAgent,
    run_dir: Path,
    *,
    expected_prompt: str,
) -> dict[str, Any] | None:
    """Finish only Skill distillation when the expensive video search is complete.

    ``VistaAgent.run`` writes ``best_video`` immediately before trajectory
    distillation.  Consequently, a trajectory containing ``best_video`` but no
    ``rule_assessments`` is a durable checkpoint for a failure in the final text
    MLLM call.  Reusing it prevents a transient provider outage from regenerating
    all ten videos for the prompt.
    """
    trajectory_path = run_dir / "trajectory.json"
    if not trajectory_path.is_file():
        return None
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(trajectory.get("user_prompt", "")) != expected_prompt:
        return None
    best_video = trajectory.get("best_video")
    if not isinstance(best_video, dict) or not best_video.get("video_path"):
        return None
    if not isinstance(best_video.get("prompt_candidate"), dict):
        return None
    if trajectory.get("rule_assessments") is not None:
        return trajectory
    if not agent._skill_learning_enabled():
        return None

    stats = trajectory.setdefault("stats", {"phases": []})
    stats.setdefault("phases", [])
    phase_started, mllm_before, t2v_before = agent._phase_start()
    episode = trajectory.get("skill_learning_episode")
    if not isinstance(episode, dict):
        episode = agent._build_batch_skill_episode(trajectory)
    initial_rewrite = trajectory.get("initial_skill_rewrite", {})
    activated_rule_ids = (
        list(initial_rewrite.get("activated_rule_ids", []))
        if isinstance(initial_rewrite, dict)
        else []
    )
    rule_assessments = agent.distiller.propose_trajectory_rule_assessments(
        episode,
        activated_rule_ids,
    )
    candidate_patches = [
        item for item in rule_assessments if item.get("status") == "candidate"
    ]
    trajectory["skill_learning_episode"] = episode
    trajectory["activated_rule_ids"] = activated_rule_ids
    trajectory["rule_assessments"] = rule_assessments
    trajectory["candidate_patches"] = candidate_patches
    agent._record_phase(
        stats,
        "trajectory_distillation",
        phase_started,
        mllm_before,
        t2v_before,
        {"resumed_from_completed_search": True},
    )
    stats["resumed_from_completed_search"] = True
    agent._write_trajectory(run_dir, trajectory)
    return trajectory


def _collect_summary(run_root: Path, metadata: dict[str, Any]) -> None:
    records = []
    for status_path in sorted((run_root / "items").glob("*/status.json")):
        records.append(json.loads(status_path.read_text(encoding="utf-8")))
    counts: dict[str, int] = {}
    for record in records:
        state = str(record.get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    _write_json(run_root / "manifest.json", {**metadata, "counts": counts, "items": records})


def main() -> None:
    args = parse_args()
    if args.data_inspection_retries < 0:
        raise ValueError("data-inspection-retries must be non-negative")
    prompt_path = _inside_repo(args.prompts)
    run_root = _inside_repo(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "items").mkdir(exist_ok=True)
    if not args.all_dry_run:
        if args.router_key_file:
            key_path = Path(args.router_key_file).expanduser()
            os.environ[ROUTER_KEY_ENV] = key_path.read_text(encoding="utf-8").strip()
        elif not os.environ.get(ROUTER_KEY_ENV, "").strip():
            raise RuntimeError(
                f"Set {ROUTER_KEY_ENV} (or pass --router-key-file) before a non-dry-run training shard."
            )

    all_rows = _load_prompts(prompt_path)
    selected = _partition(all_rows, args.split, args.promotion_fraction, args.split_seed)
    selected = selected[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    corpus_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(prompt_path),
        "corpus_sha256": corpus_sha,
        "nonempty_prompt_count": len(all_rows),
        "split": args.split,
        "split_seed": args.split_seed,
        "promotion_fraction": args.promotion_fraction,
        "selected_count": len(selected),
        "config_path": str(_inside_repo(args.config)),
        "activation_mode": args.activation_mode,
        "random_activation_count": args.random_activation_count,
        "random_activation_seed": args.random_activation_seed,
        "freeze_negative_prompt": bool(args.freeze_negative_prompt),
    }
    _collect_summary(run_root, metadata)

    config = load_config(args.config)
    if args.skill_dir:
        frozen_skill = _inside_repo(args.skill_dir)
        config["skill_learning"]["skill_dir"] = str(frozen_skill)
        config["skill_learning"]["rules_file"] = str(frozen_skill / "rules.json")
    if args.activation_mode is not None:
        config["skill_learning"]["activation_mode"] = args.activation_mode
    if args.random_activation_count is not None:
        config["skill_learning"]["random_activation_count"] = args.random_activation_count
    if args.random_activation_seed is not None:
        config["skill_learning"]["random_activation_seed"] = args.random_activation_seed
    if args.freeze_negative_prompt:
        config["skill_learning"]["freeze_negative_prompt"] = True
    allowed_gpus = {str(index) for index in range(8)}
    if args.vision_api_base_url:
        if not args.vision_api_model:
            raise ValueError("vision-api-model is required with vision-api-base-url")
        vision = config["vision_mllm"]
        vision.update(
            provider="openai_compatible",
            base_url=args.vision_api_base_url,
            model=args.vision_api_model,
            api_key_env=args.vision_api_key_env,
            api_key_required=True,
            dry_run_when_api_key_missing=False,
        )
        for local_key in ("model_path", "cuda_visible_devices", "device_map", "torch_dtype", "enable_thinking", "preserve_thinking"):
            vision.pop(local_key, None)
    if args.vision_gpu is not None:
        if args.vision_api_base_url:
            raise ValueError("vision-gpu cannot be combined with vision-api-base-url")
        if args.vision_gpu not in allowed_gpus:
            raise ValueError("vision-gpu must be one of 0,1,2,3,4,5,6,7")
        config["vision_mllm"]["cuda_visible_devices"] = args.vision_gpu
    if args.t2v_gpu is not None:
        if args.t2v_gpu not in allowed_gpus:
            raise ValueError("t2v-gpu must be one of 0,1,2,3,4,5,6,7")
        config["t2v"]["cuda_visible_devices"] = args.t2v_gpu
    if args.t2v_dry_run:
        config["t2v"]["dry_run"] = True
    if args.all_dry_run:
        config["t2v"]["dry_run"] = True
        for role in ("text_mllm", "vision_mllm"):
            config[role]["dry_run"] = True
            config[role]["api_key_required"] = True
            config[role]["api_key"] = ""
            config[role]["dry_run_when_api_key_missing"] = True
    agent = VistaAgent(config)
    try:
        for split_index, row in enumerate(selected, start=args.start_index):
            item_id = f"p{row['corpus_index']:04d}_{row['prompt_sha256'][:10]}"
            item_dir = run_root / "items" / item_id
            status_path = item_dir / "status.json"
            old_status: dict[str, Any] | None = None
            if args.resume and status_path.is_file():
                old_status = json.loads(status_path.read_text(encoding="utf-8"))
                if old_status.get("state") == "completed":
                    continue
            item_dir.mkdir(parents=True, exist_ok=True)
            first_attempt_index, attempt_errors = _prior_inspection_attempts(old_status)
            status = {**row, "item_id": item_id, "split_index": split_index}
            succeeded = False
            for local_retry in range(args.data_inspection_retries + 1):
                attempt_index = first_attempt_index + local_retry
                attempt_dir = _attempt_run_dir(item_dir, attempt_index)
                status.update(
                    state="running",
                    attempt_index=attempt_index,
                    attempt_number=attempt_index + 1,
                    active_run_dir=str(attempt_dir),
                    data_inspection_retry_count=len(attempt_errors),
                    attempt_errors=attempt_errors,
                )
                status.pop("error", None)
                _write_json(status_path, status)
                _collect_summary(run_root, metadata)
                agent.reset_telemetry()
                try:
                    trajectory = None
                    if args.resume and old_status is not None:
                        trajectory = _resume_completed_search_distillation(
                            agent,
                            attempt_dir,
                            expected_prompt=row["prompt"],
                        )
                    if trajectory is None:
                        result = agent.run(row["prompt"], output_dir=attempt_dir)
                        trajectory_path = Path(result.trajectory_path)
                        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                    else:
                        trajectory_path = attempt_dir / "trajectory.json"
                        print(
                            f"[{datetime.now(timezone.utc).isoformat()}] item={item_id} "
                            "resumed completed video search; retried Skill distillation only",
                            flush=True,
                        )
                    episode = trajectory.get("skill_learning_episode")
                    rule_assessments = trajectory.get("rule_assessments", [])
                    candidate_patches = trajectory.get("candidate_patches", [])
                    best_video = trajectory.get("best_video", {})
                    best_prompt = best_video.get("prompt_candidate", {})
                    status.update(
                        state="completed",
                        trajectory_path=str(trajectory_path),
                        best_prompt=str(best_prompt.get("prompt", "")),
                        best_video_path=str(best_video.get("video_path", "")),
                        learned_episode_count=1 if episode else 0,
                        learning_episode=episode,
                        activated_rule_ids=trajectory.get("activated_rule_ids", []),
                        rule_assessments=rule_assessments,
                        candidate_patches=candidate_patches,
                        learned=[],
                        data_inspection_retry_count=len(attempt_errors),
                        attempt_errors=attempt_errors,
                    )
                    succeeded = True
                    break
                except Exception as exc:  # Continue corpus training unless explicitly requested otherwise.
                    error = f"{type(exc).__name__}: {exc}"
                    retryable = _is_data_inspection_failure(exc)
                    if retryable:
                        attempt_errors.append(
                            {
                                "attempt_index": attempt_index,
                                "type": "vision_data_inspection_failed",
                                "message": error,
                            }
                        )
                    status.update(
                        state="retrying" if retryable and local_retry < args.data_inspection_retries else "failed",
                        error=error,
                        data_inspection_retry_count=len(attempt_errors),
                        attempt_errors=attempt_errors,
                    )
                    _write_json(status_path, status)
                    _collect_summary(run_root, metadata)
                    if retryable and local_retry < args.data_inspection_retries:
                        print(
                            f"[{datetime.now(timezone.utc).isoformat()}] item={item_id} "
                            f"vision data inspection rejected attempt={attempt_index}; "
                            f"regenerating with fresh T2V seeds",
                            flush=True,
                        )
                        continue
                    if args.fail_fast:
                        raise
                    break
            if not succeeded:
                continue
            _write_json(status_path, status)
            _collect_summary(run_root, metadata)
    finally:
        close = getattr(agent.t2v, "close", None)
        if callable(close):
            close()
    _collect_summary(run_root, metadata)


if __name__ == "__main__":
    main()
