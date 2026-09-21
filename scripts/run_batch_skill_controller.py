#!/usr/bin/env python3
"""TIDE rule-pool trainer: each wave may update one rule from four patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.clients.mllm_client import MLLMClient  # noqa: E402
from vista.core.distillation import SkillDistiller  # noqa: E402
from vista.core.waves import prompt_rows, skill_digest  # noqa: E402
from vista.utils.config import load_config  # noqa: E402

SANA_GPUS = ("0", "1", "2", "3", "4", "5", "6", "7")
SKILL_DIR = ROOT / "skills" / "rewrite-t2v-prompt"
DEFAULT_INITIAL_RULES = SKILL_DIR / "initializations" / "posneg_v1_1.json"
# The API key is read from the environment (or an optional key file) rather than a
# hard-coded path. See ``resolve_router_api_key``.
ROUTER_KEY_ENV = "EPICLLMROUTER_API_KEY"
ROUTER_KEY_FILE_ENV = "EPICLLMROUTER_API_KEY_FILE"


def resolve_router_api_key() -> str:
    """Return the router API key from the environment or an optional key file.

    Resolution order:
    1. ``EPICLLMROUTER_API_KEY`` environment variable.
    2. A file whose path is given by ``EPICLLMROUTER_API_KEY_FILE``.
    Raises when neither is available so a real run fails fast instead of calling
    the API without credentials.
    """

    key = os.environ.get(ROUTER_KEY_ENV, "").strip()
    if key:
        return key
    key_file = os.environ.get(ROUTER_KEY_FILE_ENV, "").strip()
    if key_file and Path(key_file).is_file():
        return Path(key_file).read_text(encoding="utf-8").strip()
    raise RuntimeError(
        f"Set {ROUTER_KEY_ENV} (or point {ROUTER_KEY_FILE_ENV} at a key file) "
        "before launching a non-dry-run training controller."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="training_prompts/evalcrafter_prompt700.txt")
    parser.add_argument("--config", default="config/deepseek_qwen36_35ba3b_sana_posneg_train.yaml")
    parser.add_argument("--output-root", default="outputs/evalcrafter_rulepool4_posneg")
    parser.add_argument("--wave-size", "--batch-size", dest="wave_size", type=int, default=8)
    parser.add_argument("--rule-update-size", type=int, default=4, choices=(4,))
    parser.add_argument("--max-waves", "--max-batches", dest="max_waves", type=int, default=None)
    parser.add_argument(
        "--initial-rules",
        default=str(DEFAULT_INITIAL_RULES.relative_to(ROOT)),
        help="Revision-0 rules used to initialize a run-local working Skill.",
    )
    parser.add_argument(
        "--sana-gpus",
        type=parse_sana_gpus,
        default=SANA_GPUS,
        help="Comma-separated SANA GPU IDs; each ID must be unique and in 0-7.",
    )
    parser.add_argument("--all-dry-run", action="store_true")
    parser.add_argument(
        "--activation-mode",
        choices=("normal", "all", "random"),
        default=None,
        help="Prompt-routing ablation; normal is Full skill, all bypasses triggers, random samples a fixed count.",
    )
    parser.add_argument(
        "--random-activation-count",
        type=int,
        default=None,
        help="Total active rules for random mode, including global rules; match the Full-skill mean.",
    )
    parser.add_argument("--random-activation-seed", type=int, default=None)
    parser.add_argument(
        "--freeze-negative-prompt",
        action="store_true",
        help="Keep each module's initial negative_terms fixed during distillation.",
    )
    return parser.parse_args()


def parse_sana_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {str(index) for index in range(8)}
    if not gpus:
        raise argparse.ArgumentTypeError("sana-gpus must contain at least one GPU")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("sana-gpus must not contain duplicates")
    invalid = [gpu for gpu in gpus if gpu not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError("sana-gpus must contain only GPU IDs 0-7")
    return gpus


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_working_skill(output_root: Path, initial_rules: Path) -> Path:
    """Create or validate the isolated mutable Skill used by one controller run."""
    working_skill = output_root / "working_skill"
    record_path = output_root / "skill_initialization.json"
    initial_rules = initial_rules.resolve()
    initial_value = load_json(initial_rules, {})
    if not isinstance(initial_value, dict) or int(initial_value.get("revision", -1)) != 0:
        raise ValueError("--initial-rules must contain a revision-0 rules object")
    source_sha256 = _sha256_file(initial_rules)
    record = load_json(record_path, None)
    if record is None:
        if working_skill.exists():
            raise RuntimeError("working_skill exists without skill_initialization.json; use a clean output root")
        shutil.copytree(SKILL_DIR, working_skill)
        shutil.copy2(initial_rules, working_skill / "rules.json")
        revision, digest = skill_digest(working_skill)
        if revision != 0:
            raise RuntimeError(f"Run-local Skill initialization produced revision {revision}, expected 0")
        record = {
            "schema_version": "1.0",
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "source_rules_path": str(initial_rules),
            "source_rules_sha256": source_sha256,
            "initial_revision": revision,
            "initial_skill_sha256": digest,
            "working_skill_path": str(working_skill),
            "deployed_skill_path": str(SKILL_DIR),
        }
        write_json(record_path, record)
    else:
        if not working_skill.is_dir():
            raise RuntimeError("skill_initialization.json exists but working_skill is missing")
        if record.get("source_rules_sha256") != source_sha256:
            raise RuntimeError("The requested initialization rules changed since this run was created")
        if int(record.get("initial_revision", -1)) != 0:
            raise RuntimeError("Existing run was not initialized from revision 0")
    return working_skill


def write_shard_prompts(
    batch_dir: Path,
    rows: list[dict[str, Any]],
    sana_gpus: tuple[str, ...],
) -> list[tuple[int, str, Path]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in sana_gpus]
    for index, row in enumerate(rows):
        assignments[index % len(sana_gpus)].append(row)
    jobs: list[tuple[int, str, Path]] = []
    for shard, (gpu, assigned) in enumerate(zip(sana_gpus, assignments)):
        if not assigned:
            continue
        path = batch_dir / "shard_prompts" / f"shard_{shard}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(row["prompt"] for row in assigned) + "\n", encoding="utf-8")
        write_json(path.with_suffix(".json"), assigned)
        jobs.append((shard, gpu, path))
    return jobs


def train_batch(
    batch_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
    working_skill: Path,
) -> None:
    training_started = time.perf_counter()
    frozen = batch_dir / "frozen_skill"
    if not frozen.exists():
        shutil.copytree(working_skill, frozen)
    revision, digest = skill_digest(frozen)
    manifest = load_json(batch_dir / "batch_manifest.json", {})
    configured_skill = load_config(args.config).get("skill_learning", {})
    manifest.update({
        "state": "training",
        "batch_size": len(rows),
        "skill_revision": revision,
        "skill_sha256": digest,
        "sana_gpus": list(args.sana_gpus),
        "prompt_rows": rows,
        "video_budget_per_prompt": 10,
        "layout": "raw_1+skill_init_1+optimization_4+optimization_4",
        "activation_mode": getattr(args, "activation_mode", None) or configured_skill.get("activation_mode", "normal"),
        "random_activation_count": getattr(args, "random_activation_count", None) if getattr(args, "random_activation_count", None) is not None else configured_skill.get("random_activation_count"),
        "random_activation_seed": getattr(args, "random_activation_seed", None) if getattr(args, "random_activation_seed", None) is not None else configured_skill.get("random_activation_seed", 42),
        "freeze_negative_prompt": bool(getattr(args, "freeze_negative_prompt", False) or configured_skill.get("freeze_negative_prompt", False)),
    })
    write_json(batch_dir / "batch_manifest.json", manifest)
    # Child stdout goes to regular files.  Force unbuffered Python so that
    # long Qwen critique/tournament calls remain observable while a wave runs.
    env = {**env, "PYTHONUNBUFFERED": "1"}
    processes: list[tuple[subprocess.Popen[Any], Any, Path]] = []
    for shard, gpu, prompt_file in write_shard_prompts(batch_dir, rows, args.sana_gpus):
        log_path = batch_dir / "logs" / f"shard_{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "scripts" / "run_training_corpus.py"),
            "--prompts", str(prompt_file.relative_to(ROOT)),
            "--config", args.config,
            "--skill-dir", str(frozen.relative_to(ROOT)),
            "--split", "all", "--resume",
            "--output-dir", str((batch_dir / f"shard_{shard}").relative_to(ROOT)),
            "--t2v-gpu", gpu,
        ]
        if getattr(args, "activation_mode", None) is not None:
            command += ["--activation-mode", args.activation_mode]
        if getattr(args, "random_activation_count", None) is not None:
            command += ["--random-activation-count", str(args.random_activation_count)]
        if getattr(args, "random_activation_seed", None) is not None:
            command += ["--random-activation-seed", str(args.random_activation_seed)]
        if getattr(args, "freeze_negative_prompt", False):
            command.append("--freeze-negative-prompt")
        if args.all_dry_run:
            command.append("--all-dry-run")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, handle, log_path))
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] started shard={shard} "
            f"gpu={gpu} pid={process.pid} log={log_path}",
            flush=True,
        )
    failures = []
    for process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] finished pid={process.pid} "
            f"returncode={code} log={log_path}",
            flush=True,
        )
        if code:
            failures.append({"returncode": code, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"Batch shard failures: {failures}")
    statuses = list(batch_dir.glob("shard_*/items/*/status.json"))
    completed = [load_json(path, {}) for path in statuses if load_json(path, {}).get("state") == "completed"]
    if len(completed) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} completed prompts, found {len(completed)}")
    manifest = load_json(batch_dir / "batch_manifest.json", {})
    manifest.update(
        state="trained",
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_elapsed_seconds=round(time.perf_counter() - training_started, 3),
    )
    write_json(batch_dir / "batch_manifest.json", manifest)


def collect_rule_assessments(batch_dir: Path, wave_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assessments: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("shard_*/items/*/status.json")):
        status = load_json(path, {})
        if status.get("state") == "completed" and isinstance(status.get("learning_episode"), dict):
            episode_id = str(status["learning_episode"].get("episode_id"))
            raw_assessments = status.get("rule_assessments", [])
            if isinstance(raw_assessments, list):
                assessments.extend(item for item in raw_assessments if isinstance(item, dict))
            raw_patches = status.get("candidate_patches", [])
            if not isinstance(raw_patches, list):
                raw_patches = []
            legacy_patch = status.get("candidate_patch")
            if not raw_patches and isinstance(legacy_patch, dict):
                raw_patches = [legacy_patch]
            for patch in raw_patches:
                if not isinstance(patch, dict):
                    continue
                edit = patch.get("edit", {})
                rule_id = str(edit.get("rule_id", "")) if isinstance(edit, dict) else ""
                if patch.get("status") != "candidate" or not rule_id:
                    continue
                patch_id = str(patch.get("patch_id") or f"{episode_id}:{rule_id}")
                patches.append({
                    "patch_id": patch_id,
                    "episode_id": episode_id,
                    "wave": wave_number,
                    "edit": edit,
                    "reasoning": patch.get("reasoning", ""),
                    "confidence": patch.get("confidence", 0.0),
                    "base_rule_hash": patch.get("base_rule_hash", ""),
                })
    return assessments, patches


def collect_patches(batch_dir: Path, wave_number: int) -> list[dict[str, Any]]:
    """Backward-compatible candidate-only view used by older callers."""
    return collect_rule_assessments(batch_dir, wave_number)[1]


def _new_patch_pool(current_rules: dict[str, Any]) -> dict[str, Any]:
    rule_ids = [
        str(item.get("rule_id"))
        for item in current_rules.get("modules", [])
        if isinstance(item, dict) and item.get("rule_id")
    ]
    return {
        "schema_version": "3.0",
        "queues": {rule_id: [] for rule_id in rule_ids},
        "rule_state": {
            rule_id: {
                "eligible_since_wave": None,
                "last_updated_wave": None,
                "last_processed_wave": None,
            }
            for rule_id in rule_ids
        },
        "consumed_patch_ids": [],
        "consumed_batches": [],
    }


def _load_patch_pool(pool_path: Path, current_rules: dict[str, Any]) -> dict[str, Any]:
    raw = load_json(pool_path, None)
    if not isinstance(raw, dict):
        return _new_patch_pool(current_rules)
    if isinstance(raw.get("queues"), dict):
        pool = raw
    else:
        pool = _new_patch_pool(current_rules)
        for item in raw.get("patches", []):
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("edit", {}).get("rule_id", ""))
            if rule_id:
                pool["queues"].setdefault(rule_id, []).append(item)
    pool["schema_version"] = "3.0"
    pool.setdefault("queues", {})
    pool.setdefault("rule_state", {})
    pool.setdefault("consumed_patch_ids", [])
    pool.setdefault("consumed_batches", [])
    for module in current_rules.get("modules", []):
        if not isinstance(module, dict) or not module.get("rule_id"):
            continue
        rule_id = str(module["rule_id"])
        pool["queues"].setdefault(rule_id, [])
        pool["rule_state"].setdefault(rule_id, {
            "eligible_since_wave": None,
            "last_updated_wave": None,
            "last_processed_wave": None,
        })
    return pool


def _refresh_eligibility(pool: dict[str, Any], wave_number: int, batch_size: int) -> None:
    for rule_id, queue in pool["queues"].items():
        state = pool["rule_state"].setdefault(rule_id, {})
        if len(queue) >= batch_size:
            if state.get("eligible_since_wave") is None:
                state["eligible_since_wave"] = wave_number
        else:
            state["eligible_since_wave"] = None


def _ingest_patches(
    pool: dict[str, Any],
    patches: list[dict[str, Any]],
    *,
    wave_number: int,
    batch_size: int,
) -> list[str]:
    pending_ids = {
        str(item.get("patch_id"))
        for queue in pool["queues"].values()
        for item in queue
    }
    known_ids = pending_ids | {str(value) for value in pool.get("consumed_patch_ids", [])}
    added: list[str] = []
    for patch in patches:
        patch_id = str(patch.get("patch_id", ""))
        rule_id = str(patch.get("edit", {}).get("rule_id", ""))
        if not patch_id or not rule_id or patch_id in known_ids:
            continue
        queue = pool["queues"].setdefault(rule_id, [])
        queue.append({**patch, "enqueued_wave": wave_number, "enqueue_index": len(queue)})
        pool["rule_state"].setdefault(rule_id, {
            "eligible_since_wave": None,
            "last_updated_wave": None,
            "last_processed_wave": None,
        })
        known_ids.add(patch_id)
        added.append(patch_id)
    _refresh_eligibility(pool, wave_number, batch_size)
    return added


def _select_eligible_rule(pool: dict[str, Any], batch_size: int) -> str | None:
    eligible = [rule_id for rule_id, queue in pool["queues"].items() if len(queue) >= batch_size]
    if not eligible:
        return None
    def key(rule_id: str) -> tuple[int, int, int, str]:
        state = pool["rule_state"].get(rule_id, {})
        eligible_since = state.get("eligible_since_wave")
        last_updated = state.get("last_updated_wave")
        return (
            int(eligible_since) if eligible_since is not None else 10**9,
            int(last_updated) if last_updated is not None else -1,
            -len(pool["queues"][rule_id]),
            rule_id,
        )
    return sorted(eligible, key=key)[0]


def _consume_patch_batch(
    pool: dict[str, Any],
    *,
    rule_id: str,
    patch_ids: list[str],
    wave_number: int,
    batch_size: int,
    result: dict[str, Any],
) -> None:
    already_consumed = {str(value) for value in pool.get("consumed_patch_ids", [])}
    consumed = {str(value) for value in patch_ids}
    pool["queues"][rule_id] = [
        item for item in pool["queues"].get(rule_id, [])
        if str(item.get("patch_id")) not in consumed
    ]
    for patch_id in patch_ids:
        if patch_id not in already_consumed:
            pool["consumed_patch_ids"].append(patch_id)
            already_consumed.add(patch_id)
    if not any(batch.get("patch_ids") == patch_ids for batch in pool.get("consumed_batches", [])):
        pool["consumed_batches"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wave": wave_number,
            "rule_id": rule_id,
            "patch_ids": patch_ids,
            "update_kind": result.get("update_kind", "normal_batch_update"),
            "status": result.get("status"),
            "revision": result.get("revision"),
        })
    state = pool["rule_state"].setdefault(rule_id, {})
    state["last_processed_wave"] = wave_number
    if result.get("status") == "update":
        state["last_updated_wave"] = wave_number
    state["eligible_since_wave"] = wave_number + 1 if len(pool["queues"][rule_id]) >= batch_size else None


def _distill_pool_batch(
    update_dir: Path,
    *,
    output_root: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    working_skill: Path,
    wave_number: int,
    selected_rule: str,
    chosen: list[dict[str, Any]],
    update_kind: str,
) -> dict[str, Any]:
    result_path = update_dir / "skill_update.json"
    selection_path = update_dir / "selected_rule_batch.json"
    patch_ids = [str(item["patch_id"]) for item in chosen]
    if not selection_path.exists():
        write_json(selection_path, {
            "rule_id": selected_rule,
            "update_kind": update_kind,
            "patches": chosen,
            "patch_ids": patch_ids,
        })
    current_rules = load_json(working_skill / "rules.json", {})
    pool_path = output_root / "patch_pool.json"
    pool = _load_patch_pool(pool_path, current_rules)
    if result_path.exists():
        result = load_json(result_path, {})
        _consume_patch_batch(
            pool,
            rule_id=selected_rule,
            patch_ids=patch_ids,
            wave_number=wave_number,
            batch_size=args.rule_update_size,
            result=result,
        )
        write_json(pool_path, pool)
        return result
    episode_ids = sorted(str(item["episode_id"]) for item in chosen)
    last_update = current_rules.get("last_batch_update") or {}
    if sorted(str(value) for value in last_update.get("evidence_episode_ids", [])) == episode_ids and episode_ids:
        result = {
            "schema_version": "3.0",
            "status": "update",
            "recovered_after_commit": True,
            "base_revision": int(current_rules.get("revision", 0)) - 1,
            "revision": int(current_rules.get("revision", 0)),
            "reasoning": last_update.get("reasoning", ""),
            "instruction_length": last_update.get("instruction_length"),
            "instruction_warning_chars": last_update.get("instruction_warning_chars"),
            "instruction_length_warning": last_update.get("instruction_length_warning", False),
            "evidence_episode_ids": episode_ids,
            "rule_id": selected_rule,
            "update_kind": update_kind,
        }
    else:
        config = load_config(args.config)
        text_config = config["text_mllm"]
        skill_options = config.get("skill_learning", {})
        if args.all_dry_run:
            text_config = {
                **text_config,
                "dry_run": True,
                "api_key": "",
                "api_key_required": True,
                "dry_run_when_api_key_missing": True,
            }
        old_env = os.environ.copy()
        os.environ.update(env)
        try:
            distiller = SkillDistiller(
                MLLMClient(text_config),
                working_skill,
                working_skill / "rules.json",
                freeze_negative_prompt=bool(
                    getattr(args, "freeze_negative_prompt", False)
                    or skill_options.get("freeze_negative_prompt", False)
                ),
            )
            result = distiller.distill_rule_batch(
                chosen,
                apply=not args.all_dry_run,
                audit_dir=update_dir / "batch_distillation",
                update_kind=update_kind,
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)
    write_json(result_path, result)
    pool = _load_patch_pool(pool_path, load_json(working_skill / "rules.json", {}))
    _consume_patch_batch(
        pool,
        rule_id=selected_rule,
        patch_ids=patch_ids,
        wave_number=wave_number,
        batch_size=args.rule_update_size,
        result=result,
    )
    write_json(pool_path, pool)
    return result


def update_skill(
    batch_dir: Path,
    wave_number: int,
    output_root: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    working_skill: Path,
) -> dict[str, Any]:
    update_started = time.perf_counter()
    result_path = batch_dir / "skill_update.json"
    pool_path = output_root / "patch_pool.json"
    current_rules = load_json(working_skill / "rules.json", {})
    if result_path.exists():
        result = load_json(result_path, {})
        selection = load_json(batch_dir / "selected_rule_batch.json", None)
        if isinstance(selection, dict):
            pool = _load_patch_pool(pool_path, current_rules)
            _consume_patch_batch(
                pool,
                rule_id=str(selection["rule_id"]),
                patch_ids=[str(value) for value in selection.get("patch_ids", [])],
                wave_number=wave_number,
                batch_size=args.rule_update_size,
                result=result,
            )
            write_json(pool_path, pool)
        return result
    pool = _load_patch_pool(pool_path, current_rules)
    assessments, current_patches = collect_rule_assessments(batch_dir, wave_number)
    added_patch_ids = _ingest_patches(
        pool,
        current_patches,
        wave_number=wave_number,
        batch_size=args.rule_update_size,
    )
    write_json(pool_path, pool)
    assessment_counts: dict[str, dict[str, int]] = {}
    for item in assessments:
        rule_id = str(item.get("rule_id", "unknown"))
        status = str(item.get("assessment_status") or item.get("status", "unknown"))
        assessment_counts.setdefault(rule_id, {})[status] = assessment_counts.setdefault(rule_id, {}).get(status, 0) + 1
    write_json(batch_dir / "rule_assessment_summary.json", {
        "schema_version": "3.0",
        "assessment_counts": assessment_counts,
        "added_patch_ids": added_patch_ids,
        "queue_lengths": {rule_id: len(queue) for rule_id, queue in pool["queues"].items()},
    })
    selected_rule = _select_eligible_rule(pool, args.rule_update_size)
    if selected_rule is None:
        result = {
            "status": "no_update",
            "reason": "no rule has four eligible queued patches",
            "revision": int(current_rules.get("revision", 0)),
            "queue_lengths": {rule_id: len(queue) for rule_id, queue in pool["queues"].items()},
        }
        write_json(result_path, result)
        return result
    selection_path = batch_dir / "selected_rule_batch.json"
    selected_record = load_json(selection_path, None)
    if selected_record is None:
        chosen = pool["queues"][selected_rule][: args.rule_update_size]
        selected_record = {
            "rule_id": selected_rule,
            "update_kind": "normal_batch_update",
            "patches": chosen,
            "patch_ids": [item["patch_id"] for item in chosen],
        }
        write_json(selection_path, selected_record)
    selected_rule = str(selected_record["rule_id"])
    chosen = list(selected_record["patches"])
    update = _distill_pool_batch(
        batch_dir,
        output_root=output_root,
        args=args,
        env=env,
        working_skill=working_skill,
        wave_number=wave_number,
        selected_rule=selected_rule,
        chosen=chosen,
        update_kind="normal_batch_update",
    )
    manifest = load_json(batch_dir / "batch_manifest.json", {})
    manifest.update(
        state="dry_run_validated" if args.all_dry_run else "skill_updated",
        updated_at=datetime.now(timezone.utc).isoformat(),
        selected_rule_id=selected_rule,
        consumed_patch_ids=selected_record["patch_ids"],
        new_revision=update["revision"] if not args.all_dry_run and update.get("status") == "update" else None,
        projected_revision=update["revision"],
        skill_update_elapsed_seconds=round(time.perf_counter() - update_started, 3),
    )
    write_json(batch_dir / "batch_manifest.json", manifest)
    return update


def finalize_patch_pool(
    output_root: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    working_skill: Path,
    *,
    starting_wave: int,
) -> dict[str, Any]:
    final_root = output_root / "final_updates"
    state_path = final_root / "finalization_state.json"
    state = load_json(state_path, {
        "schema_version": "3.0",
        "next_update": 0,
        "completed": False,
        "updates": [],
    })
    while not state.get("completed"):
        update_index = int(state["next_update"])
        update_dir = final_root / f"update_{update_index:04d}"
        existing_selection = load_json(update_dir / "selected_rule_batch.json", None)
        if isinstance(existing_selection, dict):
            selected_rule = str(existing_selection["rule_id"])
            chosen = list(existing_selection["patches"])
            update_kind = str(existing_selection.get("update_kind", "normal_batch_update"))
            result = _distill_pool_batch(
                update_dir,
                output_root=output_root,
                args=args,
                env=env,
                working_skill=working_skill,
                wave_number=starting_wave + update_index,
                selected_rule=selected_rule,
                chosen=chosen,
                update_kind=update_kind,
            )
            state["updates"].append({
                "update_index": update_index,
                "rule_id": selected_rule,
                "update_kind": update_kind,
                "patch_count": len(chosen),
                "patch_ids": existing_selection["patch_ids"],
                "status": result.get("status"),
                "revision": result.get("revision"),
                "recovered": True,
            })
            state["next_update"] = update_index + 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, state)
            continue
        current_rules = load_json(working_skill / "rules.json", {})
        pool_path = output_root / "patch_pool.json"
        pool = _load_patch_pool(pool_path, current_rules)
        _refresh_eligibility(pool, starting_wave + int(state["next_update"]), args.rule_update_size)
        write_json(pool_path, pool)
        selected_rule = _select_eligible_rule(pool, args.rule_update_size)
        update_kind = "normal_batch_update"
        if selected_rule is not None:
            chosen = pool["queues"][selected_rule][: args.rule_update_size]
        else:
            tail_rules = [rule_id for rule_id, queue in pool["queues"].items() if 1 <= len(queue) <= 3]
            if not tail_rules:
                state.update(
                    completed=True,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    final_revision=int(current_rules.get("revision", 0)),
                    remaining_queue_lengths={
                        rule_id: len(queue) for rule_id, queue in pool["queues"].items()
                    },
                )
                write_json(state_path, state)
                break
            def tail_key(rule_id: str) -> tuple[int, int, str]:
                rule_state = pool["rule_state"].get(rule_id, {})
                last_updated = rule_state.get("last_updated_wave")
                first_wave = min(
                    int(item.get("enqueued_wave", starting_wave))
                    for item in pool["queues"][rule_id]
                )
                return (
                    int(last_updated) if last_updated is not None else -1,
                    first_wave,
                    rule_id,
                )
            selected_rule = sorted(tail_rules, key=tail_key)[0]
            chosen = list(pool["queues"][selected_rule])
            update_kind = "final_tail_update"
        update_index = int(state["next_update"])
        update_dir = final_root / f"update_{update_index:04d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        selection_path = update_dir / "selected_rule_batch.json"
        selected_record = load_json(selection_path, None)
        if selected_record is None:
            selected_record = {
                "rule_id": selected_rule,
                "update_kind": update_kind,
                "patches": chosen,
                "patch_ids": [item["patch_id"] for item in chosen],
            }
            write_json(selection_path, selected_record)
        else:
            selected_rule = str(selected_record["rule_id"])
            chosen = list(selected_record["patches"])
            update_kind = str(selected_record.get("update_kind", update_kind))
        result = _distill_pool_batch(
            update_dir,
            output_root=output_root,
            args=args,
            env=env,
            working_skill=working_skill,
            wave_number=starting_wave + update_index,
            selected_rule=selected_rule,
            chosen=chosen,
            update_kind=update_kind,
        )
        state["updates"].append({
            "update_index": update_index,
            "rule_id": selected_rule,
            "update_kind": update_kind,
            "patch_count": len(chosen),
            "patch_ids": selected_record["patch_ids"],
            "status": result.get("status"),
            "revision": result.get("revision"),
        })
        state["next_update"] = update_index + 1
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
    return state


def main() -> None:
    args = parse_args()
    if args.wave_size <= 0:
        raise ValueError("wave-size must be positive")
    output_root = (ROOT / args.output_root).resolve()
    output_root.relative_to(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    initial_rules = _repo_path(args.initial_rules)
    working_skill = initialize_working_skill(output_root, initial_rules)
    env = os.environ.copy()
    if not args.all_dry_run:
        env["EPICLLMROUTER_API_KEY"] = resolve_router_api_key()
        config = load_config(args.config)
        vision = config.get("vision_mllm", {})
        if vision.get("provider") == "vllm":
            base_url = str(vision.get("base_url", "")).rstrip("/")
            models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
            except Exception as exc:
                raise RuntimeError(
                    f"Local Qwen3.6 endpoint is unavailable at {models_url}; "
                    "start scripts/serve_qwen36_cuda67.sh first"
                ) from exc
    env.pop("CUDA_VISIBLE_DEVICES", None)
    rows = prompt_rows((ROOT / args.prompts).resolve())
    state_path = output_root / "controller_state.json"
    state = load_json(state_path, {
        "next_wave": 0,
        "train_offset": 0,
        "completed_waves": 0,
        "finalization_complete": False,
    })
    state.setdefault("finalization_complete", False)
    working_revision, _ = skill_digest(working_skill)
    if "current_skill_revision" in state and int(state["current_skill_revision"]) != working_revision:
        raise RuntimeError(
            "controller_state current_skill_revision does not match the run-local working Skill"
        )
    while state["train_offset"] < len(rows):
        if args.max_waves is not None and state["completed_waves"] >= args.max_waves:
            break
        wave_number = int(state["next_wave"])
        batch_rows = rows[state["train_offset"]:state["train_offset"] + args.wave_size]
        batch_dir = output_root / f"wave_{wave_number:04d}"
        manifest = load_json(batch_dir / "batch_manifest.json", {})
        if "iteration_started_at" not in manifest:
            manifest["iteration_started_at"] = datetime.now(timezone.utc).isoformat()
            write_json(batch_dir / "batch_manifest.json", manifest)
        if manifest.get("state") not in {"trained", "skill_updated"}:
            train_batch(batch_dir, batch_rows, args, env, working_skill)
        update = update_skill(batch_dir, wave_number, output_root, args, env, working_skill)
        completed_at = datetime.now(timezone.utc)
        manifest = load_json(batch_dir / "batch_manifest.json", {})
        started_at = datetime.fromisoformat(manifest["iteration_started_at"])
        elapsed_seconds = round((completed_at - started_at).total_seconds(), 3)
        manifest.update(
            state="dry_run_validated" if args.all_dry_run else "completed",
            iteration_completed_at=completed_at.isoformat(),
            iteration_elapsed_seconds=elapsed_seconds,
        )
        write_json(batch_dir / "batch_manifest.json", manifest)
        state.update(
            next_wave=wave_number + 1,
            train_offset=state["train_offset"] + len(batch_rows),
            completed_waves=state["completed_waves"] + 1,
            current_skill_revision=skill_digest(working_skill)[0],
            projected_skill_revision=update["revision"],
            finalization_complete=False,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(state_path, state)
        print(
            f"[{completed_at.isoformat()}] completed wave={wave_number} "
            f"elapsed_seconds={elapsed_seconds}",
            flush=True,
        )
    remaining_prompts = len(rows) - state["train_offset"]
    if remaining_prompts == 0:
        finalization = finalize_patch_pool(
            output_root,
            args,
            env,
            working_skill,
            starting_wave=int(state["next_wave"]),
        )
        state.update(
            current_skill_revision=skill_digest(working_skill)[0],
            projected_skill_revision=skill_digest(working_skill)[0],
            finalization_complete=bool(finalization.get("completed")),
            finalization_update_count=len(finalization.get("updates", [])),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(state_path, state)
    write_json(
        output_root / "controller_summary.json",
        {**state, "remaining_prompts": remaining_prompts},
    )


if __name__ == "__main__":
    main()
