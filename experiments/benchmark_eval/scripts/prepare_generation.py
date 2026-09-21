#!/usr/bin/env python3
"""Prepare one benchmark arm for generation without running the video model."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generation_common import (  # noqa: E402
    DEFAULT_REGISTRY,
    WORKSPACE,
    arm_definition,
    atomic_json,
    atomic_jsonl,
    ensure_router_key,
    experiment_dir,
    manifest_path,
    merge_negative,
    output_path_for_task,
    read_json,
    read_jsonl,
    receipt_path_for_task,
    resolve_from_registry,
    sha256_file,
    sha256_text,
    stable_job_id,
)


VPO_TEMPLATE = """In this task, your goal is to expand the user's short query into a detailed and well-structured English prompt for generating short videos.

Please ensure that the generated video prompt adheres to the following principles:

1. **Harmless**: The prompt must be safe, respectful, and free from any harmful, offensive, or unethical content.
2. **Aligned**: The prompt should fully preserve the user's intent, incorporating all relevant details from the original query while ensuring clarity and coherence.
3. **Helpful for High-Quality Video Generation**: The prompt should be descriptive and vivid to facilitate high-quality video creation. Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query.

User Query:{}

Video Prompt:"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", required=True)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--rewrite-gpu", default="0")
    result.add_argument("--max-tasks", type=int, default=0)
    result.add_argument("--validate-only", action="store_true")
    return result


def _method_signature(
    registry_path: Path, method_name: str, method: dict[str, Any]
) -> str:
    payload: dict[str, Any] = {"name": method_name, "config": method}
    method_type = method.get("type")
    if method_type == "skill":
        skill_dir = resolve_from_registry(registry_path, str(method["skill_dir"]))
        rules_file = resolve_from_registry(registry_path, str(method["rules_file"]))
        rewrite_config = resolve_from_registry(
            registry_path, str(method["rewrite_config"])
        )
        payload["skill_md_sha256"] = sha256_file(skill_dir / "SKILL.md")
        payload["rules_sha256"] = sha256_file(rules_file)
        payload["rewrite_config_sha256"] = sha256_file(rewrite_config)
    elif method_type == "vpo":
        checkpoint = resolve_from_registry(registry_path, str(method["checkpoint"]))
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"missing VPO checkpoint: {checkpoint}")
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
            path = checkpoint / name
            if path.is_file():
                payload[f"{name}_sha256"] = sha256_file(path)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _rewrite_signature(registry_path: Path, method: dict[str, Any]) -> str:
    """Signature for calls that must produce one shared rewrite across arms."""
    cache_method = dict(method)
    # Dropping the negative is a downstream ablation, not a separate MLLM call.
    cache_method.pop("drop_conditional_negative", None)
    payload: dict[str, Any] = {"config": cache_method}
    if method.get("type") == "skill":
        skill_dir = resolve_from_registry(registry_path, str(method["skill_dir"]))
        rules_file = resolve_from_registry(registry_path, str(method["rules_file"]))
        rewrite_config = resolve_from_registry(
            registry_path, str(method["rewrite_config"])
        )
        payload["skill_md_sha256"] = sha256_file(skill_dir / "SKILL.md")
        payload["rules_sha256"] = sha256_file(rules_file)
        payload["rewrite_config_sha256"] = sha256_file(rewrite_config)
    elif method.get("type") == "vpo":
        checkpoint = resolve_from_registry(registry_path, str(method["checkpoint"]))
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ):
            path = checkpoint / name
            if path.is_file():
                payload[f"{name}_sha256"] = sha256_file(path)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _generator_signature(
    registry_path: Path, generator_name: str, generator: dict[str, Any]
) -> str:
    payload: dict[str, Any] = {"name": generator_name, "config": generator}
    if generator.get("config"):
        config_path = resolve_from_registry(registry_path, str(generator["config"]))
        if not config_path.is_file():
            raise FileNotFoundError(f"missing generator config: {config_path}")
        payload["config_sha256"] = sha256_file(config_path)
    checkpoint = resolve_from_registry(registry_path, str(generator["checkpoint"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing generator checkpoint: {checkpoint}")
    for name in ("model_index.json", "config.json", "configuration.json"):
        path = checkpoint / name
        if path.is_file():
            payload[f"{name}_sha256"] = sha256_file(path)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load_arm_tasks(arm: str, groups: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks: list[dict[str, Any]] = []
    manifests: dict[str, str] = {}
    for group in groups:
        path = manifest_path(group)
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest: {path}")
        manifests[group] = sha256_file(path)
        tasks.extend(row for row in read_jsonl(path) if row.get("arm") == arm)
    if not tasks:
        raise ValueError(f"no tasks found for arm {arm} in groups {groups}")
    ids = [stable_job_id(row) for row in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids for arm {arm}")
    return tasks, manifests


def _cache_key(prompt: str) -> str:
    return sha256_text(prompt)


def _skill_rewriter(
    registry_path: Path,
    method: dict[str, Any],
    workers: int,
) -> Callable[[list[str], dict[str, Any], Path], dict[str, Any]]:
    ensure_router_key(PROJECT_ROOT)
    rewrite_config_path = resolve_from_registry(registry_path, str(method["rewrite_config"]))
    skill_dir = resolve_from_registry(registry_path, str(method["skill_dir"]))
    rules_file = resolve_from_registry(registry_path, str(method["rules_file"]))
    def run(prompts: list[str], cache: dict[str, Any], cache_path: Path) -> dict[str, Any]:
        from vista.clients.mllm_client import MLLMClient
        from vista.core.prompt_rewriter import PromptRewriter
        from vista.utils.config import load_config

        config = load_config(rewrite_config_path)["text_mllm"]

        def rewrite(prompt: str) -> tuple[str, dict[str, Any]]:
            client = MLLMClient(config)
            result = PromptRewriter(client, skill_dir, rules_file).rewrite(prompt)
            return _cache_key(prompt), {
                "raw_prompt": prompt,
                "positive_prompt": result.rewritten_prompt,
                "conditional_negative_prompt": result.negative_prompt,
                "skill_revision": result.skill_revision,
                "skill_sha256": result.skill_sha256,
            }

        pending = [prompt for prompt in prompts if _cache_key(prompt) not in cache["records"]]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(rewrite, prompt) for prompt in pending]
            for future in as_completed(futures):
                key, record = future.result()
                cache["records"][key] = record
                atomic_json(cache_path, cache)
        return cache

    return run


def _vpo_rewriter(
    registry_path: Path,
    method: dict[str, Any],
    rewrite_gpu: str,
) -> Callable[[list[str], dict[str, Any], Path], dict[str, Any]]:
    checkpoint = resolve_from_registry(registry_path, str(method["checkpoint"]))
    seed = int(method.get("rewrite_seed", 42))
    max_new_tokens = int(method.get("max_new_tokens", 1024))
    temperature = float(method.get("temperature", 0.7))
    top_p = float(method.get("top_p", 1.0))

    def run(prompts: list[str], cache: dict[str, Any], cache_path: Path) -> dict[str, Any]:
        pending = [prompt for prompt in prompts if _cache_key(prompt) not in cache["records"]]
        if not pending:
            return cache
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rewrite_gpu)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("VPO prompt generation requires CUDA")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval().to("cuda")
        model.config.use_cache = True
        try:
            for index, prompt in enumerate(pending, start=1):
                prompt_seed = seed + int(_cache_key(prompt)[:8], 16)
                torch.manual_seed(prompt_seed)
                torch.cuda.manual_seed_all(prompt_seed)
                messages = [{"role": "user", "content": VPO_TEMPLATE.format(prompt)}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        top_p=top_p,
                        temperature=temperature,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    output[0, input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
                if not response:
                    raise RuntimeError(f"empty VPO rewrite for prompt: {prompt}")
                key = _cache_key(prompt)
                cache["records"][key] = {
                    "raw_prompt": prompt,
                    "positive_prompt": response,
                    "conditional_negative_prompt": "",
                    "rewrite_seed": prompt_seed,
                    "checkpoint": str(checkpoint),
                }
                atomic_json(cache_path, cache)
                print(f"[rewrite {index}/{len(pending)}] {key[:12]}", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
        return cache

    return run


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1 or args.max_tasks < 0:
        raise ValueError("workers must be positive and max-tasks must be non-negative")
    registry_path, registry, arm = arm_definition(args.arm, args.registry)
    groups = [str(value) for value in arm.get("groups", [])]
    tasks, manifests = _load_arm_tasks(args.arm, groups)
    expected_tasks = int(arm.get("expected_tasks", len(tasks)))
    if len(tasks) != expected_tasks:
        raise ValueError(
            f"registry expects {expected_tasks} tasks for {args.arm}, found {len(tasks)}"
        )
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    generator_name = str(arm["generator"])
    method_name = str(arm["prompt_method"])
    generator = registry["generators"][generator_name]
    method = registry["prompt_methods"][method_name]
    arm_root = experiment_dir(args.arm, registry_path) / args.arm
    method_signature = _method_signature(registry_path, method_name, method)
    rewrite_signature = _rewrite_signature(registry_path, method)
    generator_signature = _generator_signature(registry_path, generator_name, generator)
    generator_config = dict(generator)
    generator_config["checkpoint"] = str(
        resolve_from_registry(registry_path, str(generator["checkpoint"]))
    )
    if generator.get("config"):
        generator_config["config"] = str(
            resolve_from_registry(registry_path, str(generator["config"]))
        )
        generator_config["config_sha256"] = sha256_file(
            Path(generator_config["config"])
        )

    plan = {
        "arm": args.arm,
        "ready": True,
        "groups": groups,
        "task_count": len(tasks),
        "generator": generator_name,
        "prompt_method": method_name,
        "experiment_dir": str(arm_root),
        "method_signature": method_signature,
        "rewrite_signature": rewrite_signature,
        "generator_signature": generator_signature,
        "manifest_sha256": manifests,
    }
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    arm_root.mkdir(parents=True, exist_ok=True)
    local_cache_path = arm_root / "rewrite_records.json"
    method_type = method.get("type")
    shared_cache = method_type in {"skill", "vpo"}
    cache_path = (
        WORKSPACE / "cache" / "rewrites" / f"{rewrite_signature}.json"
        if shared_cache
        else local_cache_path
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = None
    if shared_cache:
        lock_handle = cache_path.with_suffix(".lock").open("a", encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    cache: dict[str, Any]
    try:
        if cache_path.is_file():
            old = read_json(cache_path)
            cache = old if old.get("rewrite_signature") == rewrite_signature else {}
        else:
            cache = {}
        cache.setdefault("schema_version", "1.0")
        cache["rewrite_signature"] = rewrite_signature
        cache.setdefault("records", {})

        unique_prompts = list(dict.fromkeys(str(task["raw_prompt"]) for task in tasks))
        if method_type == "raw":
            for prompt in unique_prompts:
                cache["records"][_cache_key(prompt)] = {
                    "raw_prompt": prompt,
                    "positive_prompt": prompt,
                    "conditional_negative_prompt": "",
                }
            atomic_json(cache_path, cache)
        elif method_type == "skill":
            cache = _skill_rewriter(registry_path, method, args.workers)(
                unique_prompts, cache, cache_path
            )
        elif method_type == "vpo":
            cache = _vpo_rewriter(registry_path, method, args.rewrite_gpu)(
                unique_prompts, cache, cache_path
            )
        else:
            raise ValueError(f"unsupported prompt method type: {method_type}")
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    selected_records = {
        _cache_key(prompt): cache["records"][_cache_key(prompt)]
        for prompt in unique_prompts
    }
    atomic_json(
        local_cache_path,
        {
            "schema_version": "1.0",
            "arm": args.arm,
            "prompt_method": method_name,
            "method_signature": method_signature,
            "rewrite_signature": rewrite_signature,
            "shared_cache": str(cache_path) if shared_cache else None,
            "drop_conditional_negative": bool(
                method.get("drop_conditional_negative", False)
            ),
            "records": selected_records,
        },
    )

    fixed_negative = str(registry.get("fixed_negative_prompt", ""))
    jobs: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for ordinal, task in enumerate(tasks):
        record = selected_records[_cache_key(str(task["raw_prompt"]))]
        output_path = output_path_for_task(arm_root, task).resolve()
        receipt_path = receipt_path_for_task(arm_root, task).resolve()
        if str(output_path) in output_paths:
            raise ValueError(f"duplicate output path: {output_path}")
        output_paths.add(str(output_path))
        conditional_negative = (
            ""
            if bool(method.get("drop_conditional_negative", False))
            else str(record.get("conditional_negative_prompt", ""))
        )
        jobs.append(
            {
                **task,
                "job_id": stable_job_id(task),
                "job_ordinal": ordinal,
                "positive_prompt": str(record["positive_prompt"]),
                "conditional_negative_prompt": conditional_negative,
                "final_negative_prompt": merge_negative(fixed_negative, conditional_negative),
                "output_path": str(output_path),
                "receipt_path": str(receipt_path),
                "method_signature": method_signature,
                "generator_signature": generator_signature,
                "rewrite_metadata": {
                    key: value
                    for key, value in record.items()
                    if key not in {"raw_prompt", "positive_prompt", "conditional_negative_prompt"}
                },
            }
        )

    jobs_path = arm_root / "jobs.jsonl"
    atomic_jsonl(jobs_path, jobs)
    experiment = {
        "schema_version": "1.0",
        **plan,
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "jobs": str(jobs_path),
        "videos_root": str(arm_root / "videos"),
        "receipts_root": str(arm_root / "receipts"),
        "generator_config": generator_config,
        "fixed_negative_prompt": fixed_negative,
    }
    atomic_json(arm_root / "experiment.json", experiment)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"jobs: {jobs_path}")


if __name__ == "__main__":
    main()
