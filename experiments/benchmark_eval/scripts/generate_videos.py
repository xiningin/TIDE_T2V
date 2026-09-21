#!/usr/bin/env python3
"""Generate one shard of a prepared benchmark arm without running evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generation_common import (  # noqa: E402
    atomic_json,
    output_path_for_task,
    read_json,
    read_jsonl,
    receipt_path_for_task,
    sha256_file,
    sha256_text,
    stable_job_id,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", type=Path, required=True)
    result.add_argument("--gpu", required=True, help="Physical CUDA device id for this shard.")
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--max-tasks", type=int, default=0)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--continue-on-error", action="store_true")
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the shard; do not load a model or write files.",
    )
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_experiment(
    experiment_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_path = experiment_path.expanduser().resolve()
    experiment = read_json(experiment_path)
    if experiment.get("schema_version") != "1.0":
        raise ValueError(f"unsupported experiment schema: {experiment.get('schema_version')}")

    registry_path = Path(str(experiment["registry"])).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"registry disappeared after preparation: {registry_path}")
    if sha256_file(registry_path) != experiment.get("registry_sha256"):
        raise RuntimeError("generation registry changed; rerun prepare_generation.py")

    generator_config = experiment.get("generator_config")
    if not isinstance(generator_config, dict):
        raise ValueError("experiment.generator_config must be an object")
    checkpoint = Path(str(generator_config["checkpoint"])).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing generator checkpoint: {checkpoint}")
    if generator_config.get("config"):
        config_path = Path(str(generator_config["config"])).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"missing generator config: {config_path}")
        if sha256_file(config_path) != generator_config.get("config_sha256"):
            raise RuntimeError("generator config changed; rerun prepare_generation.py")

    jobs_path = Path(str(experiment["jobs"])).resolve()
    jobs = read_jsonl(jobs_path)
    if len(jobs) != int(experiment["task_count"]):
        raise ValueError(
            f"task count mismatch: experiment={experiment['task_count']}, jobs={len(jobs)}"
        )

    arm = str(experiment["arm"])
    root = experiment_path.parent
    job_ids: set[str] = set()
    for expected_ordinal, job in enumerate(jobs):
        if job.get("arm") != arm:
            raise ValueError(f"foreign arm in jobs.jsonl at ordinal {expected_ordinal}")
        if job.get("job_ordinal") != expected_ordinal:
            raise ValueError(f"non-contiguous job_ordinal at ordinal {expected_ordinal}")
        if job.get("job_id") != stable_job_id(job):
            raise ValueError(f"invalid job_id at ordinal {expected_ordinal}")
        if str(job["job_id"]) in job_ids:
            raise ValueError(f"duplicate job_id: {job['job_id']}")
        job_ids.add(str(job["job_id"]))
        expected_video = output_path_for_task(root, job).resolve()
        expected_receipt = receipt_path_for_task(root, job).resolve()
        if Path(str(job["output_path"])).resolve() != expected_video:
            raise ValueError(f"output path escaped canonical layout: {job['job_id']}")
        if Path(str(job["receipt_path"])).resolve() != expected_receipt:
            raise ValueError(f"receipt path escaped canonical layout: {job['job_id']}")
        if job.get("generator_signature") != experiment.get("generator_signature"):
            raise ValueError(f"generator signature mismatch: {job['job_id']}")
        if job.get("method_signature") != experiment.get("method_signature"):
            raise ValueError(f"method signature mismatch: {job['job_id']}")
    return experiment, jobs


def _valid_receipt(job: dict[str, Any]) -> bool:
    receipt_path = Path(str(job["receipt_path"]))
    target = Path(str(job["output_path"]))
    if not receipt_path.is_file() or not target.is_file() or target.stat().st_size <= 0:
        return False
    try:
        receipt = read_json(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    checks = (
        receipt.get("schema_version") == "1.0",
        receipt.get("status") == "generated",
        receipt.get("job_id") == job.get("job_id"),
        receipt.get("arm") == job.get("arm"),
        receipt.get("seed") == job.get("seed"),
        receipt.get("video_path") == str(target),
        receipt.get("video_bytes") == target.stat().st_size,
        receipt.get("generator_signature") == job.get("generator_signature"),
        receipt.get("method_signature") == job.get("method_signature"),
        receipt.get("positive_prompt_sha256")
        == sha256_text(str(job["positive_prompt"])),
        receipt.get("negative_prompt_sha256")
        == sha256_text(str(job["final_negative_prompt"])),
        receipt.get("dry_run") is False,
    )
    return all(checks) and receipt.get("video_sha256") == sha256_file(target)


class SanaBackend:
    def __init__(self, config: dict[str, Any], gpu: str):
        from vista.clients.sana_client import SanaDiffusersT2VClient
        from vista.utils.config import load_config

        source = load_config(Path(str(config["config"])))
        runtime = dict(source["t2v"])
        runtime.update(
            {
                "backend": "sana_diffusers",
                "dry_run": False,
                "checkpoint_dir": str(config["checkpoint"]),
                "cuda_visible_devices": gpu,
                "size": f"{int(config['width'])}*{int(config['height'])}",
                "frame_num": int(config["frames"]),
                "fps": float(config["fps"]),
                "sample_steps": int(config["steps"]),
                "guidance_scale": float(config["guidance_scale"]),
                "precision": str(config["precision"]),
                "startup_timeout_seconds": int(
                    config.get("startup_timeout_seconds", 1800)
                ),
                "command_timeout_seconds": int(
                    config.get("command_timeout_seconds", 1200)
                ),
                # Each prepared job already contains the merged fixed + learned negative.
                "negative_prompt": "",
            }
        )
        self.client = SanaDiffusersT2VClient(runtime)

    def generate(self, job: dict[str, Any]) -> dict[str, Any]:
        target = Path(str(job["output_path"]))
        generation = self.client.generate(
            prompt=str(job["positive_prompt"]),
            output_dir=target.parent,
            candidate_id=target.stem,
            seed=int(job["seed"]),
            negative_prompt=str(job["final_negative_prompt"]),
        )
        generated = Path(generation.video_path).resolve()
        if generated != target.resolve():
            raise RuntimeError(f"SANA output mismatch: expected {target}, got {generated}")
        if generation.dry_run:
            raise RuntimeError("SANA unexpectedly returned a dry-run placeholder")
        call = self.client.calls[-1]
        if call["negative_prompt"] != job["final_negative_prompt"]:
            raise RuntimeError("SANA negative prompt differs from the prepared job")
        return {
            "elapsed_seconds": float(generation.elapsed_seconds),
            "worker_load_seconds": float(call["worker_load_seconds"]),
        }

    def close(self) -> None:
        self.client.close()


class CogVideoXBackend:
    def __init__(self, config: dict[str, Any], gpu: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import torch
        from diffusers import CogVideoXPipeline

        dtypes = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        started = time.perf_counter()
        self.torch = torch
        self.config = config
        self.pipe = CogVideoXPipeline.from_pretrained(
            str(config["checkpoint"]),
            torch_dtype=dtypes[str(config["precision"])],
            local_files_only=True,
        )
        if bool(config.get("vae_tiling", True)):
            self.pipe.vae.enable_tiling()
        if bool(config.get("vae_slicing", True)):
            self.pipe.vae.enable_slicing()
        if bool(config.get("enable_cpu_offload", False)):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.load_seconds = time.perf_counter() - started

    def generate(self, job: dict[str, Any]) -> dict[str, Any]:
        from diffusers.utils import export_to_video

        target = Path(str(job["output_path"]))
        partial = target.with_name(f".{target.stem}.partial{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        started = time.perf_counter()
        try:
            generator = self.torch.Generator(device="cuda").manual_seed(int(job["seed"]))
            result = self.pipe(
                prompt=str(job["positive_prompt"]),
                negative_prompt=str(job["final_negative_prompt"]),
                width=int(self.config["width"]),
                height=int(self.config["height"]),
                num_frames=int(self.config["frames"]),
                num_inference_steps=int(self.config["steps"]),
                guidance_scale=float(self.config["guidance_scale"]),
                use_dynamic_cfg=bool(self.config.get("use_dynamic_cfg", False)),
                generator=generator,
            )
            export_to_video(result.frames[0], str(partial), fps=int(self.config["fps"]))
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise RuntimeError("CogVideoX export did not create a non-empty MP4")
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            self.torch.cuda.empty_cache()
            raise
        return {
            "elapsed_seconds": time.perf_counter() - started,
            "worker_load_seconds": self.load_seconds,
        }

    def close(self) -> None:
        del self.pipe
        self.torch.cuda.empty_cache()



class WanBackend:
    def __init__(self, config: dict[str, Any], gpu: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import torch

        self.torch = torch
        self.config = config

        # Add Wan2.2 repo to path
        # Older prepared experiment manifests may omit the local Wan2.2
        # repository path.  Keep those manifests usable by falling back to
        # the repository used by the existing Wan2.2 runs.
        wan_repo = str(
            Path(config.get("wan_repo", "/tmp/Wan2.2_repo")).resolve()
        )
        if wan_repo not in sys.path:
            sys.path.insert(0, wan_repo)

        from wan.configs import WAN_CONFIGS
        from wan.textimage2video import WanTI2V

        started = time.perf_counter()
        wan_cfg = WAN_CONFIGS["ti2v-5B"]
        self.wan = WanTI2V(
            config=wan_cfg,
            checkpoint_dir=str(Path(config["checkpoint"]).resolve()),
            device_id=0,
            rank=0,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=False,
        )
        self.load_seconds = time.perf_counter() - started

    def generate(self, job: dict[str, Any]) -> dict[str, Any]:
        import imageio
        import numpy as np

        target = Path(str(job["output_path"]))
        partial = target.with_name(f".{target.stem}.partial{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)

        started = time.perf_counter()
        try:
            seed = int(job["seed"])
            video_tensor = self.wan.generate(
                input_prompt=str(job["positive_prompt"]),
                img=None,
                size=(int(self.config["width"]), int(self.config["height"])),
                max_area=int(self.config["width"]) * int(self.config["height"]),
                frame_num=int(self.config["frames"]),
                shift=float(self.config.get("sample_shift", 5.0)),
                sample_solver="unipc",
                sampling_steps=int(self.config["steps"]),
                guide_scale=float(self.config["guidance_scale"]),
                n_prompt=str(job["final_negative_prompt"]),
                seed=seed,
                offload_model=False,
            )

            # video_tensor: (C, N, H, W) in [-1, 1]
            video_tensor = video_tensor.clamp(-1, 1)
            # Convert to (N, H, W, C) uint8
            video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
            video_np = ((video_np + 1) / 2 * 255).astype(np.uint8)

            writer = imageio.get_writer(
                str(partial), fps=int(self.config["fps"]), codec="libx264", quality=8
            )
            for frame in video_np:
                writer.append_data(frame)
            writer.close()

            if not partial.is_file() or partial.stat().st_size <= 0:
                raise RuntimeError("Wan2.2 export did not create a non-empty MP4")
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            self.torch.cuda.empty_cache()
            raise

        return {
            "elapsed_seconds": time.perf_counter() - started,
            "worker_load_seconds": self.load_seconds,
        }

    def close(self) -> None:
        del self.wan
        self.torch.cuda.empty_cache()

def _make_backend(config: dict[str, Any], gpu: str) -> SanaBackend | CogVideoXBackend | WanBackend:
    backend = str(config.get("backend"))
    if backend == "sana_diffusers":
        return SanaBackend(config, gpu)
    if backend == "cogvideox_diffusers":
        return CogVideoXBackend(config, gpu)
    if backend == "wan_ti2v":
        return WanBackend(config, gpu)
    raise ValueError(f"unsupported generation backend: {backend}")


def _receipt(
    job: dict[str, Any],
    experiment: dict[str, Any],
    gpu: str,
    shard_index: int,
    num_shards: int,
    timing: dict[str, Any],
) -> dict[str, Any]:
    target = Path(str(job["output_path"]))
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"generation did not create a non-empty video: {target}")
    return {
        "schema_version": "1.0",
        "status": "generated",
        "dry_run": False,
        "job_id": job["job_id"],
        "job_ordinal": job["job_ordinal"],
        "arm": job["arm"],
        "group": job["group"],
        "benchmark": job["benchmark"],
        "prompt_set": job["prompt_set"],
        "prompt_index": job["prompt_index"],
        "sample_index": job["sample_index"],
        "seed": job["seed"],
        "raw_prompt": job["raw_prompt"],
        "positive_prompt": job["positive_prompt"],
        "conditional_negative_prompt": job["conditional_negative_prompt"],
        "final_negative_prompt": job["final_negative_prompt"],
        "positive_prompt_sha256": sha256_text(str(job["positive_prompt"])),
        "negative_prompt_sha256": sha256_text(str(job["final_negative_prompt"])),
        "prompt_method": experiment["prompt_method"],
        "generator": experiment["generator"],
        "method_signature": job["method_signature"],
        "generator_signature": job["generator_signature"],
        "rewrite_metadata": job.get("rewrite_metadata", {}),
        "generator_config": experiment["generator_config"],
        "video_path": str(target),
        "video_bytes": target.stat().st_size,
        "video_sha256": sha256_file(target),
        "gpu": gpu,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "elapsed_seconds": timing["elapsed_seconds"],
        "worker_load_seconds": timing.get("worker_load_seconds", 0.0),
        "completed_at": utc_now(),
    }


def main() -> None:
    args = parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.max_tasks < 0:
        raise ValueError("max-tasks must be non-negative")

    experiment, all_jobs = _check_experiment(args.experiment)
    jobs = [
        job
        for job in all_jobs
        if int(job["job_ordinal"]) % args.num_shards == args.shard_index
    ]
    if args.max_tasks:
        jobs = jobs[: args.max_tasks]

    preview = {
        "arm": experiment["arm"],
        "backend": experiment["generator_config"]["backend"],
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_jobs": len(jobs),
        "total_jobs": len(all_jobs),
        "dry_run": args.dry_run,
        "first_jobs": [
            {"job_id": job["job_id"], "output_path": job["output_path"]}
            for job in jobs[:5]
        ],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or not jobs:
        return

    backend: SanaBackend | CogVideoXBackend | None = None
    failures = 0
    resumed = 0
    generated = 0
    try:
        for index, job in enumerate(jobs, start=1):
            target = Path(str(job["output_path"]))
            receipt_path = Path(str(job["receipt_path"]))
            if args.resume and _valid_receipt(job):
                resumed += 1
                print(f"[{index}/{len(jobs)}] resume {job['job_id']}: {target}", flush=True)
                continue
            started = time.perf_counter()
            try:
                if backend is None:
                    backend = _make_backend(experiment["generator_config"], args.gpu)
                timing = backend.generate(job)
                atomic_json(
                    receipt_path,
                    _receipt(
                        job,
                        experiment,
                        args.gpu,
                        args.shard_index,
                        args.num_shards,
                        timing,
                    ),
                )
                generated += 1
                print(
                    f"[{index}/{len(jobs)}] generated {job['job_id']}: {target}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                failure = {
                    "schema_version": "1.0",
                    "status": "failed",
                    "dry_run": False,
                    "job_id": job["job_id"],
                    "job_ordinal": job["job_ordinal"],
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "video_path": str(target),
                    "generator_signature": job["generator_signature"],
                    "method_signature": job["method_signature"],
                    "gpu": args.gpu,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "elapsed_seconds": time.perf_counter() - started,
                    "failed_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                atomic_json(receipt_path, failure)
                print(
                    f"[{index}/{len(jobs)}] failed {job['job_id']}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
    finally:
        if backend is not None:
            backend.close()

    print(
        json.dumps(
            {
                "selected_jobs": len(jobs),
                "generated": generated,
                "resumed": resumed,
                "failed": failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
