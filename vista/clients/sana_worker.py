from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", "*").split("*")
    if len(parts) != 2:
        raise ValueError(f"invalid SANA size: {value}")
    width, height = (int(item) for item in parts)
    return height, width


def _motion_prompt(prompt: str, score: int | float) -> str:
    cleaned = prompt.rstrip()
    separator = "" if cleaned.endswith((".", "!", "?")) else "."
    return f"{cleaned}{separator} motion score: {score}."


def run_sana_worker(connection: Connection, config: dict[str, Any]) -> None:
    """Model-resident child process; import CUDA libraries only after GPU isolation."""
    visible = str(config.get("cuda_visible_devices", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    try:
        import torch
        from diffusers import SanaVideoPipeline, __version__ as diffusers_version
        from diffusers.utils import export_to_video

        expected_version = str(config.get("diffusers_version", ""))
        if expected_version and diffusers_version != expected_version:
            raise RuntimeError(f"diffusers version mismatch: expected {expected_version}, got {diffusers_version}")
        dtype_name = str(config.get("precision", "bf16"))
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtypes[dtype_name]
        checkpoint = str(config.get("checkpoint_dir") or config.get("ckpt_dir") or "")
        started = time.perf_counter()
        pipeline = SanaVideoPipeline.from_pretrained(checkpoint, torch_dtype=dtype, local_files_only=True)
        pipeline.vae.to(torch.float32)
        pipeline.text_encoder.to(dtype)
        pipeline.to("cuda")
        expected_shift = config.get("sample_shift")
        actual_shift = getattr(pipeline.scheduler.config, "flow_shift", None)
        if expected_shift is not None and actual_shift is not None and float(expected_shift) != float(actual_shift):
            raise RuntimeError(f"scheduler flow_shift mismatch: expected {expected_shift}, got {actual_shift}")
        connection.send({"status": "ready", "load_seconds": time.perf_counter() - started, "gpu": visible})
    except Exception as exc:  # noqa: BLE001
        connection.send({"status": "error", "error": f"{type(exc).__name__}:{exc}"})
        connection.close()
        return

    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request.get("op") == "close":
            break
        if request.get("op") != "generate":
            connection.send({"status": "error", "error": "unknown operation"})
            continue
        started = time.perf_counter()
        target = Path(request["video_path"])
        partial = target.with_name(f".{target.stem}.partial{target.suffix}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial.unlink(missing_ok=True)
            height, width = _parse_size(str(config.get("size", "832*480")))
            seed = int(request["seed"])
            generator = torch.Generator(device="cuda").manual_seed(seed)
            result = pipeline(
                prompt=_motion_prompt(str(request["prompt"]), config.get("motion_score", 30)),
                negative_prompt=str(request.get("negative_prompt", config.get("negative_prompt", ""))),
                height=height,
                width=width,
                frames=int(config.get("frame_num", 81)),
                guidance_scale=float(config.get("guidance_scale", config.get("sample_guide_scale", 6.0))),
                num_inference_steps=int(config.get("sample_steps", 50)),
                generator=generator,
                clean_caption=bool(config.get("clean_caption", False)),
                use_resolution_binning=bool(config.get("use_resolution_binning", True)),
                max_sequence_length=int(config.get("max_sequence_length", 300)),
            )
            frames = result.frames[0]
            export_to_video(frames, str(partial), fps=float(config.get("fps", 16.0)))
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("SANA export did not create a non-empty video")
            partial.replace(target)
            connection.send({"status": "ok", "elapsed_seconds": time.perf_counter() - started, "video_path": str(target)})
        except Exception as exc:  # noqa: BLE001
            partial.unlink(missing_ok=True)
            torch.cuda.empty_cache()
            connection.send({"status": "error", "error": f"{type(exc).__name__}:{exc}"})

    connection.close()
