from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass
class T2VGeneration:
    video_path: str
    dry_run: bool
    command: list[str] | None = None
    elapsed_seconds: float = 0.0


class WanCliT2VClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.wan_dir = Path(config.get("wan_dir", "")).resolve()
        self.python = config.get("python_executable", "python")
        self.ckpt_dir = Path(config.get("ckpt_dir", "")).resolve()
        self.dry_run = bool(config.get("dry_run", True))
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, output_dir: str | Path, candidate_id: str, seed: int | None = None, negative_prompt: str | None = None) -> T2VGeneration:
        started = perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"{candidate_id}.mp4"

        if self.dry_run or not self.ckpt_dir.exists():
            video_path.write_text("VISTA dry-run placeholder. Replace ckpt_dir and disable dry_run for real Wan generation.\n", encoding="utf-8")
            manifest = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "candidate_id": candidate_id,
                "seed": seed,
                "dry_run": True,
                "reason": "t2v.dry_run enabled or ckpt_dir missing",
                "ckpt_dir": str(self.ckpt_dir),
            }
            video_path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            generation = T2VGeneration(video_path=str(video_path), dry_run=True, elapsed_seconds=perf_counter() - started)
            self._record_call(candidate_id, prompt, seed, generation)
            return generation

        generate_py = self.wan_dir / "generate.py"
        if not generate_py.exists():
            raise FileNotFoundError(f"Wan generate.py not found: {generate_py}")

        cmd = [
            self.python,
            str(generate_py),
            "--task",
            str(self.config.get("task", "t2v-1.3B")),
            "--size",
            str(self.config.get("size", "832*480")),
            "--ckpt_dir",
            str(self.ckpt_dir),
            "--save_file",
            str(video_path),
            "--prompt",
            prompt,
            "--offload_model",
            str(bool(self.config.get("offload_model", True))),
            "--sample_solver",
            str(self.config.get("sample_solver", "unipc")),
        ]
        if self.config.get("t5_cpu", False):
            cmd.append("--t5_cpu")
        if self.config.get("t5_fsdp", False):
            cmd.append("--t5_fsdp")
        if self.config.get("dit_fsdp", False):
            cmd.append("--dit_fsdp")
        if seed is not None:
            cmd.extend(["--base_seed", str(seed)])
        for key, flag in (
            ("sample_steps", "--sample_steps"),
            ("sample_shift", "--sample_shift"),
            ("sample_guide_scale", "--sample_guide_scale"),
            ("frame_num", "--frame_num"),
            ("ulysses_size", "--ulysses_size"),
            ("ring_size", "--ring_size"),
        ):
            value = self.config.get(key)
            if value is not None:
                cmd.extend([flag, str(value)])

        timeout = int(self.config.get("command_timeout_seconds") or 0) or None
        env = os.environ.copy()
        if self.config.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.config["cuda_visible_devices"])
        extra_env = self.config.get("env")
        if isinstance(extra_env, dict):
            env.update({str(key): str(value) for key, value in extra_env.items()})

        subprocess.run(cmd, cwd=str(self.wan_dir), check=True, timeout=timeout, env=env)
        generation = T2VGeneration(video_path=str(video_path), dry_run=False, command=cmd, elapsed_seconds=perf_counter() - started)
        self._record_call(candidate_id, prompt, seed, generation)
        return generation

    def _record_call(self, candidate_id: str, prompt: str, seed: int | None, generation: T2VGeneration) -> None:
        self.calls.append(
            {
                "candidate_id": candidate_id,
                "prompt": prompt,
                "seed": seed,
                "video_path": generation.video_path,
                "dry_run": generation.dry_run,
                "elapsed_seconds": generation.elapsed_seconds,
                "command": generation.command,
            }
        )


def create_t2v_client(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend", "wan_cli")).lower()
    if backend in {"sana", "sana_video", "sana_diffusers"}:
        from vista.clients.sana_client import SanaDiffusersT2VClient

        return SanaDiffusersT2VClient(config)
    if backend == "wan_cli":
        return WanCliT2VClient(config)
    raise ValueError(f"Unsupported T2V backend: {backend}")
