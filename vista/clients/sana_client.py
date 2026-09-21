from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
from typing import Any

from vista.clients.sana_worker import run_sana_worker
from vista.clients.t2v_client import T2VGeneration


class SanaDiffusersT2VClient:
    """Persistent SANA-Video client isolated to its configured GPU subprocess."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.checkpoint_dir = Path(config.get("checkpoint_dir") or config.get("ckpt_dir") or "").resolve()
        self.dry_run = bool(config.get("dry_run", True))
        self.calls: list[dict[str, Any]] = []
        self._process: Any | None = None
        self._connection: Any | None = None
        self._load_seconds = 0.0

    def generate(
        self,
        prompt: str,
        output_dir: str | Path,
        candidate_id: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> T2VGeneration:
        started = perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"{candidate_id}.mp4"
        seed = int(seed or 0)
        if self.dry_run or not self.checkpoint_dir.exists():
            video_path.write_text("SANA dry-run placeholder.\n", encoding="utf-8")
            final_negative = self._merge_negative(negative_prompt)
            video_path.with_suffix(".json").write_text(
                json.dumps({"prompt": prompt, "negative_prompt": final_negative, "candidate_id": candidate_id, "seed": seed, "dry_run": True}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generation = T2VGeneration(str(video_path), True, elapsed_seconds=perf_counter() - started)
            self._record(candidate_id, prompt, final_negative, seed, generation)
            return generation
        self._ensure_worker()
        final_negative = self._merge_negative(negative_prompt)
        self._connection.send({"op": "generate", "prompt": prompt, "negative_prompt": final_negative, "seed": seed, "video_path": str(video_path.resolve())})
        timeout = int(self.config.get("command_timeout_seconds") or 0)
        if timeout and not self._connection.poll(timeout):
            self.close(force=True)
            raise TimeoutError(f"SANA generation timed out after {timeout}s")
        response = self._connection.recv()
        if response.get("status") != "ok":
            raise RuntimeError(f"SANA generation failed: {response.get('error')}")
        generation = T2VGeneration(
            video_path=str(video_path),
            dry_run=False,
            command=["persistent_sana_diffusers_worker", f"gpu={self.config.get('cuda_visible_devices', '0')}"],
            elapsed_seconds=float(response.get("elapsed_seconds", perf_counter() - started)),
        )
        self._record(candidate_id, prompt, final_negative, seed, generation)
        return generation

    def _merge_negative(self, conditional: str | None) -> str:
        terms: list[str] = []
        for value in (str(self.config.get("negative_prompt", "")), conditional or ""):
            for raw in value.replace(";", ",").split(","):
                term = " ".join(raw.strip().split()).rstrip(".")
                if term and term.lower() not in {item.lower() for item in terms}:
                    terms.append(term)
        return ", ".join(terms)

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._process = context.Process(target=run_sana_worker, args=(child, self.config), daemon=True)
        self._process.start()
        child.close()
        self._connection = parent
        timeout = int(self.config.get("startup_timeout_seconds", 600))
        if not parent.poll(timeout):
            self.close(force=True)
            raise TimeoutError(f"SANA worker startup timed out after {timeout}s")
        response = parent.recv()
        if response.get("status") != "ready":
            self.close(force=True)
            raise RuntimeError(f"SANA worker startup failed: {response.get('error')}")
        self._load_seconds = float(response.get("load_seconds", 0.0))

    def close(self, *, force: bool = False) -> None:
        if self._connection is not None:
            if not force:
                try:
                    self._connection.send({"op": "close"})
                except (BrokenPipeError, EOFError):
                    pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
            self._process = None

    def _record(self, candidate_id: str, prompt: str, negative_prompt: str, seed: int, generation: T2VGeneration) -> None:
        self.calls.append(
            {
                "candidate_id": candidate_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "video_path": generation.video_path,
                "dry_run": generation.dry_run,
                "elapsed_seconds": generation.elapsed_seconds,
                "worker_load_seconds": self._load_seconds,
                "command": generation.command,
            }
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
