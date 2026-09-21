#!/usr/bin/env python3
"""Generate a seed-matched raw/rewritten pair and optionally judge it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.core.agent import VistaAgent  # noqa: E402
from vista.schemas import PromptCandidate, VideoCandidate  # noqa: E402
from vista.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed-matched T2V prompt rewrite ablation.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--judge", action="store_true", help="Run the configured MLLM pairwise judge after generation.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse matching raw/skill MP4 files in output-dir.")
    return parser.parse_args()


def _inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output-dir must stay inside {ROOT}") from exc
    return resolved


def main() -> None:
    args = parse_args()
    output_dir = _inside_repo(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    agent = VistaAgent(config)

    rewrite_started = perf_counter()
    rewrite = agent.rewrite_prompt(args.prompt)
    rewrite_seconds = perf_counter() - rewrite_started
    raw_prompt = PromptCandidate("raw_p0", args.prompt, 0, metadata={"source": "raw"})
    rewritten_prompt = PromptCandidate(
        "skill_p0",
        rewrite.rewritten_prompt,
        0,
        negative_prompt=rewrite.negative_prompt,
        metadata={"source": "frozen_skill", "skill_revision": rewrite.skill_revision, "skill_sha256": rewrite.skill_sha256},
    )
    videos: list[VideoCandidate] = []
    for candidate in (raw_prompt, rewritten_prompt):
        expected_video = output_dir / candidate.id / f"{candidate.id}_v0.mp4"
        if args.reuse_existing and expected_video.is_file():
            video_path = str(expected_video)
            metadata = {"dry_run": False, "elapsed_seconds": None, "command": "reused_existing"}
        else:
            generation = agent.t2v.generate(
                candidate.prompt,
                output_dir / candidate.id,
                f"{candidate.id}_v0",
                seed=args.seed,
                negative_prompt=candidate.negative_prompt,
            )
            video_path = generation.video_path
            metadata = {"dry_run": generation.dry_run, "elapsed_seconds": generation.elapsed_seconds, "command": generation.command}
        videos.append(
            VideoCandidate(
                id=f"{candidate.id}_v0",
                prompt_candidate=candidate,
                video_path=video_path,
                round_index=0,
                seed=args.seed,
                metadata=metadata,
            )
        )

    judge = None
    judge_error = None
    if args.judge:
        try:
            judge = agent.selector.compare_for_improvement(args.prompt, videos[0], videos[1]).to_dict()
        except Exception as exc:  # Preserve expensive generations even if evaluation fails.
            judge_error = f"{type(exc).__name__}: {exc}"
    result = {
        "schema_version": "1.0",
        "prompt": args.prompt,
        "seed": args.seed,
        "rewrite": rewrite.to_dict(),
        "rewrite_seconds": rewrite_seconds,
        "videos": [item.to_dict() for item in videos],
        "t2v_calls": agent.t2v.calls,
        "mllm_calls": agent.all_mllm_calls(),
        "pairwise_judge": judge,
        "judge_error": judge_error,
        "config": config,
    }
    path = output_dir / "pair_ablation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    close = getattr(agent.t2v, "close", None)
    if callable(close):
        close()
    if judge_error:
        raise RuntimeError(f"Pair generated and saved, but judge failed: {judge_error}")
    print(json.dumps({"result": str(path), "raw_video": videos[0].video_path, "skill_video": videos[1].video_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
