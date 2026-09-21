#!/usr/bin/env python3
"""Prepare a matched 80-prompt, one-video-per-prompt VBench2 comparison.

The VISTA self-improvement run stores many candidate videos per prompt.  Its
trajectory.json records the selected ``best_video``; this script exposes those
80 selected videos with the raw benchmark filenames and pairs them with the
seed-3407 SANA raw/ours videos for an apples-to-apples preliminary comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROMPT_FILE = ROOT / "experiments/cross_method_results/vista/vista_vbench_prompts.txt"
VISTA_RUN_ROOT = ROOT / "experiments/cross_method_results/vista/vista_self_improve"
VISTA_EVAL_ROOT = ROOT / "experiments/cross_method_results/vista/vista_vbench_eval"
SANA_ROOT = ROOT / "experiments/cross_model_results/sana-2b"
OUT_ROOT = VISTA_EVAL_ROOT / "matched_triplet_1sample"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link_or_check(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() != src.resolve():
            raise RuntimeError(f"existing symlink points elsewhere: {dst}")
        return
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise RuntimeError(f"existing file differs from source: {dst}")
        return
    os.symlink(src.resolve(), dst)


def main() -> None:
    prompts = [line.strip() for line in PROMPT_FILE.read_text().splitlines() if line.strip()]
    assert len(prompts) == 80, len(prompts)

    # Locate each completed VISTA trajectory by its prompt index/hash directory.
    trajectories = {}
    for traj_path in VISTA_RUN_ROOT.glob("shard_gpu*/items/*/run/trajectory.json"):
        data = json.loads(traj_path.read_text())
        idx = int(traj_path.parents[1].name.split("_", 1)[0])
        if idx in trajectories:
            raise RuntimeError(f"duplicate VISTA trajectory for index {idx}")
        best = Path(data["best_video"]["video_path"])
        if not best.exists():
            raise FileNotFoundError(best)
        trajectories[idx] = {
            "prompt": data["user_prompt"],
            "source": str(best),
            "sha256": sha256(best),
            "best_id": data["best_video"].get("id", ""),
        }
    if sorted(trajectories) != list(range(80)):
        raise RuntimeError(f"VISTA trajectory indices are incomplete: {sorted(set(range(80)) - set(trajectories))}")

    groups = {
        "vista_self_improve": {},
        "sana_ours": {},
        "sana_raw": {},
    }
    for idx, prompt in enumerate(prompts):
        v = trajectories[idx]
        if v["prompt"] != prompt:
            raise RuntimeError(f"prompt mismatch at {idx}: {v['prompt']!r} != {prompt!r}")
        filename = f"{prompt}-0.mp4"
        vista_dst = OUT_ROOT / "vista_self_improve" / filename
        # The existing per-dimension VISTA eval tree is the audited copy of the
        # same best_video files; prefer it when available, otherwise link runs.
        existing = list(VISTA_EVAL_ROOT.glob(f"*/{filename}"))
        vista_src = existing[0] if existing else Path(v["source"])
        link_or_check(vista_dst, vista_src)
        groups["vista_self_improve"][filename] = {"source": str(vista_src), "sha256": sha256(vista_src), "prompt_index": idx, "best_id": v["best_id"]}

        for arm in ("sana_ours", "sana_raw"):
            src = SANA_ROOT / arm / "videos/main_vbench" / arm / filename
            if not src.exists():
                raise FileNotFoundError(src)
            dst = OUT_ROOT / arm / filename
            link_or_check(dst, src)
            groups[arm][filename] = {"source": str(src), "sha256": sha256(src), "prompt_index": idx, "seed_suffix": 0}

    manifest = {
        "description": "Matched VBench2 subset: 80 prompts, one video per prompt, VISTA best_video vs SANA seed-3407 raw/ours.",
        "prompt_file": str(PROMPT_FILE),
        "prompt_count": len(prompts),
        "groups": groups,
        "dimensions": ["subject_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality", "overall_consistency", "multiple_objects", "human_action", "spatial_relationship"],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"prepared {len(prompts)} prompts x {len(groups)} groups at {OUT_ROOT}")
    for group in groups:
        print(group, len(list((OUT_ROOT / group).glob("*.mp4"))))


if __name__ == "__main__":
    main()
