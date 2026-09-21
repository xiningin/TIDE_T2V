#!/usr/bin/env python3
"""Build deterministic, per-video generation manifests from evaluation_plan.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "config" / "evaluation_plan.json"
DEFAULT_OUTPUT = WORKSPACE / "manifests"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_filename(benchmark: str, prompt: str, prompt_index: int, sample_index: int) -> str:
    if benchmark == "vbench":
        return f"{prompt}-{sample_index}.mp4"
    if benchmark == "t2v_compbench":
        return f"{prompt_index:04d}.mp4"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def build_group(config: dict, group_name: str) -> list[dict]:
    group = config["groups"][group_name]
    arms = config["arms"]
    prompt_sets = config["prompt_sets"]
    samples_per_prompt = int(group["samples_per_prompt"])
    vbench_seeds = config["vbench_seeds"]
    if samples_per_prompt > len(vbench_seeds):
        raise ValueError(
            f"{group_name}: samples_per_prompt={samples_per_prompt} exceeds configured seed count"
        )

    tasks: list[dict] = []
    for arm_id in group["arms"]:
        arm = arms[arm_id]
        for prompt_set_id in group["prompt_sets"]:
            prompt_set = prompt_sets[prompt_set_id]
            benchmark = prompt_set["benchmark"]
            prompt_path = WORKSPACE / prompt_set["file"]
            prompts = read_lines(prompt_path)
            expected = int(prompt_set["expected_prompt_count"])
            if len(prompts) != expected:
                raise ValueError(f"{prompt_path}: expected {expected} prompts, found {len(prompts)}")

            for prompt_index, raw_prompt in enumerate(prompts, start=1):
                for sample_index in range(samples_per_prompt):
                    seed = vbench_seeds[sample_index]
                    filename = task_filename(
                        benchmark, raw_prompt, prompt_index, sample_index
                    )
                    if benchmark == "vbench":
                        # VBench standard mode uses os.listdir and therefore requires
                        # all videos for an arm to be directly inside one flat folder.
                        relative_path = Path("outputs") / group_name / arm_id / filename
                    else:
                        # CompBench runs one category directory at a time.
                        relative_path = (
                            Path("outputs")
                            / group_name
                            / arm_id
                            / prompt_set_id
                            / filename
                        )
                    tasks.append(
                        {
                            "group": group_name,
                            "arm": arm_id,
                            "model": arm["model"],
                            "method": arm["method"],
                            "benchmark": benchmark,
                            "prompt_set": prompt_set_id,
                            "dimensions": prompt_set["dimensions"],
                            "prompt_index": prompt_index,
                            "raw_prompt": raw_prompt,
                            "sample_index": sample_index,
                            "seed": seed,
                            "benchmark_filename": filename,
                            "output_relpath": relative_path.as_posix(),
                        }
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": config["schema_version"], "groups": {}, "total_tasks": 0}

    for group_name in config["groups"]:
        tasks = build_group(config, group_name)
        output_path = args.output_dir / f"{group_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        per_arm = len(tasks) // len(config["groups"][group_name]["arms"])
        summary["groups"][group_name] = {
            "arms": len(config["groups"][group_name]["arms"]),
            "tasks_per_arm": per_arm,
            "tasks": len(tasks),
            "manifest": output_path.relative_to(WORKSPACE).as_posix(),
        }
        summary["total_tasks"] += len(tasks)
        print(f"{group_name}: {len(tasks):,} tasks ({per_arm:,} per arm)")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"total: {summary['total_tasks']:,} tasks")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
