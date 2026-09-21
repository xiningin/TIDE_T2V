#!/usr/bin/env python3
"""Validate copied prompt data, metadata alignment, and official repository links."""

from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "config" / "evaluation_plan.json"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_prompt(value: str) -> str:
    """Ignore the five known trailing-period differences in CompBench numeracy metadata."""
    return value.strip().rstrip(".").strip()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    for link_name in ("VBench", "T2V-CompBench"):
        path = WORKSPACE / "official" / link_name
        if not path.is_symlink():
            errors.append(f"missing official symlink: {path}")
        elif not path.exists():
            errors.append(f"broken official symlink: {path}")

    for prompt_set_id, item in config["prompt_sets"].items():
        prompt_path = WORKSPACE / item["file"]
        if not prompt_path.exists():
            errors.append(f"missing prompt file: {prompt_path}")
            continue
        prompts = read_lines(prompt_path)
        expected = int(item["expected_prompt_count"])
        if len(prompts) != expected:
            errors.append(f"{prompt_set_id}: expected {expected} prompts, found {len(prompts)}")
        if len(set(prompts)) != len(prompts):
            print(
                f"warning: {prompt_set_id} contains {len(prompts) - len(set(prompts))} duplicate text entries; indexes are preserved"
            )

        metadata_rel = item.get("metadata")
        if metadata_rel:
            metadata_path = WORKSPACE / metadata_rel
            if not metadata_path.exists():
                errors.append(f"missing metadata file: {metadata_path}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if len(metadata) != expected:
                errors.append(
                    f"{prompt_set_id}: expected {expected} metadata rows, found {len(metadata)}"
                )
            metadata_prompts = [str(row.get("prompt", "")).strip() for row in metadata]
            if [normalize_prompt(value) for value in metadata_prompts] != [
                normalize_prompt(value) for value in prompts
            ]:
                errors.append(f"{prompt_set_id}: prompt file and metadata order/content differ")
            elif metadata_prompts != prompts:
                print(
                    f"warning: {prompt_set_id} prompt text differs from metadata only by trailing periods; source order is preserved"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("bundle validation passed")
    for prompt_set_id, item in config["prompt_sets"].items():
        count = len(read_lines(WORKSPACE / item["file"]))
        print(f"{prompt_set_id}: {count} prompts -> {', '.join(item['dimensions'])}")


if __name__ == "__main__":
    main()
