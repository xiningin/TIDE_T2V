#!/usr/bin/env python3
"""Sample a deterministic canonical-prompt suite from all VBench-2.0 dimensions."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

DIMENSIONS = [
    "Camera_Motion",
    "Complex_Landscape",
    "Complex_Plot",
    "Composition",
    "Diversity",
    "Dynamic_Attribute",
    "Dynamic_Spatial_Relationship",
    "Human_Anatomy",
    "Human_Clothes",
    "Human_Identity",
    "Human_Interaction",
    "Instance_Preservation",
    "Material",
    "Mechanics",
    "Motion_Order_Understanding",
    "Motion_Rationality",
    "Multi-View_Consistency",
    "Thermotics",
]


def nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_payload(source_root: Path, prompts_per_dimension: int, seed: int) -> dict:
    groups = []
    next_id = 1
    for dimension in DIMENSIONS:
        source = source_root / "prompts/prompt" / f"{dimension}.txt"
        prompts = nonempty_lines(source)
        if len(prompts) < prompts_per_dimension:
            raise ValueError(
                f"not enough prompts for {dimension}: requested={prompts_per_dimension}, "
                f"available={len(prompts)}"
            )
        indices = sorted(
            random.Random(f"{seed}:{dimension}").sample(
                range(len(prompts)), prompts_per_dimension
            )
        )
        records = []
        for index in indices:
            records.append(
                {
                    "id": f"vbench2-r{prompts_per_dimension}-{next_id:03d}",
                    "source_line": index + 1,
                    "canonical_prompt": prompts[index],
                }
            )
            next_id += 1
        groups.append(
            {
                "dimension": dimension,
                "prompt_count": len(records),
                "prompts": records,
            }
        )
    return {
        "name": "VBench-2.0 Full-18 deterministic subset",
        "version": 1,
        "source_root": str(source_root),
        "protocol": {
            "total_dimensions": len(DIMENSIONS),
            "prompts_per_dimension": prompts_per_dimension,
            "total_prompts": len(DIMENSIONS) * prompts_per_dimension,
            "selection": "per-dimension sampling without replacement",
            "random_seed": seed,
            "evaluation_and_filename_field": "canonical_prompt",
            "official_diversity_samples_per_prompt": 20,
        },
        "dimensions": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("VBENCH2_ROOT", "/path/to/VBench-2.0")),
    )
    parser.add_argument("--prompts-per-dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prompts_per_dimension < 1:
        raise ValueError("--prompts-per-dimension must be positive")
    payload = build_payload(
        args.source_root.resolve(), args.prompts_per_dimension, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        f"Wrote {payload['protocol']['total_prompts']} prompts across "
        f"{payload['protocol']['total_dimensions']} dimensions to {args.output}"
    )


if __name__ == "__main__":
    main()
