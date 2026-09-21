#!/usr/bin/env python3
"""Combine paired arm-level VBench-2.0 summaries and compute deltas from raw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment_path = args.experiment.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    root = Path(str(experiment["output"])).resolve()
    arms = [str(value) for value in experiment["arms"]]
    summaries = {}
    for arm in arms:
        path = root / arm / "evaluation_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation summary for {arm}: {path}")
        summaries[arm] = json.loads(path.read_text(encoding="utf-8"))
        if summaries[arm].get("errors"):
            raise ValueError(f"evaluation for {arm} contains errors")

    reference_dimensions = set(summaries[arms[0]]["dimensions"])
    for arm in arms[1:]:
        if set(summaries[arm]["dimensions"]) != reference_dimensions:
            raise ValueError(f"evaluated dimensions for {arm} do not match {arms[0]}")

    all_dimensions = sorted(
        {dimension for value in summaries.values() for dimension in value["dimensions"]}
    )
    rows = []
    for dimension in all_dimensions:
        scores = {
            arm: summaries[arm]["dimensions"].get(dimension, {}).get("score")
            for arm in arms
        }
        raw = scalar(scores.get("raw"))
        deltas = {
            arm: (
                scalar(score) - raw
                if scalar(score) is not None and raw is not None
                else None
            )
            for arm, score in scores.items()
            if arm != "raw"
        }
        rows.append({"dimension": dimension, "scores": scores, "delta_vs_raw": deltas})

    means: dict[str, float | None] = {}
    for arm in arms:
        values = [scalar(row["scores"].get(arm)) for row in rows]
        numeric = [value for value in values if value is not None]
        means[arm] = sum(numeric) / len(numeric) if numeric else None
    raw_mean = means.get("raw")
    mean_deltas = {
        arm: (value - raw_mean if value is not None and raw_mean is not None else None)
        for arm, value in means.items()
        if arm != "raw"
    }
    output = {
        "note": (
            "Unweighted means are diagnostic only. Per-dimension paired deltas are "
            "the primary comparison; this subset is not an official leaderboard score."
        ),
        "experiment": str(experiment_path),
        "arms": arms,
        "rows": rows,
        "unweighted_dimension_means": means,
        "mean_delta_vs_raw": mean_deltas,
    }
    output_path = (
        args.output.resolve() if args.output else root / "comparison_summary.json"
    )
    atomic_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
