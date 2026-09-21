#!/usr/bin/env python3
"""Evaluate one prepared arm with the official VBench-2.0 implementation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_VBENCH = Path(os.environ.get("VBENCH2_ROOT", "/path/to/VBench-2.0"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--vbench-dir", type=Path, default=DEFAULT_VBENCH)
    parser.add_argument("--full-info", type=Path)
    parser.add_argument("--dimensions", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_metadata(
    manifest: list[dict[str, Any]], full_info_path: Path, metadata_root: Path
) -> dict[str, Path]:
    full_info = json.loads(full_info_path.read_text(encoding="utf-8"))
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in full_info:
        lookup[str(item["prompt_en"]).strip()].append(item)

    grouped: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in manifest:
        key = (str(row["prompt_id"]), str(row["canonical_prompt"]))
        grouped[str(row["dimension"])][key].append(row)

    metadata_paths: dict[str, Path] = {}
    for dimension, prompts in grouped.items():
        output = []
        for (_, canonical), rows in prompts.items():
            matches = [
                item
                for item in lookup.get(canonical, [])
                if dimension in item.get("dimension", [])
            ]
            if not matches:
                raise ValueError(
                    f"no exact VBench metadata match for {dimension}: {canonical}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous VBench metadata for {dimension}: {canonical}"
                )
            rows = sorted(rows, key=lambda row: int(row["sample_index"]))
            paths = [Path(str(row["video_path"])) for row in rows]
            missing = [
                str(path)
                for path in paths
                if not path.is_file() or path.stat().st_size <= 0
            ]
            if missing:
                raise FileNotFoundError("missing videos:\n" + "\n".join(missing))
            item = copy.deepcopy(matches[0])
            item["dimension"] = [dimension]
            item["video_list"] = [str(path.resolve()) for path in paths]
            output.append(item)
        path = metadata_root / f"{dimension}.json"
        atomic_json(path, output)
        metadata_paths[dimension] = path.resolve()
    return metadata_paths


def main() -> None:
    args = _parser().parse_args()
    experiment_path = args.experiment.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if args.arm not in experiment["arms"]:
        raise ValueError(f"unknown arm {args.arm!r}; available: {experiment['arms']}")
    output = Path(str(experiment["output"])).resolve()
    manifest_path = output / args.arm / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    selected = {value.strip() for value in args.dimensions.split(",") if value.strip()}
    available = {str(row["dimension"]) for row in manifest}
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown dimensions for this arm: {sorted(unknown)}")
    if selected:
        manifest = [row for row in manifest if row["dimension"] in selected]
    if not manifest:
        raise ValueError("no manifest rows selected")

    vbench_dir = args.vbench_dir.resolve()
    full_info = (
        args.full_info.resolve()
        if args.full_info
        else vbench_dir / "vbench2/VBench2_full_info.json"
    )
    if not (vbench_dir / "vbench2").is_dir() or not full_info.is_file():
        raise FileNotFoundError(
            f"invalid VBench-2.0 checkout or full-info file: {vbench_dir}"
        )
    arm_root = output / args.arm
    metadata_root = arm_root / "evaluation_metadata"
    results_root = arm_root / "evaluation_results"
    metadata_paths = build_metadata(manifest, full_info, metadata_root)

    os.chdir(vbench_dir)
    if str(vbench_dir) not in sys.path:
        sys.path.insert(0, str(vbench_dir))
    import torch
    from vbench2.utils import init_submodules

    dimensions: dict[str, Any] = {}
    errors: dict[str, str] = {}
    work_root = results_root / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    for dimension, metadata_path in metadata_paths.items():
        result_path = results_root / f"{dimension}.json"
        metadata_sha = sha256_file(metadata_path)
        if args.resume and result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("metadata_sha256") == metadata_sha:
                dimensions[dimension] = {
                    "score": previous["score"],
                    "num_videos": previous["num_videos"],
                    "result_file": str(result_path.resolve()),
                    "resumed": True,
                }
                print(f"[evaluate] resume {args.arm}/{dimension}", flush=True)
                continue
        print(f"[evaluate] {args.arm}/{dimension}", flush=True)
        try:
            # Several official VBench2 modules mutate Decord's process-global
            # bridge. Camera_Motion switches it to ``torch``, while the modules
            # that follow call ``.asnumpy()`` and therefore require the native
            # NDArray bridge. Reset before every dimension; modules that need a
            # different bridge can still set it during import.
            import decord

            decord.bridge.set_bridge("native")
            module_name = (
                "multi_view_consistency"
                if dimension == "Multi-View_Consistency"
                else dimension.lower()
            )
            module = importlib.import_module(f"vbench2.{module_name}")
            compute = getattr(module, f"compute_{module_name}")
            submodules = init_submodules([dimension], local=True)[dimension]
            if dimension == "Instance_Preservation":
                with tempfile.TemporaryDirectory(
                    prefix="instance-", dir=work_root
                ) as temp:
                    os.chdir(temp)
                    try:
                        score, video_results = compute(
                            str(metadata_path), torch.device(args.device), submodules
                        )
                    finally:
                        os.chdir(vbench_dir)
            else:
                score, video_results = compute(
                    str(metadata_path), torch.device(args.device), submodules
                )
            result = {
                "arm": args.arm,
                "dimension": dimension,
                "score": convert(score),
                "num_videos": len(video_results),
                "metadata": str(metadata_path),
                "metadata_sha256": metadata_sha,
                "video_results": convert(video_results),
            }
            atomic_json(result_path, result)
            dimensions[dimension] = {
                "score": result["score"],
                "num_videos": result["num_videos"],
                "result_file": str(result_path.resolve()),
                "resumed": False,
            }
        except Exception as exc:  # noqa: BLE001
            errors[dimension] = f"{type(exc).__name__}: {exc}"
            if not args.continue_on_error:
                raise

    summary = {
        "note": (
            "Diagnostic VBench-2.0 subset; scores are paired across arms but are "
            "not directly comparable to the official complete leaderboard."
        ),
        "arm": args.arm,
        "experiment": str(experiment_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dimensions": dimensions,
        "errors": errors,
    }
    summary_path = arm_root / "evaluation_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
