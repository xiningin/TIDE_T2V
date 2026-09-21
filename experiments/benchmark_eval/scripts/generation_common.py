"""Shared helpers for manifest-driven benchmark video generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parents[1]
DEFAULT_REGISTRY = WORKSPACE / "config" / "generation_registry.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    # Accept both conventional JSONL and a JSON array emitted by older
    # preparation scripts (despite the .jsonl suffix).
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError(f"expected a list of JSON objects at {path}")
        return value
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_from_registry(registry_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (registry_path.parent / path).resolve()


def load_registry(path: Path = DEFAULT_REGISTRY) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    registry = read_json(resolved)
    if registry.get("schema_version") != "1.0":
        raise ValueError(f"unsupported registry schema: {registry.get('schema_version')}")
    return resolved, registry


def arm_definition(
    arm: str, registry_path: Path = DEFAULT_REGISTRY, *, require_ready: bool = True
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved, registry = load_registry(registry_path)
    arms = registry.get("arms")
    if not isinstance(arms, dict) or arm not in arms:
        raise KeyError(f"unknown arm {arm!r}")
    definition = arms[arm]
    if not isinstance(definition, dict):
        raise ValueError(f"invalid registry entry for arm {arm}")
    if require_ready and definition.get("ready") is not True:
        reason = definition.get("blocked_reason", "arm is not marked ready")
        raise RuntimeError(f"{arm} is blocked: {reason}")
    return resolved, registry, definition


def experiment_dir(arm: str, registry_path: Path = DEFAULT_REGISTRY) -> Path:
    resolved, _, definition = arm_definition(arm, registry_path, require_ready=False)
    return resolve_from_registry(resolved, str(definition["experiment_dir"]))


def merge_negative(*values: str) -> str:
    terms: list[str] = []
    lowered: set[str] = set()
    for value in values:
        for raw in value.replace(";", ",").split(","):
            term = " ".join(raw.strip().split()).rstrip(".")
            if term and term.lower() not in lowered:
                terms.append(term)
                lowered.add(term.lower())
    return ", ".join(terms)


def manifest_path(group: str) -> Path:
    return WORKSPACE / "manifests" / f"{group}.jsonl"


def output_path_for_task(root: Path, task: dict[str, Any]) -> Path:
    group = str(task["group"])
    arm = str(task["arm"])
    filename = str(task["benchmark_filename"])
    base = root / "videos" / group / arm
    if task["benchmark"] == "vbench":
        return base / filename
    if task["benchmark"] == "t2v_compbench":
        # CompBench officially expects 0001.mp4 ... 0200.mp4 per category.
        # For paired multi-seed experiments, keep each seed in a separate
        # subdirectory so that all seeds can be generated without collisions;
        # the evaluator is invoked once per seed directory.
        seed_dir = task.get("seed_dir")
        if seed_dir:
            return base / str(task["prompt_set"]) / str(seed_dir) / filename
        return base / str(task["prompt_set"]) / filename
    raise ValueError(f"unsupported benchmark: {task['benchmark']}")


def receipt_path_for_task(root: Path, task: dict[str, Any]) -> Path:
    output = output_path_for_task(root, task)
    relative = output.relative_to(root / "videos")
    return root / "receipts" / relative.with_suffix(".json")


def ensure_router_key(project_root: Path = PROJECT_ROOT) -> None:
    if os.getenv("EPICLLMROUTER_API_KEY"):
        return
    key_file = os.getenv("EPICLLMROUTER_API_KEY_FILE", "").strip()
    if key_file and Path(key_file).is_file():
        os.environ["EPICLLMROUTER_API_KEY"] = Path(key_file).read_text(encoding="utf-8").strip()
        return
    raise RuntimeError(
        "EPICLLMROUTER_API_KEY is unset. Export it, or point EPICLLMROUTER_API_KEY_FILE at a key file."
    )


def stable_job_id(task: dict[str, Any]) -> str:
    return ":".join(
        [
            str(task["group"]),
            str(task["arm"]),
            str(task["prompt_set"]),
            str(task["prompt_index"]),
            str(task["sample_index"]),
        ]
    )
