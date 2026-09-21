#!/usr/bin/env python3
"""Prepare paired raw/skill VBench-2.0 prompts and sharded SANA jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.clients.mllm_client import MLLMClient  # noqa: E402
from vista.core.prompt_rewriter import PromptRewriter  # noqa: E402
from vista.utils.config import load_config  # noqa: E402


ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULT_SUITE = ROOT / "vbench2_eval/data/vbench2_quick_eval_44.json"
DEFAULT_OUTPUT = ROOT / "outputs/vbench2_skill_eval"
DEFAULT_CONFIG = ROOT / "config/deepseek_qwen37plus_sana_posneg_train.yaml"
ROUTER_KEY_FILE_ENV = "EPICLLMROUTER_API_KEY_FILE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skill-arm",
        action="append",
        default=[],
        metavar="NAME=SKILL_DIR",
        help="Repeat for every skill checkpoint; the raw arm is always included.",
    )
    parser.add_argument("--rewrite-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--t2v-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--diversity-samples-per-prompt", type=int, default=20)
    parser.add_argument("--dimensions", default="")
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skill_sha256(skill_dir: Path) -> str:
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    rules = json.loads((skill_dir / "rules.json").read_text(encoding="utf-8"))
    payload = (
        skill.encode("utf-8")
        + b"\n"
        + json.dumps(rules, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def merge_negative(base: str, conditional: str) -> str:
    terms: list[str] = []
    lowered: set[str] = set()
    for source in (base, conditional):
        for raw in source.replace(";", ",").split(","):
            term = " ".join(raw.strip().split()).rstrip(".")
            if term and term.lower() not in lowered:
                terms.append(term)
                lowered.add(term.lower())
    return ", ".join(terms)


def safe_stem(prompt: str) -> str:
    stem = prompt[:180].strip().rstrip(".")
    if not stem or stem in {".", ".."} or any(x in stem for x in ("/", "\\", "\0")):
        return sha256_text(prompt)[:24]
    return stem


def parse_skill_arms(values: list[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not ARM_RE.fullmatch(name):
            raise ValueError(f"invalid --skill-arm {value!r}; expected NAME=SKILL_DIR")
        if name == "raw":
            raise ValueError("'raw' is reserved for the unrewritten control arm")
        path = Path(raw_path).expanduser().resolve()
        if name in arms:
            raise ValueError(f"duplicate skill arm: {name}")
        if not (path / "SKILL.md").is_file() or not (path / "rules.json").is_file():
            raise FileNotFoundError(f"invalid skill directory for {name}: {path}")
        arms[name] = path
    if not arms:
        raise ValueError("provide at least one --skill-arm NAME=SKILL_DIR")
    return arms


def load_rows(
    suite_path: Path, dimensions: str = "", max_prompts: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dimensions"), list):
        raise ValueError("suite must contain a top-level dimensions list")
    selected = {item.strip() for item in dimensions.split(",") if item.strip()}
    known = {str(block.get("dimension")) for block in payload["dimensions"]}
    unknown = selected.difference(known)
    if unknown:
        raise ValueError(
            f"unknown dimensions: {sorted(unknown)}; available: {sorted(known)}"
        )

    rows: list[dict[str, Any]] = []
    for block in payload["dimensions"]:
        dimension = str(block.get("dimension", "")).strip()
        if not dimension or (selected and dimension not in selected):
            continue
        prompts = block.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError(f"invalid prompts list for dimension {dimension}")
        for record in prompts:
            if max_prompts and len(rows) >= max_prompts:
                break
            prompt = str(record.get("canonical_prompt", "")).strip()
            prompt_id = str(record.get("id", "")).strip()
            if not prompt or not prompt_id:
                raise ValueError(f"empty id/canonical_prompt in dimension {dimension}")
            rows.append(
                {
                    "dimension": dimension,
                    "prompt_id": prompt_id,
                    "source_line": record.get("source_line"),
                    "canonical_prompt": prompt,
                }
            )
        if max_prompts and len(rows) >= max_prompts:
            break
    if not rows:
        raise ValueError("no prompts selected")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt ids must be unique in the selected suite")
    return payload, rows


def _sample_count(
    dimension: str, samples_per_prompt: int, diversity_samples_per_prompt: int
) -> int:
    if dimension == "Diversity":
        return max(samples_per_prompt, diversity_samples_per_prompt)
    return samples_per_prompt


def paired_seed(base_seed: int, prompt_ordinal: int, sample_index: int) -> int:
    """Return the arm-independent seed for one prompt/sample pair."""
    return base_seed + prompt_ordinal * 1000 + sample_index


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 1 or args.shards < 1:
        raise ValueError("workers and shards must be positive")
    if args.base_seed < 0 or args.max_prompts < 0:
        raise ValueError("base-seed and max-prompts must be non-negative")
    if args.samples_per_prompt < 1:
        raise ValueError("samples-per-prompt must be positive")
    if args.diversity_samples_per_prompt < 20:
        raise ValueError("diversity-samples-per-prompt must be at least 20")

    suite_path = args.suite.resolve()
    output = args.output.resolve()
    arms = parse_skill_arms(args.skill_arm)
    _, rows = load_rows(suite_path, args.dimensions, args.max_prompts)
    job_count_per_arm = sum(
        _sample_count(
            row["dimension"], args.samples_per_prompt, args.diversity_samples_per_prompt
        )
        for row in rows
    )
    plan = {
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "prompt_count": len(rows),
        "arms": ["raw", *arms],
        "job_count": job_count_per_arm * (len(arms) + 1),
        "shards": args.shards,
    }
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not os.getenv("EPICLLMROUTER_API_KEY"):
        key_file = os.getenv(ROUTER_KEY_FILE_ENV, "").strip()
        if not (key_file and Path(key_file).is_file()):
            raise RuntimeError(
                "EPICLLMROUTER_API_KEY is unset. Export it, or point "
                f"{ROUTER_KEY_FILE_ENV} at a key file."
            )
        os.environ["EPICLLMROUTER_API_KEY"] = Path(key_file).read_text(encoding="utf-8").strip()

    rewrite_config = load_config(args.rewrite_config)
    t2v_config = load_config(args.t2v_config)["t2v"]
    static_negative = str(t2v_config.get("negative_prompt", ""))
    rewrite_path = output / "rewrite_records.json"
    cache: dict[str, Any] = {
        "schema_version": "1.0",
        "suite_sha256": plan["suite_sha256"],
        "arms": {},
    }
    if rewrite_path.is_file():
        previous = json.loads(rewrite_path.read_text(encoding="utf-8"))
        if previous.get("suite_sha256") == plan["suite_sha256"]:
            cache = previous

    arm_metadata: dict[str, dict[str, Any]] = {}
    for name, skill_dir in arms.items():
        digest = skill_sha256(skill_dir)
        rules = json.loads((skill_dir / "rules.json").read_text(encoding="utf-8"))
        existing = cache.setdefault("arms", {}).get(name, {})
        if existing.get("skill_sha256") != digest:
            existing = {}
        records = existing.setdefault("records", {})
        existing.update(
            skill_dir=str(skill_dir),
            skill_revision=int(rules.get("revision", 0)),
            skill_sha256=digest,
        )
        cache["arms"][name] = existing
        arm_metadata[name] = {
            "skill_dir": str(skill_dir),
            "skill_revision": int(rules.get("revision", 0)),
            "skill_sha256": digest,
            "records": records,
        }

    def rewrite(
        name: str, skill_dir: Path, row: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        client = MLLMClient(rewrite_config["text_mllm"])
        result = PromptRewriter(client, skill_dir).rewrite(row["canonical_prompt"])
        return (
            name,
            row["prompt_id"],
            {
                "canonical_prompt": row["canonical_prompt"],
                "positive_prompt": result.rewritten_prompt,
                "negative_prompt": result.negative_prompt,
                "skill_revision": result.skill_revision,
                "skill_sha256": result.skill_sha256,
            },
        )

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, skill_dir in arms.items():
            records = arm_metadata[name]["records"]
            for row in rows:
                old = records.get(row["prompt_id"])
                if not old or old.get("canonical_prompt") != row["canonical_prompt"]:
                    futures.append(executor.submit(rewrite, name, skill_dir, row))
        for future in as_completed(futures):
            name, prompt_id, record = future.result()
            cache["arms"][name]["records"][prompt_id] = record
            atomic_json(rewrite_path, cache)
    atomic_json(rewrite_path, cache)

    arm_names = ["raw", *arms]
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}
    all_jobs: list[dict[str, Any]] = []
    for arm in arm_names:
        for ordinal, row in enumerate(rows, start=1):
            if arm == "raw":
                positive = row["canonical_prompt"]
                conditional_negative = ""
                revision = None
                digest = None
            else:
                record = cache["arms"][arm]["records"][row["prompt_id"]]
                positive = str(record["positive_prompt"])
                conditional_negative = str(record.get("negative_prompt") or "")
                revision = int(record["skill_revision"])
                digest = str(record["skill_sha256"])
            count = _sample_count(
                row["dimension"],
                args.samples_per_prompt,
                args.diversity_samples_per_prompt,
            )
            for sample_index in range(count):
                seed = paired_seed(args.base_seed, ordinal, sample_index)
                stem = safe_stem(row["canonical_prompt"])
                output_path = (
                    output
                    / arm
                    / "videos"
                    / row["dimension"]
                    / f"{stem}-{sample_index}.mp4"
                ).resolve()
                identity = (
                    f"psista-vbench2:{arm}:{row['prompt_id']}:{sample_index}:"
                    f"{seed}:{sha256_text(positive)}:{sha256_text(conditional_negative)}"
                )
                job_id = sha256_text(identity)[:24]
                manifest = {
                    **row,
                    "arm": arm,
                    "sample_index": sample_index,
                    "seed": seed,
                    "generation_prompt": positive,
                    "conditional_negative_prompt": conditional_negative,
                    "final_negative_prompt": merge_negative(
                        static_negative, conditional_negative
                    ),
                    "skill_revision": revision,
                    "skill_sha256": digest,
                    "video_path": str(output_path),
                    "job_id": job_id,
                }
                manifests[arm].append(manifest)
                all_jobs.append(
                    {
                        "job_id": job_id,
                        "arm": arm,
                        "dimension": row["dimension"],
                        "prompt_id": row["prompt_id"],
                        "sample_index": sample_index,
                        "seed": seed,
                        "prompt": positive,
                        "conditional_negative_prompt": conditional_negative,
                        "final_negative_prompt": manifest["final_negative_prompt"],
                        "output_path": str(output_path),
                    }
                )

    for arm, manifest in manifests.items():
        write_jsonl(output / arm / "manifest.jsonl", manifest)
    for shard in range(args.shards):
        write_jsonl(
            output / "jobs" / f"shard_{shard}.jsonl",
            [job for index, job in enumerate(all_jobs) if index % args.shards == shard],
        )
    experiment = {
        "schema_version": "1.0",
        "protocol": "paired-vbench2-raw-skill-checkpoints",
        **plan,
        "output": str(output),
        "rewrite_config": str(args.rewrite_config.resolve()),
        "t2v_config": str(args.t2v_config.resolve()),
        "t2v_config_sha256": sha256_file(args.t2v_config.resolve()),
        "static_negative_prompt": static_negative,
        "base_seed": args.base_seed,
        "samples_per_prompt": args.samples_per_prompt,
        "diversity_samples_per_prompt": args.diversity_samples_per_prompt,
        "dimensions": sorted({row["dimension"] for row in rows}),
        "skill_arms": {
            name: {key: value for key, value in metadata.items() if key != "records"}
            for name, metadata in arm_metadata.items()
        },
    }
    atomic_json(output / "experiment.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
