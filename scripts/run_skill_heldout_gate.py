#!/usr/bin/env python3
"""Evaluate incumbent and candidate rewrite skills on unseen, seed-matched videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promote_skill_patch import promote  # noqa: E402
from vista.clients.mllm_client import MLLMClient  # noqa: E402
from vista.clients.t2v_client import create_t2v_client  # noqa: E402
from vista.core.prompt_rewriter import PromptRewriter  # noqa: E402
from vista.core.selection import TournamentSelector  # noqa: E402
from vista.core.waves import gate_decision  # noqa: E402
from vista.schemas import PromptCandidate, VideoCandidate  # noqa: E402
from vista.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True)
    parser.add_argument("--prompts", required=True, help="JSON list with corpus_index and prompt.")
    parser.add_argument("--config", default="config/deepseek_qwen_sana_train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vision-gpu", default=None, help="Optional local vision GPU; omit for API evaluation.")
    parser.add_argument("--t2v-gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def outcome(pair: Any) -> str:
    if pair.decision == "B_BETTER" and pair.swapped_decision == "A_BETTER":
        return "candidate_win"
    if pair.decision == "A_BETTER" and pair.swapped_decision == "B_BETTER":
        return "incumbent_win"
    return "tie"


def main() -> None:
    args = parse_args()
    if (args.vision_gpu is not None and args.vision_gpu not in set("01234567")) or args.t2v_gpu not in set("01234567"):
        raise ValueError("Only GPU 0-7 are allowed")
    output = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "heldout_report.json"
    if args.resume and report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            print(json.dumps({key: existing[key] for key in existing if key != "records"}, ensure_ascii=False, indent=2))
            return
    base_skill = ROOT / "skills" / "rewrite-t2v-prompt"
    candidate_skill = output / "candidate_skill"
    if not candidate_skill.exists():
        shutil.copytree(base_skill, candidate_skill)
        patch_container = json.loads(Path(args.patch).read_text(encoding="utf-8"))
        patch = patch_container.get("patch", patch_container)
        promote(patch, {"passed": True, "guard_noninferior": True, "purpose": "temporary held-out candidate"}, candidate_skill)

    config = load_config(args.config)
    if args.vision_gpu is not None:
        config["vision_mllm"]["cuda_visible_devices"] = args.vision_gpu
    config["t2v"]["cuda_visible_devices"] = args.t2v_gpu
    if args.all_dry_run:
        config["t2v"]["dry_run"] = True
        for role in ("text_mllm", "vision_mllm"):
            config[role]["api_key"] = ""
            config[role]["dry_run_when_api_key_missing"] = True
    text = MLLMClient(config["text_mllm"])
    vision = MLLMClient(config["vision_mllm"])
    t2v = create_t2v_client(config["t2v"])
    selector = TournamentSelector(vision, config["selection"])
    incumbent = PromptRewriter(text, base_skill)
    challenger = PromptRewriter(text, candidate_skill)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if args.resume and report_path.exists():
        records = json.loads(report_path.read_text(encoding="utf-8")).get("records", [])
    completed = {int(record["corpus_index"]) for record in records}
    try:
        for row in prompts:
            corpus_index = int(row["corpus_index"])
            if corpus_index in completed:
                continue
            source = str(row["prompt"])
            old = incumbent.rewrite(source)
            new = challenger.rewrite(source)
            seed = 1_900_000_000 + corpus_index
            item_dir = output / f"p{corpus_index:04d}"
            old_gen = t2v.generate(old.rewritten_prompt, item_dir, "incumbent", seed=seed, negative_prompt=old.negative_prompt)
            new_gen = t2v.generate(new.rewritten_prompt, item_dir, "candidate", seed=seed, negative_prompt=new.negative_prompt)
            old_video = VideoCandidate("incumbent", PromptCandidate("incumbent", old.rewritten_prompt, 0, negative_prompt=old.negative_prompt), old_gen.video_path, 0, seed)
            new_video = VideoCandidate("candidate", PromptCandidate("candidate", new.rewritten_prompt, 0, negative_prompt=new.negative_prompt), new_gen.video_path, 0, seed)
            evaluation_error = None
            try:
                pair = selector.compare_for_improvement(source, old_video, new_video)
                comparison_outcome = outcome(pair)
                pairwise = pair.to_dict()
            except RuntimeError as exc:
                # A generated clip can be rejected by DashScope's content
                # inspection even when the MP4 itself is valid. This is not
                # evidence for either rewrite skill, so record a neutral result
                # and continue instead of aborting the entire wave.
                if "data_inspection_failed" not in str(exc).lower():
                    raise
                comparison_outcome = "tie"
                pairwise = None
                evaluation_error = {
                    "type": "dashscope_data_inspection_failed",
                    "message": "Video evaluation was rejected by provider content inspection; counted as neutral.",
                }
            records.append(
                {
                    "corpus_index": corpus_index,
                    "prompt": source,
                    "seed": seed,
                    "incumbent_prompt": old.rewritten_prompt,
                    "incumbent_negative_prompt": old.negative_prompt,
                    "candidate_prompt": new.rewritten_prompt,
                    "candidate_negative_prompt": new.negative_prompt,
                    "incumbent_video": old_gen.video_path,
                    "candidate_video": new_gen.video_path,
                    "outcome": comparison_outcome,
                    "pairwise": pairwise,
                    "evaluation_error": evaluation_error,
                }
            )
            write_json(report_path, {**gate_decision([r["outcome"] for r in records]), "records": records, "complete": False})
    finally:
        close = getattr(t2v, "close", None)
        if callable(close):
            close()
    report = {**gate_decision([r["outcome"] for r in records]), "records": records, "complete": True}
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in report if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
