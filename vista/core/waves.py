from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vista.core.trigger_contract import validate_trigger_candidate


def prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        prompt = " ".join(raw.strip().split())
        if not prompt:
            continue
        rows.append(
            {
                "corpus_index": len(rows),
                "line_number": line_number,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def is_promotion_prompt(row: dict[str, Any], fraction: float, seed: str) -> bool:
    cutoff = int(fraction * 10_000)
    bucket = int(hashlib.sha256(f"{seed}\0{row['prompt']}".encode()).hexdigest()[:8], 16) % 10_000
    return bucket < cutoff


def skill_digest(skill_dir: Path) -> tuple[int, str]:
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    rules = json.loads((skill_dir / "rules.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        skill.encode("utf-8") + b"\n" + json.dumps(rules, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return int(rules.get("revision", 0)), digest


def collect_candidates(wave_dir: Path, excluded_corpus_indices: set[int] | None = None) -> list[dict[str, Any]]:
    excluded = excluded_corpus_indices or set()
    candidates: list[dict[str, Any]] = []
    for status_path in sorted(wave_dir.glob("shard_*/items/*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        global_index = status.get("global_corpus_index", status.get("corpus_index"))
        if status.get("state") != "completed" or global_index in excluded:
            continue
        for learned in status.get("learned", []):
            episode = learned.get("episode") or {}
            patch = learned.get("patch") or {}
            if episode.get("status") == "accepted" and patch.get("status") == "candidate":
                candidates.append(
                    {
                        "item_id": status.get("item_id"),
                        "global_corpus_index": global_index,
                        "prompt_sha256": status.get("prompt_sha256"),
                        "episode_id": episode.get("episode_id"),
                        "patch": patch,
                    }
                )
    return candidates


def normalize_patch_scope(patch: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Keep a valid learned trigger while restoring the immutable axis and scope."""
    value = json.loads(json.dumps(patch))
    edit = value.get("edit") or {}
    if edit.get("operation") != "replace":
        return value
    rule_id = edit.get("rule_id")
    current = next((m for m in rules.get("modules", []) if m.get("rule_id") == rule_id), None)
    if current is None:
        raise ValueError(f"Candidate replaces unknown rule: {rule_id}")
    proposed_trigger = str(edit.get("trigger", "")).strip() or str(current.get("trigger", ""))
    trigger_validation = validate_trigger_candidate(
        rules,
        str(rule_id),
        proposed_trigger,
        change_rationale=str(edit.get("trigger_change_rationale", "")).strip(),
    )
    edit["trigger"] = proposed_trigger if trigger_validation["valid"] else str(current.get("trigger", ""))
    edit["trigger_axis"] = str(current.get("trigger_axis", ""))
    edit["capability_scope"] = list(current.get("capability_scope", []))
    value["trigger_changed"] = edit["trigger"] != str(current.get("trigger", ""))
    value["discarded_trigger_change"] = proposed_trigger != edit["trigger"]
    value["trigger_validation"] = trigger_validation
    return value


def representative_candidate(candidates: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any] | None:
    """Choose one complete candidate deterministically.

    Cross-episode support chooses the rule family. Within that family, the shortest
    complete instruction is preferred as a deterministic tie-breaker; this is not a
    hard length bound and long evidence-supported instructions remain eligible.
    """
    if not candidates:
        return None
    family_counts = Counter(str(c["patch"].get("edit", {}).get("rule_id", "")) for c in candidates)
    family = sorted(family_counts, key=lambda key: (-family_counts[key], key))[0]
    family_items = [c for c in candidates if c["patch"].get("edit", {}).get("rule_id") == family]
    chosen = min(
        family_items,
        key=lambda c: (
            len(str(c["patch"].get("edit", {}).get("instruction", "")))
            + sum(len(str(x)) for x in c["patch"].get("edit", {}).get("negative_terms", [])),
            str(c.get("episode_id", "")),
        ),
    )
    result = json.loads(json.dumps(chosen))
    result["patch"] = normalize_patch_scope(result["patch"], rules)
    result["selection"] = {
        "method": "majority_rule_family_then_shortest_complete_instruction",
        "family_support": family_counts[family],
        "candidate_count": len(candidates),
        "counts_by_rule": dict(family_counts),
    }
    return result


def gate_decision(outcomes: list[str], max_loss_fraction: float = 0.2) -> dict[str, Any]:
    counts = Counter(outcomes)
    n = len(outcomes)
    wins = counts["candidate_win"]
    losses = counts["incumbent_win"]
    allowed_losses = int(n * max_loss_fraction)
    guard = losses <= allowed_losses
    passed = n > 0 and wins > losses and guard
    return {
        "passed": passed,
        "guard_noninferior": guard,
        "prompt_count": n,
        "candidate_wins": wins,
        "incumbent_wins": losses,
        "ties": counts["tie"],
        "allowed_losses": allowed_losses,
        "macro_delta": (wins - losses) / n if n else 0.0,
        "target_delta": wins / n if n else 0.0,
    }
