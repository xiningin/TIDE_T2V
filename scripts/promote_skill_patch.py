#!/usr/bin/env python3
"""Promote one distillation patch after an independent held-out gate.

The script intentionally edits only the selected TIDE skill directory. A patch
is never promoted from the training episode alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.core.trigger_contract import validate_trigger_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one held-out-gated T2V skill patch.")
    parser.add_argument("--patch", required=True, help="Distiller JSON patch.")
    parser.add_argument("--heldout-report", required=True, help="Independent report JSON with passed=true.")
    parser.add_argument("--skill-dir", default=None, help="rewrite-t2v-prompt skill directory.")
    parser.add_argument("--freeze-negative-prompt", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def promote(
    patch: dict[str, Any],
    report: dict[str, Any],
    skill_dir: Path,
    *,
    freeze_negative_prompt: bool = False,
) -> dict[str, Any]:
    if patch.get("status") != "candidate":
        raise ValueError("patch status is not candidate")
    if report.get("passed") is not True:
        raise ValueError("held-out report did not pass")
    if report.get("guard_noninferior") is False:
        raise ValueError("held-out guard non-inferiority failed")
    edit = patch.get("edit")
    if not isinstance(edit, dict):
        raise ValueError("patch has no edit")
    rules_path = skill_dir / "rules.json"
    rules = _load(rules_path)
    modules = rules.get("modules")
    if not isinstance(modules, list):
        raise ValueError("rules.json has no modules list")
    operation = edit.get("operation")
    rule_id = str(edit.get("rule_id", "")).strip()
    if operation != "replace" or not rule_id:
        raise ValueError("invalid patch operation or rule_id")
    indices = [i for i, item in enumerate(modules) if isinstance(item, dict) and item.get("rule_id") == rule_id]
    if len(indices) != 1:
        raise ValueError("replace must target exactly one existing module")
    old = modules[indices[0]]
    proposed_trigger = str(edit.get("trigger", "")).strip()
    proposed_axis = str(edit.get("trigger_axis", "")).strip()
    if proposed_axis != str(old.get("trigger_axis", "")):
        raise ValueError("replacement cannot change trigger_axis")
    trigger_validation = validate_trigger_candidate(
        rules,
        rule_id,
        proposed_trigger,
        change_rationale=str(edit.get("trigger_change_rationale", "")).strip(),
    )
    if not trigger_validation["valid"]:
        raise ValueError("invalid learned trigger: " + "; ".join(trigger_validation["errors"]))
    candidate = {
        "rule_id": rule_id,
        "trigger": proposed_trigger,
        "trigger_axis": str(old.get("trigger_axis", "")),
        "instruction": str(edit.get("instruction", "")).strip(),
        "negative_terms": (
            list(old.get("negative_terms", []))
            if freeze_negative_prompt
            else [str(x).strip() for x in edit.get("negative_terms", []) if str(x).strip()][:8]
        ),
        "capability_scope": [str(x) for x in edit.get("capability_scope", [])],
        "expected_side_effects": [str(x) for x in edit.get("expected_side_effects", [])],
    }
    if not candidate["trigger"] or not candidate["instruction"]:
        raise ValueError("patch trigger/instruction is empty")
    old_scope = old.get("capability_scope", [])
    if list(old_scope) != candidate["capability_scope"]:
        raise ValueError("replacement cannot change capability_scope")
    modules[indices[0]] = candidate
    rules["schema_version"] = "2.0"
    rules["revision"] = int(rules.get("revision", 0)) + 1
    rules["last_promotion"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rule_id": rule_id,
        "operation": operation,
        "trigger_changed": proposed_trigger != str(old.get("trigger", "")),
        "previous_trigger": str(old.get("trigger", "")),
        "promoted_trigger": proposed_trigger,
        "trigger_validation": trigger_validation,
        "heldout_report": report,
    }
    _atomic_write(rules_path, rules)
    return {"promoted": True, "revision": rules["revision"], "rule_id": rule_id, "rules_path": str(rules_path)}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    skill_dir = Path(args.skill_dir or root / "skills" / "rewrite-t2v-prompt")
    result = promote(
        _load(Path(args.patch)),
        _load(Path(args.heldout_report)),
        skill_dir,
        freeze_negative_prompt=args.freeze_negative_prompt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
