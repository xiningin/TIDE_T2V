from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.core.trigger_contract import validate_trigger_candidate
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import parse_json_like


class SkillDistiller:
    """Propose one auditable rule patch; never mutates the deployed skill."""

    def __init__(
        self,
        mllm: MLLMClient,
        skill_dir: str | Path,
        rules_file: str | Path | None = None,
        *,
        freeze_negative_prompt: bool = False,
    ):
        self.mllm = mllm
        self.skill_dir = Path(skill_dir)
        self.skill_file = self.skill_dir / "SKILL.md"
        self.rules_file = Path(rules_file) if rules_file else self.skill_dir / "rules.json"
        self.freeze_negative_prompt = bool(freeze_negative_prompt)
        self.template = load_prompt("experience_distillation.txt")
        self.batch_template = load_prompt("batch_experience_distillation.txt")
        self.trajectory_template = load_prompt("trajectory_candidate_patch.txt")
        self.rule_assessments_template = load_prompt("trajectory_rule_assessments.txt")

    def propose(self, episode: dict[str, Any]) -> dict[str, Any]:
        skill = self.skill_file.read_text(encoding="utf-8") if self.skill_file.exists() else ""
        rules = self._load_rules()
        prompt = render_template(
            self.template,
            skill=skill,
            rules=json.dumps(rules, ensure_ascii=False, indent=2),
            episode=json.dumps(episode, ensure_ascii=False, indent=2),
        )
        response = self.mllm.call(prompt, expected="skill_patch", temperature=0.1, thinking=True)
        parsed = parse_json_like(response.text, default={})
        return self._validate(parsed, episode, rules, freeze_negative_prompt=self.freeze_negative_prompt)

    def _load_rules(self) -> dict[str, Any]:
        if not self.rules_file.exists():
            return {"schema_version": "1.0", "revision": 0, "modules": []}
        parsed = json.loads(self.rules_file.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {"schema_version": "1.0", "revision": 0, "modules": []}

    @staticmethod
    def _module_hash(module: dict[str, Any]) -> str:
        payload = json.dumps(module, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def propose_trajectory_rule_assessments(
        self,
        episode: dict[str, Any],
        activated_rule_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Assess every activated rule and emit zero or one candidate per rule."""
        rules = self._load_rules()
        modules = {
            str(item.get("rule_id")): item
            for item in rules.get("modules", [])
            if isinstance(item, dict) and item.get("rule_id")
        }
        ordered_ids: list[str] = []
        for value in activated_rule_ids:
            rule_id = str(value).strip()
            if rule_id in modules and rule_id not in ordered_ids:
                ordered_ids.append(rule_id)
        if not ordered_ids:
            return []
        activated_rules = [modules[rule_id] for rule_id in ordered_ids]
        prompt = render_template(
            self.rule_assessments_template,
            episode=json.dumps(episode, ensure_ascii=False, indent=2),
            activated_rules=json.dumps(activated_rules, ensure_ascii=False, indent=2),
        )
        if self.freeze_negative_prompt:
            prompt += (
                "\n\nABLATION CONSTRAINT: negative prompt terms are frozen. "
                "Do not propose, remove, or edit negative_terms; preserve the current "
                "terms exactly and assess only positive instruction/trigger changes."
            )
        response = self.mllm.call(
            prompt,
            expected="trajectory_rule_assessments",
            temperature=0.1,
            thinking=True,
        )
        parsed = parse_json_like(response.text, default={})
        return self._normalize_rule_assessments(parsed, episode, ordered_ids, modules)

    def _normalize_rule_assessments(
        self,
        parsed: Any,
        episode: dict[str, Any],
        ordered_ids: list[str],
        modules: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        episode_id = str(episode.get("episode_id", ""))
        raw_items = parsed.get("assessments", []) if isinstance(parsed, dict) else []
        by_rule: dict[str, dict[str, Any]] = {}
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                rule_id = str(item.get("rule_id", ""))
                if rule_id in modules and rule_id in ordered_ids and rule_id not in by_rule:
                    by_rule[rule_id] = item
        normalized: list[dict[str, Any]] = []
        for rule_id in ordered_ids:
            current = modules[rule_id]
            item = by_rule.get(rule_id)
            if item is None:
                normalized.append({
                    "schema_version": "3.0",
                    "patch_id": f"{episode_id}:{rule_id}",
                    "episode_id": episode_id,
                    "rule_id": rule_id,
                    "status": "reject",
                    "reasoning": "distiller omitted the activated rule assessment",
                    "confidence": 0.0,
                    "base_rule_hash": self._module_hash(current),
                    "edit": {},
                })
                continue
            status = str(item.get("status", "reject")).strip().lower()
            if status not in {"candidate", "no_change", "misrouted", "reject"}:
                status = "reject"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            edit: dict[str, Any] = {}
            discarded_trigger_change = False
            trigger_validation: dict[str, Any] = {}
            assessment_status = status
            if status in {"candidate", "misrouted"}:
                raw_edit = item.get("edit") if isinstance(item.get("edit"), dict) else {}
                proposed_trigger = str(raw_edit.get("trigger", "")).strip() or str(current.get("trigger", ""))
                trigger_rationale = str(raw_edit.get("trigger_change_rationale", "")).strip()
                trigger_validation = validate_trigger_candidate(
                    rules=self._load_rules(),
                    rule_id=rule_id,
                    trigger=proposed_trigger,
                    change_rationale=trigger_rationale,
                )
                if not trigger_validation["valid"]:
                    discarded_trigger_change = proposed_trigger != str(current.get("trigger", ""))
                    proposed_trigger = str(current.get("trigger", ""))
                if status == "misrouted" and (
                    proposed_trigger == str(current.get("trigger", "")) or not trigger_rationale
                ):
                    status = "reject"
                instruction = (
                    str(current.get("instruction", ""))
                    if assessment_status == "misrouted"
                    else str(raw_edit.get("instruction", "")).strip()
                )
                if not instruction:
                    status = "reject"
                elif status != "reject":
                    edit = {
                        "operation": "replace",
                        "rule_id": rule_id,
                        "trigger": proposed_trigger,
                        "trigger_axis": str(current.get("trigger_axis", "")),
                        "instruction": instruction,
                        "negative_terms": list(current.get("negative_terms", []))
                        if self.freeze_negative_prompt
                        else (
                            list(current.get("negative_terms", []))
                            if assessment_status == "misrouted"
                            else [
                                str(value).strip()
                                for value in raw_edit.get("negative_terms", [])
                                if str(value).strip()
                            ][:8]
                        ),
                        "capability_scope": list(current.get("capability_scope", [])),
                        "expected_side_effects": (
                            list(current.get("expected_side_effects", []))
                            if assessment_status == "misrouted"
                            else [
                                str(value).strip()
                                for value in raw_edit.get("expected_side_effects", [])
                                if str(value).strip()
                            ]
                        ),
                        "trigger_change_rationale": trigger_rationale,
                    }
                    status = "candidate"
            normalized.append({
                "schema_version": "3.0",
                "patch_id": f"{episode_id}:{rule_id}",
                "episode_id": episode_id,
                "rule_id": rule_id,
                "status": status,
                "assessment_status": assessment_status,
                "reasoning": str(item.get("reasoning", "")).strip(),
                "confidence": confidence,
                "base_rule_hash": self._module_hash(current),
                "edit": edit,
                "trigger_changed": bool(edit) and edit.get("trigger") != current.get("trigger"),
                "discarded_trigger_change": discarded_trigger_change,
                "trigger_validation": trigger_validation,
            })
        return normalized

    def distill_rule_batch(
        self,
        patches: list[dict[str, Any]],
        *,
        apply: bool = False,
        audit_dir: str | Path | None = None,
        update_kind: str = "normal_batch_update",
    ) -> dict[str, Any]:
        """Distill one rule update from one to four same-rule trajectory patches."""
        if not 1 <= len(patches) <= 4:
            raise ValueError("A rule update requires one to four patches")
        rules = self._load_rules()
        rule_ids = {str(item.get("edit", {}).get("rule_id", "")) for item in patches}
        if len(rule_ids) != 1 or not next(iter(rule_ids)):
            raise ValueError("All four patches must target the same rule_id")
        rule_id = next(iter(rule_ids))
        current_module = next((item for item in rules.get("modules", []) if item.get("rule_id") == rule_id), None)
        if current_module is None:
            raise ValueError(f"Unknown rule_id: {rule_id}")
        instruction_warning_chars = self._instruction_warning_chars(rules)
        prompt = render_template(
            self.batch_template,
            rule_id=rule_id,
            current_module=json.dumps(current_module, ensure_ascii=False, indent=2),
            patches=json.dumps(patches, ensure_ascii=False, indent=2),
            batch_size=len(patches),
            instruction_warning_chars=instruction_warning_chars,
        )
        if self.freeze_negative_prompt:
            prompt += (
                "\n\nABLATION CONSTRAINT: negative prompt terms are frozen. "
                "Return the current module negative_terms exactly; do not update them."
            )
        response = self.mllm.call(
            prompt,
            expected="skill_rule_batch_update",
            temperature=0.1,
            thinking=True,
        )
        parsed = parse_json_like(response.text, default={})
        try:
            update = self._validate_rule_batch(parsed, rules, patches, rule_id, freeze_negative_prompt=self.freeze_negative_prompt)
        except ValueError as first_error:
            self._write_batch_audit(
                audit_dir,
                "attempt_01.json",
                response=response,
                parsed=parsed,
                thinking=True,
                valid=False,
                validation_error=str(first_error),
            )
            repair_prompt = self._batch_repair_prompt(
                response.text,
                current_module=current_module,
                patches=patches,
                rule_id=rule_id,
                instruction_warning_chars=instruction_warning_chars,
            )
            repaired = self.mllm.call(
                repair_prompt,
                expected="skill_rule_batch_update_repair",
                temperature=0.0,
                thinking=False,
            )
            repaired_parsed = parse_json_like(repaired.text, default={})
            try:
                update = self._validate_rule_batch(repaired_parsed, rules, patches, rule_id, freeze_negative_prompt=self.freeze_negative_prompt)
            except ValueError as repair_error:
                self._write_batch_audit(
                    audit_dir,
                    "attempt_02_repair.json",
                    response=repaired,
                    parsed=repaired_parsed,
                    thinking=False,
                    valid=False,
                    validation_error=str(repair_error),
                )
                raise ValueError(
                    "Batch distillation remained invalid after one non-thinking schema repair: "
                    f"{repair_error}"
                ) from repair_error
            self._write_batch_audit(
                audit_dir,
                "attempt_02_repair.json",
                response=repaired,
                parsed=repaired_parsed,
                thinking=False,
                valid=True,
                validation_error=None,
            )
        else:
            self._write_batch_audit(
                audit_dir,
                "attempt_01.json",
                response=response,
                parsed=parsed,
                thinking=True,
                valid=True,
                validation_error=None,
            )
        update["update_kind"] = update_kind
        if apply and update["status"] == "update":
            self._atomic_apply_rule(update)
        return update

    @staticmethod
    def _batch_repair_prompt(
        original_text: str,
        *,
        current_module: dict[str, Any],
        patches: list[dict[str, Any]],
        rule_id: str,
        instruction_warning_chars: int,
    ) -> str:
        evidence_ids = [str(item.get("episode_id")) for item in patches]
        required_shape = {
            "schema_version": "2.0",
            "status": "update",
            "reasoning": "recover the original response's short synthesis",
            "evidence_episode_ids": evidence_ids,
            "module": {
                "rule_id": rule_id,
                "trigger": current_module.get("trigger", ""),
                "trigger_axis": current_module.get("trigger_axis", ""),
                "instruction": current_module.get("instruction", ""),
                "negative_terms": current_module.get("negative_terms", []),
                "capability_scope": current_module.get("capability_scope", []),
                "expected_side_effects": current_module.get("expected_side_effects", []),
            },
        }
        return (
            "Repair serialization/schema only. Return exactly one JSON object and no prose or markdown. "
            "Do not create a new substantive rule proposal: recover the intended fields from the original "
            "response, use the current rule as the fallback for missing module fields, keep rule_id, "
            "trigger_axis, and capability_scope fixed, and cite exactly the required episode IDs. "
            "There is no hard character limit on the replacement instruction. Preserve all "
            "evidence-supported behavior during schema repair; remove content only if it is "
            f"duplicated or contradictory. A length above {instruction_warning_chars} characters "
            "is an audit warning, not a validation failure.\n\n"
            f"Required rule_id: {json.dumps(rule_id, ensure_ascii=False)}\n"
            f"Required episode IDs: {json.dumps(evidence_ids, ensure_ascii=False)}\n"
            f"Required shape and safe fallback values:\n{json.dumps(required_shape, ensure_ascii=False, indent=2)}\n\n"
            f"Original response to serialize:\n{original_text}"
        )

    @staticmethod
    def _write_batch_audit(
        audit_dir: str | Path | None,
        filename: str,
        *,
        response: Any,
        parsed: Any,
        thinking: bool,
        valid: bool,
        validation_error: str | None,
    ) -> None:
        if audit_dir is None:
            return
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thinking": thinking,
            "valid": valid,
            "validation_error": validation_error,
            "response_text": response.text,
            "raw_response": response.raw,
            "usage": response.usage,
            "elapsed_seconds": response.elapsed_seconds,
            "base_url": response.base_url,
            "parsed": parsed,
        }
        temporary = (destination / filename).with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(destination / filename)

    def propose_trajectory_patch(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Compress one full two-round trajectory without mutating the Skill."""
        rules = self._load_rules()
        existing_rules = [
            {
                "rule_id": str(item.get("rule_id", "")),
                "trigger": str(item.get("trigger", "")),
                "trigger_axis": str(item.get("trigger_axis", "")),
                "instruction": str(item.get("instruction", "")),
            }
            for item in rules.get("modules", [])
            if isinstance(item, dict) and item.get("rule_id")
        ]
        prompt = render_template(
            self.trajectory_template,
            episode=json.dumps(episode, ensure_ascii=False, indent=2),
            existing_rules=json.dumps(existing_rules, ensure_ascii=False, indent=2),
        )
        response = self.mllm.call(
            prompt,
            expected="trajectory_patch",
            temperature=0.1,
            thinking=True,
        )
        parsed = parse_json_like(response.text, default={})
        allowed_ids = {item["rule_id"] for item in existing_rules}
        parsed_edit = parsed.get("edit") if isinstance(parsed, dict) else None
        parsed_rule_id = str(parsed_edit.get("rule_id", "")) if isinstance(parsed_edit, dict) else ""
        if isinstance(parsed, dict) and parsed.get("status") == "candidate" and parsed_rule_id not in allowed_ids:
            repair_prompt = (
                "Repair the candidate patch below. Return one JSON object only. "
                "Keep the evidence-grounded edit, but set edit.rule_id to exactly one allowed ID. "
                "Do not invent or rename IDs. If no allowed rule is supported, return status=reject "
                "with the same episode_id and an empty edit.\n\n"
                f"Allowed rule IDs: {json.dumps(sorted(allowed_ids), ensure_ascii=False)}\n"
                f"Episode ID: {json.dumps(str(episode.get('episode_id')))}\n"
                f"Candidate: {json.dumps(parsed, ensure_ascii=False, indent=2)}"
            )
            repaired = self.mllm.call(
                repair_prompt,
                expected="trajectory_patch_repair",
                temperature=0.0,
                thinking=False,
            )
            parsed = parse_json_like(repaired.text, default={})
        if not isinstance(parsed, dict):
            return self._reject_trajectory_patch(episode, "invalid distillation JSON")
        if parsed.get("status") == "reject":
            return self._reject_trajectory_patch(episode, str(parsed.get("reasoning", "insufficient evidence")))
        if parsed.get("status") != "candidate":
            return self._reject_trajectory_patch(episode, "distiller did not return candidate|reject")
        if str(parsed.get("episode_id")) != str(episode.get("episode_id")):
            return self._reject_trajectory_patch(episode, "candidate patch cited the wrong episode")
        edit = parsed.get("edit")
        if not isinstance(edit, dict):
            return self._reject_trajectory_patch(episode, "candidate patch did not contain one edit")
        rule_id = str(edit.get("rule_id", ""))
        current = next((item for item in rules.get("modules", []) if item.get("rule_id") == rule_id), None)
        if current is None:
            return self._reject_trajectory_patch(episode, "candidate patch still targeted an unknown rule_id after repair")
        proposed_trigger = str(edit.get("trigger", "")).strip() or str(current.get("trigger", ""))
        trigger_rationale = str(edit.get("trigger_change_rationale", "")).strip()
        trigger_validation = validate_trigger_candidate(
            rules, rule_id, proposed_trigger, change_rationale=trigger_rationale
        )
        if not trigger_validation["valid"]:
            proposed_trigger = str(current.get("trigger", ""))
        return {
            "schema_version": "2.0",
            "status": "candidate",
            "episode_id": str(episode.get("episode_id")),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "edit": {
                "operation": "replace",
                "rule_id": rule_id,
                "trigger": proposed_trigger,
                "trigger_axis": str(current.get("trigger_axis", "")),
                "instruction": str(edit.get("instruction", "")).strip() or str(current.get("instruction", "")),
                "negative_terms": [str(x).strip() for x in edit.get("negative_terms", []) if str(x).strip()][:8],
                "capability_scope": list(current.get("capability_scope", [])),
                "expected_side_effects": [str(x) for x in edit.get("expected_side_effects", [])],
                "trigger_change_rationale": trigger_rationale,
            },
            "trigger_changed": proposed_trigger != str(current.get("trigger", "")),
            "discarded_trigger_change": bool(
                str(edit.get("trigger", "")).strip()
                and str(edit.get("trigger", "")).strip() != str(current.get("trigger", ""))
                and not trigger_validation["valid"]
            ),
            "trigger_validation": trigger_validation,
        }

    @staticmethod
    def _reject_trajectory_patch(episode: dict[str, Any], reasoning: str) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "status": "reject",
            "episode_id": str(episode.get("episode_id")),
            "reasoning": reasoning.strip() or "insufficient evidence",
            "edit": {},
        }

    @staticmethod
    def _validate_rule_batch(
        parsed: Any,
        current: dict[str, Any],
        patches: list[dict[str, Any]],
        rule_id: str,
        *,
        freeze_negative_prompt: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(parsed, dict) or parsed.get("status") != "update":
            raise ValueError("Batch distiller did not return status=update")
        module = parsed.get("module")
        if not isinstance(module, dict) or str(module.get("rule_id")) != rule_id:
            raise ValueError("Batch update must return only the selected rule")
        instruction = str(module.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("Updated rule requires an instruction")
        instruction_warning_chars = SkillDistiller._instruction_warning_chars(current)
        if any(word in instruction.lower() for word in ("vbench", "benchmark", "video_path", "seed")):
            raise ValueError("Evaluator-specific leakage in updated rule")
        evidence_ids = {str(x) for x in parsed.get("evidence_episode_ids", [])}
        valid_ids = {str(x.get("episode_id")) for x in patches}
        if evidence_ids != valid_ids:
            raise ValueError("Rule update must cite every selected patch")
        old = next(item for item in current["modules"] if item.get("rule_id") == rule_id)
        proposed_trigger = str(module.get("trigger", "")).strip()
        proposed_axis = str(module.get("trigger_axis", "")).strip()
        old_trigger = str(old.get("trigger", ""))
        source_trigger_changes = [
            item
            for item in patches
            if str(item.get("edit", {}).get("trigger", "")).strip()
            and str(item.get("edit", {}).get("trigger", "")).strip() != old_trigger
            and str(item.get("edit", {}).get("trigger_change_rationale", "")).strip()
        ]
        support_count = len(source_trigger_changes)
        minimum_support = int(current.get("trigger_policy", {}).get("min_batch_change_support", 2))
        requested_trigger = proposed_trigger or old_trigger
        trigger_validation = validate_trigger_candidate(
            current,
            rule_id,
            requested_trigger,
            change_rationale=str(parsed.get("reasoning", "")).strip(),
        )
        accept_trigger_change = (
            requested_trigger != old_trigger
            and support_count >= minimum_support
            and trigger_validation["valid"]
        )
        normalized_trigger = requested_trigger if accept_trigger_change else old_trigger
        normalized = {
            "rule_id": rule_id,
            "trigger": normalized_trigger,
            "trigger_axis": str(old.get("trigger_axis", "")),
            "instruction": instruction,
            "negative_terms": (
                list(old.get("negative_terms", []))
                if freeze_negative_prompt
                else [str(x).strip() for x in module.get("negative_terms", []) if str(x).strip()][:8]
            ),
            "capability_scope": list(old.get("capability_scope", [])),
            "expected_side_effects": [str(x) for x in module.get("expected_side_effects", [])],
        }
        status = "no_op_consumed" if normalized == {
            key: old.get(key)
            for key in (
                "rule_id",
                "trigger",
                "trigger_axis",
                "instruction",
                "negative_terms",
                "capability_scope",
                "expected_side_effects",
            )
        } else "update"
        base_revision = int(current.get("revision", 0))
        return {
            "schema_version": "2.0",
            "status": status,
            "base_revision": base_revision,
            "revision": base_revision + (1 if status == "update" else 0),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "instruction_length": len(instruction),
            "instruction_warning_chars": instruction_warning_chars,
            "instruction_length_warning": len(instruction) > instruction_warning_chars,
            "evidence_episode_ids": sorted(evidence_ids),
            "rule_id": rule_id,
            "trigger_preserved": normalized_trigger == old_trigger,
            "trigger_changed": normalized_trigger != old_trigger,
            "trigger_support_count": support_count,
            "minimum_trigger_support": minimum_support,
            "discarded_trigger_change": bool(
                (proposed_trigger and proposed_trigger != normalized_trigger)
                or (proposed_axis and proposed_axis != normalized["trigger_axis"])
            ),
            "trigger_validation": trigger_validation,
            "module": normalized,
        }

    @staticmethod
    def _instruction_warning_chars(rules: dict[str, Any]) -> int:
        """Return a soft readability warning threshold, never a hard update limit.

        Older run-local skills stored ``max_instruction_chars``.  It is deliberately
        interpreted as the warning threshold for backward compatibility; no learned
        instruction is rejected solely because it exceeds that value.
        """
        policy = rules.get("update_policy") if isinstance(rules, dict) else {}
        if not isinstance(policy, dict):
            policy = {}
        raw = policy.get("instruction_warning_chars", policy.get("max_instruction_chars", 8000))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 8000
        return max(1, value)

    def _atomic_apply_rule(self, update: dict[str, Any]) -> None:
        current = self._load_rules()
        if int(current.get("revision", 0)) != int(update["base_revision"]):
            raise RuntimeError("Skill revision changed during batch distillation")
        modules = list(current.get("modules", []))
        indices = [i for i, item in enumerate(modules) if item.get("rule_id") == update["rule_id"]]
        if len(indices) != 1:
            raise RuntimeError("Selected rule disappeared during batch distillation")
        modules[indices[0]] = update["module"]
        value = {**current, "schema_version": "2.0", "revision": int(update["revision"]), "modules": modules,
            "last_batch_update": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_size": len(update["evidence_episode_ids"]),
                "update_kind": update.get("update_kind", "normal_batch_update"),
                "evidence_episode_ids": update["evidence_episode_ids"],
                "rule_id": update["rule_id"],
                "reasoning": update["reasoning"],
                "instruction_length": update.get("instruction_length"),
                "instruction_warning_chars": update.get("instruction_warning_chars"),
                "instruction_length_warning": update.get("instruction_length_warning", False),
                "trigger_changed": update.get("trigger_changed", False),
                "trigger_support_count": update.get("trigger_support_count", 0),
                "trigger_validation": update.get("trigger_validation", {}),
            },
        }
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.rules_file.name}.", dir=str(self.rules_file.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(name, self.rules_file)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _validate(
        parsed: Any,
        episode: dict[str, Any],
        rules: dict[str, Any],
        *,
        freeze_negative_prompt: bool = False,
    ) -> dict[str, Any]:
        reject = {
            "schema_version": "1.0",
            "status": "reject",
            "reasoning": "insufficient causal evidence",
            "edit": {},
        }
        if not isinstance(parsed, dict) or episode.get("status") != "accepted":
            return reject
        edit = parsed.get("edit")
        if parsed.get("status") != "candidate" or not isinstance(edit, dict):
            return reject
        operation = str(edit.get("operation", ""))
        rule_id = str(edit.get("rule_id", "")).strip()
        instruction = str(edit.get("instruction", "")).strip()
        negative_terms = [str(x).strip() for x in edit.get("negative_terms", []) if str(x).strip()]
        current = next(
            (item for item in rules.get("modules", []) if isinstance(item, dict) and item.get("rule_id") == rule_id),
            None,
        )
        if operation != "replace" or not rule_id or not instruction or current is None:
            return reject
        if not edit.get("evidence_episode_ids"):
            return reject
        forbidden = ("vbench", "benchmark", "judge", "seed", "video_path")
        if any(term in instruction.lower() for term in forbidden):
            return reject
        proposed_trigger = str(edit.get("trigger", "")).strip() or str(current.get("trigger", ""))
        trigger_rationale = str(edit.get("trigger_change_rationale", "")).strip()
        trigger_validation = validate_trigger_candidate(
            rules, rule_id, proposed_trigger, change_rationale=trigger_rationale
        )
        if not trigger_validation["valid"]:
            proposed_trigger = str(current.get("trigger", ""))
        return {
            "schema_version": "1.0",
            "status": "candidate",
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "edit": {
                "operation": "replace",
                "rule_id": rule_id,
                "trigger": proposed_trigger,
                "trigger_axis": str(current.get("trigger_axis", "")),
                "instruction": instruction,
                "negative_terms": (
                    list(current.get("negative_terms", []))
                    if freeze_negative_prompt
                    else negative_terms[:8]
                ),
                "capability_scope": list(current.get("capability_scope", [])),
                "evidence_episode_ids": [str(x) for x in edit.get("evidence_episode_ids", [])],
                "expected_side_effects": [str(x) for x in edit.get("expected_side_effects", [])],
                "guard_metric": str(edit.get("guard_metric", "")),
                "trigger_change_rationale": trigger_rationale,
            },
            "trigger_changed": proposed_trigger != str(current.get("trigger", "")),
            "trigger_validation": trigger_validation,
        }
