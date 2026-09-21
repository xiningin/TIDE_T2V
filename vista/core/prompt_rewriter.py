from __future__ import annotations

import hashlib
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.schemas import PromptRewriteResult
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import extract_fenced_or_raw, parse_json_like


class PromptRewriter:
    """One-pass deployment rewriter backed by a versioned local skill."""

    def __init__(
        self,
        mllm: MLLMClient,
        skill_dir: str | Path | None = None,
        rules_file: str | Path | None = None,
        *,
        activation_mode: str = "normal",
        random_activation_count: int | None = None,
        random_activation_seed: int = 42,
    ):
        self.mllm = mllm
        package_root = Path(__file__).resolve().parents[2]
        requested_skill = Path(skill_dir) if skill_dir else package_root / "skills" / "rewrite-t2v-prompt"
        self.skill_dir = requested_skill if requested_skill.is_absolute() else package_root / requested_skill
        self.skill_file = self.skill_dir / "SKILL.md"
        requested_rules = Path(rules_file) if rules_file else self.skill_dir / "rules.json"
        self.rules_file = requested_rules if requested_rules.is_absolute() else package_root / requested_rules
        self.template = load_prompt("one_pass_rewrite.txt")
        if activation_mode not in {"normal", "all", "random"}:
            raise ValueError("activation_mode must be one of normal, all, random")
        if random_activation_count is not None and random_activation_count < 0:
            raise ValueError("random_activation_count must be non-negative")
        self.activation_mode = activation_mode
        self.random_activation_count = random_activation_count
        self.random_activation_seed = int(random_activation_seed)

    def rewrite(self, user_prompt: str, *, source_prompt: str = "") -> PromptRewriteResult:
        skill, rules, revision, digest = self._load_skill()
        prompt_rules, forced_rule_ids, activation_policy = self._rules_for_activation_mode(
            rules, user_prompt
        )
        prompt = render_template(
            self.template,
            input_prompt=user_prompt,
            source_prompt=source_prompt or "(none; rewrite directly from the source request)",
            rules=json.dumps(prompt_rules, ensure_ascii=False, indent=2),
            activation_policy=activation_policy,
        )
        response = self.mllm.call(prompt, expected="rewrite", temperature=0.1, thinking=False)
        rewritten, negative, activated_rule_ids = self._parse_rewrite(response.text)
        allowed_rule_ids = {
            str(item.get("rule_id"))
            for item in prompt_rules.get("modules", [])
            if isinstance(item, dict) and item.get("rule_id")
        }
        activated_rule_ids = [
            rule_id for rule_id in activated_rule_ids if rule_id in allowed_rule_ids
        ]
        global_rule_ids = [
            str(rule_id)
            for rule_id in rules.get("trigger_policy", {}).get("global_rule_ids", [])
            if str(rule_id) in allowed_rule_ids
        ]
        if forced_rule_ids is not None:
            activated_rule_ids = list(forced_rule_ids)
        else:
            activated_rule_ids = list(dict.fromkeys(global_rule_ids + activated_rule_ids))
        normalized = " ".join(rewritten.split())
        if len(normalized) < 20 or len(normalized.split()) <= 8:
            rewritten = source_prompt.strip() or user_prompt.strip()
        allowed_negative_terms = {
            str(term).strip().lower()
            for module in rules.get("modules", [])
            if isinstance(module, dict) and str(module.get("rule_id")) in activated_rule_ids
            for term in module.get("negative_terms", [])
            if str(term).strip()
        }
        negative = self._normalize_negative_prompt(negative, allowed_negative_terms)
        return PromptRewriteResult(
            user_prompt=user_prompt,
            rewritten_prompt=rewritten,
            negative_prompt=negative,
            skill_dir=str(self.skill_dir),
            skill_revision=revision,
            skill_sha256=digest,
            activated_rule_ids=activated_rule_ids,
            dry_run=response.dry_run,
        )

    def _rules_for_activation_mode(
        self, rules: dict[str, Any], user_prompt: str
    ) -> tuple[dict[str, Any], list[str] | None, str]:
        """Prepare the rule view and, for ablations, the activation assignment.

        ``all`` and ``random`` are deployment/training ablations only. They do not
        mutate the on-disk skill or its trigger contract. Random selection is
        deterministic for a prompt, which keeps retries and paired seeds stable.
        ``random_activation_count`` counts global rules as active; global rules are
        always retained and the remaining slots are sampled from conditional rules.
        """
        modules = [item for item in rules.get("modules", []) if isinstance(item, dict) and item.get("rule_id")]
        all_ids = [str(item["rule_id"]) for item in modules]
        global_ids = [
            str(rule_id)
            for rule_id in rules.get("trigger_policy", {}).get("global_rule_ids", [])
            if str(rule_id) in all_ids
        ]
        if self.activation_mode == "normal":
            return deepcopy(rules), None, (
                "Normal routing: apply only modules whose source-trigger is satisfied. "
                "Global modules remain active when their global policy says so."
            )
        if self.activation_mode == "all":
            return deepcopy(rules), list(dict.fromkeys(global_ids + all_ids)), (
                "ABLATION all-rules mode: bypass trigger routing and apply every supplied "
                "module. Return every rule_id in activated_rule_ids."
            )

        conditional_ids = [rule_id for rule_id in all_ids if rule_id not in global_ids]
        target = self.random_activation_count
        if target is None:
            raise ValueError(
                "random activation requires skill_learning.random_activation_count; "
                "set it to the Full skill mean activation count (including global rules)"
            )
        target = max(len(global_ids), min(int(target), len(all_ids)))
        digest = hashlib.sha256(
            f"{self.random_activation_seed}\0{user_prompt}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        sampled = rng.sample(conditional_ids, target - len(global_ids))
        selected = list(dict.fromkeys(global_ids + [rule_id for rule_id in all_ids if rule_id in sampled]))
        selected_set = set(selected)
        prompt_rules = deepcopy(rules)
        # Hide inactive modules from the model rather than merely asking it to
        # ignore them; this makes the random-trigger arm a true routing ablation.
        prompt_rules["modules"] = [
            module for module in prompt_rules.get("modules", [])
            if isinstance(module, dict) and str(module.get("rule_id")) in selected_set
        ]
        return prompt_rules, selected, (
            "ABLATION random-trigger mode: use only the supplied modules. "
            f"The deterministic assignment activates exactly {len(selected)} modules, "
            "including global modules. Return exactly those rule_ids in activated_rule_ids."
        )

    def _load_skill(self) -> tuple[str, dict[str, Any], int, str]:
        skill = self.skill_file.read_text(encoding="utf-8") if self.skill_file.exists() else ""
        rules: dict[str, Any] = {"schema_version": "1.0", "revision": 0, "modules": []}
        if self.rules_file.exists():
            parsed = json.loads(self.rules_file.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                rules = parsed
        revision = int(rules.get("revision", 0))
        digest = hashlib.sha256(
            skill.encode("utf-8") + b"\n" + json.dumps(rules, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return skill, rules, revision, digest

    @staticmethod
    def _parse_rewrite(text: str) -> tuple[str, str, list[str]]:
        raw = extract_fenced_or_raw(text)
        parsed = parse_json_like(raw, default=None)
        if isinstance(parsed, dict):
            negative = parsed.get("negative_prompt", "")
            negative = negative if isinstance(negative, str) else ", ".join(str(x) for x in negative) if isinstance(negative, list) else ""
            raw_activated = parsed.get("activated_rule_ids", [])
            activated = []
            if isinstance(raw_activated, list):
                for item in raw_activated:
                    rule_id = str(item).strip()
                    if rule_id and rule_id not in activated:
                        activated.append(rule_id)
            for key in ("positive_prompt", "rewritten_prompt", "prompt", "output"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip(), negative.strip(), activated
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip(), "", []
        cleaned = raw.strip()
        # Some thinking-enabled local models explain "construction" after a
        # final-section marker and put the actual deliverable in the last
        # quoted block. Prefer that block over persisting the explanation.
        standalone_quote = re.fullmatch(r'\s*["“]([^"”\n]{20,})["”]\s*', cleaned)
        if standalone_quote:
            return standalone_quote.group(1).strip(), "", []
        return cleaned.strip('"“”'), "", []

    @staticmethod
    def _normalize_negative_prompt(value: str, allowed_terms: set[str] | None = None) -> str:
        terms: list[str] = []
        for raw in re.split(r"[,;\n]+", value):
            term = " ".join(raw.strip().split())
            if allowed_terms is not None and term.lower() not in allowed_terms:
                continue
            if term and term.lower() not in {item.lower() for item in terms}:
                terms.append(term)
        return ", ".join(terms[:8])
