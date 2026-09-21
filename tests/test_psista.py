from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import requests

from vista.clients.mllm_client import MLLMClient, MLLMResponse
from vista.core.agent import VistaAgent
from vista.core.dtpa import DTPAOptimizer
from vista.core.distillation import SkillDistiller
from vista.core.prompt_rewriter import PromptRewriter
from vista.clients.t2v_client import create_t2v_client
from vista.utils.config import load_config
from vista.utils.json_utils import parse_json_like
from vista.core.waves import gate_decision, normalize_patch_scope, representative_candidate
from scripts.promote_skill_patch import promote
from scripts.run_batch_skill_controller import (
    _consume_patch_batch,
    _ingest_patches,
    _load_patch_pool,
    _new_patch_pool,
    _select_eligible_rule,
    finalize_patch_pool,
    initialize_working_skill,
    parse_sana_gpus,
    write_shard_prompts,
)
from scripts.run_training_corpus import (
    _attempt_run_dir,
    _is_data_inspection_failure,
    _prior_inspection_attempts,
    _resume_completed_search_distillation,
)
from scripts.supervise_batch_skill_controller import retry_delay, training_complete
from scripts.run_wave_controller import SANA_GPUS


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "rewrite-t2v-prompt"


class FakeMLLM:
    dry_run = False
    calls = []

    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def call(self, prompt: str, **kwargs):
        self.calls.append(kwargs.get("expected"))
        return MLLMResponse(text=self.text, dry_run=False)


class SequenceMLLM:
    dry_run = False

    def __init__(self, texts: list[str]):
        self.texts = iter(texts)
        self.calls = []

    def call(self, prompt: str, **kwargs):
        self.calls.append(kwargs)
        return MLLMResponse(text=next(self.texts), dry_run=False)


class PromptSkillTests(unittest.TestCase):
    def test_router_rate_limit_uses_provider_scale_backoff(self):
        client = MLLMClient({
            "provider": "openai_compatible",
            "api_key": "test-only",
            "api_key_required": True,
            "dry_run_when_api_key_missing": False,
            "retry_backoff_seconds": 1,
            "transient_retry_backoff_seconds": 60,
            "max_retry_backoff_seconds": 300,
        })
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "75"
        error = requests.HTTPError(response=response)

        self.assertEqual(client._retry_delay_seconds(0, error), 75)
        self.assertEqual(client._retry_delay_seconds(1, error), 120)
        self.assertEqual(client._retry_delay_seconds(10, error), 300)

        bad_request = requests.Response()
        bad_request.status_code = 400
        self.assertEqual(
            client._retry_delay_seconds(1, requests.HTTPError(response=bad_request)),
            2,
        )

    def test_data_inspection_retry_reduces_each_video_frame_group(self):
        groups = [
            [{"type": "image_url", "id": f"a{index}"} for index in range(10)],
            [{"type": "image_url", "id": f"b{index}"} for index in range(10)],
        ]
        prompt = {"type": "text", "text": "compare"}

        retry_one = MLLMClient._inspection_retry_content(groups, prompt, retry_number=1)
        retry_two = MLLMClient._inspection_retry_content(groups, prompt, retry_number=2)
        retry_four = MLLMClient._inspection_retry_content(groups, prompt, retry_number=4)

        self.assertEqual(sum(item["type"] == "image_url" for item in retry_one), 10)
        self.assertEqual(sum(item["type"] == "image_url" for item in retry_two), 6)
        self.assertEqual(sum(item["type"] == "image_url" for item in retry_four), 2)
        self.assertIs(retry_one[-1], prompt)

        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error":"InternalError.Algo.DataInspectionFailed"}'
        self.assertTrue(MLLMClient._is_data_inspection_error(requests.HTTPError(response=response)))

    def test_dtpa_accepts_common_candidate_wrapper(self):
        wrapped = json.dumps({"candidates": [
            {"positive_prompt": "A detailed continuous shot of a fox running across a snowy field.", "negative_prompt": "jitter"},
            {"positive_prompt": "A wide shot follows a fox running smoothly through deep snow.", "negative_prompt": "ghosting"},
        ]})
        items = DTPAOptimizer._candidate_items(wrapped)
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0]["positive_prompt"].startswith("A detailed"))

    def test_deepseek_reasoning_content_gets_nonthinking_finalizer(self):
        client = MLLMClient({
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-v4-flash",
            "api_key": "test-only",
            "api_key_required": True,
            "dry_run_when_api_key_missing": False,
            "max_tokens": 8192,
        })
        thinking_response = mock.Mock()
        thinking_response.raise_for_status.return_value = None
        thinking_response.json.return_value = {
            "choices": [{"message": {"content": None, "reasoning_content": "analysis"}}],
            "usage": {"completion_tokens": 8192},
        }
        final_response = mock.Mock()
        final_response.raise_for_status.return_value = None
        final_response.json.return_value = {
            "choices": [{"message": {"content": '[{"positive_prompt":"A complete detailed video prompt.","negative_prompt":"jitter"}]'}}],
            "usage": {"completion_tokens": 32},
        }
        with mock.patch(
            "vista.clients.mllm_client.requests.post",
            side_effect=[thinking_response, final_response],
        ) as post:
            result = client.call("return JSON", thinking=True)
        self.assertIn("positive_prompt", result.text)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["thinking"], {"type": "disabled"})

    def test_vllm_empty_content_and_reasoning_gets_nonthinking_finalizer(self):
        client = MLLMClient({
            "provider": "vllm",
            "base_url": "http://127.0.0.1:30000/v1",
            "model": "qwen-test",
            "api_key_required": False,
            "dry_run_when_api_key_missing": False,
            "enable_thinking": True,
            "preserve_thinking": False,
            "max_tokens": 8192,
        })
        empty_response = mock.Mock()
        empty_response.raise_for_status.return_value = None
        empty_response.json.return_value = {
            "choices": [{"message": {"content": None, "reasoning_content": None}}],
            "usage": {"completion_tokens": 8192},
        }
        final_response = mock.Mock()
        final_response.raise_for_status.return_value = None
        final_response.json.return_value = {
            "choices": [{"message": {"content": '{"winner":"A"}'}}],
            "usage": {"completion_tokens": 8},
        }
        with mock.patch(
            "vista.clients.mllm_client.requests.post",
            side_effect=[empty_response, final_response],
        ) as post:
            result = client.call("choose A or B")
        self.assertEqual(result.text, '{"winner":"A"}')
        self.assertEqual(post.call_count, 2)
        final_payload = post.call_args.kwargs["json"]
        self.assertFalse(final_payload["chat_template_kwargs"]["enable_thinking"])
        self.assertIn("previous long-thinking pass returned no serializable", final_payload["messages"][0]["content"][-1]["text"].lower())

    def test_vllm_per_call_thinking_override_disables_thinking(self):
        client = MLLMClient({
            "provider": "vllm",
            "base_url": "http://127.0.0.1:30000/v1",
            "model": "qwen-test",
            "api_key_required": False,
            "dry_run_when_api_key_missing": False,
            "enable_thinking": True,
            "preserve_thinking": True,
        })
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"Decision":"A_BETTER"}'}}],
            "usage": {"completion_tokens": 8},
        }
        with mock.patch("vista.clients.mllm_client.requests.post", return_value=response) as post:
            client.call("compare", thinking=False)
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(payload["chat_template_kwargs"]["preserve_thinking"])

    def test_wave_candidate_selection_prefers_supported_family_and_preserves_scope(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        learned_trigger = "The source requests an observable motion trajectory or a gradual physical state transition."
        candidates = []
        for episode, rule_id, instruction in (
            ("e1", "vista.motion_physics", "A short local motion rule."),
            ("e2", "vista.motion_physics", "A substantially longer local motion rule with excess detail."),
            ("e3", "vista.scene_components", "A camera rule."),
        ):
            candidates.append(
                {
                    "episode_id": episode,
                    "patch": {
                        "status": "candidate",
                        "edit": {
                            "operation": "replace",
                            "rule_id": rule_id,
                            "trigger": learned_trigger if rule_id == "vista.motion_physics" else "Missing shot presentation details.",
                            "trigger_change_rationale": "The rule missed a source-supported motion trajectory.",
                            "instruction": instruction,
                            "capability_scope": ["invalid_scope"],
                        },
                    },
                }
        )
        selected = representative_candidate(candidates, rules)
        self.assertEqual(selected["patch"]["edit"]["rule_id"], "vista.motion_physics")
        self.assertEqual(selected["patch"]["edit"]["capability_scope"], current["capability_scope"])
        self.assertEqual(selected["patch"]["edit"]["trigger"], learned_trigger)
        self.assertEqual(selected["patch"]["edit"]["trigger_axis"], current["trigger_axis"])
        self.assertTrue(selected["patch"]["trigger_changed"])

    def test_batch_distillation_learns_supported_trigger_and_freezes_axis(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        learned_trigger = "The source requests an observable motion trajectory or a gradual physical state transition."
        patches = [
            {
                "episode_id": f"e{index}",
                "edit": {
                    "rule_id": "vista.motion_physics",
                    "trigger": learned_trigger if index < 2 else current["trigger"],
                    "trigger_change_rationale": "The rule missed a source-supported motion trajectory." if index < 2 else "",
                },
            }
            for index in range(4)
        ]
        parsed = {
            "status": "update",
            "reasoning": "Two independent patches identify the same missed motion-routing boundary.",
            "evidence_episode_ids": [f"e{index}" for index in range(4)],
            "module": {
                "rule_id": "vista.motion_physics",
                "trigger": learned_trigger,
                "trigger_axis": current["trigger_axis"],
                "instruction": "Describe observable motion without adding events.",
                "negative_terms": [],
                "capability_scope": current["capability_scope"],
                "expected_side_effects": [],
            },
        }
        update = SkillDistiller._validate_rule_batch(parsed, rules, patches, "vista.motion_physics")
        self.assertEqual(update["module"]["trigger"], learned_trigger)
        self.assertEqual(update["module"]["trigger_axis"], current["trigger_axis"])
        self.assertFalse(update["trigger_preserved"])
        self.assertTrue(update["trigger_changed"])
        self.assertEqual(update["trigger_support_count"], 2)
        self.assertFalse(update["discarded_trigger_change"])

    def test_batch_distillation_discards_trigger_change_without_repeated_support(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        learned_trigger = "The source requests an observable motion trajectory or a gradual physical state transition."
        patches = [
            {
                "episode_id": f"e{index}",
                "edit": {
                    "rule_id": "vista.motion_physics",
                    "trigger": learned_trigger if index == 0 else current["trigger"],
                    "trigger_change_rationale": "The rule missed a source-supported motion trajectory." if index == 0 else "",
                },
            }
            for index in range(4)
        ]
        parsed = {
            "status": "update",
            "evidence_episode_ids": [f"e{index}" for index in range(4)],
            "module": {
                **current,
                "trigger": learned_trigger,
                "instruction": "Describe observable motion without adding events.",
            },
        }
        update = SkillDistiller._validate_rule_batch(parsed, rules, patches, "vista.motion_physics")
        self.assertEqual(update["module"]["trigger"], current["trigger"])
        self.assertTrue(update["trigger_preserved"])
        self.assertFalse(update["trigger_changed"])
        self.assertEqual(update["trigger_support_count"], 1)
        self.assertTrue(update["discarded_trigger_change"])

    def test_batch_distillation_accepts_long_instruction_with_soft_warning(self):
        rules = json.loads((SKILL / "initializations" / "posneg_v1_1.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        long_instruction = "Describe the supported observable motion and preserve the requested event. " + ("Add evidence-backed continuity detail. " * 220)
        patches = [
            {
                "episode_id": f"long-e{index}",
                "edit": {"rule_id": "vista.motion_physics", "trigger": current["trigger"]},
            }
            for index in range(4)
        ]
        parsed = {
            "status": "update",
            "reasoning": "Long evidence-supported synthesis.",
            "evidence_episode_ids": [item["episode_id"] for item in patches],
            "module": {
                **current,
                "instruction": long_instruction,
            },
        }
        update = SkillDistiller._validate_rule_batch(parsed, rules, patches, "vista.motion_physics")
        self.assertGreater(len(long_instruction), 8000)
        self.assertEqual(update["status"], "update")
        self.assertEqual(update["instruction_length"], len(long_instruction.strip()))
        self.assertTrue(update["instruction_length_warning"])

    def test_length_policy_does_not_change_trigger_contract(self):
        rules = json.loads((SKILL / "initializations" / "posneg_v1_1.json").read_text(encoding="utf-8"))
        self.assertEqual(rules["update_policy"]["instruction_length_policy"], "unbounded_with_soft_warning")
        self.assertEqual(rules["update_policy"]["instruction_warning_chars"], 8000)
        trigger_report = subprocess.run(
            [
                sys.executable,
                str(SKILL / "scripts" / "validate_trigger_contract.py"),
                "--rules",
                str(SKILL / "initializations" / "posneg_v1_1.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(json.loads(trigger_report.stdout)["valid"])

    def test_legacy_max_instruction_chars_is_soft_warning_only(self):
        rules = json.loads((SKILL / "initializations" / "posneg_v1_1.json").read_text(encoding="utf-8"))
        rules.pop("update_policy", None)
        rules["update_policy"] = {"max_instruction_chars": 3000}
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        long_instruction = "Evidence-backed motion behavior. " + ("Preserve continuous observable progression. " * 120)
        patches = [
            {"episode_id": f"legacy-e{index}", "edit": {"rule_id": "vista.motion_physics", "trigger": current["trigger"]}}
            for index in range(4)
        ]
        parsed = {
            "status": "update",
            "reasoning": "Legacy configuration compatibility.",
            "evidence_episode_ids": [item["episode_id"] for item in patches],
            "module": {**current, "instruction": long_instruction},
        }
        update = SkillDistiller._validate_rule_batch(parsed, rules, patches, "vista.motion_physics")
        self.assertGreater(len(long_instruction), 3000)
        self.assertEqual(update["instruction_warning_chars"], 3000)
        self.assertTrue(update["instruction_length_warning"])
        self.assertEqual(update["status"], "update")

    def test_batch_distillation_audits_and_repairs_schema_without_thinking(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        patches = [
            {
                "episode_id": f"repair-e{index}",
                "edit": {"rule_id": "vista.motion_physics", "trigger": current["trigger"]},
            }
            for index in range(4)
        ]
        repaired = {
            "schema_version": "2.0",
            "status": "update",
            "reasoning": "Serialization repaired without changing the proposed rule.",
            "evidence_episode_ids": [item["episode_id"] for item in patches],
            "module": {**current, "instruction": "A compact repaired motion rule."},
        }
        client = SequenceMLLM([json.dumps(current), json.dumps(repaired)])
        with tempfile.TemporaryDirectory() as tmp:
            audit_dir = Path(tmp) / "audit"
            update = SkillDistiller(client, SKILL).distill_rule_batch(
                patches, apply=False, audit_dir=audit_dir
            )
            attempt_one = json.loads((audit_dir / "attempt_01.json").read_text(encoding="utf-8"))
            attempt_two = json.loads((audit_dir / "attempt_02_repair.json").read_text(encoding="utf-8"))
        self.assertEqual(update["status"], "update")
        self.assertFalse(attempt_one["valid"])
        self.assertTrue(attempt_two["valid"])
        self.assertTrue(client.calls[0]["thinking"])
        self.assertFalse(client.calls[1]["thinking"])

    def test_parse_json_like_prefers_outer_update_object(self):
        value = parse_json_like(
            'The result is {"status":"update","module":{"rule_id":"vista.motion_physics"}} done.'
        )
        self.assertEqual(value["status"], "update")
        self.assertEqual(value["module"]["rule_id"], "vista.motion_physics")

    def test_promotion_accepts_valid_trigger_change_and_freezes_axis(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        learned_trigger = "The source requests an observable motion trajectory or a gradual physical state transition."
        patch = {
            "status": "candidate",
            "edit": {
                **current,
                "operation": "replace",
                "trigger": learned_trigger,
                "trigger_change_rationale": "Held-out routing evidence supports the missed motion case.",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
            result = promote(patch, {"passed": True, "guard_noninferior": True}, skill_dir)
            promoted = json.loads((skill_dir / "rules.json").read_text(encoding="utf-8"))
        self.assertTrue(result["promoted"])
        updated = next(x for x in promoted["modules"] if x["rule_id"] == "vista.motion_physics")
        self.assertEqual(updated["trigger"], learned_trigger)
        self.assertEqual(updated["trigger_axis"], current["trigger_axis"])

    def test_promotion_rejects_unbounded_conditional_trigger(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(x for x in rules["modules"] if x["rule_id"] == "vista.motion_physics")
        patch = {
            "status": "candidate",
            "edit": {
                **current,
                "operation": "replace",
                "trigger": "Always apply to every prompt.",
                "trigger_change_rationale": "A claimed routing expansion.",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conditional trigger cannot become an unbounded global trigger"):
                promote(patch, {"passed": True, "guard_noninferior": True}, skill_dir)

    def test_trigger_contract_validator_passes(self):
        completed = subprocess.run(
            [sys.executable, str(SKILL / "scripts" / "validate_trigger_contract.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(json.loads(completed.stdout)["valid"])

    def test_revision_zero_initialization_contract_passes(self):
        initialization = SKILL / "initializations" / "posneg_v1_1.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(SKILL / "scripts" / "validate_trigger_contract.py"),
                "--rules",
                str(initialization),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertTrue(report["valid"])
        self.assertEqual(report["revision"], 0)

    def test_batch_controller_initializes_isolated_revision_zero_skill(self):
        deployed_before = (SKILL / "rules.json").read_bytes()
        initialization = SKILL / "initializations" / "posneg_v1_1.json"
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            working = initialize_working_skill(Path(tmp), initialization)
            rules = json.loads((working / "rules.json").read_text(encoding="utf-8"))
            record = json.loads((Path(tmp) / "skill_initialization.json").read_text(encoding="utf-8"))
        self.assertEqual(rules["revision"], 0)
        self.assertEqual(record["initial_revision"], 0)
        self.assertEqual((SKILL / "rules.json").read_bytes(), deployed_before)

    def test_batch_controller_assigns_only_requested_gpus(self):
        rows = [{"prompt": f"prompt {index}"} for index in range(8)]
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            jobs = write_shard_prompts(Path(tmp), rows, parse_sana_gpus("6,7"))
            assignments = [json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for _, _, path in jobs]
        self.assertEqual([gpu for _, gpu, _ in jobs], ["6", "7"])
        self.assertEqual([len(items) for items in assignments], [4, 4])

    def test_per_rule_fifo_scheduler_prefers_longest_eligible_wait(self):
        rules = json.loads(
            (SKILL / "initializations" / "posneg_v1_1.json").read_text(encoding="utf-8")
        )
        pool = _new_patch_pool(rules)
        def patch(rule_id: str, index: int) -> dict[str, object]:
            return {
                "patch_id": f"{rule_id}:{index}",
                "episode_id": f"episode-{rule_id}-{index}",
                "edit": {"rule_id": rule_id, "instruction": f"instruction {index}"},
            }
        identity = "vista.identity_count_guard"
        motion = "vista.motion_physics"
        _ingest_patches(
            pool,
            [patch(identity, index) for index in range(4)],
            wave_number=1,
            batch_size=4,
        )
        _ingest_patches(
            pool,
            [patch(motion, index) for index in range(4)],
            wave_number=2,
            batch_size=4,
        )
        self.assertEqual(_select_eligible_rule(pool, 4), identity)
        chosen_ids = [item["patch_id"] for item in pool["queues"][identity][:4]]
        self.assertEqual(chosen_ids, [f"{identity}:{index}" for index in range(4)])
        _consume_patch_batch(
            pool,
            rule_id=identity,
            patch_ids=chosen_ids,
            wave_number=2,
            batch_size=4,
            result={"status": "update", "revision": 1, "update_kind": "normal_batch_update"},
        )
        self.assertEqual(pool["queues"][identity], [])
        self.assertEqual(_select_eligible_rule(pool, 4), motion)

    def test_final_flush_consumes_one_to_three_remaining_patches(self):
        initialization = SKILL / "initializations" / "posneg_v1_1.json"
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            output_root = Path(tmp)
            working_skill = initialize_working_skill(output_root, initialization)
            rules = json.loads((working_skill / "rules.json").read_text(encoding="utf-8"))
            pool = _new_patch_pool(rules)
            rule_id = "vista.identity_count_guard"
            current = next(item for item in rules["modules"] if item["rule_id"] == rule_id)
            patches = [
                {
                    "patch_id": f"tail-{index}:{rule_id}",
                    "episode_id": f"tail-{index}",
                    "edit": {
                        **current,
                        "operation": "replace",
                        "instruction": f"Tail evidence {index}.",
                        "trigger_change_rationale": "",
                    },
                }
                for index in range(3)
            ]
            _ingest_patches(pool, patches, wave_number=13, batch_size=4)
            (output_root / "patch_pool.json").write_text(
                json.dumps(pool), encoding="utf-8"
            )
            args = SimpleNamespace(
                config="config/deepseek_qwen36_35ba3b_sana_posneg_train.yaml",
                all_dry_run=True,
                rule_update_size=4,
            )
            state = finalize_patch_pool(
                output_root,
                args,
                {},
                working_skill,
                starting_wave=13,
            )
            final_pool = _load_patch_pool(
                output_root / "patch_pool.json",
                json.loads((working_skill / "rules.json").read_text(encoding="utf-8")),
            )
        self.assertTrue(state["completed"])
        self.assertEqual(state["updates"][0]["update_kind"], "final_tail_update")
        self.assertEqual(state["updates"][0]["patch_count"], 3)
        self.assertEqual(final_pool["queues"][rule_id], [])

    def test_supervisor_waits_for_finalization_after_all_prompts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            output_root = Path(tmp)
            (output_root / "controller_state.json").write_text(
                json.dumps({"train_offset": 100, "finalization_complete": False}),
                encoding="utf-8",
            )
            self.assertFalse(training_complete(output_root, 100))
            (output_root / "controller_state.json").write_text(
                json.dumps({"train_offset": 100, "finalization_complete": True}),
                encoding="utf-8",
            )
            self.assertTrue(training_complete(output_root, 100))

    def test_heldout_gate_requires_more_wins_than_losses_and_strict_guard(self):
        self.assertTrue(gate_decision(["candidate_win", "candidate_win", "tie"])["passed"])
        self.assertFalse(gate_decision(["candidate_win", "tie", "incumbent_win"])["passed"])

    def test_one_pass_rewriter_returns_one_prompt_and_revision(self):
        client = FakeMLLM('{"rewritten_prompt":"A single coherent shot of a red panda walking into sunlight."}')
        result = PromptRewriter(client, SKILL).rewrite("A red panda walks into sunlight.")
        self.assertEqual(result.rewritten_prompt, "A single coherent shot of a red panda walking into sunlight.")
        self.assertEqual(result.skill_revision, json.loads((SKILL / "rules.json").read_text())["revision"])
        self.assertEqual(client.calls, ["rewrite"])

    def test_one_pass_rewriter_returns_positive_and_bounded_negative(self):
        client = FakeMLLM(json.dumps({
            "positive_prompt": "A detailed continuous shot of a cup falling from the left edge and breaking on the floor.",
            "negative_prompt": "abrupt cut, discontinuous motion, framing drift, frozen frame, morphing, warping, flickering, identity switching, abrupt appearance, missing subjects, abrupt cut",
            "activated_rule_ids": [
                "vista.duration_scene_budget",
                "vista.scene_components",
                "vista.motion_physics",
                "vista.frame_exit_reentry",
                "vista.identity_count_guard",
            ],
        }))
        result = PromptRewriter(client, SKILL).rewrite("A cup falls and breaks.")
        self.assertTrue(result.positive_prompt.startswith("A detailed"))
        self.assertEqual(len(result.negative_prompt.split(", ")), 8)
        self.assertEqual(result.negative_prompt.count("abrupt cut"), 1)

    def test_one_pass_rewriter_records_only_known_activated_rules(self):
        client = FakeMLLM(json.dumps({
            "positive_prompt": "A detailed continuous shot of a cup falling and breaking on the floor.",
            "negative_prompt": "object morphing",
            "activated_rule_ids": [
                "vista.intent_preservation",
                "vista.motion_physics",
                "invented.rule",
                "vista.motion_physics",
            ],
        }))
        result = PromptRewriter(client, SKILL).rewrite("A cup falls and breaks.")
        self.assertEqual(result.activated_rule_ids, [
            "vista.intent_preservation",
            "vista.motion_physics",
        ])

    def test_multi_rule_trajectory_assessment_emits_one_record_per_activation(self):
        response = {
            "schema_version": "3.0",
            "episode_id": "multi-e1",
            "assessments": [
                {
                    "rule_id": "vista.intent_preservation",
                    "status": "no_change",
                    "reasoning": "Intent was preserved.",
                    "confidence": 0.9,
                    "edit": {},
                },
                {
                    "rule_id": "vista.motion_physics",
                    "status": "candidate",
                    "reasoning": "The action lacked a visible end state.",
                    "confidence": 0.8,
                    "edit": {
                        "rule_id": "vista.motion_physics",
                        "instruction": "Describe a compact observable action with a visible end state.",
                        "negative_terms": ["frozen frame"],
                    },
                },
            ],
        }
        assessments = SkillDistiller(FakeMLLM(json.dumps(response)), SKILL).propose_trajectory_rule_assessments(
            {"episode_id": "multi-e1"},
            ["vista.intent_preservation", "vista.motion_physics"],
        )
        self.assertEqual([item["rule_id"] for item in assessments], [
            "vista.intent_preservation",
            "vista.motion_physics",
        ])
        self.assertEqual([item["status"] for item in assessments], ["no_change", "candidate"])
        self.assertEqual(assessments[1]["patch_id"], "multi-e1:vista.motion_physics")

    def test_misrouted_assessment_becomes_trigger_candidate(self):
        rules = json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))
        current = next(
            item for item in rules["modules"]
            if item["rule_id"] == "vista.frame_exit_reentry"
        )
        narrowed = "The source explicitly states that a subject exits the frame and later re-enters it during one continuous action."
        response = {
            "schema_version": "3.0",
            "episode_id": "misroute-e1",
            "assessments": [{
                "rule_id": "vista.frame_exit_reentry",
                "status": "misrouted",
                "reasoning": "Cross-frame motion was inferred without source support.",
                "confidence": 0.9,
                "edit": {
                    "rule_id": "vista.frame_exit_reentry",
                    "trigger": narrowed,
                    "trigger_change_rationale": "The source never requested exit or re-entry.",
                },
            }],
        }
        assessment = SkillDistiller(
            FakeMLLM(json.dumps(response)), SKILL
        ).propose_trajectory_rule_assessments(
            {"episode_id": "misroute-e1"},
            ["vista.frame_exit_reentry"],
        )[0]
        self.assertEqual(assessment["status"], "candidate")
        self.assertEqual(assessment["assessment_status"], "misrouted")
        self.assertEqual(assessment["edit"]["trigger"], narrowed)
        self.assertEqual(assessment["edit"]["instruction"], current["instruction"])

    def test_qwen_visible_thinking_keeps_only_final_rewrite(self):
        raw = '''Thinking Process:
1. Build a fidelity ledger.
2. Make the action observable.

Final Output Construction:
"A ceramic cup rests on the edge of a wooden table, tips forward and falls, strikes the floor, and shatters into several pieces."'''  # noqa: E501
        stripped = MLLMClient._strip_thinking_text(raw)
        result = PromptRewriter(FakeMLLM(stripped), SKILL).rewrite("A cup falls and breaks.")
        self.assertNotIn("Thinking Process", stripped)
        self.assertNotIn("fidelity ledger", result.rewritten_prompt)
        self.assertEqual(
            result.rewritten_prompt,
            "A ceramic cup rests on the edge of a wooden table, tips forward and falls, strikes the floor, and shatters into several pieces.",
        )

    def test_incomplete_thinking_fragment_falls_back_to_source(self):
        result = PromptRewriter(FakeMLLM('or just describe the view. The modules say'), SKILL).rewrite(
            "A ceramic cup falls from the edge of a wooden table and breaks on the floor."
        )
        self.assertEqual(
            result.rewritten_prompt,
            "A ceramic cup falls from the edge of a wooden table and breaks on the floor.",
        )

    def test_distiller_rejects_nonaccepted_episode(self):
        client = FakeMLLM('{"status":"candidate","edit":{"operation":"replace","rule_id":"event.temporal_causality","trigger":"ordered events","instruction":"bind each event to its subject","capability_scope":["complex_plot"],"evidence_episode_ids":["round-1"],"expected_side_effects":[],"guard_metric":"identity"}}')
        distiller = SkillDistiller(client, SKILL)
        patch = distiller.propose({"status": "comparable"})
        self.assertEqual(patch["status"], "reject")

    def test_trajectory_distiller_repairs_unknown_rule_id(self):
        episode = {"episode_id": "e1", "rounds": []}
        client = SequenceMLLM([
            json.dumps({
                "status": "candidate", "episode_id": "e1", "reasoning": "motion evidence",
                "edit": {"rule_id": "motion_physics", "instruction": "improve motion"},
            }),
            json.dumps({
                "status": "candidate", "episode_id": "e1", "reasoning": "motion evidence",
                "edit": {"rule_id": "vista.motion_physics", "instruction": "improve motion"},
            }),
        ])
        patch = SkillDistiller(client, SKILL).propose_trajectory_patch(episode)
        self.assertEqual(patch["status"], "candidate")
        self.assertEqual(patch["edit"]["rule_id"], "vista.motion_physics")
        current = next(
            item
            for item in json.loads((SKILL / "rules.json").read_text(encoding="utf-8"))["modules"]
            if item["rule_id"] == "vista.motion_physics"
        )
        self.assertEqual(patch["edit"]["trigger"], current["trigger"])
        self.assertEqual(patch["edit"]["trigger_axis"], current["trigger_axis"])
        self.assertEqual(client.calls[1]["expected"], "trajectory_patch_repair")
        self.assertTrue(client.calls[0]["thinking"])
        self.assertFalse(client.calls[1]["thinking"])

    def test_trajectory_distiller_rejects_unknown_rule_after_repair(self):
        episode = {"episode_id": "e2", "rounds": []}
        invalid = json.dumps({
            "status": "candidate", "episode_id": "e2", "reasoning": "weak",
            "edit": {"rule_id": "invented.rule", "instruction": "change it"},
        })
        patch = SkillDistiller(SequenceMLLM([invalid, invalid]), SKILL).propose_trajectory_patch(episode)
        self.assertEqual(patch["status"], "reject")
        self.assertEqual(patch["episode_id"], "e2")

    def test_trajectory_distiller_handles_null_edit_without_crashing(self):
        episode = {"episode_id": "e-null", "rounds": []}
        invalid = json.dumps({
            "status": "candidate",
            "episode_id": "e-null",
            "reasoning": "no serializable edit was produced",
            "edit": None,
        })
        client = SequenceMLLM([invalid, invalid])
        patch = SkillDistiller(client, SKILL).propose_trajectory_patch(episode)
        self.assertEqual(patch["status"], "reject")
        self.assertEqual(patch["episode_id"], "e-null")
        self.assertEqual(client.calls[1]["expected"], "trajectory_patch_repair")

    def test_supervisor_training_complete_uses_controller_progress(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            output_root = Path(tmp)
            (output_root / "controller_state.json").write_text(
                json.dumps({"train_offset": 99}),
                encoding="utf-8",
            )
            self.assertFalse(training_complete(output_root, 100))
            (output_root / "controller_state.json").write_text(
                json.dumps({"train_offset": 100}),
                encoding="utf-8",
            )
            self.assertTrue(training_complete(output_root, 100))

    def test_supervisor_retry_delay_is_capped(self):
        self.assertEqual(retry_delay(1, base_seconds=60, max_seconds=900), 60)
        self.assertEqual(retry_delay(4, base_seconds=60, max_seconds=900), 480)
        self.assertEqual(retry_delay(10, base_seconds=60, max_seconds=900), 900)

    def test_deploy_has_one_rewrite_and_one_t2v_call(self):
        config = load_config(ROOT / "config" / "default.yaml")
        config["mllm"]["api_key"] = ""
        config["mllm"]["dry_run_when_api_key_missing"] = True
        config["t2v"]["dry_run"] = True
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            result = VistaAgent(config).run_deploy("A cup falls and breaks.", tmp)
            trajectory = json.loads(Path(result.trajectory_path).read_text(encoding="utf-8"))
            phases = trajectory["stats"]["phases"]
            self.assertEqual(trajectory["stats"]["mllm_call_count"], 1)
            self.assertEqual(trajectory["stats"]["t2v_call_count"], 1)
            self.assertEqual([item["name"] for item in phases], ["one_pass_rewrite", "one_t2v_generation"])

    def test_sana_backend_dry_run_uses_same_client_contract(self):
        client = create_t2v_client({"backend": "sana_diffusers", "dry_run": True, "checkpoint_dir": "/missing"})
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            generation = client.generate("A fox runs.", tmp, "fox", seed=7)
            self.assertTrue(generation.dry_run)
            self.assertEqual(client.calls[0]["seed"], 7)
            self.assertTrue(Path(generation.video_path).exists())

    def test_sana_merges_static_and_conditional_negative_prompts(self):
        client = create_t2v_client({
            "backend": "sana_diffusers",
            "dry_run": True,
            "checkpoint_dir": "/missing",
            "negative_prompt": "flicker, ghosting",
        })
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            client.generate("A fox runs.", tmp, "fox", seed=7, negative_prompt="ghosting, trajectory drift")
        self.assertEqual(client.calls[0]["negative_prompt"], "flicker, ghosting, trajectory drift")

    def test_dashscope_video_mode_sends_base64_mp4_and_fps(self):
        client = MLLMClient({
            "provider": "openai_compatible",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.7-plus",
            "api_key": "test-only",
            "api_key_required": True,
            "dry_run_when_api_key_missing": False,
            "video_input": {"mode": "base64_video", "fps": 2},
        })
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            handle.write(b"fake-mp4")
            handle.flush()
            with mock.patch("vista.clients.mllm_client.requests.post", return_value=response) as post:
                client.call("describe", videos=[handle.name])
        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "video_url")
        self.assertTrue(content[0]["video_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual(content[0]["fps"], 2)
        self.assertEqual(content[-1], {"type": "text", "text": "describe"})

    def test_wave_controller_assigns_all_eight_gpus_to_sana(self):
        self.assertEqual(SANA_GPUS, ("0", "1", "2", "3", "4", "5", "6", "7"))

    def test_training_routes_text_and_video_tasks_to_different_models(self):
        config = load_config(ROOT / "config" / "default.yaml")
        dry_client = {
            "provider": "openai_compatible",
            "api_key_required": True,
            "api_key": "",
            "dry_run_when_api_key_missing": True,
        }
        config["text_mllm"] = {**dry_client, "model": "deepseek-v4-flash"}
        config["vision_mllm"] = {**dry_client, "model": "qwen3.6-27b"}
        config["t2v"]["dry_run"] = True
        config["run"].update(iterations=2, initial_prompt_samples=0, improved_prompt_samples=4,
                             include_champion_residual=False, require_exact_improved_samples=True)
        config["skill_learning"]["enabled"] = True
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            result = VistaAgent(config).run("A ball rolls off a table and lands on the floor.", tmp)
            trajectory = json.loads(Path(result.trajectory_path).read_text(encoding="utf-8"))
        calls = trajectory["stats"]["mllm_calls"]
        text_tasks = {call["expected"] for call in calls if call["model_role"] == "text"}
        vision_tasks = {call["expected"] for call in calls if call["model_role"] == "vision"}
        self.assertTrue({"rewrite", "dtpa", "prompt_sampler", "trajectory_rule_assessments"}.issubset(text_tasks))
        self.assertNotIn("planner", text_tasks)
        self.assertNotIn("skill_patch", text_tasks)
        self.assertTrue({"probing", "pairwise", "judge_normal", "judge_adversarial", "judge_meta"}.issubset(vision_tasks))
        self.assertTrue(all(call["video_count"] == 0 for call in calls if call["model_role"] == "text"))
        text_by_task = {call["expected"]: call for call in calls if call["model_role"] == "text"}
        self.assertFalse(text_by_task["rewrite"]["thinking"])
        self.assertFalse(text_by_task["dtpa"]["thinking"])
        self.assertFalse(text_by_task["prompt_sampler"]["thinking"])
        self.assertTrue(text_by_task["trajectory_rule_assessments"]["thinking"])
        self.assertTrue(all(call["thinking"] is False for call in calls if call["model_role"] == "vision"))
        initial_sources = {item["prompt_candidate"]["metadata"]["source"] for item in trajectory["rounds"][0]["videos"]}
        self.assertIn("frozen_skill", initial_sources)
        self.assertIn("raw_baseline", initial_sources)
        self.assertEqual(trajectory["rounds"][0]["search_start"], "r0_skill_incumbent")
        self.assertEqual([len(item["videos"]) for item in trajectory["rounds"]], [2, 4, 4])
        self.assertIn("skill_learning_episode", trajectory)

    def test_training_cli_exposes_qwen_api_mode(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_training_corpus.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--vision-api-base-url", completed.stdout)
        self.assertIn("--vision-api-model", completed.stdout)
        self.assertIn("--data-inspection-retries", completed.stdout)

    def test_training_recognizes_and_numbers_data_inspection_retries(self):
        error = (
            "RuntimeError: MLLM request failed after retries: "
            "InternalError.Algo.DataInspectionFailed: input image was rejected"
        )
        self.assertTrue(_is_data_inspection_failure(error))
        self.assertFalse(_is_data_inspection_failure("RuntimeError: request timed out"))
        first_index, history = _prior_inspection_attempts(
            {"state": "failed", "attempt_index": 0, "error": error}
        )
        self.assertEqual(first_index, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["type"], "vision_data_inspection_failed")
        item_dir = Path("/tmp/example_item")
        self.assertEqual(_attempt_run_dir(item_dir, 0), item_dir / "run")
        self.assertEqual(_attempt_run_dir(item_dir, 1), item_dir / "run_retry_01")

    def test_training_resumes_only_distillation_from_completed_search(self):
        class FakeAgent:
            def __init__(self):
                self.distiller = mock.Mock()
                self.distiller.propose_trajectory_rule_assessments.return_value = [{
                    "status": "candidate",
                    "episode_id": "episode-1",
                    "rule_id": "vista.motion_physics",
                    "edit": {"rule_id": "vista.motion_physics"},
                }]
                self._record_phase = mock.Mock()
                self._write_trajectory = VistaAgent._write_trajectory

            @staticmethod
            def _skill_learning_enabled():
                return True

            @staticmethod
            def _phase_start():
                return 1.0, (0, 0), 0

            @staticmethod
            def _build_batch_skill_episode(trajectory):
                return {"episode_id": "episode-1", "prompt": trajectory["user_prompt"]}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trajectory = {
                "user_prompt": "A boat crosses a calm ocean.",
                "initial_skill_rewrite": {"activated_rule_ids": ["vista.motion_physics"]},
                "rounds": [{"round_index": 0}],
                "best_video": {
                    "video_path": str(run_dir / "best.mp4"),
                    "prompt_candidate": {"prompt": "A detailed boat prompt."},
                },
                "stats": {"phases": []},
            }
            (run_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
            agent = FakeAgent()
            recovered = _resume_completed_search_distillation(
                agent,
                run_dir,
                expected_prompt="A boat crosses a calm ocean.",
            )

            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["candidate_patches"][0]["status"], "candidate")
            self.assertEqual(recovered["skill_learning_episode"]["episode_id"], "episode-1")
            self.assertTrue(recovered["stats"]["resumed_from_completed_search"])
            agent.distiller.propose_trajectory_rule_assessments.assert_called_once()
            agent._record_phase.assert_called_once()

    def test_training_does_not_resume_incomplete_or_mismatched_trajectory(self):
        agent = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "trajectory.json").write_text(
                json.dumps({"user_prompt": "prompt", "rounds": []}),
                encoding="utf-8",
            )
            self.assertIsNone(
                _resume_completed_search_distillation(agent, run_dir, expected_prompt="prompt")
            )
            agent.reset_mock()
            (run_dir / "trajectory.json").write_text(
                json.dumps(
                    {
                        "user_prompt": "different prompt",
                        "best_video": {
                            "video_path": "best.mp4",
                            "prompt_candidate": {"prompt": "rewritten"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(
                _resume_completed_search_distillation(agent, run_dir, expected_prompt="prompt")
            )

    def test_transformers_input_mover_keeps_grid_and_pixels_on_same_device(self):
        class FakeTensor:
            def __init__(self):
                self.device = "cpu"

            def to(self, device):
                moved = FakeTensor()
                moved.device = str(device)
                return moved

        moved = MLLMClient._move_inputs_to_device(
            {"pixel_values": FakeTensor(), "image_grid_thw": FakeTensor()}, "cuda:0"
        )
        self.assertEqual(moved["pixel_values"].device, "cuda:0")
        self.assertEqual(moved["image_grid_thw"].device, "cuda:0")

    def test_local_transformers_explicit_gpu_overrides_broad_parent_mask(self):
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
        client = MLLMClient(
            {
                "provider": "transformers",
                "model": "/missing",
                "cuda_visible_devices": "4",
                "api_key_required": False,
            }
        )
        # Stop immediately after the mask is applied, before importing a model.
        client._preload_cuda_runtime_libraries = lambda: (_ for _ in ()).throw(RuntimeError("stop"))
        with self.assertRaisesRegex(RuntimeError, "stop"):
            client._load_transformers_local()
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


if __name__ == "__main__":
    unittest.main()
