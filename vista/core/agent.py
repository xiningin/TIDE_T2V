from __future__ import annotations

import json
import hashlib
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.clients.t2v_client import create_t2v_client
from vista.core.dtpa import DTPAOptimizer
from vista.core.distillation import SkillDistiller
from vista.core.mmac import MMACCritic
from vista.core.planner import PromptPlanner
from vista.core.prompt_rewriter import PromptRewriter
from vista.core.selection import TournamentSelector
from vista.core.strict_constraints import strict_action_enabled
from vista.schemas import PromptCandidate, VideoCandidate, VistaRunResult
from vista.utils.config import ensure_dir


class VistaAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.run_config = config.get("run", {})
        self.rng = random.Random(int(self.run_config.get("random_seed", 42)))
        legacy_mllm = config.get("mllm", {})
        if "text_mllm" not in config and "vision_mllm" not in config:
            shared_mllm = MLLMClient(legacy_mllm)
            self.text_mllm = shared_mllm
            self.vision_mllm = shared_mllm
        else:
            self.text_mllm = MLLMClient(config.get("text_mllm", legacy_mllm))
            self.vision_mllm = MLLMClient(config.get("vision_mllm", legacy_mllm))
        self.mllm = self.vision_mllm  # Backward-compatible evaluator alias.
        self.t2v = create_t2v_client(config.get("t2v", {}))
        self.selection_config = config.get("selection", {})
        self.planner = PromptPlanner(self.text_mllm, self.run_config)
        self.selector = TournamentSelector(self.vision_mllm, self.selection_config)
        self.critic = MMACCritic(self.vision_mllm, self.selection_config)
        self.optimizer = DTPAOptimizer(self.text_mllm, self.run_config, self.selection_config)
        self.skill_config = config.get("skill_learning", {})
        self.rewriter = PromptRewriter(
            self.text_mllm,
            self.skill_config.get("skill_dir"),
            self.skill_config.get("rules_file"),
            activation_mode=str(self.skill_config.get("activation_mode", "normal")),
            random_activation_count=self.skill_config.get("random_activation_count"),
            random_activation_seed=int(self.skill_config.get("random_activation_seed", self.run_config.get("random_seed", 42))),
        )
        self.distiller = SkillDistiller(
            self.text_mllm,
            self.rewriter.skill_dir,
            self.rewriter.rules_file,
            freeze_negative_prompt=bool(self.skill_config.get("freeze_negative_prompt", False)),
        )

    def rewrite_prompt(self, user_prompt: str, *, source_prompt: str = "") -> Any:
        """Run the deployment path: one text-only rewrite and no video judges."""
        return self.rewriter.rewrite(user_prompt, source_prompt=source_prompt)

    def reset_telemetry(self) -> None:
        """Clear per-prompt counters without unloading persistent models/workers."""
        self.text_mllm.calls.clear()
        if self.vision_mllm is not self.text_mllm:
            self.vision_mllm.calls.clear()
        self.t2v.calls.clear()

    def all_mllm_calls(self) -> list[dict[str, Any]]:
        if self.text_mllm is self.vision_mllm:
            return self._tagged_calls(self.text_mllm.calls, "shared")
        calls = self._tagged_calls(self.text_mllm.calls, "text")
        calls += self._tagged_calls(self.vision_mllm.calls, "vision")
        return calls

    def run_deploy(self, user_prompt: str, output_dir: str | Path | None = None) -> VistaRunResult:
        """Rewrite once and generate exactly one video; never invoke VISTA search."""
        run_started = perf_counter()
        output_dir = self._make_output_dir(output_dir)
        stats: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "deploy",
            "phases": [],
            "mllm_calls": [],
            "t2v_calls": [],
            "usage_total": self._zero_usage(),
        }
        rewrite_started, mllm_before, t2v_before = self._phase_start()
        rewrite = self.rewrite_prompt(user_prompt)
        self._record_phase(stats, "one_pass_rewrite", rewrite_started, mllm_before, t2v_before)
        prompt_candidate = PromptCandidate(
            id="deploy_p0",
            prompt=rewrite.rewritten_prompt,
            round_index=0,
            negative_prompt=rewrite.negative_prompt,
            metadata={"source": "frozen_skill", "skill_revision": rewrite.skill_revision, "skill_sha256": rewrite.skill_sha256},
        )
        generation_started, mllm_before, t2v_before = self._phase_start()
        deploy_seed = self.rng.randint(0, 2**31 - 1)
        generated = self.t2v.generate(
            prompt_candidate.prompt,
            output_dir / "deploy",
            "deploy_p0_v0",
            seed=deploy_seed,
            negative_prompt=prompt_candidate.negative_prompt,
        )
        video = VideoCandidate(
            id="deploy_p0_v0",
            prompt_candidate=prompt_candidate,
            video_path=generated.video_path,
            round_index=0,
            seed=deploy_seed,
            metadata={"dry_run": generated.dry_run, "command": generated.command, "elapsed_seconds": generated.elapsed_seconds},
        )
        self._record_phase(stats, "one_t2v_generation", generation_started, mllm_before, t2v_before, {"video_count": 1})
        trajectory = {
            "mode": "deploy",
            "user_prompt": user_prompt,
            "rewrite": rewrite.to_dict(),
            "best_video": video.to_dict(),
            "config": self.config,
            "stats": stats,
        }
        self._finalize_stats(stats, run_started)
        trajectory_path = self._write_trajectory(output_dir, trajectory)
        return VistaRunResult(user_prompt, video, prompt_candidate, str(trajectory_path), str(output_dir))

    def run(self, user_prompt: str, output_dir: str | Path | None = None) -> VistaRunResult:
        run_started = perf_counter()
        output_dir = self._make_output_dir(output_dir)
        stats: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "phases": [],
            "mllm_calls": [],
            "t2v_calls": [],
            "usage_total": self._zero_usage(),
        }
        trajectory: dict[str, Any] = {
            "user_prompt": user_prompt,
            "config": self.config,
            "text_mllm_dry_run": self.text_mllm.dry_run,
            "vision_mllm_dry_run": self.vision_mllm.dry_run,
            "t2v_dry_run": self.t2v.dry_run,
            "stats": stats,
            "rounds": [],
        }

        phase_started, mllm_before, t2v_before = self._phase_start()
        # PSISTA initialization deliberately replaces VISTA's structured-video
        # planner: keep one raw baseline and one frozen-skill rewrite. Search starts
        # from the skill rewrite; the raw video is diagnostic evidence only.
        prompts = [
            PromptCandidate(
                id="r0_raw",
                prompt=user_prompt,
                round_index=0,
                metadata={"source": "raw_baseline"},
            )
        ]
        initial_skill_rewrite: dict[str, Any] | None = None
        if self._skill_learning_enabled():
            rewrite = self.rewrite_prompt(user_prompt)
            initial_skill_rewrite = rewrite.to_dict()
            prompts.append(
                PromptCandidate(
                    id="r0_skill_incumbent",
                    prompt=rewrite.rewritten_prompt,
                    round_index=0,
                    negative_prompt=rewrite.negative_prompt,
                    metadata={
                        "source": "frozen_skill",
                        "skill_revision": rewrite.skill_revision,
                        "skill_sha256": rewrite.skill_sha256,
                    },
                )
            )
        self._record_phase(stats, "planning", phase_started, mllm_before, t2v_before, {"round_index": 0})
        if initial_skill_rewrite is not None:
            trajectory["initial_skill_rewrite"] = initial_skill_rewrite

        phase_started, mllm_before, t2v_before = self._phase_start()
        videos = self._generate_videos(prompts, output_dir / "round_0", round_index=0)
        self._record_phase(stats, "t2v_generation", phase_started, mllm_before, t2v_before, {"round_index": 0, "video_count": len(videos)})

        skill_video = next((item for item in videos if item.prompt_candidate.id == "r0_skill_incumbent"), None)
        if skill_video is None:
            raise RuntimeError("PSISTA training requires an r0_skill_incumbent video")
        champion = skill_video
        trajectory["rounds"].append(
            {
                "round_index": 0,
                "phase": "initialization",
                "prompts": [item.to_dict() for item in prompts],
                "videos": [item.to_dict() for item in videos],
                "search_start": "r0_skill_incumbent",
                "raw_is_diagnostic_only": True,
                "champion": champion.to_dict(),
            }
        )

        unchanged_count = 0
        max_iterations = int(self.run_config.get("iterations", 4))
        patience = int(self.run_config.get("early_stop_patience", 2))
        for round_index in range(1, max_iterations + 1):
            phase_started, mllm_before, t2v_before = self._phase_start()
            critiques = self.critic.critique(user_prompt, champion)
            self._record_phase(stats, "critique", phase_started, mllm_before, t2v_before, {"round_index": round_index})

            phase_started, mllm_before, t2v_before = self._phase_start()
            improved_prompts = self.optimizer.improve(user_prompt, champion, critiques, round_index)
            self._record_phase(stats, "optimization", phase_started, mllm_before, t2v_before, {"round_index": round_index})

            phase_started, mllm_before, t2v_before = self._phase_start()
            improved_videos = self._generate_videos(improved_prompts, output_dir / f"round_{round_index}", round_index=round_index)
            self._record_phase(stats, "t2v_generation", phase_started, mllm_before, t2v_before, {"round_index": round_index, "video_count": len(improved_videos)})

            phase_started, mllm_before, t2v_before = self._phase_start()
            tournament = self.selector.select(user_prompt, improved_videos)
            self._record_phase(stats, "selection", phase_started, mllm_before, t2v_before, {"round_index": round_index, "video_count": len(improved_videos)})
            selected_champion = tournament.winner
            if selected_champion.prompt == champion.prompt:
                unchanged_count += 1
            else:
                unchanged_count = 0
            champion = selected_champion

            round_record: dict[str, Any] = {
                "round_index": round_index,
                "phase": "self_improvement",
                "critiques": [item.to_dict() for item in critiques],
                "prompts": [item.to_dict() for item in improved_prompts],
                "videos": [item.to_dict() for item in improved_videos],
                "tournament": tournament.to_dict(),
                "unchanged_count": unchanged_count,
            }
            trajectory["rounds"].append(round_record)
            self._write_trajectory(output_dir, trajectory)
            if unchanged_count >= patience:
                trajectory["early_stop"] = {"round_index": round_index, "reason": "champion prompt unchanged"}
                break

        trajectory["best_video"] = champion.to_dict()
        if self._skill_learning_enabled():
            # Assess every rule that actually activated in the initial frozen-Skill
            # rewrite. These per-rule assessments cannot mutate the Skill; the
            # controller schedules same-rule candidates across trajectories.
            distill_started, mllm_before, t2v_before = self._phase_start()
            episode = self._build_batch_skill_episode(trajectory)
            activated_rule_ids = list(
                (initial_skill_rewrite or {}).get("activated_rule_ids", [])
            )
            rule_assessments = self.distiller.propose_trajectory_rule_assessments(
                episode,
                activated_rule_ids,
            )
            candidate_patches = [
                item for item in rule_assessments if item.get("status") == "candidate"
            ]
            trajectory["skill_learning_episode"] = episode
            trajectory["activated_rule_ids"] = activated_rule_ids
            trajectory["rule_assessments"] = rule_assessments
            trajectory["candidate_patches"] = candidate_patches
            self._record_phase(stats, "trajectory_distillation", distill_started, mllm_before, t2v_before)
        self._finalize_stats(stats, run_started)
        trajectory_path = self._write_trajectory(output_dir, trajectory)
        return VistaRunResult(
            user_prompt=user_prompt,
            best_video=champion,
            best_prompt=champion.prompt_candidate,
            trajectory_path=str(trajectory_path),
            output_dir=str(output_dir),
        )

    def _generate_videos(self, prompts: list[PromptCandidate], output_dir: Path, round_index: int) -> list[VideoCandidate]:
        videos_per_prompt = int(self.run_config.get("videos_per_prompt", 1))
        videos: list[VideoCandidate] = []
        for prompt_candidate in prompts:
            for sample_index in range(videos_per_prompt):
                seed = self.rng.randint(0, 2**31 - 1)
                video_id = f"{prompt_candidate.id}_v{sample_index}"
                generation = self.t2v.generate(
                    prompt_candidate.prompt,
                    output_dir,
                    video_id,
                    seed=seed,
                    negative_prompt=prompt_candidate.negative_prompt,
                )
                videos.append(
                    VideoCandidate(
                        id=video_id,
                        prompt_candidate=prompt_candidate,
                        video_path=generation.video_path,
                        round_index=round_index,
                        seed=seed,
                        metadata={
                            "dry_run": generation.dry_run,
                            "command": generation.command,
                            "elapsed_seconds": generation.elapsed_seconds,
                        },
                    )
                )
        return videos

    def _make_output_dir(self, output_dir: str | Path | None) -> Path:
        if output_dir:
            return ensure_dir(output_dir)
        root = self.config.get("outputs", {}).get("root_dir", "outputs")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ensure_dir(Path(root) / f"vista_run_{stamp}")

    def _strict_improvement_gate_enabled(self) -> bool:
        strict_config = self.selection_config.get("strict_action_constraints")
        if not isinstance(strict_config, dict):
            return False
        return strict_action_enabled(self.selection_config) and bool(strict_config.get("improvement_gate", True))

    def _skill_learning_enabled(self) -> bool:
        return bool(self.skill_config.get("enabled", False))

    def _improvement_gate_enabled(self) -> bool:
        return self._strict_improvement_gate_enabled() or self._skill_learning_enabled()

    def _build_skill_episode(
        self,
        user_prompt: str,
        *,
        champion_before: dict[str, Any],
        challenger: VideoCandidate,
        critiques: list[Any],
        tournament: Any,
        gate: dict[str, Any] | None,
        round_index: int,
    ) -> dict[str, Any]:
        accepted = bool(gate and gate.get("accepted"))
        status = "accepted" if accepted else "comparable"
        parent_prompt = champion_before.get("prompt") or champion_before.get("prompt_candidate", {}).get("prompt", "")
        parent_negative = champion_before.get("negative_prompt") or champion_before.get("prompt_candidate", {}).get("negative_prompt", "")
        return {
            "episode_id": f"{hashlib.sha256(user_prompt.encode('utf-8')).hexdigest()[:12]}-round-{round_index:04d}",
            "schema_version": "1.0",
            "status": status,
            "source_prompt": user_prompt,
            "parent_prompt": parent_prompt,
            "parent_negative_prompt": parent_negative,
            "candidate_prompt": challenger.prompt,
            "candidate_negative_prompt": challenger.negative_prompt,
            "failure_signature": [item.dimension for item in critiques],
            "observed_feedback": [item.to_dict() for item in critiques],
            "outcome": {"accepted_by_gate": accepted, "gate": gate or {}, "tournament": tournament.to_dict()},
            "seed_and_generator": {"seed": challenger.seed, "video_path": challenger.video_path},
            "heldout": False,
        }

    @staticmethod
    def _build_batch_skill_episode(trajectory: dict[str, Any]) -> dict[str, Any]:
        rounds = trajectory.get("rounds", [])
        initial = rounds[0] if rounds else {}
        return {
            "schema_version": "2.0",
            "episode_id": hashlib.sha256(trajectory["user_prompt"].encode("utf-8")).hexdigest()[:12],
            "source_prompt": trajectory["user_prompt"],
            "initial_skill_rewrite": trajectory.get("initial_skill_rewrite", {}),
            "raw_baseline_video": next(
                (v for v in initial.get("videos", []) if v.get("prompt_candidate", {}).get("id") == "r0_raw"),
                {},
            ),
            "initial_rewrite_video": next(
                (v for v in initial.get("videos", []) if v.get("prompt_candidate", {}).get("id") == "r0_skill_incumbent"),
                {},
            ),
            "optimization_rounds": [
                {
                    "round_index": item.get("round_index"),
                    "critiques": item.get("critiques", []),
                    "candidate_prompts": item.get("prompts", []),
                    "tournament": item.get("tournament", {}),
                }
                for item in rounds[1:]
            ],
            "final_champion": trajectory.get("best_video", {}),
        }

    @staticmethod
    def _accept_challenger_from_gate(gate_result: Any) -> bool:
        if gate_result.decision != "B_BETTER":
            return False
        if gate_result.swapped_decision is not None and gate_result.swapped_decision != "A_BETTER":
            return False
        return True

    @staticmethod
    def _write_trajectory(output_dir: Path, trajectory: dict[str, Any]) -> Path:
        path = output_dir / "trajectory.json"
        path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
        if "stats" in trajectory:
            (output_dir / "stats.json").write_text(json.dumps(trajectory["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _phase_start(self) -> tuple[float, tuple[int, int], int]:
        vision_count = -1 if self.text_mllm is self.vision_mllm else len(self.vision_mllm.calls)
        return perf_counter(), (len(self.text_mllm.calls), vision_count), len(self.t2v.calls)

    def _record_phase(
        self,
        stats: dict[str, Any],
        name: str,
        started: float,
        mllm_before: tuple[int, int],
        t2v_before: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        text_before, vision_before = mllm_before
        if vision_before < 0:
            mllm_calls = self._tagged_calls(self.text_mllm.calls[text_before:], "shared")
        else:
            mllm_calls = self._tagged_calls(self.text_mllm.calls[text_before:], "text")
            mllm_calls += self._tagged_calls(self.vision_mllm.calls[vision_before:], "vision")
        t2v_calls = self.t2v.calls[t2v_before:]
        phase = {
            "name": name,
            "elapsed_seconds": perf_counter() - started,
            "mllm_call_count": len(mllm_calls),
            "t2v_call_count": len(t2v_calls),
            "mllm_elapsed_seconds": sum(float(call.get("elapsed_seconds", 0.0)) for call in mllm_calls),
            "t2v_elapsed_seconds": sum(float(call.get("elapsed_seconds", 0.0)) for call in t2v_calls),
            "usage": self._sum_usage(call.get("usage") for call in mllm_calls),
        }
        if extra:
            phase.update(extra)
        stats["phases"].append(phase)

    def _finalize_stats(self, stats: dict[str, Any], run_started: float) -> None:
        stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
        stats["total_elapsed_seconds"] = perf_counter() - run_started
        all_mllm_calls = self.all_mllm_calls()
        stats["mllm_calls"] = all_mllm_calls
        stats["t2v_calls"] = self.t2v.calls
        stats["mllm_call_count"] = len(all_mllm_calls)
        stats["t2v_call_count"] = len(self.t2v.calls)
        stats["mllm_elapsed_seconds"] = sum(float(call.get("elapsed_seconds", 0.0)) for call in all_mllm_calls)
        stats["t2v_elapsed_seconds"] = sum(float(call.get("elapsed_seconds", 0.0)) for call in self.t2v.calls)
        stats["usage_total"] = self._sum_usage(call.get("usage") for call in all_mllm_calls)

    @staticmethod
    def _tagged_calls(calls: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        return [{**call, "model_role": role} for call in calls]

    @classmethod
    def _sum_usage(cls, usages: list[dict[str, int] | None] | Any) -> dict[str, int]:
        total = cls._zero_usage()
        for usage in usages:
            if not usage:
                continue
            total["input_tokens"] += int(usage.get("input_tokens") or 0)
            total["output_tokens"] += int(usage.get("output_tokens") or 0)
            total["total_tokens"] += int(usage.get("total_tokens") or 0)
        return total

    @staticmethod
    def _zero_usage() -> dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
