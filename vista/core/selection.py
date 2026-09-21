from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.core.strict_constraints import strict_action_block
from vista.schemas import PairwiseResult, TournamentResult, VideoCandidate
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import parse_json_like


class TournamentSelector:
    def __init__(self, mllm: MLLMClient, config: dict[str, Any]):
        self.mllm = mllm
        self.config = config
        self.probing_template = load_prompt("probing_critique.txt")
        self.compare_template = load_prompt("pairwise_compare.txt")

    def _thinking(self) -> bool:
        """Keep tournament thinking off unless a non-training caller explicitly opts in."""
        return bool(self.config.get("tournament_thinking", False))

    def select(self, user_prompt: str, videos: list[VideoCandidate]) -> TournamentResult:
        if not videos:
            raise ValueError("Tournament requires at least one video candidate.")
        if len(videos) == 1:
            return TournamentResult(winner=videos[0], probing_feedback={}, rounds=[])

        feedback = self._generate_feedback(user_prompt, videos)
        active = list(videos)
        rounds: list[list[PairwiseResult]] = []
        while len(active) > 1:
            next_active: list[VideoCandidate] = []
            round_results: list[PairwiseResult] = []
            slots: list[VideoCandidate | tuple[VideoCandidate, VideoCandidate]] = []
            pairs: list[tuple[VideoCandidate, VideoCandidate]] = []
            iterator = iter(active)
            for video_a in iterator:
                video_b = next(iterator, None)
                if video_b is None:
                    slots.append(video_a)
                    continue
                pair = (video_a, video_b)
                slots.append(pair)
                pairs.append(pair)
            pair_results = self._map_batch(
                pairs,
                lambda pair: self._compare(user_prompt, pair[0], pair[1], feedback),
            )
            result_iter = iter(pair_results)
            for slot in slots:
                if isinstance(slot, tuple):
                    video_a, video_b = slot
                    result = next(result_iter)
                    round_results.append(result)
                    next_active.append(video_a if result.winner_id == video_a.id else video_b)
                else:
                    next_active.append(slot)
            rounds.append(round_results)
            active = next_active
        return TournamentResult(winner=active[0], probing_feedback=feedback, rounds=rounds)

    def _generate_feedback(self, user_prompt: str, videos: list[VideoCandidate]) -> dict[str, str]:
        def generate_one(video: VideoCandidate) -> tuple[str, str]:
            prompt = render_template(self.probing_template, input_prompt=user_prompt, scene_prompt=video.prompt)
            prompt = self._with_strict_constraints(prompt, purpose="probing")
            response = self.mllm.call(
                prompt,
                videos=[video.video_path],
                expected="probing",
                thinking=self._thinking(),
            )
            return video.id, response.text

        return dict(self._map_batch(videos, generate_one))

    def compare_for_improvement(
        self,
        user_prompt: str,
        incumbent: VideoCandidate,
        challenger: VideoCandidate,
    ) -> PairwiseResult:
        feedback = self._generate_feedback(user_prompt, [incumbent, challenger])
        return self._compare(user_prompt, incumbent, challenger, feedback, purpose="improvement_gate")

    def _compare(
        self,
        user_prompt: str,
        video_a: VideoCandidate,
        video_b: VideoCandidate,
        feedback: dict[str, str],
        purpose: str = "pairwise",
    ) -> PairwiseResult:
        forward_prompt = render_template(
            self.compare_template,
            input_prompt=user_prompt,
            prompt_a=video_a.prompt,
            feedback_a=feedback.get(video_a.id, ""),
            prompt_b=video_b.prompt,
            feedback_b=feedback.get(video_b.id, ""),
        )
        forward_prompt = self._with_strict_constraints(forward_prompt, purpose=purpose)
        forward = self.mllm.call(
            forward_prompt,
            videos=[video_a.video_path, video_b.video_path],
            expected="pairwise",
            thinking=self._thinking(),
        )
        decision = self._decision_from_text(forward.text)

        swapped_decision: str | None = None
        swapped_raw: str | None = None
        if self.config.get("double_check_swapped_order", True):
            swapped_prompt = render_template(
                self.compare_template,
                input_prompt=user_prompt,
                prompt_a=video_b.prompt,
                feedback_a=feedback.get(video_b.id, ""),
                prompt_b=video_a.prompt,
                feedback_b=feedback.get(video_a.id, ""),
            )
            swapped_prompt = self._with_strict_constraints(swapped_prompt, purpose=purpose)
            swapped = self.mllm.call(
                swapped_prompt,
                videos=[video_b.video_path, video_a.video_path],
                expected="pairwise",
                thinking=self._thinking(),
            )
            swapped_decision = self._decision_from_text(swapped.text)
            swapped_raw = swapped.text

        winner = self._resolve_winner(video_a, video_b, decision, swapped_decision)
        return PairwiseResult(
            video_a_id=video_a.id,
            video_b_id=video_b.id,
            decision=decision,
            swapped_decision=swapped_decision,
            winner_id=winner.id,
            raw_forward=forward.text,
            raw_swapped=swapped_raw,
        )

    def _decision_from_text(self, text: str) -> str:
        parsed = parse_json_like(text, default={})
        if isinstance(parsed, dict):
            decision = str(parsed.get("Decision", "")).strip().upper()
            if decision in {"A_BETTER", "B_BETTER", "COMPARABLE"}:
                return decision
        upper = text.upper()
        for decision in ("A_BETTER", "B_BETTER", "COMPARABLE"):
            if decision in upper:
                return decision
        return "COMPARABLE"

    def _resolve_winner(
        self,
        video_a: VideoCandidate,
        video_b: VideoCandidate,
        decision: str,
        swapped_decision: str | None,
    ) -> VideoCandidate:
        forward_winner = self._winner_from_decision(video_a, video_b, decision)
        if swapped_decision is None:
            return forward_winner or random.choice([video_a, video_b])

        swapped_winner = self._winner_from_decision(video_b, video_a, swapped_decision)
        if forward_winner and swapped_winner and forward_winner.id == swapped_winner.id:
            return forward_winner
        if forward_winner and swapped_decision == "COMPARABLE":
            return forward_winner
        if swapped_winner and decision == "COMPARABLE":
            return swapped_winner
        if self.config.get("random_on_disagreement", True):
            return random.choice([video_a, video_b])
        return forward_winner or swapped_winner or video_a

    @staticmethod
    def _winner_from_decision(video_a: VideoCandidate, video_b: VideoCandidate, decision: str) -> VideoCandidate | None:
        if decision == "A_BETTER":
            return video_a
        if decision == "B_BETTER":
            return video_b
        return None

    def _map_batch(self, items: list[Any], func: Any) -> list[Any]:
        if not items:
            return []
        max_workers = max(1, int(getattr(self.mllm, "batch_size", 1)))
        if max_workers == 1 or len(items) == 1:
            return [func(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
            return list(executor.map(func, items))

    def _with_strict_constraints(self, prompt: str, *, purpose: str) -> str:
        return prompt + strict_action_block(self.config, purpose=purpose)
