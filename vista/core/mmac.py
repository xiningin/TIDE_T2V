from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.core.strict_constraints import strict_action_block
from vista.schemas import CritiqueResult, VideoCandidate
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import find_scores, parse_json_like


class MMACCritic:
    DIMENSIONS = ("visual", "context")

    def __init__(self, mllm: MLLMClient, config: dict[str, Any] | None = None):
        self.mllm = mllm
        self.config = config or {}
        self.meta_template = load_prompt("judge_meta.txt")
        self.templates = {
            ("visual", "normal"): load_prompt("judge_normal_visual.txt"),
            ("visual", "adversarial"): load_prompt("judge_adversarial_visual.txt"),
            ("context", "normal"): load_prompt("judge_normal_context.txt"),
            ("context", "adversarial"): load_prompt("judge_adversarial_context.txt"),
        }

    def critique(self, user_prompt: str, video: VideoCandidate) -> list[CritiqueResult]:
        return self._map_batch(
            list(self.DIMENSIONS),
            lambda dimension: self._critique_dimension(user_prompt, video, dimension),
        )

    def _critique_dimension(self, user_prompt: str, video: VideoCandidate, dimension: str) -> CritiqueResult:
        normal = self._judge(user_prompt, video, dimension, "normal")
        adversarial = self._judge(user_prompt, video, dimension, "adversarial")
        meta_prompt = render_template(
            self.meta_template,
            input_prompt=user_prompt,
            scene_prompt=video.prompt,
            dimension=dimension,
            normal_judge=normal,
            adversarial_judge=adversarial,
        )
        meta_prompt = self._with_strict_constraints(meta_prompt)
        meta = self.mllm.call(
            meta_prompt,
            videos=[video.video_path],
            expected="judge_meta",
            thinking=False,
        ).text
        scores = self._merge_scores(normal, adversarial, meta)
        return CritiqueResult(dimension=dimension, normal=normal, adversarial=adversarial, meta=meta, scores=scores)

    def _judge(self, user_prompt: str, video: VideoCandidate, dimension: str, role: str) -> str:
        template = self.templates[(dimension, role)]
        prompt = render_template(template, input_prompt=user_prompt, scene_prompt=video.prompt)
        prompt = self._with_strict_constraints(prompt)
        expected = "judge_normal" if role == "normal" else "judge_adversarial"
        return self.mllm.call(
            prompt,
            videos=[video.video_path],
            expected=expected,
            thinking=False,
        ).text

    def _merge_scores(self, *texts: str) -> dict[str, float]:
        merged: dict[str, list[float]] = {}
        for text in texts:
            parsed: Any = parse_json_like(text, default={})
            for key, value in find_scores(parsed).items():
                merged.setdefault(key, []).append(value)
        return {key: sum(values) / len(values) for key, values in merged.items() if values}

    def _map_batch(self, items: list[Any], func: Any) -> list[Any]:
        if not items:
            return []
        max_workers = max(1, int(getattr(self.mllm, "batch_size", 1)))
        if max_workers == 1 or len(items) == 1:
            return [func(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
            return list(executor.map(func, items))

    def _with_strict_constraints(self, prompt: str) -> str:
        return prompt + strict_action_block(self.config, purpose="critique")


def critiques_to_text(critiques: list[CritiqueResult]) -> str:
    payload = [critique.to_dict() for critique in critiques]
    return json.dumps(payload, ensure_ascii=False, indent=2)
