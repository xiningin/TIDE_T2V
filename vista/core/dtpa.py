from __future__ import annotations

from typing import Any

from vista.clients.mllm_client import MLLMClient
from vista.core.mmac import critiques_to_text
from vista.core.strict_constraints import strict_action_block
from vista.schemas import CritiqueResult, PromptCandidate, VideoCandidate
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import parse_json_like, parse_python_list


class DTPAOptimizer:
    def __init__(self, mllm: MLLMClient, run_config: dict[str, Any], selection_config: dict[str, Any] | None = None):
        self.mllm = mllm
        self.run_config = run_config
        self.selection_config = selection_config or {}
        self.dtpa_template = load_prompt("dtpa.txt")
        self.sampler_template = load_prompt("prompt_sampler.txt")

    def improve(
        self,
        user_prompt: str,
        winner: VideoCandidate,
        critiques: list[CritiqueResult],
        round_index: int,
    ) -> list[PromptCandidate]:
        feedback = critiques_to_text(critiques)
        dtpa_prompt = render_template(
            self.dtpa_template,
            input_prompt=user_prompt,
            scene_prompt=winner.prompt,
            all_feedback=feedback,
        )
        dtpa_prompt = self._with_strict_constraints(dtpa_prompt, purpose="dtpa")
        # Video evidence has already been converted to textual critiques by
        # the vision evaluator. Keep optimization text-only for model routing.
        dtpa_response = self.mllm.call(dtpa_prompt, expected="dtpa", thinking=False)
        modifications = parse_python_list(dtpa_response.text)
        if not modifications:
            modifications = ["Keep the current prompt because no clear major issue was identified."]

        num_scenes = int(self.run_config.get("improved_prompt_samples", 3))
        sampler_prompt = render_template(
            self.sampler_template,
            duration_seconds=self.run_config.get("duration_seconds", 5),
            input_prompt=user_prompt,
            scene_prompt=winner.prompt,
            negative_prompt=winner.negative_prompt or "(none; use only source-triggered artifact terms)",
            suggested_modifications=modifications,
            num_scenes=num_scenes,
        )
        sampler_prompt = self._with_strict_constraints(sampler_prompt, purpose="sampler")
        sampler_response = self.mllm.call(
            sampler_prompt,
            expected="prompt_sampler",
            temperature=0.7,
            thinking=False,
        )
        parsed = self._candidate_items(sampler_response.text)
        if len(parsed) != num_scenes:
            repair_prompt = (
                f"Convert the response below into exactly {num_scenes} usable prompt pairs. "
                "Return only a JSON list. Every element must be an object with non-empty "
                "positive_prompt and negative_prompt string fields. Preserve the proposed "
                "content; do not provide explanations or markdown.\n\n"
                f"Response to repair:\n{sampler_response.text[-30000:]}"
            )
            repaired = self.mllm.call(
                repair_prompt,
                expected="prompt_sampler_repair",
                temperature=0,
                thinking=False,
            )
            parsed = self._candidate_items(repaired.text)

        candidates: list[PromptCandidate] = []
        if isinstance(parsed, list):
            for index, item in enumerate(parsed[:num_scenes]):
                if isinstance(item, dict):
                    text = str(item.get("positive_prompt") or item.get("prompt") or "")
                    negative = str(item.get("negative_prompt") or "")
                else:
                    text = item if isinstance(item, str) else str(item)
                    negative = winner.negative_prompt
                if self._is_usable_prompt(text):
                    candidates.append(
                        PromptCandidate(
                            id=f"r{round_index}_p{index}",
                            prompt=text.strip(),
                            round_index=round_index,
                            negative_prompt=self._normalize_negative(negative),
                            parent_id=winner.prompt_candidate.id,
                            metadata={"source": "dtpa", "modifications": modifications},
                        )
                    )

        if bool(self.run_config.get("include_champion_residual", True)):
            candidates.append(
                PromptCandidate(
                    id=f"r{round_index}_champion_residual",
                    prompt=winner.prompt,
                    round_index=round_index,
                    negative_prompt=winner.negative_prompt,
                    parent_id=winner.prompt_candidate.id,
                    metadata={"source": "champion_residual", "modifications": modifications},
                )
            )
        if len(candidates) != num_scenes and bool(self.run_config.get("require_exact_improved_samples", False)):
            raise ValueError(f"Prompt optimizer returned {len(candidates)} usable candidates; expected exactly {num_scenes}")
        return candidates

    @staticmethod
    def _candidate_items(text: str) -> list[Any]:
        """Accept the requested list plus common harmless JSON wrappers."""
        parsed = parse_json_like(text, default=[])
        if isinstance(parsed, dict):
            for key in ("candidates", "prompts", "prompt_pairs", "results", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _normalize_negative(value: str) -> str:
        terms: list[str] = []
        for raw in value.replace(";", ",").split(","):
            term = " ".join(raw.strip().split())
            if term and term.lower() not in {x.lower() for x in terms}:
                terms.append(term)
        return ", ".join(terms[:8])

    @staticmethod
    def _is_usable_prompt(text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        placeholders = {
            "prompt",
            "prompt string",
            "video prompt",
            "scene prompt",
            "string",
            "...",
            "n/a",
            "none",
        }
        return (
            len(normalized) >= 20
            and normalized not in placeholders
            and not normalized.startswith("scene description")
            and not normalized.endswith("...")
        )

    def _with_strict_constraints(self, prompt: str, *, purpose: str) -> str:
        return prompt + strict_action_block(self.selection_config, purpose=purpose)
