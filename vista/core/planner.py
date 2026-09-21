from __future__ import annotations

from vista.clients.mllm_client import MLLMClient
from vista.schemas import PromptCandidate
from vista.utils.config import load_prompt, render_template
from vista.utils.json_utils import parse_json_like


class PromptPlanner:
    def __init__(self, mllm: MLLMClient, run_config: dict):
        self.mllm = mllm
        self.run_config = run_config
        self.template = load_prompt("planner.txt")

    def plan(self, user_prompt: str) -> list[PromptCandidate]:
        num_scenes = int(self.run_config.get("initial_prompt_samples", 3))
        prompt = render_template(
            self.template,
            video_type=self.run_config.get("video_type", "real-world"),
            duration_seconds=self.run_config.get("duration_seconds", 5),
            input_prompt=user_prompt,
            num_scenes=num_scenes,
        )
        response = self.mllm.call(prompt, expected="planner")
        parsed = parse_json_like(response.text, default=[])

        candidates: list[PromptCandidate] = []
        if isinstance(parsed, list):
            for index, item in enumerate(parsed[:num_scenes]):
                text = item if isinstance(item, str) else str(item)
                if self._is_usable_prompt(text):
                    candidates.append(
                        PromptCandidate(
                            id=f"r0_p{index}",
                            prompt=text.strip(),
                            round_index=0,
                            metadata={"source": "planner", "dry_run": response.dry_run},
                        )
                    )

        candidates.append(
            PromptCandidate(
                id="r0_residual",
                prompt=user_prompt,
                round_index=0,
                metadata={"source": "user_residual"},
            )
        )
        return candidates

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
