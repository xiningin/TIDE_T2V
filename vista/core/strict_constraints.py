from __future__ import annotations

from typing import Any


def strict_action_enabled(config: dict[str, Any] | None) -> bool:
    strict_config = _strict_config(config)
    return bool(strict_config.get("enabled", False))


def strict_action_block(config: dict[str, Any] | None, *, purpose: str) -> str:
    strict_config = _strict_config(config)
    if not bool(strict_config.get("enabled", False)):
        return ""

    min_phases = int(strict_config.get("min_action_phases", 3))
    enforce_candidate_prompt = bool(strict_config.get("enforce_candidate_prompt", True))
    require_action_progress = bool(strict_config.get("require_temporal_action_progress", True))
    require_meaningful_improvement = bool(strict_config.get("require_meaningful_improvement", True))
    examples = strict_config.get("domain_examples") or []
    extra_instructions = strict_config.get("extra_instructions") or []

    lines = [
        "",
        "### Strict Action Constraints",
        "This run uses a stricter action-quality policy. Apply these rules before rewarding visual style, lighting, camera beauty, or broad user-prompt satisfaction.",
    ]

    if require_action_progress:
        lines.extend(
            [
                f"- The core action must visibly progress through at least {min_phases} distinct, physically connected phases across the clip. A single pose, a stretched glide, a frozen subject with moving background/water, or a character sliding through the scene without visible propulsion is a severe failure.",
                "- Judge whether the action is actually performed over time, not merely depicted in one attractive frame.",
                "- Penalize motion that is too slow to complete a plausible action cycle unless the user explicitly asks for slow motion.",
            ]
        )

    if enforce_candidate_prompt:
        lines.extend(
            [
                "- Treat the Candidate Prompt as enforceable detail, not optional decoration. If it specifies an action subtype, camera relation, environment, object count, or physical interaction, a video that contradicts those details should lose even if it loosely matches the short User Prompt.",
                "- If the candidate prompt names a specific movement type, do not accept a different movement type as equivalent merely because both are in the same broad category.",
            ]
        )

    if require_meaningful_improvement:
        lines.extend(
            [
                "- Prefer COMPARABLE when the difference is only a seed/style/camera-distance variation and does not materially improve core action completion, physical plausibility, temporal consistency, or prompt adherence.",
                "- Choose a winner only when one video is meaningfully better on the concrete failure modes above.",
            ]
        )

    if examples:
        lines.append("- Domain examples to apply when relevant:")
        lines.extend(f"  - {item}" for item in examples)

    if extra_instructions:
        lines.append("- Additional run-specific instructions:")
        lines.extend(f"  - {item}" for item in extra_instructions)

    if purpose == "probing":
        lines.extend(
            [
                "- In probing feedback, explicitly state whether the core action is active or static, whether the candidate-prompt action subtype is obeyed, and what exact temporal/physical failure should be fixed next.",
            ]
        )
    elif purpose == "pairwise":
        lines.extend(
            [
                "- In pairwise comparison, apply the strict action constraints as hard gates. A prettier video should not beat a less pretty video if it has a worse core-action or physics failure.",
                "- In the JSON Rationale, mention the decisive action/physics/prompt-adherence difference, or say that no material improvement was found.",
            ]
        )
    elif purpose == "critique":
        lines.extend(
            [
                "- In critique, do not conclude that there is no major issue when the video still shows a static glide, incomplete action cycle, wrong action subtype, implausible propulsion, or prompt-detail contradiction.",
            ]
        )
    elif purpose == "dtpa":
        lines.extend(
            [
                "- Suggested modifications must target the concrete action/physics failures. Avoid generic quality words unless they directly address a named failure.",
                "- Prefer edits that specify a short temporal action sequence with visible start, middle, and end phases.",
            ]
        )
    elif purpose == "sampler":
        lines.extend(
            [
                "- Each revised prompt should encode the core action as a temporal sequence, not just a static scene description.",
                "- Include concise negative constraints against static posing, sliding/gliding without propulsion, wrong action subtype, and disconnected environmental reactions when relevant.",
            ]
        )
    elif purpose == "improvement_gate":
        lines.extend(
            [
                "- Video A is the incumbent champion and Video B is the new challenger. Accept Video B only if it is materially better on core action completion, physical plausibility, temporal consistency, or candidate-prompt adherence. If it is merely different or comparable, keep Video A.",
            ]
        )

    return "\n".join(lines)


def _strict_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    value = config.get("strict_action_constraints")
    return value if isinstance(value, dict) else {}
