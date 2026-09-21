from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class PromptCandidate:
    id: str
    prompt: str
    round_index: int
    negative_prompt: str = ""
    parent_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class VideoCandidate:
    id: str
    prompt_candidate: PromptCandidate
    video_path: str
    round_index: int
    seed: int | None = None
    metadata: JsonDict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.prompt_candidate.prompt

    @property
    def negative_prompt(self) -> str:
        return self.prompt_candidate.negative_prompt

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        return data


@dataclass
class CritiqueResult:
    dimension: str
    normal: str
    adversarial: str
    meta: str
    scores: dict[str, float] = field(default_factory=dict)

    def mean_score(self) -> float | None:
        if not self.scores:
            return None
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class PairwiseResult:
    video_a_id: str
    video_b_id: str
    decision: str
    swapped_decision: str | None
    winner_id: str
    raw_forward: str
    raw_swapped: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class TournamentResult:
    winner: VideoCandidate
    probing_feedback: dict[str, str]
    rounds: list[list[PairwiseResult]]

    def to_dict(self) -> JsonDict:
        return {
            "winner": self.winner.to_dict(),
            "probing_feedback": self.probing_feedback,
            "rounds": [[item.to_dict() for item in round_items] for round_items in self.rounds],
        }


@dataclass
class VistaRunResult:
    user_prompt: str
    best_video: VideoCandidate
    best_prompt: PromptCandidate
    trajectory_path: str
    output_dir: str

    def to_dict(self) -> JsonDict:
        return {
            "user_prompt": self.user_prompt,
            "best_video": self.best_video.to_dict(),
            "best_prompt": self.best_prompt.to_dict(),
            "trajectory_path": str(Path(self.trajectory_path)),
            "output_dir": str(Path(self.output_dir)),
        }


@dataclass
class PromptRewriteResult:
    user_prompt: str
    rewritten_prompt: str
    negative_prompt: str
    skill_dir: str
    skill_revision: int
    skill_sha256: str
    activated_rule_ids: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> JsonDict:
        return {**asdict(self), "positive_prompt": self.rewritten_prompt}

    @property
    def positive_prompt(self) -> str:
        return self.rewritten_prompt
