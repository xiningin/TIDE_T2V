#!/usr/bin/env python3
"""Compatibility entry point for :mod:`vbench2_eval.prepare_skill_arms`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbench2_eval.prepare_skill_arms import main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) == 1:
        skill_root = ROOT / "outputs/evalcrafter_rulepool4_qwen36_local"
        sys.argv.extend(
            [
                "--skill-arm",
                f"initial={skill_root / 'wave_0000/frozen_skill'}",
                "--skill-arm",
                f"trained={skill_root / 'wave_0011/frozen_skill'}",
            ]
        )
    main()
