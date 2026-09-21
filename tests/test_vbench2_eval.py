from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vbench2_eval.prepare_skill_arms import (
    _sample_count,
    load_rows,
    merge_negative,
    paired_seed,
    parse_skill_arms,
    safe_stem,
)


class VBench2PrepareTests(unittest.TestCase):
    def test_merge_negative_is_stable_and_case_insensitive(self) -> None:
        self.assertEqual(
            merge_negative("jitter, ghosting", "Ghosting; deformed hands."),
            "jitter, ghosting, deformed hands",
        )

    def test_safe_stem_falls_back_for_paths(self) -> None:
        value = safe_stem("unsafe/path")
        self.assertEqual(len(value), 24)
        self.assertNotIn("/", value)

    def test_diversity_sample_floor(self) -> None:
        self.assertEqual(_sample_count("Diversity", 3, 20), 20)
        self.assertEqual(_sample_count("Composition", 3, 20), 3)

    def test_seed_schedule_is_arm_independent(self) -> None:
        raw = [paired_seed(42, 7, index) for index in range(3)]
        trained = [paired_seed(42, 7, index) for index in range(3)]
        self.assertEqual(raw, trained)
        self.assertEqual(raw, [7042, 7043, 7044])

    def test_parse_skill_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            skill = Path(temporary) / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("test", encoding="utf-8")
            (skill / "rules.json").write_text('{"revision": 1}', encoding="utf-8")
            parsed = parse_skill_arms([f"initial={skill}"])
            self.assertEqual(parsed["initial"], skill.resolve())
            with self.assertRaises(ValueError):
                parse_skill_arms([f"raw={skill}"])

    def test_load_rows_filters_and_preserves_order(self) -> None:
        payload = {
            "dimensions": [
                {
                    "dimension": "Composition",
                    "prompts": [
                        {"id": "a", "source_line": 1, "canonical_prompt": "A"},
                        {"id": "b", "source_line": 2, "canonical_prompt": "B"},
                    ],
                },
                {
                    "dimension": "Mechanics",
                    "prompts": [{"id": "c", "source_line": 3, "canonical_prompt": "C"}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary) / "suite.json"
            suite.write_text(json.dumps(payload), encoding="utf-8")
            _, rows = load_rows(suite, dimensions="Composition", max_prompts=1)
        self.assertEqual([row["prompt_id"] for row in rows], ["a"])


if __name__ == "__main__":
    unittest.main()
