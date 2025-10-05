from __future__ import annotations

import unittest
from pathlib import Path

from dataset_utils import build_prompt_completion_pairs, load_html_tag_dataset


class DatasetUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_path = Path(__file__).resolve().parent.parent / "dataset.txt"

    def test_load_pairs_contains_known_entry(self) -> None:
        pairs = load_html_tag_dataset(self.dataset_path)
        mapping = dict(pairs)
        self.assertIn("<html> </html>", mapping)
        self.assertTrue(
            mapping["<html> </html>"].lower().startswith("creates an html document")
        )

    def test_multiline_descriptions_are_collapsed(self) -> None:
        pairs = load_html_tag_dataset(self.dataset_path)
        mapping = dict(pairs)
        key = "<select multiple name=? size=?> </select>"
        self.assertIn(key, mapping)
        self.assertIn(
            "menu items visible before user needs to scroll",
            mapping[key],
        )

    def test_prompt_completion_format(self) -> None:
        pairs = load_html_tag_dataset(self.dataset_path)
        prompt_completion = build_prompt_completion_pairs(pairs[:2])
        self.assertEqual(len(prompt_completion), 2)
        self.assertTrue(prompt_completion[0]["prompt"].startswith("Describe"))
        self.assertTrue(prompt_completion[0]["completion"].startswith(" "))


if __name__ == "__main__":
    unittest.main()

