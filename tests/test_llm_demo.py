from __future__ import annotations

import unittest

from llm_demo import LocalHTMLTagLLM, choose_prompts, generate_text


class LLMDemoHelpersTestCase(unittest.TestCase):
    def test_choose_prompts_returns_first_three(self) -> None:
        prompts = choose_prompts(
            [
                {"prompt": "p0", "completion": ""},
                {"prompt": "p1", "completion": ""},
                {"prompt": "p2", "completion": ""},
                {"prompt": "p3", "completion": ""},
            ]
        )
        self.assertEqual(prompts, ["p0", "p1", "p2"])

    def test_generate_text_uses_local_llm(self) -> None:
        generator = LocalHTMLTagLLM({"<a>": "Defines a hyperlink."})
        output = generate_text(generator, "Describe the HTML element `<a>`.", max_length=80)
        self.assertEqual(output, "Defines a hyperlink.")

    def test_generate_text_handles_unknown_prompt(self) -> None:
        generator = LocalHTMLTagLLM({"<a>": "Defines a hyperlink."})
        output = generate_text(generator, "What is a widget?", max_length=50)
        self.assertIn("tiny in-repo language model", output)


if __name__ == "__main__":
    unittest.main()
