from __future__ import annotations

import unittest

from llm_demo import LocalHTMLTagLLM, choose_prompts, generate_text
from llm_demo import choose_prompts, generate_text


class DummyGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, prompt: str, max_length: int, num_return_sequences: int):
        self.calls.append((prompt, max_length, num_return_sequences))
        return [{"generated_text": f"reply to: {prompt}"}]


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
    def test_generate_text_uses_generator(self) -> None:
        generator = DummyGenerator()
        output = generate_text(generator, "Hello world", max_length=42)
        self.assertEqual(output, "reply to: Hello world")
        self.assertEqual(generator.calls, [("Hello world", 42, 1)])


if __name__ == "__main__":
    unittest.main()
