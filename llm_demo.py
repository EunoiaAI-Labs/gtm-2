"""Showcase a tiny self-contained LLM that knows about HTML tags."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dataset_utils import build_prompt_completion_pairs, load_html_tag_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository's built-in miniature LLM that specialises in "
            "HTML tag descriptions."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("dataset.txt"),
        help="Dataset file that will be converted to prompts.",
    )
    parser.add_argument(
        "--model",
        default="html-tag-llm",
        help=(
            "Name of the local LLM persona to load (only 'html-tag-llm' is "
            "currently available)."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=80,
        help=(
            "Maximum length for generated completions. The local model deals in "
            "characters, so this acts as a soft character limit."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Prompt for manual inputs instead of showcasing prompts derived from "
            "the dataset."
        ),
    )
    return parser.parse_args()


def choose_prompts(prompt_completion: Sequence[dict[str, str]]) -> list[str]:
    """Pick a handful of prompts to showcase in the demo."""

    examples = []
    for example in prompt_completion[:3]:
        examples.append(example["prompt"])
    return examples


@dataclass
class LocalHTMLTagLLM:
    """A deliberately tiny "LLM" backed by the bundled HTML dataset."""

    tag_to_description: dict[str, str]

    _TAG_PATTERN = re.compile(r"<[^>]+>")

    def generate(self, prompt: str, max_length: int) -> str:
        """Return a completion for ``prompt`` using the local knowledge base."""

        tag = self._extract_tag(prompt)
        if tag and tag in self.tag_to_description:
            return self._trim(self.tag_to_description[tag], max_length)

        # Fall back to fuzzy matching: look for any known tag name inside the
        # prompt (even without angle brackets).
        lowered = prompt.lower()
        for known_tag, description in self.tag_to_description.items():
            plain = known_tag.strip("<>").lower()
            if len(plain) <= 1:
                continue
            if re.search(rf"\b{re.escape(plain)}\b", lowered):
                return self._trim(description, max_length)

        return self._trim(
            (
                "I'm a tiny in-repo language model that specialises in HTML "
                "elements, but I don't recognise that prompt yet. Try asking "
                "about a specific tag like <a> or <section>."
            ),
            max_length,
        )

    def _extract_tag(self, prompt: str) -> str | None:
        match = self._TAG_PATTERN.search(prompt)
        if match:
            return match.group(0)
        return None

    @staticmethod
    def _trim(text: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        trimmed = text[: max_length - 1].rstrip()
        return f"{trimmed}…"


def create_generator(model: str, pairs: Sequence[tuple[str, str]]) -> LocalHTMLTagLLM:
    """Create the requested local LLM instance."""

    model = model.lower()
    if model != "html-tag-llm":
        raise SystemExit(
            "Only the built-in 'html-tag-llm' persona is available. Got: "
            f"{model!r}."
        )

    tag_to_description = {tag: description for tag, description in pairs}
    return LocalHTMLTagLLM(tag_to_description)


def generate_text(generator: LocalHTMLTagLLM, prompt: str, max_length: int) -> str:
    """Generate a completion for the provided prompt using the local LLM."""

    return generator.generate(prompt, max_length)


def run_dataset_demo(
    prompts: Sequence[str], generator: LocalHTMLTagLLM, max_length: int
) -> None:
    """Run the canned dataset demonstration."""

    for prompt in prompts:
        print("=" * 80)
        print("Prompt:")
        print(prompt)
        print("\nGenerated:")
        print(generate_text(generator, prompt, max_length))


def run_interactive_session(generator: LocalHTMLTagLLM, max_length: int) -> None:
    """Interactively prompt the user for text to feed to the LLM."""

    print(
        "Enter a prompt to query the built-in HTML tag LLM "
        "(press Ctrl-D to exit)."
    )
    while True:
        try:
            prompt = input("Prompt> ")
        except EOFError:
            print()  # Add a trailing newline for clean exits.
            break

        if not prompt.strip():
            continue

        print("\nGenerated:")
        print(generate_text(generator, prompt, max_length))
        print("=" * 80)


def main() -> None:
    args = parse_args()
    pairs = load_html_tag_dataset(args.dataset)
    generator = create_generator(args.model, pairs)
    print("Loaded the repository's local html-tag-llm persona.")

    if args.interactive:
        run_interactive_session(generator, args.max_length)
        return

    prompt_completion = build_prompt_completion_pairs(pairs)
    prompts = choose_prompts(prompt_completion)
    run_dataset_demo(prompts, generator, args.max_length)


if __name__ == "__main__":
    main()

