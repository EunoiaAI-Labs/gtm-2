"""Showcase a tiny self-contained LLM that knows about HTML tags."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dataset_utils import build_prompt_completion_pairs, load_html_tag_dataset
from model import LocalHTMLTagLLM


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


def _call_hf_generator(
    generator: Callable[[str, int, int], Sequence[dict[str, Any]]],
    prompt: str,
    max_length: int,
    num_return_sequences: int,
) -> str:
    """Invoke a HuggingFace-style generator and extract the text output."""

    results = generator(prompt, max_length, num_return_sequences)
    if not results:
        return ""

    first = results[0]
    if isinstance(first, dict) and "generated_text" in first:
        text = first["generated_text"]
        if isinstance(text, str):
            return text
    if isinstance(first, str):
        return first
    raise TypeError(
        "Unsupported response from generator: expected a mapping containing "
        "'generated_text' or a plain string."
    )


def generate_text(
    generator: LocalHTMLTagLLM | Callable[[str, int, int], Sequence[dict[str, Any]]],
    prompt: str,
    max_length: int,
    *,
    num_return_sequences: int = 1,
) -> str:
    """Generate a completion for the provided prompt using the selected LLM."""

    if hasattr(generator, "generate"):
        # ``LocalHTMLTagLLM`` exposes a ``generate`` method matching this API.
        return generator.generate(prompt, max_length)  # type: ignore[return-value]

    if callable(generator):
        return _call_hf_generator(generator, prompt, max_length, num_return_sequences)

    raise TypeError("Generator must expose a 'generate' method or be callable.")


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

