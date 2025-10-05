"""Small script that demonstrates prompting an off-the-shelf LLM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from dataset_utils import build_prompt_completion_pairs, load_html_tag_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a quick generation demo using a 🤗 Transformers text-generation "
            "pipeline."
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
        default="distilgpt2",
        help="Model identifier compatible with transformers.pipeline().",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=80,
        help="Maximum length for generated text (in tokens).",
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


def create_generator(model: str):
    """Create a text-generation pipeline for the requested model."""

    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised manually.
        raise SystemExit(
            "transformers is required for this demo. Install it via "
            "`pip install transformers torch` and run the script again."
        ) from exc

    return pipeline("text-generation", model=model)


def generate_text(generator: Any, prompt: str, max_length: int) -> str:
    """Generate a completion for the provided prompt using the pipeline."""

    outputs = generator(prompt, max_length=max_length, num_return_sequences=1)
    return outputs[0]["generated_text"]


def run_dataset_demo(prompts: Sequence[str], generator: Any, max_length: int) -> None:
    """Run the canned dataset demonstration."""

    for prompt in prompts:
        print("=" * 80)
        print("Prompt:")
        print(prompt)
        print("\nGenerated:")
        print(generate_text(generator, prompt, max_length))


def run_interactive_session(generator: Any, max_length: int) -> None:
    """Interactively prompt the user for text to feed to the LLM."""

    print("Enter a prompt to generate a completion (press Ctrl-D to exit).")
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
    generator = create_generator(args.model)

    if args.interactive:
        run_interactive_session(generator, args.max_length)
        return

    pairs = load_html_tag_dataset(args.dataset)
    prompt_completion = build_prompt_completion_pairs(pairs)
    prompts = choose_prompts(prompt_completion)
    run_dataset_demo(prompts, generator, args.max_length)


if __name__ == "__main__":
    main()

