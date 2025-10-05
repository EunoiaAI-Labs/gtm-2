"""Command-line helper that converts ``dataset.txt`` into JSONL format."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_utils import (
    build_prompt_completion_pairs,
    load_html_tag_dataset,
    save_prompt_completion_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the raw HTML cheatsheet into prompt/completion JSONL that "
            "can be used for language-model fine-tuning."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path(__file__).with_name("dataset.txt"),
        type=Path,
        help="Path to the source dataset (defaults to dataset.txt in the repo).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=Path("data/html_cheatsheet.jsonl"),
        type=Path,
        help="Destination for the generated JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_html_tag_dataset(args.input)
    prompt_completion = build_prompt_completion_pairs(pairs)
    target_path = save_prompt_completion_jsonl(prompt_completion, args.output)
    print(f"Wrote {len(prompt_completion)} prompt/completion pairs to {target_path}.")


if __name__ == "__main__":
    main()

