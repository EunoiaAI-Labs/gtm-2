"""Utilities for turning the raw HTML cheatsheet into model-ready data.

The repository ships a single ``dataset.txt`` file that contains a long list
of HTML tags with short descriptions.  The goal of this module is to transform
those loosely formatted notes into structured prompt/completion pairs that can
be consumed by a language model fine-tuning pipeline.

Two small steps are involved:

``load_html_tag_dataset``
    Reads ``dataset.txt`` (or another file following the same format) and
    extracts ``(tag, description)`` tuples.  The helper is robust to
    multi-line descriptions and gracefully skips headings or blank lines.

``build_prompt_completion_pairs``
    Converts the ``(tag, description)`` tuples into dictionaries that follow
    the common prompt/completion JSONL convention used by many LLM training
    utilities.

Both helpers are lightweight and avoid pulling heavy dependencies so that they
can be used in simple data preparation scripts and unit tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, MutableSequence, Sequence, Tuple

__all__ = [
    "load_html_tag_dataset",
    "build_prompt_completion_pairs",
    "save_prompt_completion_jsonl",
]


_TAG_PATTERN = re.compile(r"<[^>]+>")


def _looks_like_tag_line(line: str) -> bool:
    """Return ``True`` when ``line`` resembles an HTML tag declaration."""

    if not line:
        return False
    return bool(_TAG_PATTERN.search(line))


def load_html_tag_dataset(path: Path | str) -> List[Tuple[str, str]]:
    """Parse ``dataset.txt`` into ``(tag, description)`` tuples.

    Parameters
    ----------
    path:
        Location of the text dataset.  Either a :class:`pathlib.Path` instance
        or a plain string.

    Returns
    -------
    list of tuple[str, str]
        A list containing ``(tag, description)`` pairs.  Multi-line
        descriptions are collapsed into a single whitespace-normalised string.
    """

    dataset_path = Path(path)
    raw_lines = dataset_path.read_text(encoding="utf-8").splitlines()

    pairs: List[Tuple[str, str]] = []
    current_tag: str | None = None
    description_lines: MutableSequence[str] = []

    def flush_current() -> None:
        nonlocal current_tag, description_lines
        if current_tag and description_lines:
            description = " ".join(part.strip() for part in description_lines)
            # Normalise whitespace so that descriptions look tidy when fed to
            # an LLM training pipeline.
            description = " ".join(description.split())
            pairs.append((current_tag, description))
        current_tag = None
        description_lines = []

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            # Blank lines simply separate sections – flush accumulated text.
            flush_current()
            continue

        if _looks_like_tag_line(line):
            flush_current()
            current_tag = line
            continue

        if current_tag is None:
            # Skip headings or explanatory text that is not tied to a tag.
            continue

        description_lines.append(line)

    # Capture the trailing pair (if any) once the loop finishes.
    flush_current()
    return pairs


def build_prompt_completion_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    instruction: str = "Describe the HTML element",
) -> List[dict[str, str]]:
    """Turn ``(tag, description)`` tuples into prompt/completion dictionaries."""

    results: List[dict[str, str]] = []
    for tag, description in pairs:
        prompt = f"{instruction} `{tag}`."
        # Fine-tuning utilities such as the OpenAI JSONL format expect the
        # completion to start with a leading space so that the model learns to
        # separate the prompt from the answer.
        completion = " " + description.strip()
        results.append({"prompt": prompt, "completion": completion})
    return results


def save_prompt_completion_jsonl(
    data: Iterable[dict[str, str]], path: Path | str
) -> Path:
    """Persist prompt/completion dictionaries as a JSONL file."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fh:
        for record in data:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")
    return target_path

