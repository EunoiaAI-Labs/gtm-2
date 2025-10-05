#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="distilgpt2"
MAX_LENGTH=80
DATASET="${SCRIPT_DIR}/dataset.txt"
INTERACTIVE=1

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --model NAME          Hugging Face model identifier (default: ${MODEL}).
  --max-length TOKENS   Maximum generation length (default: ${MAX_LENGTH}).
  --dataset PATH        Dataset file for non-interactive demos.
  --demo                Run the canned dataset demo instead of interactive mode.
  --help                Show this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --demo)
      INTERACTIVE=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${INTERACTIVE}" -eq 1 ]]; then
  exec python "${SCRIPT_DIR}/llm_demo.py" --model "${MODEL}" --max-length "${MAX_LENGTH}" --interactive
else
  exec python "${SCRIPT_DIR}/llm_demo.py" --model "${MODEL}" --max-length "${MAX_LENGTH}" --dataset "${DATASET}"
fi

