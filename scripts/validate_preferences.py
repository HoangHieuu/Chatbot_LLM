from __future__ import annotations

import argparse

from chatbot_llm.preferences import load_preference_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Vietnamese chatbot preference JSONL file.")
    parser.add_argument("path", help="Path to a preference JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_preference_jsonl(args.path)
    print(f"valid preference file: {args.path}")
    print(f"examples: {len(examples)}")


if __name__ == "__main__":
    main()
