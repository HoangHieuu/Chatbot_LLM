from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from chatbot_llm.inference import Chatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vietnamese quality evaluation prompts through a chatbot model.")
    parser.add_argument("--prompts", default="data/eval/vietnamese_quality_prompts.jsonl")
    parser.add_argument("--output", default="outputs/eval_responses.jsonl")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_prompts(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()
    chatbot = Chatbot.from_pretrained(model_id=args.model_id)

    prompts = read_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in prompts:
            messages = row["messages"]
            last_user = next(message for message in reversed(messages) if message["role"] == "user")
            history = [message for message in messages if message is not last_user]
            response = chatbot.reply(last_user["content"], history=history)
            file.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "category": row.get("category"),
                        "messages": messages,
                        "response": response,
                        "rubric": row.get("rubric", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote evaluation responses to {output_path}")


if __name__ == "__main__":
    main()
