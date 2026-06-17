from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = "Bạn là một trợ lý AI thân thiện, hãy trả lời bằng tiếng Việt."
DEFAULT_DATASET_NAME = "5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca"


def convert_to_chat_messages(
    conversations: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]

    for message in conversations:
        source_role = message.get("from")
        if source_role == "human":
            role = "user"
        elif source_role in {"gpt", "assistant"}:
            role = "assistant"
        else:
            continue

        content = str(message.get("value", "")).strip()
        if content:
            messages.append({"role": role, "content": content})

    return messages


def format_training_example(example: dict[str, Any], tokenizer: Any, system_prompt: str) -> dict[str, str]:
    messages = convert_to_chat_messages(example["conversations"], system_prompt=system_prompt)
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }
