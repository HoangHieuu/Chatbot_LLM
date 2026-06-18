from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class PreferenceExample:
    id: str
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    source: str | None = None
    notes: str | None = None


def _validate_messages(value: Any, field_name: str, line_number: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"line {line_number}: {field_name} must be a non-empty list of messages")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"line {line_number}: {field_name}[{index}] must be an object")

        role = item.get("role")
        content = item.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"line {line_number}: {field_name}[{index}].role must be one of {sorted(VALID_ROLES)}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"line {line_number}: {field_name}[{index}].content must be a non-empty string")

        messages.append({"role": role, "content": content.strip()})

    return messages


def parse_preference_record(record: dict[str, Any], line_number: int) -> PreferenceExample:
    example_id = record.get("id") or f"line_{line_number}"
    if not isinstance(example_id, str):
        raise ValueError(f"line {line_number}: id must be a string when provided")

    prompt = _validate_messages(record.get("prompt"), "prompt", line_number)
    chosen = _validate_messages(record.get("chosen"), "chosen", line_number)
    rejected = _validate_messages(record.get("rejected"), "rejected", line_number)

    if chosen == rejected:
        raise ValueError(f"line {line_number}: chosen and rejected must not be identical")

    source = record.get("source")
    notes = record.get("notes")
    return PreferenceExample(
        id=example_id,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        source=source if isinstance(source, str) else None,
        notes=notes if isinstance(notes, str) else None,
    )


def load_preference_jsonl(path: str | Path) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each JSONL row must be an object")
            examples.append(parse_preference_record(record, line_number))

    if not examples:
        raise ValueError(f"{path} does not contain any preference examples")
    return examples


def preference_examples_to_rows(
    examples: Iterable[PreferenceExample],
    tokenizer: Any,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for example in examples:
        rows.append(
            {
                "id": example.id,
                "chosen": tokenizer.apply_chat_template(
                    example.prompt + example.chosen,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                "rejected": tokenizer.apply_chat_template(
                    example.prompt + example.rejected,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
            }
        )
    return rows


def preference_examples_to_prompts(examples: Iterable[PreferenceExample], tokenizer: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for example in examples:
        rows.append(
            {
                "id": example.id,
                "query": tokenizer.apply_chat_template(
                    example.prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            }
        )
    return rows
