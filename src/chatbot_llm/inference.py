from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from chatbot_llm.data import DEFAULT_SYSTEM_PROMPT


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def _first_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def history_to_messages(history: list[Any] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_message = str(item[0] or "").strip()
            assistant_message = str(item[1] or "").strip()
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
    return messages


@dataclass
class Chatbot:
    model: Any
    tokenizer: Any
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        system_prompt: str | None = None,
        load_in_4bit: bool = False,
    ) -> "Chatbot":
        model_id = model_id or os.getenv("CHATBOT_MODEL_ID") or DEFAULT_MODEL_ID
        token = os.getenv("HF_TOKEN") or None

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "token": token,
            "trust_remote_code": True,
            "torch_dtype": _select_dtype(),
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt or os.getenv("CHATBOT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        )

    def reply(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        user_message = message.strip()
        if not user_message:
            return ""

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history_to_messages(history))
        messages.append({"role": "user", "content": user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = _first_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
