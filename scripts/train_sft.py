from __future__ import annotations

import argparse
import os

import torch
from dotenv import load_dotenv

from chatbot_llm.data import DEFAULT_DATASET_NAME, DEFAULT_SYSTEM_PROMPT, format_training_example


DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune the Vietnamese chatbot with QLoRA.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="./checkpoint/llama3-1b-multi-conversation")
    parser.add_argument("--hub-model-id", default=None, help="Optional Hugging Face repo id for merged upload.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training with Unsloth requires an NVIDIA CUDA GPU.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=dtype,
        token=os.getenv("HF_TOKEN") or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "gate_proj",
        ],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    raw_dataset = load_dataset(args.dataset_name, split=args.split)
    train_dataset = raw_dataset.map(
        lambda example: format_training_example(example, tokenizer, args.system_prompt),
        remove_columns=raw_dataset.column_names,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_total_limit=4,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
        remove_unused_columns=True,
        max_steps=args.max_steps,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set.")
        model.push_to_hub_merged(
            args.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer.push_to_hub(args.hub_model_id, token=os.getenv("HF_TOKEN") or None)
        print(f"Pushed merged model and tokenizer to {args.hub_model_id}")


if __name__ == "__main__":
    main()
