from __future__ import annotations

import argparse
import os

import torch
from dotenv import load_dotenv

from chatbot_llm.preferences import load_preference_jsonl, preference_examples_to_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reward model from Vietnamese preference pairs.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--preferences", default="data/preferences/sample_preferences.jsonl")
    parser.add_argument("--output-dir", default="./checkpoint/reward-model")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
    from trl import RewardTrainer

    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_preference_jsonl(args.preferences)
    dataset = Dataset.from_list(preference_examples_to_rows(examples, tokenizer))

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        token=token,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["score"],
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        bf16=dtype is torch.bfloat16,
        fp16=torch.cuda.is_available() and dtype is not torch.bfloat16,
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_length=args.max_length,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved reward model to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set.")
        trainer.model.push_to_hub(args.hub_model_id, token=token)
        tokenizer.push_to_hub(args.hub_model_id, token=token)
        print(f"pushed reward model to {args.hub_model_id}")


if __name__ == "__main__":
    main()
