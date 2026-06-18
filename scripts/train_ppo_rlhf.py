from __future__ import annotations

import argparse
import os
from itertools import cycle, islice

import torch
from dotenv import load_dotenv

from chatbot_llm.preferences import load_preference_jsonl, preference_examples_to_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO/RLHF using a Vietnamese SFT policy and reward model.")
    parser.add_argument("--sft-model-id", required=True, help="SFT policy model id or local path.")
    parser.add_argument("--reward-model-id", required=True, help="Reward model id or local path.")
    parser.add_argument("--preferences", default="data/preferences/sample_preferences.jsonl")
    parser.add_argument("--output-dir", default="./checkpoint/ppo-rlhf")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--total-episodes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def first_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_reward_model(model_id: str, token: str | None):
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    try:
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model, tokenizer


def score_with_reward_model(
    reward_model: torch.nn.Module,
    reward_tokenizer,
    texts: list[str],
) -> list[torch.Tensor]:
    device = first_device(reward_model)
    inputs = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.inference_mode():
        outputs = reward_model(**inputs)

    scores = outputs.logits.squeeze(-1).detach().float().cpu()
    return [score for score in scores]


def main() -> None:
    load_dotenv()
    args = parse_args()
    token = os.getenv("HF_TOKEN") or None

    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

    policy_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_id, token=token, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model_id,
        token=token,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        policy.to("cuda")
    reference_policy = create_reference_model(policy)

    reward_model, reward_tokenizer = load_reward_model(args.reward_model_id, token)

    examples = load_preference_jsonl(args.preferences)
    prompt_rows = preference_examples_to_prompts(examples, policy_tokenizer)

    config = PPOConfig(
        model_name=args.sft_model_id,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
    )
    ppo_trainer = PPOTrainer(config, policy, reference_policy, policy_tokenizer)
    policy_device = first_device(policy)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": policy_tokenizer.eos_token_id,
        "eos_token_id": policy_tokenizer.eos_token_id,
    }

    episode_iter = islice(cycle(prompt_rows), args.total_episodes)
    batch: list[dict[str, str]] = []
    completed = 0

    for row in episode_iter:
        batch.append(row)
        if len(batch) < args.batch_size:
            continue

        query_tensors: list[torch.Tensor] = []
        response_tensors: list[torch.Tensor] = []
        reward_inputs: list[str] = []

        for item in batch:
            query_tensor = policy_tokenizer(item["query"], return_tensors="pt").input_ids.squeeze(0).to(policy_device)
            response_tensor = ppo_trainer.generate(
                query_tensor,
                return_prompt=False,
                **generation_kwargs,
            )
            if response_tensor.ndim > 1:
                response_tensor = response_tensor.squeeze(0)

            response_text = policy_tokenizer.decode(response_tensor, skip_special_tokens=True)
            query_tensors.append(query_tensor)
            response_tensors.append(response_tensor.to(policy_device))
            reward_inputs.append(item["query"] + response_text)

        rewards = score_with_reward_model(reward_model, reward_tokenizer, reward_inputs)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        completed += len(batch)
        print(f"completed PPO episodes: {completed}/{args.total_episodes}; reward_mean={float(torch.stack(rewards).mean()):.4f}")
        ppo_trainer.log_stats(stats, batch, rewards)
        batch = []

    ppo_trainer.save_pretrained(args.output_dir)
    policy_tokenizer.save_pretrained(args.output_dir)
    print(f"saved PPO/RLHF policy to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set.")
        policy.push_to_hub(args.hub_model_id, token=token)
        policy_tokenizer.push_to_hub(args.hub_model_id, token=token)
        print(f"pushed PPO/RLHF policy to {args.hub_model_id}")


if __name__ == "__main__":
    main()
