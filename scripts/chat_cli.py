from __future__ import annotations

import argparse

from dotenv import load_dotenv

from chatbot_llm.inference import Chatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the Vietnamese LLM in the terminal.")
    parser.add_argument("--model-id", default=None, help="Hugging Face model id or local model path.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading on CUDA.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    chatbot = Chatbot.from_pretrained(model_id=args.model_id, load_in_4bit=args.load_in_4bit)
    history: list[dict[str, str]] = []

    print("Type /exit to stop.")
    while True:
        message = input("You: ").strip()
        if message in {"/exit", "/quit"}:
            break
        answer = chatbot.reply(message, history)
        print(f"Assistant: {answer}")
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
        )


if __name__ == "__main__":
    main()
