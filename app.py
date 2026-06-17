from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from chatbot_llm.inference import Chatbot


def main() -> None:
    load_dotenv()
    chatbot = Chatbot.from_pretrained(load_in_4bit=os.getenv("CHATBOT_LOAD_IN_4BIT") == "1")

    demo = gr.ChatInterface(
        fn=chatbot.reply,
        title="Vietnamese Chatbot LLM",
        examples=[
            "Xin chào, bạn có thể giúp tôi học tiếng Anh không?",
            "Tóm tắt đoạn văn sau bằng tiếng Việt ngắn gọn.",
            "Gợi ý cho tôi một kế hoạch học machine learning trong 4 tuần.",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
