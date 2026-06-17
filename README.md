# Chatbot LLM - Vietnamese

Vietnamese chatbot project for learning the full LLM application path:

1. Start from `Llama-3.2-1B-Instruct`.
2. Fine-tune with QLoRA/SFT on Vietnamese multi-turn chat data.
3. Save or push the tuned model to Hugging Face Hub.
4. Serve the model through a terminal chat loop or Gradio web UI.
5. Extend later with reward modeling and PPO/RLHF.

The original learning notebook is `[Training]_LLM_SFT.ipynb`. The reusable project code lives in `src/`, `scripts/`, and `app.py`.

## Project Layout

```text
.
├── [Training]_LLM_SFT.ipynb      # notebook walkthrough for SFT
├── app.py                        # Gradio chatbot UI
├── scripts/
│   ├── chat_cli.py               # terminal chatbot
│   └── train_sft.py              # QLoRA/SFT training script
├── src/chatbot_llm/
│   ├── data.py                   # dataset formatting helpers
│   └── inference.py              # model loading and generation
├── requirements.txt              # app/inference install
├── requirements-train.txt        # CUDA training install
└── .env.example                  # environment template
```

## 1. Configure Secrets

Create `.env` from the template:

```bash
cp .env.example .env
```

Edit `.env`:

```text
HF_TOKEN=hf_your_token_here
CHATBOT_MODEL_ID=your-huggingface-username/Llama-3.2-1B-Instruct-Chat-sft
CHATBOT_SYSTEM_PROMPT=Bạn là một trợ lý AI thân thiện, hãy trả lời bằng tiếng Việt.
```

Do not commit `.env`. It is ignored by Git.

## 2. Install for Chat/Inference

Use this when you already have a model on Hugging Face Hub or a local model path.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Run the Chatbot

Terminal:

```bash
python scripts/chat_cli.py --model-id "$CHATBOT_MODEL_ID"
```

Gradio web UI:

```bash
python app.py
```

If your model is private or gated, make sure `HF_TOKEN` is set in `.env`.

## 4. Install for SFT Training

QLoRA training with Unsloth is intended for Linux with an NVIDIA CUDA GPU. On macOS, use the notebook for study and run training on Colab, Kaggle, RunPod, or another CUDA machine.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

## 5. Run SFT Training

Dataset: [`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`](https://huggingface.co/datasets/5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca)

Base model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`

Quick smoke run:

```bash
python scripts/train_sft.py \
  --max-steps 5 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4
```

Longer training run:

```bash
python scripts/train_sft.py \
  --max-steps 400 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --output-dir ./checkpoint/llama3-1b-multi-conversation
```

Train and push a merged model to Hugging Face Hub:

```bash
python scripts/train_sft.py \
  --max-steps 400 \
  --output-dir ./checkpoint/llama3-1b-multi-conversation \
  --push-to-hub \
  --hub-model-id your-huggingface-username/Llama-3.2-1B-Instruct-Chat-sft
```

After pushing, set this in `.env`:

```text
CHATBOT_MODEL_ID=your-huggingface-username/Llama-3.2-1B-Instruct-Chat-sft
```

Then run `python app.py`.

## Current Status

Implemented:

- SFT notebook walkthrough.
- Reusable dataset formatting helpers.
- QLoRA/SFT training script.
- Hugging Face Hub push path for merged model.
- Terminal chat loop.
- Gradio web chat UI.

Next milestones:

- Add evaluation prompts for Vietnamese response quality.
- Add a small preference dataset format.
- Train a reward model.
- Add PPO/RLHF training.
- Package the final chatbot with Open WebUI or a hosted Gradio Space.
