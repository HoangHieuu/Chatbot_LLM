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
│   ├── run_evaluation_prompts.py # generate responses for quality checks
│   ├── train_ppo_rlhf.py         # PPO/RLHF training script
│   ├── train_reward_model.py     # reward model training script
│   ├── train_sft.py              # QLoRA/SFT training script
│   └── validate_preferences.py   # preference data validator
├── data/
│   ├── eval/                     # Vietnamese evaluation prompts
│   └── preferences/              # sample preference JSONL
├── deploy/                       # Open WebUI and Gradio Space notes
├── src/chatbot_llm/
│   ├── data.py                   # dataset formatting helpers
│   ├── preferences.py            # preference JSONL helpers
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
REWARD_MODEL_ID=your-huggingface-username/Llama-3.2-1B-Instruct-Chat-reward
PPO_MODEL_ID=your-huggingface-username/Llama-3.2-1B-Instruct-Chat-ppo
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

## 6. Evaluate Vietnamese Response Quality

Evaluation prompts live in:

```text
data/eval/vietnamese_quality_prompts.jsonl
```

Generate model responses:

```bash
python scripts/run_evaluation_prompts.py \
  --model-id "$CHATBOT_MODEL_ID" \
  --output outputs/eval_sft_responses.jsonl
```

Review the generated answers against each row's `rubric`. This is a lightweight manual evaluation set, not an automatic benchmark.

## 7. Build Preference Data

Preference data lives in JSONL format:

```text
data/preferences/sample_preferences.jsonl
```

Each row contains:

- `prompt`: user or multi-turn context.
- `chosen`: preferred assistant answer.
- `rejected`: worse assistant answer.
- `notes`: why the chosen answer is better.

See the full schema in `docs/preference_dataset.md`.

Validate the sample file:

```bash
python scripts/validate_preferences.py data/preferences/sample_preferences.jsonl
```

For real reward modeling, expand this file from 3 examples to hundreds or thousands of human-labeled pairs.

## 8. Train a Reward Model

The reward model learns to score `chosen` answers higher than `rejected` answers.

Smoke run:

```bash
python scripts/train_reward_model.py \
  --preferences data/preferences/sample_preferences.jsonl \
  --output-dir ./checkpoint/reward-model-smoke \
  --num-train-epochs 1
```

Train and push:

```bash
python scripts/train_reward_model.py \
  --model-id "$CHATBOT_MODEL_ID" \
  --preferences data/preferences/preferences_train.jsonl \
  --output-dir ./checkpoint/reward-model \
  --push-to-hub \
  --hub-model-id "$REWARD_MODEL_ID"
```

## 9. Run PPO/RLHF

PPO starts from your SFT model and uses the reward model to optimize generated answers.

Smoke run:

```bash
python scripts/train_ppo_rlhf.py \
  --sft-model-id "$CHATBOT_MODEL_ID" \
  --reward-model-id "$REWARD_MODEL_ID" \
  --preferences data/preferences/sample_preferences.jsonl \
  --total-episodes 4 \
  --output-dir ./checkpoint/ppo-smoke
```

Longer run and push:

```bash
python scripts/train_ppo_rlhf.py \
  --sft-model-id "$CHATBOT_MODEL_ID" \
  --reward-model-id "$REWARD_MODEL_ID" \
  --preferences data/preferences/preferences_train.jsonl \
  --total-episodes 1000 \
  --push-to-hub \
  --hub-model-id "$PPO_MODEL_ID"
```

After pushing PPO/RLHF output, set:

```text
CHATBOT_MODEL_ID=your-huggingface-username/Llama-3.2-1B-Instruct-Chat-ppo
```

Then run `python app.py`.

## 10. Deploy

Gradio Space notes:

```text
deploy/huggingface-space/README.md
```

Open WebUI/Ollama notes:

```text
deploy/openwebui/README.md
deploy/openwebui/Modelfile.template
```

## Current Status

Implemented:

- SFT notebook walkthrough.
- Reusable dataset formatting helpers.
- QLoRA/SFT training script.
- Hugging Face Hub push path for merged model.
- Terminal chat loop.
- Gradio web chat UI.
- Add evaluation prompts for Vietnamese response quality.
- Add a small preference dataset format.
- Reward model training script.
- PPO/RLHF training script.
- Open WebUI and Hugging Face Gradio Space deployment notes.

Still required for real quality:

- Run SFT, reward model training, and PPO on a CUDA machine.
- Replace `sample_preferences.jsonl` with a real human-labeled preference dataset.
- Compare base, SFT, and PPO outputs with the evaluation prompts.
