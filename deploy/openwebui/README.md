# Open WebUI Deployment

Open WebUI commonly connects to Ollama. Ollama serves GGUF models, so a Transformers checkpoint from Hugging Face must be converted and quantized before Open WebUI can use it through Ollama.

## Flow

```text
SFT/RLHF Hugging Face model
  -> convert to GGUF with llama.cpp
  -> quantize GGUF
  -> create Ollama model from Modelfile
  -> select the model in Open WebUI
```

## Commands

After converting and quantizing your final model to `model.gguf`, place it next to `Modelfile` and run:

```bash
ollama create vietnamese-chatbot -f Modelfile
ollama run vietnamese-chatbot
```

Start Open WebUI:

```bash
docker run -d \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Then open:

```text
http://localhost:3000
```

## Notes

- Use the SFT or PPO/RLHF model as the final model, not the base model, if you want to demonstrate your training pipeline.
- Keep the system prompt in `Modelfile` aligned with the prompt used during SFT and RLHF.
- Test safety and Vietnamese quality with `data/eval/vietnamese_quality_prompts.jsonl`.
