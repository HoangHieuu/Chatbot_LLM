# Preference Dataset Format

Reward modeling and PPO need preference data, not just SFT conversations. Each JSONL row stores one prompt and two candidate answers: the preferred `chosen` response and the lower-quality `rejected` response.

## JSONL Schema

```json
{
  "id": "pref_001",
  "prompt": [
    {"role": "user", "content": "User request in Vietnamese"}
  ],
  "chosen": [
    {"role": "assistant", "content": "Preferred answer"}
  ],
  "rejected": [
    {"role": "assistant", "content": "Worse answer"}
  ],
  "source": "manual_seed",
  "notes": "Why chosen is better"
}
```

`prompt`, `chosen`, and `rejected` are lists of chat messages so multi-turn prompts are supported. The reward trainer converts each row into TRL-compatible `chosen` and `rejected` strings by applying the tokenizer chat template to:

- `prompt + chosen`
- `prompt + rejected`

## Labeling Guidelines

Prefer answers that are:

- Vietnamese, natural, and directly useful.
- Faithful to the prompt and conversation context.
- Specific enough to act on.
- Safe, especially for illegal, privacy-invasive, or harmful requests.
- Honest about uncertainty.

Reject answers that are:

- Vague, generic, or not actionable.
- In the wrong language.
- Hallucinated or overconfident.
- Unsafe or policy-violating.
- Needlessly long for a simple request.

Validate a file before training:

```bash
python scripts/validate_preferences.py data/preferences/sample_preferences.jsonl
```
