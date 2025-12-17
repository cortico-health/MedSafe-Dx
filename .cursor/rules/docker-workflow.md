# Docker Workflow

## General Principle

**Always use Docker for running benchmark tasks, utility scripts, and webserver.** This ensures reproducibility and prevents environment-specific issues.

## Available Services

### 1. converter
Converts raw DDXPlus CSV data to benchmark-ready JSON format.

```bash
docker compose run --rm converter
```

### 2. inference
Runs model inference on benchmark cases via OpenRouter API.

```bash
docker compose run --rm inference python3 -m inference.run_inference \
  --limit 10 \
  --out results/artifacts/model-name.json
```

### 3. evaluator
Evaluates model predictions against gold labels.

```bash
docker compose run --rm evaluator python3 -m evaluator.cli \
  --cases data/ddxplus_v0/cases.json \
  --predictions results/artifacts/model-name.json \
  --model-name "model-name" \
  --model-version "2025-01" \
  --out results/artifacts/model-name-eval.json
```

## Environment Variables

API keys are loaded from `.env.local` (gitignored).

Supported variables:
- `OPENROUTER_API_KEY` or `OPENROUTER_KEY`

## Why Docker?

1. **Reproducibility**: Same environment for all users
2. **Isolation**: No local Python environment conflicts  
3. **Simplicity**: No need to manage virtualenvs or pip installs
4. **Consistency**: All benchmark runs use identical dependencies

## Building

After changing dependencies in `requirements.txt`:

```bash
docker compose build
```

