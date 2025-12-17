# SafeDiag-Bench

## Clinical Diagnostic Assistant Safety Benchmark

A safety-first benchmark for evaluating large language models (LLMs) used as **clinician-facing diagnostic decision support tools**.

> 📖 **[Read the Full Methodology](methodology.html)**  
> For a detailed explanation of the safety testing framework, scoring logic, and design principles, please see our [methodology page](methodology.html).

This benchmark explicitly measures **Diagnostic usefulness** and **Safety-critical escalation behavior**. Unsafe behavior is surfaced, not averaged away.

---

## Running the benchmark

### Prerequisites

* Docker and Docker Compose
* OpenRouter API key (for inference)
* Download ICD10 code reference - to data/section111_valid_icd10_october2025.xlsx
* Download [DDXPlus dataset](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585) and extract to data/ddxplus_v0

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env.local
docker compose build
```

---

### Workflow

#### 1. Generate test cases (reproducible random subset)

```bash
# Create standard test sets
docker compose run --rm evaluator ./scripts/create_standard_test_sets.sh

# Or create custom test set
docker compose run --rm evaluator python3 scripts/prep_test_cases.py \
  --output data/test_sets/my-test.json \
  --num-cases 100 \
  --seed 42
```

Generated test sets: `dev-v0.json` (10), `test-v0.json` (100), `eval-v0.json` (500), `full-eval-v0.json` (2000)

#### 2. Run inference

```bash
docker compose run --rm inference python3 -m inference.run_inference \
  --cases data/test_sets/test-v0.json \
  --model "anthropic/claude-3.5-sonnet" \
  --out results/artifacts/claude-predictions.json
```

#### 3. Evaluate model

```bash
docker compose run --rm evaluator python3 -m evaluator.cli \
  --cases data/test_sets/test-v0.json \
  --predictions results/artifacts/claude-predictions.json \
  --model-name "claude-3.5-sonnet" \
  --model-version "2025-01" \
  --out results/artifacts/claude-eval.json
```

or, use this utility that wraps the openrouter runner for several models at once - 100 cases each.

```
./scripts/test_models.sh 100 anthropic/claude-haiku-4.5 anthropic/claude-sonnet-4.5 openai/gpt-oss-120b google/gemini-2.5-flash-lite deepseek/deepseek-chat-v3-0324 openai/gpt-4o-mini
```

#### 4. Review results

**1. Summary Metrics:**

```bash
cat results/artifacts/claude-eval.json
```

**2. Clinical Review Transcript:**

Generate a human-readable log for doctor review:

```bash
docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
  --cases data/test_sets/test-v0.json \
  --predictions results/artifacts/claude-predictions.json \
  --model-name "Claude 3.5 Sonnet" \
  --out results/artifacts/claude-transcript.txt
```

**3. Interactive Leaderboard:**

```bash
./scripts/serve_leaderboard.sh
# Open http://localhost:18080/
```

Example output:

```json
{
  "model": "claude-sonnet-4",
  "version": "2025-01",
  "cases": 100,
  "safety": {
    "missed_escalations": 5,
    "overconfident_wrong": 0,
    "unsafe_reassurance": 1
  },
  "effectiveness": {
    "top1_recall": 0.43,
    "top3_recall": 0.71
  }
}
```

---

### Key Benefits

* **Reproducible**: Same seed → identical test cases for fair model comparison
* **Traceable**: Full metadata (seed, timestamp) embedded in outputs
* **Auditable**: Safety failures explicitly counted, not averaged away

---

## Submitting results

To contribute results:

1. **Use a frozen test set** (e.g., `eval-v0.json`, seed=42)
2. **Run inference & evaluation** using the standard pipeline
3. **Copy the result to the leaderboard** (`cp results/artifacts/model-name.json leaderboard/`)
4. **Commit the artifact**
5. **Open a Pull Request** with your inference script

Results are curated to ensure integrity.

---

## Status

* **Current version:** v0
* Spec is frozen
* Evaluator and API under active development

Early feedback and collaboration are welcome.

---

## Other Projects

* This project takes inspiration and builds on some approach in [MedS-Ins](https://github.com/MAGIC-AI4Med/MedS-Ins)

## License & governance

Copyright © Cortico Health Technologies Inc 2025

This work is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

Initial releases are curated to preserve evaluation integrity.


