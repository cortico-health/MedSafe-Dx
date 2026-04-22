# MedSafe-Dx (v0)

## A Safety-Focused Benchmark for Evaluating LLMs in Clinical Diagnostic Decision Support

A safety-first benchmark for evaluating large language models (LLMs) used as **clinician-facing diagnostic decision support tools**. MedSafe-Dx evaluates three dimensions: **escalation sensitivity**, **avoidance of false reassurance**, and **calibration of uncertainty**. Unsafe behavior is surfaced, not averaged away.

**Authors:** Clark Van Oyen, Namrah Mirza-Haq (Cortico Health Technologies)

- 📊 **Live leaderboard:** <https://msdx.cortico.health/>
- 📄 **Methodology & results:** <https://msdx.cortico.health/report.html> (rendered from [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md))
- 📝 **Preprint (medRxiv):** <https://doi.org/10.64898/2026.04.14.26350711>

> **Cite as:** Van Oyen C, Mirza-Haq N. *MedSafe-Dx (v0): A Safety-Focused Benchmark for Evaluating LLMs in Clinical Diagnostic Decision Support.* medRxiv 2026.04.14.26350711; doi: <https://doi.org/10.64898/2026.04.14.26350711>

---

## Running the benchmark

### Prerequisites

* Docker and Docker Compose
* OpenRouter API key (for inference)
* Download [ICD10 code reference](https://www.cms.gov/files/document/valid-icd-10-list.xlsx-0) → `data/section111_valid_icd10_october2025.xlsx`
* Download [DDXPlus dataset](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585) and extract to `data/ddxplus_v0`

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env.local
docker compose build
```

---

### Workflow

#### 1. Generate test cases (reproducible random subset based on seed)

```bash
# Create standard test sets
docker compose run --rm evaluator ./scripts/create_standard_test_sets.sh

# Or create a custom test set
docker compose run --rm evaluator python3 scripts/prep_test_cases.py \
  --output data/test_sets/my-test.json \
  --num-cases 250 \
  --seed 42
```

Standard test sets: `dev-v0.json` (10), `test-v0.json` (100), **`eval-250-v0.json` (250 — primary published eval set, seed=42)**, `eval-v0.json` (500), `full-eval-v0.json` (2000).

#### 2. Run inference

```bash
docker compose run --rm inference python3 -m inference.run_inference \
  --cases data/test_sets/eval-250-v0.json \
  --model "anthropic/claude-sonnet-4.5" \
  --out results/artifacts/claude-sonnet-4.5-predictions.json
```

#### 3. Evaluate model

```bash
docker compose run --rm evaluator python3 -m evaluator.cli \
  --cases data/test_sets/eval-250-v0.json \
  --predictions results/artifacts/claude-sonnet-4.5-predictions.json \
  --model-name "claude-sonnet-4.5" \
  --model-version "2025-09" \
  --out results/artifacts/claude-sonnet-4.5-eval.json
```

Or use this utility that wraps the OpenRouter runner for several models at once:

```bash
./scripts/test_models.sh 250 anthropic/claude-haiku-4.5 anthropic/claude-sonnet-4.5 \
  openai/gpt-oss-120b google/gemini-2.5-flash-lite \
  deepseek/deepseek-chat-v3-0324 openai/gpt-4o-mini
```

#### 4. Review results

**Summary metrics:**

```bash
cat results/artifacts/claude-sonnet-4.5-eval.json
```

**Clinical review transcript** — human-readable log for doctor review:

```bash
docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
  --cases data/test_sets/eval-250-v0.json \
  --predictions results/artifacts/claude-sonnet-4.5-predictions.json \
  --model-name "Claude Sonnet 4.5" \
  --out results/artifacts/claude-sonnet-4.5-transcript.txt
```

**Interactive leaderboard** (local):

```bash
./scripts/serve_leaderboard.sh
# Open http://localhost:18080/
```

The same UI is published at <https://msdx.cortico.health/>.

Example output:

```json
{
  "model": "claude-sonnet-4.5",
  "version": "2025-09",
  "cases": 250,
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

### Key benefits

* **Reproducible:** same seed → identical test cases for fair model comparison
* **Traceable:** full metadata (seed, timestamp, dataset hash, prediction hash) embedded in outputs
* **Auditable:** safety failures explicitly counted, not averaged away

---

## Submitting results

To contribute results:

1. Use a frozen test set (the published eval set is `eval-250-v0.json`, seed=42).
2. Run inference and evaluation using the standard pipeline above.
3. Copy the eval output to the leaderboard: `cp results/artifacts/<model-name>-eval.json leaderboard/`.
4. Commit the artifact.
5. Open a Pull Request with your inference command and any prompt/config details.

Results are curated to preserve evaluation integrity.

---

## Related work

* Builds on approaches from [MedS-Ins](https://github.com/MAGIC-AI4Med/MedS-Ins).
* See `BENCHMARK_REPORT.md` §1.5 for positioning relative to MedQA/USMLE-style knowledge benchmarks and rubric-based systems like HealthBench.

---

## License

Copyright © Cortico Health Technologies Inc 2026

This work is licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), matching the medRxiv preprint.

For commercial use or derivative works, contact <solutions@cortico.health>.
