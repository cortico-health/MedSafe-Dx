# MedSafe-Dx: findings from the 2026-09 leaderboard refresh

**Run:** `run-2026-09-healthbench-refresh` | **Date:** 2026-09-12 | **Status:** complete. 11 models added, 2 old rows re-run at max_tokens 16000, 23 rows on the board
**Board:** [current leaderboard](/) | [previous 12-model board (archived)](/archive/run-2026-05-v0-refresh/) | [run registry](https://github.com/cortico-health/MedSafe-Dx/blob/main/docs/RUNS.md)

## Summary

1. **GPT-5.6 Terra is the new leader on Triage Success Rate (73.2%)**, one point ahead of the previous leader GPT-5 Chat (72.4%). GPT-5.4 Mini ties for second at 72.4%.
2. **Newer frontier models mostly did not beat their predecessors.** Grok 4.6 (-3.2 pts), GPT-5.6 Sol (-2.0) and Claude Opus 5 (-0.4) all rank below the model they replace, because each over-escalates more routine cases. Only GPT-5.4 Mini improved (+2.8 over the re-run GPT-5 Mini).
3. **HealthBench rank does not predict MedSafe-Dx rank.** The three HealthBench Pro leaders we ran, GPT-6 Astra (#17, 64.4%), Claude Fable 5 (#18, 62.8%) and Claude Opus 5 (#21, 62.0%), hold the best top-3 diagnostic recall on the board (79.5%, 88.8%, 85.4%) and among the highest over-escalation. They are the best diagnosticians and the most cautious triagers, and this benchmark's primary metric penalises the caution.
4. **The v0 harness understated reasoning models, and the re-runs confirm it.** Every 2026-01/05 format failure was an empty or mid-sentence response. Re-running at max_tokens 16000 took GPT-5 Mini from 29 format failures to 5 (TSR 68.0% -> 69.6%) and DeepSeek R1 from 3 to 1 (61.6% -> 62.8%). Gemini 3 Pro Preview's 65 failures were 41 empty responses and 23 truncations under 365 characters, the same signature, but the model has been withdrawn from OpenRouter so its row cannot be corrected.
5. **Two evaluator/harness defects were found and fixed during the run**: the evaluator rejected whole predictions when a model listed more than 5 ranked diagnoses even though the prompt never states a count (it now truncates to the top 5), and API failures were silently scored as safety failures with no retry, no checkpoint, and no diagnostic log.

## What was run

| Field | Value |
|---|---|
| Eval set | `eval-250-v0.json`, N=250, identical case IDs and order to the published board |
| Prompt | v4, `intake` workflow, temperature 0.0 |
| max_tokens | 16000 (v0 rows: 2000) |
| Backend | OpenRouter API for all rows except GPT-6 Astra (local Codex CLI on the ChatGPT plan, after OpenRouter rejected the run on credits) |
| Added | Claude Opus 5, Claude Fable 5, GPT-5.6 Sol / Terra / Luna, GPT-5.4 Mini, GPT-6 Astra, Kimi K3, GLM 5.3, Grok 4.6; Gemini 3.1 Pro Preview derived from the N=500 run |
| Re-run at 16000 | GPT-5 Mini, DeepSeek R1 (original rows kept on the archived board) |
| Dropped | Qwen3.8 Max, Muse Spark 1.3 (blocked by the account's zero-data-retention policy) |
| Roster rule for OpenAI | latest 5-series model of each class (Astra, Sol, Terra, Luna) plus the newest "mini" |

## Results

Primary metric: **Triage Success Rate (TSR)** = Safety Pass Rate - over-escalations / all cases. Higher is better.

![Triage Success Rate by model, coloured by run era](/figures/tsr-by-era.png)

| # | Model | Era | TSR | SPR | Over-esc | Top-3 recall | Format fails | Missed esc | max_tokens | Backend |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | openai-gpt-5.6-terra | 2026-09 | **73.2%** | 94.4% | 21.2% | 81.4% | 2 | 12 | 16000 | openrouter |
| 2 | openai-gpt-5.4-mini | 2026-09 | **72.4%** | 96.4% | 24.0% | 80.5% | 0 | 7 | 16000 | openrouter |
| 3 | openai-gpt-5-chat | 2026-01 | **72.4%** | 94.0% | 21.6% | 79.6% | 0 | 8 | 2000 | openrouter |
| 4 | meta-llama-llama-4-maverick | 2026-05 | **71.2%** | 96.8% | 25.6% | 66.5% | 2 | 6 | 2000 | openrouter |
| 5 | x-ai-grok-4.20 | 2026-05 | **71.2%** | 89.6% | 18.4% | 78.1% | 0 | 26 | 2000 | openrouter |
| 6 | openai-o3-pro | 2026-05 | **70.8%** | 92.8% | 22.0% | 79.3% | 0 | 13 | 2000 | openrouter |
| 7 | openai-gpt-5.2 | 2026-01 | **70.8%** | 97.6% | 26.8% | 71.3% | 0 | 5 | 2000 | openrouter |
| 8 | anthropic-claude-haiku-4.5 | 2026-01 | **70.8%** | 95.6% | 24.8% | 69.9% | 0 | 11 | 2000 | openrouter |
| 9 | z-ai-glm-5.3 | 2026-09 | **69.6%** | 93.2% | 23.6% | 82.4% | 1 | 9 | 16000 | openrouter |
| 10 | openai-gpt-5.6-luna | 2026-09 | **69.6%** | 92.0% | 22.4% | 73.5% | 0 | 18 | 16000 | openrouter |
| 11 | anthropic-claude-sonnet-4.6 | 2026-05 | **69.6%** | 94.8% | 25.2% | 80.2% | 0 | 11 | 2000 | openrouter |
| 12 | openai-gpt-5-mini | 2026-09-rerun | **69.6%** | 88.8% | 19.2% | 77.5% | 5 | 18 | 16000 | openrouter |
| 13 | openai-gpt-5.6-sol | 2026-09 | **68.8%** | 91.6% | 22.8% | 76.4% | 0 | 15 | 16000 | openrouter |
| 14 | moonshotai-kimi-k3 | 2026-09 | **68.4%** | 88.8% | 20.4% | 80.6% | 0 | 18 | 16000 | openrouter |
| 15 | x-ai-grok-4.6 | 2026-09 | **68.0%** | 90.8% | 22.8% | 83.3% | 0 | 21 | 16000 | openrouter |
| 16 | openai-gpt-oss-120b | 2026-01 | **66.8%** | 85.2% | 18.4% | 78.9% | 1 | 17 | 2000 | openrouter |
| 17 | openai-gpt-6-astra | 2026-09 | **64.4%** | 89.6% | 25.2% | 79.5% | 0 | 8 | 16000 | codex |
| 18 | anthropic-claude-fable-5 | 2026-09 | **62.8%** | 86.0% | 23.2% | 88.8% | 0 | 9 | 16000 | openrouter |
| 19 | deepseek-deepseek-r1 | 2026-09-rerun | **62.8%** | 90.4% | 27.6% | 78.8% | 1 | 5 | 16000 | openrouter |
| 20 | anthropic-claude-opus-4.7 | 2026-05 | **62.4%** | 86.4% | 24.0% | 85.2% | 0 | 5 | 2000 | openrouter |
| 21 | anthropic-claude-opus-5 | 2026-09 | **62.0%** | 87.6% | 25.6% | 85.4% | 0 | 3 | 16000 | openrouter |
| 22 | google-gemini-3.1-pro-preview (derived) | 2026-03 | **57.6%** | 78.4% | 20.8% | 86.7% | 3 | 21 | 2000 | openrouter |
| 23 | google-gemini-3-pro-preview | 2026-01 | **47.2%** | 62.4% | 15.2% | 87.2% | 65 | 9 | 2000 | openrouter |

Rows marked "(derived)" were evaluated on the 250-case subset of an existing N=500 run with the same prompt at the old 2000 cap. Rows marked `codex` were produced through the Codex CLI, which wraps the prompt in its own agent system prompt and does not expose temperature. Treat them as indicative.

### Successor vs predecessor

![Generation deltas](/figures/generation-deltas.png)

| Successor | Predecessor | TSR old -> new | SPR old -> new | Over-esc old -> new |
|---|---|---|---|---|
| GPT-5.4 Mini | GPT-5 Mini (re-run at 16000) | 69.6% -> 72.4% (+2.8) | 88.8% -> 96.4% | 19.2% -> 24.0% |
| Claude Opus 5 | Claude Opus 4.7 | 62.4% -> 62.0% (-0.4) | 86.4% -> 87.6% | 24.0% -> 25.6% |
| GPT-5.6 Sol | GPT-5.2 | 70.8% -> 68.8% (-2.0) | 97.6% -> 91.6% | 26.8% -> 22.8% |
| Grok 4.6 | Grok 4.20 | 71.2% -> 68.0% (-3.2) | 89.6% -> 90.8% | 18.4% -> 22.8% |

The GPT-5 Mini comparison uses its 16000 re-run so both sides share the same token cap; against the original 2000-cap row the gain would read +4.4.

### Re-runs at max_tokens 16000 (same model, same cases)

| Model | Format fails 2000 -> 16000 | SPR | Over-esc | TSR |
|---|---|---|---|---|
| GPT-5 Mini | 29 -> 5 | 84.8% -> 88.8% | 16.8% -> 19.2% | 68.0% -> 69.6% |
| DeepSeek R1 | 3 -> 1 | 90.4% -> 90.4% | 28.8% -> 27.6% | 61.6% -> 62.8% |

GPT-5 Mini's 5 remaining failures are still truncations: each response stops after the follow-up text at 470-515 characters with no closing brace, so even 16000 tokens is occasionally consumed by reasoning. Recording finish reason and token usage per prediction is the next harness step.

### Safety vs over-escalation

![Safety Pass Rate vs over-escalation, with iso-TSR contours](/figures/tradeoff-scatter.png)

The 2026-09 models cluster in a narrow band: Safety Pass Rate 86-96%, over-escalation 19-26% of non-urgent cases. No new model moved toward the ideal corner; the gains that exist come from fewer missed escalations, not from escalating less.

## Harness and evaluator findings

### 1. Diagnosis-count rule was implicit, and it cost Claude Opus 5 seven cases

The v4 prompt shows a 5-row schema example but never says "exactly 5" (only the retired v1 system prompt did). Opus 5 returned 6 or 7 ranked diagnoses on 7 broad presentations (hemoptysis, hematemesis, zoonotic fever). The evaluator rejected each whole prediction as a format failure, which counts as a safety failure, even though all 7 escalations were correct. No other model in the repo's history ever exceeded 5.

Fix: `evaluator/schemas.py` truncates ranked differentials to the top 5 and still rejects fewer than 5. Opus 5 moved from SPR 84.8% / TSR 60.8% to 87.6% / 62.0%. No published row changed. The pre-fix eval is kept as `anthropic-claude-opus-5-250cases-eval.exact5.json`.

### 2. OpenRouter credit exhaustion looked like model failure

Six parallel runs at max_tokens 16000 tripped OpenRouter's in-flight budget check (HTTP 402) when the account balance fell to about $27. GPT-6 Astra lost 39/250 cases, GLM 5.3 lost 4. The old client scored each as a format failure with no retry and only wrote predictions at the end of 250 cases, so a crash lost the whole pass.

Fixes: retries with backoff on 429/5xx/timeouts; checkpoint every 10 cases; resume re-runs errored cases and preserves test-set order; the refresh script only skips a model when every case has a non-error prediction. GLM was recovered by resume (2 of its original 12 failures remain, both genuine: one placeholder value, one persistent empty response).

### 3. Empty responses were invisible

GLM 5.3 returned empty content on 7 cases with no error line anywhere. The client now logs finish reason and token usage for empty content and retries once. On the second attempt 10 of 11 GLM cases succeeded, so the provider behaviour is transient.

### 4. The v0 max_tokens=2000 cap produced the old board's format failures

| Model (v0 row) | Format failures | Of which empty | Of which truncated mid-JSON |
|---|---|---|---|
| Gemini 3 Pro Preview | 65 | 41 | 23 |
| GPT-5 Mini | 29 | 29 | 0 |
| DeepSeek R1 | 3 | 3 | 0 |

Truncated responses are 4 to 365 characters and stop mid-sentence, consistent with reasoning tokens consuming the cap before the JSON finished. Gemini 3 Pro Preview has since been removed from OpenRouter, so it cannot be re-run; GPT-5 Mini and DeepSeek R1 were re-run at 16000 (results above). A Gemini 3.1 Pro Preview row was derived from the existing N=500 run (same prompt, cap 2000: 3 format failures, TSR 57.6%) as the nearest available Google datapoint.

### 5. Codex CLI as a backend

`inference/codex_cli.py` adds `--backend codex`, which runs `codex exec` sandboxed read-only in an empty directory with an ephemeral session. It is flat-rate on the ChatGPT plan and took 16 s per case for GPT-6 Astra. Output metadata now records `backend`, `reasoning_effort` and `git_commit`. Caveats: Codex prepends its own agent system prompt, temperature is not controllable, reasoning effort was medium. Astra returned 250/250 clean JSON.

## Follow-ups

| Item | Why | Cost |
|---|---|---|
| Gemini 3.1 Pro Preview at 16000 | replace the derived cap-2000 row with a like-for-like run | ~$5 |
| Record finish_reason and token usage per prediction | make truncation diagnosable without forensics | harness only |
| Structured-output mode as an option | test whether JSON-schema enforcement changes safety scores | harness only |

Harness additions this run: `--workers N` (parallel cases per model; 8 workers cut a reasoning-model pass from hours to minutes), retries, checkpoint/resume, Codex CLI backend, provenance fields in output metadata.

This page is rendered from `docs/FINDINGS-2026-09.md`. The paper has not been updated.
