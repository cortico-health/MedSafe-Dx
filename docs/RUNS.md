# MedSafe-Dx Run Registry

Each leaderboard refresh gets a run ID, a frozen roster, and a git tag. This file is the
audit trail because the published leaderboard only shows the latest state.

## Conventions

- **Run ID:** `run-YYYY-MM-<descriptor>`
- **Eval set:** frozen JSON in `data/test_sets/` (gitignored; regenerate with
  `scripts/create_standard_test_sets.sh`). Integrity is pinned by `cases_sha256`
  inside every eval artifact.
- **Tag:** `leaderboard-<run-id>` on the commit containing that run's `leaderboard/*.json`.
- **Versioning rule:** never overwrite a prior run's artifacts. New models get new files;
  superseded rosters stay reachable via their git tag.

---

## run-2026-05-v0-refresh (tag: `leaderboard-run-2026-05-v0-refresh`)

The current published leaderboard (https://msdx.cortico.health/). 12 models on the
frozen primary eval set.

| Field | Value |
|---|---|
| Eval set | `data/test_sets/eval-250-v0.json` (N=250, seed=42) |
| Prompt | v4, `intake` workflow, temperature 0.0, **max_tokens 16000** (v0 used 2000; see smoke finding 4) |
| Roster window | model versions 2026-01 to 2026-05 |
| Artifacts | `results/artifacts/*-250cases{,-eval}.json`, published copies in `leaderboard/` |
| Paper | medRxiv 2026.04.14.26350711 (v2/v3 text matches this roster) |

Roster (model, version label in artifact):

1. openai-gpt-5.2 (2026-01)
2. openai-gpt-5-chat (2026-01)
3. openai-gpt-5-mini (2026-01)
4. openai-gpt-oss-120b (2026-01)
5. openai-o3-pro (2026-05)
6. anthropic-claude-opus-4.7 (2026-05)
7. anthropic-claude-sonnet-4.6 (2026-05)
8. anthropic-claude-haiku-4.5 (2026-01)
9. google-gemini-3-pro-preview (2026-01)
10. deepseek-deepseek-r1 (2026-05)
11. meta-llama-llama-4-maverick (2026-05)
12. x-ai-grok-4.20 (2026-05)

Historical context: earlier 100-case and 500-case runs exist in `results/artifacts/`
and in git history; the 500-case draft analysis is shelved under `shelved/`. The
N=250 set above is the only published, preprint-aligned run.

---

## run-2026-09-healthbench-refresh (tag: `leaderboard-run-2026-09-healthbench-refresh`)

Adds models that rose to the top of HealthBench (OpenAI's rubric-graded clinical
benchmark) after the v0 roster froze, because MedSafe-Dx should track whether
HealthBench gains translate into hard safety-gate performance.

| Field | Value |
|---|---|
| Eval set | `data/test_sets/eval-250-v0.json` (same frozen set, for comparability) |
| Script | `scripts/run_healthbench_refresh_250.sh` |
| Version label | `2026-09` |
| Status | **run on 2026-09-12**, 10 models added + 2 re-runs + 1 derived row (12 -> 23 rows); see "Outcome" below |

Planned slate (7 models, ~$10-41): Claude Opus 5, GPT-5.6 Sol, Kimi K3 (MAST #1;
supersedes K2 Thinking), Qwen3.8 Max (dated slug `qwen/qwen3.8-max-0902`), Muse Spark 1.3,
GLM 5.3 (no published HealthBench score - novel datapoint), Grok 4.6 (refreshes
grok-4.20). `--full` adds Claude Fable 5 + GPT-6 Astra (~$22-97 total). Grok 4.7 lands
mid-Sept 2026; run it as a follow-up rather than blocking this run.

Clinical-deployed tier: dropped 2026-09-08. Baichuan-M3, MedGemma, and Meditron are
not listed on OpenRouter; OpenEvidence, Hippocratic Polaris, and UpToDate Expert AI
are closed systems. A clinical tier needs a non-OpenRouter runner.

### Outcome (2026-09-12)

Roster actually run (all on `eval-250-v0.json`, prompt v4, intake, temp 0.0, max_tokens 16000):

| Model | Backend | Notes |
|---|---|---|
| anthropic-claude-opus-5 | OpenRouter | 7 cases returned 6-7 diagnoses; scored after the evaluator change below |
| openai-gpt-5.6-sol | OpenRouter | clean |
| openai-gpt-5.6-terra | OpenRouter | added 2026-09-12 (latest 5-series per class); 2 malformed-JSON cases |
| openai-gpt-5.6-luna | OpenRouter | added 2026-09-12; clean |
| openai-gpt-5.4-mini | OpenRouter | added 2026-09-12 (latest "mini"); clean |
| openai-gpt-6-astra | **Codex CLI** | OpenRouter attempt hit HTTP 402 (39/250); re-run in full via `--backend codex` (ChatGPT plan). Codex adds its own agent system prompt; no temperature control; reasoning effort medium. Not strictly comparable to API rows. The partial OpenRouter artifact is kept as `*-openrouter-partial*.json`. |
| moonshotai-kimi-k3 | OpenRouter | 13 responses fenced in markdown; cleaner handles it |
| z-ai-glm-5.3 | OpenRouter | 4 cases lost to 402 + 7 empty-content responses; recovered by resume (see harness changes) |
| x-ai-grok-4.6 | OpenRouter | clean |
| anthropic-claude-fable-5 | OpenRouter | clean; best top-3 recall on the board (88.8%) |
| google-gemini-3.1-pro-preview | derived | 250-case subset of the 2026-03 N=500 run (same prompt, cap 2000), version label 2026-03 |
| openai-gpt-5-mini (re-run) | OpenRouter | `*-250cases-mt16000*`, version `2026-09-rerun`; fmt fails 29 -> 5. Old row moved to `leaderboard/archived/*.mt2000.json` |
| deepseek-deepseek-r1 (re-run) | OpenRouter | same scheme; fmt fails 3 -> 1 |

Dropped: Qwen3.8 Max and Muse Spark 1.3 (ZDR policy, not relaxed). Gemini 3 Pro Preview could not be re-run (withdrawn from OpenRouter).

Decisions and harness changes made during the run:

1. **Evaluator: top-5 truncation.** The v4 prompt never states "exactly 5" (only the v1
   system prompt did); the count is implied by the schema example. Opus 5 returned 6-7
   ranked diagnoses on 7 broad presentations and the evaluator rejected the whole
   prediction as a safety failure. `evaluator/schemas.py` now truncates to the top 5 and
   still rejects fewer than 5. No prior run ever exceeded 5, so published rows are
   unaffected. Opus 5 as scored before/after: SPR 84.8% -> 87.6%, TSR 60.8% -> 62.0%
   (`anthropic-claude-opus-5-250cases-eval.exact5.json` keeps the original).
2. **OpenRouter credit exhaustion mid-run (HTTP 402 "in-flight budget").** Six parallel
   runs at max_tokens 16000 reserve more than a low balance. Astra lost 39 cases, GLM 4.
   Fixes: retries with backoff on 429/5xx/timeouts (`inference/openrouter.py`),
   checkpoint every 10 cases + resume that re-runs errored entries
   (`inference/run_inference.py`), and the refresh script only skips inference when every
   case has a non-error prediction. Runs started before these fixes are not resumable.
3. **Empty-content responses (GLM 5.3, 7 cases)** were scored as `api_failure` with no
   log line. The client now logs finish_reason + usage and retries once.
4. **Codex CLI backend** (`inference/codex_cli.py`, `--backend codex`) for models on the
   ChatGPT plan. Output metadata now records `backend`, `reasoning_effort`, `git_commit`.
5. **Old-era caveat is broader than DeepSeek R1.** Every 2026-01/05 format failure is
   "differential_diagnoses missing" (no content at all): Gemini 3 Pro 65/250, GPT-5 Mini
   29/250, R1 3/250. That matches max_tokens 2000 starvation, so Gemini's last place is
   probably a harness artifact. Re-run those three at 16000 as a follow-up.

6. **Parallel inference.** `--workers N` / `INFERENCE_WORKERS` runs N cases concurrently per
   model (thread pool, chunked so checkpoints and ordering are unchanged). Sequential
   reasoning-model passes at 16000 were 2-5 h; 8 workers finish in 15-40 min.
7. **Re-run naming.** Re-runs of published models use `LABEL_OVERRIDE` (artifact suffix,
   e.g. `250cases-mt16000`) and `VERSION_OVERRIDE` (e.g. `2026-09-rerun`) so the original
   artifacts are never overwritten; the old leaderboard row moves to `leaderboard/archived/`.

Findings write-up: `docs/FINDINGS-2026-09.md`, rendered at `/findings-2026-09.html`.

Tooling: `scripts/compare_runs.py OLD NEW` prints roster deltas, TSR ranking with old
ranks, and successor-vs-predecessor deltas for release notes. The superseded 12-model
board is archived at `/archive/run-2026-05-v0-refresh/` on the site.

### Smoke pass findings (N=10, dev-v0, 2026-09-08)

Purpose was flushing harness issues, not scoring. Findings:

1. **Harness bug (fixed):** `data/convert_csv_to_json.py` emits `id`, but inference
   expects `case_id` - cases.json must go through `data/cases.py` (the two-stage
   pipeline). Fresh checkouts hit a `KeyError: 'case_id'` wall without it.
2. **Harness improvement (committed):** `inference/openrouter.py` now logs the error
   response body; OpenRouter puts the real reason (invalid model ID, ZDR blocks)
   only there.
3. **ZDR policy blocks 2 models:** Qwen3.8 Max and Muse Spark 1.3 route only via
   first-party providers (Alibaba, Meta) that the account's zero-data-retention
   policy rejects. Decision needed: relax ZDR for this run (DDXPlus is synthetic,
   no PHI) or drop both.
4. **Reasoning models + `max_tokens=2000` starve content (fixed):** GLM 5.3
   failed 4-6/10 and Kimi K3 1/10 because chain-of-thought tokens count against
   the cap; when reasoning runs long the JSON content is truncated (GLM to a
   literal `"{\n"` stub). Validated fix: raise the cap to 16000 - GLM and Kimi
   then go 10/10 on dev-v0 (worst GLM case used ~7,000 reasoning tokens).
   The v0 leaderboard ran with cap 2000; deepseek-r1's 3/250 format failures there
   are likely the same starvation, so cross-era comparisons of reasoning models
   should note this harness change (v0 cap vs v0.1 cap 16000).
5. **Frozen-set recovery:** the original `eval-250-v0.json` was gitignored and lost.
   Forensics show it equals the first 250 (sorted by case_id) of `eval-v0.json`
   (N=500, seed=42), which reproduces exactly. The 250-set was rebuilt that way and
   case-level equivalence is verified (same 250 IDs as published predictions; symptom
   counts match prediction audits). Byte-level sha256 still differs from the published
   `cases_sha256`, so the original file had cosmetic differences. Note: seed-42
   resampling at N=250 does NOT reproduce the frozen set - the derivation rule above
   is the reproducible path, now documented here.
