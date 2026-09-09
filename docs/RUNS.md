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

## run-2026-09-healthbench-refresh (planned)

Adds models that rose to the top of HealthBench (OpenAI's rubric-graded clinical
benchmark) after the v0 roster froze, because MedSafe-Dx should track whether
HealthBench gains translate into hard safety-gate performance.

| Field | Value |
|---|---|
| Eval set | `data/test_sets/eval-250-v0.json` (same frozen set, for comparability) |
| Script | `scripts/run_healthbench_refresh_250.sh` |
| Version label | `2026-09` |
| Status | smoke pass done (N=10), 250-case run pending decisions |

Planned slate (7 models, ~$10-41): Claude Opus 5, GPT-5.6 Sol, Kimi K3 (MAST #1;
supersedes K2 Thinking), Qwen3.8 Max (dated slug `qwen/qwen3.8-max-0902`), Muse Spark 1.3,
GLM 5.3 (no published HealthBench score - novel datapoint), Grok 4.6 (refreshes
grok-4.20). `--full` adds Claude Fable 5 + GPT-6 Astra (~$22-97 total). Grok 4.7 lands
mid-Sept 2026; run it as a follow-up rather than blocking this run.

Clinical-deployed tier: dropped 2026-09-08. Baichuan-M3, MedGemma, and Meditron are
not listed on OpenRouter; OpenEvidence, Hippocratic Polaris, and UpToDate Expert AI
are closed systems. A clinical tier needs a non-OpenRouter runner.

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
