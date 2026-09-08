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
| Prompt | v4, `intake` workflow, temperature 0.0 |
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
| Status | designed, not yet executed |

Planned slate and rationale: see artifact report (2026-09-08) and the script header.
