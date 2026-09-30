# v0.3 full run: roster expansion, smoke test and cost projection

Date: 2026-09-29. Decisions 26-29: we add the current model roster to the v0.3 full run (results/v03_full/, 900 cases, seed 20261005), arm 4aj only, OpenAI models through the API. This file covers the smoke test and the cost projection. The full run has not started.

## What we ran

- Prompts and settings: the full run's (`inference/run_config_v03_abj.json`: prompt v7a4aj, decoder v02, temperature 0, max_tokens 16000, reasoning effort "medium", one retry on an empty or truncated response). We added each new model to a copy of that config with reasoning effort "medium", because the full run gives every reasoning-capable model "medium". The copy is outside the repo; add the same entries to the config before the full run.
- Cases: the first 3 cases of `data/test_sets/eval-v03-full.json` (ddxplus_84317, ddxplus_33912, ddxplus_113916), the same 3 the full run's smoke used (`LIMIT=3`), for every model.
- Runner: `python3 -m inference.run_inference --prompt-version v7a4aj --limit 3`, as `scripts/run_v03_ab.sh` calls it.
- Prices: OpenRouter's `GET /api/v1/models`, read 2026-09-29, in USD per million tokens.

## Roster

"ZDR list" is whether OpenRouter's zero-data-retention endpoint list (`GET /api/v1/endpoints/zdr`) has an endpoint for the model. "Account policy" is whether the request went through under the key's data policy, which is the mechanism the harness relies on: it sends no per-request provider setting, and the account policy is what blocked Qwen3.8 Max and Muse Spark 1.3 in the 2026-09 refresh (docs/RUNS.md). Tokens per case are the smoke mean; completion includes reasoning. The projection is 900 x the mean smoke cost per case; the range is 900 x the cheapest and dearest smoke case.

| # | Model | OpenRouter id | Price in / out | ZDR list | Account policy | Smoke parse | Tokens per case: prompt / completion (reasoning) | Projected 900 x 4aj (range) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Grok 4.7 | x-ai/grok-4.7 | 2 / 6 | yes (xAI) | ok | 3/3 ok | 1742 / 3704 (3578) | $21.58 (19.82-23.44) | Newest Grok (4.6 also listed). xAI adds about 1,200 prompt tokens of its own. Longest reasoning: max 4,050 completion tokens, a quarter of the 16,000 cap. |
| 2 | Claude Fable 5.1 | anthropic/claude-fable-5.1 | 10 / 50 | **no** | ok | 3/3 ok | 814 / 292 (57) | $20.47 (18.03-25.24) | Served by Anthropic (2) and Google (1); neither is on the ZDR list. See decision 1. |
| 3 | GPT-6 Astra | openai/gpt-6-astra | 10 / 50 | yes (Azure only) | ok | 3/3 ok | 514 / 322 (212) | $19.13 (18.14-19.94) | Served by OpenAI, not the listed Azure endpoint. Short differentials (2 diagnoses per case). |
| 4 | Claude Opus 5.5 | anthropic/claude-opus-5.5 | 4 / 20 | yes | ok | 3/3 ok | 814 / 464 (234) | $11.28 (10.17-12.05) | |
| 5 | GPT-5.4 mini | openai/gpt-5.4-mini | 0.75 / 4.5 | yes (Azure only) | ok | 3/3 ok | 514 / 1597 (1413) | $6.82 (3.11-9.94) | Newest GPT mini. Reasoning length varies 3x across cases, hence the wide range. |
| 6 | GPT-6.1 Sol | openai/gpt-6.1-sol | 2 / 10 | yes (Azure only) | ok | 3/3 ok | 514 / 353 (231) | $4.11 (3.93-4.42) | Newest Sol (listed 2026-09-29). Alternative: GPT-5.6 Sol, row A2. |
| 7 | Kimi K3 | moonshotai/kimi-k3 | 3 / 15 | yes | ok | 3/3 ok | 603 / 217 (37) | $3.76 (2.20-4.55) | Newest Kimi. Relace charged about half of Modal's price for the same tokens. |
| 8 | Claude Sonnet 5.5 | anthropic/claude-sonnet-5.5 | 2 / 10 | yes | ok | 3/3 ok | 814 / 241 (0) | $3.63 (3.48-3.74) | Newest Sonnet (listed 2026-09-28). Used no reasoning tokens at "medium". Alternative: Sonnet 5, row A1. |
| 9 | Gemini 3.8 Flash | google/gemini-3.8-flash | 0.75 / 3.75 | yes | ok | 3/3 ok | 531 / 792 (634) | $3.03 (2.89-3.24) | No Gemini Pro newer than 3.1 Pro is listed; 3.8 Flash is the newest Gemini. See decision 3. |
| 10 | DeepSeek V4.1 Flash | deepseek/deepseek-v4.1-flash | 0.3 / 1.2 | yes | ok | 3/3 ok | 531 / 2279 (2124) | $1.62 (0.94-2.68) | Newest DeepSeek (2026-09-10). The first smoke attempt hung for over 10 minutes with no reply; a rerun finished in seconds. Alternative: V4 Pro 0813, row A4. |
| 11 | GPT-6 Luna | openai/gpt-6-luna | 0.1 / 0.5 | yes (Azure only) | ok | 3/3 ok | 514 / 1040 (898) | $0.51 (0.39-0.58) | Newest Luna. Alternative: GPT-5.6 Luna, row A3. |

Every smoke response finished with `stop` on the first attempt, with no retry, and parsed to a flag and a differential. No response came near the 16,000 max_tokens cap: the highest was 4,050 completion tokens (Grok 4.7).

### Alternatives the brief named (smoked, not in the totals)

We applied the roster rule "latest model of each class" to every family, so the brief's GPT-5.6 Sol, GPT-5.6 Luna and Sonnet 5 give way to newer models released since. Swapping any of them back changes the total by under $2.

| # | Model | OpenRouter id | Price in / out | ZDR list | Smoke parse | Tokens per case: prompt / completion (reasoning) | Projected 900 x 4aj (range) |
|---|---|---|---|---|---|---|---|
| A1 | Claude Sonnet 5 | anthropic/claude-sonnet-5 | 2 / 10 | yes | 3/3 ok | 812 / 246 (36) | $3.68 (3.17-4.05) |
| A2 | GPT-5.6 Sol | openai/gpt-5.6-sol | 2 / 10 | yes (Azure only) | 3/3 ok | 514 / 523 (357) | $5.63 (4.65-6.17) |
| A3 | GPT-5.6 Luna | openai/gpt-5.6-luna | 0.2 / 1.2 | yes (Azure only) | 3/3 ok | 514 / 638 (487) | $0.78 (0.70-0.82) |
| A4 | DeepSeek V4 Pro 0813 | deepseek/deepseek-v4-pro-0813 | 0.4 / 3.49 | yes | 3/3 ok | 531 / 2908 (2740) | $7.36 (6.65-8.44) |

## Totals

| Roster | Projected 900 x 4aj | Range |
|---|---|---|
| All 11 (rows 1-11) | **$95.94** | $83.09-109.79 |
| Trimmed: without Grok 4.7, Fable 5.1 and GPT-6 Astra | **$34.75** | $27.10-41.19 |

How far to trust a 3-case projection: for the seven models already in the full run, 900 x their 3-case smoke cost landed within 0.82-1.40 x the actual 900-case cost (Sonnet 4.6 0.82, gpt-oss-120b 1.40, the other five 0.93-1.23).

Smoke spend: $0.38 (the sum of per-request costs in the smoke files; the key's usage rose by $0.37 over the same period).

## Blocker and decisions

The OpenRouter key has $16.49 of headroom, below both totals, so the full run needs more budget before it starts. OpenRouter also holds max_tokens x price against the limit for each in-flight request (16,000 x $50 per million = $0.80 per Fable or Astra request), so the run needs headroom above the projection.

1. **Fable 5.1 and the ZDR rule.** Default: keep Fable, because the harness's own mechanism (the account data policy) accepted it, as it accepted GPT-5.6 Terra through OpenAI in the full run. If the rule means OpenRouter's strict ZDR list, we would add `provider: {zdr: true}` to each request: Fable drops (no ZDR endpoint), the OpenAI models route to Azure, and the existing seven models would then have run under a different policy from the new ones.
2. **Budget.** Default: the trimmed roster ($35) as the first batch, because it needs about $20 of new headroom rather than $80; the three flagships ($61) follow as a second batch.
3. **Gemini.** Default: include Gemini 3.8 Flash ($3), because it is the newest Gemini and no Pro successor to 3.1 Pro exists.
