# v0.3 full run: run settings per model (arm 4aj)

Built by `scripts/analysis/v03_run_settings.py` from `inference/run_config_v03_abj.json`, the provenance files and the
arm-4aj prediction files under `results/v03_full/runs/` (git-ignored). Tokens are means per case as OpenRouter
reported them; "Served by" counts requests per provider.

- **Shared settings:** prompt v7a4aj, decoder v02, temperature 0.0, max_tokens 16000. The OpenRouter client makes up to four attempts on an empty response; the runner then retries once on an empty or truncated one (empty_or_truncated_retries 1); an answer that failed to parse was asked again once.
- **v0.3 full run** (`provenance.json`): launched 2026-09-27T19:24:21Z from commit 49c0aa5 with git_dirty true (uncommitted changes in the working tree); run config sha256 3b3c05b1a1df.
- **v0.3 full run, roster expansion** (`provenance-expansion.json`): launched 2026-09-30T00:23:54Z from commit c47a81c with git_dirty true (uncommitted changes in the working tree); run config sha256 effc223a8a0e.

| Model | Reasoning effort sent | Reasoning tokens per case | Completion tokens per case | Cases answered | Served by (requests) |
|---|---|---|---|---|---|
| anthropic/claude-opus-5.5 | medium | 283 | 507 | 900 | Claude Platform on AWS 898; Amazon Bedrock 2 |
| anthropic/claude-fable-5.1 | medium | 62 | 291 | 900 | Anthropic 887; Google 13 |
| openai/gpt-6.1-sol | medium | 221 | 366 | 900 | OpenAI 900 |
| openai/gpt-6-astra | medium | 229 | 372 | 900 | OpenAI 900 |
| google/gemini-3.1-pro-preview | medium | 615 | 777 | 900 | Google 900 |
| openai/gpt-6-luna | medium | 753 | 898 | 900 | OpenAI 900 |
| openai/gpt-5.6-terra | medium | 350 | 470 | 900 | OpenAI 900 |
| google/gemini-3.8-flash | medium | 665 | 821 | 900 | Google 890; Google AI Studio 10 |
| moonshotai/kimi-k3 | medium | 64 | 237 | 900 | Together 376; InferenceNet 362; Modal 97; Wafer 13; 12 more providers (52 requests) |
| openai/gpt-5.4-mini | medium | 1400 | 1565 | 900 | OpenAI 900 |
| anthropic/claude-sonnet-5.5 | medium | 0 | 224 | 900 | Claude Platform on AWS 900 |
| anthropic/claude-sonnet-4.6 | medium | 957 | 1159 | 900 | Claude Platform on AWS 900 |
| z-ai/glm-5.3 | medium | 44 | 201 | 900 | InferenceNet 298; Wafer 206; Makora 191; PrimeIntellect 68; 20 more providers (137 requests) |
| openai/gpt-oss-120b | medium | 549 | 768 | 900 | CoreWeave 191; AkashML 158; DeepInfra 143; DekaLLM 133; 12 more providers (275 requests) |
| anthropic/claude-haiku-4.5 | none (no reasoning parameter sent) | 0 | 183 | 900 | Amazon Bedrock 900 |
| meta-llama/llama-3.1-8b-instruct | none (no reasoning parameter sent) | 0 | 147 | 900 | DeepInfra 789; Groq 111 |
| deepseek/deepseek-v4.1-flash (unfinished, not scored) | medium | 1877 | 2023 | 819 | Together 398; Wafer 219; CoreWeave 88; DeepInfra 35; 13 more providers (80 requests) |
| x-ai/grok-4.7 (unfinished, not scored) | medium | 5198 | 5329 | 800 | xAI 800 |
