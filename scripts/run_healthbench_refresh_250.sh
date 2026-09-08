#!/bin/bash
# run-2026-09-healthbench-refresh
#
# Adds HealthBench-leading models to MedSafe-Dx on the frozen eval-250-v0 set,
# because the v0 leaderboard predates the current HealthBench frontier
# (Claude Fable 5, GPT-6 Astra, GPT-5.6 Sol, Claude Opus 5, Muse Spark,
# Qwen3.8 Max, Kimi K2 Thinking).
#
# Same frozen test set + prompt v4 + temp 0.0 as run-2026-05-v0-refresh,
# so scores are directly comparable with the published leaderboard.
#
# Cost: see docs/RUNS.md and the 2026-09-08 planning artifact. Roughly
# $7-28 for the default 5-model slate, $19-84 with the two flagships.
#
# Usage:
#   ./scripts/run_healthbench_refresh_250.sh           # default 5-model slate
#   ./scripts/run_healthbench_refresh_250.sh --full    # adds the 2 flagship models
#   SMOKE=1 ./scripts/run_healthbench_refresh_250.sh   # 10-case dev-v0 dry run first

set -e
cd "$(dirname "$0")/.."

if [ "${SMOKE:-0}" = "1" ]; then
    TEST_SET="data/test_sets/dev-v0.json"
    LABEL="10cases-smoke"
else
    TEST_SET="data/test_sets/eval-250-v0.json"
    LABEL="250cases"
fi

# Default slate: strong HealthBench performers at moderate price.
MODELS=(
    "anthropic/claude-opus-5"       # HealthBench Pro ~59.8
    "openai/gpt-5.6-sol"            # HealthBench Pro ~60.5
    "qwen/qwen3.8-max"              # top open-weights, HealthBench ~0.602
    "moonshotai/kimi-k2-thinking"   # HealthBench ~0.580, very cheap
    "meta/muse-spark-1.3"           # HealthBench Pro ~59.3 (1.1); OR carries 1.3
)

# --full adds the two $10/$50 flagships.
if [ "${1:-}" = "--full" ]; then
    MODELS+=(
        "anthropic/claude-fable-5"  # HealthBench Pro #1, ~0.660
        "openai/gpt-6-astra"        # HealthBench Pro ~63.4
    )
fi

VERSION="2026-09"

if [ ! -f "$TEST_SET" ]; then
    echo "Test set $TEST_SET missing. Regenerate standard sets first:"
    echo "  docker compose run --rm evaluator ./scripts/create_standard_test_sets.sh"
    exit 1
fi

echo "run-2026-09-healthbench-refresh | $TEST_SET | ${#MODELS[@]} models"

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    pred_file="results/artifacts/${model_safe}-${LABEL}.json"
    eval_file="results/artifacts/${model_safe}-${LABEL}-eval.json"

    echo "========================================"
    echo "Model: $model"
    echo "========================================"

    if [ -f "$pred_file" ]; then
        echo "Predictions exist, skipping inference"
    else
        docker compose run --rm inference python3 -m inference.run_inference \
            --cases "$TEST_SET" \
            --model "$model" \
            --out "$pred_file" \
            --temperature 0.0 || { echo "Inference failed for $model"; continue; }
    fi

    if [ -f "$eval_file" ]; then
        echo "Eval exists, skipping"
    else
        docker compose run --rm evaluator python3 -m evaluator.cli \
            --cases "$TEST_SET" \
            --predictions "$pred_file" \
            --model-name "$model_safe" \
            --model-version "$VERSION" \
            --out "$eval_file" || echo "Evaluation failed for $model"
    fi
done

echo ""
echo "Next: review eval JSONs, then copy to leaderboard/ and commit:"
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    echo "  cp results/artifacts/${model_safe}-${LABEL}-eval.json leaderboard/"
done
