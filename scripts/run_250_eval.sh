#!/bin/bash
# Run 250-case evaluation for all models
set -e
cd /home/claude/cortico/MedSafe-Dx

TEST_SET="data/test_sets/eval-250-v0.json"
MODELS=(
    "anthropic/claude-haiku-4.5"
    "anthropic/claude-sonnet-4.5"
    "openai/gpt-5.2"
    "openai/gpt-5-mini"
    "openai/gpt-4o-mini"
    "openai/gpt-5-chat"
    "deepseek/deepseek-chat-v3-0324"
    "openai/gpt-oss-120b"
    "openai/gpt-4.1"
    "google/gemini-2.5-flash-lite-preview-05-20"
    "google/gemini-3-pro-preview"
    "google/gemini-2.0-flash-001"
)

echo "Running 250-case evaluation for ${#MODELS[@]} models"
echo "Test set: $TEST_SET"
echo ""

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    pred_file="results/artifacts/${model_safe}-250cases.json"
    eval_file="results/artifacts/${model_safe}-250cases-eval.json"

    echo "========================================"
    echo "Model: $model"
    echo "========================================"

    # Run inference if predictions don't exist
    if [ -f "$pred_file" ]; then
        echo "Predictions exist, skipping inference"
    else
        echo "Running inference..."
        docker compose run --rm inference python3 -m inference.run_inference \
            --cases "$TEST_SET" \
            --model "$model" \
            --out "$pred_file" \
            --temperature 0.0 || echo "Inference failed for $model"
    fi

    # Run evaluation if predictions exist
    if [ -f "$pred_file" ]; then
        echo "Running evaluation..."
        docker compose run --rm evaluator python3 -m evaluator.cli \
            --cases "$TEST_SET" \
            --predictions "$pred_file" \
            --model-name "$model_safe" \
            --model-version "2026-01" \
            --out "$eval_file" || echo "Evaluation failed for $model"
    fi

    echo ""
done

echo "========================================"
echo "Results Summary"
echo "========================================"
printf "%-45s %6s %8s %6s %6s %6s\n" "Model" "Cases" "SafePass" "Missed" "OvConf" "Unsafe"
printf "%-45s %6s %8s %6s %6s %6s\n" "-----" "-----" "--------" "------" "------" "------"

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    eval_file="results/artifacts/${model_safe}-250cases-eval.json"

    if [ -f "$eval_file" ]; then
        cases=$(jq -r '.valid_predictions' "$eval_file")
        safety=$(jq -r '.safety_pass_rate | . * 100 | floor' "$eval_file")
        missed=$(jq -r '.safety.missed_escalations' "$eval_file")
        overconf=$(jq -r '.safety.overconfident_wrong' "$eval_file")
        unsafe=$(jq -r '.safety.unsafe_reassurance' "$eval_file")
        printf "%-45s %6s %7s%% %6s %6s %6s\n" "$model_safe" "$cases" "$safety" "$missed" "$overconf" "$unsafe"
    fi
done

echo ""
echo "Done!"
