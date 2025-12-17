#!/bin/bash
#
# General model testing script
# Usage: ./test_models.sh <num_cases> <model1> <model2> [model3] ...
#

set -e  # Exit on error

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_cases> <model1> <model2> [model3] ..."
    echo ""
    echo "Examples:"
    echo "  $0 1 anthropic/claude-sonnet-4.5"
    echo "  $0 10 anthropic/claude-haiku-4.5 openai/gpt-4o-mini"
    echo "  $0 100 anthropic/claude-haiku-4.5 anthropic/claude-sonnet-4.5 openai/gpt-4o-mini"
    echo ""
    echo "Available models (examples):"
    echo "  anthropic/claude-haiku-4.5"
    echo "  anthropic/claude-sonnet-4.5"
    echo "  openai/gpt-oss-120b"
    echo "  google/gemini-2.5-flash-lite"
    echo "  deepseek/deepseek-chat-v3-0324"
    echo "  openai/gpt-4o-mini"
    exit 1
fi

NUM_CASES="$1"
shift  # Remove first argument, rest are models

# Build models array
MODELS=("$@")

echo "========================================"
echo "Testing ${#MODELS[@]} Models on $NUM_CASES Cases"
echo "========================================"

# Configuration
TEST_SET="data/test_sets/test-${NUM_CASES}cases-${#MODELS[@]}models.json"
SEED=42

echo ""
echo "Models to test:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Step 1: Generate test set if needed
if [ -f "$TEST_SET" ]; then
    echo "✓ Test set already exists: $TEST_SET"
else
    echo "Generating test set..."
    docker compose run --rm evaluator python3 scripts/prep_test_cases.py \
        --output "$TEST_SET" \
        --num-cases "$NUM_CASES" \
        --seed "$SEED" \
        --name "test-${NUM_CASES}cases-${#MODELS[@]}models"
    echo "✓ Test set created"
fi

echo ""
echo "========================================"
echo "Running Inference"
echo "========================================"

# Step 2: Run inference for each model
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions_path="results/artifacts/${model_safe}-${NUM_CASES}cases.json"
    
    echo ""
    echo "Model: $model"
    echo "Output: $predictions_path"
    
    if [ -f "$predictions_path" ]; then
        echo "⚠ Predictions exist, skipping inference (delete to rerun)"
    else
        docker compose run --rm inference python3 -m inference.run_inference \
            --cases "$TEST_SET" \
            --model "$model" \
            --out "$predictions_path" \
            --temperature 0.0
        echo "✓ Inference complete"
    fi
done

echo ""
echo "========================================"
echo "Running Evaluations"
echo "========================================"

# Step 3: Evaluate each model
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions_path="results/artifacts/${model_safe}-${NUM_CASES}cases.json"
    eval_path="results/artifacts/${model_safe}-${NUM_CASES}cases-eval.json"
    
    echo ""
    echo "Model: $model"
    
    if [ ! -f "$predictions_path" ]; then
        echo "✗ No predictions found, skipping evaluation"
        continue
    fi
    
    if [ -f "$eval_path" ]; then
        echo "⚠ Evaluation exists, skipping (delete to rerun)"
    else
        docker compose run --rm evaluator python3 -m evaluator.cli \
            --cases "$TEST_SET" \
            --predictions "$predictions_path" \
            --model-name "$model_safe" \
            --model-version "2025-01" \
            --out "$eval_path"
        echo "✓ Evaluation complete"
    fi
done

echo ""
echo "========================================"
echo "Generating Clinical Review Transcripts"
echo "========================================"

# Step 4: Generate review transcripts
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions_path="results/artifacts/${model_safe}-${NUM_CASES}cases.json"
    transcript_path="results/artifacts/${model_safe}-${NUM_CASES}cases-transcript.txt"
    
    if [ ! -f "$predictions_path" ]; then
        continue
    fi
    
    if [ -f "$transcript_path" ]; then
        echo "  ⚠ Transcript exists: $transcript_path"
    else
        docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
            --cases "$TEST_SET" \
            --predictions "$predictions_path" \
            --model-name "$model" \
            --out "$transcript_path"
    fi
done

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""

# Step 5: Display results
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    eval_path="results/artifacts/${model_safe}-${NUM_CASES}cases-eval.json"
    
    if [ ! -f "$eval_path" ]; then
        echo "Model: $model"
        echo "  Status: No results"
        echo ""
        continue
    fi
    
    echo "Model: $model"
    echo "  Cases: $(jq -r '.cases' "$eval_path") / $(jq -r '.total_attempted' "$eval_path") (valid / attempted)"
    echo "  Format failures: $(jq -r '.format_failures' "$eval_path")"
    echo "  Safety:"
    echo "    Missed escalations: $(jq -r '.safety.missed_escalations' "$eval_path")"
    echo "    Overconfident wrong: $(jq -r '.safety.overconfident_wrong' "$eval_path")"
    echo "    Unsafe reassurance: $(jq -r '.safety.unsafe_reassurance' "$eval_path")"
    echo "  Effectiveness:"
    echo "    Top-1 recall: $(jq -r '.effectiveness.top1_recall' "$eval_path")"
    echo "    Top-3 recall: $(jq -r '.effectiveness.top3_recall' "$eval_path")"
    echo ""
done

echo "========================================"
echo "Complete!"
echo "========================================"
echo ""
echo "Review clinical transcripts (for doctor review):"
for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    echo "  cat results/artifacts/${model_safe}-${NUM_CASES}cases-transcript.txt"
done
echo ""
echo "Compare safety metrics:"
echo "  jq '.safety' results/artifacts/*-${NUM_CASES}cases-eval.json"
echo ""
echo "Compare effectiveness metrics:"
echo "  jq '.effectiveness' results/artifacts/*-${NUM_CASES}cases-eval.json"
echo ""

