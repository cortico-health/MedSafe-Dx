#!/bin/bash
#
# Example: Run multiple models on the same random test set
#
# This demonstrates the reproducible evaluation workflow:
# 1. Prepare a seeded test set once
# 2. Run multiple models on the same test set
# 3. Compare results fairly
#

set -e  # Exit on error

echo "========================================"
echo "Multi-Model Evaluation Example"
echo "========================================"

# Configuration
TEST_SET_NAME="standard-eval-100"
TEST_SET_PATH="data/test_sets/${TEST_SET_NAME}.json"
NUM_CASES=100
SEED=42

# Models to evaluate
MODELS=(
    "anthropic/claude-sonnet-4"
    "openai/gpt-4o"
    "openai/gpt-4o-mini"
)

# Step 1: Prepare test set (only once)
echo ""
echo "Step 1: Preparing test set..."
echo "----------------------------------------"

if [ -f "$TEST_SET_PATH" ]; then
    echo "Test set already exists: $TEST_SET_PATH"
    echo "Skipping preparation (delete to regenerate)"
else
    python3 scripts/prep_test_cases.py \
        --input data/ddxplus_v0/cases.json \
        --output "$TEST_SET_PATH" \
        --num-cases "$NUM_CASES" \
        --seed "$SEED" \
        --name "$TEST_SET_NAME"
fi

# Step 2: Run inference for each model
echo ""
echo "Step 2: Running inference on all models..."
echo "----------------------------------------"

for model in "${MODELS[@]}"; do
    # Create safe filename from model name
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions_path="results/artifacts/${model_safe}-predictions.json"
    
    echo ""
    echo "Running: $model"
    
    if [ -f "$predictions_path" ]; then
        echo "  ⚠ Predictions already exist: $predictions_path"
        echo "  ⚠ Skipping (delete to regenerate)"
        continue
    fi
    
    # Note: This requires OPENROUTER_API_KEY in environment
    # For demo purposes, we'll just show the command
    echo "  Command: python3 -m inference.run_inference \\"
    echo "    --cases $TEST_SET_PATH \\"
    echo "    --model $model \\"
    echo "    --out $predictions_path"
    
    # Uncomment to actually run:
    # python3 -m inference.run_inference \
    #     --cases "$TEST_SET_PATH" \
    #     --model "$model" \
    #     --out "$predictions_path" \
    #     --temperature 0.0
done

# Step 3: Evaluate all models
echo ""
echo "Step 3: Evaluating all models..."
echo "----------------------------------------"

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions_path="results/artifacts/${model_safe}-predictions.json"
    eval_path="results/artifacts/${model_safe}-eval.json"
    
    if [ ! -f "$predictions_path" ]; then
        echo ""
        echo "Skipping $model (no predictions found)"
        continue
    fi
    
    if [ -f "$eval_path" ]; then
        echo ""
        echo "Evaluation already exists: $eval_path"
        echo "Skipping (delete to regenerate)"
        continue
    fi
    
    echo ""
    echo "Evaluating: $model"
    
    python3 -m evaluator.cli \
        --cases "$TEST_SET_PATH" \
        --predictions "$predictions_path" \
        --model-name "$model_safe" \
        --model-version "2025-01" \
        --out "$eval_path"
    
    echo "  ✓ Results: $eval_path"
done

# Step 4: Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo ""
echo "Test Set: $TEST_SET_PATH"
echo "  Cases: $NUM_CASES"
echo "  Seed: $SEED"
echo ""
echo "All models evaluated on identical cases!"
echo ""
echo "To view results:"
echo "  ls -lh results/artifacts/*-eval.json"
echo ""
echo "To compare safety metrics:"
echo "  jq '.safety' results/artifacts/*-eval.json"
echo ""

