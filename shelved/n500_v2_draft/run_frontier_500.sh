#!/bin/bash
# Run MedSafe-Dx benchmark: 500 cases × 3 frontier models (parallel)
set -e

TEST_SET="data/test_sets/eval-v0.json"
WORKERS=10
MODELS=(
    "openai/gpt-5.4-pro"
    "google/gemini-3.1-pro-preview"
    "anthropic/claude-opus-4.6"
)

echo "========================================"
echo "MedSafe-Dx Frontier Benchmark (Parallel)"
echo "========================================"
echo "Cases: 500 | Workers: $WORKERS"
echo ""

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    predictions="results/artifacts/${model_safe}-500cases.json"
    eval_out="results/artifacts/${model_safe}-500cases-eval.json"
    transcript="results/artifacts/${model_safe}-500cases-transcript.txt"
    
    echo "----------------------------------------"
    echo "Model: $model"
    echo "----------------------------------------"
    
    # Inference
    if [ -f "$predictions" ] && python3 -c "import json; d=json.load(open('$predictions')); exit(0 if d.get('metadata',{}).get('status')=='complete' else 1)" 2>/dev/null; then
        echo "⚠ Predictions exist and complete, skipping"
    else
        echo "Running inference (${WORKERS} workers)..."
        docker compose run --rm inference python3 -m inference.run_inference_parallel \
            --cases "$TEST_SET" \
            --model "$model" \
            --out "$predictions" \
            --workers "$WORKERS" \
            --temperature 0.0 \
            --resume
        echo "✓ Inference complete"
    fi
    
    # Evaluation
    if [ -f "$eval_out" ]; then
        echo "⚠ Evaluation exists, skipping"
    else
        echo "Running evaluation..."
        docker compose run --rm evaluator python3 -m evaluator.cli \
            --cases "$TEST_SET" \
            --predictions "$predictions" \
            --model-name "$model_safe" \
            --model-version "2026-03" \
            --out "$eval_out"
        echo "✓ Evaluation complete"
    fi
    
    # Transcript
    if [ -f "$transcript" ]; then
        echo "⚠ Transcript exists, skipping"
    else
        echo "Generating transcript..."
        docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
            --cases "$TEST_SET" \
            --predictions "$predictions" \
            --model-name "$model" \
            --out "$transcript"
        echo "✓ Transcript complete"
    fi
    
    echo ""
done

echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""

for model in "${MODELS[@]}"; do
    model_safe=$(echo "$model" | sed 's/\//-/g')
    eval_path="results/artifacts/${model_safe}-500cases-eval.json"
    
    if [ ! -f "$eval_path" ]; then
        echo "Model: $model — No results"
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
