#!/bin/bash
#
# Re-run evaluations for all existing predictions with updated methodology
# This preserves predictions but regenerates eval files with new rules
#

set -e

cd /home/claude/cortico/MedSafe-Dx

echo "========================================"
echo "MedSafe-Dx: Re-running Evaluations"
echo "========================================"
echo ""
echo "This will re-evaluate all existing predictions with the updated methodology:"
echo "  - Double jeopardy fix (unsafe reassurance rule)"
echo "  - Expected harm scoring"
echo ""

TEST_SET="data/test_sets/test-v0.json"

# Find all prediction files (exclude -eval.json and -transcript.txt)
PREDICTION_FILES=$(ls results/artifacts/*-100cases.json 2>/dev/null | grep -v '\-eval\.json' | grep -v '\-transcript\.txt' || true)

if [ -z "$PREDICTION_FILES" ]; then
    echo "No prediction files found in results/artifacts/"
    exit 1
fi

echo "Found prediction files:"
for pred in $PREDICTION_FILES; do
    echo "  - $(basename "$pred")"
done
echo ""

# Step 1: Re-run evaluations
echo "========================================"
echo "Step 1: Re-running Evaluations"
echo "========================================"

for pred_path in $PREDICTION_FILES; do
    base=$(basename "$pred_path" .json)
    model_name="${base%-100cases}"
    eval_path="results/artifacts/${base}-eval.json"

    echo ""
    echo "Model: $model_name"
    echo "  Input: $pred_path"
    echo "  Output: $eval_path"

    # Remove old evaluation if exists
    if [ -f "$eval_path" ]; then
        echo "  Removing old evaluation..."
        rm "$eval_path"
    fi

    # Run evaluation
    docker compose run --rm evaluator python3 -m evaluator.cli \
        --cases "$TEST_SET" \
        --predictions "$pred_path" \
        --model-name "$model_name" \
        --model-version "2026-01" \
        --out "$eval_path"

    echo "  ✓ Evaluation complete"
done

# Step 2: Copy to leaderboard
echo ""
echo "========================================"
echo "Step 2: Updating Leaderboard"
echo "========================================"

mkdir -p leaderboard

for pred_path in $PREDICTION_FILES; do
    base=$(basename "$pred_path" .json)
    eval_path="results/artifacts/${base}-eval.json"
    leaderboard_eval="leaderboard/${base}-eval.json"
    leaderboard_pred="leaderboard/${base}.json"

    if [ -f "$eval_path" ]; then
        cp "$eval_path" "$leaderboard_eval"
        echo "  ✓ Copied: $(basename "$leaderboard_eval")"
    fi

    # Also copy prediction file to leaderboard for reference
    if [ -f "$pred_path" ] && [ ! -f "$leaderboard_pred" ]; then
        cp "$pred_path" "$leaderboard_pred"
        echo "  ✓ Copied: $(basename "$leaderboard_pred")"
    fi
done

# Step 3: Regenerate transcripts
echo ""
echo "========================================"
echo "Step 3: Regenerating Transcripts"
echo "========================================"

for pred_path in $PREDICTION_FILES; do
    base=$(basename "$pred_path" .json)
    model_name="${base%-100cases}"
    transcript_path="results/artifacts/${base}-transcript.txt"

    if [ -f "$transcript_path" ]; then
        echo "  Removing old transcript: $(basename "$transcript_path")"
        rm "$transcript_path"
    fi

    docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
        --cases "$TEST_SET" \
        --predictions "$pred_path" \
        --model-name "$model_name" \
        --out "$transcript_path"

    echo "  ✓ Generated: $(basename "$transcript_path")"
done

# Step 4: Summary
echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo ""

printf "%-45s %6s %6s %6s %8s %8s %8s\n" "Model" "Cases" "Missed" "OvConf" "UnsafeR" "OverEsc" "Top1"
printf "%-45s %6s %6s %6s %8s %8s %8s\n" "-----" "-----" "------" "------" "-------" "-------" "----"

for pred_path in $PREDICTION_FILES; do
    base=$(basename "$pred_path" .json)
    model_name="${base%-100cases}"
    eval_path="results/artifacts/${base}-eval.json"

    if [ -f "$eval_path" ]; then
        cases=$(jq -r '.cases' "$eval_path")
        missed=$(jq -r '.safety.missed_escalations' "$eval_path")
        overconf=$(jq -r '.safety.overconfident_wrong' "$eval_path")
        unsafe=$(jq -r '.safety.unsafe_reassurance' "$eval_path")
        overesc=$(jq -r '.effectiveness.over_escalation' "$eval_path")
        top1=$(jq -r '.effectiveness.top1_recall | (. * 100 | floor) / 100' "$eval_path")

        printf "%-45s %6s %6s %6s %8s %8s %8s\n" "$model_name" "$cases" "$missed" "$overconf" "$unsafe" "$overesc" "$top1"
    fi
done

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo ""
echo "View leaderboard:"
echo "  ./scripts/serve_leaderboard.sh"
echo ""
echo "View individual evaluation:"
echo "  jq '.' results/artifacts/<model>-100cases-eval.json"
echo ""
