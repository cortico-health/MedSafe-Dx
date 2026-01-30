#!/bin/bash
# Regenerate transcripts for all prediction files
set -e
cd /home/claude/cortico/MedSafe-Dx

TEST_SET="data/test_sets/test-v0.json"

for pred in results/artifacts/*-100cases.json; do
    # Skip eval files
    [[ "$pred" == *-eval.json ]] && continue

    base=$(basename "$pred" .json)
    model_name="${base%-100cases}"
    transcript="results/artifacts/${base}-transcript.txt"

    echo "Generating: $transcript"
    docker compose run --rm evaluator python3 scripts/generate_review_transcript.py \
        --cases "$TEST_SET" \
        --predictions "$pred" \
        --out "$transcript" \
        --model-name "$model_name"
done

echo ""
echo "All transcripts regenerated."
