#!/bin/bash
#
# Create standard test sets for the benchmark
#
# This creates a suite of test sets with different sizes and purposes:
# - dev: Small set for quick iteration (10 cases)
# - test: Medium set for regular testing (100 cases)
# - eval: Larger set for formal evaluation (500 cases)
# - full-eval: Large comprehensive set (2000 cases)
#
# All sets use fixed seeds for reproducibility
#

set -e

echo "========================================"
echo "Creating Standard Test Sets"
echo "========================================"

INPUT_CASES="data/ddxplus_v0/cases.json"
OUTPUT_DIR="data/test_sets"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Define test sets
declare -A TEST_SETS=(
    ["dev"]="10:1"
    ["test"]="100:42"
    # NOTE: These seeds are intentionally aligned with the committed v0 test sets in
    # data/test_sets/*.json to avoid “seed drift” between the script and the repo.
    ["eval"]="500:42"
    ["full-eval"]="2000:42"
)

# Create each test set
for name in "${!TEST_SETS[@]}"; do
    IFS=':' read -r num_cases seed <<< "${TEST_SETS[$name]}"
    
    output_file="${OUTPUT_DIR}/${name}-v0.json"
    
    echo ""
    echo "Creating: $name"
    echo "  Cases: $num_cases"
    echo "  Seed: $seed"
    echo "  Output: $output_file"
    
    if [ -f "$output_file" ]; then
        echo "  ⚠ Already exists, skipping (delete to regenerate)"
        continue
    fi
    
    python3 scripts/prep_test_cases.py \
        --input "$INPUT_CASES" \
        --output "$output_file" \
        --num-cases "$num_cases" \
        --seed "$seed" \
        --name "${name}-v0"
    
    echo "  ✓ Created"
done

echo ""
echo "========================================"
echo "Test Sets Created"
echo "========================================"
echo ""
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No test sets created (all existed)"
echo ""
echo "Usage:"
echo "  python3 -m inference.run_inference --cases $OUTPUT_DIR/dev-v0.json ..."
echo "  python3 -m inference.run_inference --cases $OUTPUT_DIR/test-v0.json ..."
echo "  python3 -m inference.run_inference --cases $OUTPUT_DIR/eval-v0.json ..."
echo ""
