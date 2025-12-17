# Preparing Random Test Cases

The `prep_test_cases.py` script allows you to create reproducible random subsets of cases for model evaluation.

## Why use this?

- **Reproducibility**: Multiple models can be tested on the exact same random subset
- **Seed-based**: Results are deterministic and traceable
- **Metadata tracking**: Each test set includes full provenance information

## Quick Start

### 1. Prepare a test set

```bash
python3 scripts/prep_test_cases.py \
  --input data/ddxplus_v0/cases.json \
  --output data/test_sets/eval-100-seed42.json \
  --num-cases 100 \
  --seed 42 \
  --name "v0-eval-100"
```

This creates a test set with:
- 100 randomly selected cases
- Seed 42 (for reproducibility)
- Metadata embedded in the output

### 2. Run inference on multiple models

Now you can run multiple models on the **exact same cases**:

```bash
# Model 1
python3 -m inference.run_inference \
  --cases data/test_sets/eval-100-seed42.json \
  --model "anthropic/claude-sonnet-4" \
  --out results/artifacts/claude-sonnet-4-eval.json

# Model 2
python3 -m inference.run_inference \
  --cases data/test_sets/eval-100-seed42.json \
  --model "openai/gpt-4o" \
  --out results/artifacts/gpt-4o-eval.json

# Model 3
python3 -m inference.run_inference \
  --cases data/test_sets/eval-100-seed42.json \
  --model "meta-llama/llama-3.1-70b-instruct" \
  --out results/artifacts/llama-3-eval.json
```

### 3. Evaluate results

```bash
python3 -m evaluator.cli \
  --cases data/test_sets/eval-100-seed42.json \
  --predictions results/artifacts/claude-sonnet-4-eval.json \
  --model-name "claude-sonnet-4" \
  --model-version "2025-01" \
  --out results/artifacts/claude-sonnet-4-eval-results.json
```

## Options

```
--input       Input cases file (default: data/ddxplus_v0/cases.json)
--output      Output path for sampled cases (required)
--num-cases   Number of cases to sample (required)
--seed        Random seed (default: 42)
--name        Optional test set name for tracking
```

## Output Format

The prep script creates a JSON file with this structure:

```json
{
  "metadata": {
    "source_file": "data/ddxplus_v0/cases.json",
    "total_available_cases": 10000,
    "sampled_cases": 100,
    "seed": 42,
    "timestamp": "2025-12-16T12:00:00Z",
    "test_set_name": "v0-eval-100"
  },
  "cases": [
    { "case_id": "...", ... },
    ...
  ]
}
```

## Backward Compatibility

The inference and evaluation pipeline supports both:
- **Old format**: Plain list of cases
- **New format**: Object with metadata + cases

Existing workflows continue to work without changes.

## Example Workflow

```bash
# Create different test sets for different purposes
python3 scripts/prep_test_cases.py \
  --output data/test_sets/quick-test-10.json \
  --num-cases 10 \
  --seed 123 \
  --name "quick-test"

python3 scripts/prep_test_cases.py \
  --output data/test_sets/standard-eval-500.json \
  --num-cases 500 \
  --seed 42 \
  --name "standard-eval"

python3 scripts/prep_test_cases.py \
  --output data/test_sets/large-eval-2000.json \
  --num-cases 2000 \
  --seed 42 \
  --name "large-eval"

# Run multiple models on the standard eval set
for model in "anthropic/claude-sonnet-4" "openai/gpt-4o" "openai/gpt-4o-mini"; do
  python3 -m inference.run_inference \
    --cases data/test_sets/standard-eval-500.json \
    --model "$model" \
    --out "results/artifacts/$(basename $model)-eval.json"
done
```

## Benefits

1. **Reproducible comparisons**: All models tested on identical cases
2. **Traceable**: Seed and source recorded in output
3. **Flexible**: Create multiple test sets for different purposes
4. **Compatible**: Works with existing evaluation pipeline

