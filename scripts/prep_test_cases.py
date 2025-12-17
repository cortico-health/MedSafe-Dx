#!/usr/bin/env python3
"""
Prepare random test cases from the full case set based on a seed.
This allows multiple models to run on identical random subsets.
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime


def prep_test_cases(
    input_cases: str,
    output_path: str,
    num_cases: int,
    seed: int,
    metadata: dict = None,
):
    """
    Generate a random subset of cases based on seed.
    
    Args:
        input_cases: Path to full cases.json
        output_path: Path to save sampled cases
        num_cases: Number of cases to sample
        seed: Random seed for reproducibility
        metadata: Additional metadata to include in output
    """
    # Load all cases
    with open(input_cases) as f:
        all_cases = json.load(f)
    
    if num_cases > len(all_cases):
        print(f"Warning: Requested {num_cases} cases but only {len(all_cases)} available")
        num_cases = len(all_cases)
    
    # Set seed and sample
    random.seed(seed)
    sampled_cases = random.sample(all_cases, num_cases)
    
    # Sort by case_id for consistent ordering in output
    sampled_cases.sort(key=lambda c: c.get("case_id", ""))
    
    # Prepare output with metadata
    output_data = {
        "metadata": {
            "source_file": str(input_cases),
            "total_available_cases": len(all_cases),
            "sampled_cases": num_cases,
            "seed": seed,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(metadata or {})
        },
        "cases": sampled_cases
    }
    
    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Sampled {num_cases} cases from {len(all_cases)} total")
    print(f"✓ Seed: {seed}")
    print(f"✓ Output: {output_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare random test cases based on seed for reproducible model evaluation"
    )
    parser.add_argument(
        "--input",
        default="data/ddxplus_v0/cases.json",
        help="Input cases file (full dataset)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for sampled cases",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        required=True,
        help="Number of cases to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--name",
        help="Optional test set name (e.g., 'v0-eval', 'quick-test')",
    )
    
    args = parser.parse_args()
    
    metadata = {}
    if args.name:
        metadata["test_set_name"] = args.name
    
    prep_test_cases(
        input_cases=args.input,
        output_path=args.output,
        num_cases=args.num_cases,
        seed=args.seed,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()

