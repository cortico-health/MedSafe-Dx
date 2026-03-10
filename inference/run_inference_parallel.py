#!/usr/bin/env python3
"""
Parallel inference runner for MedSafe-Dx benchmark.
Runs N concurrent workers per model for much faster throughput.
"""

import json
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from inference.openrouter import call_openrouter, load_cases, write_predictions
from inference.run_inference import (
    run_inference_on_case,
    get_system_prompt,
    format_case_for_prompt,
)

# Thread-safe counter
class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.count = 0
        self.success = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment(self, success=True):
        with self.lock:
            self.count += 1
            if success:
                self.success += 1
            else:
                self.failed += 1
            if self.count % 10 == 0 or self.count == self.total:
                elapsed = time.time() - self.start_time
                rate = self.count / elapsed if elapsed > 0 else 0
                eta = (self.total - self.count) / rate if rate > 0 else 0
                print(f"Progress: {self.count}/{self.total} "
                      f"(ok={self.success}, fail={self.failed}) "
                      f"[{rate:.1f}/s, ETA {eta/60:.0f}m]",
                      flush=True)


def run_single_case(case, model, workflow, temperature):
    """Worker function for thread pool."""
    result = run_inference_on_case(case, model=model, workflow=workflow, temperature=temperature)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Path to test cases JSON")
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers (default: 10)")
    parser.add_argument("--workflow", choices=["intake", "chart_review"], default="intake")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from partial output file")
    
    args = parser.parse_args()
    
    print(f"Loading cases from {args.cases}...")
    cases, metadata = load_cases(args.cases)
    
    if metadata:
        print(f"Test set: {metadata.get('test_set_name', '?')}, "
              f"seed={metadata.get('seed', '?')}, "
              f"cases={metadata.get('sampled_cases', '?')}")
    
    if args.limit:
        cases = cases[:args.limit]
    
    # Resume support: skip already-completed cases
    completed_ids = set()
    existing_predictions = []
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            existing = json.load(f)
        if isinstance(existing, dict) and "predictions" in existing:
            existing_predictions = existing["predictions"]
        elif isinstance(existing, list):
            existing_predictions = existing
        completed_ids = {p["case_id"] for p in existing_predictions if "case_id" in p}
        print(f"Resuming: {len(completed_ids)} cases already done, "
              f"{len(cases) - len(completed_ids)} remaining")
    
    remaining_cases = [c for c in cases if c["case_id"] not in completed_ids]
    
    if not remaining_cases:
        print("All cases already completed!")
        return
    
    print(f"Running inference on {len(remaining_cases)} cases with {args.workers} workers...")
    print(f"Model: {args.model}")
    
    counter = ProgressCounter(len(remaining_cases))
    predictions = list(existing_predictions)  # Start with existing
    predictions_lock = threading.Lock()
    
    # Periodic save every 50 cases
    save_counter = threading.Lock()
    last_save_count = [len(existing_predictions)]
    
    def save_checkpoint():
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_metadata = {
            "model": args.model,
            "temperature": args.temperature,
            "workflow": args.workflow,
            "prompt_version": "v4",
            "total_cases": len(predictions),
            "successful_predictions": counter.success,
            "failed_predictions": counter.failed,
            "status": "in_progress",
        }
        if metadata:
            output_metadata["test_set_metadata"] = metadata
        with predictions_lock:
            write_predictions(output_path, list(predictions), output_metadata)
        print(f"  [checkpoint saved: {len(predictions)} predictions]", flush=True)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_case, case, args.model, args.workflow, args.temperature): case
            for case in remaining_cases
        }
        
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
                success = isinstance(result, dict) and "error" not in result
                counter.increment(success=success)
                with predictions_lock:
                    predictions.append(result)
                
                # Checkpoint every 50 new predictions
                with save_counter:
                    if len(predictions) - last_save_count[0] >= 50:
                        last_save_count[0] = len(predictions)
                        save_checkpoint()
                        
            except Exception as e:
                print(f"Worker exception for case {case.get('case_id', '?')}: {e}", flush=True)
                counter.increment(success=False)
                with predictions_lock:
                    predictions.append({
                        "case_id": case["case_id"],
                        "workflow": args.workflow,
                        "error": f"worker_exception: {str(e)}",
                        "raw_response": None,
                    })
    
    # Final save
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata = {
        "model": args.model,
        "temperature": args.temperature,
        "workflow": args.workflow,
        "prompt_version": "v4",
        "total_cases": len(predictions),
        "successful_predictions": counter.success,
        "failed_predictions": counter.failed,
        "status": "complete",
    }
    if metadata:
        output_metadata["test_set_metadata"] = metadata
    write_predictions(output_path, predictions, output_metadata)
    
    elapsed = time.time() - counter.start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes!")
    print(f"Successful: {counter.success}")
    print(f"Failed: {counter.failed}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
