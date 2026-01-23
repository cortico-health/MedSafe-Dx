import json
import argparse
from datetime import datetime, timezone
import sys
from evaluator.evaluator import evaluate
from evaluator.harm import HarmWeights


def load_harm_weights(path: str) -> HarmWeights:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("harm weights JSON must be an object")
    allowed = set(HarmWeights().to_dict().keys())
    unknown = sorted(set(data.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown harm weight keys: {unknown}")
    return HarmWeights(**{k: float(v) for k, v in data.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the run if there are missing/extra/duplicate/invalid predictions",
    )
    parser.add_argument(
        "--harm-weights",
        default=None,
        help="Optional path to JSON object of harm weights (overrides defaults)",
    )

    args = parser.parse_args()

    harm_weights = load_harm_weights(args.harm_weights) if args.harm_weights else None
    try:
        artifact = evaluate(
            args.cases,
            args.predictions,
            args.model_name,
            args.model_version,
            harm_weights=harm_weights,
            strict=args.strict,
        )
    except Exception as e:
        print(f"ERROR: evaluation failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    # Add precise timestamp
    artifact["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)


if __name__ == "__main__":
    main()
