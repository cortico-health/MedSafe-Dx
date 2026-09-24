import fcntl
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import sys
from evaluator.evaluator import evaluate, sha256_file
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


def lock_predictions(predictions_path: str):
    """Share-lock <predictions>.lock so no inference run can write while we score.

    inference/run_inference.py holds an exclusive lock on the same file for its
    whole run. If that lock is held the file is still being written, so we refuse
    to score: the eval JSON would record a sha256 for a file that then changes.
    """
    lock_path = Path(predictions_path + ".lock")
    if not lock_path.exists():
        return None
    fd = open(lock_path)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise RuntimeError(
            f"inference is still writing {predictions_path} (lock {lock_path} is held); "
            "score it after that run finishes"
        )
    return fd


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
        lock_fd = lock_predictions(args.predictions)  # noqa: F841 - held until exit
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

    # The recorded sha256 must describe the file as it is now, or the row cannot be
    # rebuilt from its predictions later.
    if sha256_file(args.predictions) != artifact["predictions_sha256"]:
        print(f"ERROR: {args.predictions} changed while it was being scored", file=sys.stderr)
        raise SystemExit(1)

    # Add precise timestamp
    artifact["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)


if __name__ == "__main__":
    main()
