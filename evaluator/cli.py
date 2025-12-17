import json
import argparse
from datetime import datetime
from evaluator.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    artifact = evaluate(
        args.cases,
        args.predictions,
        args.model_name,
        args.model_version,
    )
    
    # Add precise timestamp
    artifact["timestamp"] = datetime.utcnow().isoformat() + 'Z'

    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)


if __name__ == "__main__":
    main()
