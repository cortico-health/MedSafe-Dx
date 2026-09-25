#!/bin/bash
# MedSafe-Dx v0.3 run (spec/v0.3-scoring.md draft 2, section 9: as in v0.2 section 9).
#
# Runs each model on the main 470-case sample and both pool subsets (atypical 150,
# high risk 100) with prompt v6 and inference/run_config_v03.json (the v0.2 values:
# max_tokens 16000, reasoning effort "medium", one retry on an empty or truncated
# response, per-prediction finish reason, provider, request id and token usage),
# then scores every row at once. We score only after every inference worker has
# exited, because the board's bootstrap pairs all rows on the same draws and a
# half-written file would change a sha256 the provenance already recorded.
#
# Rules the script keeps:
# 1. One writer per prediction file: run_inference holds an exclusive lock on
#    <file>.lock for the whole run, and the scorer refuses a file whose lock is held.
# 2. A prediction file that already covers every case of its set is not re-run;
#    a partial file resumes (errored cases are re-run).
# 3. A model OpenRouter no longer lists is skipped and named in the provenance.
# 4. Provenance (git commit, input sha256s, config, timings, per-file sha256 and
#    token cost) goes to $OUT_DIR/provenance.json.
#
# Usage:
#   ./scripts/run_v03.sh                         # the two test models (asks for confirmation)
#   MODELS_OVERRIDE="a/b c/d" ./scripts/run_v03.sh
#   DRY_RUN=1 ./scripts/run_v03.sh               # render every request, send nothing
#   NO_SCORE=1 ./scripts/run_v03.sh              # run inference only; re-run later to score
#   RUNNER=docker PARALLEL=6 INFERENCE_WORKERS=8 ./scripts/run_v03.sh

set -euo pipefail
cd "$(dirname "$0")/.."
export MEDSAFE_GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

RUN_CONFIG="inference/run_config_v03.json"
OUT_DIR="${OUT_DIR:-results/v03/runs}"
PARALLEL="${PARALLEL:-6}"
export INFERENCE_WORKERS="${INFERENCE_WORKERS:-8}"
RUNNER="${RUNNER:-local}"
DRY_RUN="${DRY_RUN:-0}"
NO_SCORE="${NO_SCORE:-0}"

# Set label -> case file. The label names the prediction file.
SETS=(
    "470cases:data/test_sets/eval-v02-adult.json"
    "pool-atypical:data/test_sets/eval-v03-pool-atypical.json"
    "pool-high-risk:data/test_sets/eval-v03-pool-high-risk.json"
)

# Test roster for draft 2 (spec section header: frozen for the two test runs).
ROSTER=("openai/gpt-5.6-terra" "openai/gpt-oss-120b")
if [ -n "${MODELS_OVERRIDE:-}" ]; then
    read -r -a ROSTER <<< "$MODELS_OVERRIDE"
fi

for s in "${SETS[@]}"; do
    [ -f "${s#*:}" ] || { echo "Missing ${s#*:}; build it with scripts/build_v03_key.py" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"

AVAILABLE=$(python3 - "$RUN_CONFIG" "${ROSTER[@]}" <<'PY'
import json, sys, urllib.request
cfg = json.load(open(sys.argv[1]))["models"]
roster = sys.argv[2:]
try:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        listed = {m["id"] for m in json.load(r)["data"]}
except Exception as e:  # offline: trust the roster, the run itself will fail loudly
    print(f"WARNING: could not fetch the OpenRouter model list ({e})", file=sys.stderr)
    listed = set(roster)
for m in roster:
    if m not in cfg:
        print(f"WARNING: {m} is not in {sys.argv[1]}; its row will carry config_overridden", file=sys.stderr)
    if m in listed:
        print(m)
    else:
        print(f"SKIP: {m} is not listed on OpenRouter", file=sys.stderr)
PY
)
read -r -a MODELS <<< "$(echo $AVAILABLE)"
SKIPPED=$(comm -23 <(printf '%s\n' "${ROSTER[@]}" | sort) <(printf '%s\n' "${MODELS[@]}" | sort) | tr '\n' ' ')

run_py() {
    if [ "$RUNNER" = "local" ]; then
        python3 "$@"
    else
        docker compose run --rm -e INFERENCE_WORKERS inference python3 "$@"
    fi
}

pred_file() { echo "$OUT_DIR/$(echo "$1" | sed 's/\//-/g')-v03-$2.json"; }

complete() {
    [ -f "$1" ] && python3 - "$1" "$2" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); preds = p["predictions"] if isinstance(p, dict) else p
c = json.load(open(sys.argv[2]))["cases"]
have = {x.get("case_id") for x in preds if isinstance(x, dict) and "error" not in x}
sys.exit(0 if {x["case_id"] for x in c} <= have else 1)
PY
}

or_usage() {  # account usage in USD, for the provenance spend line (a free GET)
    python3 - <<'PY'
import json, os, urllib.request
from dotenv import load_dotenv
load_dotenv(".env.local"); load_dotenv(".env")
key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
try:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    print(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
except Exception:
    print("null")
PY
}

echo "MedSafe-Dx v0.3 | ${#MODELS[@]} models x ${#SETS[@]} sets | commit $MEDSAFE_GIT_COMMIT | dry run: $DRY_RUN"
[ -n "$SKIPPED" ] && echo "Skipped (not on OpenRouter): $SKIPPED"

if [ "$DRY_RUN" = "1" ]; then
    for model in "${MODELS[@]}"; do
        for s in "${SETS[@]}"; do
            run_py -m inference.run_inference --prompt-version v6 --run-config "$RUN_CONFIG" \
                --cases "${s#*:}" --model "$model" --out "$(pred_file "$model" "${s%%:*}")" --dry-run | tail -1
        done
    done
    echo "Dry run done: request bodies in $OUT_DIR/*.dryrun.json. Nothing was sent."
    exit 0
fi

TODO=()
for model in "${MODELS[@]}"; do
    for s in "${SETS[@]}"; do
        complete "$(pred_file "$model" "${s%%:*}")" "${s#*:}" || TODO+=("$model|$s")
    done
done

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
USAGE_BEFORE=$(or_usage)
if [ ${#TODO[@]} -gt 0 ]; then
    if [ "${CONFIRM:-}" != "yes" ]; then
        read -r -p "This sends paid requests for ${#TODO[@]} model x set runs. Type yes to continue: " ans
        [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
    fi
    LOG_DIR="$OUT_DIR/logs"; mkdir -p "$LOG_DIR"
    pids=()
    for item in "${TODO[@]}"; do
        model="${item%%|*}"; s="${item#*|}"; label="${s%%:*}"; cases="${s#*:}"
        f=$(pred_file "$model" "$label")
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
        echo "start: $model / $label -> $f"
        ( run_py -m inference.run_inference --prompt-version v6 --run-config "$RUN_CONFIG" \
              --cases "$cases" --model "$model" --out "$f" --workers "$INFERENCE_WORKERS" \
              > "$LOG_DIR/$(basename "$f" .json).log" 2>&1 \
          || echo "FAILED: $model / $label (see $LOG_DIR)" ) &
        pids+=($!)
    done
    # Score only after every worker has exited.
    for p in "${pids[@]}"; do wait "$p" || true; done
fi
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
USAGE_AFTER=$(or_usage)

FILES=()
INCOMPLETE=()
for model in "${MODELS[@]}"; do
    for s in "${SETS[@]}"; do
        f=$(pred_file "$model" "${s%%:*}")
        if complete "$f" "${s#*:}"; then FILES+=("$f"); else INCOMPLETE+=("$model/${s%%:*}"); fi
    done
done
if [ ${#INCOMPLETE[@]} -gt 0 ]; then
    echo "Incomplete after the run (re-run this script to resume): ${INCOMPLETE[*]}"
fi

python3 - "$OUT_DIR/provenance.json" "$START" "$END" "$RUN_CONFIG" "$SKIPPED" "${INCOMPLETE[*]:-}" \
    "$USAGE_BEFORE" "$USAGE_AFTER" "$(IFS=,; echo "${SETS[*]}")" "${FILES[@]}" <<'PY'
import hashlib, json, os, platform, subprocess, sys
out, start, end, cfg, skipped, incomplete, u0, u1, sets, *files = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def meta(p):
    d = json.load(open(p)); m = d.get("metadata", {})
    preds = d.get("predictions", [])
    usage = [x.get("usage") or {} for x in preds if isinstance(x, dict)]
    return {**{k: m.get(k) for k in ("model", "prompt_version", "decoder_version", "max_tokens", "reasoning_effort",
                                     "config_version", "config_overridden", "backend", "git_commit",
                                     "successful_predictions", "failed_predictions")},
            "test_set": (m.get("test_set_metadata") or {}).get("test_set_name"),
            "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 4),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage)}
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
num = lambda s: None if s in ("", "null") else float(s)
prev = json.load(open(out)) if os.path.exists(out) else {}
spend = prev.get("account_spend_usd", [])
if num(u0) is not None and num(u1) is not None and num(u1) > num(u0):
    spend.append({"started": start, "finished": end, "usd": round(num(u1) - num(u0), 4)})
test_sets = {}
for s in sets.split(","):
    label, path = s.split(":", 1)
    test_sets[label] = {"path": path, "sha256": sha(path),
                        "case_ids_sha256": json.load(open(path))["metadata"].get("case_ids_sha256")}
json.dump({
    "run": "v0.3 test", "spec": "spec/v0.3-scoring.md draft 2", "started": prev.get("started", start), "finished": end,
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "test_sets": test_sets,
    "run_config": {"path": cfg, "sha256": sha(cfg), "content": json.load(open(cfg))},
    "skipped_not_on_openrouter": skipped.split(), "incomplete": incomplete.split(),
    "account_spend_usd": spend,
    "predictions": [{"path": f, "sha256": sha(f), **meta(f)} for f in files],
    "python": platform.python_version(),
}, open(out, "w"), indent=1)
print(f"wrote {out}")
PY

if [ "$NO_SCORE" = "1" ]; then echo "NO_SCORE=1: skipping the scorer."; exit 0; fi
if [ ${#FILES[@]} -eq 0 ]; then echo "No complete prediction files; nothing to score."; exit 1; fi
python3 -m evaluator.v03_score --predictions "${FILES[@]}" \
    --out "$OUT_DIR/v03-scores.json" --report "$OUT_DIR/v03-report.md"
