#!/bin/bash
# MedSafe-Dx v0.3 draft-3 prompt test (spec/v0.3-scoring.md section 12).
#
# Runs every model on the 150-case prompt-test set (data/test_sets/eval-v03-ab150.json)
# under each prompt v7 arm, with inference/run_config_v03_ab.json (max_tokens 16000,
# reasoning effort "medium" where supported, one retry on an empty or truncated
# response, per-prediction finish reason, provider, request id and token usage), then
# scores every file at once with evaluator/v03b_score.py.
#
# Rules the script keeps:
# 1. One writer per prediction file: run_inference holds <file>.lock for the whole run,
#    and the scorer refuses a file whose lock is held. We score only after every worker exits.
# 2. A file that already covers its cases is not re-run; a partial file resumes.
# 3. The small Llama is meta-llama/llama-3.1-8b-instruct; if OpenRouter does not list it,
#    we run meta-llama/llama-3.3-70b-instruct and name the swap in the provenance.
# 4. Provenance (git commit, input sha256s, config, per-file sha256 and token cost, account
#    spend) goes to $OUT_DIR/provenance.json.
#
# Usage:
#   LIMIT=10 OUT_DIR=results/v03/ab/smoke CONFIRM=yes ./scripts/run_v03_ab.sh   # the smoke
#   CONFIRM=yes ./scripts/run_v03_ab.sh                                          # the 150 x arms x models
#   ARMS_OVERRIDE="v7a4a v7a4b" MODELS_OVERRIDE="a/b" NO_SCORE=1 ...
#   RUN_CONFIG=inference/run_config_v03_abj.json PROVENANCE=results/v03/ab/runs/provenance-4j.json \
#     ARMS_OVERRIDE="v7a4aj v7a4bj" MODELS_OVERRIDE="..." NO_SCORE=1 CONFIRM=yes ./scripts/run_v03_ab.sh
#   (amendment A1; a separate provenance file keeps the five-arm run's record)
#   CASES=data/test_sets/eval-v03-phase2.json OUT_DIR=results/phase2/runs RUN_CONFIG=inference/run_config_v03_abj.json \
#     RUN_LABEL="v0.3 Phase 2" ARMS_OVERRIDE="v7a4aj v7a4bj" MODELS_OVERRIDE="..." NO_SCORE=1 CONFIRM=yes ./scripts/run_v03_ab.sh
#   (the Phase 2 model run, docs/v0.3-case-selection-rules.md section 7; scored by scripts/analysis/v03_phase2_scores.py)

set -euo pipefail
cd "$(dirname "$0")/.."
export MEDSAFE_GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

RUN_CONFIG="${RUN_CONFIG:-inference/run_config_v03_ab.json}"
CASES="${CASES:-data/test_sets/eval-v03-ab150.json}"
OUT_DIR="${OUT_DIR:-results/v03/ab/runs}"
PROVENANCE="${PROVENANCE:-$OUT_DIR/provenance.json}"
PARALLEL="${PARALLEL:-8}"
export INFERENCE_WORKERS="${INFERENCE_WORKERS:-8}"
export PYTHONUNBUFFERED=1
LIMIT="${LIMIT:-}"
NO_SCORE="${NO_SCORE:-0}"

ARMS=(v7a1 v7a2 v7a3 v7a4a v7a4b)
[ -n "${ARMS_OVERRIDE:-}" ] && read -r -a ARMS <<< "$ARMS_OVERRIDE"
ROSTER=("openai/gpt-5.6-terra" "openai/gpt-oss-120b" "anthropic/claude-haiku-4.5" "meta-llama/llama-3.1-8b-instruct")
[ -n "${MODELS_OVERRIDE:-}" ] && read -r -a ROSTER <<< "$MODELS_OVERRIDE"
[ -f "$CASES" ] || { echo "Missing $CASES; build it with scripts/build_v03_ab_set.py" >&2; exit 1; }
mkdir -p "$OUT_DIR"

AVAILABLE=$(python3 - "${ROSTER[@]}" <<'PY'
import json, sys, urllib.request
roster = sys.argv[1:]
with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
    listed = {m["id"] for m in json.load(r)["data"]}
for m in roster:
    if m == "meta-llama/llama-3.1-8b-instruct" and m not in listed:
        print("SWAP: llama-3.1-8b-instruct not listed; running llama-3.3-70b-instruct", file=sys.stderr)
        m = "meta-llama/llama-3.3-70b-instruct"
    print(m) if m in listed else print(f"SKIP: {m} is not listed on OpenRouter", file=sys.stderr)
PY
)
read -r -a MODELS <<< "$(echo $AVAILABLE)"

pred_file() { echo "$OUT_DIR/$(echo "$1" | sed 's/\//-/g')-$2.json"; }

complete() {  # file covers the (first LIMIT) cases with no errored entry
    [ -f "$1" ] && python3 - "$1" "$CASES" "${LIMIT:-0}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); preds = p["predictions"] if isinstance(p, dict) else p
c = json.load(open(sys.argv[2]))["cases"]; n = int(sys.argv[3]) or len(c)
have = {x.get("case_id") for x in preds if isinstance(x, dict) and "error" not in x}
sys.exit(0 if {x["case_id"] for x in c[:n]} <= have else 1)
PY
}

or_usage() {
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

echo "MedSafe-Dx v0.3 prompt test | ${#MODELS[@]} models x ${#ARMS[@]} arms | limit ${LIMIT:-none} | commit $MEDSAFE_GIT_COMMIT"
TODO=()
for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do complete "$(pred_file "$m" "$a")" || TODO+=("$m|$a"); done; done

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
USAGE_BEFORE=$(or_usage)
if [ ${#TODO[@]} -gt 0 ]; then
    if [ "${CONFIRM:-}" != "yes" ]; then
        read -r -p "This sends paid requests for ${#TODO[@]} model x arm runs. Type yes to continue: " ans
        [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
    fi
    LOG_DIR="$OUT_DIR/logs"; mkdir -p "$LOG_DIR"
    pids=()
    for item in "${TODO[@]}"; do
        m="${item%%|*}"; a="${item#*|}"; f=$(pred_file "$m" "$a")
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 3; done
        echo "start: $m / $a -> $f"
        ( python3 -m inference.run_inference --prompt-version "$a" --run-config "$RUN_CONFIG" --cases "$CASES" \
              --model "$m" --out "$f" --workers "$INFERENCE_WORKERS" ${LIMIT:+--limit "$LIMIT"} \
              > "$LOG_DIR/$(basename "$f" .json).log" 2>&1 || echo "FAILED: $m / $a (see $LOG_DIR)" ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
fi
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
USAGE_AFTER=$(or_usage)

FILES=(); INCOMPLETE=()
for m in "${MODELS[@]}"; do for a in "${ARMS[@]}"; do
    f=$(pred_file "$m" "$a"); if complete "$f"; then FILES+=("$f"); else INCOMPLETE+=("$m/$a"); fi
done; done
[ ${#INCOMPLETE[@]} -gt 0 ] && echo "Incomplete (re-run to resume): ${INCOMPLETE[*]}"

python3 - "$PROVENANCE" "$START" "$END" "$RUN_CONFIG" "$CASES" "${LIMIT:-}" "${INCOMPLETE[*]:-}" \
    "$USAGE_BEFORE" "$USAGE_AFTER" "${ROSTER[*]}" "${MODELS[*]}" "${FILES[@]}" <<'PY'
import hashlib, json, os, platform, subprocess, sys
out, start, end, cfg, cases, limit, incomplete, u0, u1, roster, models, *files = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def meta(p):
    d = json.load(open(p)); m = d.get("metadata", {}); preds = d.get("predictions", [])
    usage = [x.get("usage") or {} for x in preds if isinstance(x, dict)]
    return {**{k: m.get(k) for k in ("model", "prompt_version", "decoder_version", "max_tokens", "reasoning_effort",
                                     "config_version", "config_overridden", "git_commit", "successful_predictions",
                                     "failed_predictions")},
            "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 4),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage)}
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
num = lambda s: None if s in ("", "null") else float(s)
prev = json.load(open(out)) if os.path.exists(out) else {}
spend = prev.get("account_spend_usd", [])
if num(u0) is not None and num(u1) is not None and num(u1) > num(u0):
    spend.append({"started": start, "finished": end, "usd": round(num(u1) - num(u0), 4)})
design = cases.replace(".json", ".design.csv")
json.dump({"run": os.getenv("RUN_LABEL", "v0.3 prompt test"), "spec": os.getenv("RUN_SPEC", "spec/v0.3-scoring.md draft 3 section 12"),
           "limit": limit or None,
           "started": prev.get("started", start), "finished": end,
           "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
           "cases": {"path": cases, "sha256": sha(cases), "design_sha256": sha(design) if os.path.exists(design) else None},
           "run_config": {"path": cfg, "sha256": sha(cfg), "content": json.load(open(cfg))},
           "roster": roster.split(), "models_run": models.split(), "incomplete": incomplete.split(),
           "account_spend_usd": spend, "predictions": [{"path": f, "sha256": sha(f), **meta(f)} for f in files],
           "python": platform.python_version()}, open(out, "w"), indent=1)
print(f"wrote {out}")
PY

[ "$NO_SCORE" = "1" ] && { echo "NO_SCORE=1: skipping the scorer."; exit 0; }
[ ${#FILES[@]} -eq 0 ] && { echo "No complete prediction files."; exit 1; }
python3 -m evaluator.v03b_score --predictions "${FILES[@]}" ${LIMIT:+--limit "$LIMIT"} \
    --out "$OUT_DIR/ab-scores.json" --report "$OUT_DIR/ab-report.md"
