#!/bin/bash
# MedSafe-Dx v0.2 run (spec/v0.2-scoring.md revision 4.2, section 9).
#
# Runs the roster on the frozen 470-case adult sample with prompt v5 and
# inference/run_config_v02.json, then scores every row at once. We score only
# after every inference worker has exited, because the board's bootstrap pairs
# all rows on the same draws and a half-written file would change a sha256 the
# provenance already recorded.
#
# Rules the script keeps:
# 1. One writer per prediction file: run_inference holds an exclusive lock on
#    <file>.lock for the whole run, and the scorer refuses a file whose lock is held.
# 2. A model whose prediction file already covers all 470 cases is not re-run;
#    a partial file resumes (errored cases are re-run).
# 3. A model OpenRouter no longer lists is skipped and named in the provenance.
# 4. Provenance (git commit, input sha256s, config, timings, per-file sha256) goes
#    to results/v02/runs/provenance.json.
#
# Usage:
#   ./scripts/run_v02.sh                    # paid run of the whole roster (asks for confirmation)
#   DRY_RUN=1 ./scripts/run_v02.sh          # render every request, send nothing, no spend
#   MODELS_OVERRIDE="a/b c/d" ./scripts/run_v02.sh   # a subset of the roster
#   RUNNER=local ./scripts/run_v02.sh       # system python3 instead of docker compose
#   PARALLEL=4 INFERENCE_WORKERS=8 ./scripts/run_v02.sh
#
# Cost: run `python3 scripts/estimate_v02_cost.py` after a dry run.

set -euo pipefail
cd "$(dirname "$0")/.."
export MEDSAFE_GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

TEST_SET="data/test_sets/eval-v02-adult.json"
RUN_CONFIG="inference/run_config_v02.json"
OUT_DIR="results/v02/runs"
LABEL="v02-470cases"
PARALLEL="${PARALLEL:-4}"
export INFERENCE_WORKERS="${INFERENCE_WORKERS:-8}"
RUNNER="${RUNNER:-docker}"
DRY_RUN="${DRY_RUN:-0}"

# Roster: the 23 rows of the v0.1 board (leaderboard/*-eval.json) that OpenRouter
# still lists. Gone on 2026-09-23: google/gemini-3-pro-preview, openai/gpt-5-chat.
ROSTER=(
    "anthropic/claude-fable-5"
    "anthropic/claude-opus-5"
    "anthropic/claude-opus-4.7"
    "anthropic/claude-sonnet-4.6"
    "anthropic/claude-haiku-4.5"
    "openai/gpt-6-astra"
    "openai/gpt-5.6-sol"
    "openai/gpt-5.6-terra"
    "openai/gpt-5.6-luna"
    "openai/gpt-5.4-mini"
    "openai/gpt-5.2"
    "openai/gpt-5-mini"
    "openai/o3-pro"
    "openai/gpt-oss-120b"
    "google/gemini-3.1-pro-preview"
    "moonshotai/kimi-k3"
    "z-ai/glm-5.3"
    "x-ai/grok-4.6"
    "x-ai/grok-4.20"
    "deepseek/deepseek-r1"
    "meta-llama/llama-4-maverick"
)
if [ -n "${MODELS_OVERRIDE:-}" ]; then
    read -r -a ROSTER <<< "$MODELS_OVERRIDE"
fi

if [ ! -f "$TEST_SET" ]; then
    echo "Missing $TEST_SET; build it with scripts/prep_v02_sample.py" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

# Pre-flight: every roster model must be in the run config (otherwise its row is
# flagged config_overridden) and listed on OpenRouter. The model list is a free GET.
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

pred_file() { echo "$OUT_DIR/$(echo "$1" | sed 's/\//-/g')-$LABEL.json"; }

complete() {
    [ -f "$1" ] && python3 - "$1" "$TEST_SET" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); preds = p["predictions"] if isinstance(p, dict) else p
c = json.load(open(sys.argv[2]))["cases"]
have = {x.get("case_id") for x in preds if isinstance(x, dict) and "error" not in x}
sys.exit(0 if {x["case_id"] for x in c} <= have else 1)
PY
}

echo "MedSafe-Dx v0.2 | $TEST_SET | ${#MODELS[@]} models | commit $MEDSAFE_GIT_COMMIT | dry run: $DRY_RUN"
[ -n "$SKIPPED" ] && echo "Skipped (not on OpenRouter): $SKIPPED"

if [ "$DRY_RUN" = "1" ]; then
    for model in "${MODELS[@]}"; do
        run_py -m inference.run_inference --prompt-version v5 --run-config "$RUN_CONFIG" \
            --cases "$TEST_SET" --model "$model" --out "$(pred_file "$model")" --dry-run | tail -1
    done
    echo "Dry run done: request bodies in $OUT_DIR/*.dryrun.json. Nothing was sent."
    exit 0
fi

if [ "${CONFIRM:-}" != "yes" ]; then
    read -r -p "This sends paid requests for ${#MODELS[@]} models x 470 cases. Type yes to continue: " ans
    [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
fi

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG_DIR="$OUT_DIR/logs"; mkdir -p "$LOG_DIR"
pids=()
for model in "${MODELS[@]}"; do
    f=$(pred_file "$model")
    if complete "$f"; then
        echo "complete, skipping: $model"
        continue
    fi
    # Bound concurrency: wait for a slot.
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
    echo "start: $model -> $f"
    ( run_py -m inference.run_inference --prompt-version v5 --run-config "$RUN_CONFIG" \
          --cases "$TEST_SET" --model "$model" --out "$f" --workers "$INFERENCE_WORKERS" \
          > "$LOG_DIR/$(basename "$f" .json).log" 2>&1 \
      || echo "FAILED: $model (see $LOG_DIR)" ) &
    pids+=($!)
done
# Score only after every worker has exited.
for p in "${pids[@]}"; do wait "$p" || true; done
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

FILES=()
INCOMPLETE=()
for model in "${MODELS[@]}"; do
    f=$(pred_file "$model")
    if complete "$f"; then FILES+=("$f"); else INCOMPLETE+=("$model"); fi
done
if [ ${#INCOMPLETE[@]} -gt 0 ]; then
    echo "Incomplete after the run (re-run this script to resume): ${INCOMPLETE[*]}"
fi

python3 - "$OUT_DIR/provenance.json" "$START" "$END" "$TEST_SET" "$RUN_CONFIG" "$SKIPPED" "${INCOMPLETE[*]:-}" "${FILES[@]}" <<'PY'
import hashlib, json, platform, subprocess, sys
out, start, end, test_set, cfg, skipped, incomplete, *files = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def meta(p):
    m = json.load(open(p)).get("metadata", {})
    return {k: m.get(k) for k in ("model", "prompt_version", "decoder_version", "max_tokens", "reasoning_effort",
                                   "config_version", "config_overridden", "backend", "git_commit",
                                   "successful_predictions", "failed_predictions")}
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
json.dump({
    "run": "v0.2", "spec": "spec/v0.2-scoring.md revision 4.2", "started": start, "finished": end,
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "test_set": {"path": test_set, "sha256": sha(test_set)},
    "case_ids_sha256": json.load(open(test_set))["metadata"]["case_ids_sha256"],
    "run_config": {"path": cfg, "sha256": sha(cfg), "content": json.load(open(cfg))},
    "skipped_not_on_openrouter": skipped.split(), "incomplete": incomplete.split(),
    "predictions": [{"path": f, "sha256": sha(f), **meta(f)} for f in files],
    "python": platform.python_version(),
}, open(out, "w"), indent=1)
print(f"wrote {out}")
PY

if [ ${#FILES[@]} -eq 0 ]; then echo "No complete prediction files; nothing to score."; exit 1; fi
python3 -m evaluator.v02_score --cases "$TEST_SET" --predictions "${FILES[@]}" --out leaderboard/v02-scores.json
echo "Board data: leaderboard/v02-scores.json (served at /v02-scores.json; it replaces the SYNTHETIC preview)."
