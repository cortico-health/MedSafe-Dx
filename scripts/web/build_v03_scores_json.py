#!/usr/bin/env python3
"""Write the v0.3 board's data file from the full-run scores.

The board at / (web/static/leaderboard.html) fetches data/v03-scores.json.
We cut results/v03_full/scores.json down to the figures the board shows, so
the page loads a small file and every figure on it traces to one source.

Usage:
    python3 scripts/web/build_v03_scores_json.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "v03_full" / "scores.json"
OUT = ROOT / "web" / "static" / "data" / "v03-scores.json"
SOURCE_COMMIT = "4ac4c79"  # the commit that scored the full run

HEADLINE, SECONDARY = "4aj", "4bj"
REFERENCE_ROWS = [
    ("zero point (A2)", "zero_point"),
    ("naive Bayes", "naive_bayes"),
    ("always-routine", "always_routine"),
]


def metric(row, key):
    """One metric as {value, ci} with one decimal place, as scores.md prints it."""
    m = row[key]
    return {"value": round(m["value"], 1), "ci": [round(x, 1) for x in m["ci"]]}


def miss_equivalent(row, partial):
    """U + P/7 in % of SERIOUS, with U's interval shifted by P/7.

    The balanced cost is O + 7U + P (evaluator/v03_valid_reason.py, `_balanced_cost`), so on the
    axes (O, U + P/7) every line of equal score is straight. The interval is U's, shifted: an
    approximation, because the scores file keeps no draws of U + P/7."""
    shift = partial / 7
    return {"value": round(row["U"]["value"] + shift, 2),
            "ci": [round(x + shift, 2) for x in row["U"]["ci"]]}


def serious_cost(row, serious, benign):
    """7U + P in % of SERIOUS for a row whose scores file gives only the sample-mix cost.

    The mix cost is (7 x misses + partials + over-escalations) over all cases; costs are whole
    numbers per case, so we round the SERIOUS cost to a whole count before dividing."""
    total = row["cost"]["value"] / 100 * (serious + benign)
    count = round(total - row["O"]["value"] / 100 * benign)
    return 100 * count / serious


def main():
    src = json.loads(SRC.read_text())
    within = src["headline_within_condition"]
    boot = src["condition_bootstrap"]

    models = sorted({k.split("|")[0] for k in within["rows"]})
    rows = []
    for model in models:
        a = within["rows"][f"{model}|{HEADLINE}"]
        b = within["rows"][f"{model}|{SECONDARY}"]
        rows.append({
            "model": model,
            "score_4aj": metric(a, "score_z_bal"),
            "score_4aj_condition_bootstrap_ci": metric(boot["rows"][f"{model}|{HEADLINE}"], "score_z_bal")["ci"],
            "U": metric(a, "U"),
            "O": metric(a, "O"),
            "partial": {
                "total": round(a["partial"]["value"], 1),
                "in_list": round(a["partial_inlist"]["value"], 1),
                "off_list": round(a["partial_offlist"]["value"], 1),
                "truth": round(a["partial_truth"]["value"], 1),
            },
            "escalated": round(a["esc"]["value"], 1),
            "miss_equivalent": miss_equivalent(a, a["partial"]["value"]),
            "score_4bj": metric(b, "score_z_bal"),
            "parse_4bj_unreadable": src["parse"][f"{model}|{SECONDARY}"]["unreadable"],
        })
    rows.sort(key=lambda r: -r["score_4aj"]["value"])

    refs = []
    for name, rid in REFERENCE_ROWS:
        r = src["reference_rows"][name]
        partial = max(0.0, serious_cost(r, within["serious"], within["benign"]) - 7 * r["U"]["value"])
        refs.append({
            "id": rid,
            "score": metric(r, "score_z_bal"),
            "U": metric(r, "U"),
            "O": metric(r, "O"),
            "escalated": round(r["esc"]["value"], 1),
            "partial": round(partial, 1),
            "miss_equivalent": miss_equivalent(r, partial),
        })
    # The zero point's balanced cost: every model's score is 100 x (1 - (O + 7U + P) / this).
    zero = src["reference_rows"]["zero point (A2)"]
    zero_cost_bal = zero["O"]["value"] + serious_cost(zero, within["serious"], within["benign"])

    # model_pairs keys read "<arm>|<model A>|<model B>"; a pair is unseparated
    # when its within-condition difference interval includes 0.
    def unseparated(arm):
        return [
            key.split("|")[1:]
            for key, p in within["model_pairs"].items()
            if key.startswith(arm + "|") and not p["separated"]
        ]

    anchor = src["anchor_test"]
    s = src["set"]
    data = {
        "source": {
            "file": "results/v03_full/scores.json",
            "commit": SOURCE_COMMIT,
            "built_by": "scripts/web/build_v03_scores_json.py",
        },
        "set": {
            "cases": s["cases"],
            "seed": s["seed"],
            "strata": s["strata"],
            "public_twins": s["twins"],
            "serious": within["serious"],
            "benign": within["benign"],
            "zero_point": within["zero_reference"],
            "zero_cost_bal": round(zero_cost_bal, 3),
        },
        "models": rows,
        "reference_rows": refs,
        "pair_separation": {
            arm: {
                "separated": src["pair_separation"][arm]["within"],
                "separated_condition_bootstrap": src["pair_separation"][arm]["condition_bootstrap"],
                "pairs": src["pair_separation"][arm]["pairs"],
                "not_separated": unseparated(arm),
            }
            for arm in (HEADLINE, SECONDARY)
        },
        "anchor_test": {
            "models_surviving_holm": sum(1 for v in anchor.values() if v["reject"]),
            "models": len(anchor),
        },
        "spend_usd": round(src["account_spend_usd"], 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} models, {len(refs)} reference rows")


if __name__ == "__main__":
    main()
