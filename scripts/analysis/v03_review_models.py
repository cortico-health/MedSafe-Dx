"""v0.3 review, part 3: the 19 v0.1 rows on the 250-case v0 set, scored as if they were v0.3 runs.

Stand-ins: ESCALATE_NOW = serious_concern YES, ROUTINE_CARE = NO, unreadable stays unreadable;
the top-5 differential is the flag list. The v0.1 prompt asked a different question, so the
levels are indicative, not a forecast. What we learn: interval width, separation, how much of
H is the YES rate, what lenient matching changes, and how often YES comes with no tier-1 flag.

Run: .venv/bin/python scripts/analysis/v03_review_models.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from v03_review_common import (LENIENT, OUT, ROOT, STRICT, Matcher, RedHerring, build_key, cluster_bootstrap_h,
                               fmt, load_adults, load_tiers, sample_ids, score, write_csv)

sys.path.insert(0, str(ROOT / "scripts"))
from scripts.analysis import failure_shape as fs  # noqa: E402


def load_rows(ids: list[str]) -> dict[str, list[dict]]:
    pos = {cid: i for i, cid in enumerate(ids)}
    rows = {}
    for r in fs.load_rows():
        path, status = fs.resolve_predictions(r)
        if path is None:
            continue
        raw = json.loads(path.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        ans = [{"yes": None, "flags": []} for _ in ids]
        seen = set()
        for p in preds:
            cid = p.get("case_id") if isinstance(p, dict) else None
            if cid not in pos or cid in seen:
                continue
            seen.add(cid)
            esc = p.get("escalation_decision")
            yes = True if esc == "ESCALATE_NOW" else False if esc == "ROUTINE_CARE" else None
            codes = [d.get("code") for d in (p.get("differential_diagnoses") or []) if isinstance(d, dict) and d.get("code")]
            ans[pos[cid]] = {"yes": yes, "flags": codes[:5]}
        if len(seen) >= 200:
            rows[r["name"]] = ans
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_adults()
    tiers = load_tiers()["v03"]
    rh = RedHerring()
    m = Matcher()
    ids = sample_ids("250")
    key = build_key(df, ids, tiers, rh)
    n10 = sum(1 for c in key if c["R10"])
    print(f"250 set: R10 cases {n10}, R5 cases {sum(1 for c in key if c['R5'])}, clearly low {sum(1 for c in key if c['clearly_low'])}, "
          f"truth tier 1 {sum(1 for c in key if c['tier'] == 1)}")
    rows = load_rows(ids)
    print(f"model rows with per-case outputs on the 250 set: {len(rows)}")

    out = []
    per_case_h = {}
    for name, ans in rows.items():
        s = score(key, ans, m, LENIENT, tiers=tiers)
        ss = score(key, ans, m, STRICT, tiers=tiers)
        st = score(key, ans, m, LENIENT, target_field="truth_only", tiers=tiers)
        s5 = score(key, ans, m, LENIENT, target_field="R5", tiers=tiers)
        pt, lo, hi = cluster_bootstrap_h(key, ans)
        unread = sum(1 for a in ans if a["yes"] is None)
        unread_h = sum(1 for c, a in zip(key, ans) if c["R10"] and a["yes"] is None)
        per_case_h[name] = np.array([a["yes"] is not True for c, a in zip(key, ans) if c["R10"]])
        out.append({"model": name, "H": fmt(s["H"]), "H_lo": fmt(lo), "H_hi": fmt(hi), "H_width": fmt(hi - lo),
                    "H_truth_only": fmt(st["H"]), "H_R5": fmt(s5["H"]),
                    "unreadable_all": unread, "unreadable_in_H": unread_h, "H_events": s["H_ev"],
                    "COV_lenient": fmt(s["COV"]), "COV_strict": fmt(ss["COV"]),
                    "MT_truth_lenient": fmt(s["MT_truth"]), "MT_truth_strict": fmt(ss["MT_truth"]),
                    "MT_dxa_lenient": fmt(s["MT_dxa"]), "MT_dxa_strict": fmt(ss["MT_dxa"]),
                    "OC": fmt(s["OC"]), "CON": fmt(s["CON"]), "YES_rate": fmt(s["YES_rate"]),
                    "YES_without_tier1_flag": fmt(s["YES_no_tier1_flag"]), "YES_without_tier1_flag_strict": fmt(ss["YES_no_tier1_flag"])})
    out.sort(key=lambda r: float(r["H"]))
    write_csv(OUT / "models250.csv", out)
    cols = ["model", "H", "H_lo", "H_hi", "unreadable_in_H", "H_truth_only", "COV_lenient", "COV_strict", "MT_truth_lenient", "MT_truth_strict", "OC", "YES_rate", "YES_without_tier1_flag"]
    print(" | ".join(cols))
    for r in out:
        print(" | ".join(str(r[c]) for c in cols))

    # separation: paired differences with a condition bootstrap
    names = [r["model"] for r in out]
    conds = np.array([c["truth"] for c in key if c["R10"]])
    uniq = sorted(set(conds))
    rng = np.random.default_rng(20260923)
    B = 1000
    draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(B)]
    idx_by_cond = {u: np.where(conds == u)[0] for u in uniq}
    def boot_diff(a, b):
        d = []
        for dr in draws:
            sel = np.concatenate([idx_by_cond[uniq[i]] for i in dr])
            d.append(a[sel].mean() - b[sel].mean())
        d = np.array(d)
        return np.percentile(d, 2.5), np.percentile(d, 97.5)
    sep = 0
    pairs = 0
    adjacent_sep = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo, hi = boot_diff(per_case_h[names[i]], per_case_h[names[j]])
            pairs += 1
            if lo > 0 or hi < 0:
                sep += 1
            if j == i + 1:
                adjacent_sep.append((names[i], names[j], round(lo, 3), round(hi, 3)))
    print(f"\nmodel pairs separated at 95% (paired condition bootstrap): {sep} of {pairs}")
    print("adjacent ranks separated:", sum(1 for _, _, lo, hi in adjacent_sep if lo > 0 or hi < 0), "of", len(adjacent_sep))
    # correlation between H and YES rate
    h = np.array([float(r["H"]) for r in out]); y = np.array([float(r["YES_rate"]) for r in out])
    print(f"Spearman(H, YES rate) over models: {np.corrcoef(np.argsort(np.argsort(h)), np.argsort(np.argsort(y)))[0,1]:.2f}")

    # what the H events are: truth-target misses vs DXA-derived-target misses
    ev = Counter()
    for name, ans in rows.items():
        for c, a in zip(key, ans):
            if c["R10"] and a["yes"] is not True:
                ev["truth is tier 1" if c["tier"] == 1 else "DXA-derived only"] += 1
    print("H events across models by target kind:", dict(ev))
    dxa_only_cases = [c for c in key if c["R10"] and c["tier"] != 1]
    print("DXA-derived-only R10 cases on the 250 set:", len(dxa_only_cases), Counter(c["truth"] for c in dxa_only_cases).most_common(8))
    # per DXA-derived-only case: share of models saying NO
    no_share = []
    for c in dxa_only_cases:
        i = key.index(c)
        no_share.append((c["truth"], sorted(c["R10"]), round(np.mean([rows[n][i]["yes"] is not True for n in rows]), 2)))
    no_share.sort(key=lambda x: -x[2])
    print("share of models saying NO on DXA-derived-only cases (top 12):")
    for t in no_share[:12]:
        print("  ", t)


if __name__ == "__main__":
    main()
