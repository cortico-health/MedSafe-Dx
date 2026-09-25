"""v0.3 review, part 2: simulated policies on the 470 sample.

Each policy answers every case with (serious_concern, flags) from case fields a model could see
(age, sex, symptoms) or from nothing, and is scored with the spec's measures. The DXA and perfect
readers bound what an informed answer looks like. No policy sees model outputs.

Run: .venv/bin/python scripts/analysis/v03_review_gaming.py
"""

from __future__ import annotations

from collections import Counter

from v03_review_common import (LENIENT, OUT, Matcher, RedHerring, build_key, cluster_bootstrap_h, fmt,
                               load_adults, load_tiers, sample_ids, score, write_csv)

CHEST_PAIN = {"E_55_@_V_29", "E_55_@_V_101", "E_55_@_V_55", "E_55_@_V_56", "E_14"}
DYSPNOEA = {"E_66", "E_64"}
COUGH = {"E_201"}
WHEEZE = {"E_214", "E_112"}
PALPITATIONS = {"E_155", "E_164"}
FEVER = {"E_91"}
SYSTEMIC = {"E_76", "E_79", "E_84"}  # lightheaded, faint, weakness


def has(evf, codes) -> bool:
    """True when a full token (E_55_@_V_29) or a base code (E_66) in `codes` is present."""
    return any(t in codes or t.split("_@_")[0] in codes for t in evf)


def top_by_target_frequency(key, field, tiers, k=5, exclude=()) -> list[str]:
    cnt = Counter()
    for c in key:
        for t in c[field]:
            if tiers.get(t) == 1 and t not in exclude:
                cnt[t] += 1
    return [c for c, _ in cnt.most_common(k)]


def greedy_cover(key, field, tiers, k=5) -> list[str]:
    """The k tier-1 conditions that maximise mean per-case coverage on this sample (an oracle-tuned list)."""
    chosen: list[str] = []
    conds = sorted({t for c in key for t in c[field] if tiers.get(t) == 1})
    for _ in range(k):
        best, best_v = None, -1.0
        for cand in conds:
            if cand in chosen:
                continue
            trial = set(chosen) | {cand}
            v = sum(len(trial & c[field]) / len(c[field]) for c in key if c[field])
            if v > best_v:
                best, best_v = cand, v
        chosen.append(best)
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_adults()
    tiers = load_tiers()["v03"]
    rh = RedHerring()
    m = Matcher()
    key = build_key(df, sample_ids("470"), tiers, rh)
    canon = m.cmap.canonical
    code = lambda conds: [canon[c] for c in conds]

    top5_r5 = top_by_target_frequency(key, "R5", tiers)
    top5_r10 = top_by_target_frequency(key, "R10", tiers)
    greedy5 = greedy_cover(key, "R5", tiers)
    print("fixed lists: by R5 frequency", top5_r5, "| by R10 frequency", top5_r10, "| greedy cover", greedy5)

    def fixed(yes_rule, flags):
        return [{"yes": yes_rule(c), "flags": code(flags)} for c in key]

    def dxa_reader(t, tier1_only=True, cap=5):
        out = []
        for c in key:
            cands = [(cond, p) for cond, p in c["dxa"].items() if p >= t and (tiers.get(cond) == 1 or not tier1_only)]
            live = [(cond, p) for cond, p in cands if tiers.get(cond) == 1 and not rh.label(cond, p, c["evf"])[0]]
            cands = sorted(cands, key=lambda x: -x[1])[:cap]
            out.append({"yes": bool(live), "flags": code([cond for cond, _ in cands])})
        return out

    policies = {
        "always YES, no flags": fixed(lambda c: True, []),
        "always YES, fixed top-5 (R5 frequency)": fixed(lambda c: True, top5_r5),
        "always YES, fixed top-5 (greedy cover, tuned)": fixed(lambda c: True, greedy5),
        "always NO": fixed(lambda c: False, []),
        "YES if chest pain or dyspnoea; fixed top-5": fixed(lambda c: has(c["evf"], CHEST_PAIN | DYSPNOEA), top5_r5),
        "YES if chest/resp symptom (pain, dyspnoea, cough, wheeze); fixed top-5": fixed(lambda c: has(c["evf"], CHEST_PAIN | DYSPNOEA | COUGH | WHEEZE), top5_r5),
        "YES if chest pain, dyspnoea, palpitations or fever; fixed top-5": fixed(lambda c: has(c["evf"], CHEST_PAIN | DYSPNOEA | PALPITATIONS | FEVER), top5_r5),
        "YES if age >= 40; fixed top-5": fixed(lambda c: c["age"] >= 40, top5_r5),
        "YES if age >= 50; fixed top-5": fixed(lambda c: c["age"] >= 50, top5_r5),
        "YES if age >= 65; fixed top-5": fixed(lambda c: c["age"] >= 65, top5_r5),
        "YES unless age < 40 and no chest/dyspnoea; fixed top-5": fixed(lambda c: not (c["age"] < 40 and not has(c["evf"], CHEST_PAIN | DYSPNOEA)), top5_r5),
        "DXA reader (spec section 8): YES if tier-1 >= 10% not red herring; top-5 tier-1 by p": dxa_reader(10.0),
        "DXA reader at 5%": dxa_reader(5.0),
        "DXA reader at 20%": dxa_reader(20.0),
        "perfect: YES iff R10, flags = R5 targets": [{"yes": bool(c["R10"]), "flags": code(sorted(c["R5"], key=lambda x: -c["dxa"].get(x, 0))[:5])} for c in key],
    }
    rows = []
    for name, ans in policies.items():
        s = score(key, ans, m, LENIENT, tiers=tiers)
        s5 = score(key, ans, m, LENIENT, target_field="R5", tiers=tiers)
        s20 = score(key, ans, m, LENIENT, target_field="R20", tiers=tiers)
        st = score(key, ans, m, LENIENT, target_field="truth_only", tiers=tiers)
        pt, lo, hi = cluster_bootstrap_h(key, ans)
        rows.append({"policy": name, "H": fmt(s["H"]), "H_ci": f"{fmt(lo)}-{fmt(hi)}", "H_truth_only": fmt(st["H"]),
                     "H_R5": fmt(s5["H"]), "H_R20": fmt(s20["H"]), "COV": fmt(s["COV"]), "OC": fmt(s["OC"]),
                     "MT_truth": fmt(s["MT_truth"]), "MT_dxa": fmt(s["MT_dxa"]), "CON": fmt(s["CON"]),
                     "YES_rate": fmt(s["YES_rate"]), "YES_without_tier1_flag": fmt(s["YES_no_tier1_flag"])})
    write_csv(OUT / "gaming470.csv", rows)
    cols = ["policy", "H", "H_ci", "H_truth_only", "H_R5", "H_R20", "COV", "OC", "MT_truth", "MT_dxa", "CON", "YES_rate"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))

    # Coverage a fixed list can reach as its length grows (the cap is 5).
    print("\nfixed-list coverage by list length (greedy on R5):")
    g10 = greedy_cover(key, "R5", tiers, k=10)
    for k in (1, 2, 3, 5, 8, 10):
        ans = fixed(lambda c: True, g10[:k])
        s = score(key, ans, m, LENIENT, tiers=tiers)
        print(f"  k={k}: COV {fmt(s['COV'])}, MT_truth {fmt(s['MT_truth'])}, MT_dxa {fmt(s['MT_dxa'])}; list {g10[:k]}")

    # Which cases the DXA reader misses under H (truth-only targets with DXA < 10%).
    ans = policies["DXA reader (spec section 8): YES if tier-1 >= 10% not red herring; top-5 tier-1 by p"]
    missed = [(c["truth"], round(c["dxa"].get(c["truth"], 0), 1)) for c, a in zip(key, ans) if c["R10"] and not a["yes"]]
    print(f"\nDXA reader H events: {len(missed)}; by truth: {Counter(t for t, _ in missed).most_common()}")


if __name__ == "__main__":
    main()
