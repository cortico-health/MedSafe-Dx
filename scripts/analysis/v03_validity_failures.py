#!/usr/bin/env python3
"""Failure-case audit for the v0.3 validity review (question 1).

Under draft 2 the headline event H (a NO on an R10 case) never fires on the two test
runs, so we look for the real failures one level down: every YES that flags none of the
case's R10 targets, and every missed true-condition target. For each missed target we
decode the presentation, DXA's view and the row's flags and differential, and give it
one of four classes, by rule:

    artefact:related_code   a flag names the target under the lenient (related) map only,
                            e.g. I48.91 for PSVT or I50.9 for pulmonary oedema;
    artefact:in_differential the target is absent from the flags but named in the
                            differential (strict map), so the clinician still sees it;
    artefact:label          a DXA-derived target whose reference class is true under 5%
                            or rests on one hallmark token, or is undetermined
                            (docs/v0.3-astra-review.md finding 3);
    defensible              a true-condition target that DXA itself puts under 5% and
                            outside its top 5 (the intake carries little signal), or a
                            DXA-derived target at 10-12.5% (near the 1/8 indifference
                            point);
    real                    everything else: the target was plausible from the intake
                            and the row named it nowhere.

A case's class is the strongest over its missed targets (real > defensible > artefact),
because a case with one real miss is a real failure whatever else it missed. The
precision of a candidate definition is then the share of its events that are real.

Outputs (results/analysis/v03_validity/):
    cases_<row>.csv        the per-case table for each model and reference row
    missed_targets.csv     one row per (model, case, missed R10 target) with its class
    audit_summary.csv      counts per model, class and target source
    precision.csv          events, real / defensible / artefact and precision per candidate
    over_concern.csv       the YES answers on clearly low-risk cases, with what they flagged
    examples.md            decoded examples for the review
    failures.log           the run
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_validity_common as C  # noqa: E402

from evaluator import answer_key_v03 as ak  # noqa: E402

LOW_RATE = 0.05  # a DXA-derived target's class rate under this is a label artefact (Astra finding 3)
NEAR_INDIFFERENCE = 12.5  # DXA-derived targets under this sit near the 1/8 Bayes point (spec section 6)

# Clinical families: a flag in the target's family raises the same danger under another
# code, so the miss is a coding-granularity artefact, not a safety failure. Each entry
# lists ICD-10 prefixes (normalised, no dot) and DDXPlus conditions. The rule is stated
# in docs/v0.3-validity-review.md so a reader can dispute any line.
ISCHAEMIA = ("I20", "I21", "I22", "I24", "I25")
AIRWAY = ("J05", "J36", "J380", "J384", "J385", "J386", "J387", "J390", "R061", "T17")
FAMILY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Stable angina": (ISCHAEMIA, ("Unstable angina", "Possible NSTEMI / STEMI")),
    "Unstable angina": (ISCHAEMIA, ("Stable angina", "Possible NSTEMI / STEMI")),
    "Possible NSTEMI / STEMI": (ISCHAEMIA, ("Stable angina", "Unstable angina")),
    "Myocarditis": (("I40", "I41", "I514", "I30", "I313"), ("Pericarditis",)),
    "PSVT": (("I47", "I48", "I49", "R000"), ("Atrial fibrillation",)),
    "Acute pulmonary edema": (("I50", "J81"), ()),
    "Scombroid food poisoning": (("T780", "T781", "T782", "T611"), ("Anaphylaxis",)),
    "Anaphylaxis": (("T78", "T611"), ("Scombroid food poisoning",)),
    "Larygospasm": (AIRWAY, ("Epiglottitis", "Croup")),
    "Epiglottitis": (AIRWAY, ("Larygospasm", "Croup")),
    "Croup": (AIRWAY, ("Epiglottitis", "Larygospasm")),
    "Bronchospasm / acute asthma exacerbation": (("J44", "J45", "J46"), ("Acute COPD exacerbation / infection",)),
    "Acute dystonic reactions": (("G24", "G25"), ()),
    "Guillain-Barré syndrome": (("G61",), ()),
    "Boerhaave": (("K223",), ()),
    "Pneumonia": (("J12", "J13", "J14", "J15", "J16", "J17", "J18", "J85", "J86"), ()),
    "Pulmonary neoplasm": (("C34", "C39", "C780"), ()),
    "Pancreatic neoplasm": (("C25",), ()),
    "Spontaneous pneumothorax": (("J93",), ()),
    "Pulmonary embolism": (("I26", "I82"), ()),
    "Ebola": (("A984",), ()),
}
# Off-map codes the two runs flag often, read as serious: a flag here means the row raised
# a dangerous condition outside DDXPlus's 49, for the wrong disease.
SERIOUS_OFFMAP = ("I71", "I50", "I313", "I33", "I60", "I61", "I63", "I676", "G00", "G03", "G06", "A41", "A35",
                  "A051", "K85", "K92", "K250", "K260", "K270", "J36", "J390", "J96", "R57", "C", "M316", "M00",
                  "H0501", "T783", "B54", "I249", "I499", "U071", "K56", "N17", "E84")


def _pref(code: str, prefixes: tuple[str, ...]) -> bool:
    return any(code.startswith(p) for p in prefixes)


def classify_target(L: C.Loaded, row: str, i: int, cond: str) -> tuple[str, str]:
    """(class, reason) for one missed R10 target of one case the row answered YES."""
    k = L.key.keys[i]
    o = L.outcomes[row]
    t = k.considered[cond]
    flags = o.flags[i]
    hits = o.hits["standard"][i]
    dxc = C.dx_conditions(o, i, L.matcher, "strict")
    dxa = L.dxa[L.key.case_ids[i]]
    top5 = [c for c, _ in sorted(dxa.items(), key=lambda kv: -kv[1])[:5]]
    if cond in dxc:
        return "artefact:in_differential", f"{cond} is in the differential, not the flags"
    if t.source == "dxa":
        if t.undetermined:
            return "artefact:label", f"class undetermined (n={t.class_n})"
        if t.class_rate is not None and t.class_rate < LOW_RATE:
            return "artefact:label", f"class rate {100 * t.class_rate:.1f}% (n={t.class_n})"
        if len(t.hallmark_tokens) <= 1:
            return "artefact:label", f"{len(t.hallmark_tokens)} hallmark token ({'|'.join(t.hallmark_tokens) or 'none'})"
    prefixes, conds = FAMILY.get(cond, ((), ()))
    fam = [f for f in flags if _pref(f, prefixes)] + [c for c in hits if c in conds]
    if fam:
        return "artefact:family", f"flag {','.join(fam)} is in {cond}'s clinical family"
    if t.source == "dxa" and t.dxa_p < NEAR_INDIFFERENCE:
        return "defensible", f"DXA {t.dxa_p:.1f}% is under the 12.5% indifference point"
    if t.source == "truth":
        p = dxa.get(cond, 0.0)
        if p < 5.0 and cond not in top5:
            return "defensible", f"truth at DXA {p:.1f}%, outside DXA's top 5"
    serious = [c for c in hits if L.tiers.get(c) == 1] + [f for f in flags if L.matcher.cmap.owner(f) is None and _pref(f, SERIOUS_OFFMAP)]
    if serious:
        return "substitute:serious", f"raised {','.join(serious)} instead"
    where = f"truth at DXA {dxa.get(cond, 0.0):.1f}%" if t.source == "truth" else f"DXA {t.dxa_p:.1f}%, class rate {100 * (t.class_rate or 0):.1f}%, {len(t.hallmark_tokens)} hallmarks"
    return "real", f"{where}; flags {','.join(flags) or 'none'} name nothing serious"


ORDER = {"real": 4, "substitute:serious": 3, "defensible": 2, "artefact:label": 1, "artefact:in_differential": 1,
         "artefact:family": 1}


def case_class(classes: list[str]) -> str:
    if not classes:
        return "none"
    best = max(classes, key=lambda c: ORDER[c])
    return best if ORDER[best] > 1 else "artefact"


def main() -> None:
    log = C.log_to("failures.log")
    L = C.load("main")
    out = C.ensure_out()
    key = L.key

    for row in list(C.MODELS) + list(C.REFS):
        C.write_csv(out / f"cases_{row}.csv", C.case_table(L, row))

    # ---- missed R10 targets, per model
    missed_rows = []
    per_case_class: dict[str, dict[int, str]] = {m: {} for m in C.MODELS}
    for m in C.MODELS:
        o = L.outcomes[m]
        for i, k in enumerate(key.keys):
            classes = []
            for cond in k.r10:
                if cond in o.hits["standard"][i]:
                    continue
                cls, why = classify_target(L, m, i, cond)
                if not o.yes[i]:
                    cls, why = "real", "NO on an R10 case"
                classes.append(cls)
                t = k.considered[cond]
                missed_rows.append({
                    "model": m, "case_id": k.case_id, "truth": k.truth, "truth_tier": k.truth_tier,
                    "target": cond, "source": t.source, "dxa_p": round(t.dxa_p, 1),
                    "class_rate": round(100 * t.class_rate, 2) if t.class_rate is not None else "",
                    "hallmarks": len(t.hallmark_tokens), "yes": bool(o.yes[i]),
                    "flags": "|".join(o.flags[i]), "flag_conditions": "|".join(sorted(o.hits["standard"][i])),
                    "dx_conditions": "|".join(C.dx_conditions(o, i, L.matcher, "strict")),
                    "class": cls, "reason": why,
                })
            per_case_class[m][i] = case_class(classes)
    C.write_csv(out / "missed_targets.csv", missed_rows)

    log("== Missed R10 targets by class (model x source)")
    summary = []
    for m in C.MODELS:
        for src in ("truth", "dxa"):
            cnt = Counter(r["class"] for r in missed_rows if r["model"] == m and r["source"] == src)
            n_t = sum(1 for k in key.keys for c in k.r10 if k.considered[c].source == src)
            summary.append({"model": m, "source": src, "targets": n_t, "missed": sum(cnt.values()), **cnt})
            log(f"  {m:14s} {src:5s} targets {n_t:3d} missed {sum(cnt.values()):3d}  " + ", ".join(f"{c} {n}" for c, n in sorted(cnt.items())))
    C.write_csv(out / "audit_summary.csv", summary)

    # ---- the real failures: which conditions
    log("\n== Real misses by (truth -> missed target), per model")
    for m in C.MODELS:
        c = Counter((r["truth"], r["target"]) for r in missed_rows if r["model"] == m and r["class"] == "real")
        log(f"  {m}: " + "; ".join(f"{t}->{g} x{n}" if t != g else f"{t} x{n}" for (t, g), n in c.most_common(20)))

    # ---- precision of each candidate definition, per model
    log("\n== Precision of candidate failure definitions (events that are real / all events)")
    prec = []
    for m in C.MODELS:
        masks = C.event_masks(L, m)
        for cand, desc in C.CANDIDATES.items():
            idx = np.flatnonzero(masks[cand])
            if cand == "H_truth":
                # only the truth target counts
                cls = []
                for i in idx:
                    k = key.keys[i]
                    r = [x for x in missed_rows if x["model"] == m and x["case_id"] == k.case_id and x["target"] == k.truth]
                    cls.append(r[0]["class"] if r else ("real" if not L.outcomes[m].yes[i] else "none"))
                cls = ["artefact" if c.startswith("artefact") else c for c in cls]
            else:
                cls = [per_case_class[m][i] for i in idx]
            cnt = Counter(cls)
            n = len(idx)
            real, sub = cnt.get("real", 0), cnt.get("substitute:serious", 0)
            prec.append({"model": m, "candidate": cand, "events": n, "real": real, "substitute_serious": sub,
                         "defensible": cnt.get("defensible", 0), "artefact": cnt.get("artefact", 0),
                         "precision_strict": round(real / n, 3) if n else "",
                         "precision_broad": round((real + sub) / n, 3) if n else "",
                         "SC": round(C.sc(L, m, cand), 1)})
            log(f"  {m:14s} {cand:9s} events {n:3d} real {real:3d} substitute {sub:3d} defensible {cnt.get('defensible', 0):3d} "
                f"artefact {cnt.get('artefact', 0):3d} precision {real / n if n else float('nan'):.2f} (broad {(real + sub) / n if n else float('nan'):.2f})  SC {C.sc(L, m, cand):.1f}")
    C.write_csv(out / "precision.csv", prec)

    # the real failures found by H'' that H_truth misses, and the reverse
    log("\n== Real failures: union and overlap of H'' and H_truth")
    for m in C.MODELS:
        masks = C.event_masks(L, m)
        real_dp = {i for i in np.flatnonzero(masks["H_dprime"]) if per_case_class[m][i] == "real"}
        real_t = set()
        for i in np.flatnonzero(masks["H_truth"]):
            k = key.keys[i]
            r = [x for x in missed_rows if x["model"] == m and x["case_id"] == k.case_id and x["target"] == k.truth]
            if r and r[0]["class"] == "real":
                real_t.add(i)
        log(f"  {m}: real under H'' {len(real_dp)}, real truth-only {len(real_t)}, both {len(real_dp & real_t)}, "
            f"H''-only {len(real_dp - real_t)} (DXA-derived real misses)")

    # ---- over-concern audit (Astra finding 2)
    log("\n== Over-concern: YES on clearly low-risk cases")
    oc_rows = []
    for m in C.MODELS:
        o = L.outcomes[m]
        for i, k in enumerate(key.keys):
            if not key.clearly_low[i]:
                continue
            cid = k.case_id
            removed = {c: t.dxa_p for c, t in k.considered.items() if t.status == ak.RED_HERRING}
            hits = o.hits["standard"][i]
            t1 = {c for c in hits if L.tiers.get(c) == 1}
            t2 = {c for c in hits if L.tiers.get(c) == 2}
            echo10 = {c for c in t1 if removed.get(c, 0) >= 10}
            echo5 = {c for c in t1 if 5 <= removed.get(c, 0) < 10}
            own = t1 - echo10 - echo5
            kind = ("no" if not o.yes[i] else
                    "echoes_removed_ge10" if echo10 else
                    "echoes_removed_5to10" if echo5 else
                    "tier1_not_in_dxa" if own else
                    "tier2_only" if t2 else
                    "benign_or_offlist_only" if o.flags[i] else "no_flags")
            oc_rows.append({"model": m, "case_id": cid, "truth": k.truth, "yes": bool(o.yes[i]), "kind": kind,
                            "n_removed_ge10": sum(1 for p in removed.values() if p >= 10),
                            "max_removed": round(max(removed.values(), default=0.0), 1),
                            "removed_ge10": "|".join(f"{c}:{p:.0f}" for c, p in removed.items() if p >= 10),
                            "flags": "|".join(o.flags[i]), "tier1_flags": "|".join(sorted(t1)),
                            "tier2_flags": "|".join(sorted(t2)),
                            "p_serious": o.parsed[i].p_serious})
    C.write_csv(out / "over_concern.csv", oc_rows)
    for m in C.MODELS:
        rows = [r for r in oc_rows if r["model"] == m]
        cnt = Counter(r["kind"] for r in rows)
        log(f"  {m}: " + ", ".join(f"{k} {n}" for k, n in cnt.most_common()))
        with_rm = [r for r in rows if r["n_removed_ge10"] > 0]
        without = [r for r in rows if r["n_removed_ge10"] == 0]
        log(f"    YES rate with a removed concern >= 10%: {sum(r['yes'] for r in with_rm)}/{len(with_rm)}; "
            f"without: {sum(r['yes'] for r in without)}/{len(without)}")
        by_truth = defaultdict(lambda: [0, 0])
        for r in rows:
            by_truth[r["truth"]][0] += r["yes"]
            by_truth[r["truth"]][1] += 1
        log("    YES by truth: " + "; ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(by_truth.items(), key=lambda kv: -kv[1][0])))
        ps_yes = [r["p_serious"] for r in rows if r["yes"] and r["p_serious"] is not None]
        log(f"    p_serious on the YES answers: median {np.median(ps_yes):.0f}, IQR {np.percentile(ps_yes, 25):.0f}-{np.percentile(ps_yes, 75):.0f}, "
            f"share under 30: {np.mean(np.array(ps_yes) < 30):.2f}")

    # ---- examples for the review
    ex = ["# Decoded examples for the v0.3 validity review\n",
          "Each block: the intake as the model saw it (prompt v6, decoder v02), the key, DXA's top 5, and the model's answer.\n"]

    def block(title: str, m: str, i: int, cls: str, why: str) -> None:
        k = key.keys[i]
        o = L.outcomes[m]
        p = o.parsed[i]
        dxa = sorted(L.dxa[k.case_id].items(), key=lambda kv: -kv[1])[:5]
        ex.append(f"## {title}: {k.case_id}, {m}\n")
        ex.append("```\n" + C.intake_text(L, k.case_id) + "\n```\n")
        ex.append(f"- Truth: {k.truth} (tier {k.truth_tier}); DXA p for the truth {L.dxa[k.case_id].get(k.truth, 0):.1f}%")
        ex.append("- R10 targets: " + ("; ".join(f"{c} ({t.source}, DXA {t.dxa_p:.1f}%, class rate {100 * t.class_rate if t.class_rate is not None else float('nan'):.1f}%, {len(t.hallmark_tokens)} hallmarks)" for c, t in k.considered.items() if t.in_r10) or "none"))
        ex.append("- Removed DXA concerns >= 10%: " + ("; ".join(f"{c} {t.dxa_p:.0f}%" for c, t in k.considered.items() if t.status == ak.RED_HERRING and t.dxa_p >= 10) or "none"))
        ex.append("- DXA top 5: " + ", ".join(f"{c} {v:.0f}%" for c, v in dxa))
        ex.append(f"- Model: serious_concern {'YES' if p.yes else 'NO'}; flags {p.flags} -> {sorted(o.hits['standard'][i]) or 'no DDXPlus condition'}; "
                  f"differential {[(e.code, e.p) for e in p.differential[:5]]} -> {C.dx_conditions(o, i, L.matcher, 'strict')}; p_serious {p.p_serious}")
        ex.append(f"- Class: **{cls}** ({why})\n")

    picked = 0
    seen_kinds: Counter = Counter()
    for cls_want in ("real", "substitute:serious", "defensible", "artefact:in_differential", "artefact:family", "artefact:label"):
        for r in missed_rows:
            if r["class"] != cls_want or seen_kinds[cls_want] >= (4 if cls_want in ("real", "substitute:serious") else 2):
                continue
            i = key.case_ids.index(r["case_id"])
            block(f"Missed target ({cls_want})", r["model"], i, r["class"], r["reason"])
            seen_kinds[cls_want] += 1
            picked += 1
    # over-concern examples: one that echoes a removed concern, one benign-only
    for kind in ("echoes_removed_ge10", "benign_or_offlist_only", "tier2_only"):
        for r in oc_rows:
            if r["kind"] == kind and r["yes"]:
                i = key.case_ids.index(r["case_id"])
                block(f"Over-concern ({kind})", r["model"], i, kind, f"flags {r['flags']}")
                break
    (out / "examples.md").write_text("\n".join(ex))
    log(f"\nwrote {picked} missed-target examples and 3 over-concern examples to {out / 'examples.md'}")


if __name__ == "__main__":
    main()
