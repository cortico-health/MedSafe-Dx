#!/usr/bin/env python3
"""
Score arms 4aj and 4bj of the v0.3 prompt test (spec/v0.3-scoring.md amendment A1) under the
valid-reason rule (evaluator/v03_valid_reason.py), and write results/v03/ab/ab-4j-scores.json
plus generated tables (ab-4j-tables.md) for results/v03/ab/ab-4j-report.md. No model is called.

We score the 7-model roster on the same 150 cases, with:

1. partial cost 1 (sensitivity rows at 2 and 3.5) and the pair row of
   docs/wrong-serious-condition-cost.md (Boerhaave escalated under a non-surgical label costs 3.5);
2. off-list validity from spec/offlist_tiers_nhamcs.csv (primary-diagnosis NHAMCS rates, tier 1
   only), with the all-valid and all-invalid bounds and a row without the weak-evidence tier-1 rows;
3. both weightings (sample mix and balanced 50/50), U, O, the partial rate split in-list / off-list,
   and the escalation share, each with a within-condition 95% interval (evaluator/v03_stats.py);
   the condition bootstrap is kept for the two scores as the superpopulation sensitivity;
4. the reference rows that could define 0: always escalate with one fixed flag on the most common
   tier-1 target, the committed-five differential, always routine, the DXA reader and naive Bayes;
   every row is also rescaled so the fixed single flag scores 0;
5. paired comparisons: 4aj - 4bj per model, 4aj - 4a and 4bj - 4b for the models that ran both,
   and every model pair within each arm;
6. a descriptive audit of the justification sentence, which the scorer never reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import v03_score as vs  # noqa: E402
from evaluator import v03_stats as st  # noqa: E402
from evaluator import v03_valid_reason as vr  # noqa: E402
from evaluator import v03b_score as sb  # noqa: E402
from evaluator.schemas_v03b import normalise_code  # noqa: E402

AB = ROOT / "results" / "v03" / "ab"
RUNS = AB / "runs"
JARMS = ("v7a4aj", "v7a4bj")
BASE = {"v7a4aj": "v7a4a", "v7a4bj": "v7a4b"}
MODELS = ("openai/gpt-5.6-terra", "openai/gpt-oss-120b", "anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview",
          "z-ai/glm-5.3", "anthropic/claude-haiku-4.5", "meta-llama/llama-3.1-8b-instruct")
MEASURES = ("score_mix", "score_bal", "U", "O", "partial", "partial_inlist", "partial_offlist", "pass", "bare", "esc",
            "score_mix@2", "score_bal@2", "score_mix@3.5", "score_bal@3.5", "score_mix@pair", "score_bal@pair")
PAIRED = ("score_mix", "score_bal", "U", "O", "partial", "esc")
ICD_ORDER = ROOT / "data" / "external" / "icd10cm" / "icd10cm_order_2026.txt"


def file_for(model: str, arm: str) -> Path:
    return RUNS / f"{model.replace('/', '-')}-{arm}.json"


# ---------------------------------------------------------------- one row


def score_row(a, ab, rule, rules_extra, W, kcase, Mc):
    """(summary, within draws, point, outcome) for one row, plus bounds and the condition-bootstrap scores."""
    o = vr.outcomes(a, ab, rule)
    extra = {"pair": vr.pair_partials(o, ab.key.truth)}
    point, draws = vr.stats(o, ab, W, extra=extra, key=kcase)
    summ = vr.summarise(point, draws)
    out = {"measures": {m: summ[m] for m in MEASURES}}
    cp, cd = vr.stats(o, ab, Mc, extra=extra)
    cs = vr.summarise(cp, cd)
    out["condition_bootstrap"] = {m: cs[m] for m in ("score_mix", "score_bal")}
    if a.arm is not None:
        out["bounds"] = {}
        for name, (r, mode) in {"offlist_all_valid": (rule, "escalate"), "offlist_all_invalid": (rule, "routine"),
                                **{k: (v, "rule") for k, v in rules_extra.items()}}.items():
            bo = vr.outcomes(a, ab, r, offlist=mode)
            bp, bd = vr.stats(bo, ab, W, partials=(vr.PARTIAL,), key=kcase)
            bs = vr.summarise(bp, bd)
            out["bounds"][name] = {m: bs[m] for m in ("score_mix", "score_bal", "U", "O", "partial", "esc")}
    reasons = o.reasons
    out["counts"] = {**{k: int(v) for k, v in Counter(o.outcome).items()},
                     "partial_inlist": sum(1 for r, x in zip(reasons, o.outcome) if x == vr.PARTIAL_OUT and r.kind == vr.OTHER_TIER1),
                     "partial_offlist": sum(1 for r, x in zip(reasons, o.outcome) if x == vr.PARTIAL_OUT and r.kind == vr.OFFLIST),
                     "pair_row_cases": int(((extra["pair"] > vr.PARTIAL) & (np.array(o.outcome) == vr.PARTIAL_OUT)).sum()),
                     "unreadable": int((~a.readable).sum()), "escalations": int(o.esc.sum())}
    return out, draws, point, o


def rescale_draws(d_row, d_ref, m):
    """SCORE against a reference that defines 0: 100 x (C_ref - C) / C_ref, from each score's own anchor form
    (C / C_anchor = 1 - score / 100), per draw."""
    r, f = 1 - d_row[m] / 100.0, 1 - d_ref[m] / 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(f > 0, 100.0 * (1 - r / np.where(f > 0, f, 1.0)), np.nan)


def ci(x):
    lo, hi = vs.interval(x)
    return [None if v is None else round(v, 2) for v in (lo, hi)]


def pdiff(da, db, pa, pb, measures=PAIRED):
    return vr.diff(da, db, pa, pb, measures)


def excludes0(d):
    lo, hi = d["ci"]
    return lo is not None and (lo > 0 or hi < 0)


# ---------------------------------------------------------------- references


def code_passes(ab, code: str) -> int:
    """SERIOUS cases on which `code` names an R10 target (standard map with the family rows)."""
    t1 = {h for h in ab.matcher.conditions_hit([code], vr.POLICY) if ab.key.tiers.get(h) == 1}
    return sum(1 for k, s in zip(ab.key.keys, ab.serious) if s and t1 & set(k.r10))


def best_single_flag(ab) -> tuple[str, str, int]:
    """(code, condition, SERIOUS cases passed) for the flag on the most common tier-1 target: the condition that is
    an R10 target on the most SERIOUS cases; ties go to the condition whose code passes the most SERIOUS cases (the
    family rows let one code name several conditions), then to the shortest code."""
    counts = Counter(t for k, s in zip(ab.key.keys, ab.serious) if s for t in k.r10)
    top = max(counts.values())
    best = None
    for cond in sorted(c for c, n in counts.items() if n == top):
        for code in sorted({c for c in ab.matcher.map_codes() if cond in ab.matcher.conditions_hit([c], "strict")}):
            cand = (code_passes(ab, code), -len(code), code, cond)
            best = cand if best is None or cand[:2] > best[:2] else best
    return best[2], best[3], best[0]


def max_single_code(ab) -> tuple[str, int]:
    """The map code that names a target on the most SERIOUS cases: the ceiling of any one fixed flag."""
    return max(((c, code_passes(ab, c)) for c in sorted(ab.matcher.map_codes())), key=lambda x: (x[1], -len(x[0])))


def reference_rows(ab):
    refs = sb.reference_answers(ab)
    code, cond, n = best_single_flag(ab)
    one = sb.fixed_answers("single-flag", ab, np.ones(ab.key.n, bool), [[code]] * ab.key.n)
    ordered = {"single-flag": (one, f"Always escalate, one fixed flag on the most common target: {code} ({cond}; names "
                                     f"a target on {n} of {int(ab.serious.sum())} SERIOUS cases)"),
               "always-escalate": (refs["always-escalate"], sb.REFS["always-escalate"]),
               "always-routine": (refs["always-routine"], sb.REFS["always-routine"]),
               "dxa": (refs["dxa"], sb.REFS["dxa"]), "naive-bayes": (refs["naive-bayes"], sb.REFS["naive-bayes"])}
    ideal_codes = []
    code_of = {}
    for c in ab.matcher.map_codes():
        for h in ab.matcher.conditions_hit([c], "strict"):
            code_of.setdefault(h, c)
    for k in ab.key.keys:
        ideal_codes.append([code_of[k.r10[0]]] if k.r10 else [])
    ordered["target-named"] = (sb.fixed_answers("target-named", ab, np.ones(ab.key.n, bool), ideal_codes),
                               "Always escalate, naming a target on every SERIOUS case (the current 0)")
    return ordered, (code, cond, n)


# ---------------------------------------------------------------- justification audit

STOP = set("""unspecified other acute chronic disease diseases disorder disorders without with complication complications
specified elsewhere classified syndrome initial encounter type site part sites unknown due following and the for
not of in to or by as at on from other nos organism mention stated condition conditions primary secondary
malignant neoplasm neoplasms""".split())
# Conditions and code groups the sentence may name in other words.
SYNONYMS = {
    "I20": "angina|coronary|acs|ischaem|ischem|cardiac", "I21": "myocardial|infarct|coronary|acs|stemi|nstemi|heart attack|\\bmi\\b|cardiac",
    "I22": "myocardial|infarct|coronary|acs|\\bmi\\b", "I24": "coronary|acs|ischaem|ischem|infarct|angina|cardiac",
    "I25": "coronary|ischaem|ischem|angina|cardiac", "I26": "pulmonary embol|\\bpe\\b|embol|thrombo|clot",
    "I80": "thromb|dvt|clot", "I82": "thromb|dvt|clot", "I71": "dissection|aneurysm|aortic", "I60": "subarachnoid|haemorrh|hemorrh|bleed",
    "I61": "haemorrh|hemorrh|bleed|stroke", "I63": "stroke|infarct|cerebrovascular|ischaem|ischem", "I64": "stroke",
    "G45": "tia|transient|stroke", "A41": "sepsis|septic", "A40": "sepsis|septic", "R65": "sepsis|septic",
    "G00": "mening", "G01": "mening", "G02": "mening", "G03": "mening", "G04": "encephal", "A39": "mening",
    "C": "cancer|malignan|neoplasm|tumou?r|lymphoma|carcinoma|malignancy|metasta|leukaem|leukem",
    "K85": "pancreatit", "K92": "bleed|haemorrh|hemorrh|melaena|melena", "J18": "pneumonia", "J12": "pneumonia",
    "J13": "pneumonia", "J15": "pneumonia", "J93": "pneumothorax", "I47": "tachycardia|svt|arrhythm", "I49": "arrhythm|tachycardia",
    "I48": "fibrillation|\\baf\\b|arrhythm", "I50": "heart failure|pulmonary oedema|pulmonary edema|cardiac failure|chf",
    "J81": "pulmonary oedema|pulmonary edema", "I40": "myocarditis", "I51": "myocarditis|cardiomyopathy",
    "I30": "pericarditis", "J45": "asthma|bronchospasm", "J46": "asthma", "J05": "epiglott|croup|airway", "J38": "laryngospasm|airway|larynx|laryngeal",
    "T78": "anaphyla|allergic|angioedema", "T78.2": "anaphyla", "T78.3": "angioedema|angio-oedema", "T80": "anaphyla", "G61": "guillain|gbs|polyneuropath",
    "G70": "myasthen", "K22": "oesophag|esophag|boerhaave|perforat|rupture", "A98": "ebola|haemorrhagic fever|hemorrhagic fever",
    "B57": "chagas", "C56": "ovar", "A15": "tuberculosis|\\btb\\b", "B20": "hiv", "D86": "sarcoid", "M32": "lupus|sle",
    "G24": "dystoni", "T44": "dystoni", "T43": "dystoni", "J44": "copd|obstructive", "A37": "pertussis|whooping",
}


def _norm(t: str) -> str:
    t = t.lower().replace("æ", "e").replace("é", "e")
    return re.sub(r"(?<=[a-z])ae|oe(?=[a-z])", "e", t)


def load_descriptions() -> dict[str, str]:
    desc = {}
    if ICD_ORDER.exists():
        with ICD_ORDER.open(encoding="latin-1") as fh:
            for line in fh:
                code = line[6:13].strip()
                if code:
                    desc[code] = line[77:].strip() or line[16:76].strip()
    return desc


class Namer:
    """Does a sentence name the flagged condition? We match the code's ICD-10-CM description and its parent's,
    the DDXPlus conditions it maps to, and a synonym list per code group, on 6-letter word stems."""

    def __init__(self, ab):
        self.ab = ab
        self.desc = load_descriptions()

    def patterns(self, code: str) -> tuple[set[str], list[str]]:
        c = normalise_code(code)
        texts = [self.desc.get(c[:n], "") for n in range(len(c), 2, -1)]
        texts += list(self.ab.matcher.conditions_hit([code], vr.POLICY))
        stems = {w[:6] for t in texts for w in re.findall(r"[a-z]+", _norm(t)) if len(w) >= 5 and w not in STOP}
        syn = []
        for pre in (c[:4] if len(c) >= 4 else None, c[:3], c[:1]):
            key = pre if pre is None else (pre[:3] + "." + pre[3:] if len(pre) == 4 else pre)
            for k in (key, pre):
                if k and k in SYNONYMS:
                    syn.append(SYNONYMS[k])
        return stems, syn

    def names(self, code: str, text: str) -> bool:
        stems, syn = self.patterns(code)
        t = _norm(text)
        words = {w[:6] for w in re.findall(r"[a-z]+", t) if len(w) >= 5}
        return bool(stems & words) or any(re.search(p, t) for p in syn)


URGENT = re.compile(r"\burgent(ly)?\b", re.I)
URGENT_BROAD = re.compile(r"\burgent(ly)?\b|\bemergen(t|cy|tly)\b|\bimmediate(ly)?\b", re.I)
NEGATION = re.compile(r"\b(no|not|non|without|nor|doesn't|don't|isn't|does not|do not|is not)\b(\W+\w+){0,3}\W*$", re.I)
NO_ESC = re.compile(r"\b(does not|doesn't|do not|not) (need|require|warrant)\w*\b[^.]{0,40}\b(escalat|urgent|immediate)"
                    r"|\bno (need for |urgent |immediate )?escalation\b", re.I)


def says(pattern, text: str) -> bool:
    """The pattern occurs at least once without a negation in the three words before it."""
    for m in pattern.finditer(text):
        if not NEGATION.search(text[:m.start()]):
            return True
    return False


def flag_tier(code: str, ab, rule) -> str:
    hit = ab.matcher.conditions_hit([code], vr.POLICY)
    if hit:
        return str(min(ab.key.tiers.get(h, 3) for h in hit))
    return rule.tier(code)[0]


def justification_audit(model, arm, a, o, ab, rule, namer) -> dict:
    esc_n = esc_named = low_n = low_urgent = low_broad = esc_noesc = 0
    good, bad = [], []
    for i, (p, k) in enumerate(zip(a.parsed, ab.key.keys)):
        if p is None or not p.readable or not p.flag:
            continue
        text = p.justification or ""
        tier = flag_tier(p.flag, ab, rule)
        row = {"model": model, "arm": sb.ARM_LABELS[arm], "case_id": k.case_id, "truth": k.truth,
               "class": ab.klass[i], "flag": p.flag, "flag_tier": tier, "outcome": o.outcome[i], "text": text}
        if o.esc[i]:
            esc_n += 1
            named = namer.names(p.flag, text)
            esc_named += named
            no_esc = bool(NO_ESC.search(text))
            esc_noesc += no_esc
            if named and o.outcome[i] == vr.PASS and not no_esc:
                good.append(row)
            if no_esc or not named:
                bad.append({**row, "why": "escalating flag, sentence says no escalation" if no_esc
                            else "escalating flag, sentence does not name the flagged condition"})
        elif tier in ("2", "3"):
            low_n += 1
            u, ub = says(URGENT, text), says(URGENT_BROAD, text)
            low_urgent += u
            low_broad += ub
            if u:
                bad.append({**row, "why": f"tier-{tier} flag (no escalation), sentence says urgent"})
    return {"escalating_flags": esc_n, "escalating_named": esc_named,
            "escalating_named_pct": round(100 * esc_named / esc_n, 1) if esc_n else None,
            "escalating_says_no_escalation": esc_noesc,
            "tier23_flags": low_n, "tier23_urgent": low_urgent,
            "tier23_urgent_pct": round(100 * low_urgent / low_n, 1) if low_n else None,
            "tier23_urgent_broad": low_broad,
            "tier23_urgent_broad_pct": round(100 * low_broad / low_n, 1) if low_n else None,
            "_good": good, "_bad": bad}


# ---------------------------------------------------------------- run


def run(n_boot: int) -> dict:
    ab = sb.load_ab()
    rule = vr.TierFileRule()
    rules_extra = {"weak_excluded": vr.TierFileRule(exclude_weak=True)}
    W, kcase, _ = st.within_setup(ab.key, None, n_boot, vs.BOOTSTRAP_SEED)
    Mc = vs.cluster_draws(ab.key.k, n_boot, vs.BOOTSTRAP_SEED)
    namer = Namer(ab)
    rows, raw, audit, examples = {}, {}, {}, {"good": [], "bad": []}
    arms = JARMS + tuple(BASE.values())
    for model in MODELS:
        for arm in arms:
            f = file_for(model, arm)
            if not f.exists():
                continue
            preds, meta = sb.load_predictions(f)
            assert meta["prompt_version"] == arm and meta["model"] == model, f
            a = sb.row_answers(preds, arm, ab, f"{model}|{arm}")
            out, draws, point, o = score_row(a, ab, rule, rules_extra, W, kcase, Mc)
            name = f"{model}|{arm}"
            rows[name] = {"model": model, "arm": arm, "arm_label": sb.ARM_LABELS[arm], "file": str(f.relative_to(ROOT)),
                          "sha256": sb.ak.sha256_file(f), **out}
            raw[name] = (point, draws)
            if arm in JARMS:
                au = justification_audit(model, arm, a, o, ab, rule, namer)
                examples["good"] += au.pop("_good")
                examples["bad"] += au.pop("_bad")
                audit[name] = au

    ref_answers, single = reference_rows(ab)
    refs, ref_raw = {}, {}
    for r, (a, label) in ref_answers.items():
        out, draws, point, _ = score_row(a, ab, rule, {}, W, kcase, Mc)
        refs[r] = {"label": label, **out}
        ref_raw[r] = (point, draws)

    # Rescaled against the single fixed flag (the proposed 0).
    zp, zd = ref_raw["single-flag"]
    for coll, rr in ((rows, raw), (refs, ref_raw)):
        for name, (p, d) in rr.items():
            coll[name]["rescaled_single_flag"] = {}
            for m in ("score_mix", "score_bal"):
                pv = float(rescale_draws({m: np.array([p[m]])}, {m: np.array([zp[m]])}, m)[0])
                coll[name]["rescaled_single_flag"][m] = {"value": round(pv, 2), "ci": ci(rescale_draws(d, zd, m))}

    paired = {}
    for model in MODELS:
        for x, y in (("v7a4aj", "v7a4bj"), ("v7a4aj", "v7a4a"), ("v7a4bj", "v7a4b")):
            nx, ny = f"{model}|{x}", f"{model}|{y}"
            if nx in raw and ny in raw:
                paired[f"{model}: {sb.ARM_LABELS[x]} - {sb.ARM_LABELS[y]}"] = {
                    "model": model, "a": x, "b": y, **pdiff(raw[nx][1], raw[ny][1], raw[nx][0], raw[ny][0])}
    within = {}
    for arm in JARMS:
        names = [f"{m}|{arm}" for m in MODELS if f"{m}|{arm}" in raw]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                d = pdiff(raw[x][1], raw[y][1], raw[x][0], raw[y][0], ("score_mix", "score_bal", "U", "O"))
                within[f"{sb.ARM_LABELS[arm]}: {sb.short(x.split('|')[0])} - {sb.short(y.split('|')[0])}"] = {
                    **d, "separated_bal": excludes0(d["score_bal"]), "separated_mix": excludes0(d["score_mix"])}

    return {"spec": "spec/v0.3-scoring.md draft 3, section 12 and amendment A1; valid-reason rule",
            "rule": {"partial_cost": vr.PARTIAL, "partial_sensitivity": list(vr.PARTIAL_SENSITIVITY),
                     "pair_row": f"Boerhaave truth escalated under a non-surgical label costs {vr.PAIR_COST}; "
                                 f"surgical labels are off-list codes under {', '.join(vr.SURGICAL_PREFIXES)}",
                     "offlist_source": str(vr.OFFLIST_TIERS_CSV.relative_to(ROOT)),
                     "offlist_source_sha256": sb.ak.sha256_file(vr.OFFLIST_TIERS_CSV),
                     "weak_rows": sorted(rule.weak), "unreadable": "no escalation (cost 7 on SERIOUS, 0 on BENIGN)"},
            "sample": {"cases": int(ab.key.n), "serious": int(ab.serious.sum()), "benign": int(ab.benign.sum()),
                       "middle": int((ab.klass == "middle").sum()), "conditions": int(ab.key.k),
                       "interval": {"primary": st.INTERVAL_NOTE, "draws": n_boot, "seed": vs.BOOTSTRAP_SEED}},
            "single_flag": {"code": single[0], "condition": single[1], "serious_passed": single[2],
                            "best_any_single_code": dict(zip(("code", "serious_passed"), max_single_code(ab)))},
            "rows": rows, "references": refs, "paired": paired, "within_arm": within,
            "justification_audit": audit, "justification_examples": examples}


# ---------------------------------------------------------------- tables


def f(m) -> str:
    return sb._fmt(m)


def v(m) -> str:
    return sb._v(m)


def tables(b: dict) -> str:
    rows = b["rows"]
    L = []
    for arm in JARMS:
        lab = sb.ARM_LABELS[arm]
        L += [f"## Arm {lab}", "",
              "| Model | Balanced [95% CI] | Sample mix [95% CI] | U % [CI] | O % [CI] | Partial % (in / off) | ESC % | Rescaled, balanced | Rescaled, mix |",
              "|---|---|---|---|---|---|---|---|---|"]
        for mdl in MODELS:
            r = rows.get(f"{mdl}|{arm}")
            if not r:
                continue
            m, c, z = r["measures"], r["counts"], r["rescaled_single_flag"]
            L.append(f"| {sb.short(mdl)} | {f(m['score_bal'])} | {f(m['score_mix'])} | {f(m['U'])} | {f(m['O'])} | "
                     f"{v(m['partial'])} ({c['partial_inlist']} / {c['partial_offlist']}) | {v(m['esc'])} | "
                     f"{f(z['score_bal'])} | {f(z['score_mix'])} |")
        L += ["", f"### Arm {lab}: sensitivity rows (balanced; sample mix in the JSON)", "",
              "| Model | Partial 2 | Partial 3.5 | Boerhaave pair (cases) | Off-list all valid | Off-list all invalid | Weak rows excluded | Condition bootstrap |",
              "|---|---|---|---|---|---|---|---|"]
        for mdl in MODELS:
            r = rows.get(f"{mdl}|{arm}")
            if not r:
                continue
            m, bd = r["measures"], r["bounds"]
            L.append(f"| {sb.short(mdl)} | {f(m['score_bal@2'])} | {f(m['score_bal@3.5'])} | {f(m['score_bal@pair'])} "
                     f"({r['counts']['pair_row_cases']}) | {f(bd['offlist_all_valid']['score_bal'])} | "
                     f"{f(bd['offlist_all_invalid']['score_bal'])} | {f(bd['weak_excluded']['score_bal'])} | "
                     f"{f(r['condition_bootstrap']['score_bal'])} |")
        L.append("")
    L += ["## Reference rows (same 150 cases)", "",
          "| Reference | Balanced [95% CI] | Sample mix [95% CI] | U % | O % | Partial % | Rescaled, balanced | Rescaled, mix |",
          "|---|---|---|---|---|---|---|---|"]
    for r in b["references"].values():
        m, z = r["measures"], r["rescaled_single_flag"]
        L.append(f"| {r['label']} | {f(m['score_bal'])} | {f(m['score_mix'])} | {v(m['U'])} | {v(m['O'])} | {v(m['partial'])} | "
                 f"{f(z['score_bal'])} | {f(z['score_mix'])} |")
    L += ["", "## Paired differences (a - b, same cases and draws)", "",
          "| Comparison | Balanced | Sample mix | U (pp) | O (pp) | Partial (pp) | ESC (pp) |", "|---|---|---|---|---|---|---|"]
    for k, d in b["paired"].items():
        L.append(f"| {sb.short(d['model'])}: {sb.ARM_LABELS[d['a']]} - {sb.ARM_LABELS[d['b']]} | {f(d['score_bal'])} | "
                 f"{f(d['score_mix'])} | {f(d['U'])} | {f(d['O'])} | {f(d['partial'])} | {f(d['esc'])} |")
    L += ["", "## Model pairs within each arm", "", "| Pair | Balanced | Sample mix | U (pp) | O (pp) | Separated (bal / mix) |",
          "|---|---|---|---|---|---|"]
    for k, d in b["within_arm"].items():
        L.append(f"| {k} | {f(d['score_bal'])} | {f(d['score_mix'])} | {f(d['U'])} | {f(d['O'])} | "
                 f"{'yes' if d['separated_bal'] else 'no'} / {'yes' if d['separated_mix'] else 'no'} |")
    L += ["", "## Justification audit", "",
          "| Model | Arm | Escalating flags | Name the flagged condition | Say no escalation | Tier-2/3 flags | Say \"urgent\" | Urgent, emergency or immediate |",
          "|---|---|---|---|---|---|---|---|"]
    for k, a in b["justification_audit"].items():
        mdl, arm = k.split("|")
        L.append(f"| {sb.short(mdl)} | {sb.ARM_LABELS[arm]} | {a['escalating_flags']} | {a['escalating_named']} "
                 f"({a['escalating_named_pct']}%) | {a['escalating_says_no_escalation']} | {a['tier23_flags']} | "
                 f"{a['tier23_urgent']} ({a['tier23_urgent_pct']}%) | {a['tier23_urgent_broad']} ({a['tier23_urgent_broad_pct']}%) |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(AB / "ab-4j-scores.json"))
    ap.add_argument("--tables", default=str(AB / "ab-4j-tables.md"))
    ap.add_argument("--n-boot", type=int, default=vs.N_BOOTSTRAP)
    args = ap.parse_args()
    b = run(args.n_boot)
    Path(args.out).write_text(json.dumps(sb._jsonable(b), indent=1, ensure_ascii=False) + "\n")
    Path(args.tables).write_text(tables(b))
    print(f"single flag: {b['single_flag']}")
    for n, r in b["rows"].items():
        m = r["measures"]
        print(f"{sb.short(r['model']):24s} {r['arm_label']:4s} bal {sb._fmt(m['score_bal']):26s} mix {sb._fmt(m['score_mix']):28s} "
              f"U {m['U']['value']:5.1f} O {m['O']['value']:5.1f} part {m['partial']['value']:5.1f} esc {m['esc']['value']:5.1f}")


if __name__ == "__main__":
    main()
