"""Reference triage policies, per-case ambiguity, and reporting variants.

Why: the board's oracle (escalate iff P(severe) >= T) scores 100 because it reads
DDXPlus's exact posterior, which no clinician has. To say how far models sit from a
human-achievable level we need reference policies that use only what the model
sees (age, sex, coded evidence) plus ordinary clinical knowledge, and a per-case
measure of how determinable the triage call is from that information.

This script (no inference, no API calls):

1. Scores reference policies with evaluator/triage_score.py: red-flag rules over
   DDXPlus evidence codes, a Manchester-style discriminator set, "severity of the
   top-1 diagnosis", "any severe diagnosis in the top k", a naive-Bayes clinician
   that learns symptom likelihoods from the DDXPlus test set, noisy-posterior
   clinicians, rank-limited readers of the differential, and the model consensus.
2. Computes per-case ambiguity: distance of P(severe) from T, the probability that
   one more question flips the call (value of one more question), red-flag
   presence, and the naive-Bayes posterior from positives only.
3. Re-scores every model with per-case data inside and outside the ambiguity
   band, on determinable cases only, with the band down-weighted, and in a
   context-seeking mode where "INSUFFICIENT" on a band case counts as abstaining.

Usage:
    .venv/bin/python scripts/analysis/reference_policies.py

Outputs go to results/analysis/failure_modes/human_baseline/ (gitignored):
    reference_policies.csv, per_case_ambiguity.csv, model_band_rates.csv,
    ranking_variants.csv, noisy_posterior.csv, summary.json,
    fig_reference_band.png, ddxplus_likelihoods.json (cache of the 134k-row scan)
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import triage_score as ts  # noqa: E402
from evaluator.schemas import ModelPrediction  # noqa: E402
from scripts.analysis.failure_shape import SHORT  # noqa: E402
import build_triage_board as btb  # noqa: E402

CASES_PATH = ROOT / "data/test_sets/eval-250-v0.json"
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
EVID_PATH = ROOT / "data/ddxplus_v0/release_evidences.json"
OUT = ROOT / "results/analysis/failure_modes/human_baseline"  # own folder: siblings write to the parent
LIKELIHOOD_CACHE = OUT / "ddxplus_likelihoods.json"

T = ts.URGENT_THRESHOLD
U, O = ts.TOLERATED_UNDER_TRIAGE, ts.TOLERATED_OVER_TRIAGE
SEED = ts.BOOTSTRAP_SEED
N_NOISE_DRAWS = 400
BAND_MARGIN = 0.05  # |P(severe) - T| <= this puts a case in the ambiguity band
BAND_FLIP = 0.25  # a single question that flips the call with >= this probability does too
BAND_WEIGHT = 0.5  # weight of band cases in the down-weighted variant
# Published human telephone-triage rates (spec/triage_tolerances.md rows 6-10, verified there):
# under-triage 3.7% (Graversen 2020, nurse) to 11% (Huibers 2011, high-urgency contacts);
# over-triage 4.3% (Graversen 2020, GP) to 20.2% (Smits 2020, paper cases). Drawn as a band, not a point.
HUMAN_UNDER = (0.037, 0.11)
HUMAN_OVER = (0.043, 0.202)

# ------------------------------------------------------------------ evidence sets
# DDXPlus location values (E_55 pain, E_57 radiation, E_152 swelling).
CHEST = {"V_29", "V_101", "V_55", "V_56", "V_159", "V_160", "V_170", "V_171"}
RADIATION = {"V_121", "V_163", "V_194", "V_195", "V_30", "V_31", "V_27", "V_28", "V_53", "V_54", "V_26",
             "V_127", "V_128", "V_175", "V_176", "V_38", "V_39"}
FACE_MOUTH = {"V_116", "V_117", "V_188", "V_189", "V_61", "V_162", "V_32", "V_115", "V_148", "V_121", "V_118",
              "V_108", "V_109", "V_125", "V_126", "V_122", "V_89", "V_20", "V_21"}


class Presentation:
    """One case's evidence, queryable by code and value."""

    def __init__(self, tokens: list[str]):
        self.codes: set[str] = set()
        self.values: dict[str, set[str]] = defaultdict(set)
        for t in tokens:
            code, _, val = t.partition("_@_")
            self.codes.add(code)
            if val:
                self.values[code].add(val)

    def has(self, code: str) -> bool:
        return code in self.codes

    def at(self, code: str, where: set[str]) -> bool:
        return bool(self.values.get(code, set()) & where)

    def num(self, code: str) -> int:
        vals = [int(v) for v in self.values.get(code, ()) if v.lstrip("-").isdigit()]
        return max(vals) if vals else -1


def red_flags(pr: Presentation) -> dict[str, bool]:
    """Named red flags built from DDXPlus evidence codes. Each is a clinical discriminator
    a triage nurse could apply from the intake sheet alone."""
    dyspnea = pr.has("E_66") or pr.has("E_64") or pr.has("E_75") or pr.has("E_67")
    chest_pain = pr.at("E_55", CHEST) or pr.has("E_14")
    typical = chest_pain and (pr.at("E_57", RADIATION) or pr.has("E_50") or pr.has("E_218") or pr.has("E_13"))
    wheeze = pr.has("E_214") or pr.has("E_112")
    rash = pr.has("E_129")
    f = {
        "chest_pain": chest_pain,
        "chest_pain_typical": typical,
        "chest_pain_with_dyspnea": chest_pain and dyspnea,
        "dyspnea": dyspnea,
        "syncope": pr.has("E_159"),
        "seizure": pr.has("E_43"),
        "hemoptysis": pr.has("E_45"),
        "gi_bleed": pr.has("E_210") or pr.has("E_140") or pr.has("E_179"),
        "bleeding_bruising": pr.has("E_178"),
        "face_mouth_swelling": pr.has("E_151") and pr.at("E_152", FACE_MOUTH),
        "stridor": pr.has("E_194"),
        "allergen_reaction": pr.has("E_42") and (rash or wheeze or dyspnea),
        "rash_with_wheeze_or_dyspnea": rash and (wheeze or dyspnea),
        "wheeze": wheeze,
        "focal_neuro": any(pr.has(c) for c in ("E_176", "E_63", "E_156", "E_52", "E_84", "E_83", "E_172", "E_180")),
        "confusion": pr.has("E_39"),
        "airway": pr.has("E_65") and (pr.has("E_190") or pr.has("E_194")),
        "severe_pain": pr.num("E_56") >= 8,
        "fever_with_dyspnea": pr.has("E_91") and dyspnea,
        "palpitations": pr.has("E_155") or pr.has("E_164"),
        "presyncope": pr.has("E_82"),
        "choking": pr.has("E_75") or pr.has("E_128"),
        "dystonia": pr.has("E_193") or pr.has("E_168") or pr.has("E_192"),
    }
    f["anaphylaxis_signs"] = f["face_mouth_swelling"] or f["stridor"] or f["allergen_reaction"] or f["rash_with_wheeze_or_dyspnea"]
    return f


NARROW_FLAGS = ("chest_pain_typical", "chest_pain_with_dyspnea", "syncope", "seizure", "hemoptysis", "gi_bleed",
                "anaphylaxis_signs", "stridor", "focal_neuro", "confusion", "airway")
BROAD_FLAGS = ("chest_pain", "dyspnea", "syncope", "seizure", "hemoptysis", "gi_bleed", "bleeding_bruising",
               "anaphylaxis_signs", "stridor", "wheeze", "focal_neuro", "confusion", "airway", "severe_pain",
               "fever_with_dyspnea", "palpitations", "presyncope", "choking", "dystonia")


# ------------------------------------------------------------------ data loading


def load_cases():
    cases = json.loads(CASES_PATH.read_text())["cases"]
    return cases, [c["case_id"] for c in cases]


def load_ddxplus_rows(case_ids):
    want = {int(cid.split("_")[1]): cid for cid in case_ids}
    out = {}
    with open(DDX_CSV) as f:
        for i, row in enumerate(csv.DictReader(f)):
            cid = want.get(i)
            if cid is None:
                continue
            out[cid] = {
                "differential": ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]),
                "pathology": row["PATHOLOGY"],
                "evidences": ast.literal_eval(row["EVIDENCES"]),
            }
    return out


def parse_evidence_list(s: str) -> list[str]:
    s = s.strip()[1:-1]
    return [t.strip().strip("'") for t in s.split(",") if t.strip()]


def age_band(age: int) -> str:
    for hi, name in ((18, "0-17"), (35, "18-34"), (50, "35-49"), (65, "50-64")):
        if age < hi:
            return name
    return "65+"


def build_likelihoods(cond_names):
    """Scan the DDXPlus test set once: P(evidence code present | pathology) and P(pathology | age band, sex)."""
    if LIKELIHOOD_CACHE.exists():
        return json.loads(LIKELIHOOD_CACHE.read_text())
    n_path = Counter()
    code_counts = defaultdict(Counter)
    prior = defaultdict(Counter)
    with open(DDX_CSV) as f:
        for row in csv.DictReader(f):
            c = row["PATHOLOGY"]
            n_path[c] += 1
            prior[f"{age_band(int(row['AGE']))}|{row['SEX']}"][c] += 1
            codes = {t.partition("_@_")[0] for t in parse_evidence_list(row["EVIDENCES"])}
            for q in codes:
                code_counts[c][q] += 1
    out = {
        "n_rows": sum(n_path.values()),
        "n_pathology": dict(n_path),
        "code_counts": {c: dict(v) for c, v in code_counts.items()},
        "prior": {k: dict(v) for k, v in prior.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LIKELIHOOD_CACHE.write_text(json.dumps(out))
    return out


def load_model_outputs(case_ids):
    """Per model: escalate flag, readable flag, INSUFFICIENT flag, in case order. Only rows with per-case data."""
    import glob

    idx = {cid: k for k, cid in enumerate(case_ids)}
    models = {}
    for f in sorted(glob.glob(str(ROOT / "leaderboard/*-eval.json"))):
        ev = json.loads(Path(f).read_text())
        path, status, _ = btb.resolve(ev)
        if path is None:
            continue
        raw = json.loads(path.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        esc = np.zeros(len(case_ids), dtype=np.int8)
        readable = np.zeros(len(case_ids), dtype=np.int8)
        insuff = np.zeros(len(case_ids), dtype=np.int8)
        asks_question = np.zeros(len(case_ids), dtype=np.int8)
        seen = set()
        for p in preds:
            if not isinstance(p, dict) or p.get("case_id") not in idx or p["case_id"] in seen:
                continue
            seen.add(p["case_id"])
            k = idx[p["case_id"]]
            try:
                mp = ModelPrediction(**p)
            except Exception:
                continue
            readable[k] = 1
            esc[k] = int(mp.escalation_decision == ts.ESCALATE)
            insuff[k] = int(getattr(mp, "information_sufficiency", None) == "INSUFFICIENT")
            asks_question[k] = int(str(p.get("followup_kind") or "").upper() == "QUESTION")
        models[ev["model"]] = {"escalate": esc, "readable": readable, "insufficient": insuff,
                               "asks_question": asks_question, "status": status}
    return models


# ------------------------------------------------------------------ posteriors


def naive_bayes(pr: Presentation, age: int, sex: str, lik, cond_names, all_codes, use_negatives: bool):
    """Posterior over conditions from the case's evidence codes with likelihoods learned from DDXPlus."""
    key = f"{age_band(age)}|{'M' if sex == 'male' else 'F'}"
    prior = lik["prior"].get(key) or lik["n_pathology"]
    logp = np.full(len(cond_names), -np.inf)
    for i, c in enumerate(cond_names):
        n = lik["n_pathology"].get(c, 0)
        if n == 0:
            continue
        cc = lik["code_counts"].get(c, {})
        lp = np.log((prior.get(c, 0) + 0.5) / (sum(prior.values()) + 0.5 * len(cond_names)))
        for q in all_codes:
            L = (cc.get(q, 0) + 0.5) / (n + 1.0)
            if q in pr.codes:
                lp += np.log(L)
            elif use_negatives:
                lp += np.log(1.0 - L)
        logp[i] = lp
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def one_more_question(P: dict[str, float], pr: Presentation, lik, severe: set[str], all_codes, evid):
    """Value of one more question. For each evidence code the case does not list, treat the
    answer as unknown, update the DDXPlus posterior with the learned likelihood of that code
    per condition, and record the probability the urgent call flips. Returns the best code.

    Under DDXPlus's generator a condition never produces evidence outside its list, so a "yes"
    to such a question removes that condition. That is why the best question is often a
    benign-specific symptom or antecedent rather than a classic red-flag question."""
    names = list(P)
    p = np.array([P[c] for c in names])
    p = p / p.sum()
    sev = np.array([c in severe for c in names])
    base = float(p[sev].sum())
    urgent = base >= T
    best = {"code": None, "flip": 0.0, "abs_change": 0.0, "p_yes": None, "p_sev_yes": None, "p_sev_no": None}
    for q in all_codes:
        if q in pr.codes:
            continue
        L = np.array([(lik["code_counts"].get(c, {}).get(q, 0) + 0.5) / (lik["n_pathology"].get(c, 1) + 1.0) for c in names])
        p_yes = float((p * L).sum())
        if p_yes <= 1e-9 or p_yes >= 1 - 1e-9:
            continue
        post_yes = p * L / p_yes
        post_no = p * (1 - L) / (1 - p_yes)
        s_yes, s_no = float(post_yes[sev].sum()), float(post_no[sev].sum())
        flip = p_yes * ((s_yes >= T) != urgent) + (1 - p_yes) * ((s_no >= T) != urgent)
        change = p_yes * abs(s_yes - base) + (1 - p_yes) * abs(s_no - base)
        if flip > best["flip"] or (flip == best["flip"] and change > best["abs_change"]):
            best = {"code": q, "flip": float(flip), "abs_change": float(change), "p_yes": p_yes,
                    "p_sev_yes": s_yes, "p_sev_no": s_no}
    if best["code"]:
        best["question"] = evid[best["code"]]["question_en"]
    return best


# ------------------------------------------------------------------ scoring helpers


def score(p, e, U_=U, O_=O):
    m = ts.score_model(p, e, U_, O_)
    return m.under, m.over, m.score


def weighted_rates(p, e, w):
    urgent = ts.urgent_labels(p)
    e = np.asarray(e, dtype=bool)
    under = float((w * (urgent & ~e)).sum() / max((w * urgent).sum(), 1e-9))
    over = float((w * (~urgent & e)).sum() / max((w * ~urgent).sum(), 1e-9))
    return under, over, ts.triage_score(under, over)


def would_rank(s, model_scores):
    return int(1 + (np.asarray(model_scores) > s).sum())


# ------------------------------------------------------------------ main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cases, case_ids = load_cases()
    cond = json.loads(COND_PATH.read_text())
    evid = json.loads(EVID_PATH.read_text())
    cond_names = sorted(cond)
    severe = {c for c in cond if cond[c]["severity"] <= ts.SEVERE_MAX}
    all_codes = sorted(evid)
    rows = load_ddxplus_rows(case_ids)
    p_map = ts.load_p_severe(case_ids, DDX_CSV, COND_PATH)
    p = np.array([p_map[c] for c in case_ids])
    urgent = ts.urgent_labels(p)
    required = np.array([bool(c["escalation_required"]) for c in cases])
    lik = build_likelihoods(cond_names)
    print(f"likelihoods from {lik['n_rows']} DDXPlus test rows")

    models = load_model_outputs(case_ids)
    model_names = sorted(models, key=lambda k: -score(p, models[k]["escalate"])[2])
    model_scores = {k: score(p, models[k]["escalate"]) for k in model_names}
    ms = np.array([model_scores[k][2] for k in model_names])
    top = model_names[0]
    print(f"{len(models)} models with per-case data; top {SHORT.get(top, top)} {ms[0]:.1f}")

    # ---- per-case features
    per_case = []
    flags_by_case = []
    nb_pos = np.zeros(len(cases))
    nb_full = np.zeros(len(cases))
    top1_sev = np.zeros(len(cases), dtype=int)
    sev_rank = np.full(len(cases), 99)
    p_topk = {k: np.zeros(len(cases)) for k in (1, 2, 3, 5)}
    p_sev3 = np.zeros(len(cases))  # mass on severity <= 3: what a clinician who calls TB, COPD or pneumonia urgent sees
    sev3_top3 = np.zeros(len(cases), dtype=bool)
    voi_flip = np.zeros(len(cases))
    voi_change = np.zeros(len(cases))
    voi_flip_symptoms = np.zeros(len(cases))  # same, but only symptom questions (no antecedents)
    symptom_codes = [q for q in all_codes if not evid[q]["is_antecedent"]]
    outcome_severe = np.zeros(len(cases), dtype=bool)  # the true DDXPlus pathology is severity 1-2
    nb_top1_hit = 0
    for i, (c, cid) in enumerate(zip(cases, case_ids)):
        r = rows[cid]
        pr = Presentation(c["presenting_symptoms"])
        f = red_flags(pr)
        flags_by_case.append(f)
        diff = r["differential"]
        P = {name: prob for name, prob in diff}
        top1_sev[i] = cond[diff[0][0]]["severity"]
        for rank, (name, _) in enumerate(diff, 1):
            if name in severe and rank < sev_rank[i]:
                sev_rank[i] = rank
        for k in p_topk:
            p_topk[k][i] = sum(prob for name, prob in diff[:k] if name in severe)
        p_sev3[i] = sum(prob for name, prob in diff if cond[name]["severity"] <= 3)
        sev3_top3[i] = any(cond[name]["severity"] == 3 for name, _ in diff[:3])
        pn = naive_bayes(pr, c["age"], c["sex"], lik, cond_names, all_codes, use_negatives=False)
        pf = naive_bayes(pr, c["age"], c["sex"], lik, cond_names, all_codes, use_negatives=True)
        sev_mask = np.array([n in severe for n in cond_names])
        nb_pos[i], nb_full[i] = float(pn[sev_mask].sum()), float(pf[sev_mask].sum())
        nb_top1_hit += int(cond_names[int(np.argmax(pn))] == r["pathology"])
        outcome_severe[i] = cond[r["pathology"]]["severity"] <= ts.SEVERE_MAX
        v = one_more_question(P, pr, lik, severe, all_codes, evid)
        voi_flip[i], voi_change[i] = v["flip"], v["abs_change"]
        voi_flip_symptoms[i] = one_more_question(P, pr, lik, severe, symptom_codes, evid)["flip"]
        per_case.append({
            "case_id": cid, "age": c["age"], "sex": c["sex"], "pathology": r["pathology"],
            "pathology_severity": cond[r["pathology"]]["severity"],
            "p_severe": round(float(p[i]), 4), "urgent_T": int(urgent[i]), "escalation_required_v0": int(required[i]),
            "margin": round(float(p[i] - T), 4), "top1": diff[0][0], "top1_severity": int(top1_sev[i]),
            "severe_rank": int(sev_rank[i]) if sev_rank[i] < 99 else "",
            "p_severity_le3": round(float(p_sev3[i]), 4), "severity3_in_top3": int(sev3_top3[i]),
            "n_evidence": len(c["presenting_symptoms"]),
            "narrow_flags": ";".join(k for k in NARROW_FLAGS if f[k]),
            "broad_flag_count": sum(f[k] for k in BROAD_FLAGS),
            "nb_positives_p_severe": round(nb_pos[i], 4), "nb_full_p_severe": round(nb_full[i], 4),
            "voi_best_code": v["code"], "voi_best_question": v.get("question", ""),
            "voi_flip_prob": round(v["flip"], 4), "voi_abs_change": round(v["abs_change"], 4),
            "voi_flip_prob_symptoms_only": round(float(voi_flip_symptoms[i]), 4),
            "voi_p_yes": None if v["p_yes"] is None else round(v["p_yes"], 4),
            "voi_p_severe_if_yes": None if v["p_sev_yes"] is None else round(v["p_sev_yes"], 4),
            "voi_p_severe_if_no": None if v["p_sev_no"] is None else round(v["p_sev_no"], 4),
        })

    # ---- reference policies
    narrow = np.array([any(f[k] for k in NARROW_FLAGS) for f in flags_by_case], dtype=np.int8)
    broad_count = np.array([sum(f[k] for k in BROAD_FLAGS) for f in flags_by_case])
    consensus = np.mean([models[k]["escalate"] for k in model_names], axis=0)
    # kind: what the policy reads. "evidence" = the case sheet only (what models see);
    # "differential" = DDXPlus's own differential; "models" = the models' answers.
    policies = {
        "oracle": ("Oracle: escalate iff P(severe) >= T (exact DDXPlus posterior)", urgent.astype(np.int8), "differential"),
        "v0_label": ("v0 label: a severity 1-2 diagnosis in the DDXPlus top 3", required.astype(np.int8), "differential"),
        "top1_severe": ("Top-1 diagnosis has severity <= 2", (top1_sev <= 2).astype(np.int8), "differential"),
        "top1_sev3": ("Top-1 diagnosis has severity <= 3", (top1_sev <= 3).astype(np.int8), "differential"),
        "any_severe_top2": ("Any severe diagnosis in the top 2", (sev_rank <= 2).astype(np.int8), "differential"),
        "any_severe_top5": ("Any severe diagnosis in the top 5", (sev_rank <= 5).astype(np.int8), "differential"),
        "any_severe_anywhere": ("Any severe diagnosis anywhere in the differential", (sev_rank < 99).astype(np.int8), "differential"),
        "read_top1_T": ("Reads only the top 1: severe mass in top 1 >= T", (p_topk[1] >= T).astype(np.int8), "differential"),
        "read_top2_T": ("Reads only the top 2: severe mass in top 2 >= T", (p_topk[2] >= T).astype(np.int8), "differential"),
        "read_top3_T": ("Reads only the top 3: severe mass in top 3 >= T", (p_topk[3] >= T).astype(np.int8), "differential"),
        "read_top5_T": ("Reads only the top 5: severe mass in top 5 >= T", (p_topk[5] >= T).astype(np.int8), "differential"),
        "cautious_sev3_T": ("Cautious clinician: severity 1-3 all count as urgent, escalate iff P(severity <= 3) >= T",
                            (p_sev3 >= T).astype(np.int8), "differential"),
        "cautious_sev3_half": ("Cautious clinician: escalate iff P(severity <= 3) >= 0.5", (p_sev3 >= 0.5).astype(np.int8), "differential"),
        "redflag_narrow": ("Red flags, narrow: typical chest pain, chest pain + dyspnea, syncope, seizure, hemoptysis, "
                           "GI bleed, anaphylaxis signs, stridor, focal neuro, confusion, airway", narrow, "evidence"),
        "redflag_broad_1": ("MTS-style discriminators, any one of 19 (adds any chest pain, any dyspnea, wheeze, "
                            "severe pain >= 8/10, palpitations, presyncope, choking, dystonia, fever + dyspnea)",
                            (broad_count >= 1).astype(np.int8), "evidence"),
        "redflag_broad_2": ("MTS-style discriminators, two or more", (broad_count >= 2).astype(np.int8), "evidence"),
        "redflag_broad_3": ("MTS-style discriminators, three or more", (broad_count >= 3).astype(np.int8), "evidence"),
        "nb_positives_T": ("Naive-Bayes clinician from listed evidence only, escalate iff posterior P(severe) >= T",
                           (nb_pos >= T).astype(np.int8), "evidence"),
        "nb_full_T": ("Naive-Bayes clinician treating unlisted evidence as absent, escalate iff P(severe) >= T",
                      (nb_full >= T).astype(np.int8), "evidence"),
        "consensus": (f"Majority vote of the {len(model_names)} models", (consensus >= 0.5).astype(np.int8), "models"),
        "always_escalate": ("Always escalate", np.ones(len(cases), dtype=np.int8), "none"),
    }
    # Best threshold for the naive-Bayes clinicians, to show their ceiling.
    for key, arr in (("nb_positives", nb_pos), ("nb_full", nb_full)):
        grid = np.arange(0.001, 0.80, 0.001)
        best_t = max(grid, key=lambda t: score(p, (arr >= t).astype(np.int8))[2])
        policies[f"{key}_best_t"] = (f"{policies[key + '_T'][0].split(', escalate')[0]}, best threshold {best_t:.3f}",
                                     (arr >= best_t).astype(np.int8), "evidence")

    decisions = {k: v["escalate"] for k, v in models.items()}
    decisions.update({k: v[1] for k, v in policies.items()})
    ci = ts.bootstrap(p, decisions, ranked=model_names)

    ref_rows = []
    for key, (desc, e, kind) in policies.items():
        under, over, s = score(p, e)
        lab = ts.label_counts(e, urgent)
        ref_rows.append({
            "policy": key, "kind": kind, "description": desc, "escalate_rate": round(float(e.mean()), 4),
            "under": round(under, 4), "over": round(over, 4), "score": round(s, 2),
            "score_ci_lo": round(ci[key]["score"][0], 2), "score_ci_hi": round(ci[key]["score"][1], 2),
            "missed": lab["missed_escalations"], "over_escalations": lab["over_escalations"],
            "would_rank": would_rank(s, ms), "n_models": len(model_names),
        })
    with open(OUT / "reference_policies.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_rows[0]))
        w.writeheader()
        w.writerows(ref_rows)

    # ---- noisy-posterior clinicians
    logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    logit_T = np.log(T / (1 - T))
    noise_rows = []
    thresholds = np.arange(0.03, 0.60, 0.01)
    frontier = {}
    for sigma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        Z = logit[None, :] + sigma * rng.standard_normal((N_NOISE_DRAWS, len(p)))
        # threshold at T
        e_T = (Z >= logit_T).astype(np.int8)
        sc_T = np.array([score(p, e_T[d])[2] for d in range(N_NOISE_DRAWS)])
        uo_T = np.array([score(p, e_T[d])[:2] for d in range(N_NOISE_DRAWS)])
        # best fixed threshold for this noise level
        curve = []
        for t in thresholds:
            lt = np.log(t / (1 - t))
            e_t = (Z >= lt).astype(np.int8)
            uo = np.array([score(p, e_t[d])[:2] for d in range(N_NOISE_DRAWS)])
            s = np.array([ts.triage_score(a, b) for a, b in uo])
            curve.append((float(t), float(uo[:, 0].mean()), float(uo[:, 1].mean()), float(s.mean())))
        best = max(curve, key=lambda x: x[3])
        frontier[sigma] = curve
        noise_rows.append({
            "sigma_logit": sigma, "threshold_T_score_mean": round(float(sc_T.mean()), 2),
            "threshold_T_score_p5": round(float(np.percentile(sc_T, 5)), 2),
            "threshold_T_score_p95": round(float(np.percentile(sc_T, 95)), 2),
            "threshold_T_under": round(float(uo_T[:, 0].mean()), 4), "threshold_T_over": round(float(uo_T[:, 1].mean()), 4),
            "best_threshold": round(best[0], 2), "best_threshold_score_mean": round(best[3], 2),
            "best_threshold_under": round(best[1], 4), "best_threshold_over": round(best[2], 4),
            "share_of_calls_flipped_vs_oracle": round(float((e_T != urgent[None, :]).mean()), 4),
        })
    with open(OUT / "noisy_posterior.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(noise_rows[0]))
        w.writeheader()
        w.writerows(noise_rows)
    # Noise level whose ceiling (best threshold) first drops to the top model's score.
    sig_match = next((r["sigma_logit"] for r in noise_rows if r["best_threshold_score_mean"] <= ms[0]), None)
    sig_match_T = next((r["sigma_logit"] for r in noise_rows if r["threshold_T_score_mean"] <= ms[0]), None)

    # ---- ambiguity band
    in_margin = np.abs(p - T) <= BAND_MARGIN
    in_flip = voi_flip >= BAND_FLIP
    band = in_margin | in_flip
    determinable = ~band
    weights = np.where(band, BAND_WEIGHT, 1.0)
    for pc, b, m_, fl in zip(per_case, band, in_margin, in_flip):
        pc["band_margin"], pc["band_flip"], pc["ambiguous"] = int(m_), int(fl), int(b)
    band_summary = {
        "margin_rule": f"|P(severe) - T| <= {BAND_MARGIN}", "flip_rule": f"one-question flip probability >= {BAND_FLIP}",
        "n_margin": int(in_margin.sum()), "n_flip": int(in_flip.sum()), "n_band": int(band.sum()),
        "n_determinable": int(determinable.sum()),
        "band_urgent": int((band & urgent).sum()), "band_nonurgent": int((band & ~urgent).sum()),
        "determinable_urgent": int((determinable & urgent).sum()), "determinable_nonurgent": int((determinable & ~urgent).sum()),
        "band_with_narrow_red_flag": int((band & (narrow == 1)).sum()),
        "determinable_with_narrow_red_flag": int((determinable & (narrow == 1)).sum()),
        "voi_flip_quantiles": {q: round(float(np.percentile(voi_flip, q)), 3) for q in (25, 50, 75, 90)},
        "voi_best_questions": Counter(pc["voi_best_question"] for pc in per_case if pc["band_flip"]).most_common(8),
        "band_pathology_severe": int(sum(1 for pc, b in zip(per_case, band) if b and pc["pathology_severity"] <= 2)),
        "n_band_if_symptom_questions_only": int(((np.abs(p - T) <= BAND_MARGIN) | (voi_flip_symptoms >= BAND_FLIP)).sum()),
        "n_flip_symptom_questions_only": int((voi_flip_symptoms >= BAND_FLIP).sum()),
        "urgent_with_severe_pathology": int((urgent & outcome_severe).sum()),
        "urgent_with_benign_pathology_in_band": int((urgent & ~outcome_severe & band).sum()),
        "urgent_with_benign_pathology_outside_band": int((urgent & ~outcome_severe & ~band).sum()),
    }
    # How the ambiguity band relates to where models actually fail.
    err_matrix = np.stack([(models[k]["escalate"] != urgent) for k in model_names])  # models x cases
    band_summary["mean_model_error_rate_in_band"] = round(float(err_matrix[:, band].mean()), 4)
    band_summary["mean_model_error_rate_outside"] = round(float(err_matrix[:, determinable].mean()), 4)
    band_summary["share_of_all_model_errors_in_band"] = round(float(err_matrix[:, band].sum() / err_matrix.sum()), 4)
    band_summary["share_of_missed_in_band"] = round(float((err_matrix & urgent[None, :])[:, band].sum() / (err_matrix & urgent[None, :]).sum()), 4)
    band_summary["share_of_over_in_band"] = round(float((err_matrix & ~urgent[None, :])[:, band].sum() / (err_matrix & ~urgent[None, :]).sum()), 4)

    # Determinable non-urgent cases: which conditions drive the over-triage that remains outside the band.
    det_non = determinable & ~urgent
    esc_share = np.mean([models[k]["escalate"] for k in model_names], axis=0)
    by_top1 = defaultdict(lambda: {"cases": 0, "escalate_share": 0.0})
    for i in np.flatnonzero(det_non):
        key = f"{per_case[i]['top1']} (sev {per_case[i]['top1_severity']})"
        by_top1[key]["cases"] += 1
        by_top1[key]["escalate_share"] += float(esc_share[i])
    det_non_table = sorted(({"top1": k, "cases": v["cases"], "mean_escalate_share": round(v["escalate_share"] / v["cases"], 3)}
                            for k, v in by_top1.items()), key=lambda r: -r["cases"])
    band_summary["determinable_nonurgent_by_top1"] = det_non_table[:12]
    band_summary["determinable_nonurgent_with_sev3_in_top3"] = int((det_non & sev3_top3).sum())
    band_summary["determinable_nonurgent_escalate_share_sev3"] = round(float(esc_share[det_non & sev3_top3].mean()), 3) if (det_non & sev3_top3).any() else None
    band_summary["determinable_nonurgent_escalate_share_no_sev3"] = round(float(esc_share[det_non & ~sev3_top3].mean()), 3) if (det_non & ~sev3_top3).any() else None

    # ---- per-model rates in and out of the band, and reporting variants
    variants = {"full": {}, "determinable_only": {}, "band_downweighted": {}, "context_seeking": {}, "outcome_label": {},
                "avoidable_only": {}}
    p_outcome = outcome_severe.astype(float)  # 1 where the true pathology is severe: an outcome label, not a pre-test one
    # Solid cases: outside the band, and not a non-urgent case with a severity-3 diagnosis in the top 3.
    solid = determinable & ~(~urgent & sev3_top3)
    band_summary["solid_cases"] = {"n": int(solid.sum()), "urgent": int((solid & urgent).sum()), "nonurgent": int((solid & ~urgent).sum())}
    band_rows = []
    for k in model_names:
        e = models[k]["escalate"].astype(bool)
        ins = models[k]["insufficient"].astype(bool)
        err = e != urgent
        u_in, u_out = urgent & band, urgent & determinable
        n_in, n_out = ~urgent & band, ~urgent & determinable
        variants["full"][k] = score(p, e)
        variants["determinable_only"][k] = score(p[determinable], e[determinable])
        variants["band_downweighted"][k] = weighted_rates(p, e, weights)
        # context-seeking: on band cases, INSUFFICIENT counts as a deferral (neither under nor over)
        keep = ~(band & ins)
        variants["context_seeking"][k] = score(p[keep], e[keep])
        variants["outcome_label"][k] = score(p_outcome, e)
        variants["avoidable_only"][k] = score(p[solid], e[solid])
        # Three-way split of this model's errors: information-limited (band), label-disputed
        # (outside the band, over-triage on a case with a severity-3 diagnosis in the top 3), avoidable (the rest).
        info_limited = int(err[band].sum())
        label_disputed = int((err & determinable & ~urgent & sev3_top3).sum())
        avoidable_under = int((err & determinable & urgent).sum())
        avoidable_over = int((err & determinable & ~urgent & ~sev3_top3).sum())
        band_rows.append({
            "model": k, "name": SHORT.get(k, k), "status": models[k]["status"],
            "errors_total": int(err.sum()), "errors_information_limited": info_limited,
            "errors_label_disputed": label_disputed, "errors_avoidable_under": avoidable_under,
            "errors_avoidable_over": avoidable_over,
            "error_rate_in_band": round(float(err[band].mean()), 4), "error_rate_outside": round(float(err[determinable].mean()), 4),
            "under_in_band": round(float((~e & u_in).sum() / max(u_in.sum(), 1)), 4),
            "under_outside": round(float((~e & u_out).sum() / max(u_out.sum(), 1)), 4),
            "over_in_band": round(float((e & n_in).sum() / max(n_in.sum(), 1)), 4),
            "over_outside": round(float((e & n_out).sum() / max(n_out.sum(), 1)), 4),
            "errors_in_band": int(err[band].sum()), "errors_outside": int(err[determinable].sum()),
            "asks_question_rate_in_band": round(float(models[k]["asks_question"][band].mean()), 4),
            "insufficient_rate_in_band": round(float(ins[band].mean()), 4),
            "insufficient_rate_outside": round(float(ins[determinable].mean()), 4),
            "band_cases_deferred": int((band & ins).sum()),
            "band_errors_deferred": int((band & ins & err).sum()),
            "score_full": round(variants["full"][k][2], 2),
            "score_determinable": round(variants["determinable_only"][k][2], 2),
            "score_downweighted": round(variants["band_downweighted"][k][2], 2),
            "score_context_seeking": round(variants["context_seeking"][k][2], 2),
            "score_avoidable_only": round(variants["avoidable_only"][k][2], 2),
            "score_outcome_label": round(variants["outcome_label"][k][2], 2),
            "under_determinable": round(variants["determinable_only"][k][0], 4),
            "over_determinable": round(variants["determinable_only"][k][1], 4),
        })
    with open(OUT / "model_band_rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(band_rows[0]))
        w.writeheader()
        w.writerows(band_rows)

    base_scores = np.array([variants["full"][k][2] for k in model_names])
    base_rank = dict(zip(model_names, ts.competition_ranks(base_scores).tolist()))
    rank_rows, variant_summary = [], {}
    for v, d in variants.items():
        s = np.array([d[k][2] for k in model_names])
        r = dict(zip(model_names, ts.competition_ranks(s).tolist()))
        moves = {k: r[k] - base_rank[k] for k in model_names}
        variant_summary[v] = {
            "spearman_vs_full": round(float(spearmanr(base_scores, s).statistic), 3),
            "max_rank_move": int(max(abs(m) for m in moves.values())),
            "models_moving_3_or_more": [SHORT.get(k, k) for k in model_names if abs(moves[k]) >= 3],
            "top5": [SHORT.get(k, k) for k in sorted(model_names, key=lambda k: -d[k][2])[:5]],
            "score_range": [round(float(s.min()), 1), round(float(s.max()), 1)],
            "mean_score": round(float(s.mean()), 1),
        }
        for k in model_names:
            rank_rows.append({"variant": v, "model": k, "name": SHORT.get(k, k), "under": round(d[k][0], 4),
                              "over": round(d[k][1], 4), "score": round(d[k][2], 2), "rank": r[k], "rank_full": base_rank[k]})
    # Reference policies on determinable cases only, for the same comparison.
    ref_det = {}
    for key, (_, e, _kind) in policies.items():
        ref_det[key] = round(score(p[determinable], e[determinable])[2], 2)
    with open(OUT / "ranking_variants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0]))
        w.writeheader()
        w.writerows(rank_rows)
    with open(OUT / "per_case_ambiguity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case[0]))
        w.writeheader()
        w.writerows(per_case)

    # ---- posterior reconstruction checks
    recon = {
        "nb_positives_vs_ddxplus": {"pearson": round(float(pearsonr(nb_pos, p).statistic), 3),
                                    "spearman": round(float(spearmanr(nb_pos, p).statistic), 3),
                                    "urgent_label_agreement": round(float(((nb_pos >= T) == urgent).mean()), 3)},
        "nb_full_vs_ddxplus": {"pearson": round(float(pearsonr(nb_full, p).statistic), 3),
                               "spearman": round(float(spearmanr(nb_full, p).statistic), 3),
                               "urgent_label_agreement": round(float(((nb_full >= T) == urgent).mean()), 3)},
        "differential_mass_sum_mean": round(float(np.mean([sum(pr for _, pr in rows[c]["differential"]) for c in case_ids])), 4),
        "nb_top1_equals_pathology": round(nb_top1_hit / len(cases), 3),
        "ddxplus_top1_equals_pathology": round(float(np.mean([rows[c]["differential"][0][0] == rows[c]["pathology"] for c in case_ids])), 3),
        "cases_with_severe_pathology": int(outcome_severe.sum()),
    }
    human_band = {
        "under_range": HUMAN_UNDER, "over_range": HUMAN_OVER,
        "score_corners": {f"under={u:g},over={o:g}": round(ts.triage_score(u, o), 1) for u in HUMAN_UNDER for o in HUMAN_OVER},
        "score_range": [round(ts.triage_score(HUMAN_UNDER[1], HUMAN_OVER[1]), 1), round(ts.triage_score(HUMAN_UNDER[0], HUMAN_OVER[0]), 1)],
        "models_inside_band": [SHORT.get(k, k) for k in model_names
                               if HUMAN_UNDER[0] <= model_scores[k][0] <= HUMAN_UNDER[1] and HUMAN_OVER[0] <= model_scores[k][1] <= HUMAN_OVER[1]],
        "models_within_under_range": [SHORT.get(k, k) for k in model_names if HUMAN_UNDER[0] <= model_scores[k][0] <= HUMAN_UNDER[1]],
    }

    # ---- figure
    make_figure(p, models, model_names, policies, frontier)

    summary = {
        "T": T, "U": U, "O": O, "n_cases": len(cases), "n_urgent": int(urgent.sum()),
        "models_scored": [SHORT.get(k, k) for k in model_names],
        "model_scores": {SHORT.get(k, k): round(model_scores[k][2], 2) for k in model_names},
        "reference_policies": ref_rows, "reference_policies_determinable_only": ref_det,
        "noisy_posterior": noise_rows,
        "noise_matching_top_model": {"top_model": SHORT.get(top, top), "top_score": round(float(ms[0]), 2),
                                     "sigma_at_best_threshold": sig_match, "sigma_at_threshold_T": sig_match_T},
        "band": band_summary, "variants": variant_summary, "model_band_rates": band_rows,
        "posterior_reconstruction": recon, "published_human_band": human_band,
        "flag_prevalence": {k: int(sum(f[k] for f in flags_by_case)) for k in BROAD_FLAGS + ("chest_pain_typical", "chest_pain_with_dyspnea")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # ---- console digest
    print("\nreference policies (score, under, over, would rank among models):")
    for r in sorted(ref_rows, key=lambda r: -r["score"]):
        print(f"  {r['policy']:22s} {r['score']:5.1f} [{r['score_ci_lo']:.0f}-{r['score_ci_hi']:.0f}]  under {100*r['under']:4.1f}%  over {100*r['over']:5.1f}%  rank {r['would_rank']}/{r['n_models']}")
    print("\nnoisy posterior (sigma: score at T / best-threshold score):")
    for r in noise_rows:
        print(f"  {r['sigma_logit']:.2f}: {r['threshold_T_score_mean']:5.1f} / {r['best_threshold_score_mean']:5.1f} at t={r['best_threshold']}")
    print(f"\nband: {band_summary['n_band']} cases ({band_summary['n_margin']} margin, {band_summary['n_flip']} flip); "
          f"mean model error in band {band_summary['mean_model_error_rate_in_band']:.3f} vs outside {band_summary['mean_model_error_rate_outside']:.3f}; "
          f"{100*band_summary['share_of_all_model_errors_in_band']:.0f}% of errors in band")
    for v, d in variant_summary.items():
        print(f"  {v:18s} rho {d['spearman_vs_full']:.2f} max move {d['max_rank_move']} top5 {d['top5']} range {d['score_range']}")
    print("posterior reconstruction:", json.dumps(recon))
    print("published human band:", json.dumps(human_band))
    print(f"wrote {OUT.relative_to(ROOT)}/")


def make_figure(p, models, model_names, policies, frontier):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # iso-score contours
    xs = np.linspace(0, 1, 200)
    for s_ in (20, 40, 60, 80):
        d = ts.isoscore_distance(s_)
        ys = U * np.sqrt(np.clip(d**2 - (xs / O) ** 2, 0, None))
        ax.plot(xs, ys, color="#e4e3df", lw=1, zorder=1)
        ax.text(xs[np.argmax(ys > 0)] + 0.005 if s_ > 40 else 0.005, min(U * d, 0.42), f"{s_}", color="#898781", fontsize=7)
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((HUMAN_OVER[0], HUMAN_UNDER[0]), HUMAN_OVER[1] - HUMAN_OVER[0], HUMAN_UNDER[1] - HUMAN_UNDER[0],
                           facecolor="#eda100", alpha=0.18, edgecolor="#eda100", lw=1, zorder=1.5,
                           label="Published human telephone triage (under 3.7-11%, over 4-20%)"))
    for sigma, alpha, at in ((0.5, 0.5, 0.55), (1.0, 0.9, 0.42), (2.0, 0.5, 0.30)):
        curve = frontier[sigma]
        ax.plot([c[2] for c in curve], [c[1] for c in curve], color="#1baf7a", lw=2, alpha=alpha, zorder=2)
        c0 = curve[int(len(curve) * at)]
        ax.text(c0[2] + 0.012, c0[1], f"sigma {sigma:g}", color="#0b0b0b", fontsize=7, va="center")
    mx = [ts.score_model(p, models[k]["escalate"]).over for k in model_names]
    my = [ts.score_model(p, models[k]["escalate"]).under for k in model_names]
    ax.scatter(mx, my, s=36, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1, zorder=4, label="Models (19)")
    offsets = {model_names[0]: (6, 2), model_names[1]: (6, -8)}
    for k in model_names[:2]:
        m = ts.score_model(p, models[k]["escalate"])
        ax.annotate(SHORT.get(k, k), (m.over, m.under), xytext=offsets[k], textcoords="offset points", fontsize=7, color="#0b0b0b")
    # (label, marker, offset). Diamonds read the case sheet only; triangles read the DDXPlus differential.
    show = {
        "oracle": ("oracle", "^", (6, 4)), "any_severe_anywhere": ("any severe dx in differential", "^", (6, 4)),
        "cautious_sev3_T": ("severity 1-3 urgent", "^", (6, -8)),
        "redflag_narrow": ("red flags, narrow", "D", (6, -3)), "redflag_broad_1": ("MTS-style, any one", "D", (-8, 8)),
        "redflag_broad_2": ("MTS-style, two or more", "D", (6, 4)), "top1_severe": ("top-1 severe", "^", (6, -10)),
        "nb_positives_T": ("naive Bayes", "D", (6, 6)), "consensus": ("model consensus", "D", (6, -9)),
        "always_escalate": ("always escalate", "D", (-70, 6)),
    }
    for marker, lab in (("D", "Reference policies (case sheet only)"), ("^", "Reference policies (DDXPlus differential)")):
        keys = [k for k, v in show.items() if v[1] == marker]
        px = [ts.score_model(p, policies[k][1]).over for k in keys]
        py = [ts.score_model(p, policies[k][1]).under for k in keys]
        ax.scatter(px, py, s=54, marker=marker, color="#eb6834", edgecolor="#fcfcfb", linewidth=1, zorder=5, label=lab)
        for k, x, y in zip(keys, px, py):
            ax.annotate(show[k][0], (x, y), xytext=show[k][2], textcoords="offset points", fontsize=7, color="#0b0b0b")
    ax.plot([], [], color="#1baf7a", lw=2, label="Noisy-posterior frontier (sigma 0.5, 1, 2)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 0.45)
    ax.set_xlabel("Over-triage rate (non-urgent cases escalated)", color="#52514e")
    ax.set_ylabel("Under-triage rate (urgent cases not escalated)", color="#52514e")
    ax.set_title("Where models sit against human-achievable reference policies", loc="left", fontsize=11, color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors="#898781")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_reference_band.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
