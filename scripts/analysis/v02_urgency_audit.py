"""Audit computations for the v0.2 switch from DDXPlus severity to NTS urgency.

Why: docs/v0.2-urgency-audit.md lists every claim on the board and in the docs
that rests on the severity key, the P(severe) >= 0.15 label or the "2+ levels
off" rule. This script recomputes, from DDXPlus data and stored predictions
only (no inference), the numbers that decide whether each finding survives:

1. Condition-level agreement between DDXPlus severity and the NTS level
   (spec/acuity_reference_levels.csv), on 49 conditions and on the 47 adult ones.
2. On eval-250-v0: the urgent share under each label, the cross-tabs between
   the NTS key and the old labels, and the label ceiling (the severity key
   scored as a model against the NTS key), with case and condition bootstraps.
3. Red-flag cases (bleeding code, thunderclap headache, fever with
   immunosuppression) and how many are urgent under NTS.
4. The 19 rows with per-case predictions re-scored against the NTS key.
5. The planned v0.2 sample (10 adults per condition, seed 20260923): level mix,
   modifier fires, ceiling with a condition bootstrap, constant-policy floors.
6. A naive-Bayes reader of the case sheet, scored against the NTS key, to test
   the "label is not recoverable from the sheet" finding.

Usage:
    .venv/bin/python scripts/analysis/v02_urgency_audit.py

Outputs go to results/analysis/v02_urgency_audit/ (gitignored).
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import label_ceiling as lc  # noqa: E402
from evaluator import triage_score as ts  # noqa: E402

COND = ROOT / "data/ddxplus_v0/release_conditions.json"
TEST_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
CASES = ROOT / "data/test_sets/eval-250-v0.json"
ACUITY = ROOT / "spec/acuity_reference_levels.csv"
SEVREF = ROOT / "spec/ddxplus_severity_reference.csv"
BOARD = ROOT / "leaderboard/triage-scores.json"
NB_CACHE = ROOT / "results/analysis/failure_modes/human_baseline/ddxplus_likelihoods.json"
OUT = ROOT / "results/analysis/v02_urgency_audit"

SEED = 20260923
PER_CONDITION = 10
URGENT_MAX_NTS = 3  # spec section 4: level 1-3 is urgent
URGENT_MAX_SEV = 2  # v0.1: severity 1-2 is severe
T = ts.URGENT_THRESHOLD
U, O = ts.TOLERATED_UNDER_TRIAGE, ts.TOLERATED_OVER_TRIAGE
N_BOOT = 2000

BLEED = {"E_210", "E_140", "E_179", "E_45"}
IMMUNO = {"E_227", "E_2", "E_44"}
FEVER = "E_91"
HEAD_LOCATIONS = {"V_89", "V_125", "V_126", "V_166", "V_167", "V_108", "V_109", "V_25", "V_62", "V_124"}

CHRONIC = {"E_123", "E_31", "E_124", "E_106", "E_69", "E_113", "E_126"}
IMMUNO_WIDE = {"E_227", "E_2", "E_34"}


# ---------------------------------------------------------------- scales and modifiers


def load_levels():
    cond = json.loads(COND.read_text())
    sev = {k: int(v["severity"]) for k, v in cond.items()}
    with open(SEVREF) as f:
        ref = {r["condition"]: int(r["reference_level"]) for r in csv.DictReader(f)}
    with open(ACUITY) as f:
        nts = {r["condition"]: int(r["scale_level"]) for r in csv.DictReader(f)}
    missing = set(cond) - set(nts)
    if missing:
        raise SystemExit(f"{ACUITY} lacks {sorted(missing)}")
    return sev, ref, nts


def nts_level(path: str, age: int, codes: set[str], nts: dict[str, int]) -> int:
    """NTS level of the true condition with the patient-level modifiers from spec/acuity_reference_levels.csv."""
    base = nts[path]
    if path == "Bronchiolitis" and age < 1 and codes & {"E_160", "E_139"}:
        return 2
    if path == "Bronchospasm / acute asthma exacerbation" and codes & {"E_101", "E_46"}:
        return 1
    if path in ("Influenza", "Bronchitis") and (age >= 65 or "E_167" in codes or codes & CHRONIC or codes & IMMUNO_WIDE):
        return 3
    if path == "Pneumonia" and (age >= 65 or codes & {"E_123", "E_31", "E_106"} or codes & IMMUNO_WIDE):
        return 2
    if path == "Acute otitis media" and codes & {"E_227", "E_69", "E_106", "E_123", "E_31", "E_113"}:
        return 3
    if path in ("Acute rhinosinusitis", "Viral pharyngitis") and "E_227" in codes:
        return 3
    if path == "Whooping cough" and age < 1:
        return 3
    return base


def bases(ev_list) -> set[str]:
    return {e.split("_@_")[0] for e in ev_list}


def red_flags(ev_list) -> dict[str, bool]:
    codes = bases(ev_list)
    vals = {}
    for e in ev_list:
        if "_@_" in e:
            k, v = e.split("_@_", 1)
            vals.setdefault(k, []).append(v)
    intensity = max((int(v) for v in vals.get("E_56", []) if v.isdigit()), default=None)
    onset = max((int(v) for v in vals.get("E_59", []) if v.isdigit()), default=None)
    head = any(v in HEAD_LOCATIONS for v in vals.get("E_55", []))
    thunder = bool(head and onset is not None and onset >= 7 and intensity is not None and intensity >= 8)
    return {
        "bleed": bool(codes & BLEED),
        "thunderclap": thunder,
        "fever_immuno": FEVER in codes and bool(codes & IMMUNO),
    }


# ---------------------------------------------------------------- alternative keys

# Conditions at the urgent line whose NTS placement docs/triage-scale-anchor.md marks low or medium confidence.
# "up": the two level-4 placements (both low confidence) move to 3. "down": the three level-3 placements the
# anchor doc adds relative to the evidence reference move to 4. Scoring one reading against the other bounds
# how much of a model's A and B the mapping's own uncertainty can explain.
LINE_UP = {"Atrial fibrillation": 3, "Stable angina": 3}
LINE_DOWN = {"Scombroid food poisoning": 4, "Cluster headache": 4, "Localized edema": 4}


def alt_keys(rows: list[dict], nts: dict[str, int]) -> dict[str, np.ndarray]:
    out = {}
    for tag, moves in (("mapping_up", LINE_UP), ("mapping_down", LINE_DOWN), ("mapping_both", {**LINE_UP, **LINE_DOWN})):
        alt = {**nts, **moves}
        lvl = np.array([nts_level(r["path"], r["age"], bases(r["ev"]), alt) for r in rows])
        out[tag] = lvl
    return out


# ---------------------------------------------------------------- agreement helpers


def qwk(a, b, k=5) -> float:
    return lc.cohen_kappa(np.asarray(a) - 1, np.asarray(b) - 1, n_levels=k, weights="quadratic")


def crosstab(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    return {"both": int((x & y).sum()), "x_only": int((x & ~y).sum()), "y_only": int((~x & y).sum()), "neither": int((~x & ~y).sum())}


def score_against(key: np.ndarray, escalated: np.ndarray) -> dict:
    under, over = ts.rates_against_label(key, escalated)
    return {"under": under, "over": over, "score": ts.triage_score(under, over), "kappa": lc.cohen_kappa(key.astype(int), escalated.astype(int))}


def cluster_bootstrap(key: np.ndarray, decisions: dict[str, np.ndarray], clusters: list[str], n_boot=N_BOOT, seed=SEED) -> dict:
    """Resample conditions (clusters) with replacement; 95% intervals for under, over, score, kappa."""
    key = np.asarray(key, bool)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    names = sorted(groups)
    idx_by = [np.array(groups[c]) for c in names]
    rng = np.random.default_rng(seed)
    out = {k: {"under": [], "over": [], "score": [], "kappa": []} for k in decisions}
    for _ in range(n_boot):
        draw = rng.integers(0, len(names), size=len(names))
        idx = np.concatenate([idx_by[j] for j in draw])
        kk = key[idx]
        for k, e in decisions.items():
            ee = np.asarray(e, bool)[idx]
            under, over = ts.rates_against_label(kk, ee)
            out[k]["under"].append(under)
            out[k]["over"].append(over)
            out[k]["score"].append(ts.triage_score(under, over))
            out[k]["kappa"].append(lc.cohen_kappa(kk.astype(int), ee.astype(int)))
    return {k: {m: np.nanpercentile(v, [2.5, 97.5]).tolist() for m, v in d.items()} for k, d in out.items()}


def case_bootstrap(key: np.ndarray, decisions: dict[str, np.ndarray], n_boot=N_BOOT, seed=SEED) -> dict:
    n = len(key)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    G = np.asarray(key, float)[idx]
    n_urg = np.maximum(G.sum(axis=1), 1.0)
    n_non = np.maximum((1 - G).sum(axis=1), 1.0)
    out = {}
    for k, e in decisions.items():
        E = np.asarray(e, float)[idx]
        under = ((1 - E) * G).sum(axis=1) / n_urg
        over = (E * (1 - G)).sum(axis=1) / n_non
        score = ts.triage_score_from_distance(ts.tolerance_distance(under, over))
        kap = lc.kappa_from_draws(np.asarray(e, int)[idx], np.asarray(key, int)[idx])
        out[k] = {m: np.nanpercentile(v, [2.5, 97.5]).tolist() for m, v in
                  {"under": under, "over": over, "score": score, "kappa": kap}.items()}
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


# ---------------------------------------------------------------- data


def read_test_rows(want: dict[int, str] | None = None) -> list[dict]:
    rows = []
    with open(TEST_CSV) as f:
        for i, r in enumerate(csv.DictReader(f)):
            if want is not None and i not in want:
                continue
            rows.append({"i": i, "case_id": want[i] if want else f"ddxplus_{i}", "path": r["PATHOLOGY"], "age": int(r["AGE"]),
                         "sex": r["SEX"], "ev": ast.literal_eval(r["EVIDENCES"]), "diff": ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"])})
    return rows


def stratified_adults(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if r["age"] >= 18:
            by[r["path"]].append(r)
    rng = np.random.default_rng(SEED)
    out = []
    for p in sorted(by):
        pool = by[p]
        k = min(PER_CONDITION, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        out.extend(pool[j] for j in sorted(idx))
    return out


def label_frame(rows: list[dict], sev, ref, nts) -> dict[str, np.ndarray]:
    """Every label variant per row, in row order."""
    p_sev = np.array([sum(p for n, p in r["diff"] if sev[n] <= URGENT_MAX_SEV) for r in rows])
    p_nts = np.array([sum(p for n, p in r["diff"] if nts[n] <= URGENT_MAX_NTS) for r in rows])
    lvl = np.array([nts_level(r["path"], r["age"], bases(r["ev"]), nts) for r in rows])
    lvl_nomod = np.array([nts[r["path"]] for r in rows])
    return {
        "p_severe": p_sev,
        "p_nts_urgent": p_nts,
        "nts_level": lvl,
        "nts_level_nomod": lvl_nomod,
        "sev_level": np.array([sev[r["path"]] for r in rows]),
        "ref_level": np.array([ref[r["path"]] for r in rows]),
        "L_T": p_sev >= T,                                   # v0.1 board label
        "L_path_sev": np.array([sev[r["path"]] <= URGENT_MAX_SEV for r in rows]),  # severity key at its own line
        "L_path_ref": np.array([ref[r["path"]] <= URGENT_MAX_SEV for r in rows]),  # evidence reference key
        "L_nts": lvl <= URGENT_MAX_NTS,                      # v0.2 primary key
        "L_nts_nomod": lvl_nomod <= URGENT_MAX_NTS,
        "L_diff_nts": p_nts >= T,                            # NTS mass on the differential, at T
    }


# ---------------------------------------------------------------- sections


def section_conditions(sev, ref, nts) -> dict:
    names = sorted(sev)
    adults = [n for n in names if n not in ("Croup", "Bronchiolitis")]
    out = {}
    for tag, ns in (("all49", names), ("adult47", adults)):
        s = np.array([sev[n] for n in ns]); r = np.array([ref[n] for n in ns]); t = np.array([nts[n] for n in ns])
        out[tag] = {
            "n": len(ns),
            "nts_mix": dict(Counter(int(x) for x in t)),
            "sev_mix": dict(Counter(int(x) for x in s)),
            "qwk_sev_vs_nts": qwk(s, t), "qwk_ref_vs_nts": qwk(r, t), "qwk_sev_vs_ref": qwk(s, r),
            "within_one_sev_nts": int((np.abs(s - t) <= 1).sum()),
            "urgent_nts": int((t <= 3).sum()), "urgent_sev": int((s <= 2).sum()), "urgent_ref": int((r <= 2).sum()),
            "crosstab_sev2_vs_nts3": crosstab(s <= 2, t <= 3),
            "sev_urgent_not_nts": [n for n in ns if sev[n] <= 2 and nts[n] > 3],
            "nts_urgent_not_sev": [n for n in ns if nts[n] <= 3 and sev[n] > 2],
            "two_plus_apart": [(n, sev[n], nts[n]) for n in ns if abs(sev[n] - nts[n]) >= 2],
            "binary_kappa_sev2_vs_nts3": lc.cohen_kappa((s <= 2).astype(int), (t <= 3).astype(int)),
        }
    return out


def load_board_decisions(case_ids: list[str]) -> dict[str, np.ndarray]:
    """Escalation vectors for every row with a prediction file, resolved as scripts/build_triage_board.py does."""
    from scripts.build_triage_board import resolve
    decisions = {}
    for f in sorted(glob.glob(str(ROOT / "leaderboard/*-eval.json"))):
        ev = json.loads(Path(f).read_text())
        path, status, _ = resolve(ev)
        if path is None:
            continue
        raw = json.loads(path.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        decisions[ev["model"]] = ts.escalation_decisions(preds, case_ids)
    return decisions


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def naive_bayes_posteriors(rows: list[dict], cond_names: list[str]) -> np.ndarray | None:
    """Posterior over conditions from listed evidence codes (positives only), using the cached likelihood scan."""
    if not NB_CACHE.exists():
        return None
    lik = json.loads(NB_CACHE.read_text())
    n_path = lik["n_pathology"]
    total = sum(n_path.values())
    P = np.zeros((len(rows), len(cond_names)))
    for k, r in enumerate(rows):
        codes = bases(r["ev"])
        logp = np.full(len(cond_names), -np.inf)
        for i, c in enumerate(cond_names):
            n = n_path.get(c, 0)
            if n == 0:
                continue
            cc = lik["code_counts"].get(c, {})
            lp = np.log((n + 0.5) / (total + 0.5 * len(cond_names)))
            for q in codes:
                lp += np.log((cc.get(q, 0) + 0.5) / (n + 1.0))
            logp[i] = lp
        logp -= logp.max()
        p = np.exp(logp)
        P[k] = p / p.sum()
    return P


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sev, ref, nts = load_levels()
    cond_names = sorted(sev)
    res = {"conditions": section_conditions(sev, ref, nts)}
    c47 = res["conditions"]["adult47"]
    print(f"[1] conditions: 47 adults: NTS mix {c47['nts_mix']}, urgent (<=3) {c47['urgent_nts']} vs severity<=2 {c47['urgent_sev']}; "
          f"QWK sev-vs-NTS {c47['qwk_sev_vs_nts']:.3f} (49: {res['conditions']['all49']['qwk_sev_vs_nts']:.3f}); "
          f"binary kappa at the line {c47['binary_kappa_sev2_vs_nts3']:.2f}")
    print(f"    severity-urgent but NTS-routine: {c47['sev_urgent_not_nts']}; NTS-urgent but severity-routine: {c47['nts_urgent_not_sev']}")

    # ---- eval-250-v0
    cases = json.loads(CASES.read_text())["cases"]
    case_ids = [c["case_id"] for c in cases]
    want = {int(cid.split("_")[1]): cid for cid in case_ids}
    rows = read_test_rows(want)
    order = {r["case_id"]: r for r in rows}
    rows = [order[c] for c in case_ids]
    v0 = np.array([bool(c["escalation_required"]) for c in cases])
    L = label_frame(rows, sev, ref, nts)
    key = L["L_nts"]
    paths = [r["path"] for r in rows]
    ages = np.array([r["age"] for r in rows])
    e250 = {
        "n": len(rows), "adults": int((ages >= 18).sum()), "age_min": int(ages.min()),
        "urgent_counts": {"v0_top3": int(v0.sum()), "T_0.15": int(L["L_T"].sum()), "pathology_sev<=2": int(L["L_path_sev"].sum()),
                          "pathology_ref<=2": int(L["L_path_ref"].sum()), "nts<=3": int(key.sum()), "nts<=3_no_modifiers": int(L["L_nts_nomod"].sum()),
                          "diff_nts_mass>=T": int(L["L_diff_nts"].sum())},
        "nts_level_mix": dict(Counter(int(x) for x in L["nts_level"])),
        "modifier_fires": int((L["nts_level"] != L["nts_level_nomod"]).sum()),
        "modifier_fires_by_condition": dict(Counter(p for p, a, b in zip(paths, L["nts_level"], L["nts_level_nomod"]) if a != b)),
        "crosstab_vs_nts": {k: crosstab(L[k], key) for k in ("L_T", "L_path_sev", "L_path_ref", "L_diff_nts")},
        "crosstab_vs_nts_v0": crosstab(v0, key),
        "qwk_sev_vs_nts_cases": qwk(L["sev_level"], L["nts_level"]),
        "p_severe_mean": float(L["p_severe"].mean()), "p_nts_urgent_mean": float(L["p_nts_urgent"].mean()),
    }
    # label ceiling against the NTS key: alternate labels scored as models
    ak = alt_keys(rows, nts)
    alts = {f"nts_{k}": v <= URGENT_MAX_NTS for k, v in ak.items()}
    e250["alt_key_levels_changed"] = {k: int((v != L["nts_level"]).sum()) for k, v in ak.items()}
    e250["alt_key_qwk_vs_primary"] = {k: qwk(v, L["nts_level"]) for k, v in ak.items()}
    e250["qwk_no_modifiers_vs_primary"] = qwk(L["nts_level_nomod"], L["nts_level"])
    alts |= {"severity_key_pathology<=2": L["L_path_sev"], "reference_key_pathology<=2": L["L_path_ref"],
            "v0.1_board_label_P(severe)>=0.15": L["L_T"], "v0_top3_label": v0, "nts_no_modifiers": L["L_nts_nomod"],
            "differential_nts_mass>=0.15": L["L_diff_nts"], "always_escalate": np.ones(len(rows), bool), "never_escalate": np.zeros(len(rows), bool)}
    e250["ceiling"] = {k: score_against(key, v) for k, v in alts.items()}
    cb = case_bootstrap(key, alts)
    kb = cluster_bootstrap(key, alts, paths)
    for k in alts:
        e250["ceiling"][k]["ci_case"] = cb[k]
        e250["ceiling"][k]["ci_condition"] = kb[k]
    # the old ceiling for comparison: reference label vs the T label (what the board shows now)
    e250["old_ceiling_reference_vs_T"] = score_against(L["L_T"], np.array([lc.p_severe(r["diff"], ref) >= T for r in rows]))
    print(f"[2] eval-250: urgent {e250['urgent_counts']}; modifiers fire on {e250['modifier_fires']} cases {e250['modifier_fires_by_condition']}")
    for k in ("severity_key_pathology<=2", "v0.1_board_label_P(severe)>=0.15", "reference_key_pathology<=2", "nts_no_modifiers",
              "nts_mapping_up", "nts_mapping_down", "nts_mapping_both", "always_escalate"):
        c = e250["ceiling"][k]
        print(f"    {k:36s} vs NTS key: A {100*c['under']:4.1f}% B {100*c['over']:4.1f}% score {c['score']:5.1f} "
              f"[cond CI {c['ci_condition']['score'][0]:.0f}-{c['ci_condition']['score'][1]:.0f}] kappa {c['kappa']:.2f} [{c['ci_condition']['kappa'][0]:.2f}-{c['ci_condition']['kappa'][1]:.2f}]")
    oc = e250["old_ceiling_reference_vs_T"]
    print(f"    (current board ceiling, reference label vs T label: A {100*oc['under']:.1f}% B {100*oc['over']:.1f}% score {oc['score']:.1f} kappa {oc['kappa']:.2f})")

    # ---- red flags on eval-250
    fl = [red_flags(r["ev"]) for r in rows]
    tier1 = np.array([f["bleed"] or f["thunderclap"] or f["fever_immuno"] for f in fl])
    e250["red_flags"] = {
        "tier1_cases": int(tier1.sum()),
        "tier1_by_flag": {k: int(sum(f[k] for f in fl)) for k in ("bleed", "thunderclap", "fever_immuno")},
        "tier1_urgent_nts": int((tier1 & key).sum()), "tier1_nonurgent_nts": int((tier1 & ~key).sum()),
        "tier1_urgent_T": int((tier1 & L["L_T"]).sum()), "tier1_nonurgent_T": int((tier1 & ~L["L_T"]).sum()),
        "tier1_nonurgent_nts_by_condition": dict(Counter(p for p, t1, k in zip(paths, tier1, key) if t1 and not k)),
        "nonurgent_nts_after_removal": int((~key & ~tier1).sum()),
    }
    rf = e250["red_flags"]
    print(f"[3] red flags (TIER1) on eval-250: {rf['tier1_cases']} cases {rf['tier1_by_flag']}; NTS-urgent {rf['tier1_urgent_nts']}, "
          f"NTS-routine {rf['tier1_nonurgent_nts']} {rf['tier1_nonurgent_nts_by_condition']}; B denominator after removal {rf['nonurgent_nts_after_removal']}")

    # ---- models re-scored against the NTS key on eval-250
    decisions = load_board_decisions(case_ids)
    board = json.loads(BOARD.read_text())
    models = {}
    for name, e in decisions.items():
        cur = board["rows"].get(name, {})
        m_nts = score_against(key, e)
        m_nts_rf = score_against(key[~tier1], e[~tier1])  # red flags out of both denominators (approximation; spec removes them from B only)
        under_rf, _ = ts.rates_against_label(key, e)
        _, over_rf = ts.rates_against_label(key[~tier1], e[~tier1])
        models[name] = {
            "current_score": cur.get("score"), "current_under": cur.get("under"), "current_over": cur.get("over"),
            "nts_under": m_nts["under"], "nts_over": m_nts["over"], "nts_score": m_nts["score"], "nts_kappa": m_nts["kappa"],
            "nts_over_redflags_removed": over_rf, "nts_score_redflags_removed": ts.triage_score(under_rf, over_rf),
            "escalate_rate": float(np.asarray(e).mean()),
        }
    names = [n for n in models if models[n]["current_score"] is not None]
    cur = np.array([models[n]["current_score"] for n in names])
    new = np.array([models[n]["nts_score"] for n in names])
    new_rf = np.array([models[n]["nts_score_redflags_removed"] for n in names])
    rank_cur = ts.competition_ranks(cur); rank_new = ts.competition_ranks(new); rank_rf = ts.competition_ranks(new_rf)
    for n, a, b, c in zip(names, rank_cur, rank_new, rank_rf):
        models[n].update(rank_current=int(a), rank_nts=int(b), rank_nts_redflags_removed=int(c))
    inside_new = [n for n in names if models[n]["nts_under"] <= U and models[n]["nts_over"] <= O]
    inside_rf = [n for n in names if models[n]["nts_under"] <= U and models[n]["nts_over_redflags_removed"] <= O]
    ae = score_against(key, np.ones(len(rows), bool))
    e250["models"] = models
    e250["model_summary"] = {
        "n_rows": len(names), "spearman_current_vs_nts": spearman(cur, new), "spearman_current_vs_nts_redflags_removed": spearman(cur, new_rf),
        "movers_3plus": [n for n, a, b in zip(names, rank_cur, rank_new) if abs(int(a) - int(b)) >= 3],
        "inside_box_nts": inside_new, "inside_box_nts_redflags_removed": inside_rf,
        "nts_under_range": [float(min(models[n]["nts_under"] for n in names)), float(max(models[n]["nts_under"] for n in names))],
        "nts_over_range": [float(min(models[n]["nts_over"] for n in names)), float(max(models[n]["nts_over"] for n in names))],
        "nts_score_range": [float(new.min()), float(new.max())],
        "always_escalate_nts": ae, "always_escalate_would_rank": int(1 + (new > ae["score"]).sum()),
        "best_model_nts_kappa": max(models[n]["nts_kappa"] for n in names),
        "top5_nts": [n for n, _ in sorted(((n, models[n]["nts_score"]) for n in names), key=lambda x: -x[1])[:5]],
        "top5_current": [n for n, _ in sorted(((n, models[n]["current_score"]) for n in names), key=lambda x: -x[1])[:5]],
    }
    ms = e250["model_summary"]
    print(f"[4] {ms['n_rows']} rows re-scored vs NTS key: score range {ms['nts_score_range'][0]:.1f}-{ms['nts_score_range'][1]:.1f}, "
          f"under {100*ms['nts_under_range'][0]:.1f}-{100*ms['nts_under_range'][1]:.1f}%, over {100*ms['nts_over_range'][0]:.1f}-{100*ms['nts_over_range'][1]:.1f}%; "
          f"rho vs current {ms['spearman_current_vs_nts']:.2f} (red flags removed {ms['spearman_current_vs_nts_redflags_removed']:.2f}); "
          f"movers>=3: {len(ms['movers_3plus'])}; inside box: {ms['inside_box_nts']} / rf-removed {ms['inside_box_nts_redflags_removed']}; "
          f"always-escalate scores {ae['score']:.1f}, would rank {ms['always_escalate_would_rank']}; best model kappa {ms['best_model_nts_kappa']:.2f}")
    print("    top 5 now:", ms["top5_current"]); print("    top 5 NTS:", ms["top5_nts"])

    # ---- naive-Bayes reader vs the NTS key (eval-250)
    P = naive_bayes_posteriors(rows, cond_names)
    if P is not None:
        top1 = [cond_names[i] for i in P.argmax(axis=1)]
        nb_urgent_mass = np.array([sum(P[k, i] for i, c in enumerate(cond_names) if nts[c] <= URGENT_MAX_NTS) for k in range(len(rows))])
        nb_sev_mass = np.array([sum(P[k, i] for i, c in enumerate(cond_names) if sev[c] <= URGENT_MAX_SEV) for k in range(len(rows))])
        nb_call = np.array([nts_level(t1, r["age"], bases(r["ev"]), nts) <= URGENT_MAX_NTS for t1, r in zip(top1, rows)])
        e250["naive_bayes"] = {
            "top1_pathology_accuracy": float(np.mean([t == p for t, p in zip(top1, paths)])),
            "agreement_with_nts_key_top1_call": float((nb_call == key).mean()),
            "agreement_with_T_label_mass>=T": float(((nb_sev_mass >= T) == L["L_T"]).mean()),
            "auroc_nts_mass_vs_nts_key": auroc(nb_urgent_mass, key),
            "auroc_sev_mass_vs_T_label": auroc(nb_sev_mass, L["L_T"]),
            "auroc_sev_mass_vs_pathology_sev_key": auroc(nb_sev_mass, L["L_path_sev"]),
            "scored_vs_nts_key_top1_call": score_against(key, nb_call),
            "scored_vs_nts_key_mass>=0.5": score_against(key, nb_urgent_mass >= 0.5),
        }
        nb = e250["naive_bayes"]
        print(f"[6] naive Bayes: top-1 pathology {100*nb['top1_pathology_accuracy']:.0f}%; agrees with NTS key {100*nb['agreement_with_nts_key_top1_call']:.0f}% "
              f"(with T label {100*nb['agreement_with_T_label_mass>=T']:.0f}%); AUROC vs NTS key {nb['auroc_nts_mass_vs_nts_key']:.2f}, vs T label {nb['auroc_sev_mass_vs_T_label']:.2f}; "
              f"scored vs NTS key: A {100*nb['scored_vs_nts_key_top1_call']['under']:.1f}% B {100*nb['scored_vs_nts_key_top1_call']['over']:.1f}% score {nb['scored_vs_nts_key_top1_call']['score']:.1f}")
    res["eval250"] = e250

    # ---- planned v0.2 sample: 10 adults per condition
    all_rows = read_test_rows()
    sample = stratified_adults(all_rows)
    Ls = label_frame(sample, sev, ref, nts)
    ks = Ls["L_nts"]
    sp = [r["path"] for r in sample]
    fls = [red_flags(r["ev"]) for r in sample]
    t1s = np.array([f["bleed"] or f["thunderclap"] or f["fever_immuno"] for f in fls])
    aks = alt_keys(sample, nts)
    alts_s = {f"nts_{k}": v <= URGENT_MAX_NTS for k, v in aks.items()}
    alts_s |= {"severity_key_pathology<=2": Ls["L_path_sev"], "reference_key_pathology<=2": Ls["L_path_ref"], "nts_no_modifiers": Ls["L_nts_nomod"],
              "differential_severity_mass>=0.15": Ls["L_T"], "differential_nts_mass>=0.15": Ls["L_diff_nts"],
              "always_escalate": np.ones(len(sample), bool), "always_U3_or_more_urgent": np.ones(len(sample), bool), "never_escalate": np.zeros(len(sample), bool)}
    ceil_s = {k: score_against(ks, v) for k, v in alts_s.items()}
    kb_s = cluster_bootstrap(ks, alts_s, sp)
    cb_s = case_bootstrap(ks, alts_s)
    for k in alts_s:
        ceil_s[k]["ci_condition"] = kb_s[k]; ceil_s[k]["ci_case"] = cb_s[k]
    # DDXPlus natural adult mix
    adults_all = [r for r in all_rows if r["age"] >= 18]
    nat_lvl = np.array([nts_level(r["path"], r["age"], bases(r["ev"]), nts) for r in adults_all])
    nat_sev = np.array([sev[r["path"]] for r in adults_all])
    v02 = {
        "n": len(sample), "conditions": len(set(sp)), "age65plus": int(sum(r["age"] >= 65 for r in sample)),
        "nts_level_mix": dict(Counter(int(x) for x in Ls["nts_level"])), "sev_level_mix": dict(Counter(int(x) for x in Ls["sev_level"])),
        "urgent_nts": int(ks.sum()), "urgent_sev": int(Ls["L_path_sev"].sum()), "urgent_ref": int(Ls["L_path_ref"].sum()),
        "modifier_fires": int((Ls["nts_level"] != Ls["nts_level_nomod"]).sum()),
        "modifier_fires_by_condition": dict(Counter(p for p, a, b in zip(sp, Ls["nts_level"], Ls["nts_level_nomod"]) if a != b)),
        "crosstab_sev_key_vs_nts": crosstab(Ls["L_path_sev"], ks),
        "qwk_sev_vs_nts_cases": qwk(Ls["sev_level"], Ls["nts_level"]),
        "ceiling": ceil_s,
        "red_flags": {"tier1_cases": int(t1s.sum()), "tier1_by_flag": {k: int(sum(f[k] for f in fls)) for k in ("bleed", "thunderclap", "fever_immuno")},
                      "tier1_urgent_nts": int((t1s & ks).sum()), "tier1_nonurgent_nts": int((t1s & ~ks).sum()),
                      "tier1_nonurgent_by_condition": dict(Counter(p for p, a, k in zip(sp, t1s, ks) if a and not k)),
                      "B_denominator_headline": int((~ks & ~t1s).sum())},
        "natural_adult_mix": {"n": len(adults_all), "urgent_nts_share": float((nat_lvl <= 3).mean()), "severe_sev_share": float((nat_sev <= 2).mean()),
                              "nts_level_mix": {int(k): float(v / len(adults_all)) for k, v in Counter(nat_lvl.tolist()).items()}},
        "precision_A_at_5pct_binomial_halfwidth": float(1.96 * np.sqrt(0.05 * 0.95 / max(int(ks.sum()), 1))),
    }
    # constant-level policies under the v0.2 strict definitions (A: t urgent, a not; B: t not urgent, a urgent)
    # Under the v0.2 line every constant level is either always-urgent (A 0, B 100% of non-urgent cases) or
    # always-routine (A 100% of urgent cases, B 0); the share-of-all-cases figures show the mix.
    v02["constant_policies"] = {f"always_{a}": {"A": 0.0 if a <= 3 else 1.0, "B": 1.0 if a <= 3 else 0.0,
                                               "events_share_of_all_cases": float((~ks).mean()) if a <= 3 else float(ks.mean())}
                                for a in range(1, 6)}
    res["v02_sample"] = v02
    print(f"[5] v0.2 sample: {v02['n']} adults over {v02['conditions']} conditions; NTS mix {v02['nts_level_mix']}; urgent {v02['urgent_nts']} ({100*v02['urgent_nts']/v02['n']:.1f}%) "
          f"vs severity<=2 {v02['urgent_sev']}; modifiers fire on {v02['modifier_fires']} {v02['modifier_fires_by_condition']}")
    v02["alt_key_levels_changed"] = {k: int((v != Ls["nts_level"]).sum()) for k, v in aks.items()}
    v02["alt_key_qwk_vs_primary"] = {k: qwk(v, Ls["nts_level"]) for k, v in aks.items()}
    v02["qwk_no_modifiers_vs_primary"] = qwk(Ls["nts_level_nomod"], Ls["nts_level"])
    print(f"    alt keys: levels changed {v02['alt_key_levels_changed']}, QWK vs primary { {k: round(v, 3) for k, v in v02['alt_key_qwk_vs_primary'].items()} }, no-modifier QWK {v02['qwk_no_modifiers_vs_primary']:.3f}")
    for k in ("severity_key_pathology<=2", "reference_key_pathology<=2", "differential_severity_mass>=0.15", "nts_no_modifiers",
              "nts_mapping_up", "nts_mapping_down", "nts_mapping_both", "always_escalate"):
        c = ceil_s[k]
        print(f"    {k:36s} vs NTS key: A {100*c['under']:4.1f}% [{100*c['ci_condition']['under'][0]:.1f}-{100*c['ci_condition']['under'][1]:.1f}] "
              f"B {100*c['over']:4.1f}% [{100*c['ci_condition']['over'][0]:.1f}-{100*c['ci_condition']['over'][1]:.1f}] score {c['score']:5.1f} "
              f"[cond {c['ci_condition']['score'][0]:.0f}-{c['ci_condition']['score'][1]:.0f}; case {c['ci_case']['score'][0]:.0f}-{c['ci_case']['score'][1]:.0f}] "
              f"kappa {c['kappa']:.2f} [{c['ci_condition']['kappa'][0]:.2f}-{c['ci_condition']['kappa'][1]:.2f}]")
    r2 = v02["red_flags"]
    print(f"    red flags: {r2['tier1_cases']} {r2['tier1_by_flag']}; NTS-urgent {r2['tier1_urgent_nts']}, routine {r2['tier1_nonurgent_nts']} {r2['tier1_nonurgent_by_condition']}; headline B denominator {r2['B_denominator_headline']}")
    print(f"    natural adult mix: NTS-urgent {100*v02['natural_adult_mix']['urgent_nts_share']:.1f}%, severity<=2 {100*v02['natural_adult_mix']['severe_sev_share']:.1f}%; "
          f"A at 5% has +/- {100*v02['precision_A_at_5pct_binomial_halfwidth']:.1f} points (binomial)")

    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    with open(OUT / "eval250_per_case.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "pathology", "age", "sev", "ref", "nts", "nts_no_mod", "p_severe", "p_nts_urgent", "v0", "L_T", "L_nts", "tier1"])
        for r, s_, rf_, n_, nm_, ps, pn, a, b, c, t1 in zip(rows, L["sev_level"], L["ref_level"], L["nts_level"], L["nts_level_nomod"], L["p_severe"], L["p_nts_urgent"], v0, L["L_T"], key, tier1):
            w.writerow([r["case_id"], r["path"], r["age"], s_, rf_, n_, nm_, f"{ps:.3f}", f"{pn:.3f}", int(a), int(b), int(c), int(t1)])
    with open(OUT / "models_nts_rescore.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["rank_current", "current_score", "current_under", "current_over", "rank_nts", "nts_score", "nts_under", "nts_over", "nts_kappa",
                "rank_nts_redflags_removed", "nts_score_redflags_removed", "nts_over_redflags_removed", "escalate_rate"]
        w.writerow(["model"] + cols)
        for n in sorted(names, key=lambda n: models[n]["rank_nts"]):
            w.writerow([n] + [models[n].get(c) for c in cols])
    print(f"wrote {OUT.relative_to(ROOT)}/summary.json, eval250_per_case.csv, models_nts_rescore.csv")


if __name__ == "__main__":
    main()
