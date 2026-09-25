"""Pilot of a "dangerous if missed" (DIM) scoring on the v0.1 model outputs.

The proposed design: a clinician sees the patient; the model lists the
dangerous conditions the patient may have, so the clinician does not miss
them. We score the list against a per-case "dangerous plausible set" (the
key). This script builds candidate keys, scores the 19 v0.1 rows with per-case
outputs on the 250-case set (their top-5 differential is the flag list),
simulates gaming policies and honest readers, and compares with the v0.2
review decision. No inference spend.

Danger scale: spec/dangerous_if_missed_tiers.csv when it exists (columns
`condition` and a tier column), else DDXPlus severity. 1 is the most dangerous.

Keys. Each key is a pair (target set, excuse set). A miss is a target element
the model did not flag; an over-flag is a severe flag that is neither a target
nor excused.
  K1    target = the true condition; nothing excused.
  K2_t  target = DXA conditions with p >= t plus the truth; the target excuses.
  K3_t  the same from a naive-Bayes posterior trained on DDXPlus (dataset ceiling).
  K4_t  K2_t restricted to conditions with a specific DDXPlus symptom present.
  K5_t  target = the true condition with its danger tier; DXA p >= t only excuses
        over-flags (the v0.2 stance: a DXA quirk can excuse but never create a miss).
DDXPlus is a closed microcosm (clamped priors, off-list mass deleted), so K5 is
the primary key and the plausible-set keys are descriptive.

Run:
    .venv/bin/python scripts/analysis/dim_pilot.py

Outputs go to results/analysis/dim_pilot/.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import answer_key_v02 as ak  # noqa: E402
from evaluator.icd10 import normalize_icd10  # noqa: E402
from evaluator.v02_references import NaiveBayes, train_counts  # noqa: E402
from evaluator.v02_score import ConditionMap  # noqa: E402
from scripts.analysis import failure_shape as fs  # noqa: E402

CASES_PATH = ROOT / "data/test_sets/eval-250-v0.json"
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
TIERS_CSV = ROOT / "spec/dangerous_if_missed_tiers.csv"
OFFLIST_CSV = ROOT / "spec/ddxplus_offlist_categories.csv"
V01_RANKS = ROOT / "results/analysis/failure_shape/scoring_ranks_published23.csv"
OUT = ROOT / "results/analysis/dim_pilot"
NB_CACHE = OUT / "nb_counts_holdout250.json"  # own cache: the v0.2 cache holds the 470 holdout

THRESHOLDS = (5, 10, 15, 25)  # percent
SERIOUS_TIER = 2  # tier <= 2 is "severe" for the over-flag count
GAP = 2  # a miss this many tiers more dangerous than the best flag is a severe gap
DX_WEIGHT = 0.25  # true-diagnosis credit, provisional; the sibling agent sets the final weight
NO_FLAG_TIER = 6  # the tier of "flagged nothing on the list", one below the least dangerous tier
N_BOOT = 1000
SEED = 20260924
SPECIFIC_SYMPTOM_MAX_CONDITIONS = 8  # K4: a symptom listed for at most this many conditions is "specific"
PRIMARY_KEY = "K5_10"
PLAUSIBLE_KEY = "K2_10"  # the plausible-set comparison key
DEFAULT_CAP = 5


# ---------------------------------------------------------------- data


def load_danger_scale(conditions: dict) -> tuple[dict[str, int], str]:
    """condition -> danger tier (1 = most dangerous). The tier table wins when it exists."""
    if TIERS_CSV.exists():
        with open(TIERS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        tier_col = next(c for c in rows[0] if "tier" in c.lower())
        tiers = {r["condition"]: int(r[tier_col]) for r in rows}
        missing = set(conditions) - set(tiers)
        if missing:
            raise SystemExit(f"tier table lacks {len(missing)} conditions: {sorted(missing)[:5]}")
        return tiers, f"{TIERS_CSV.relative_to(ROOT)} ({tier_col})"
    return {c: int(v["severity"]) for c, v in conditions.items()}, "DDXPlus severity"


def load_cases() -> list[dict]:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    want = {int(c["case_id"].split("_")[1]): c for c in cases}
    with open(DDX_CSV) as f:
        for i, row in enumerate(csv.DictReader(f)):
            c = want.get(i)
            if c is None:
                continue
            c["dxa"] = {n: float(p) for n, p in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])}
            c["truth"] = row["PATHOLOGY"]
            c["codes"] = {str(e).split("_@_")[0] for e in c["presenting_symptoms"]}
            c["red_flags"] = ak.red_flags(c["presenting_symptoms"])
    missing = [c["case_id"] for c in cases if "dxa" not in c]
    if missing:
        raise SystemExit(f"{len(missing)} cases not in DDXPlus CSV")
    return cases


def load_offlist() -> list[tuple[str, str]]:
    with open(OFFLIST_CSV, newline="", encoding="utf-8") as f:
        rows = [(normalize_icd10(r["code_prefix"]).upper(), r["category"]) for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: -len(r[0]))  # longest prefix first


def offlist_category(code: str, table) -> str:
    c = normalize_icd10(code).upper()
    for prefix, cat in table:
        if c.startswith(prefix):
            return cat
    return "unmapped"


def load_models(cases, cmap: ConditionMap, offlist) -> dict[str, dict]:
    """name -> per-case flag lists from the v0.1 top-5 differential, plus the v0.1 review decision."""
    ids = {c["case_id"]: i for i, c in enumerate(cases)}
    models = {}
    for r in fs.load_rows():
        path, status = fs.resolve_predictions(r)
        if path is None:
            continue
        raw = json.loads(path.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        flags = [[] for _ in cases]
        off = [[] for _ in cases]
        esc = np.zeros(len(cases), bool)
        readable = np.zeros(len(cases), bool)
        seen = set()
        for p in preds:
            cid = p.get("case_id") if isinstance(p, dict) else None
            if cid not in ids or cid in seen:
                continue
            seen.add(cid)
            i = ids[cid]
            codes = [d.get("code") for d in (p.get("differential_diagnoses") or []) if isinstance(d, dict) and d.get("code")]
            if not codes:
                continue
            readable[i] = True
            esc[i] = p.get("escalation_decision") == "ESCALATE_NOW"
            for code in codes:
                own = cmap.owner(code)
                if own is None:
                    off[i].append((code, offlist_category(code, offlist)))
                    flags[i].append(None)  # keeps list position for the cap
                else:
                    flags[i].append(own)
        models[r["name"]] = {"flags": flags, "offlist": off, "escalated": esc, "readable": readable,
                             "status": status, "kind": "model"}
    return models


def v01_board() -> dict[str, dict]:
    with open(V01_RANKS, newline="") as f:
        return {r["model"]: {"tsr": float(r["TSR value"]), "harm": float(r["harm.py expected_harm (100:2) value"])}
                for r in csv.DictReader(f)}


# ---------------------------------------------------------------- keys


def specific_symptoms(raw_conditions: dict) -> dict[str, set[str]]:
    """Per condition, its DDXPlus symptom codes listed for at most N conditions in total."""
    listed = Counter(s for v in raw_conditions.values() for s in v["symptoms"])
    return {c: {s for s in v["symptoms"] if listed[s] <= SPECIFIC_SYMPTOM_MAX_CONDITIONS} for c, v in raw_conditions.items()}


Key = tuple[set[str], set[str]]  # (target set, excuse set)


def build_keys(cases, nb_post: dict[str, dict[str, float]], specific: dict[str, set[str]]) -> dict[str, list[Key]]:
    keys: dict[str, list[Key]] = {"K1": [({c["truth"]}, set()) for c in cases]}
    for t in THRESHOLDS:
        dxa = [{n for n, p in c["dxa"].items() if 100 * p >= t} for c in cases]
        k2 = [d | {c["truth"]} for d, c in zip(dxa, cases)]
        keys[f"K2_{t}"] = [(k, k) for k in k2]
        k3 = [{n for n, p in nb_post[c["case_id"]].items() if 100 * p >= t} | {c["truth"]} for c in cases]
        keys[f"K3_{t}"] = [(k, k) for k in k3]
        k4 = [{n for n in k if n == c["truth"] or (specific[n] & c["codes"])} for c, k in zip(cases, k2)]
        keys[f"K4_{t}"] = [(k, k) for k in k4]
        keys[f"K5_{t}"] = [({c["truth"]}, d) for d, c in zip(dxa, cases)]
    return keys


def nb_posteriors(cases) -> dict[str, dict[str, float]]:
    counts = train_counts({c["case_id"] for c in cases}, cache=NB_CACHE)
    nb = NaiveBayes(counts)
    return {c["case_id"]: dict(zip(nb.conditions, map(float, nb.posterior(c["presenting_symptoms"])))) for c in cases}


def key_summary(cases, keys, tier, out_rows: list[dict], examples: dict) -> list[dict]:
    rows = []
    for name, pairs in keys.items():
        sets = [k for k, _ in pairs]
        excuses = [e for _, e in pairs]
        sizes = np.array([len(s) for s in sets])
        top_tier = np.array([min(tier[n] for n in s) for s in sets])
        truth_top = np.array([tier[c["truth"]] == tt for c, tt in zip(cases, top_tier)])
        serious_key = top_tier <= SERIOUS_TIER
        truth_serious = np.array([tier[c["truth"]] <= SERIOUS_TIER for c in cases])
        quirk = serious_key & ~truth_serious  # the key's danger comes only from DXA / NB mass
        pairs = Counter()
        for c, s, tt, q in zip(cases, sets, top_tier, quirk):
            if q:
                tops = sorted(n for n in s if tier[n] == tt)
                pairs[(c["truth"], "; ".join(tops))] += 1
        examples[name] = [{"truth": a, "key_top": b, "cases": n} for (a, b), n in pairs.most_common(8)]
        rows.append({
            "key": name, "mean_size": round(float(sizes.mean()), 2), "median_size": float(np.median(sizes)),
            "max_size": int(sizes.max()),
            "truth_is_most_dangerous_pct": round(100 * truth_top.mean(), 1),
            "serious_key_cases": int(serious_key.sum()), "truth_serious_cases": int(truth_serious.sum()),
            "serious_key_but_truth_not_pct": round(100 * quirk.mean(), 1),
            "top_tier_1_cases": int((top_tier == 1).sum()),
            "mean_excuse_size": round(float(np.mean([len(e - s) for e, s in zip(excuses, sets)])), 2),
        })
        for c, s, tt in zip(cases, sets, top_tier):
            out_rows.append({"case_id": c["case_id"], "key": name, "truth": c["truth"], "truth_tier": tier[c["truth"]],
                             "size": len(s), "top_tier": int(tt),
                             "top_elements": "; ".join(sorted(n for n in s if tier[n] == tt)),
                             "members": "; ".join(sorted(s))})
    return rows


# ---------------------------------------------------------------- per-case scoring


def case_score(flags: list, key: set[str], excuse: set[str], truth: str, tier: dict[str, int], cap: int | None) -> dict:
    """Per-case measures for one flag list (None entries are off-list codes; they take a slot).
    Off-list codes are never a miss and never an over-flag."""
    seq = flags[:cap] if cap else flags
    on_list = [f for f in seq if f]
    fset = set(on_list)
    top = min(tier[n] for n in key)
    tops = {n for n in key if tier[n] == top}
    missed = key - fset
    missed_tier = min((tier[n] for n in missed), default=NO_FLAG_TIER)
    flagged_tier_any = min((tier[n] for n in fset), default=NO_FLAG_TIER)
    flagged_tier_key = min((tier[n] for n in fset & key), default=NO_FLAG_TIER)
    false_severe = sum(1 for n in fset - key - excuse if tier[n] <= SERIOUS_TIER)
    return {
        "hit": float(bool(tops & fset)),
        "gap_any": float(missed_tier + GAP <= flagged_tier_any),
        "gap_key": float(missed_tier + GAP <= flagged_tier_key),
        "top1": float(bool(on_list) and on_list[0] == truth and seq[0] == truth),
        "top5": float(truth in fset),
        "false_severe": float(false_severe),
        "false_any": float(len(fset - key - excuse)),
        "n_flags": float(len(fset)),
        "n_offlist": float(sum(1 for f in seq if f is None)),
        "precision": float(len(fset & (key | excuse)) / len(fset)) if fset else 0.0,
        "serious_key": float(top <= SERIOUS_TIER),
        "empty": float(not seq),
    }


DESIGNS = {
    # name: (formula label, function of the per-case measure dict)
    "proposal": ("hit - gap_any + w*top1", lambda m: m["hit"] - m["gap_any"] + DX_WEIGHT * m["top1"]),
    "proposal_gapkey": ("hit - gap_key + w*top1", lambda m: m["hit"] - m["gap_key"] + DX_WEIGHT * m["top1"]),
    "prec_0.1": ("hit - 0.1*false_severe + w*top1", lambda m: m["hit"] - 0.1 * m["false_severe"] + DX_WEIGHT * m["top1"]),
    "prec_0.25": ("hit - 0.25*false_severe + w*top1", lambda m: m["hit"] - 0.25 * m["false_severe"] + DX_WEIGHT * m["top1"]),
    "prec_0.5": ("hit - 0.5*false_severe + w*top1", lambda m: m["hit"] - 0.5 * m["false_severe"] + DX_WEIGHT * m["top1"]),
    "hit_x_precision": ("hit*(0.5+0.5*precision) + w*top1", lambda m: m["hit"] * (0.5 + 0.5 * m["precision"]) + DX_WEIGHT * m["top1"]),
    "false_any_0.1": ("hit - 0.1*false_any + w*top1", lambda m: m["hit"] - 0.1 * m["false_any"] + DX_WEIGHT * m["top1"]),
}


def score_flags(flag_lists, keys_for_key, cases, tier, cap) -> dict[str, np.ndarray]:
    ms = [case_score(f, k, e, c["truth"], tier, cap) for f, (k, e), c in zip(flag_lists, keys_for_key, cases)]
    out = {k: np.array([m[k] for m in ms]) for k in ms[0]}
    for d, (_, fn) in DESIGNS.items():
        out[d] = np.array([fn(m) for m in ms])
    return out


def summarize(arrs: dict[str, np.ndarray]) -> dict[str, float]:
    sk = arrs["serious_key"] > 0
    s = {k: float(v.mean()) for k, v in arrs.items()}
    s["hit_serious_key"] = float(arrs["hit"][sk].mean()) if sk.any() else float("nan")
    s["gap_any_serious_key"] = float(arrs["gap_any"][sk].mean()) if sk.any() else float("nan")
    s["n_serious_key"] = int(sk.sum())
    return s


# ---------------------------------------------------------------- simulated policies


def policies(cases, nb_post, tier, keys) -> dict[str, list[list[str]]]:
    """Flag lists for gaming policies and honest readers. Order matters under a cap."""
    conds = sorted(tier)
    severe = sorted((n for n in conds if tier[n] <= SERIOUS_TIER), key=lambda n: (tier[n], n))
    # Oracle-tuned fixed order: severe conditions by how often they top the primary key in this sample.
    top_freq = Counter()
    for s, _ in keys[PLAUSIBLE_KEY]:
        tt = min(tier[n] for n in s)
        for n in s:
            if tier[n] == tt:
                top_freq[n] += 1
    severe_tuned = sorted(severe, key=lambda n: (-top_freq[n], tier[n], n))
    tier1 = [n for n in severe if tier[n] == 1]
    fixed_core = [n for n in ("Possible NSTEMI / STEMI", "Pulmonary embolism", "Anaphylaxis", "Acute pulmonary edema",
                              "Spontaneous pneumothorax", "Unstable angina", "Myocarditis", "Boerhaave") if n in tier]
    n = len(cases)

    def reader(post, t, danger_first):
        out = []
        for c in cases:
            items = [(k, p) for k, p in post[c["case_id"]].items() if 100 * p >= t]
            items.sort(key=(lambda kp: (tier[kp[0]], -kp[1])) if danger_first else (lambda kp: -kp[1]))
            out.append([k for k, _ in items])
        return out

    dxa = {c["case_id"]: c["dxa"] for c in cases}
    pol = {
        "flag_all_severe": [list(severe)] * n,
        "flag_all_severe_tuned": [list(severe_tuned)] * n,
        "fixed_tier1": [list(tier1)] * n,
        "fixed_core8": [list(fixed_core)] * n,
        "flag_everything": [list(conds)] * n,
        "empty": [[] for _ in range(n)],
    }
    for t in THRESHOLDS:
        pol[f"dxa_p>={t}"] = reader(dxa, t, False)
        pol[f"dxa_p>={t}_danger_first"] = reader(dxa, t, True)
        pol[f"nb_p>={t}"] = reader(nb_post, t, False)
    pol["dxa_top5"] = reader(dxa, 0, False)
    pol["nb_top5"] = reader(nb_post, 0, False)
    pol["dxa_top5_plus_severe"] = [(r[:5] + [s for s in severe_tuned if s not in r[:5]])[:8] for r in pol["dxa_top5"]]
    return pol


def brier_rows(cases, nb_post, tier, keys_primary) -> list[dict]:
    """Danger-weighted Brier on flag probabilities: only policies with probabilities can be scored.
    Target: membership of the primary key. Weight: 1 for tier 1-2, 0.25 otherwise."""
    conds = sorted(tier)
    w = np.array([1.0 if tier[n] <= SERIOUS_TIER else 0.25 for n in conds])
    w /= w.sum()
    dxa = {c["case_id"]: c["dxa"] for c in cases}

    def brier(prob_fn):
        tot = 0.0
        for c, (k, _) in zip(cases, keys_primary):
            y = np.array([1.0 if n in k else 0.0 for n in conds])
            q = np.array([prob_fn(c, n) for n in conds])
            tot += float((w * (q - y) ** 2).sum())
        return tot / len(cases)

    severe = {n for n in conds if tier[n] <= SERIOUS_TIER}
    rows = [
        {"policy": "dxa probabilities", "brier": brier(lambda c, n: dxa[c["case_id"]].get(n, 0.0))},
        {"policy": "dxa membership p>=10 as 1/0", "brier": brier(lambda c, n: float(100 * dxa[c["case_id"]].get(n, 0.0) >= 10))},
        {"policy": "nb probabilities", "brier": brier(lambda c, n: nb_post[c["case_id"]].get(n, 0.0))},
        {"policy": "flag_all_severe (p=1)", "brier": brier(lambda c, n: float(n in severe))},
        {"policy": "flag_all_severe (p=0.5)", "brier": brier(lambda c, n: 0.5 if n in severe else 0.0)},
        {"policy": "flag nothing", "brier": brier(lambda c, n: 0.0)},
        {"policy": "truth only (p=1)", "brier": brier(lambda c, n: float(n == c["truth"]))},
    ]
    for r in rows:
        r["brier"] = round(r["brier"], 5)
    return rows


# ---------------------------------------------------------------- bootstrap


def cluster_bootstrap(values: np.ndarray, cond_idx: np.ndarray, n_cond: int, rng) -> tuple[float, float]:
    sums = np.bincount(cond_idx, weights=values, minlength=n_cond)
    cnt = np.bincount(cond_idx, minlength=n_cond).astype(float)
    draws = rng.integers(0, n_cond, size=(N_BOOT, n_cond))
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(draws[b], minlength=n_cond)
        stats[b] = (w * sums).sum() / (w * cnt).sum()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


# ---------------------------------------------------------------- v0.2 comparison


def v02_compare(cases, models, keys, tier, cap, key_name) -> tuple[list[dict], list[dict]]:
    truth_serious = np.array([tier[c["truth"]] <= SERIOUS_TIER for c in cases])
    rows, examples = [], []
    for name, m in models.items():
        arrs = score_flags(m["flags"], keys[key_name], cases, tier, cap)
        sk = arrs["serious_key"] > 0
        a_event = truth_serious & ~m["escalated"]  # v0.2 A: serious patient not escalated (unreadable counts)
        e_event = truth_serious & (arrs["top5"] == 0)  # v0.2 E: serious diagnosis missing from the top 5
        dim_miss = sk & (arrs["hit"] == 0)
        rows.append({
            "model": name,
            "v02_A_events": int(a_event.sum()), "v02_E_events": int(e_event.sum()), "dim_misses": int(dim_miss.sum()),
            "dim_miss_and_A": int((dim_miss & a_event).sum()),
            "dim_miss_escalated": int((dim_miss & m["escalated"] & truth_serious).sum()),
            "dim_miss_truth_not_serious": int((dim_miss & ~truth_serious).sum()),
            "A_but_dim_hit": int((a_event & ~dim_miss).sum()),
            "E_but_dim_hit": int((e_event & ~dim_miss).sum()),
            "dim_miss_but_top5_truth": int((dim_miss & (arrs["top5"] == 1)).sum()),
        })
        for i, c in enumerate(cases):
            kind = None
            if dim_miss[i] and m["escalated"][i] and truth_serious[i]:
                kind = "escalated but did not name the most dangerous plausible condition (DIM catches, v0.2 A does not)"
            elif a_event[i] and not dim_miss[i] and m["readable"][i]:
                kind = "named the most dangerous condition but sent the patient to routine care (v0.2 A catches, DIM does not)"
            elif dim_miss[i] and not truth_serious[i]:
                kind = "truth not serious; DIM miss comes from DXA mass on a severe condition (DIM penalises, v0.2 excuses)"
            if kind:
                kset = keys[key_name][i][0]
                tops = sorted(n for n in kset if tier[n] == min(tier[x] for x in kset))
                examples.append({"model": name, "case_id": c["case_id"], "truth": c["truth"], "truth_tier": tier[c["truth"]],
                                 "key_top": "; ".join(tops), "flags": "; ".join(f or "(off-list)" for f in m["flags"][i][:cap]),
                                 "escalated": bool(m["escalated"][i]), "kind": kind})
    return rows, examples


# ---------------------------------------------------------------- off-list


def offlist_rows(cases, models) -> tuple[list[dict], list[dict]]:
    watch = ("GI haemorrhage", "Intracranial haemorrhage", "Meningitis")
    flagged_any = {c["case_id"]: bool(c["red_flags"]) for c in cases}
    bleeding = {c["case_id"]: "bleeding" in c["red_flags"] for c in cases}
    thunder = {c["case_id"]: "thunderclap_headache" in c["red_flags"] for c in cases}
    rows, by_case = [], []
    for name, m in models.items():
        cats = Counter()
        cat_on_flag = defaultdict(lambda: Counter())
        for c, off in zip(cases, m["offlist"]):
            seen = {cat for _, cat in off}
            for cat in seen:
                cats[cat] += 1
                cat_on_flag[cat]["red_flag" if flagged_any[c["case_id"]] else "no_red_flag"] += 1
                if cat in ("GI haemorrhage",):
                    cat_on_flag[cat]["bleeding_flag" if bleeding[c["case_id"]] else "no_bleeding_flag"] += 1
                if cat in ("Intracranial haemorrhage",):
                    cat_on_flag[cat]["thunderclap" if thunder[c["case_id"]] else "no_thunderclap"] += 1
                if cat in watch or cat.lower().startswith("mening"):
                    by_case.append({"model": name, "case_id": c["case_id"], "truth": c["truth"], "category": cat,
                                    "red_flags": ";".join(c["red_flags"]) or "-"})
        total_off = sum(len(o) for o in m["offlist"])
        rows.append({"model": name, "offlist_codes": total_off,
                     "cases_with_offlist": sum(1 for o in m["offlist"] if o),
                     **{f"{cat}": cats.get(cat, 0) for cat in watch},
                     "GI_on_bleeding_flag": cat_on_flag["GI haemorrhage"]["bleeding_flag"],
                     "GI_off_bleeding_flag": cat_on_flag["GI haemorrhage"]["no_bleeding_flag"],
                     "ICH_on_thunderclap": cat_on_flag["Intracranial haemorrhage"]["thunderclap"],
                     "ICH_off_thunderclap": cat_on_flag["Intracranial haemorrhage"]["no_thunderclap"],
                     "top_categories": "; ".join(f"{k}={v}" for k, v in cats.most_common(5))})
    return rows, by_case


# ---------------------------------------------------------------- main


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_conditions = json.loads(COND_PATH.read_text())
    conditions = ak.load_conditions()
    tier, scale_source = load_danger_scale(raw_conditions)
    cmap = ConditionMap()
    offlist = load_offlist()
    cases = load_cases()
    nb_post = nb_posteriors(cases)
    keys = build_keys(cases, nb_post, specific_symptoms(raw_conditions))
    models = load_models(cases, cmap, offlist)
    board = v01_board()
    cond_names = sorted({c["truth"] for c in cases})
    cond_idx = np.array([cond_names.index(c["truth"]) for c in cases])
    rng = np.random.default_rng(SEED)

    # 1. keys
    key_rows: list[dict] = []
    quirk_examples: dict = {}
    ksum = key_summary(cases, keys, tier, key_rows, quirk_examples)
    write_csv(OUT / "keys_per_case.csv", key_rows)
    write_csv(OUT / "key_summary.csv", ksum)
    write_csv(OUT / "key_quirk_examples.csv", [{"key": k, **e} for k, es in quirk_examples.items() for e in es])

    # 2. model scores under each key (flag list = the top-5 differential, no cap beyond the list itself)
    score_rows, boot_rows = [], []
    per_key_scores: dict[str, dict[str, dict]] = defaultdict(dict)
    for kname, ksets in keys.items():
        for name, m in models.items():
            arrs = score_flags(m["flags"], ksets, cases, tier, None)
            s = summarize(arrs)
            per_key_scores[kname][name] = s
            score_rows.append({"key": kname, "model": name, **{k: round(v, 4) for k, v in s.items()},
                               "v01_tsr": board.get(name, {}).get("tsr"), "v01_harm": board.get(name, {}).get("harm")})
            if kname in (PRIMARY_KEY, PLAUSIBLE_KEY, "K1", "K2_15", "K3_10", "K5_25"):
                for meas in ("hit", "prec_0.25", "proposal"):
                    lo, hi = cluster_bootstrap(arrs[meas], cond_idx, len(cond_names), rng)
                    boot_rows.append({"key": kname, "model": name, "measure": meas, "value": round(float(arrs[meas].mean()), 4),
                                      "lo": round(lo, 4), "hi": round(hi, 4)})
                sk = arrs["serious_key"] > 0  # the serious subset, clustered by its own conditions
                for meas in ("hit", "gap_any"):
                    lo, hi = cluster_bootstrap(arrs[meas][sk], cond_idx[sk], len(cond_names), rng)
                    boot_rows.append({"key": kname, "model": name, "measure": f"{meas}_serious_key",
                                      "value": round(float(arrs[meas][sk].mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    write_csv(OUT / "model_scores.csv", score_rows)
    write_csv(OUT / "model_bootstrap.csv", boot_rows)

    # correlations with the v0.1 board and across keys
    names = [n for n in models if n in board]
    corr_rows = []
    for kname in keys:
        for meas in ("hit", "hit_serious_key", "proposal", "prec_0.25", "top5"):
            x = [per_key_scores[kname][n][meas] for n in names]
            corr_rows.append({"key": kname, "measure": meas,
                              "spearman_vs_v01_tsr": round(spearman(x, [board[n]["tsr"] for n in names]), 3),
                              "spearman_vs_v01_harm": round(spearman(x, [-board[n]["harm"] for n in names]), 3),
                              "spearman_vs_K1_top5": round(spearman(x, [per_key_scores["K1"][n]["top5"] for n in names]), 3),
                              "spread": round(max(x) - min(x), 4)})
    write_csv(OUT / "correlations.csv", corr_rows)

    # 3. gaming: policies and models, every design, caps 3/5/8/none, primary key and K2_15
    pol = policies(cases, nb_post, tier, keys)
    gaming_rows = []
    for kname in (PRIMARY_KEY, PLAUSIBLE_KEY, "K2_15", "K2_5", "K1", "K5_25"):
        for cap in (3, 5, 8, None):
            for name, flags in list(pol.items()) + [(n, m["flags"]) for n, m in models.items()]:
                arrs = score_flags(flags, keys[kname], cases, tier, cap)
                s = summarize(arrs)
                gaming_rows.append({"key": kname, "cap": cap or "none", "row": name,
                                    "kind": "policy" if name in pol else "model",
                                    **{k: round(v, 4) for k, v in s.items()}})
    write_csv(OUT / "gaming.csv", gaming_rows)
    write_csv(OUT / "brier.csv", brier_rows(cases, nb_post, tier, keys[PLAUSIBLE_KEY]))  # descriptive: agreement with the microcosm

    # Which designs resist gaming: the best policy vs the best honest reader vs the best model, per design.
    verdict_rows = []
    gamers = {"flag_all_severe", "flag_all_severe_tuned", "fixed_tier1", "fixed_core8", "flag_everything", "dxa_top5_plus_severe"}
    honest = {f"dxa_p>={t}" for t in THRESHOLDS} | {f"nb_p>={t}" for t in THRESHOLDS} | {"dxa_top5", "nb_top5"} \
        | {f"dxa_p>={t}_danger_first" for t in THRESHOLDS}
    for kname in (PRIMARY_KEY, PLAUSIBLE_KEY, "K2_15", "K1"):
        for cap in (3, 5, 8, None):
            sub = [r for r in gaming_rows if r["key"] == kname and r["cap"] == (cap or "none")]
            for d in DESIGNS:
                g = max((r for r in sub if r["row"] in gamers), key=lambda r: r[d])
                h = max((r for r in sub if r["row"] in honest), key=lambda r: r[d])
                mrows = [r for r in sub if r["kind"] == "model"]
                mbest = max(mrows, key=lambda r: r[d])
                mworst = min(mrows, key=lambda r: r[d])
                verdict_rows.append({"key": kname, "cap": cap or "none", "design": d, "formula": DESIGNS[d][0],
                                     "best_gamer": g["row"], "best_gamer_score": g[d],
                                     "best_honest": h["row"], "best_honest_score": h[d],
                                     "best_model": mbest["row"], "best_model_score": mbest[d],
                                     "worst_model": mworst["row"], "worst_model_score": mworst[d],
                                     "gamer_beats_honest": g[d] > h[d], "gamer_beats_best_model": g[d] > mbest[d]})
    write_csv(OUT / "gaming_verdict.csv", verdict_rows)

    # 4. off-list flags
    off_rows, off_cases = offlist_rows(cases, models)
    write_csv(OUT / "offlist.csv", off_rows)
    write_csv(OUT / "offlist_cases.csv", off_cases)

    # 5. v0.2 comparison
    for kn, tag in ((PRIMARY_KEY, "truth"), (PLAUSIBLE_KEY, "plausible")):
        cmp_rows, cmp_examples = v02_compare(cases, models, keys, tier, DEFAULT_CAP, kn)
        write_csv(OUT / f"v02_compare_{tag}.csv", cmp_rows)
        write_csv(OUT / f"v02_compare_{tag}_examples.csv", cmp_examples)

    red = Counter(f for c in cases for f in c["red_flags"])
    summary = {
        "danger_scale": scale_source, "cases": len(cases), "conditions_in_sample": len(cond_names),
        "models": {n: m["status"] for n, m in models.items()},
        "primary_key": PRIMARY_KEY, "plausible_key": PLAUSIBLE_KEY, "dx_weight": DX_WEIGHT, "cap": DEFAULT_CAP,
        "red_flag_cases": dict(red),
        "key_summary": ksum,
        "tier_counts": dict(Counter(tier.values())),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(models)} models, {len(keys)} keys, {len(pol)} policies; scale = {scale_source}")


if __name__ == "__main__":
    main()
