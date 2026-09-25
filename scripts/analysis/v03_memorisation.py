"""Memorisation checks for MedSafe-Dx v0.3 (docs/v0.3-memorisation-checks.md; Astra review finding 7).

A model can score well on DDXPlus without reading the patient in three ways, and each
part here bounds one of them, with no inference spend:

1. Duplicate audit. Recalling public DDXPlus rows helps only where a sample case has a
   twin in the public data. For the 470 main sample and both v0.3 pools we find exact
   twins (same age band, sex and evidence tokens) and near twins (Jaccard >= 0.9 on the
   tokens, same band and sex) in the validate and test splits in this checkout, and
   whether the twins share the true condition. The train split (1,025,602 rows) is not
   in this checkout; the doc extrapolates from the validate split's rate.
2. Shortcut analysis. The generator's signature may sit in the history line: DDXPlus
   samples risk-factor antecedents only for the true condition. We retrain the
   naive-Bayes reference (evaluator/v02_references.py) on symptoms only and on
   antecedents only, and report top-1 accuracy and the serious/benign AUC of each
   beside the full reader and the DXA reader.
3. Rendering variants. The paired rendering test (inference/rendering.py) needs the
   most frequent decoder strings for its phrase table; we write them, the table's
   coverage, and three example cases rendered each way.
4. Flag-rule inputs. Cluster-bootstrap intervals of the references on the main sample,
   so the pre-registered memorisation flag can set its margins from measured noise.

Inputs:
  data/ddxplus_v0/release_test_patients, release_validate_patients, release_evidences.json
  data/test_sets/eval-v02-adult.json, eval-v03-pool-*.json, their keys and refs
  spec/dangerous_if_missed_tiers_v03.csv

Outputs, under results/analysis/v03_memorisation/:
  duplicates.csv        per sample case: twins per split, nearest neighbour, condition agreement
  duplicates_summary.csv  per set and split: the counts the doc cites
  shortcuts.csv         per set and reader: top-1, top-5, YES rate, AUCs, with intervals on the main sample
  frequent_strings.csv  the v0.2-decoder strings by frequency over the 720 cases, and table coverage
  rendering_examples.md three cases rendered standard, shuffled and paraphrased
  summary.json          every number the doc cites
  cache/                parsed splits, safe to delete

Run: python3 scripts/analysis/v03_memorisation.py   (numpy only; about two minutes cold)
"""
from __future__ import annotations

import ast
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator.v02_references import NaiveBayes, train_counts  # noqa: E402
from evaluator.v03_score import BOOTSTRAP_SEED, N_BOOTSTRAP, cluster_draws, interval  # noqa: E402
from inference import rendering as R  # noqa: E402
from inference.run_inference import format_case_for_prompt_v7  # noqa: E402
from inference.symptom_decoder import EVIDENCE_DATA, decode_symptoms_versioned  # noqa: E402

DDX = ROOT / "data" / "ddxplus_v0"
SPLITS = {"test": DDX / "release_test_patients", "validate": DDX / "release_validate_patients"}
TRAIN_ROWS = 1_025_602  # DDXPlus train split (Fansi Tchango et al. 2022); not in this checkout
TS = ROOT / "data" / "test_sets"
OUT = ROOT / "results" / "analysis" / "v03_memorisation"
CACHE = OUT / "cache"
SETS = {
    "main": ("eval-v02-adult.json", "eval-v03-key.csv", "eval-v03-key.sha256", "eval-v03-adult.refs.json"),
    "pool-atypical": ("eval-v03-pool-atypical.json", "eval-v03-pool-atypical.key.csv",
                      "eval-v03-pool-atypical.key.sha256", "eval-v03-pool-atypical.refs.json"),
    "pool-high-risk": ("eval-v03-pool-high-risk.json", "eval-v03-pool-high-risk.key.csv",
                       "eval-v03-pool-high-risk.key.sha256", "eval-v03-pool-high-risk.refs.json"),
}
NEAR = 0.9  # Jaccard threshold for a near twin
READER_T = 10.0  # the DXA and naive-Bayes readers say YES at a tier-1 p >= 10% (scripts/build_v03_key.py)
EXAMPLE_CONDITIONS = ("Possible NSTEMI / STEMI", "Acute laryngitis", "Pulmonary embolism")
TOP_STRINGS = 40


# ---------------------------------------------------------------- data


def age_band(age: int) -> str:
    if age < 18:
        return "<18"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def load_split(name: str) -> dict:
    """Adults of one split: arrays of age, sex, pathology, and a tuple of evidence tokens per row.
    `row` is the 0-based row index, which is the case id's number for the test split."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rows, ages, sexes, paths, toks = [], [], [], [], []
    with open(SPLITS[name]) as f:
        for i, r in enumerate(csv.DictReader(f)):
            age = int(r["AGE"])
            if age < 18:
                continue
            rows.append(i)
            ages.append(age)
            sexes.append(r["SEX"])
            paths.append(r["PATHOLOGY"])
            toks.append(tuple(sorted(set(str(e) for e in ast.literal_eval(r["EVIDENCES"])))))
    d = {"row": np.array(rows), "age": np.array(ages), "sex": np.array(sexes), "pathology": np.array(paths),
         "tokens": toks}
    cache.write_bytes(pickle.dumps(d))
    return d


def load_sets() -> dict[str, dict]:
    """Per set: cases (file order), key, refs, and the per-case arrays the analyses use."""
    tiers = ak.load_tiers()
    out = {}
    for name, (cases_f, key_f, sha_f, refs_f) in SETS.items():
        cases = json.loads((TS / cases_f).read_text())["cases"]
        keys = ak.load_key(TS / key_f, TS / sha_f)
        refs = json.loads((TS / refs_f).read_text())["references"]
        kk = [keys[c["case_id"]] for c in cases]
        out[name] = {
            "cases": cases, "keys": kk, "refs": refs,
            "truth": np.array([k.truth for k in kk]),
            "truth_tier": np.array([k.truth_tier for k in kk]),
            "has_r10": np.array([bool(k.r10) for k in kk]),
            "clearly_low": np.array([k.clearly_low_risk for k in kk]),
            "tier1": np.array([tiers.get(k.truth, 3) == 1 for k in kk]),
        }
    return out


# ---------------------------------------------------------------- 1. duplicates


def token_matrix(token_lists, vocab: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(token_lists), len(vocab)), np.float32)
    for i, toks in enumerate(token_lists):
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                X[i, j] = 1.0
    return X


def duplicate_audit(sets: dict, splits: dict) -> tuple[list[dict], list[dict]]:
    """Per sample case and split: exact twins, near twins (Jaccard >= NEAR), the nearest neighbour's
    Jaccard and condition agreement. A twin must share the age band and sex; a case is never its own twin."""
    vocab = {t: j for j, t in enumerate(sorted({t for s in splits.values() for toks in s["tokens"] for t in toks}))}
    sample = []
    for sname, s in sets.items():
        for c, k in zip(s["cases"], s["keys"]):
            sample.append({"set": sname, "case_id": c["case_id"], "row": int(c["case_id"].split("_")[1]),
                           "truth": k.truth, "band": age_band(int(c["age"])),
                           "sex": "M" if c["sex"] == "male" else "F", "age": int(c["age"]),
                           "tokens": tuple(sorted(set(c["presenting_symptoms"])))})
    S = token_matrix([x["tokens"] for x in sample], vocab)
    ns = S.sum(1)
    per_case = {(x["set"], x["case_id"]): {"case_id": x["case_id"], "set": x["set"], "truth": x["truth"],
                                          "n_tokens": int(ns[i])} for i, x in enumerate(sample)}
    for split, d in splits.items():
        X = token_matrix(d["tokens"], vocab)
        nb = X.sum(1)
        band = np.array([age_band(a) for a in d["age"]])
        best = np.full(len(sample), -1.0)
        best_same = np.zeros(len(sample))  # share of max-J neighbours with the truth
        best_n = np.zeros(len(sample), int)
        best_age = np.zeros(len(sample), bool)  # a nearest neighbour with the exact age
        n_exact = np.zeros(len(sample), int)
        n_exact_same = np.zeros(len(sample), int)
        n_exact_age = np.zeros(len(sample), int)
        n_near = np.zeros(len(sample), int)
        n_near_same = np.zeros(len(sample), int)
        for i, x in enumerate(sample):
            ok = (band == x["band"]) & (d["sex"] == x["sex"])
            if split == "test":
                ok &= d["row"] != x["row"]
            idx = np.flatnonzero(ok)
            inter = X[idx] @ S[i]
            J = inter / (ns[i] + nb[idx] - inter)
            same = d["pathology"][idx] == x["truth"]
            exact = J >= 1.0 - 1e-6
            near = J >= NEAR
            n_exact[i], n_exact_same[i] = int(exact.sum()), int((exact & same).sum())
            n_exact_age[i] = int((exact & (d["age"][idx] == x["age"])).sum())
            n_near[i], n_near_same[i] = int(near.sum()), int((near & same).sum())
            if len(idx):
                m = J.max()
                at = J >= m - 1e-6
                best[i], best_n[i], best_same[i] = float(m), int(at.sum()), float(same[at].mean())
                best_age[i] = bool((d["age"][idx][at] == x["age"]).any())
        for i, x in enumerate(sample):
            pc = per_case[(x["set"], x["case_id"])]
            pc.update({f"{split}_exact": int(n_exact[i]), f"{split}_exact_same_truth": int(n_exact_same[i]),
                       f"{split}_exact_same_age": int(n_exact_age[i]),
                       f"{split}_near": int(n_near[i]), f"{split}_near_same_truth": int(n_near_same[i]),
                       f"{split}_max_jaccard": round(float(best[i]), 4), f"{split}_nn_count": int(best_n[i]),
                       f"{split}_nn_same_truth_share": round(float(best_same[i]), 4)})
    rows = list(per_case.values())
    summary = []
    for sname in list(SETS) + ["all"]:
        sub = [r for r in rows if sname == "all" or r["set"] == sname]
        n = len(sub)
        for split in list(SPLITS) + ["either"]:
            def col(r, k):
                if split != "either":
                    return r[f"{split}_{k}"]
                return sum(r[f"{s}_{k}"] for s in SPLITS)
            exact_cases = sum(1 for r in sub if col(r, "exact") > 0)
            exact_rows = sum(col(r, "exact") for r in sub)
            exact_same = sum(col(r, "exact_same_truth") for r in sub)
            exact_all_same = sum(1 for r in sub if col(r, "exact") > 0 and col(r, "exact_same_truth") == col(r, "exact"))
            near_cases = sum(1 for r in sub if col(r, "near") > 0)
            near_rows = sum(col(r, "near") for r in sub)
            near_same = sum(col(r, "near_same_truth") for r in sub)
            entry = {"set": sname, "split": split, "cases": n,
                     "cases_with_exact_twin": exact_cases, "exact_twin_rows": exact_rows,
                     "exact_twin_rows_same_truth": exact_same, "cases_whose_exact_twins_all_share_truth": exact_all_same,
                     "cases_with_exact_twin_same_age": sum(1 for r in sub if col(r, "exact_same_age") > 0),
                     "cases_with_near_twin": near_cases, "near_twin_rows": near_rows,
                     "near_twin_rows_same_truth": near_same}
            if split != "either":
                mj = np.array([r[f"{split}_max_jaccard"] for r in sub])
                nn_hit = np.array([r[f"{split}_nn_same_truth_share"] for r in sub])
                entry.update({"mean_max_jaccard": round(float(mj.mean()), 4),
                              "median_max_jaccard": round(float(np.median(mj)), 4),
                              "nn_accuracy": round(float((nn_hit > 0.5).mean()), 4),
                              "nn_accuracy_expected": round(float(nn_hit.mean()), 4)})
            summary.append(entry)
    return rows, summary


def twin_scaling(rows: list[dict], splits: dict) -> dict:
    """How the twin rate grows with the public rows, to bound the train split we do not have.
    The Poisson reading treats each case's twin count as proportional to the rows searched;
    predicting the test split from the validate split checks it at a ratio of about 1, and
    it under-predicts there (twins cluster on short evidence lists), so the train figure is
    a lower bound."""
    n_v, n_t = len(splits["validate"]["row"]), len(splits["test"]["row"])
    v = np.array([r["validate_exact"] for r in rows])
    t = np.array([r["test_exact"] for r in rows])
    train_adults = TRAIN_ROWS * (n_v + n_t) / (132_448 + 134_529)  # the adult share of the two local splits
    return {
        "cases": len(rows),
        "share_with_twin_validate_only": float((v > 0).mean()),
        "share_with_twin_test_only": float((t > 0).mean()),
        "share_with_twin_both_splits": float(((v + t) > 0).mean()),
        "rows_searched_validate": n_v, "rows_searched_both": n_v + n_t,
        "poisson_check_predicted_test_cases": float((1 - np.exp(-(n_t / n_v) * v)).sum()),
        "poisson_check_observed_test_cases": int((t > 0).sum()),
        "train_adult_rows_estimate": int(train_adults),
        "train_ratio_to_local": train_adults / (n_v + n_t),
        "poisson_train_cases_lower_bound": float((1 - np.exp(-(train_adults / (n_v + n_t)) * (v + t))).sum()),
    }


# ---------------------------------------------------------------- 2. shortcuts


def is_antecedent(code: str) -> bool:
    return bool(EVIDENCE_DATA.get(code, {}).get("is_antecedent", False))


def filtered_counts(counts: dict, keep) -> dict:
    return {"n": counts["n"], "codes": {c: {q: k for q, k in v.items() if keep(q)} for c, v in counts["codes"].items()}}


def readers(sets: dict) -> dict[str, NaiveBayes]:
    holdout = {c["case_id"] for s in sets.values() for c in s["cases"]}
    counts = train_counts(holdout, SPLITS["test"], ROOT / "results" / "analysis" / "v03_key" / "nb_counts_holdout_v03.json")
    n_sym = sum(1 for q in {q for v in counts["codes"].values() for q in v} if not is_antecedent(q))
    n_ant = sum(1 for q in {q for v in counts["codes"].values() for q in v} if is_antecedent(q))
    return {
        "nb-all": NaiveBayes(counts),
        "nb-symptoms": NaiveBayes(filtered_counts(counts, lambda q: not is_antecedent(q))),
        "nb-antecedents": NaiveBayes(filtered_counts(counts, is_antecedent)),
    }, {"training_rows": counts["rows"], "holdout": len(holdout), "symptom_codes": n_sym, "antecedent_codes": n_ant}


def reader_outputs(sets: dict, nbs: dict[str, NaiveBayes], tiers: dict[str, int]) -> dict[str, dict[str, dict]]:
    """Per set and reader: per-case top-1, top-5, tier-1 mass (percent) and YES at READER_T."""
    t1 = set(ak.tier1_conditions(tiers))
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for sname, s in sets.items():
        n = len(s["cases"])
        for rname, nb in nbs.items():
            t1_idx = np.array([c in t1 for c in nb.conditions])
            top1, top5, mass, yes = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), np.zeros(n, bool)
            for i, c in enumerate(s["cases"]):
                p = nb.posterior(c["presenting_symptoms"])
                order = np.argsort(-p, kind="stable")
                top1[i] = nb.conditions[order[0]] == s["truth"][i]
                top5[i] = s["truth"][i] in {nb.conditions[j] for j in order[:5]}
                mass[i] = 100.0 * float(p[t1_idx].sum())
                yes[i] = bool((100.0 * p[t1_idx] >= READER_T).any())
            out[sname][rname] = {"top1": top1, "top5": top5, "mass": mass, "yes": yes}
        # DXA: the case's own differential; YES from the reference row (red-herring rule applied).
        top1, top5, mass = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)
        dxa_yes = {r["case_id"]: r["serious_concern"] == "YES" for r in s["refs"]["dxa"]}
        for i, c in enumerate(s["cases"]):
            diff = sorted(c["ddxplus_differential"], key=lambda x: -float(x[1]))
            top1[i] = diff[0][0] == s["truth"][i]
            top5[i] = s["truth"][i] in {d[0] for d in diff[:5]}
            mass[i] = 100.0 * sum(float(p) for name, p in diff if name in t1)
        out[sname]["dxa"] = {"top1": top1, "top5": top5, "mass": mass,
                             "yes": np.array([dxa_yes[c["case_id"]] for c in s["cases"]])}
    return out


def auc_point(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    sp, sn = score[pos], score[neg]
    if not len(sp) or not len(sn):
        return float("nan")
    C = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
    return float(C.mean())


def auc_draws(score: np.ndarray, pos: np.ndarray, neg: np.ndarray, cond_idx: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Weighted AUC per bootstrap draw, cases weighted by their condition's multiplicity."""
    sp, sn = score[pos], score[neg]
    C = ((sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])).astype(float)
    Wp, Wn = M[:, cond_idx[pos]], M[:, cond_idx[neg]]
    num = ((Wp @ C) * Wn).sum(1)
    den = Wp.sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def rate_draws(v: np.ndarray, cond_idx: np.ndarray, k: int, M: np.ndarray) -> np.ndarray:
    num = np.bincount(cond_idx, weights=v.astype(float), minlength=k)
    den = np.bincount(cond_idx, minlength=k).astype(float)
    return (M @ num) / (M @ den)


def shortcut_table(sets: dict, outs: dict) -> tuple[list[dict], dict]:
    """Per set and reader: rates and AUCs; on the main sample with cluster-bootstrap intervals
    (conditions resampled, as in evaluator/v03_score.py) and paired differences against DXA."""
    rows, boots = [], {}
    main = sets["main"]
    conds = sorted(set(main["truth"].tolist()))
    cidx = np.array([conds.index(t) for t in main["truth"]])
    M = cluster_draws(len(conds), N_BOOTSTRAP, BOOTSTRAP_SEED)
    for sname, s in sets.items():
        pos_t1, neg_t1 = s["tier1"], ~s["tier1"]
        pos_h, neg_h = s["has_r10"], s["clearly_low"]
        for rname, o in outs[sname].items():
            row = {"set": sname, "reader": rname, "cases": len(s["cases"]),
                   "top1": 100 * o["top1"].mean(), "top5": 100 * o["top5"].mean(), "yes_rate": 100 * o["yes"].mean(),
                   "yes_on_tier1_truth": 100 * o["yes"][pos_t1].mean() if pos_t1.any() else float("nan"),
                   "yes_on_r10": 100 * o["yes"][pos_h].mean() if pos_h.any() else float("nan"),
                   "yes_on_clearly_low": 100 * o["yes"][neg_h].mean() if neg_h.any() else float("nan"),
                   "auc_tier1_truth": auc_point(o["mass"], pos_t1, neg_t1),
                   "auc_r10_vs_clearly_low": auc_point(o["mass"], pos_h, neg_h)}
            if sname == "main":
                b = {"top1": 100 * rate_draws(o["top1"], cidx, len(conds), M),
                     "top5": 100 * rate_draws(o["top5"], cidx, len(conds), M),
                     "auc_tier1_truth": auc_draws(o["mass"], pos_t1, neg_t1, cidx, M),
                     "auc_r10_vs_clearly_low": auc_draws(o["mass"], pos_h, neg_h, cidx, M)}
                boots[rname] = b
                for m, d in b.items():
                    lo, hi = interval(d)
                    row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
            rows.append(row)
    diffs = {}
    for rname in boots:
        for other in ("dxa", "nb-all"):
            if rname == other:
                continue
            for m in ("top1", "auc_tier1_truth", "auc_r10_vs_clearly_low"):
                pa = [r for r in rows if r["set"] == "main" and r["reader"] == rname][0][m]
                pb = [r for r in rows if r["set"] == "main" and r["reader"] == other][0][m]
                diffs[f"{rname} minus {other}: {m}"] = {"diff": pa - pb, "ci": interval(boots[rname][m] - boots[other][m])}
    # Pools together: tier-1 truths (atypical) against non-tier-1 truths (high-risk); every case has an R10 target.
    pooled = {}
    for rname in outs["pool-atypical"]:
        mass = np.concatenate([outs["pool-atypical"][rname]["mass"], outs["pool-high-risk"][rname]["mass"]])
        pos = np.concatenate([np.ones(len(sets["pool-atypical"]["cases"]), bool), np.zeros(len(sets["pool-high-risk"]["cases"]), bool)])
        pooled[rname] = auc_point(mass, pos, ~pos)
    return rows, {"paired": diffs, "pools_auc_tier1_truth": pooled, "n_conditions_main": len(conds)}


# ---------------------------------------------------------------- 3. rendering


def frequent_strings(sets: dict) -> tuple[list[dict], dict]:
    counts, ante = Counter(), {}
    per_set = {}
    for sname, s in sets.items():
        replaced, cases_hit, occurrences = 0, 0, 0
        for c in s["cases"]:
            a, h, _ = decode_symptoms_versioned(c["presenting_symptoms"], "v02", c["sex"])
            for x in a:
                counts[x] += 1
                ante[x] = False
            for x in h:
                counts[x] += 1
                ante[x] = True
            hit = sum(1 for x in a + h if x in R.PARAPHRASE)
            replaced += hit
            occurrences += len(a) + len(h)
            cases_hit += hit > 0
        per_set[sname] = {"cases": len(s["cases"]), "string_occurrences": occurrences, "replaced": replaced,
                          "replaced_share": replaced / occurrences, "cases_with_a_replacement": cases_hit,
                          "mean_replaced_per_case": replaced / len(s["cases"])}
    total = sum(counts.values())
    rows, cum = [], 0
    for rank, (s, k) in enumerate(counts.most_common(), 1):
        cum += k
        rows.append({"rank": rank, "count": k, "share": k / total, "cumulative_share": cum / total,
                     "is_antecedent": ante[s], "in_paraphrase_table": s in R.PARAPHRASE, "string": s})
    top = {r["string"] for r in rows[:TOP_STRINGS]}
    return rows, {"distinct_strings": len(counts), "occurrences": total, "per_set": per_set,
                  "top40_share": rows[TOP_STRINGS - 1]["cumulative_share"],
                  "table_is_top40": top == set(R.PARAPHRASE)}


def rendering_examples(sets: dict) -> str:
    lines = []
    for cond in EXAMPLE_CONDITIONS:
        c = next(c for c, k in zip(sets["main"]["cases"], sets["main"]["keys"]) if k.truth == cond)
        c = {**c, "working_diagnosis_name": c.get("working_diagnosis_name", "")}
        lines.append(f"### {c['case_id']}: {cond}, {c['age']}-year-old {c['sex']}\n")
        for rendering in R.RENDERINGS:
            text = format_case_for_prompt_v7(dict(c), "v7a1", "v02", rendering)
            body = [ln for ln in text.splitlines() if ln.startswith(("Chief Complaints:", "Medical History / Context:"))]
            lines.append(f"**{rendering}**\n")
            lines.append("```\n" + "\n".join(body) + "\n```\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- output


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = load_sets()
    tiers = ak.load_tiers()
    print("sets:", {k: len(v["cases"]) for k, v in sets.items()})

    splits = {name: load_split(name) for name in SPLITS}
    print("adult rows:", {k: len(v["row"]) for k, v in splits.items()})
    dup_rows, dup_summary = duplicate_audit(sets, splits)
    write_csv(OUT / "duplicates.csv", dup_rows)
    write_csv(OUT / "duplicates_summary.csv", dup_summary)
    for r in dup_summary:
        if r["set"] == "all" or r["split"] == "either":
            print(r)
    scaling = twin_scaling(dup_rows, splits)
    print("twin scaling:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scaling.items()})
    by_length = {"with_twin": float(np.median([r["n_tokens"] for r in dup_rows if r["test_exact"] + r["validate_exact"] > 0])),
                 "without_twin": float(np.median([r["n_tokens"] for r in dup_rows if r["test_exact"] + r["validate_exact"] == 0]))}
    twin_conditions = Counter(r["truth"] for r in dup_rows if r["test_exact"] + r["validate_exact"] > 0).most_common(8)
    nn_wrong = [(r["case_id"], r["truth"]) for r in dup_rows if r["validate_nn_same_truth_share"] <= 0.5]

    nbs, nb_meta = readers(sets)
    outs = reader_outputs(sets, nbs, tiers)
    short_rows, short_meta = shortcut_table(sets, outs)
    write_csv(OUT / "shortcuts.csv", short_rows)
    for r in short_rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith(("_lo", "_hi"))})
    for k, v in short_meta["paired"].items():
        print(k, round(v["diff"], 3), [round(x, 3) for x in v["ci"]])
    print("pools AUC", short_meta["pools_auc_tier1_truth"])

    freq_rows, freq_meta = frequent_strings(sets)
    write_csv(OUT / "frequent_strings.csv", freq_rows)
    (OUT / "rendering_examples.md").write_text(rendering_examples(sets))
    print("strings:", freq_meta)

    def jsonable(x):
        if isinstance(x, dict):
            return {str(k): jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [jsonable(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        return x

    summary = {
        "splits_adult_rows": {k: len(v["row"]) for k, v in splits.items()},
        "train_rows_not_in_checkout": TRAIN_ROWS,
        "near_threshold": NEAR,
        "duplicates": dup_summary,
        "twin_scaling": scaling,
        "twin_median_tokens": by_length,
        "twin_conditions": twin_conditions,
        "nn_wrong_cases": nn_wrong,
        "naive_bayes": nb_meta,
        "shortcuts": short_rows,
        "shortcut_paired": short_meta["paired"],
        "pools_auc_tier1_truth": short_meta["pools_auc_tier1_truth"],
        "strings": freq_meta,
        "bootstrap": {"draws": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "clusters": short_meta["n_conditions_main"]},
    }
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
