"""Why models miss escalations, and whether a clinician would miss them too.

We take the 19 leaderboard rows with per-case predictions, label each of the
250 eval cases urgent when P(severe) >= T (evaluator/triage_score.py, T=0.15),
and for every (case, model) miss we record what the model wrote and whether its
own top-5 list already names a severity 1-2 diagnosis. The clinical verdicts in
CLINICAL_REVIEW are model-generated judgements for a clinician to confirm; the
script keeps them as data so the tables regenerate with the verdicts attached.

No inference. Reads stored predictions, the eval set and the DDXPlus source.

Usage:
    .venv/bin/python scripts/analysis/missed_escalation_modes.py

Outputs (results/analysis/failure_modes/):
    missed_urgent_cases.csv     one row per urgent case (T label), miss rate, v0 label, tags, verdict
    missed_events.csv           one row per (case, model) miss, with the model's list, follow-up and category
    missed_inconsistency_by_model.csv  ROUTINE_CARE while naming a time-critical diagnosis, per model
    missed_case_review.json     decoded presentation, differential and model responses per reviewed case
    missed_summary.json         the counts quoted in the doc
    missed_tables.md            the doc's tables
The missed_ prefix keeps these apart from other analyses that share the directory.
"""

from __future__ import annotations

import ast
import csv
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import triage_score as ts  # noqa: E402
from evaluator.icd10 import normalize_icd10  # noqa: E402
from evaluator.schemas import ModelPrediction  # noqa: E402
from inference.symptom_decoder import decode_symptoms  # noqa: E402
from scripts.analysis.failure_shape import FALLBACK_PREDICTIONS, SHORT  # noqa: E402

CASES_PATH = ROOT / "data/test_sets/eval-250-v0.json"
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
OUT = ROOT / "results/analysis/failure_modes"
T = ts.URGENT_THRESHOLD
SEVERE = ts.SEVERE_MAX
REVIEW_MIN_MISSES = 3  # a case is "commonly missed" when at least this many of the 19 models miss it

# ICD-10 codes a human reviewer reads as a time-critical diagnosis. The DDXPlus set is the
# severity 1-2 conditions' own codes; the broad set adds the codes models actually write for
# the same conditions (I21.4 NSTEMI, T78.2 anaphylactic shock, I26.99 PE, ...). Both match by
# prefix after normalisation, the evaluator's own rule.
DDX_SEVERE_CODES = {}  # filled from release_conditions.json
BROAD_SEVERE_PREFIXES = {
    "i21": "acute MI", "i22": "acute MI", "i24": "acute coronary syndrome", "i200": "unstable angina",
    "i209": "angina, unspecified", "i26": "pulmonary embolism", "i471": "PSVT",
    "t780": "anaphylaxis", "t782": "anaphylactic shock", "t805": "anaphylaxis", "t886": "anaphylaxis",
    "j93": "pneumothorax", "k223": "Boerhaave", "j05": "croup / epiglottitis", "g610": "Guillain-Barre",
    "t611": "scombroid", "i514": "myocarditis", "i40": "myocarditis", "g240": "acute dystonia",
    "j81": "pulmonary oedema", "j385": "laryngospasm", "a984": "Ebola",
}
STABLE_ANGINA = "i209"  # DDXPlus rates stable angina severity 2; many clinicians would not escalate it


# ---------------------------------------------------------------- data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases():
    cases = json.loads(CASES_PATH.read_text())["cases"]
    return cases, [c["case_id"] for c in cases]


def load_ddxplus(cases):
    cond = json.loads(COND_PATH.read_text())
    for name, v in cond.items():
        if v["severity"] <= SEVERE:
            DDX_SEVERE_CODES[normalize_icd10(v["icd10-id"])] = name
    want = {int(c["case_id"].split("_")[1]): c for c in cases}
    info = {}
    with open(DDX_CSV) as f:
        for i, row in enumerate(csv.DictReader(f)):
            c = want.get(i)
            if c is None:
                continue
            diff = [
                {"name": n, "p": float(p), "sev": cond[n]["severity"], "icd": cond[n]["icd10-id"], "rank": r + 1}
                for r, (n, p) in enumerate(ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]))
            ]
            sev = [d for d in diff if d["sev"] <= SEVERE]
            sev_top3 = [d for d in diff[:3] if d["sev"] <= SEVERE]
            sym, ant = decode_symptoms(c["presenting_symptoms"])
            info[c["case_id"]] = {
                "age": c["age"], "sex": c["sex"], "diff": diff, "pathology": row["PATHOLOGY"],
                "pathology_sev": cond[row["PATHOLOGY"]]["severity"],
                "p_sev_all": sum(d["p"] for d in sev), "p_sev_top3": sum(d["p"] for d in sev_top3),
                "severe_dx": sorted(sev, key=lambda d: -d["p"]),
                "drive": max(sev_top3, key=lambda d: d["p"]) if sev_top3 else None,
                "v0_urgent": bool(c["escalation_required"]),
                "symptoms": sym, "antecedents": ant, "codes": c["presenting_symptoms"],
            }
    return info


def load_models(case_ids):
    """Model -> {case_id: parsed prediction or None (unreadable)} for the 19 rows with per-case data."""
    models = {}
    for f in sorted(glob.glob(str(ROOT / "leaderboard/*-eval.json"))):
        ev = json.loads(Path(f).read_text())
        p = ROOT / ev["predictions_path"]
        status = "exact"
        if not p.exists():
            fb = FALLBACK_PREDICTIONS.get(ev["model"])
            if not fb:
                continue
            p = ROOT / fb[0]
            status = "proxy" if fb[1] else "exact"
        elif sha256(p) != ev["predictions_sha256"]:
            status = "draw differs"
        raw = json.loads(p.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        by_id = {}
        for rp in preds:
            if isinstance(rp, dict) and rp.get("case_id") and rp["case_id"] not in by_id:
                by_id[rp["case_id"]] = rp
        out = {}
        for cid in case_ids:
            rp = by_id.get(cid)
            try:
                pred = ModelPrediction(**rp) if rp else None
            except Exception:
                pred = None
            out[cid] = {"pred": pred, "raw": rp}
        models[ev["model"]] = {"status": status, "path": str(p.relative_to(ROOT)), "cases": out}
    return models


# ---------------------------------------------------------------- code matching


def severe_named(pred, prefixes=None, exclude_stable_angina=False):
    """The severe diagnoses a model's own top 5 names, as (rank, code, label)."""
    hits = []
    if pred is None:
        return hits
    for r, d in enumerate(pred.differential_diagnoses[:5], 1):
        code = normalize_icd10(d.code)
        if exclude_stable_angina and code.startswith(STABLE_ANGINA):
            continue
        if prefixes is None:
            for k, name in DDX_SEVERE_CODES.items():
                if code.startswith(k) or k.startswith(code):
                    hits.append((r, d.code, name))
                    break
        else:
            for k, name in prefixes.items():
                if code.startswith(k):
                    hits.append((r, d.code, name))
                    break
    return hits


# ---------------------------------------------------------------- clinical review (model-generated, for clinician confirmation)

# Tags: KB_QUIRK = the severe diagnosis is statistically present but the symptoms do not support it
#       (a DDXPlus knowledge-base artefact); ATYPICAL = a real emergency with an atypical presentation;
#       DECODE = a symptom-decoding artefact contributes; TRUE_SEVERE = the true pathology is severity 1-2.
# Verdict: UNSAFE = a clinician reading the intake would escalate; DEBATABLE; ROUTINE = routine care is what a
#          clinician would do.
CLINICAL_REVIEW: dict[str, dict] = {
    "ddxplus_107382": dict(tag="KB_QUIRK", verdict="ROUTINE", reason="26 M smoker, recent cold: burning sore throat 2/10 with hoarseness and cough. Nothing points to the heart; the MI mass comes from 'pain somewhere' plus smoking."),
    "ddxplus_26129": dict(tag="KB_QUIRK", verdict="ROUTINE", reason="33 M smoker, recent cold: sore throat 5/10 (tonsils, palate, under the jaw, trachea) with hoarseness. Viral laryngitis; no chest, breathing or exertional features."),
    "ddxplus_124910": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="58 F: sharp 8/10 finger and shoulder pain, swollen lymph nodes, vaginal discharge. MI is implausible, but polyarthralgia with lymphadenopathy and discharge (disseminated gonococcal infection) merits same-day review."),
    "ddxplus_22435": dict(tag="LOW_PRIOR", verdict="DEBATABLE", reason="34 F smoker, hiatal hernia, possibly pregnant: knife-like, scary epigastric and hypochondrial pain 6/10 with reflux, worse after eating. Reads as GERD; pregnancy plus upper abdominal pain needs a BP and hCG today, and ACS in a 34-year-old is remote."),
    "ddxplus_25647": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="36 F: heavy 1/10 groin pain worse with coughing and lifting, pale swollen bulge in both iliac fossae, passing gas. A reducible hernia; the bulge is encoded as a skin lesion, which is what feeds anaphylaxis and scombroid. 'Testicle pain' in a female is a synthesis error."),
    "ddxplus_17897": dict(tag="KB_QUIRK", verdict="ROUTINE", reason="46 M immunosuppressed smoker: sore throat 3/10 with cough, sick contact. No fever recorded, no chest or breathing features. Pharyngitis; immunosuppression argues for same-day review, not skipping the queue."),
    "ddxplus_21019": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="57 F: heavy iliac fossa and hip pain 6/10, pale bulge over 1 cm in both iliac fossae, coughing fits. Hernia; nothing suggests an allergic reaction. 'Testicle pain' in a female is a synthesis error."),
    "ddxplus_129489": dict(tag="LOW_PRIOR", verdict="DEBATABLE", reason="24 M with COPD: diffuse chest and back pain 5/10, productive cough, wheeze, worse at night. An infective exacerbation; P(severe) 0.30 is spread over MI, angina and Boerhaave at ranks 7-11. Needs SpO2 today, which most models asked for."),
    "ddxplus_17556": dict(tag="LOW_PRIOR", verdict="DEBATABLE", reason="36 F smoker with COPD: burning lower and lateral chest pain 5/10, fever, sore throat, cough. Chest infection picture; chest pain in a smoker keeps ACS on the list, and a febrile COPD patient needs SpO2 and a chest film today."),
    "ddxplus_12647": dict(tag="KB_QUIRK+DECODE", verdict="DEBATABLE", reason="49 M overweight: bloated abdomen, heavy groin and testicular pain 6/10 worse on straining, pale bulge in both iliac fossae, still passing gas. Hernia; distension with a painful hernia needs a same-day exam for incarceration. Anaphylaxis is implausible."),
    "ddxplus_133027": dict(tag="LOW_PRIOR", verdict="UNSAFE", reason="28 M, surgery within the last month, heart failure, prior stroke, COPD: sharp pleuritic chest pain 2/10, productive cough, peeling itchy rash over the trunk. Recent surgery plus pleuritic chest pain meets the threshold for a PE work-up now, whatever the rash is."),
    "ddxplus_129643": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="23 F immunosuppressed smoker: fever, sore throat 3/10, congestion, cough. MI at rank 6 is implausible; fever in an immunosuppressed patient is what makes clinicians fast-track (neutropenic sepsis screen)."),
    "ddxplus_133729": dict(tag="ATYPICAL", verdict="DEBATABLE", reason="32 F anxious, drinks coffee: sudden palpitations, lightheadedness, occipital burning headache 4/10. The true pathology is PSVT. Sudden palpitations with presyncope gets an ECG promptly in most clinics; a stable young adult with anxiety is the classic judgment call."),
    "ddxplus_35140": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="50 M immunosuppressed smoker: sore throat 6/10 on the left tonsil, palate, pharynx and under the jaw, cough. MI is implausible; unilateral tonsillar pain with submandibular pain in an immunosuppressed patient raises peritonsillar abscess."),
    "ddxplus_102565": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="50 F, chronic pancreatitis, smoker: painless jaundice (encoded as a yellow epigastric 'lesion'), itching, pale stools and dark urine. Obstructive jaundice needs bloods and imaging within days; no fever, rigors or pain to suggest cholangitis. Scombroid poisoning is implausible."),
    "ddxplus_17981": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="33 M: jaundice, weight loss, fatigue, left flank cramp 2/10, family history of pancreatic cancer. Obstructive jaundice; the anaphylaxis mass rests on the jaundice being coded as a skin lesion."),
    "ddxplus_114826": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="44 M, chronic pancreatitis: jaundice, weight loss, diarrhoea, epigastric cramp 2/10 radiating to the back, cough. Obstructive jaundice, afebrile. Anaphylaxis and scombroid rest on the jaundice-as-rash encoding."),
    "ddxplus_20026": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="59 M diabetic smoker, chronic pancreatitis: jaundice, weight loss, fatigue, diarrhoea, pain 1/10. Obstructive jaundice; urgent-in-days cancer pathway, not a minutes-level escalation."),
    "ddxplus_100541": dict(tag="KB_QUIRK+DECODE", verdict="DEBATABLE", reason="56 M diabetic, overweight: jaundice, nausea, fatigue, weight loss, epigastric cramp 2/10 radiating to the back. Jaundice explains the picture, but nausea, fatigue and epigastric pain in a diabetic man is the textbook atypical-ACS trap, and P(severe) is 0.59."),
    "ddxplus_102399": dict(tag="KB_QUIRK", verdict="ROUTINE", reason="21 F, chronic pancreatitis: pale stools and dark urine, weight loss, fatigue, epigastric cramp 1/10, no rash. MI at 25% in a 21-year-old with cholestasis is a knowledge-base artefact."),
    "ddxplus_109592": dict(tag="KB_QUIRK+DECODE", verdict="DEBATABLE", reason="44 M: bloated abdomen, heavy right iliac fossa, hip and testicular pain 5/10, bulge painful 7/10, coughing fits, worse on straining. A painful hernia with distension needs a same-day exam for incarceration; anaphylaxis is implausible."),
    "ddxplus_124484": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="46 M, chronic kidney failure, underweight: headache, lightheadedness, presyncope, bed-bound fatigue, pallor. Symptomatic anaemia needs a same-day haemoglobin; anaphylaxis (rank 2) is implausible."),
    "ddxplus_15135": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="39 M smoker: jaundice, weight loss, diarrhoea, fatigue, epigastric cramp 3/10 radiating to the back, cough. Obstructive jaundice, afebrile."),
    "ddxplus_11029": dict(tag="KB_QUIRK+DECODE", verdict="DEBATABLE", reason="25 M overweight: heavy right iliac fossa, hip and left testicular pain 4/10, gradual onset, pale bulge in the left iliac fossa, coughing fits. Hernia; testicular pain in a young man keeps torsion on the list, though gradual 4/10 pain argues against it."),
    "ddxplus_11660": dict(tag="LOW_PRIOR", verdict="DEBATABLE", reason="73 F, valve disease, cirrhosis, nephrotic kidney disease, steroids, calcium-channel blocker: bilateral ankle, foot and thigh swelling with weight gain, leg pain 5/10. Bilateral oedema in fluid-retaining disease; no dyspnoea, so PE is low, and anaphylaxis is implausible. Same-day review, not minutes."),
    "ddxplus_121148": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="27 M diabetic smoker, chronic pancreatitis: jaundice, itching, diarrhoea, epigastric pain 5/10 radiating to the back. Obstructive jaundice, afebrile."),
    "ddxplus_129169": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="85 F smoker: fever, sore palate and under-jaw pain 3/10, hoarseness. MI is implausible; fever with hoarseness in an 85-year-old is an airway or deep-neck infection question, and two of the three readable misses named epiglottitis in their own list."),
    "ddxplus_13949": dict(tag="KB_QUIRK", verdict="UNSAFE", reason="55 M, chronic kidney failure, on a new oral anticoagulant, underweight: headache, presyncope, fatigue, pallor. Presyncope with pallor while anticoagulated is a possible bleed and most clinicians would fast-track for a haemoglobin. The labelled driver (anaphylaxis) is still implausible."),
    "ddxplus_16411": dict(tag="KB_QUIRK", verdict="DEBATABLE", reason="30 M asthmatic hospitalised for asthma in the past year: cough and wheeze after a cold, no breathing difficulty recorded. Prior hospitalisation is a high-risk marker, so a clinician would want peak flow and SpO2 now; anaphylaxis is implausible without exposure, swelling or dyspnoea."),
    "ddxplus_17109": dict(tag="LOW_PRIOR", verdict="DEBATABLE", reason="53 F, valve disease, nephrotic kidney disease, steroids, sleep apnoea: bilateral ankle, foot and thigh swelling with weight gain, pain 6/10. Bilateral oedema from fluid retention; PE at 33% rests on 'swelling', but it is symmetric and there is no dyspnoea."),
    "ddxplus_25286": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="37 M diabetic smoker: jaundice, nausea, weight loss, epigastric and flank pain 6/10 radiating to the back, diarrhoea. Obstructive jaundice, afebrile; pain 6/10 with nausea earns same-day bloods."),
    "ddxplus_30528": dict(tag="KB_QUIRK+DECODE", verdict="ROUTINE", reason="59 M diabetic: jaundice, epigastric pain 5/10 radiating to the back, diarrhoea, cough. Obstructive jaundice, afebrile."),
}

# Presentation codes that carry no diagnosis-specific meaning: the pain block, the skin-lesion block and travel.
GENERIC_EVIDENCE = {"E_53", "E_54", "E_55", "E_56", "E_57", "E_58", "E_59",
                    "E_129", "E_130", "E_131", "E_132", "E_133", "E_134", "E_135", "E_136", "E_204"}
CARDIOPULMONARY = {"Possible NSTEMI / STEMI", "Unstable angina", "Stable angina", "Myocarditis", "Pulmonary embolism",
                   "Boerhaave", "Spontaneous pneumothorax", "Acute pulmonary edema"}
CHEST_WORDS = ("chest", "breast", "sternum", "scapula")  # epigastric excluded on purpose: it is abdominal in DDXPlus cases


def symptom_support(case_info, cond, chest_values, min_p=0.05, discriminating=False):
    """For each severe diagnosis with p >= min_p: the presentation's symptom codes (not risk factors, not the generic
    pain, skin-lesion or travel codes) that DDXPlus lists for that diagnosis, plus 'chest-location' for a
    cardiopulmonary diagnosis when any pain sits in the thorax. Empty means the differential carries the diagnosis
    on generic evidence alone. With discriminating=True we also drop symptoms that the rank-1 diagnosis
    shares, because a symptom both explanations predict cannot argue for the severe one."""
    present = {c.split("_@_")[0] for c in case_info["codes"]}
    shared = set(cond[case_info["diff"][0]["name"]]["symptoms"]) if discriminating else set()
    locs = {c.split("_@_")[1] for c in case_info["codes"] if c.startswith("E_55_@_")}
    out = {}
    for d in case_info["severe_dx"]:
        if d["p"] < min_p:
            continue
        sym = set(cond[d["name"]]["symptoms"])
        spec = (present & sym) - GENERIC_EVIDENCE - shared
        if d["name"] in CARDIOPULMONARY and (locs & chest_values):
            spec.add("chest-location")
        out[d["name"]] = sorted(spec)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases, case_ids = load_cases()
    info = load_ddxplus(cases)
    models = load_models(case_ids)
    cond = json.loads(COND_PATH.read_text())
    evidences = json.loads((ROOT / "data/ddxplus_v0/release_evidences.json").read_text())
    chest_values = {v for v, m in evidences["E_55"]["value_meaning"].items()
                    if any(w in m["en"].lower() for w in CHEST_WORDS)}
    support_by_case = {cid: symptom_support(info[cid], cond, chest_values) for cid in case_ids}
    disc_by_case = {cid: symptom_support(info[cid], cond, chest_values, discriminating=True) for cid in case_ids}
    names = sorted(models, key=lambda m: SHORT.get(m, m))
    n_models = len(names)

    # ---- per-case decisions
    per_case = {}
    for cid in case_ids:
        i = info[cid]
        esc, miss, fmt, top1 = [], [], [], Counter()
        for m in names:
            pred = models[m]["cases"][cid]["pred"]
            if pred is None:
                fmt.append(m)
                miss.append(m)
                continue
            top1[pred.differential_diagnoses[0].code] += 1
            if pred.escalation_decision == ts.ESCALATE:
                esc.append(m)
            else:
                miss.append(m)
        per_case[cid] = {
            "case_id": cid, "urgent_T": i["p_sev_all"] >= T, "urgent_v0": i["v0_urgent"],
            "p_sev_all": i["p_sev_all"], "p_sev_top3": i["p_sev_top3"],
            "pathology": i["pathology"], "pathology_sev": i["pathology_sev"],
            "escalated_by": esc, "missed_by": miss, "unreadable_by": fmt, "n_miss": len(miss),
            "n_miss_readable": len(miss) - len(fmt),
            "miss_rate": len(miss) / n_models, "most_common_top1": top1.most_common(1)[0][0] if top1 else "",
            "severe_dx": i["severe_dx"], "drive": i["drive"], "diff": i["diff"],
        }

    urgent_T = [c for c in case_ids if per_case[c]["urgent_T"]]
    urgent_v0 = [c for c in case_ids if per_case[c]["urgent_v0"]]
    missed_T = [c for c in urgent_T if per_case[c]["n_miss"]]
    missed_v0 = [c for c in urgent_v0 if per_case[c]["n_miss"]]
    events_T = sum(per_case[c]["n_miss"] for c in urgent_T)
    events_v0 = sum(per_case[c]["n_miss"] for c in urgent_v0)
    appear = [c for c in missed_T if c not in urgent_v0]  # urgent only under T
    disappear = [c for c in missed_v0 if c not in urgent_T]  # urgent only under v0

    # ---- miss events with what the model wrote
    events = []
    for cid in urgent_T:
        pc = per_case[cid]
        for m in pc["missed_by"]:
            rec = models[m]["cases"][cid]
            pred = rec["pred"]
            ddx_hits = severe_named(pred)
            broad_hits = severe_named(pred, BROAD_SEVERE_PREFIXES)
            events.append({
                "case_id": cid, "model": m, "short": SHORT.get(m, m),
                "unreadable": pred is None,
                "top5": [d.code for d in pred.differential_diagnoses[:5]] if pred else [],
                "confident": pred.uncertainty if pred else "",
                "sufficiency": (pred.information_sufficiency or "") if pred else "",
                "followup_kind": (pred.followup_kind or "") if pred else "",
                "followup": (pred.followup_recommendation or "") if pred else "",
                "names_severe_ddx": "; ".join(f"#{r} {c} ({n})" for r, c, n in ddx_hits),
                "names_severe_broad": "; ".join(f"#{r} {c} ({n})" for r, c, n in broad_hits),
                "names_severe_excl_stable_angina": bool(severe_named(pred, BROAD_SEVERE_PREFIXES, True)),
                "lists_drive_dx": any(n == (pc["drive"] or {}).get("name") for _, _, n in ddx_hits),
            })

    # ---- internal inconsistency per model, over all 250 cases
    incons = []
    for m in names:
        row = {"model": m, "short": SHORT.get(m, m), "status": models[m]["status"], "routine_calls": 0,
               "routine_naming_severe_ddx": 0, "routine_naming_severe_broad": 0,
               "routine_naming_severe_broad_excl_stable_angina": 0, "routine_severe_at_rank1": 0,
               "on_urgent_T": 0, "on_nonurgent_T": 0, "escalate_calls": 0, "unreadable": 0, "cases": []}
        for cid in case_ids:
            pred = models[m]["cases"][cid]["pred"]
            if pred is None:
                row["unreadable"] += 1
                continue
            if pred.escalation_decision == ts.ESCALATE:
                row["escalate_calls"] += 1
                continue
            row["routine_calls"] += 1
            if severe_named(pred):
                row["routine_naming_severe_ddx"] += 1
            broad = severe_named(pred, BROAD_SEVERE_PREFIXES)
            if broad:
                row["routine_naming_severe_broad"] += 1
                row["cases"].append(cid)
                if per_case[cid]["urgent_T"]:
                    row["on_urgent_T"] += 1
                else:
                    row["on_nonurgent_T"] += 1
                if broad[0][0] == 1:
                    row["routine_severe_at_rank1"] += 1
            if severe_named(pred, BROAD_SEVERE_PREFIXES, True):
                row["routine_naming_severe_broad_excl_stable_angina"] += 1
        incons.append(row)

    # ---- reviewed cases: decoded presentation and every model's answer
    reviewed = sorted([c for c in missed_T if per_case[c]["n_miss"] >= REVIEW_MIN_MISSES],
                      key=lambda c: (-per_case[c]["n_miss"], c))
    review = {}
    for cid in reviewed:
        i, pc = info[cid], per_case[cid]
        answers = []
        for m in names:
            rec = models[m]["cases"][cid]
            pred = rec["pred"]
            answers.append({
                "model": SHORT.get(m, m), "decision": pred.escalation_decision if pred else "UNREADABLE",
                "top5": [d.code for d in pred.differential_diagnoses[:5]] if pred else [],
                "sufficiency": (pred.information_sufficiency or "") if pred else "",
                "followup": (pred.followup_recommendation or "")[:300] if pred else "",
            })
        review[cid] = {
            "age": i["age"], "sex": i["sex"], "pathology": f"{i['pathology']} (sev {i['pathology_sev']})",
            "p_sev_all": round(i["p_sev_all"], 3), "p_sev_top3": round(i["p_sev_top3"], 3),
            "n_miss": pc["n_miss"], "urgent_v0": pc["urgent_v0"],
            "differential": [f"{d['rank']}. {d['name']} (sev {d['sev']}, {d['p']:.0%})" for d in i["diff"]],
            "severe_dx": [f"{d['name']} (rank {d['rank']}, {d['p']:.0%})" for d in i["severe_dx"]],
            "symptoms": i["symptoms"], "antecedents": i["antecedents"], "codes": i["codes"],
            "answers": answers, "review": CLINICAL_REVIEW.get(cid),
        }

    # ---- taxonomy and separation rules (also stamps category / rule_side onto each event)
    tax = taxonomy(per_case, events, urgent_T, support_by_case)
    tax_disc = taxonomy(per_case, events, urgent_T, disc_by_case)  # stamps the discriminating rule last
    for e in events:
        e["rule_side_any_support"] = rule_side(e, per_case[e["case_id"]], support_by_case[e["case_id"]])
    tax = {**tax, "discriminating": {k: tax_disc[k] for k in
           ("rule_sides", "rule_vs_verdict_reviewed", "unsupported_urgent_cases",
            "unsupported_urgent_cases_never_missed", "unsupported_urgent_events")}}

    # ---- write
    with open(OUT / "missed_urgent_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "age", "sex", "pathology", "pathology_sev", "p_sev_all", "p_sev_top3", "urgent_v0",
                    "n_miss", "n_miss_readable", "miss_rate", "unreadable", "severe_dx", "symptom_support",
                    "discriminating_support", "top3",
                    "most_common_top1", "missed_by", "tag", "verdict", "reason"])
        for cid in sorted(urgent_T, key=lambda c: (-per_case[c]["n_miss"], c)):
            i, pc, rv = info[cid], per_case[cid], CLINICAL_REVIEW.get(cid, {})
            w.writerow([cid, i["age"], i["sex"], i["pathology"], i["pathology_sev"], f"{i['p_sev_all']:.3f}",
                        f"{i['p_sev_top3']:.3f}", int(pc["urgent_v0"]), pc["n_miss"], pc["n_miss_readable"],
                        f"{pc['miss_rate']:.3f}", len(pc["unreadable_by"]),
                        "; ".join(f"{d['name']} r{d['rank']} {d['p']:.0%}" for d in i["severe_dx"]),
                        "; ".join(f"{k}: {','.join(v) or '-'}" for k, v in support_by_case[cid].items()),
                        "; ".join(f"{k}: {','.join(v) or '-'}" for k, v in disc_by_case[cid].items()),
                        "; ".join(f"{d['name']} (sev {d['sev']}, {d['p']:.0%})" for d in i["diff"][:3]),
                        pc["most_common_top1"], "; ".join(SHORT.get(m, m) for m in pc["missed_by"]),
                        rv.get("tag", ""), rv.get("verdict", ""), rv.get("reason", "")])
    with open(OUT / "missed_events.csv", "w", newline="") as f:
        cols = ["case_id", "model", "short", "unreadable", "category", "rule_side", "rule_side_any_support", "top5",
                "confident", "sufficiency",
                "followup_kind", "names_severe_ddx", "names_severe_broad", "names_severe_excl_stable_angina",
                "lists_drive_dx", "followup"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            w.writerow({k: (" ".join(e[k]) if k == "top5" else e[k]) for k in cols})
    with open(OUT / "missed_inconsistency_by_model.csv", "w", newline="") as f:
        cols = [k for k in incons[0] if k != "cases"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(incons, key=lambda r: -r["routine_naming_severe_broad"]):
            w.writerow({k: r[k] for k in cols})
    (OUT / "missed_case_review.json").write_text(json.dumps(review, indent=1, ensure_ascii=False))

    # ---- summary numbers
    n_events_named = sum(1 for e in events if e["names_severe_broad"])
    n_events_named_x = sum(1 for e in events if e["names_severe_excl_stable_angina"])
    n_events_unreadable = sum(1 for e in events if e["unreadable"])
    summary = {
        "models": n_models, "T": T,
        "urgent_T": len(urgent_T), "urgent_v0": len(urgent_v0),
        "urgent_both": len(set(urgent_T) & set(urgent_v0)),
        "urgent_T_only": len(set(urgent_T) - set(urgent_v0)), "urgent_v0_only": len(set(urgent_v0) - set(urgent_T)),
        "cases_missed_T": len(missed_T), "events_T": events_T,
        "cases_missed_v0": len(missed_v0), "events_v0": events_v0,
        "missed_cases_appear_under_T": {c: per_case[c]["n_miss"] for c in appear},
        "missed_cases_disappear_under_T": {c: per_case[c]["n_miss"] for c in disappear},
        "events_appear": sum(per_case[c]["n_miss"] for c in appear),
        "events_disappear": sum(per_case[c]["n_miss"] for c in disappear),
        "miss_histogram_T": dict(sorted(Counter(per_case[c]["n_miss"] for c in urgent_T).items())),
        "reviewed_cases": len(reviewed), "reviewed_events": sum(per_case[c]["n_miss"] for c in reviewed),
        "events_unreadable": n_events_unreadable,
        "events_model_names_severe_broad": n_events_named,
        "events_model_names_severe_excl_stable_angina": n_events_named_x,
        "events_model_lists_drive_dx": sum(1 for e in events if e["lists_drive_dx"]),
        "events_insufficient": sum(1 for e in events if e["sufficiency"] == "INSUFFICIENT"),
        "events_followup_test": sum(1 for e in events if e["followup_kind"] == "TEST"),
        "inconsistency_by_model": {r["short"]: r["routine_naming_severe_broad"] for r in incons},
        "review_tags": dict(Counter(v.get("tag") for v in CLINICAL_REVIEW.values())),
        "review_verdicts": dict(Counter(v.get("verdict") for v in CLINICAL_REVIEW.values())),
    }
    # taxonomy and separation rules are filled in after the review table exists
    summary.update(tax)
    (OUT / "missed_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
    for k in ("taxonomy", "rule_sides", "rule_vs_verdict_reviewed", "review_tags", "review_verdicts", "decode_artifact",
              "discriminating"):
        print(k, json.dumps(summary[k], indent=1))
    write_tables(per_case, info, events, incons, urgent_T, support_by_case, disc_by_case, summary, names)
    print("appear:", summary["missed_cases_appear_under_T"])
    print("disappear:", summary["missed_cases_disappear_under_T"])
    print("histogram:", summary["miss_histogram_T"])
    print("inconsistency:", summary["inconsistency_by_model"])


def event_category(e, pc, support):
    """One cause per miss event, in precedence order. Model-side causes come first because they hold
    whatever the label says; case-side causes come from the clinical review tags."""
    if e["unreadable"]:
        return "unreadable output"
    if e["names_severe_broad"]:
        return "named a time-critical diagnosis, still ROUTINE_CARE"
    rv = CLINICAL_REVIEW.get(e["case_id"])
    if rv is None:
        return "not reviewed (case missed by 1-2 models)"
    tag = rv["tag"]
    if tag.startswith("KB_QUIRK"):
        return "severe diagnosis clinically implausible (knowledge-base quirk)"
    if tag == "LOW_PRIOR":
        return "severe diagnosis plausible but low, no red flag recorded"
    if tag == "ATYPICAL":
        return "real emergency, atypical presentation"
    return tag


def rule_side(e, pc, support):
    """Computable separation: which side of the human-plausible / model-error line a miss event falls on."""
    if e["unreadable"]:
        return "model error: unreadable output"
    if e["names_severe_broad"]:
        return "model error: named time-critical dx, chose ROUTINE_CARE"
    if pc["pathology_sev"] <= SEVERE or (pc["diff"][0]["sev"] <= SEVERE):
        return "model error: true pathology severe or severe dx ranks 1st"
    if not any(support.values()):
        return "human-plausible: no severe dx has symptom support"
    return "debatable: some severe dx has symptom support"


def taxonomy(per_case, events, urgent_T, support_by_case):
    cat_cases, cat_events = defaultdict(set), Counter()
    side_cases, side_events = defaultdict(set), Counter()
    for e in events:
        pc = per_case[e["case_id"]]
        c = event_category(e, pc, support_by_case[e["case_id"]])
        cat_cases[c].add(e["case_id"])
        cat_events[c] += 1
        e["category"] = c
        s = rule_side(e, pc, support_by_case[e["case_id"]])
        side_cases[s].add(e["case_id"])
        side_events[s] += 1
        e["rule_side"] = s
    decode_cases = {c for c, v in CLINICAL_REVIEW.items() if "DECODE" in v["tag"]}
    # rule vs verdict, on reviewed cases (case level, using the case-side rule only)
    cross = Counter()
    for cid, rv in CLINICAL_REVIEW.items():
        pc = per_case[cid]
        sup = support_by_case[cid]
        if pc["pathology_sev"] <= SEVERE or pc["diff"][0]["sev"] <= SEVERE:
            side = "clear error"
        elif not any(sup.values()):
            side = "human-plausible"
        else:
            side = "debatable"
        cross[(side, rv["verdict"])] += 1
    unsupported_all = [c for c in urgent_T if not any(support_by_case[c].values())]
    return {
        "taxonomy": {k: {"cases": len(cat_cases[k]), "events": v} for k, v in cat_events.most_common()},
        "decode_artifact": {"cases": len(decode_cases),
                            "events": sum(per_case[c]["n_miss"] for c in decode_cases)},
        "rule_sides": {k: {"cases": len(side_cases[k]), "events": v} for k, v in side_events.most_common()},
        "rule_vs_verdict_reviewed": {f"{a} | {b}": n for (a, b), n in sorted(cross.items())},
        "unsupported_urgent_cases": len(unsupported_all),
        "unsupported_urgent_cases_never_missed": sum(1 for c in unsupported_all if per_case[c]["n_miss"] == 0),
        "unsupported_urgent_events": sum(per_case[c]["n_miss"] for c in unsupported_all),
    }


def write_tables(per_case, info, events, incons, urgent_T, support, disc, summary, names):
    """Markdown tables for docs/failure-modes/missed-escalation.md, so the doc's numbers come from one place."""
    L = []
    L.append("## per-case (urgent under T, missed by >= %d models)\n" % REVIEW_MIN_MISSES)
    L.append("| Case | Age/sex | True pathology (sev) | Top 3 of the differential | Severe diagnoses (rank, p) | P(severe) | v0 urgent | Missed (readable) | Tag | Verdict | Reason |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in sorted(urgent_T, key=lambda c: (-per_case[c]["n_miss"], c)):
        pc, i = per_case[cid], info[cid]
        if pc["n_miss"] < REVIEW_MIN_MISSES:
            continue
        rv = CLINICAL_REVIEW.get(cid, {})
        L.append("| %s | %s %s | %s (%d) | %s | %s | %.2f | %s | %d (%d) | %s | %s | %s |" % (
            cid.replace("ddxplus_", ""), i["age"], i["sex"][0].upper(), i["pathology"], i["pathology_sev"],
            "; ".join(f"{d['name']} {d['p']:.0%}" for d in i["diff"][:3]),
            "; ".join(f"{d['name']} r{d['rank']} {d['p']:.0%}" for d in i["severe_dx"] if d["p"] >= 0.05),
            i["p_sev_all"], "yes" if pc["urgent_v0"] else "no", pc["n_miss"], pc["n_miss_readable"],
            rv.get("tag", ""), rv.get("verdict", ""), rv.get("reason", "")))
    L.append("\n## taxonomy\n")
    L.append("| Cause | Cases | Events | Share of events |")
    L.append("|---|---|---|---|")
    tot = summary["events_T"]
    for k, v in summary["taxonomy"].items():
        L.append(f"| {k} | {v['cases']} | {v['events']} | {v['events'] / tot:.0%} |")
    L.append("\n## inconsistency per model (all 250 cases)\n")
    L.append("| Model | Status | ROUTINE_CARE calls | ... naming a time-critical code | ... at rank 1 | On urgent (T) | On non-urgent | Unreadable | Cases |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(incons, key=lambda r: (-r["routine_naming_severe_broad"], r["short"])):
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d | %s |" % (
            r["short"], r["status"], r["routine_calls"], r["routine_naming_severe_broad"], r["routine_severe_at_rank1"],
            r["on_urgent_T"], r["on_nonurgent_T"], r["unreadable"], ", ".join(c.replace("ddxplus_", "") for c in r["cases"])))
    for label, key in (("any symptom support", "rule_sides"), ("discriminating symptom support", None)):
        rs = summary["rule_sides"] if key else summary["discriminating"]["rule_sides"]
        L.append(f"\n## separation rule, {label}\n")
        L.append("| Side | Cases | Events |")
        L.append("|---|---|---|")
        for k, v in rs.items():
            L.append(f"| {k} | {v['cases']} | {v['events']} |")
        cv = summary["rule_vs_verdict_reviewed"] if key else summary["discriminating"]["rule_vs_verdict_reviewed"]
        L.append("\nRule side vs review verdict, 32 reviewed cases: " + "; ".join(f"{k}: {v}" for k, v in cv.items()))
    L.append("\n## label comparison\n")
    L.append("| Case | Missed | P(severe) | Severe diagnoses | True pathology |")
    L.append("|---|---|---|---|---|")
    for cid in sorted(summary["missed_cases_appear_under_T"], key=lambda c: -per_case[c]["n_miss"]):
        i = info[cid]
        L.append("| %s | %d | %.2f | %s | %s |" % (cid.replace("ddxplus_", ""), per_case[cid]["n_miss"], i["p_sev_all"],
                 "; ".join(f"{d['name']} r{d['rank']} {d['p']:.0%}" for d in i["severe_dx"][:3]), i["pathology"]))
    (OUT / "missed_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
