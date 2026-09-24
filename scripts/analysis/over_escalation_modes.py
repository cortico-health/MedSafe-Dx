"""Why models over-escalate on eval-250-v0, and when a clinician would do the same.

Why: under the T = 0.15 label (evaluator/triage_score.py) 62 cases are not
urgent, and models escalate 40-60% of them. The benchmark author suspects that
part of this is a task a human would also fail given only this intake. This
script separates the causes so the board can tell model over-caution from
"a careful clinician would escalate this too".

What it does (no inference, no API calls):
1. Reads the 250 cases, the DDXPlus source rows, the condition and evidence
   metadata, and the stored predictions of every leaderboard row that has
   per-case data (19 rows, resolved the same way as failure_shape.py).
2. For each case computes P(severe) and the T = 0.15 label, decodes the
   presentation exactly as the prompt showed it to the model, and extracts
   computable features: red-flag evidence codes, severe diagnoses outside
   the DDXPlus top 3, the P(severe) band, and which time-critical ICD-10
   codes the models ranked.
3. Applies the hand-assigned cause category and clinical-reasonableness
   verdict per case (CASE_LABELS below; model-generated clinical judgement,
   for later clinician review) and counts cases and escalation events per
   category.
4. Evaluates candidate separation rules (computable from the intake and the
   DDXPlus differential only) against the verdicts.

Outputs, under results/analysis/failure_modes/:
    over_escalation_cases.csv              one row per case with P(severe) < 0.15
    over_escalation_events.csv             one row per (case, model) escalation on those cases
    over_escalation_dossiers.md            decoded presentation, differential and model answers per case
    over_escalation_followup_keywords.csv  what the follow-up text asks for, over all events
    over_escalation_summary.json           taxonomy counts, separation rules, relabel impact, reverse check
    over_escalation_urgent_reasons.csv     what escalators named on urgent cases (the reverse check)

Usage:
    .venv/bin/python scripts/analysis/over_escalation_modes.py
"""

from __future__ import annotations

import ast
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import triage_score as ts  # noqa: E402
from evaluator.schemas import ModelPrediction  # noqa: E402
from inference.symptom_decoder import decode_symptoms_with_audit  # noqa: E402

CASES_PATH = ROOT / "data/test_sets/eval-250-v0.json"
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
EVID_PATH = ROOT / "data/ddxplus_v0/release_evidences.json"
OUT_DIR = ROOT / "results/analysis/failure_modes"
T = ts.URGENT_THRESHOLD

FALLBACK_PREDICTIONS = {
    "anthropic-claude-haiku-4.5": "leaderboard/anthropic-claude-haiku-4.5-250cases.json",
    "anthropic-claude-sonnet-4.6": "results/artifacts/anthropic-claude-sonnet-4.6-500cases.json",
}

SHORT = {
    "anthropic-claude-fable-5": "Fable 5",
    "anthropic-claude-haiku-4.5": "Haiku 4.5",
    "anthropic-claude-opus-5": "Opus 5",
    "anthropic-claude-sonnet-4.6": "Sonnet 4.6",
    "deepseek-deepseek-r1": "DeepSeek R1",
    "google-gemini-3-pro-preview": "Gemini 3 Pro",
    "google-gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "moonshotai-kimi-k3": "Kimi K3",
    "openai-gpt-5-chat": "GPT-5 Chat",
    "openai-gpt-5-mini": "GPT-5 Mini",
    "openai-gpt-5.2": "GPT-5.2",
    "openai-gpt-5.4-mini": "GPT-5.4 Mini",
    "openai-gpt-5.6-luna": "GPT-5.6 Luna",
    "openai-gpt-5.6-sol": "GPT-5.6 Sol",
    "openai-gpt-5.6-terra": "GPT-5.6 Terra",
    "openai-gpt-6-astra": "GPT-6 Astra",
    "openai-gpt-oss-120b": "GPT-OSS 120B",
    "x-ai-grok-4.6": "Grok 4.6",
    "z-ai-glm-5.3": "GLM 5.3",
}

# Evidence codes a triage clinician reads as red flags. Labels are what the
# prompt showed the model (decoded question text), shortened.
RED_FLAG_CODES = {
    "E_14": "chest pain at rest",
    "E_66": "significant shortness of breath",
    "E_64": "breathless with minimal effort",
    "E_67": "nocturnal choking / dyspnea",
    "E_128": "brief suffocation episodes",
    "E_75": "choking / suffocating",
    "E_194": "stridor",
    "E_159": "loss of consciousness",
    "E_43": "seizure / absence",
    "E_82": "presyncope",
    "E_155": "palpitations",
    "E_164": "very irregular heartbeat",
    "E_45": "hemoptysis",
    "E_210": "hematemesis",
    "E_140": "melena",
    "E_179": "blood in stool",
    "E_178": "unusual bleeding or bruising",
    "E_162": "involuntary weight loss (3 months)",
    "E_174": "unintentional weight loss / anorexia",
    "E_91": "fever",
    "E_94": "chills",
    "E_39": "confusion",
    "E_63": "dysarthria",
    "E_156": "one-sided facial weakness",
    "E_176": "limb weakness / paralysis",
    "E_84": "weakness in both arms or legs",
    "E_88": "too tired to do usual activities / stuck in bed",
    "E_154": "much paler than usual",
    "E_65": "dysphagia",
    "E_220": "pleuritic pain",
    "E_218": "symptoms worse with exertion, relieved by rest",
    "E_217": "worse lying down, better sitting up",
}
CHEST_LOCATIONS = {"V_29": "lower chest", "V_101": "upper chest", "V_55": "side of chest (R)", "V_56": "side of chest (L)",
                   "V_197": "epigastric", "V_170": "posterior chest wall (R)", "V_171": "posterior chest wall (L)"}
HEAD_LOCATIONS = {"V_89", "V_125", "V_126", "V_166", "V_167", "V_108", "V_109", "V_25", "V_62", "V_124"}
NECK_LOCATIONS = {"V_26", "V_53", "V_54", "V_38"}
ARM_JAW_LOCATIONS = {"V_30", "V_31", "V_194", "V_195", "V_121", "V_127", "V_128", "V_175", "V_176", "V_177", "V_178"}

# Comorbidities that raise a clinician's index of suspicion on their own.
RISK_ANTECEDENTS = {
    "E_105": "prior MI or angina", "E_106": "heart failure", "E_146": "new oral anticoagulant", "E_113": "chronic kidney failure",
    "E_8": "dialysis", "E_2": "HIV positive", "E_227": "immunosuppressed", "E_44": "corticosteroids", "E_61": "IV drug use",
    "E_34": "active cancer", "E_37": "metastatic cancer", "E_31": "severe COPD", "E_123": "COPD", "E_69": "diabetes",
    "E_104": "hypertension", "E_71": "high cholesterol", "E_79": "smoker", "E_109": "prior DVT", "E_110": "immobile >3 days",
    "E_167": "pregnant", "E_12": "severe food allergy", "E_139": "heart defect", "E_22": "valve disease", "E_24": "prior anemia",
    "E_126": "cirrhosis", "E_18": "cystic fibrosis", "E_101": "asthma admission past year", "E_46": "2+ asthma attacks past year",
}

# ICD-10 prefixes the models use for time-critical diagnoses. Used to see what
# the model was escalating FOR, independent of the DDXPlus differential.
TIME_CRITICAL_ICD = {
    "I21": "MI", "I20": "angina", "I24": "ACS", "I26": "PE", "J93": "pneumothorax", "T78.0": "anaphylaxis", "T78.2": "anaphylaxis",
    "T88.6": "anaphylaxis", "I47": "PSVT/tachyarrhythmia", "I48": "AF/flutter", "I49": "arrhythmia", "J81": "pulmonary edema",
    "I50": "heart failure", "K22.3": "Boerhaave", "G61.0": "GBS", "I40": "myocarditis", "I51.4": "myocarditis", "J05": "croup/epiglottitis",
    "G24": "dystonia", "T61.1": "scombroid", "K92": "GI bleed", "A41": "sepsis", "R65": "sepsis", "I63": "stroke", "G45": "TIA",
    "J96": "respiratory failure", "I71": "aortic dissection", "K35": "appendicitis", "J44.1": "COPD exacerbation", "J45.9": "asthma",
    "J46": "status asthmaticus", "J18": "pneumonia", "J15": "pneumonia", "A15": "TB", "A16": "TB", "R04.2": "hemoptysis",
    "D62": "acute blood-loss anemia", "J38.5": "laryngospasm", "I31": "pericardial", "I30": "pericarditis", "R57": "shock",
    "T50": "poisoning", "E10.1": "DKA", "E11.1": "DKA", "R40": "coma", "G40": "seizure", "R56": "seizure",
    "I60": "SAH", "I61": "intracranial bleed", "I62": "intracranial bleed", "G03": "meningitis", "G00": "meningitis", "A39": "meningococcal",
    "A87": "meningitis", "G04": "encephalitis", "M31.5": "GCA", "M31.6": "GCA", "I33": "endocarditis", "B54": "malaria", "B50": "malaria",
    "H40.2": "angle-closure glaucoma", "H70": "mastoiditis", "N44": "testicular torsion", "K40.3": "incarcerated hernia", "K40.0": "incarcerated hernia",
    "K40.4": "incarcerated hernia", "K40.1": "incarcerated hernia", "K46.0": "incarcerated hernia",
}

FOLLOWUP_ASKS = {
    "vitals": r"\bvital|blood pressure|\bBP\b|heart rate|pulse|SpO2|oxygen sat|saturation|temperature|respiratory rate",
    "ecg": r"\bECG\b|\bEKG\b|electrocardiogram|12-lead",
    "troponin": r"troponin",
    "cxr_imaging": r"chest x-ray|chest X-ray|\bCXR\b|chest radiograph|\bCT\b|imaging|ultrasound|echocardiogram",
    "labs": r"\bCBC\b|hemoglobin|haemoglobin|blood count|\bBNP\b|d-dimer|D-dimer|lactate|\bINR\b|glucose|blood test|labs?\b",
    "rule_out": r"rule out|exclude|cannot rule|can't rule|to exclude|r/o\b",
    "airway": r"airway|stridor|swelling of (the )?(tongue|lips|throat)|angioedema",
    "onset_timing": r"onset|when did|how long|duration|timing|sudden",
    "exam": r"exam|auscultat|palpat|inspect",
}

# ---------------------------------------------------------------- hand labels
# Cause category and clinical-reasonableness verdict per non-urgent case that
# at least half the models escalate. These are model-generated clinical
# judgements (Claude, 2026-09), written after reading dossiers.md, for later
# clinician review. They are not clinician labels.
#
# Categories (a case may match several features; the primary cause is the one
# named first in the follow-up text of most models):
#   RF-CARDIOPULM   red-flag symptom that reads as ACS / PE / airway on paper
#   SEVERE-LOW      a DDXPlus severity 1-2 diagnosis is in the differential below the top 3 (or at low mass)
#   COMORBID-RISK   age, comorbidity or medication turns a benign complaint into a risk
#   OFF-LIST-SEVERE the model escalates for a severe diagnosis DDXPlus never generates (GI bleed, sepsis, stroke)
#   LABEL-SEV3      the true pathology is DDXPlus severity 3 (TB, COPD exacerbation, lung cancer, pneumonia) and reads urgent
#   ENCODING        an encoding or decoding artifact (pain scale, "stuck in bed", contradictory sex/symptom)
#   OVERCAUTION     no red flag, no risk factor, no severe mass: plain over-caution
# Verdicts: DEFENSIBLE, DEBATABLE, UNNECESSARY.
CASE_LABELS: dict[str, tuple[str, str, str]] = {
    # OFF-LIST-BLEED: GI bleeding symptoms that DDXPlus generates for anemia or GERD; no bleeding diagnosis exists in its 49 conditions.
    "ddxplus_118799": ("OFF-LIST-BLEED", "DEFENSIBLE", "Melena plus hematochezia with presyncope on a NOAC, CKD, prior anemia: an upper GI bleed until proven otherwise."),
    "ddxplus_112823": ("OFF-LIST-BLEED", "DEFENSIBLE", "Melena, presyncope and prostration on a NOAC with CKD at 61: same-hour hemoglobin and vitals."),
    "ddxplus_129862": ("OFF-LIST-BLEED", "DEFENSIBLE", "Melena plus hematochezia with presyncope on a NOAC, underweight: active bleed likely."),
    "ddxplus_131101": ("OFF-LIST-BLEED", "DEFENSIBLE", "Melena, pallor, menorrhagia and prostration on a NOAC with CKD: symptomatic anemia with a bleeding source."),
    "ddxplus_1515": ("OFF-LIST-BLEED", "DEFENSIBLE", "Coffee-ground hematemesis plus melena in a heavy drinker: upper GI bleed, needs vitals and hemoglobin now."),
    "ddxplus_26719": ("OFF-LIST-BLEED", "DEFENSIBLE", "Hematemesis with chest and epigastric pain in a 59-year-old smoker who drinks: bleed and ACS both need excluding."),
    # RF-HEMOPTYSIS: hemoptysis with fever in an immunosuppressed injecting drug user; TB is severity 3 in DDXPlus.
    "ddxplus_104758": ("RF-HEMOPTYSIS", "DEFENSIBLE", "Hemoptysis, fever, weight loss on steroids with diabetes and IV drug use: isolate and assess now (TB, septic emboli)."),
    "ddxplus_24966": ("RF-HEMOPTYSIS", "DEFENSIBLE", "Hemoptysis and fever in an HIV-positive, steroid-treated, underweight IV drug user: isolate and assess now."),
    # OFF-LIST-SAH: cluster headache encoded as sudden-onset (E_59 >= 7) severe head pain; DDXPlus has no subarachnoid hemorrhage.
    "ddxplus_108509": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 8/10 knife-like violent headache with no stated prior episodes: thunderclap pattern, CT head within hours."),
    "ddxplus_34166": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 10/10 violent headache at 58 on a vasodilator: thunderclap pattern, CT head within hours."),
    "ddxplus_106124": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 8/10 knife-like periorbital headache with lacrimation; cluster features are strong, but first-episode status is unknown."),
    "ddxplus_109101": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 9/10 knife-like headache; cluster family history and nasal congestion argue for cluster, but a 9/10 sudden headache gets imaged."),
    "ddxplus_20181": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 9/10 headache at 20 with lacrimation and congestion: typical cluster, but sudden 9/10 with unknown history is imaged."),
    "ddxplus_36986": ("OFF-LIST-SAH", "DEFENSIBLE", "Sudden 9/10 knife-like headache with lacrimation on a vasodilator: thunderclap pattern by the intake alone."),
    "ddxplus_104547": ("OFF-LIST-SAH", "DEBATABLE", "Sudden 7/10 periorbital headache with family history of cluster headache: cluster pattern; one question (first episode?) would settle it."),
    "ddxplus_120434": ("OFF-LIST-SAH", "DEBATABLE", "Sudden 7/10 unilateral periorbital pain with congestion and family history: cluster pattern; glaucoma is the only same-hour concern."),
    "ddxplus_31967": ("OFF-LIST-SAH", "DEBATABLE", "Sudden 7/10 periorbital pain with lacrimation and family history: cluster pattern; same-day rather than same-minute."),
    # OFF-LIST-MENINGITIS: influenza encoded with fever, headache, neck pain and a rash; DDXPlus has no meningitis.
    "ddxplus_15697": ("OFF-LIST-MENINGITIS", "DEFENSIBLE", "Fever, 8/10 headache, neck pain, rash and prostration in an immunosuppressed smoker: meningitis screen cannot wait."),
    "ddxplus_11773": ("OFF-LIST-MENINGITIS", "DEFENSIBLE", "Fever, chills, headache, neck pain, rash and prostration while immunosuppressed: febrile immunosuppressed pathway."),
    "ddxplus_36629": ("OFF-LIST-MENINGITIS", "DEBATABLE", "Fever, chills, 6/10 headache, neck pain, forehead rash at 58, not immunosuppressed: flu picture, but a nurse would check for meningism first."),
    "ddxplus_19428": ("OFF-LIST-MENINGITIS", "DEBATABLE", "Fever, 7/10 headache, neck pain and a neck rash at 22: flu picture; a glass test and neck-stiffness check decide it."),
    "ddxplus_31957": ("OFF-LIST-MENINGITIS", "DEBATABLE", "Chills, headache, neck pain, itchy neck rash, no fever recorded: most read this as a viral illness; meningism check is quick."),
    # COMORBID-RISK: fever with IV drug use, HIV exposure, weight loss or prostration; HIV seroconversion is severity 3.
    "ddxplus_101125": ("COMORBID-RISK", "DEBATABLE", "Fever, lymphadenopathy, weight loss, prostration and mucosal ulcers in an IV drug user: same-day blood cultures, not same-minute."),
    "ddxplus_121707": ("COMORBID-RISK", "DEBATABLE", "Fever, diarrhea, prostration, oral and genital ulcers in an IV drug user: endocarditis is the reason models give; same-day workup."),
    "ddxplus_107066": ("COMORBID-RISK", "DEBATABLE", "Fever, weight loss, prostration, mucosal ulcers, IV drug use and HIV exposure: seroconversion; same-day, not urgent."),
    "ddxplus_114855": ("COMORBID-RISK", "DEBATABLE", "Fever, weight loss, painful mucosal lesions, IV drug use at 58: same-day assessment is enough."),
    "ddxplus_110219": ("COMORBID-RISK", "UNNECESSARY", "Fever, weight loss, prostration and mucosal ulcers at 64 after HIV exposure: seroconversion illness; routine same-day."),
    "ddxplus_119501": ("COMORBID-RISK", "UNNECESSARY", "Fever, night sweats, weight loss, mucosal ulcers at 36 after HIV exposure: seroconversion; routine same-day."),
    # AGE-RISK: benign respiratory illness where age alone drives the escalation.
    "ddxplus_25951": ("AGE-RISK", "DEBATABLE", "Fever, myalgia and temporal headache at 93: a sepsis screen and GCA check are reasonable, but nothing here says minutes."),
    "ddxplus_16646": ("AGE-RISK", "DEBATABLE", "New 7/10 temporal and occipital headache with myalgia at 81: GCA needs same-day ESR; escalation is cautious, not wrong."),
    # EXAM-NEEDED: a complaint that is routine unless the examination says otherwise.
    "ddxplus_12868": ("EXAM-NEEDED", "DEBATABLE", "Gradual 6/10 groin and left testicular pain worse on straining at 51: hernia; torsion is unlikely with gradual onset but testicular pain gets examined promptly."),
    # OVERCAUTION: no red flag, no risk factor, no severe mass.
    "ddxplus_115087": ("OVERCAUTION", "UNNECESSARY", "Bilateral 4/10 ear pain with fever at 72: otitis media; routine."),
    "ddxplus_104179": ("OVERCAUTION", "UNNECESSARY", "Sudden 9/10 bilateral ear pain with coryza at 73: otitis media; routine."),
    "ddxplus_32315": ("OVERCAUTION", "UNNECESSARY", "9/10 bilateral ear pain at 26: otitis media; routine."),
    "ddxplus_133786": ("OVERCAUTION", "UNNECESSARY", "Fever, 7/10 headache, sore throat, myalgia, cough at 19: influenza-like illness without neck pain or rash; routine."),
    "ddxplus_120600": ("OVERCAUTION", "UNNECESSARY", "Fever, 7/10 headache, sore throat, myalgia at 56: influenza-like illness; routine."),
    "ddxplus_103580": ("OVERCAUTION", "UNNECESSARY", "Fever, mild headache, sore throat, cough at 71 smoker: URTI; routine (vitals at intake would be enough)."),
    "ddxplus_115418": ("OVERCAUTION", "UNNECESSARY", "Fever, 7/10 frontal headache, myalgia at 33: URTI; routine."),
    "ddxplus_10945": ("OVERCAUTION", "UNNECESSARY", "Fever, productive cough, sore throat at 55: bronchitis or pneumonia; routine with a chest exam."),
    "ddxplus_1167": ("OVERCAUTION", "UNNECESSARY", "Fever, sore throat, congestion at 62: URTI; the one escalation asked about the 'N' travel artifact."),
    "ddxplus_32805": ("OVERCAUTION", "UNNECESSARY", "Sore throat with fever at 18, no dysphagia or stridor: pharyngitis; routine."),
    "ddxplus_109070": ("OVERCAUTION", "UNNECESSARY", "Finger pain, lymph nodes and red eyes at 37: sarcoidosis; the one escalation chased the 'N' travel artifact."),
    "ddxplus_122212": ("OVERCAUTION", "UNNECESSARY", "Finger pain and lymph nodes at 20: sarcoidosis; the one escalation chased the 'N' travel artifact."),
    "ddxplus_11469": ("OVERCAUTION", "UNNECESSARY", "Coughing fits with post-tussive vomiting after pertussis contact: whooping cough; routine."),
    "ddxplus_122296": ("OVERCAUTION", "UNNECESSARY", "Facial pain, anosmia and purulent discharge at 18: sinusitis; routine."),
    "ddxplus_36111": ("OVERCAUTION", "UNNECESSARY", "Facial pain, anosmia and purulent discharge at 60: sinusitis; routine."),
}


# ---------------------------------------------------------------- loading


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    """Leaderboard rows with per-case predictions, resolved like failure_shape.py."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "leaderboard/*-eval.json"))):
        ev = json.loads(Path(f).read_text())
        p = ROOT / ev["predictions_path"]
        note = "exact file"
        if not p.exists():
            fb = FALLBACK_PREDICTIONS.get(ev["model"])
            if not fb:
                continue
            p = ROOT / fb
            note = "fallback file"
        elif sha256(p) != ev["predictions_sha256"]:
            note = "file changed after scoring"
        raw = json.loads(p.read_text())
        preds = raw["predictions"] if isinstance(raw, dict) else raw
        by_id = {}
        for x in preds:
            if isinstance(x, dict) and x.get("case_id") and x["case_id"] not in by_id:
                by_id[x["case_id"]] = x
        rows.append({"model": ev["model"], "name": SHORT.get(ev["model"], ev["model"]), "path": str(p.relative_to(ROOT)),
                     "note": note, "preds": by_id})
    return rows


def parse_pred(rp):
    """(escalate, confident, sufficiency, followup_kind, followup, codes) or None if unreadable."""
    if rp is None:
        return None
    try:
        pred = ModelPrediction(**rp)
    except Exception:
        return None
    return {
        "escalate": int(pred.escalation_decision == "ESCALATE_NOW"),
        "confident": int(pred.uncertainty == "CONFIDENT"),
        "sufficiency": pred.information_sufficiency or "",
        "followup_kind": pred.followup_kind or "",
        "followup": (pred.followup_recommendation or "").replace("\n", " ").strip(),
        "codes": [d.code for d in pred.differential_diagnoses],
        "raw": rp.get("raw_response") or "",
    }


def load_ddx(case_ids):
    want = {int(c.split("_")[1]): c for c in case_ids}
    out = {}
    with open(DDX_CSV) as f:
        for i, row in enumerate(csv.DictReader(f)):
            cid = want.get(i)
            if cid is None:
                continue
            out[cid] = {
                "differential": [(n, float(p)) for n, p in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])],
                "pathology": row["PATHOLOGY"],
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "initial": row["INITIAL_EVIDENCE"],
            }
    return out


# ---------------------------------------------------------------- features


def decode_presentation(case):
    active, ante, _ = decode_symptoms_with_audit(case["presenting_symptoms"])
    return active, ante


def case_features(case, ddx, cond, evid):
    codes = case["presenting_symptoms"]
    bases = {c.split("_@_")[0] for c in codes}
    values = {c.split("_@_")[1] for c in codes if "_@_" in c}
    red = [RED_FLAG_CODES[b] for b in RED_FLAG_CODES if b in bases]
    chest = [CHEST_LOCATIONS[v] for c in codes if c.startswith("E_55_@_") for v in [c.split("_@_")[1]] if v in CHEST_LOCATIONS]
    radiates = [evid["E_57"]["value_meaning"][c.split("_@_")[1]]["en"] for c in codes if c.startswith("E_57_@_")]
    arm_jaw = any(c.split("_@_")[1] in ARM_JAW_LOCATIONS for c in codes if c.startswith(("E_55_@_", "E_57_@_")))
    risk = [RISK_ANTECEDENTS[b] for b in RISK_ANTECEDENTS if b in bases]
    intensity = next((int(c.split("_@_")[1]) for c in codes if c.startswith("E_56_@_")), None)
    onset = next((int(c.split("_@_")[1]) for c in codes if c.startswith("E_59_@_")), None)
    diff = ddx["differential"]
    sev = {n: cond[n]["severity"] for n, _ in diff}
    p_sev = sum(p for n, p in diff if sev[n] <= ts.SEVERE_MAX)
    severe_entries = [(i + 1, n, p) for i, (n, p) in enumerate(diff) if sev[n] <= ts.SEVERE_MAX]
    top_severe = severe_entries[0] if severe_entries else None
    head = [c for c in codes if c.startswith("E_55_@_") and c.split("_@_")[1] in HEAD_LOCATIONS]
    neck = [c for c in codes if c.startswith("E_55_@_") and c.split("_@_")[1] in NECK_LOCATIONS]
    return {
        "bases": bases, "head_locations": head, "neck_locations": neck,
        "red_flags": red, "chest_pain": chest, "radiates": radiates, "arm_jaw": arm_jaw, "risk": risk,
        "pain_intensity": intensity, "pain_onset": onset, "p_severe": p_sev, "severe_entries": severe_entries,
        "top_severe": top_severe, "pathology_sev": cond[ddx["pathology"]]["severity"], "pathology_name": ddx["pathology"], "n_codes": len(codes),
    }


def time_critical(codes):
    hits = []
    for c in codes:
        cu = c.upper()
        for pre, lab in TIME_CRITICAL_ICD.items():
            if cu.startswith(pre.upper()):
                hits.append((c, lab))
                break
    return hits


def followup_asks(text):
    t = text or ""
    return [k for k, rx in FOLLOWUP_ASKS.items() if re.search(rx, t, flags=re.I)]


# ---------------------------------------------------------------- main


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = json.loads(CASES_PATH.read_text())["cases"]
    cond = json.loads(COND_PATH.read_text())
    evid = json.loads(EVID_PATH.read_text())
    ddx = load_ddx([c["case_id"] for c in cases])
    rows = load_rows()
    print(f"{len(rows)} rows with per-case predictions")

    # Per-case, per-model parsed predictions.
    parsed = {r["model"]: {c["case_id"]: parse_pred(r["preds"].get(c["case_id"])) for c in cases} for r in rows}

    # Cross-check escalation decisions against the failure-shape outcomes file.
    fs = ROOT / "results/analysis/failure_shape/case_model_outcomes.csv"
    if fs.exists():
        mism = 0
        for r in csv.DictReader(open(fs)):
            p = parsed.get(r["model"], {}).get(r["case_id"])
            if r["escalate"] == "" or p is None:
                continue
            if int(r["escalate"]) != p["escalate"]:
                mism += 1
        print(f"escalation decisions disagreeing with failure_shape outcomes: {mism}")

    feats = {c["case_id"]: case_features(c, ddx[c["case_id"]], cond, evid) for c in cases}
    non_urgent = [c for c in cases if feats[c["case_id"]]["p_severe"] < T]
    urgent = [c for c in cases if feats[c["case_id"]]["p_severe"] >= T]
    print(f"urgent {len(urgent)}, non-urgent {len(non_urgent)} at T={T}")

    # ---- per-case table for non-urgent cases
    per_case = []
    events = []
    for c in non_urgent:
        cid = c["case_id"]
        f = feats[cid]
        d = ddx[cid]
        preds = {m: parsed[m][cid] for m in parsed}
        valid = {m: p for m, p in preds.items() if p}
        esc = {m: p for m, p in valid.items() if p["escalate"]}
        top1 = Counter(p["codes"][0] for p in valid.values())
        tc_top3 = Counter(lab for p in esc.values() for _, lab in time_critical(p["codes"][:3]))
        tc_top1 = Counter(lab for p in esc.values() for _, lab in time_critical(p["codes"][:1]))
        asks = Counter(k for p in esc.values() for k in followup_asks(p["followup"]))
        insuff = sum(1 for p in esc.values() if p["sufficiency"] == "INSUFFICIENT")
        conf = sum(1 for p in esc.values() if p["confident"])
        lab = CASE_LABELS.get(cid, ("", "", ""))
        per_case.append({
            "case_id": cid, "age": c["age"], "sex": c["sex"], "pathology": d["pathology"], "pathology_sev": f["pathology_sev"],
            "p_severe": round(f["p_severe"], 4), "v0_escalation_required": int(c["escalation_required"]),
            "top3": "; ".join(f"{n} ({cond[n]['severity']}, {p:.0%})" for n, p in d["differential"][:3]),
            "top_severe": f"{f['top_severe'][1]} r{f['top_severe'][0]} {f['top_severe'][2]:.1%}" if f["top_severe"] else "",
            "red_flags": "; ".join(f["red_flags"]), "chest_pain": "; ".join(f["chest_pain"]), "radiates": "; ".join(f["radiates"]),
            "arm_jaw": int(f["arm_jaw"]), "risk_factors": "; ".join(f["risk"]), "pain_intensity": f["pain_intensity"] if f["pain_intensity"] is not None else "",
            "models_valid": len(valid), "models_escalate": len(esc), "share_escalate": round(len(esc) / len(valid), 3) if valid else "",
            "escalate_confident": conf, "escalate_insufficient": insuff,
            "most_common_top1": ", ".join(f"{k} x{v}" for k, v in top1.most_common(3)),
            "time_critical_top1": ", ".join(f"{k} x{v}" for k, v in tc_top1.most_common(3)),
            "time_critical_top3": ", ".join(f"{k} x{v}" for k, v in tc_top3.most_common(4)),
            "followup_asks": ", ".join(f"{k} x{v}" for k, v in asks.most_common(4)),
            "category": lab[0], "verdict": lab[1], "reason": lab[2], "tier": "",
        })
        for m, p in esc.items():
            events.append({"case_id": cid, "model": SHORT.get(m, m), "codes": " ".join(p["codes"]), "confident": p["confident"],
                           "sufficiency": p["sufficiency"], "followup_kind": p["followup_kind"],
                           "time_critical_top3": "; ".join(l for _, l in time_critical(p["codes"][:3])),
                           "asks": "; ".join(followup_asks(p["followup"])), "followup": p["followup"][:600]})
    per_case.sort(key=lambda r: (-(r["share_escalate"] or 0), -r["p_severe"]))

    with open(OUT_DIR / "over_escalation_events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)

    # ---- dossiers for reading
    with open(OUT_DIR / "over_escalation_dossiers.md", "w") as fh:
        fh.write(f"# Non-urgent cases (P(severe) < {T}) and what the models said\n\n")
        fh.write("Presentation text is what the prompt showed the model (decoded by inference/symptom_decoder.py). ")
        fh.write("Codes the decoder could not name are shown raw in brackets.\n\n")
        for r in per_case:
            cid = r["case_id"]
            c = next(x for x in cases if x["case_id"] == cid)
            d = ddx[cid]
            f = feats[cid]
            active, ante = decode_presentation(c)
            fh.write(f"## {cid}: {c['age']} {c['sex']}, pathology {d['pathology']} (sev {f['pathology_sev']}), "
                     f"P(severe) {f['p_severe']:.3f}, escalated {r['models_escalate']}/{r['models_valid']}\n\n")
            fh.write(f"- Chief complaints: {', '.join(active) or 'none'}\n")
            fh.write(f"- History: {', '.join(ante) or 'none'}\n")
            fh.write(f"- Raw codes: {' '.join(c['presenting_symptoms'])}\n")
            fh.write("- Differential: " + "; ".join(f"{n} (sev {cond[n]['severity']}, {p:.1%})" for n, p in d["differential"]) + "\n")
            fh.write(f"- Red flags: {r['red_flags'] or '-'} | chest: {r['chest_pain'] or '-'} | radiates: {r['radiates'] or '-'} | risk: {r['risk_factors'] or '-'}\n")
            fh.write(f"- Top-1 codes: {r['most_common_top1']} | time-critical in top 3 among escalators: {r['time_critical_top3'] or '-'}\n")
            fh.write(f"- Follow-up asks: {r['followup_asks'] or '-'} | escalators CONFIDENT {r['escalate_confident']}, INSUFFICIENT {r['escalate_insufficient']}\n\n")
            for m in parsed:
                p = parsed[m][cid]
                if not p:
                    fh.write(f"  - {SHORT.get(m, m)}: unreadable\n")
                    continue
                tag = "ESC" if p["escalate"] else "routine"
                fh.write(f"  - {SHORT.get(m, m)} [{tag}, {'CONF' if p['confident'] else 'unc'}, {p['sufficiency'] or '-'}]: "
                         f"{' '.join(p['codes'])} | {p['followup'][:300]}\n")
            fh.write("\n")

    # ---- reverse check: urgent cases, what did escalators escalate for?
    urg_rows = []
    for c in urgent:
        cid = c["case_id"]
        f = feats[cid]
        d = ddx[cid]
        valid = {m: p for m, p in ((m, parsed[m][cid]) for m in parsed) if p}
        esc = {m: p for m, p in valid.items() if p["escalate"]}
        tc = Counter(lab for p in esc.values() for _, lab in time_critical(p["codes"][:3]))
        driving = f["top_severe"]
        # Does the model's top-3 name the DDXPlus severe diagnosis family?
        drive_lab = None
        if driving:
            icd = cond[driving[1]]["icd10-id"].upper()
            for pre, lab in TIME_CRITICAL_ICD.items():
                if icd.startswith(pre.upper()):
                    drive_lab = lab
                    break
        names_driving = sum(1 for p in esc.values() if drive_lab and any(l == drive_lab for _, l in time_critical(p["codes"][:3])))
        names_any_tc = sum(1 for p in esc.values() if time_critical(p["codes"][:3]))
        urg_rows.append({"case_id": cid, "age": c["age"], "sex": c["sex"], "pathology": d["pathology"], "p_severe": round(f["p_severe"], 3),
                         "driving_dx": driving[1] if driving else "", "driving_rank": driving[0] if driving else "",
                         "driving_p": round(driving[2], 3) if driving else "", "red_flags": "; ".join(f["red_flags"]),
                         "models_valid": len(valid), "models_escalate": len(esc), "escalators_naming_driving_dx_top3": names_driving,
                         "escalators_naming_any_time_critical_top3": names_any_tc,
                         "time_critical_top3": ", ".join(f"{k} x{v}" for k, v in tc.most_common(4))})
    with open(OUT_DIR / "over_escalation_urgent_reasons.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(urg_rows[0].keys()))
        w.writeheader()
        w.writerows(urg_rows)

    # ---- follow-up keyword table over all escalation events on non-urgent cases
    ask_tot = Counter(k for e in events for k in e["asks"].split("; ") if k)
    with open(OUT_DIR / "over_escalation_followup_keywords.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ask", "escalation_events", "share_of_events"])
        for k, v in ask_tot.most_common():
            w.writerow([k, v, round(v / len(events), 3)])

    summary = summarize(per_case, events, feats, non_urgent, urg_rows, rows)
    summary["within_pathology"] = within_pathology(cases, feats, parsed)
    summary["escalated_for"] = escalated_for(events)
    summary["relabel_impact"] = relabel_impact(cases, feats, parsed, per_case)
    summary["travel_artifact"] = {
        "cases_shown_as_recent_travel_to_N": sum(1 for c in cases if "E_204_@_V_10" in c["presenting_symptoms"]),
        "over_escalation_followups_mentioning_travel": sum(1 for e in events if re.search(r"travel", e["followup"], re.I)),
        "over_escalation_followups_asking_to_clarify_N": sum(1 for e in events if re.search(r"travel", e["followup"], re.I) and re.search(r"'N'|\"N\"|destination|clarify|specify", e["followup"], re.I)),
    }
    summary["prompt_framing"] = {
        "over_escalation_followups_noting_red_flags_field_says_none": sum(1 for e in events if re.search(r"no recorded red flags|no red flags|intake form|intake label|despite the intake|lists no red|red flag despite", e["followup"], re.I)),
    }
    with open(OUT_DIR / "over_escalation_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_case[0].keys()))
        w.writeheader()
        w.writerows(per_case)
    (OUT_DIR / "over_escalation_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k in ("counts", "taxonomy", "rules")}, indent=1)[:6000])


# ---------------------------------------------------------------- taxonomy and rules


def within_pathology(cases, feats, parsed):
    """For each true pathology: urgent and non-urgent case counts under T and the mean share of models escalating in each group.

    Why: if the same pathology (so the same generated symptom pattern) lands on both sides of T, the label splits
    like presentations by which co-diagnoses DDXPlus happened to sample, not by anything the model can see.
    """
    out = defaultdict(lambda: {"urgent": [], "nonurgent": []})
    for c in cases:
        cid = c["case_id"]
        valid = [parsed[m][cid] for m in parsed if parsed[m][cid]]
        share = sum(p["escalate"] for p in valid) / len(valid) if valid else 0.0
        key = "urgent" if feats[cid]["p_severe"] >= T else "nonurgent"
        out[feats[cid]["pathology_name"]][key].append(share)
    rows = []
    for path, d in out.items():
        if d["nonurgent"]:
            rows.append({"pathology": path, "urgent_cases": len(d["urgent"]), "nonurgent_cases": len(d["nonurgent"]),
                         "mean_share_escalate_urgent": round(sum(d["urgent"]) / len(d["urgent"]), 3) if d["urgent"] else None,
                         "mean_share_escalate_nonurgent": round(sum(d["nonurgent"]) / len(d["nonurgent"]), 3)})
    return sorted(rows, key=lambda r: -r["nonurgent_cases"])


def relabel_impact(cases, feats, parsed, per_case):
    """Per model: under- and over-triage rates and triage score under the T label, then with TIER1 cases counted urgent,
    then with TIER1 and TIER2 counted urgent. Shows how much of the measured over-triage the rules reclassify."""
    import numpy as np
    ids = [c["case_id"] for c in cases]
    p = np.array([feats[c]["p_severe"] for c in ids])
    tier = {r["case_id"]: r["tier"] for r in per_case}
    variants = {
        "T=0.15": p,
        "T=0.15 + TIER1 urgent": np.array([1.0 if tier.get(c) == "TIER1" else v for c, v in zip(ids, p)]),
        "T=0.15 + TIER1+TIER2 urgent": np.array([1.0 if tier.get(c) in ("TIER1", "TIER2") else v for c, v in zip(ids, p)]),
    }
    out = {}
    for m in parsed:
        e = np.array([parsed[m][c]["escalate"] if parsed[m][c] else 0 for c in ids])
        row = {}
        for name, pv in variants.items():
            r = ts.score_model(pv, e)
            row[name] = {"under": round(r.under, 3), "over": round(r.over, 3), "score": round(r.score, 1)}
        out[SHORT.get(m, m)] = row
    counts = {name: {"urgent": int((pv >= T).sum()), "nonurgent": int((pv < T).sum())} for name, pv in variants.items()}
    mean = {name: {k: round(float(np.mean([out[m][name][k] for m in out])), 3) for k in ("under", "over", "score")} for name in variants}
    return {"label_counts": counts, "mean_over_models": mean, "per_model": out}


def escalated_for(events):
    """Which time-critical diagnosis families the escalating models ranked in their top 3, over all over-escalation events."""
    fam = Counter()
    none = 0
    for e in events:
        labs = set(l for l in e["time_critical_top3"].split("; ") if l)
        if not labs:
            none += 1
        for l in labs:
            fam[l] += 1
    return {"events": len(events), "events_with_no_time_critical_code_in_top3": none, "families": dict(fam.most_common())}



def summarize(per_case, events, feats, non_urgent, urg_rows, rows):
    n_models = len(rows)
    total_events = sum(r["models_escalate"] for r in per_case)
    by_share = Counter()
    for r in per_case:
        s = r["share_escalate"] or 0
        by_share["all models" if r["models_escalate"] == r["models_valid"] else ">=half" if s >= 0.5 else "<half" if s > 0 else "none"] += 1

    taxonomy = defaultdict(lambda: {"cases": 0, "events": 0, "verdicts": Counter()})
    verdicts = defaultdict(lambda: {"cases": 0, "events": 0})
    for r in per_case:
        if not r["category"]:
            continue
        t = taxonomy[r["category"]]
        t["cases"] += 1
        t["events"] += r["models_escalate"]
        t["verdicts"][r["verdict"]] += 1
        verdicts[r["verdict"]]["cases"] += 1
        verdicts[r["verdict"]]["events"] += r["models_escalate"]
    taxonomy = {k: {"cases": v["cases"], "events": v["events"], "verdicts": dict(v["verdicts"])} for k, v in taxonomy.items()}

    # Candidate separation rules, computable from the intake evidence codes and the DDXPlus differential.
    def has(r, *codes):
        return any(c in feats[r["case_id"]]["bases"] for c in codes)

    def head_pain(r):
        return bool(feats[r["case_id"]]["head_locations"])

    def neck_pain(r):
        return bool(feats[r["case_id"]]["neck_locations"])

    def bleed(r):
        return has(r, "E_210", "E_140", "E_179", "E_45")

    def thunderclap(r, min_intensity):
        f = feats[r["case_id"]]
        return head_pain(r) and (f["pain_onset"] or 0) >= 7 and (f["pain_intensity"] or 0) >= min_intensity

    def fever_immunosuppressed(r):
        return has(r, "E_91") and has(r, "E_227", "E_2", "E_44")

    def fever_ivdu(r):
        return has(r, "E_91") and has(r, "E_61")

    def fever_elderly(r):
        return has(r, "E_91") and r["age"] >= 80

    def meningism_picture(r):
        return has(r, "E_91", "E_94") and head_pain(r) and neck_pain(r) and has(r, "E_129")

    def severe_in_diff(r):
        return bool(feats[r["case_id"]]["severe_entries"])

    def tier1(r):
        return bleed(r) or thunderclap(r, 8) or fever_immunosuppressed(r)

    def tier2(r):
        return thunderclap(r, 7) or fever_ivdu(r) or fever_elderly(r) or meningism_picture(r)

    rules = {
        "R1 bleeding code (hematemesis, melena, hematochezia, hemoptysis)": bleed,
        "R2 thunderclap headache (head pain, onset >= 7, intensity >= 8)": lambda r: thunderclap(r, 8),
        "R2b thunderclap headache, intensity >= 7": lambda r: thunderclap(r, 7),
        "R3 fever + immunosuppressed / HIV / corticosteroids": fever_immunosuppressed,
        "R4 fever + IV drug use": fever_ivdu,
        "R5 fever + age >= 80": fever_elderly,
        "R6 fever or chills + headache + neck pain + rash": meningism_picture,
        "P1 P(severe) >= 0.05": lambda r: r["p_severe"] >= 0.05,
        "P2 P(severe) >= 0.10": lambda r: r["p_severe"] >= 0.10,
        "S1 a severity 1-2 diagnosis anywhere in the DDXPlus differential": severe_in_diff,
        "TIER1 = R1 or R2 or R3 (clinician would escalate)": tier1,
        "TIER2 = R2b or R4 or R5 or R6, not TIER1 (debatable)": lambda r: tier2(r) and not tier1(r),
        "TIER1 or TIER2": lambda r: tier1(r) or tier2(r),
    }
    rule_out = {}
    labelled = [r for r in per_case if r["verdict"]]
    for name, fn in rules.items():
        hit = [r for r in per_case if fn(r)]
        hit_lab = [r for r in labelled if fn(r)]
        miss_lab = [r for r in labelled if not fn(r)]
        rule_out[name] = {
            "cases_flagged": len(hit), "events_on_flagged": sum(r["models_escalate"] for r in hit),
            "share_of_over_escalation_events": round(sum(r["models_escalate"] for r in hit) / total_events, 3) if total_events else None,
            "mean_share_escalate_flagged": round(sum(r["share_escalate"] or 0 for r in hit) / len(hit), 3) if hit else None,
            "mean_share_escalate_unflagged": round(sum(r["share_escalate"] or 0 for r in per_case if not fn(r)) / max(1, len(per_case) - len(hit)), 3),
            "flagged_verdicts": dict(Counter(r["verdict"] for r in hit_lab)),
            "unflagged_verdicts": dict(Counter(r["verdict"] for r in miss_lab)),
            "flagged_cases": [r["case_id"] for r in hit],
        }
    for r in per_case:
        r["tier"] = "TIER1" if tier1(r) else "TIER2" if tier2(r) else "none"

    # Per-model reclassification: how many of each model's over-escalations fall in each tier.
    per_model = {}
    for e in events:
        m = per_model.setdefault(e["model"], Counter())
        m[next(r["tier"] for r in per_case if r["case_id"] == e["case_id"])] += 1
        m["total"] += 1
    per_model = {m: dict(c) for m, c in sorted(per_model.items())}

    # Reverse check on urgent cases.
    urg_esc = sum(r["models_escalate"] for r in urg_rows)
    urg_named = sum(r["escalators_naming_driving_dx_top3"] for r in urg_rows)
    urg_any = sum(r["escalators_naming_any_time_critical_top3"] for r in urg_rows)

    return {
        "T": T, "models": n_models, "nonurgent_cases": len(non_urgent), "urgent_cases": len(urg_rows),
        "counts": {"over_escalation_events": total_events, "cases_by_share": dict(by_share),
                   "mean_over_escalation_rate": round(total_events / sum(r["models_valid"] for r in per_case), 3)},
        "taxonomy": taxonomy, "verdicts": dict(verdicts), "rules": rule_out, "per_model_tiers": per_model,
        "reverse_check": {"escalations_on_urgent_cases": urg_esc, "naming_driving_dx_in_top3": urg_named,
                          "naming_any_time_critical_in_top3": urg_any},
        "rows": [{"name": r["name"], "path": r["path"], "note": r["note"]} for r in rows],
    }


if __name__ == "__main__":
    main()
