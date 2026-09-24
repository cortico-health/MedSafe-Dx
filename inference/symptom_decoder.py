"""
Decode DDXPlus symptom codes to human-readable clinical findings.
"""

import json
from pathlib import Path


def load_evidence_data():
    """Load full symptom/evidence data including value meanings."""
    evidence_path = Path(__file__).parent.parent / "data" / "ddxplus_v0" / "release_evidences.json"
    try:
        with open(evidence_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load evidence data: {e}")
        return {}


EVIDENCE_DATA = load_evidence_data()


def _decode_symptom_with_audit(symptom_code):
    """
    Decode symptom code to human-readable clinical text, plus decode fidelity signals.
    Returns: (description, is_antecedent, audit) tuple.
    """
    audit = {
        "input_code": symptom_code,
        "unknown_evidence_code": False,
        "unknown_value_code": False,
    }

    # Handle compound codes like "E_55_@_V_29" or "E_58_@_2"
    parts = symptom_code.split("_@_")
    base_code = parts[0]
    audit["base_code"] = base_code

    # Get base evidence data
    evidence = EVIDENCE_DATA.get(base_code)
    if not evidence:
        audit["unknown_evidence_code"] = True
        evidence = {}

    question = evidence.get("question_en", symptom_code)
    data_type = evidence.get("data_type", "")
    is_antecedent = evidence.get("is_antecedent", False)

    # Convert question to clinical statement
    description = question.replace("Do you have ", "").replace("Have you ", "").replace("?", "")
    description = description.replace("Do you ", "").replace("Are you ", "").replace("Is ", "")
    description = description.replace("Does the ", "").replace("Did the ", "")
    description = description.replace("How ", "").replace("What ", "")
    description = description.strip()

    # Handle specific evidence codes for cleaner presentation
    if base_code == "E_53":
        description = "Pain present"
    elif base_code == "E_57" and not parts[1:]:
        description = "Pain radiation"

    # If there's a value modifier
    if len(parts) > 1:
        value_code = parts[1]

        value_meanings = evidence.get("value_meaning", {})
        if value_code in value_meanings:
            value_text = value_meanings[value_code].get("en", value_code)

            if "feel pain somewhere" in question.lower():
                if value_text.lower() != "nowhere":
                    description = f"pain in {value_text}"
            elif "characterize your pain" in question.lower():
                description = f"pain character {value_text}"
            elif "irradiat" in question.lower() or "radiate" in question.lower():
                description = f"pain radiating to {value_text}"
            elif base_code == "E_204":
                return (f"Recent travel to {value_text}", True, audit)
            else:
                description = f"{description}: {value_text}"
        else:
            audit["unknown_value_code"] = True

            # Check if it's a numeric scale (data_type C with numeric value)
            if data_type == "C" and value_code.isdigit():
                if "intense" in question.lower():
                    description = f"pain intensity {value_code}/10"
                elif "precisely" in question.lower():
                    description = f"pain is diffuse (localization {value_code}/10)"
                elif "fast" in question.lower():
                    description = "sudden pain onset" if int(value_code) >= 7 else "gradual pain onset"
                else:
                    description = f"{description} {value_code}/10"
            else:
                # Try to find value in other evidences (for location codes)
                found = False
                for other_code, other_evidence in EVIDENCE_DATA.items():
                    if value_code in other_evidence.get("value_meaning", {}):
                        value_text = other_evidence["value_meaning"][value_code].get("en", value_code)
                        description = f"{description}: {value_text}"
                        found = True
                        break

                if not found and not value_code.isdigit():
                    description = f"{description}: {value_code}"

    # Capitalize first letter
    if description and description[0].islower():
        description = description[0].upper() + description[1:]

    return (description, is_antecedent, audit)


def decode_symptom(symptom_code):
    """
    Decode symptom code to human-readable clinical text.
    Returns: (description, is_antecedent) tuple
    
    Examples:
        E_19 -> ("Hyperthyroidism", True)
        E_55_@_V_29 -> ("Pain in lower chest", False)
    """
    description, is_antecedent, _audit = _decode_symptom_with_audit(symptom_code)
    return (description, is_antecedent)


def decode_symptoms(symptom_codes):
    """
    Decode a list of symptom codes to human-readable text.
    Returns: (active_symptoms, antecedents) tuple of lists
    """
    active_symptoms = []
    antecedents = []
    
    for code in symptom_codes:
        result = decode_symptom(code)
        if not result:
            continue
            
        text, is_antecedent = result
        if not text:  # Skip None values
            continue
            
        if is_antecedent:
            antecedents.append(text)
        else:
            active_symptoms.append(text)
            
    return active_symptoms, antecedents


def decode_symptoms_with_audit(symptom_codes):
    """
    Decode a list of symptom codes to human-readable text, plus decode fidelity stats.
    Returns: (active_symptoms, antecedents, audit) tuple.
    """
    active_symptoms = []
    antecedents = []
    unknown_evidence_codes: list[str] = []
    unknown_value_codes: list[str] = []

    for code in symptom_codes or []:
        text, is_antecedent, audit = _decode_symptom_with_audit(code)
        if audit.get("unknown_evidence_code"):
            unknown_evidence_codes.append(str(code))
        if audit.get("unknown_value_code"):
            unknown_value_codes.append(str(code))

        if not text:
            continue
        if is_antecedent:
            antecedents.append(text)
        else:
            active_symptoms.append(text)

    return active_symptoms, antecedents, {
        "total_codes": len(symptom_codes or []),
        "unknown_evidence_codes": unknown_evidence_codes[:25],
        "unknown_value_codes": unknown_value_codes[:25],
        "unknown_evidence_count": len(unknown_evidence_codes),
        "unknown_value_count": len(unknown_value_codes),
    }


# ---------------------------------------------------------------------------
# v0.2 decoder (spec/v0.2-scoring.md section 2)
#
# The v0 functions above stay byte-for-byte as they were, so the v0 test sets
# decode to the prompts the published v0 runs saw. Callers pick the v0.2
# rendering with version="v02". docs/v0.2-decoder-audit.md lists every defect
# this fixes and every one it leaves in place.
# ---------------------------------------------------------------------------

DECODER_VERSIONS = ("v0", "v02")

# DDXPlus answer values that mean "no answer" or "nowhere". v0 printed them as
# findings ("Pain character NA", "Pain radiating to nowhere", "Feel pain somewhere").
_V02_EMPTY_VALUES = {"V_11", "V_123"}

# English value labels that mistranslate the French original (release_evidences.json).
_V02_VALUE_TEXT = {
    # E_54 pain character
    "V_71": "tearing",                 # "déchirante"; v0: "heartbreaking"
    "V_112": "lancinating (shooting)", # "lancinante"; v0: "haunting"
    "V_154": "unpleasant",             # "pénible"; v0: "tedious"
    "V_161": "tender",                 # "sensible"; v0: "sensitive"
    "V_179": "stabbing",               # "un coup de couteau"; v0: "a knife stroke"
    "V_184": "throbbing",              # "une pulsation"; v0: "a pulse"
    "V_182": "cramping",               # "une crampe"; v0: "a cramp"
    # body locations
    "V_137": "palate",                 # "palais"; v0: "palace"
    "V_105": "hamstring (R)",          # "ischio(D)"; v0: "ischio jambier(R)"
    "V_106": "hamstring (L)",
    "V_197": "epigastrium",            # v0: "epigastric" (adjective)
    # E_204 travel
    "V_8": "the Caribbean",            # v0: "Caraibes"
}

# Genital sites that contradict the patient's recorded sex. DDXPlus generates
# testicular pain for female inguinal-hernia patients and labial lesions for male
# HIV patients; v0.2 names the region without the sex-specific organ.
_V02_MALE_SITES = {"V_168": "genital area (R)", "V_169": "genital area (L)", "V_155": "genitals", "V_158": "genitals"}
_V02_FEMALE_SITES = {"V_95": "genitals (R)", "V_96": "genitals (L)", "V_146": "genitals (R)", "V_147": "genitals (L)"}

# Whole-evidence renderings where the v0 text changes the meaning.
_V02_EVIDENCE_TEXT = {
    "E_4": "History of croup in the patient or a family member",
    "E_202": "Barking cough",  # "toux aboyante"; the English question says "whooping cough"
}

_YES_NO = {"V_12": "yes", "V_10": "no"}

# Detail questions and the question that opens them. DDXPlus fills the details
# with default answers (0, "NA", "nowhere", "N") even when the patient answered
# no to the opening question, and v0 printed them: a PSVT patient without pain
# read "Feel pain somewhere, Gradual pain onset". v0.2 drops a detail code whose
# opening question is absent. Every such code in the test split is a default value.
_V02_DETAIL_OF = {
    **{c: "E_53" for c in ("E_54", "E_55", "E_56", "E_57", "E_58", "E_59")},  # pain
    **{c: "E_129" for c in ("E_130", "E_131", "E_132", "E_133", "E_134", "E_135", "E_136")},  # skin lesions
    "E_152": "E_151",  # swelling
}


def _v02_value_text(evidence: dict, value_code: str, sex: str | None) -> str | None:
    """Readable text for a categorical value, or None when the value means 'none'."""
    if value_code in _V02_EMPTY_VALUES:
        return None
    s = (sex or "").lower()
    if s in ("female", "f") and value_code in _V02_MALE_SITES:
        return _V02_MALE_SITES[value_code]
    if s in ("male", "m") and value_code in _V02_FEMALE_SITES:
        return _V02_FEMALE_SITES[value_code]
    if value_code in _V02_VALUE_TEXT:
        return _V02_VALUE_TEXT[value_code]
    meaning = evidence.get("value_meaning", {}).get(value_code)
    if meaning:
        return meaning.get("en", value_code).replace("(R)", " (R)").replace("(L)", " (L)").replace("  ", " ")
    return None


def _decode_symptom_v02(symptom_code: str, sex: str | None = None):
    """
    v0.2 rendering. Returns (description or None, is_antecedent, audit).
    None means the code carries no finding ("nowhere", "NA") and is left out of the prompt.
    """
    parts = str(symptom_code).split("_@_")
    base = parts[0]
    value = parts[1] if len(parts) > 1 else None
    evidence = EVIDENCE_DATA.get(base)
    audit = {"input_code": symptom_code, "base_code": base,
             "unknown_evidence_code": evidence is None, "unknown_value_code": False}
    if evidence is None:
        return (symptom_code, False, audit)
    is_ante = bool(evidence.get("is_antecedent", False))

    if value is None and base in _V02_EVIDENCE_TEXT:
        return (_V02_EVIDENCE_TEXT[base], is_ante, audit)

    if value is not None and value.isdigit():
        n = int(value)
        scales = {
            "E_56": f"Pain intensity {n}/10",
            "E_58": f"How precisely the pain can be located: {n}/10 (10 = most precise)",
            "E_59": f"How fast the pain appeared: {n}/10 (10 = fastest)",
            "E_132": f"Skin lesions raised: {n}/10",
            "E_134": f"Pain caused by the skin lesions: {n}/10",
            "E_136": f"Itching of the skin lesions: {n}/10",
        }
        if base in scales:
            return (scales[base], is_ante, audit)

    if value is not None:
        if value not in evidence.get("value_meaning", {}) and not value.isdigit():
            audit["unknown_value_code"] = True
        if base == "E_204":
            if value == "V_10":
                return ("No travel outside the country in the last 4 weeks", True, audit)
            text = _v02_value_text(evidence, value, sex) or value
            return (f"Travel outside the country in the last 4 weeks: {text}", True, audit)
        if base in ("E_131", "E_135"):
            yn = _YES_NO.get(value, value)
            label = "Skin lesions peel off" if base == "E_131" else "Skin lesion larger than 1 cm"
            return (f"{label}: {yn}", is_ante, audit)
        if base == "E_57" and value == "V_123":
            return ("Pain does not radiate", is_ante, audit)
        text = _v02_value_text(evidence, value, sex)
        if text is None:
            return (None, is_ante, audit)
        templates = {
            "E_54": "Pain character: {}",
            "E_55": "Pain location: {}",
            "E_57": "Pain radiates to: {}",
            "E_130": "Skin lesion colour: {}",
            "E_133": "Skin lesion location: {}",
            "E_152": "Swelling location: {}",
        }
        if base in templates:
            return (templates[base].format(text), is_ante, audit)

    # Binary evidences, and any code the tables above do not cover: the v0 rendering.
    description, is_ante_v0, audit_v0 = _decode_symptom_with_audit(symptom_code)
    audit_v0["unknown_evidence_code"] = audit["unknown_evidence_code"]
    return (description, is_ante_v0, audit_v0)


def decode_symptoms_versioned(symptom_codes, version: str = "v0", sex: str | None = None):
    """
    Decode codes with the chosen decoder. Returns (active_symptoms, antecedents, audit),
    the same shape as decode_symptoms_with_audit. version="v0" reproduces v0 exactly
    (sex is ignored); version="v02" applies the v0.2 fixes.
    """
    if version == "v0":
        return decode_symptoms_with_audit(symptom_codes)
    if version != "v02":
        raise ValueError(f"unknown decoder version {version!r}; expected one of {DECODER_VERSIONS}")
    active, antecedents = [], []
    unknown_ev, unknown_val, dropped, orphans = [], [], [], []
    present = {str(c).split("_@_")[0] for c in symptom_codes or []}
    for code in symptom_codes or []:
        parent = _V02_DETAIL_OF.get(str(code).split("_@_")[0])
        if parent and parent not in present:
            orphans.append(str(code))
            continue
        text, is_ante, audit = _decode_symptom_v02(code, sex)
        if audit.get("unknown_evidence_code"):
            unknown_ev.append(str(code))
        if audit.get("unknown_value_code"):
            unknown_val.append(str(code))
        if not text:
            dropped.append(str(code))
            continue
        (antecedents if is_ante else active).append(text)
    return active, antecedents, {
        "decoder_version": version,
        "total_codes": len(symptom_codes or []),
        "unknown_evidence_codes": unknown_ev[:25],
        "unknown_value_codes": unknown_val[:25],
        "unknown_evidence_count": len(unknown_ev),
        "unknown_value_count": len(unknown_val),
        "dropped_empty_codes": dropped,
        "dropped_orphan_detail_codes": orphans,
    }
