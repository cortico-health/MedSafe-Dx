"""
Meaning-preserving renderings of the intake, for the paired rendering test
(docs/v0.3-memorisation-checks.md section 3, docs/v0.3-rendering-variants.md).

A model that has learned DDXPlus's surface patterns (the decoder's exact phrases in
the decoder's exact order) can score well without reading the patient. Each
rendering here keeps every finding and changes only the surface, so a paired run
across renderings measures surface-pattern reliance: the findings are the same, so a
large score change can only come from the wording or the order.

- "standard": the v0.2 decoder's strings in DDXPlus's evidence order (every run so far).
- "shuffled": the same strings, symptoms and history each shuffled with a seed fixed
  per case, so a rerun renders the same order and a scorer can pair cases.
- "paraphrased": the same order, with each of the 40 most frequent strings replaced by
  a fixed alternative phrase (PARAPHRASE); every other string is unchanged.

`render_strings` applies one rendering to a decoded case. The tests in
evaluator/tests/test_rendering_variants.py check that the content is identical.
"""

from __future__ import annotations

import random
from typing import Sequence

RENDERINGS = ("standard", "shuffled", "paraphrased")
RENDER_SEED = 20260925

# The 40 most frequent v0.2-decoder strings over the 470 main sample and both v0.3 pools
# (720 cases; scripts/analysis/v03_memorisation.py writes the counts to
# results/analysis/v03_memorisation/frequent_strings.csv). Together they are 42% of the
# string occurrences in those cases. Each alternative names the same finding; a
# sex-specific site or a scale value keeps its value.
PARAPHRASE: dict[str, str] = {
    "No travel outside the country in the last 4 weeks": "Has not travelled abroad in the past 4 weeks",
    "Pain present": "Reports pain",
    "Experiencing shortness of breath or difficulty breathing in a significant way":
        "Marked breathlessness or difficulty breathing",
    "Pain does not radiate": "The pain stays in one place and does not spread",
    "A cough": "Coughing",
    "Smoke cigarettes": "Current cigarette smoker",
    "Pain character: tender": "The pain feels tender",
    "Pain character: stabbing": "The pain feels stabbing",
    "Pain character: sharp": "The pain feels sharp",
    "Pain location: lower chest": "Pain felt in the lower chest",
    "Pain location: epigastrium": "Pain felt in the epigastrium",
    "Significantly overweight compared to people of the same height as you": "Markedly overweight for their height",
    "Pain location: upper chest": "Pain felt in the upper chest",
    "A fever (either felt or measured with a thermometer)": "Fever, felt or measured with a thermometer",
    "Pain location: breast (L)": "Pain felt in the left breast",
    "Drink alcohol excessively or do you have an addiction to alcohol": "Excessive alcohol use or alcohol dependence",
    "Pain location: breast (R)": "Pain felt in the right breast",
    "Asthma or have you ever had to use a bronchodilator in the past": "History of asthma or past bronchodilator use",
    "Pain location: side of the chest (R)": "Pain felt in the right side of the chest",
    "Any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for":
        "Skin lesions, redness or other skin changes the patient links to this complaint",
    "Constantly feel fatigued or do you have non-restful sleep": "Persistent fatigue or unrefreshing sleep",
    "Pain location: side of the chest (L)": "Pain felt in the left side of the chest",
    "Diabetes": "Diabetic",
    "Pain intensity 6/10": "Pain rated 6 out of 10",
    "Pain character: heavy": "The pain feels heavy",
    "Pain character: burning": "The pain feels burning",
    "Pain character: unpleasant": "The pain feels unpleasant",
    "Pain radiates to: thoracic spine": "Pain spreads to the thoracic spine",
    "Ever had a heart attack or do you have angina (chest pain)": "Previous heart attack, or known angina (chest pain)",
    "Skin lesions peel off: no": "The skin lesions do not peel",
    "Skin lesion larger than 1 cm: yes": "At least one skin lesion is over 1 cm across",
    "How precisely the pain can be located: 6/10 (10 = most precise)":
        "Pain localisation 6 out of 10, where 10 is most precisely located",
    "Feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations":
        "Palpitations: heart racing, beating irregularly or missing beats",
    "High blood pressure or do you take medications to treat high blood pressure":
        "Hypertension, or on blood-pressure medication",
    "Feeling nauseous or do you feel like vomiting": "Nausea or the urge to vomit",
    "Pain location: forehead": "Pain felt in the forehead",
    "Pain intensity 5/10": "Pain rated 5 out of 10",
    "Pain character: exhausting": "The pain feels exhausting",
    "How fast the pain appeared: 4/10 (10 = fastest)": "Speed of pain onset 4 out of 10, where 10 is fastest",
    "How precisely the pain can be located: 2/10 (10 = most precise)":
        "Pain localisation 2 out of 10, where 10 is most precisely located",
}


def inverse_table(table: dict[str, str] = PARAPHRASE) -> dict[str, str]:
    """alternative -> original. Raises when two originals share an alternative, because
    then a paraphrased prompt could not be read back to its findings."""
    inv: dict[str, str] = {}
    for k, v in table.items():
        if v in inv:
            raise ValueError(f"paraphrase {v!r} stands for both {inv[v]!r} and {k!r}")
        inv[v] = k
    return inv


def paraphrase_strings(strings: Sequence[str], table: dict[str, str] = PARAPHRASE) -> list[str]:
    return [table.get(s, s) for s in strings]


def case_rng(case_id: str, seed: int = RENDER_SEED) -> random.Random:
    """One generator per case, seeded from the fixed seed and the case id, so every run
    shuffles a case the same way (random.Random hashes a str seed with sha512, which is
    stable across Python versions)."""
    return random.Random(f"{seed}:{case_id}")


def shuffle_strings(active: Sequence[str], antecedents: Sequence[str], case_id: str,
                    seed: int = RENDER_SEED) -> tuple[list[str], list[str]]:
    rng = case_rng(case_id, seed)
    a, h = list(active), list(antecedents)
    rng.shuffle(a)
    rng.shuffle(h)
    return a, h


def render_strings(active: Sequence[str], antecedents: Sequence[str], case_id: str,
                   rendering: str = "standard") -> tuple[list[str], list[str]]:
    """The decoded strings under one rendering. Symptoms and history are rendered
    separately because the prompt prints them on separate lines."""
    if rendering == "standard":
        return list(active), list(antecedents)
    if rendering == "shuffled":
        return shuffle_strings(active, antecedents, case_id)
    if rendering == "paraphrased":
        return paraphrase_strings(active), paraphrase_strings(antecedents)
    raise ValueError(f"unknown rendering {rendering!r}; expected one of {RENDERINGS}")
