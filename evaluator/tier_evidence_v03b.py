"""
Tier evidence strength for MedSafe-Dx v0.3b (key fix #11; docs/v0.3-key-fixes.md section 5).

We record, per DDXPlus condition, which source rows name it, and derive from them:

1. `final_tier`: the v0.3 rule (docs/tier-upgrade-sources.md section 2). Tier 1 when a
   Newman-Toker 2023 Table 1 row names the condition, or when at least 2 of the 3
   independent sources (S1 Singh 2013, S2 Hussain 2019, S3 Miyagami 2023) name it;
   otherwise the severity crosswalk. v0.3 counts one category row, "Other Cancers" for
   pancreatic neoplasm, and no other.
2. `final_tier_formal`: the same rule with the category exception stated as a general
   rule (CATEGORY_RULE below) and applied to every source row.
3. `final_tier_count_floor`: the v0.3 rule when a source row needs a count of at
   least 2 to name a condition (the asthma sensitivity row of
   docs/tier-upgrade-sources.md section 4).
4. `evidence_level` of the final tier, strongest first: Newman-Toker named row,
   2-of-3 sources, Newman-Toker category row, DDXPlus-severity-only.

Absence from a source never lowers a tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

NT, S1, S2, S3 = "NT2023", "S1", "S2", "S3"
INDEPENDENT = (S1, S2, S3)

NAMED, CATEGORY = "named", "category"

# How we read a category row under the formal rule.
CANCER, FAMILY, NOT_FAMILY = "cancer", "disease_family", "not_a_family"

CATEGORY_RULE = (
    "A category row names a condition when (a) the category names the condition's cancer "
    "(any malignant neoplasm, because each cancer row of the source is an organ-system cancer "
    "and a residual cancer row is the rest of that same list) or the condition's disease family "
    "(one disease process: an arrhythmia, a fracture, an otitis, heart failure), and (b) the same "
    "source has no condition-level row for that condition. Rows that span unrelated processes or "
    "organ systems (Newman-Toker 'Other Infections' and 'Other Vascular Events', Singh 'Viral "
    "syndrome' and 'Psychiatric disorder') name no family and never count."
)

LEVEL_NT_NAMED = "Newman-Toker named row"
LEVEL_TWO_OF_THREE = "2-of-3 sources"
LEVEL_NT_CATEGORY = "Newman-Toker category row"
LEVEL_SEVERITY = "DDXPlus-severity-only"


@dataclass(frozen=True)
class SourceRow:
    """One row of a source's per-diagnosis table that mentions a condition."""

    source: str  # NT, S1, S2 or S3
    row: str  # the row text as printed
    kind: str  # NAMED or CATEGORY
    count: Optional[int] = None  # cases in the row, when the source prints one and we recorded it
    family: str = ""  # for CATEGORY rows: CANCER, FAMILY or NOT_FAMILY


# Every source row that mentions one of our conditions (docs/tier-upgrade-sources.md
# sections 1 and 3; docs/dangerous-if-missed-labels.md section 2). Conditions absent
# here are named by no row.
SOURCE_ROWS: dict[str, tuple[SourceRow, ...]] = {
    "Possible NSTEMI / STEMI": (
        SourceRow(NT, "Myocardial Infarction", NAMED),
        SourceRow(S1, "Angina/myocardial infarction/acute coronary syndrome", NAMED),
        SourceRow(S2, "MI", NAMED, 161),
        SourceRow(S3, "AMI", NAMED, 5),
    ),
    "Pulmonary embolism": (
        SourceRow(NT, "Venous Thromboembolism", NAMED),
        SourceRow(S1, "Pulmonary embolism", NAMED),
        SourceRow(S2, "PE", NAMED, 34),
    ),
    "Pneumonia": (
        SourceRow(NT, "Pneumonia", NAMED),
        SourceRow(S1, "Pneumonia", NAMED, 14),
        SourceRow(S2, "Pneumonia", NAMED, 8),
        SourceRow(S3, "pneumonia (Table 4 final diagnosis)", NAMED, 1),
    ),
    "Pulmonary neoplasm": (
        SourceRow(NT, "Lung Cancer", NAMED),
        SourceRow(S1, "Cancer (primary)", CATEGORY, 11, CANCER),
    ),
    "Pancreatic neoplasm": (
        SourceRow(NT, "Other Cancers", CATEGORY, None, CANCER),
        SourceRow(S1, "Cancer (primary)", CATEGORY, 11, CANCER),
    ),
    "Bronchospasm / acute asthma exacerbation": (
        SourceRow(S1, "Asthma exacerbation", NAMED, 1),
        SourceRow(S3, "Bronchial asthma", NAMED, 2),
    ),
    "Stable angina": (SourceRow(S1, "Angina/myocardial infarction/acute coronary syndrome", NAMED),),
    "Unstable angina": (SourceRow(S1, "Angina/myocardial infarction/acute coronary syndrome", NAMED),),
    "Epiglottitis": (SourceRow(S3, "Epiglottitis", NAMED, 4),),
    "Acute pulmonary edema": (SourceRow(S1, "Decompensated congestive heart failure", CATEGORY, 12, FAMILY),),
    "PSVT": (SourceRow(S1, "Cardiac dysrhythmia", CATEGORY, 3, FAMILY),),
    "Atrial fibrillation": (
        SourceRow(S1, "Atrial fibrillation (new onset)", NAMED, 1),
        SourceRow(S1, "Cardiac dysrhythmia", CATEGORY, 3, FAMILY),
    ),
    "HIV (initial infection)": (SourceRow(S1, "HIV", NAMED, 1),),
    "Anemia": (SourceRow(S1, "Symptomatic anemia", NAMED, 9),),
    "SLE": (SourceRow(S1, "Complicated lupus", NAMED, 1),),
    "Spontaneous rib fracture": (
        SourceRow(S1, "Fracture", CATEGORY, 2, FAMILY),
        SourceRow(S2, "Fracture", CATEGORY, 1007, FAMILY),
    ),
    "Acute otitis media": (SourceRow(S1, "Otitis", CATEGORY, 3, FAMILY),),
    "Influenza": (SourceRow(S1, "Viral syndrome", CATEGORY, 1, NOT_FAMILY),),
    "URTI": (SourceRow(S1, "Viral syndrome", CATEGORY, 1, NOT_FAMILY),),
    "Panic attack": (SourceRow(S1, "Psychiatric disorder", CATEGORY, 2, NOT_FAMILY),),
}

# v0.3 counted exactly this category row (docs/dangerous-if-missed-labels.md section 3, rule A).
V03_CATEGORY_EXCEPTIONS = {("Pancreatic neoplasm", NT, "Other Cancers")}


def base_tier(severity: int) -> int:
    """DDXPlus severity crosswalk: 1-2 -> tier 1, 3 -> tier 2, 4-5 -> tier 3."""
    return 1 if severity <= 2 else 2 if severity == 3 else 3


def counts_formal(condition: str, row: SourceRow, rows: tuple[SourceRow, ...]) -> bool:
    """Whether a row names the condition under the formal category rule."""
    if row.kind == NAMED:
        return True
    if row.family not in (CANCER, FAMILY):
        return False
    return not any(r.source == row.source and r.kind == NAMED for r in rows)


def counts_v03(condition: str, row: SourceRow, rows: tuple[SourceRow, ...]) -> bool:
    """Whether a row names the condition under the v0.3 rule: named rows plus the one listed exception."""
    return row.kind == NAMED or (condition, row.source, row.row) in V03_CATEGORY_EXCEPTIONS


def naming_sources(condition: str, rows: tuple[SourceRow, ...], counts, count_floor: int = 0) -> set[str]:
    """Sources with at least one row that names the condition. A row with a recorded count
    below `count_floor` does not name; an unrecorded count passes (only S1's severity 1-2 rows
    and NT rows lack one, and neither decides a tier through the 2-of-3 rule)."""
    out = set()
    for r in rows:
        if r.count is not None and r.count < count_floor:
            continue
        if counts(condition, r, rows):
            out.add(r.source)
    return out


def tier_under(condition: str, severity: int, counts, count_floor: int = 0) -> tuple[int, str]:
    """(final tier, evidence level) under a row-counting rule."""
    rows = SOURCE_ROWS.get(condition, ())
    src = naming_sources(condition, rows, counts, count_floor)
    nt_rows = [r for r in rows if r.source == NT and counts(condition, r, rows)]
    n_indep = len(src & set(INDEPENDENT))
    if any(r.kind == NAMED for r in nt_rows):
        return 1, LEVEL_NT_NAMED
    if n_indep >= 2:
        return 1, LEVEL_TWO_OF_THREE
    if nt_rows:
        return 1, LEVEL_NT_CATEGORY
    return base_tier(severity), LEVEL_SEVERITY


def independent_count(condition: str, counts) -> int:
    rows = SOURCE_ROWS.get(condition, ())
    return len(naming_sources(condition, rows, counts) & set(INDEPENDENT))


def table_row(condition: str, icd10: str, severity: int) -> dict:
    """One row of spec/dangerous_if_missed_tiers_v03b.csv."""
    rows = SOURCE_ROWS.get(condition, ())
    tier, level = tier_under(condition, severity, counts_v03)
    tier_f, level_f = tier_under(condition, severity, counts_formal)
    tier_c, _ = tier_under(condition, severity, counts_v03, count_floor=2)
    cats = [r for r in rows if r.kind == CATEGORY]
    exc = []
    for r in cats:
        v03 = counts_v03(condition, r, rows)
        formal = counts_formal(condition, r, rows)
        exc.append(f"{r.source} '{r.row}' ({r.family}): v0.3 {'counts' if v03 else 'no'}, formal {'counts' if formal else 'no'}")
    return {
        "condition": condition, "icd10": icd10, "ddxplus_severity": severity, "base_tier": base_tier(severity),
        "final_tier": tier, "evidence_level": level,
        "nt_row": "; ".join(f"{r.row} ({r.kind})" for r in rows if r.source == NT),
        "independent_sources_named": independent_count(condition, counts_v03),
        "source_rows": " | ".join(f"{r.source} '{r.row}' {r.kind}" + (f" n={r.count}" if r.count is not None else "")
                                  for r in rows),
        "category_exception": "; ".join(exc),
        "final_tier_formal": tier_f, "evidence_level_formal": level_f, "tier_change_formal": tier_f != tier,
        "final_tier_count_floor": tier_c, "count_floor_sensitive": tier_c != tier,
    }


TABLE_COLUMNS = ("condition", "icd10", "ddxplus_severity", "base_tier", "final_tier", "evidence_level", "nt_row",
                 "independent_sources_named", "source_rows", "category_exception", "final_tier_formal",
                 "evidence_level_formal", "tier_change_formal", "final_tier_count_floor", "count_floor_sensitive")
