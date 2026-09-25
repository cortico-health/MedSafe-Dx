"""
Scorer-facing views of the v0.3b answer key (evaluator/answer_key_v03b.py; key fixes
#7, #8 and #10 of docs/v0.3-key-fixes.md). Each view returns the case IDs a measure
scores, so the scorer keeps one cost rule and only changes its denominators.

Views:

1. `headline`: SERIOUS (an R10 target) against chargeable BENIGN. The default headline.
2. `truth_only_headline`: SERIOUS_TRUTH (tier-1 truth) against chargeable BENIGN, the
   evidence stratum where the synthetic diagnosis itself is serious (Astra finding 1).
3. `dxa_supported_risk_ignored`: on SERIOUS_DXA_ONLY cases, the share the model did not
   escalate. Printed as its own line, because these misses disagree with a DXA-derived
   concern rather than with the true condition.
4. `chargeable_benign`: BENIGN cases where over-concern costs; `strict=True` also drops
   the cases whose truth has a documented serious off-list complication.
5. `pool_with_controls`: a pool's SERIOUS cases against its matched controls, so blanket
   escalation no longer scores perfectly on a pool.

Every function takes the loaded key, a mapping case_id -> CaseKeyV03b.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Optional

from evaluator import answer_key_v03b as akb

Key = Mapping[str, "akb.CaseKeyV03b"]


def ids_of(key: Key, evidence_class: str) -> list[str]:
    return [cid for cid, k in key.items() if k.evidence_class == evidence_class]


def serious(key: Key) -> list[str]:
    """Cases with an R10 target (SERIOUS_TRUTH and SERIOUS_DXA_ONLY)."""
    return [cid for cid, k in key.items() if k.evidence_class in (akb.SERIOUS_TRUTH, akb.SERIOUS_DXA_ONLY)]


def chargeable_benign(key: Key, strict: bool = False) -> list[str]:
    """BENIGN cases where escalation costs; strict=True also drops documented off-list complications."""
    return [cid for cid, k in key.items() if k.oc_chargeable and not (strict and k.offlist_complication)]


def exempt_benign(key: Key) -> list[str]:
    """BENIGN cases with a removed or uncertain DXA tier-1 concern at p >= 5%: reported, never charged."""
    return [cid for cid, k in key.items() if k.evidence_class == akb.BENIGN and not k.oc_chargeable]


def headline(key: Key) -> tuple[list[str], list[str]]:
    """(serious, benign) ID lists for the default headline."""
    return serious(key), chargeable_benign(key)


def truth_only_headline(key: Key) -> tuple[list[str], list[str]]:
    """(serious, benign) ID lists where serious means the true condition is tier 1."""
    return ids_of(key, akb.SERIOUS_TRUTH), chargeable_benign(key)


def dxa_supported_risk_ignored(key: Key, escalated: Mapping[str, Optional[bool]]) -> dict:
    """The 'DXA-supported risk ignored' line: SERIOUS_DXA_ONLY cases not escalated.
    A case missing from `escalated`, or mapped to None (unreadable), counts as not escalated."""
    ids = ids_of(key, akb.SERIOUS_DXA_ONLY)
    missed = [cid for cid in ids if not escalated.get(cid)]
    return {"n": len(ids), "not_escalated": len(missed), "rate": len(missed) / len(ids) if ids else None,
            "ids": missed}


def pool_with_controls(pool_key: Key, control_key: Key) -> tuple[list[str], list[str]]:
    """(serious, benign) for a pool scored against its matched controls."""
    return serious(pool_key), chargeable_benign(control_key)


def class_counts(key: Key) -> dict:
    """Counts per evidence class, plus the view denominators."""
    c = Counter(k.evidence_class for k in key.values())
    return {**{k: c.get(k, 0) for k in (akb.SERIOUS_TRUTH, akb.SERIOUS_DXA_ONLY, akb.BENIGN, akb.MIDDLE)},
            "chargeable_benign": len(chargeable_benign(key)),
            "chargeable_benign_strict": len(chargeable_benign(key, strict=True)),
            "exempt_benign": len(exempt_benign(key))}
