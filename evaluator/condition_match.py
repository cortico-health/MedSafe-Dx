"""
Flag matching for MedSafe-Dx v0.3: does a flagged ICD-10 code name a DDXPlus condition?

We read the relation of a code to a condition from spec/ddxplus_icd10_map.csv
through `ConditionMap` (evaluator/v02_score.py), which applies the longest-prefix
rule per condition. A policy names the relations that count as a match:

| Policy | Relations counted | Use |
|---|---|---|
| `standard` | equivalent, narrower, broader | v0.3 flag matching (review fix 3) |
| `lenient` | the above plus related | sensitivity row only |
| `strict` | equivalent, narrower | diagnosis matching (DX) and a sensitivity row |

"Related" rows name a different diagnosis (GERD K21.9 is related to MI), so they
count only in the lenient row, because counting them credits a flag for a
different and often less dangerous condition (docs/v0.3-adversarial-review.md
section 3). Codes off the map match nothing under any policy.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional

from evaluator.v02_score import MAP_CSV, ConditionMap

POLICIES: dict[str, tuple[str, ...]] = {
    "strict": ("equivalent", "narrower"),
    "standard": ("equivalent", "narrower", "broader"),
    "lenient": ("equivalent", "narrower", "broader", "related"),
}
DEFAULT_POLICY = "standard"


def relations_for(policy: str) -> tuple[str, ...]:
    try:
        return POLICIES[policy]
    except KeyError:
        raise ValueError(f"unknown matching policy {policy!r}; use one of {sorted(POLICIES)}") from None


class FlagMatcher:
    """Match flagged codes to DDXPlus conditions under a named policy."""

    def __init__(self, cmap: Optional[ConditionMap] = None, path: Path = MAP_CSV):
        self.cmap = cmap or ConditionMap(path)

    def relation(self, code: str, condition: str) -> Optional[str]:
        return self.cmap.relation(code, condition) if code else None

    def matches(self, code: str, condition: str, policy: str = DEFAULT_POLICY) -> bool:
        return self.relation(code, condition) in relations_for(policy)

    def matched(self, codes: Iterable[str], condition: str, policy: str = DEFAULT_POLICY) -> bool:
        """True when any of `codes` matches `condition`."""
        return any(self.matches(c, condition, policy) for c in codes if c)

    def conditions_hit(self, codes: Iterable[str], policy: str = DEFAULT_POLICY) -> set[str]:
        """Every condition that at least one of `codes` matches."""
        rels = relations_for(policy)
        hit: set[str] = set()
        for c in codes:
            if not c:
                continue
            for cond, (_, rel) in self.cmap.resolve(c).items():
                if rel in rels:
                    hit.add(cond)
        return hit

    def tier1_hit(self, codes: Iterable[str], tiers: Mapping[str, int], policy: str = DEFAULT_POLICY) -> set[str]:
        """The tier-1 conditions that the flags match."""
        return {c for c in self.conditions_hit(codes, policy) if tiers.get(c) == 1}

    def map_codes(self) -> list[str]:
        """Every distinct code in the map, normalised, sorted."""
        return sorted({code for rows in self.cmap.rows.values() for code in rows})

    def multiplicity(self, tiers: Mapping[str, int], policy: str = DEFAULT_POLICY) -> dict[str, set[str]]:
        """For each map code: the tier-1 conditions it matches under `policy` (codes matching none omitted)."""
        out: dict[str, set[str]] = defaultdict(set)
        for code in self.map_codes():
            hit = self.tier1_hit([code], tiers, policy)
            if hit:
                out[code] = hit
        return dict(out)


def multiplicity_summary(matcher: FlagMatcher, tiers: Mapping[str, int]) -> dict[str, dict]:
    """Per policy: map codes that match any tier-1 condition, and those that match more than one."""
    out = {}
    for policy in POLICIES:
        m = matcher.multiplicity(tiers, policy)
        multi = {code: conds for code, conds in m.items() if len(conds) > 1}
        worst = max((len(v) for v in m.values()), default=0)
        out[policy] = {
            "codes_matching_tier1": len(m),
            "codes_matching_more_than_one_tier1": len(multi),
            "max_tier1_per_code": worst,
            "examples": {code: sorted(conds) for code, conds in sorted(multi.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]},
        }
    return out
