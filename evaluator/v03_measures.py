"""
Per-case helpers for the v0.3 reassurance variant H' and the extended consistency
check (review fix 4, docs/v0.3-adversarial-review.md section 4.3). The scorer that
aggregates them over cases comes later; these functions judge one parsed answer.

- `reassures`: the answer says nothing is serious: NO, or unreadable.
- `reassures_prime` (H'): as `reassures`, and also a YES that flags no tier-1
  condition, because "YES, watch for sinusitis" gives the clinician nothing to act on.
- `consistency_issues`: YES with no flags, YES with no tier-1 flag, and NO while
  flagging a tier-1 condition.

Tier-1 flags are matched with evaluator/condition_match.py under its default
(standard) policy unless the caller passes another.
"""

from __future__ import annotations

from typing import Mapping

from evaluator.condition_match import DEFAULT_POLICY, FlagMatcher
from evaluator.schemas_v03 import ParsedV03

YES_WITHOUT_FLAGS = "yes_without_flags"
YES_WITHOUT_TIER1_FLAG = "yes_without_tier1_flag"
NO_WITH_TIER1_FLAG = "no_with_tier1_flag"


def tier1_flagged(parsed: ParsedV03, matcher: FlagMatcher, tiers: Mapping[str, int],
                  policy: str = DEFAULT_POLICY) -> set[str]:
    """The tier-1 conditions the answer's flags match."""
    return matcher.tier1_hit(parsed.flags, tiers, policy)


def reassures(parsed: ParsedV03) -> bool:
    """NO, or unreadable."""
    return parsed.yes is not True


def reassures_prime(parsed: ParsedV03, tier1_hit: set[str]) -> bool:
    """H': NO, unreadable, or YES with no tier-1 flag."""
    return reassures(parsed) or not tier1_hit


def consistency_issues(parsed: ParsedV03, tier1_hit: set[str]) -> list[str]:
    """The consistency tags that apply to one readable answer (none for an unreadable one)."""
    if parsed.yes is None:
        return []
    out = []
    if parsed.yes:
        if not parsed.flags:
            out.append(YES_WITHOUT_FLAGS)
        if not tier1_hit:
            out.append(YES_WITHOUT_TIER1_FLAG)
    elif tier1_hit:
        out.append(NO_WITH_TIER1_FLAG)
    return out
