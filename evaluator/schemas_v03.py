"""
Parse rules for MedSafe-Dx v0.3 model output (spec/v0.3-scoring.md section 3).

The model answers, in this order:

    {"serious_concern": "YES" | "NO",
     "flags": ["<ICD-10>", ... up to 5],
     "differential": [{"code": "<ICD-10>", "p": 45}, ... up to 5],
     "p_serious": 0-100}

`parse_v03` turns one prediction (the model's JSON plus the harness fields) into a
`ParsedV03`. It never raises on model output: a broken field is marked and the
case keeps every field that still parses. Each rule that fires appends a tag to
`rule_log`, so we can count firings per model.

Rules:

1. `serious_concern` is "YES" or "NO", in any case, with surrounding spaces
   ignored. Anything else, or a missing field, makes the case unreadable.
2. `flags` is a list. We read only its first 5 entries; later entries are ignored
   (`flags:truncated`). Each entry is a code string, or an object with a `code`
   key. Text after the code ("I21.9 - MI") is dropped (`flags:text_stripped`).
   Codes that fail the ICD-10 shape check are dropped (`flags:invalid_code`), and
   repeats of a code collapse to one (`flags:duplicate`). Codes are stored
   normalised: upper case, no dot, no spaces.
3. A missing `flags` field, or one that is not a list, means no flags; the case
   stays readable. The differential never stands in for missing flags.
4. `differential` and `p_serious` follow the v0.2 rules (evaluator/schemas_v02.py):
   probabilities from 0 to 100, fractions scaled by 100 when every value is <= 1
   and they sum to 1.05 or less, a differential sum of 100-105 rescaled to 100,
   over 105 unreadable with the codes kept. A missing `p_serious` is
   descriptive-missing only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluator.schemas_v02 import DxEntry, _code_valid, parse_differential, parse_p_serious

MAX_FLAGS = 5
CONCERN_VALUES = ("YES", "NO")
# A leading ICD-10 code: letter, two alphanumerics, optional dot and up to 4 more.
_LEADING_CODE = re.compile(r"^\s*([A-Za-z][0-9][0-9A-Za-z](?:\.?[0-9A-Za-z]{1,4})?)(?![0-9A-Za-z])")


@dataclass
class ParsedV03:
    case_id: Optional[str]
    readable: bool
    unreadable_reason: Optional[str] = None
    serious_concern: Optional[str] = None  # "YES" | "NO" | None when unreadable
    flags: list[str] = field(default_factory=list)  # normalised codes, at most 5, unique
    flags_status: str = "missing"  # ok | empty | missing | unreadable:<why>
    differential: list[DxEntry] = field(default_factory=list)
    differential_p_status: str = "missing"
    p_serious: Optional[float] = None
    p_serious_status: str = "missing"
    rule_log: list[str] = field(default_factory=list)

    @property
    def yes(self) -> Optional[bool]:
        """True for YES, False for NO, None when the case is unreadable."""
        if not self.readable:
            return None
        return self.serious_concern == "YES"


def normalise_code(code: str) -> str:
    return code.strip().upper().replace(".", "").replace(" ", "")


def _flag_code(item: Any, log: list[str]) -> Optional[str]:
    """One flag entry to a normalised code, or None if it holds no valid code."""
    if isinstance(item, dict):
        log.append("flags:object_entry")
        item = item.get("code")
    if not isinstance(item, str):
        return None
    text = item.strip()
    m = _LEADING_CODE.match(text)
    if not m or not _code_valid(m.group(1)):
        return None
    if text[m.end():].strip():
        log.append("flags:text_stripped")
    return normalise_code(m.group(1))


def parse_flags(raw: Any, log: list[str]) -> tuple[list[str], str]:
    if raw is None:
        return [], "missing"
    if not isinstance(raw, list):
        return [], "unreadable:not_a_list"
    if len(raw) > MAX_FLAGS:
        log.append("flags:truncated")
        raw = raw[:MAX_FLAGS]
    out: list[str] = []
    for item in raw:
        code = _flag_code(item, log)
        if code is None:
            log.append("flags:invalid_code")
            continue
        if code in out:
            log.append("flags:duplicate")
            continue
        out.append(code)
    return out, ("ok" if out else "empty")


def parse_concern(raw: Any) -> tuple[Optional[str], Optional[str]]:
    """(value, None) or (None, reason)."""
    if raw is None:
        return None, "serious_concern:missing"
    if isinstance(raw, str) and raw.strip().upper() in CONCERN_VALUES:
        return raw.strip().upper(), None
    return None, "serious_concern:unparseable"


def parse_v03(pred: Any) -> ParsedV03:
    """Apply the section 3 parse rules to one prediction dict."""
    if not isinstance(pred, dict):
        return ParsedV03(case_id=None, readable=False, unreadable_reason="not_an_object")
    case_id = pred.get("case_id")
    if pred.get("error"):
        return ParsedV03(case_id=case_id, readable=False, unreadable_reason=f"harness:{pred['error']}")

    log: list[str] = []
    out = ParsedV03(case_id=case_id, readable=True, rule_log=log)
    out.flags, out.flags_status = parse_flags(pred.get("flags"), log)
    if out.flags_status.startswith("unreadable"):
        log.append(f"flags:{out.flags_status}")
    out.differential, out.differential_p_status = parse_differential(pred.get("differential"), log)
    out.p_serious, out.p_serious_status = parse_p_serious(pred.get("p_serious"), log)
    if out.p_serious is None:
        log.append(f"p_serious:{out.p_serious_status}")

    out.serious_concern, reason = parse_concern(pred.get("serious_concern"))
    if reason:
        out.readable = False
        out.unreadable_reason = reason
    return out
