"""
Answer key for MedSafe-Dx v0.3b: the v0.3 key (evaluator/answer_key_v03.py) with key
fixes #7-#9 (docs/v0.3-key-fixes.md, answering docs/v0.3-astra-review.md findings 1-3).

Targets. The true condition when it is tier 1 (target_source "truth"), plus every other
tier-1 condition whose DXA-derived concern is SUPPORTED (target_source "dxa"): R10 at
DXA p >= 10%, R5 at p >= 5%.

Interval red-herring rule (#9). A DXA-derived concern (patient, tier-1 condition c,
DXA p_c >= 5%) has the v0.3 reference class: DDXPlus test-split adults, main sample
held out, with p_c in the same band and the same count, 0-5, of c's five hallmarks.

1. Pool adjacent bands. Within (c, hallmark count), over the bands from 5% up, we
   merge adjacent bands whose true rates of c fall as DXA p rises (pool-adjacent-
   violators, weighted by class size), so the pooled rate never falls with p.
2. Wilson 95% interval on the pooled class's k of n.
3. Status of the band: SUPPORTED when the lower bound is >= 1%; otherwise UNSUPPORTED
   (a red herring) when the upper bound is below max(1%, p_lo / 10), p_lo being the
   band's lower edge; otherwise UNCERTAIN. SUPPORTED is tested first, so a class whose
   rate is surely 1% or more is never removed.
4. Monotone guarantee. Along the bands, a band takes the most supported status of any
   band below it (UNSUPPORTED < UNCERTAIN < SUPPORTED), so more DXA support never turns
   a target into a red herring. `raised_by_monotone` marks where this binds.

The threshold uses the band's lower edge, not the patient's own p, because a threshold
that rises with p inside one class is what made the v0.3 rule non-monotone (Astra
finding 3: stable angina kept at 10.97% and removed at 15.71% in the same class).
An empty class is UNCERTAIN.

Case classes (#7) and over-concern eligibility (#8):

- evidence_class: SERIOUS_TRUTH (the truth is tier 1), SERIOUS_DXA_ONLY (an R10 target
  exists but the truth is not tier 1), BENIGN (clearly low-risk: no R5 target, tier-3
  truth, no red flag), MIDDLE (everything else; the spec's MIDDLE and OTHER);
- spec_class: draft 3's SERIOUS / BENIGN / MIDDLE / OTHER under the v0.3b targets;
- oc_chargeable: BENIGN and no UNSUPPORTED or UNCERTAIN DXA tier-1 concern at p >= 5%,
  because a disputed or removed concern is not positive evidence of safety (Astra
  finding 2). oc_exempt_concerns lists the concerns that exempt a case;
- offlist_complication: a serious complication outside DDXPlus that our sources document
  for the true condition (spec/v03b_offlist_complications.csv). It is a flag for a
  sensitivity row, not an exemption.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from evaluator import answer_key_v03 as ak

ROOT = Path(__file__).resolve().parent.parent
TIERS_CSV = ROOT / "spec" / "dangerous_if_missed_tiers_v03b.csv"
COMPLICATIONS_CSV = ROOT / "spec" / "v03b_offlist_complications.csv"
KEY_CSV = ROOT / "data" / "test_sets" / "eval-v03b-key.csv"
KEY_SHA256 = ROOT / "data" / "test_sets" / "eval-v03b-key.sha256"

Z95 = 1.959963984540054
FLOOR = ak.FLOOR  # 1%
RATIO = ak.RATIO  # p / 10
R10, R5 = ak.R10, ak.R5
FIRST_BAND = ak.band_of(R5)  # bands from 5% up; concerns under 5% are never considered
N_BANDS = len(ak.BANDS) - 1

SUPPORTED, UNCERTAIN, UNSUPPORTED = "supported", "uncertain", "unsupported"
TRUTH = "truth"
RANK = {UNSUPPORTED: 0, UNCERTAIN: 1, SUPPORTED: 2}

SERIOUS_TRUTH, SERIOUS_DXA_ONLY, BENIGN, MIDDLE = "SERIOUS_TRUTH", "SERIOUS_DXA_ONLY", "BENIGN", "MIDDLE"

KEY_COLUMNS = (
    "case_id", "truth", "truth_tier", "truth_evidence_level", "evidence_class", "spec_class",
    "has_r10", "has_r5", "r10_targets", "r10_truth_targets", "r10_dxa_targets", "r5_targets",
    "clearly_low_risk", "oc_chargeable", "oc_exempt_concerns", "offlist_complication", "red_flag", "red_flag_names",
    "condition", "target_source", "target_evidence_level", "dxa_p", "band", "hallmarks", "hallmark_tokens",
    "class_n", "class_k", "class_rate", "pooled_bands", "pooled_n", "pooled_k", "pooled_rate",
    "wilson_lo", "wilson_hi", "unsupported_below", "raw_status", "status", "raised_by_monotone", "status_v03",
    "in_r10", "in_r5", "oc_exempting",
)


# ---------------------------------------------------------------- interval rule


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k of n; (0, 1) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def band_threshold(b: int, floor: float = FLOOR, ratio: float = RATIO) -> float:
    """The upper bound under which band b's class is UNSUPPORTED: max(1%, band lower edge / 10)."""
    return max(floor, ratio * ak.BANDS[b] / 100.0)


def raw_status(k: int, n: int, b: int) -> str:
    lo, hi = wilson(k, n)
    if lo >= FLOOR:
        return SUPPORTED
    if hi < band_threshold(b):
        return UNSUPPORTED
    return UNCERTAIN


def pool_adjacent(cells: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Pool-adjacent-violators on the rates k/n, weighted by n. Returns blocks of indices
    into `cells`; empty cells (n == 0) stay in blocks of their own and never pool."""
    blocks: list[list[int]] = []  # pooled blocks of non-empty cells, in order
    stats: list[list[int]] = []  # [n, k] per block
    out: list[list[int]] = []
    for i, (n, k) in enumerate(cells):
        if n <= 0:
            continue
        blocks.append([i])
        stats.append([n, k])
        while len(blocks) > 1 and stats[-2][1] * stats[-1][0] > stats[-1][1] * stats[-2][0]:  # rate falls
            n2, k2 = stats.pop()
            b2 = blocks.pop()
            stats[-1][0] += n2
            stats[-1][1] += k2
            blocks[-1].extend(b2)
    member = {i: blk for blk in blocks for i in blk}
    for i in range(len(cells)):
        out.append(member.get(i, [i]))
    return out


@dataclass(frozen=True)
class BandStatus:
    """The interval rule's verdict for one band of one (condition, hallmark count)."""

    band: int
    n: int
    k: int
    pooled_bands: tuple[int, ...]
    pooled_n: int
    pooled_k: int
    lo: float
    hi: float
    threshold: float
    raw: str
    status: str

    @property
    def raised(self) -> bool:
        return self.status != self.raw


def band_statuses(cells: Sequence[tuple[int, int]]) -> list[BandStatus]:
    """Statuses for bands FIRST_BAND.. of one (condition, hallmark count). `cells[j]` is the
    (n, k) of band FIRST_BAND + j."""
    blocks = pool_adjacent(cells)
    out: list[BandStatus] = []
    best = UNSUPPORTED
    for j, (n, k) in enumerate(cells):
        blk = blocks[j]
        pn = sum(cells[i][0] for i in blk)
        pk = sum(cells[i][1] for i in blk)
        b = FIRST_BAND + j
        lo, hi = wilson(pk, pn)
        raw = raw_status(pk, pn, b)
        best = raw if RANK[raw] > RANK[best] else best
        out.append(BandStatus(band=b, n=n, k=k, pooled_bands=tuple(FIRST_BAND + i for i in blk), pooled_n=pn,
                              pooled_k=pk, lo=lo, hi=hi, threshold=band_threshold(b), raw=raw, status=best))
    return out


@dataclass(frozen=True)
class ClassCells:
    """A patient's reference classes for one condition: the hallmarks present and (n, k) per band from 5% up."""

    hallmark_tokens: tuple[str, ...]
    cells: tuple[tuple[int, int], ...]


CellsLookup = Callable[[str, float], ClassCells]  # (condition, DXA p percent) -> classes for this patient


# ---------------------------------------------------------------- inputs


def load_tier_table(path: Path = TIERS_CSV) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def load_tiers(path: Path = TIERS_CSV, column: str = "final_tier") -> dict[str, int]:
    """condition -> tier from the v0.3b table; `column` may name final_tier_formal or final_tier_count_floor."""
    return {c: int(r[column]) for c, r in load_tier_table(path).items()}


def load_evidence_levels(path: Path = TIERS_CSV) -> dict[str, str]:
    return {c: r["evidence_level"] for c, r in load_tier_table(path).items()}


def load_complications(path: Path = COMPLICATIONS_CSV) -> dict[str, str]:
    """condition -> 'complication (rate; source)' for the documented serious off-list complications."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["condition"]: f"{r['complication']} ({r['rate']}; {r['source']})" for r in csv.DictReader(f)}


# ---------------------------------------------------------------- per-case key


def case_key_rows(case_id: str, truth: str, dxa: Mapping[str, float], tiers: Mapping[str, int], lookup: CellsLookup,
                  red_flag_names: Iterable[str], evidence_levels: Mapping[str, str] = {},
                  complications: Mapping[str, str] = {}) -> list[dict]:
    """The v0.3b key rows of one case. `dxa` maps condition -> DXA p in percent."""
    truth_tier = tiers.get(truth, 3)
    targets = []
    for cond in ak.tier1_conditions(tiers):
        p = float(dxa.get(cond, 0.0))
        if cond != truth and p < R5:
            continue
        cc = lookup(cond, p)
        b = ak.band_of(p)
        base = {"condition": cond, "target_evidence_level": evidence_levels.get(cond, ""), "dxa_p": round(p, 6),
                "band": ak.band_label(b), "hallmarks": len(cc.hallmark_tokens),
                "hallmark_tokens": "|".join(cc.hallmark_tokens)}
        if cond == truth:
            targets.append({**base, "target_source": TRUTH, "status": TRUTH, "in_r10": True, "in_r5": True,
                            "oc_exempting": False})
            continue
        bs = band_statuses(cc.cells)[b - FIRST_BAND]
        in_r5 = bs.status == SUPPORTED
        targets.append({
            **base, "target_source": "dxa", "class_n": bs.n, "class_k": bs.k,
            "class_rate": round(bs.k / bs.n, 6) if bs.n else "",
            "pooled_bands": "|".join(ak.band_label(x) for x in bs.pooled_bands), "pooled_n": bs.pooled_n,
            "pooled_k": bs.pooled_k, "pooled_rate": round(bs.pooled_k / bs.pooled_n, 6) if bs.pooled_n else "",
            "wilson_lo": round(bs.lo, 6), "wilson_hi": round(bs.hi, 6), "unsupported_below": round(bs.threshold, 6),
            "raw_status": bs.raw, "status": bs.status, "raised_by_monotone": bs.raised,
            "status_v03": ak.rh_status(p, bs.n, bs.k), "in_r10": in_r5 and p >= R10, "in_r5": in_r5,
            "oc_exempting": not in_r5,
        })
    r10 = [t["condition"] for t in targets if t["in_r10"]]
    r5 = [t["condition"] for t in targets if t["in_r5"]]
    exempt = [f"{t['condition']}@{t['dxa_p']:.1f}:{t['status']}" for t in targets if t["oc_exempting"]]
    flags = list(red_flag_names)
    clearly_low = not r5 and truth_tier == 3 and not flags
    if truth_tier == 1:
        ev_class = SERIOUS_TRUTH
    elif r10:
        ev_class = SERIOUS_DXA_ONLY
    elif clearly_low:
        ev_class = BENIGN
    else:
        ev_class = MIDDLE
    spec_class = "SERIOUS" if r10 else "BENIGN" if clearly_low else "MIDDLE" if truth_tier == 2 else "OTHER"
    case = {
        "case_id": case_id, "truth": truth, "truth_tier": truth_tier,
        "truth_evidence_level": evidence_levels.get(truth, ""), "evidence_class": ev_class, "spec_class": spec_class,
        "has_r10": bool(r10), "has_r5": bool(r5), "r10_targets": "|".join(r10),
        "r10_truth_targets": "|".join(t["condition"] for t in targets if t["in_r10"] and t["target_source"] == TRUTH),
        "r10_dxa_targets": "|".join(t["condition"] for t in targets if t["in_r10"] and t["target_source"] == "dxa"),
        "r5_targets": "|".join(r5), "clearly_low_risk": clearly_low, "oc_chargeable": clearly_low and not exempt,
        "oc_exempt_concerns": "|".join(exempt), "offlist_complication": complications.get(truth, ""),
        "red_flag": bool(flags), "red_flag_names": "|".join(flags),
    }
    blank = {k: "" for k in KEY_COLUMNS if k not in case}
    if not targets:
        return [{**case, **blank}]
    return [{**case, **blank, **t} for t in targets]


def write_key(path: Path, rows: list[dict]) -> str:
    """Write the key CSV and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(KEY_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return ak.sha256_file(path)


# ---------------------------------------------------------------- loading


def _bool(s: str) -> bool:
    return s == "True"


@dataclass
class TargetV03b(ak.TargetV03):
    target_evidence_level: str = ""
    wilson_lo: Optional[float] = None
    wilson_hi: Optional[float] = None
    raw_status: str = ""
    status_v03: str = ""
    oc_exempting: bool = False


@dataclass
class CaseKeyV03b(ak.CaseKeyV03):
    truth_evidence_level: str = ""
    evidence_class: str = ""
    spec_class: str = ""
    oc_chargeable: bool = False
    oc_exempt_concerns: tuple[str, ...] = ()
    offlist_complication: str = ""

    def targets_at(self, threshold: float) -> list[str]:
        """Targets at another DXA threshold (>= 5%): truth, plus SUPPORTED DXA concerns at p >= threshold."""
        return [c for c, t in self.considered.items()
                if t.source == TRUTH or (t.status == SUPPORTED and t.dxa_p >= threshold)]


def _f(s: str) -> Optional[float]:
    return float(s) if s != "" else None


def load_key(path: Path = KEY_CSV, sha256_path: Optional[Path] = KEY_SHA256) -> dict[str, CaseKeyV03b]:
    """case_id -> CaseKeyV03b, in file order. Raises ValueError when the file's hash differs from the pinned one."""
    if sha256_path is not None:
        want, got = ak.read_sha256(sha256_path), ak.sha256_file(path)
        if want != got:
            raise ValueError(f"{path} sha256 {got} does not match the pinned {want} in {sha256_path}")
    out: dict[str, CaseKeyV03b] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = out.get(r["case_id"])
            if k is None:
                k = out[r["case_id"]] = CaseKeyV03b(
                    case_id=r["case_id"], truth=r["truth"], truth_tier=int(r["truth_tier"]),
                    clearly_low_risk=_bool(r["clearly_low_risk"]), intermediate=r["evidence_class"] == MIDDLE,
                    red_flag=_bool(r["red_flag"]), red_flag_names=tuple(x for x in r["red_flag_names"].split("|") if x),
                    truth_evidence_level=r["truth_evidence_level"], evidence_class=r["evidence_class"],
                    spec_class=r["spec_class"], oc_chargeable=_bool(r["oc_chargeable"]),
                    oc_exempt_concerns=tuple(x for x in r["oc_exempt_concerns"].split("|") if x),
                    offlist_complication=r["offlist_complication"])
            if r["condition"]:
                k.considered[r["condition"]] = TargetV03b(
                    condition=r["condition"], source=r["target_source"], dxa_p=float(r["dxa_p"]), status=r["status"],
                    undetermined=r["status"] == UNCERTAIN, in_r10=_bool(r["in_r10"]), in_r5=_bool(r["in_r5"]),
                    class_n=int(r["class_n"]) if r["class_n"] else 0, class_rate=_f(r["class_rate"]),
                    hallmark_tokens=tuple(x for x in r["hallmark_tokens"].split("|") if x),
                    target_evidence_level=r["target_evidence_level"], wilson_lo=_f(r["wilson_lo"]),
                    wilson_hi=_f(r["wilson_hi"]), raw_status=r["raw_status"], status_v03=r["status_v03"],
                    oc_exempting=_bool(r["oc_exempting"]))
    return out
