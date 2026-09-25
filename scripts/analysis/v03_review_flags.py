"""v0.3 review, part 4: flag leniency.

The spec counts equivalent, narrower, broader and related codes as flag matches. We measure:
1. how many map rows each tier-1 condition has per relation, and which related codes are owned
   (equivalent or narrower) by a DDXPlus condition of a lower tier, or are symptom codes;
2. codes that match more than one tier-1 condition under the lenient rule (one flag, several targets);
3. on the 250 set with the v0.1 top-5 lists: COV and MT under lenient, strict, and two restrictions.

Run: .venv/bin/python scripts/analysis/v03_review_flags.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict

from v03_review_common import (LENIENT, OUT, ROOT, STRICT, Matcher, RedHerring, build_key, fmt, load_adults,
                               load_tiers, sample_ids, score, write_csv)
from v03_review_models import load_rows

MAP_CSV = ROOT / "spec/ddxplus_icd10_map.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tiers = load_tiers()["v03"]
    m = Matcher()
    rows = list(csv.DictReader(open(MAP_CSV, newline="", encoding="utf-8")))
    tier1 = sorted(c for c, t in tiers.items() if t == 1)

    # 1. per-condition relation counts and suspect related codes
    per = defaultdict(Counter)
    for r in rows:
        per[r["condition"]][r["relation"]] += 1
    print("relation rows per tier-1 condition (equivalent/narrower/broader/related):")
    for c in tier1:
        p = per[c]
        print(f"  {c[:40]:40} {p['equivalent']:3} {p['narrower']:3} {p['broader']:3} {p['related']:4}")

    suspect = []
    for r in rows:
        if r["relation"] not in ("related", "broader") or tiers.get(r["condition"]) != 1:
            continue
        code = r["code"]
        owner = m.cmap.owner(code)
        owner_tier = tiers.get(owner) if owner else None
        symptom = bool(re.match(r"^R\d", code))
        if (owner and owner != r["condition"] and (owner_tier or 3) > 1) or symptom:
            suspect.append({"condition": r["condition"], "code": code, "relation": r["relation"], "owner": owner or "",
                            "owner_tier": owner_tier or "", "symptom_code": symptom, "note": r["note"][:100]})
    write_csv(OUT / "flags_suspect_map_rows.csv", suspect)
    print(f"\nlenient rows for tier-1 conditions that are symptom codes or owned by a lower-tier condition: {len(suspect)}")
    print("  by condition:", Counter(s["condition"] for s in suspect).most_common())
    print("  owned by a lower-tier DDXPlus condition:")
    for s in suspect:
        if s["owner"]:
            print(f"    {s['condition'][:30]:30} <- {s['code']:8} ({s['relation']}) owner {s['owner']} tier {s['owner_tier']}")

    # 2. codes matching several tier-1 conditions under the lenient rule
    multi = Counter()
    examples = {}
    for r in rows:
        code = r["code"]
        hit = {c for c, (_, rel) in m.cmap.resolve(code).items() if rel in LENIENT and tiers.get(c) == 1}
        if len(hit) >= 2:
            multi[len(hit)] += 1
            examples.setdefault(len(hit), []).append((code, sorted(hit)))
    print("\nmap codes that match >= 2 tier-1 conditions leniently, by count:", dict(sorted(multi.items())))
    for k in sorted(examples, reverse=True)[:3]:
        for code, hit in examples[k][:4]:
            print(f"  {code}: {hit}")

    # 3. on the 250 set with v0.1 lists
    df = load_adults()
    rh = RedHerring()
    ids = sample_ids("250")
    key = build_key(df, ids, tiers, rh)
    models = load_rows(ids)
    NO_SYMPTOM = tuple(LENIENT)

    class Restricted(Matcher):
        """Lenient, minus symptom codes (R chapter) and codes owned by a lower-tier DDXPlus condition."""

        def __init__(self, base: Matcher, drop_symptom=True, drop_lower_owner=True):
            self.cmap = base.cmap
            self.drop_symptom, self.drop_lower_owner = drop_symptom, drop_lower_owner

        def _ok(self, code, cond, rel):
            if rel in STRICT:
                return True
            if self.drop_symptom and re.match(r"^R\d", code.upper()):
                return False
            if self.drop_lower_owner:
                own = self.cmap.owner(code)
                if own and own != cond and tiers.get(own, 3) > 1:
                    return False
            return True

        def matched(self, codes, cond, relations=LENIENT):
            for c in codes:
                if not c:
                    continue
                rel = self.cmap.relation(c, cond)
                if rel in relations and self._ok(c, cond, rel):
                    return True
            return False

        def conditions_hit(self, codes, relations=LENIENT):
            hit = set()
            for c in codes:
                if not c:
                    continue
                for cond, (_, rel) in self.cmap.resolve(c).items():
                    if rel in relations and self._ok(c, cond, rel):
                        hit.add(cond)
            return hit

    variants = {
        "lenient (spec)": (m, LENIENT),
        "broader only (no related)": (m, ("equivalent", "narrower", "broader")),
        "lenient minus symptom codes and lower-tier-owned codes": (Restricted(m), LENIENT),
        "strict (DX rule)": (m, STRICT),
    }
    out = []
    for name, ans in models.items():
        row = {"model": name}
        for v, (mm, rel) in variants.items():
            s = score(key, ans, mm, rel, tiers=tiers)
            row[f"COV {v}"] = fmt(s["COV"])
            row[f"MT_truth {v}"] = fmt(s["MT_truth"])
            row[f"MT_dxa {v}"] = fmt(s["MT_dxa"])
        out.append(row)
    write_csv(OUT / "flags_models250.csv", out)
    print("\nCOV per model under the matching variants (lenient / broader-only / restricted / strict):")
    for r in sorted(out, key=lambda r: -float(r["COV lenient (spec)"])):
        print(f"  {r['model'][:16]:16} COV {r['COV lenient (spec)']:>5} {r['COV broader only (no related)']:>5} {r['COV lenient minus symptom codes and lower-tier-owned codes']:>5} {r['COV strict (DX rule)']:>5} | "
              f"MT_truth {r['MT_truth lenient (spec)']:>5} {r['MT_truth strict (DX rule)']:>5}")

    # which related/broader codes earned matches, and what they name
    earned = Counter()
    for name, ans in models.items():
        for c, a in zip(key, ans):
            for t in c["R5"]:
                if m.matched(a["flags"], t, STRICT):
                    continue
                for code in a["flags"]:
                    rel = m.cmap.relation(code, t) if code else None
                    if rel in ("broader", "related"):
                        earned[(t, code, rel, m.cmap.owner(code) or "")] += 1
                        break
    print("\nbroader/related codes that earned a target match when no strict code did (target, code, relation, owner; model x case counts):")
    for (t, code, rel, own), n in earned.most_common(25):
        print(f"  {n:4} {t[:28]:28} {code:8} {rel:8} owner={own}")
    write_csv(OUT / "flags_earned_related.csv", [{"target": t, "code": code, "relation": rel, "owner": own, "n": n} for (t, code, rel, own), n in earned.most_common()])


if __name__ == "__main__":
    main()
