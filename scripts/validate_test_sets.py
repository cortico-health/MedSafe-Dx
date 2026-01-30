#!/usr/bin/env python3
"""
Validate MedSafe-Dx test set JSON files for schema consistency and label derivation drift.

Checks:
- Test set JSON structure (metadata + cases)
- Unique case_ids within each file
- Required fields exist and have expected types
- Re-derives escalation_required / uncertainty_acceptable from DDXPlus severity metadata
  (when severity can be resolved for all gold codes) to catch regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _normalize_icd10(code: str) -> str:
    return (code or "").strip().lower().replace(".", "").replace(" ", "")


def _explode_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    for c in codes or []:
        if not c:
            continue
        for part in str(c).split(","):
            n = _normalize_icd10(part)
            if n:
                out.append(n)
    return out


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_severity_index(conditions_path: Path) -> dict[str, int]:
    conditions = _load_json(conditions_path)
    if not isinstance(conditions, dict):
        raise ValueError(f"Expected object in {conditions_path}")

    idx: dict[str, int] = {}
    for _, data in conditions.items():
        if not isinstance(data, dict):
            continue
        code = data.get("icd10-id") or data.get("icd10_code") or data.get("icd10")
        severity = data.get("severity")
        if code is None or severity is None:
            continue
        try:
            sev = int(severity)
        except Exception:
            continue
        # Some DDXPlus entries use multi-code ICD fields like "J17, J18".
        # To make label re-derivation possible, assign the same severity to each component.
        parts = [p.strip() for p in str(code).split(",")] if isinstance(code, str) else [str(code)]
        for part in parts:
            n = _normalize_icd10(part)
            if not n:
                continue
            # If multiple conditions map to the same code, keep the most severe (min).
            idx[n] = min(idx.get(n, sev), sev)
    return idx


def _severity_for_code(code_norm: str, severity_idx: dict[str, int]) -> int | None:
    if not code_norm:
        return None
    if code_norm in severity_idx:
        return severity_idx[code_norm]

    # Common ICD-10 pattern: letter + 2 digits (+ optional suffix). Prefer prefix matches
    # by progressively shortening the code down to the 3-char category.
    for i in range(len(code_norm), 2, -1):
        pref = code_norm[:i]
        if pref in severity_idx:
            return severity_idx[pref]

    # If the provided code is very short, check for more-specific matches.
    # Choose the most severe among candidates to avoid missing a critical mapping.
    candidates = [severity_idx[k] for k in severity_idx.keys() if k.startswith(code_norm)]
    if candidates:
        return min(candidates)
    return None


@dataclass
class FileReport:
    path: Path
    total_cases: int = 0
    duplicate_case_ids: int = 0
    schema_errors: int = 0
    derivation_mismatches: int = 0
    unknown_severity_cases: int = 0
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def validate_test_set_file(
    path: Path,
    severity_idx: dict[str, int],
    max_list: int = 50,
    show_unknown: bool = False,
) -> FileReport:
    report = FileReport(path=path)
    data = _load_json(path)

    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
        metadata = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}
    elif isinstance(data, list):
        cases = data
        metadata = {}
    else:
        report.schema_errors += 1
        report.notes.append("File is neither a list nor an object with 'cases'")
        return report

    if not isinstance(cases, list):
        report.schema_errors += 1
        report.notes.append("'cases' is not a list")
        return report

    report.total_cases = len(cases)

    # Optional seed sanity check for canonical v0 files.
    expected_seed_by_name = {
        "dev-v0.json": 1,
        "test-v0.json": 42,
        "eval-v0.json": 42,
        "full-eval-v0.json": 42,
    }
    expected_seed = expected_seed_by_name.get(path.name)
    if expected_seed is not None:
        actual_seed = metadata.get("seed")
        if actual_seed != expected_seed:
            report.notes.append(f"Seed mismatch: expected {expected_seed}, got {actual_seed!r}")

    seen: set[str] = set()
    for idx, c in enumerate(cases):
        if not isinstance(c, dict):
            report.schema_errors += 1
            if report.schema_errors <= max_list:
                report.notes.append(f"Case[{idx}] is not an object")
            continue

        cid = c.get("case_id")
        if not isinstance(cid, str) or not cid:
            report.schema_errors += 1
            if report.schema_errors <= max_list:
                report.notes.append(f"Case[{idx}] missing/invalid case_id")
            continue

        if cid in seen:
            report.duplicate_case_ids += 1
        seen.add(cid)

        gold_top3 = c.get("gold_top3")
        if not isinstance(gold_top3, list) or not gold_top3:
            report.schema_errors += 1
            if report.schema_errors <= max_list:
                report.notes.append(f"{cid}: missing/invalid gold_top3")
            continue

        esc = c.get("escalation_required")
        unc_ok = c.get("uncertainty_acceptable")
        if not isinstance(esc, bool) or not isinstance(unc_ok, bool):
            report.schema_errors += 1
            if report.schema_errors <= max_list:
                report.notes.append(f"{cid}: escalation_required/uncertainty_acceptable not bool")
            continue

        gold_codes = _explode_codes([str(x) for x in gold_top3])
        severities = []
        for code_norm in gold_codes:
            sev = _severity_for_code(code_norm, severity_idx)
            if sev is None:
                severities = None
                break
            severities.append(sev)

        if severities is None or not severities:
            report.unknown_severity_cases += 1
            if show_unknown and report.unknown_severity_cases <= max_list:
                report.notes.append(
                    f"{cid}: unable to resolve severity for gold_top3={gold_top3}"
                )
            continue

        # Re-derive labels (mirrors spec intent: severity <= 2 => escalation).
        derived_esc = any(sev <= 2 for sev in severities)
        derived_unc_ok = len(severities) > 1 and (max(severities) - min(severities) <= 1)

        if derived_esc != esc or derived_unc_ok != unc_ok:
            report.derivation_mismatches += 1
            if report.derivation_mismatches <= max_list:
                report.notes.append(
                    f"{cid}: label mismatch "
                    f"(esc {esc}->{derived_esc}, unc_ok {unc_ok}->{derived_unc_ok}, severities={severities})"
                )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MedSafe-Dx test set JSON files")
    parser.add_argument(
        "--dir",
        default="data/test_sets",
        help="Directory containing test set JSON files (default: data/test_sets)",
    )
    parser.add_argument(
        "--conditions",
        default="data/ddxplus_v0/release_conditions.json",
        help="Path to DDXPlus release_conditions.json (default: data/ddxplus_v0/release_conditions.json)",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="Max per-file notes to print (default: 50)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0 (print warnings only)",
    )
    parser.add_argument(
        "--show-unknown",
        action="store_true",
        help="Include per-case notes for unknown severity mappings",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    test_dir = (base_dir / args.dir).resolve()
    conditions_path = (base_dir / args.conditions).resolve()

    if not test_dir.exists():
        print(f"ERROR: test dir not found: {test_dir}", file=sys.stderr)
        return 2
    if not conditions_path.exists():
        print(f"ERROR: conditions file not found: {conditions_path}", file=sys.stderr)
        return 2

    severity_idx = _build_severity_index(conditions_path)
    files = sorted(test_dir.glob("*.json"))
    if not files:
        print(f"ERROR: no *.json files in {test_dir}", file=sys.stderr)
        return 2

    any_errors = False
    for f in files:
        report = validate_test_set_file(
            f,
            severity_idx,
            max_list=args.max_list,
            show_unknown=args.show_unknown,
        )
        header = (
            f"{f.name}: cases={report.total_cases}, dup={report.duplicate_case_ids}, "
            f"schema_err={report.schema_errors}, mismatches={report.derivation_mismatches}, "
            f"unknown_sev={report.unknown_severity_cases}"
        )
        print(header)
        for note in report.notes[: args.max_list]:
            print(f"  - {note}")

        if report.schema_errors or report.derivation_mismatches or report.duplicate_case_ids:
            any_errors = True

    if args.warn_only:
        return 0
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
