#!/usr/bin/env python3
"""Test the critical code matching logic."""

import sys
sys.path.insert(0, '.')

from scripts.analysis.moe_physician_panel import (
    is_critical_diagnosis, 
    check_critical_diagnoses,
    CRITICAL_ICD10_CODES
)

# Test cases from the failed results
test_codes = [
    "K92.0",    # Hematemesis - should match
    "I21.9",    # MI - should match
    "K25.0",    # Gastric ulcer with hemorrhage - should match
    "T78.2",    # Anaphylaxis - should match
    "J02.9",    # Pharyngitis - should NOT match
    "I49.9",    # Cardiac arrhythmia - should match I49.0?
]

print("=== CRITICAL CODES IN LIST ===")
for code in sorted(CRITICAL_ICD10_CODES):
    print(f"  {code}")

print("\n=== TESTING MATCHES ===")
for code in test_codes:
    result = is_critical_diagnosis(code)
    print(f"{code}: {'MATCH' if result else 'no match'}")

print("\n=== CHECK SPECIFIC CASE ===")
case_dx = ["K92.0", "K25.0", "I21.9"]
critical_found = check_critical_diagnoses(case_dx)
print(f"Differential: {case_dx}")
print(f"Critical found: {critical_found}")
