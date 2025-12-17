"""
Convert DDXPlus CSV format to JSON format expected by cases.py.

DDXPlus CSV columns:
- AGE
- DIFFERENTIAL_DIAGNOSIS (JSON string: list of [condition_name, probability])
- SEX
- PATHOLOGY (actual diagnosis)
- EVIDENCES (JSON string: list of evidence IDs)
- INITIAL_EVIDENCE
"""

import json
import csv
import ast
from pathlib import Path

def convert_csv_to_json(csv_path: Path, output_path: Path):
    """Convert DDXPlus CSV to JSON format."""
    cases = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Parse differential diagnosis (Python list syntax, may be quoted in CSV)
            try:
                diff_str = row['DIFFERENTIAL_DIAGNOSIS'].strip()
                # Remove outer quotes if present
                if diff_str.startswith('"') and diff_str.endswith('"'):
                    diff_str = diff_str[1:-1].replace('""', '"')  # Handle escaped quotes
                # Use ast.literal_eval for Python list syntax
                diff_diag = ast.literal_eval(diff_str)
            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"Warning: Failed to parse DIFFERENTIAL_DIAGNOSIS for row {idx}: {e}")
                diff_diag = []
            
            # Parse evidences (Python list syntax, may be quoted in CSV)
            try:
                ev_str = row['EVIDENCES'].strip()
                # Remove outer quotes if present
                if ev_str.startswith('"') and ev_str.endswith('"'):
                    ev_str = ev_str[1:-1].replace('""', '"')  # Handle escaped quotes
                # Use ast.literal_eval for Python list syntax
                evidences = ast.literal_eval(ev_str)
            except Exception as e:
                if idx < 5:  # Only print first few errors
                    print(f"Warning: Failed to parse EVIDENCES for row {idx}: {e}")
                evidences = []
            
            # Build diagnoses list (top 3 from differential)
            diagnoses = []
            for condition_name, prob in diff_diag[:3]:
                diagnoses.append({
                    "condition_name": condition_name,
                    "probability": prob
                })
            
            # Build case structure
            case = {
                "id": idx,
                "age": int(row['AGE']),
                "sex": row['SEX'],
                "symptoms": evidences,  # Evidence IDs as symptoms
                "diagnoses": diagnoses,
                "pathology": row['PATHOLOGY'],  # Actual diagnosis
                "initial_evidence": row.get('INITIAL_EVIDENCE', ''),
                "duration": "unknown"  # Not in CSV
            }
            
            cases.append(case)
    
    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cases, f, indent=2)
    
    print(f"Converted {len(cases)} cases from CSV to JSON")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "ddxplus_v0"
    csv_path = base_dir / "release_test_patients"
    output_path = base_dir / "cases.json"
    
    convert_csv_to_json(csv_path, output_path)

