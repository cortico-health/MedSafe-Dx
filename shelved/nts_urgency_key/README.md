# Shelved: NTS urgency answer key

NTS is not used in v0.2 (user decision 2026-09-23); kept as research record.

`acuity_key.py` gave each v0.2 patient a Netherlands Triage Standard (NTS) urgency level from spec/acuity_reference_levels.csv, with patient-level modifier rules from docs/triage-scale-anchor.md. Spec revision 4 (spec/v0.2-scoring.md) scores against DDXPlus severity instead, so nothing in v0.2 imports this code.

The section 7 red-flag rules moved to evaluator/answer_key_v02.py, which v0.2 uses. The copy here is frozen.

To run the old tests: `cd shelved/nts_urgency_key && python3 -m unittest test_acuity_key`. The main suite (`python3 -m unittest discover -s evaluator/tests -t .`) does not run them.
