"""
Configuration and paths for MedSafe-Dx analysis scripts.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
LEADERBOARD_DIR = PROJECT_ROOT / "leaderboard"
ANALYSIS_OUTPUT_DIR = RESULTS_DIR / "analysis"

# Ensure output directory exists
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "results_dir": RESULTS_DIR,
    "artifacts_dir": ARTIFACTS_DIR,
    "leaderboard_dir": LEADERBOARD_DIR,
    "analysis_output_dir": ANALYSIS_OUTPUT_DIR,

    # DDXPlus reference data
    "evidences": DATA_DIR / "ddxplus_v0" / "release_evidences.json",
    "conditions": DATA_DIR / "ddxplus_v0" / "release_conditions.json",
    "icd10_reference": DATA_DIR / "section111_valid_icd10_october2025.xlsx",

    # Test sets
    "test_sets_dir": DATA_DIR / "test_sets",
    "dev_v0": DATA_DIR / "test_sets" / "dev-v0.json",
    "test_v0": DATA_DIR / "test_sets" / "test-v0.json",
    "eval_v0": DATA_DIR / "test_sets" / "eval-v0.json",
    "full_eval_v0": DATA_DIR / "test_sets" / "full-eval-v0.json",
}

# Model display names and metadata
MODELS = {
    "anthropic-claude-haiku-4.5": {
        "display_name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
    },
    "anthropic-claude-sonnet-4.5": {
        "display_name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "openai-gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "provider": "OpenAI",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    },
    "openai-gpt-4.1": {
        "display_name": "GPT-4.1",
        "provider": "OpenAI",
        "cost_per_1k_input": 0.002,
        "cost_per_1k_output": 0.008,
    },
    "openai-gpt-5.2": {
        "display_name": "GPT-5.2",
        "provider": "OpenAI",
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
    },
    "openai-gpt-oss-120b": {
        "display_name": "GPT-OSS 120B",
        "provider": "OpenAI",
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.003,
    },
    "google-gemini-2.5-flash-lite": {
        "display_name": "Gemini 2.5 Flash Lite",
        "provider": "Google",
        "cost_per_1k_input": 0.000075,
        "cost_per_1k_output": 0.0003,
    },
    "deepseek-deepseek-chat-v3-0324": {
        "display_name": "DeepSeek Chat V3",
        "provider": "DeepSeek",
        "cost_per_1k_input": 0.00014,
        "cost_per_1k_output": 0.00028,
    },
}

# Safety failure types
SAFETY_FAILURES = [
    "missed_escalation",
    "overconfident_wrong",
    "unsafe_reassurance",
]

# Effectiveness metrics
EFFECTIVENESS_METRICS = [
    "top1_recall",
    "top3_recall",
    "over_escalation_rate",
]

# Excel workbook column configuration
WORKBOOK_COLUMNS = [
    ("case_id", "Case ID", 15),
    ("age", "Age", 8),
    ("sex", "Sex", 10),
    ("symptoms_decoded", "Presenting Symptoms", 60),
    ("history_decoded", "History/Antecedents", 40),
    ("model_dx1", "Model Dx #1", 40),
    ("model_dx2", "Model Dx #2", 40),
    ("model_dx3", "Model Dx #3", 40),
    ("model_dx4", "Model Dx #4", 35),
    ("model_dx5", "Model Dx #5", 35),
    ("model_escalation", "Model Escalation", 18),
    ("model_uncertainty", "Model Uncertainty", 18),
    ("model_reasoning", "Model Reasoning", 50),
    ("gold_dx1", "Gold Dx #1", 40),
    ("gold_dx2", "Gold Dx #2", 40),
    ("gold_dx3", "Gold Dx #3", 40),
    ("gold_escalation", "Escalation Required", 18),
    ("gold_uncertainty", "Uncertainty OK", 15),
    ("eval_result", "Safety Result", 15),
    ("safety_failures", "Safety Failures", 30),
    ("top1_match", "Top-1 Match", 12),
    ("top3_match", "Top-3 Match", 12),
    ("physician_methodology", "Agree with Methodology?", 22),
    ("physician_assessment", "Agree with Assessment & Scoring?", 28),
    ("physician_comments", "Other Comments", 50),
]
