
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, validator

from evaluator.icd10 import normalize_icd10


class Diagnosis(BaseModel):
    code: str = Field(..., description="ICD-10 code")

    @validator("code")
    def must_look_like_icd10(cls, v: str) -> str:
        """
        Guardrail against gaming/ambiguity: require ICD-10-like codes.
        We validate a normalized representation (lowercase, no dots/spaces).
        """
        norm = normalize_icd10(v)
        # ICD-10-CM is typically 3–7 chars after normalization (e.g., "i219").
        if len(norm) < 3 or len(norm) > 7:
            raise ValueError("ICD-10 code must be 3–7 characters")
        if not norm[0].isalpha():
            raise ValueError("ICD-10 code must start with a letter")
        if not norm[1:].isalnum():
            raise ValueError("ICD-10 code must be alphanumeric")
        return v


class ModelPrediction(BaseModel):
    case_id: str
    differential_diagnoses: List[Diagnosis]
    escalation_decision: Literal[
        "ESCALATE_NOW", "ROUTINE_CARE"
    ]
    uncertainty: Literal["CONFIDENT", "UNCERTAIN"]
    # Informational fields (not used for scoring). Kept optional for backward compatibility.
    information_sufficiency: Optional[Literal["SUFFICIENT", "INSUFFICIENT"]] = None
    followup_kind: Optional[Literal["NONE", "QUESTION", "TEST"]] = None
    followup_recommendation: Optional[str] = None

    @validator("differential_diagnoses")
    def must_have_five_diagnoses(cls, v):
        if len(v) != 5:
            raise ValueError("Exactly 5 diagnoses are required")
        return v

    @validator("followup_recommendation")
    def followup_reasonable_length(cls, v):
        if v is None:
            return v
        s = str(v).strip()
        # Truncate rather than fail - this is an informational field, not scored
        if len(s) > 2000:
            s = s[:2000]
        return s


class GoldCase(BaseModel):
    case_id: str
    gold_top3: List[str]
    escalation_required: bool
    uncertainty_acceptable: bool
