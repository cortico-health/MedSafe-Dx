
from typing import List, Literal
from pydantic import BaseModel, Field, validator


class Diagnosis(BaseModel):
    code: str = Field(..., description="ICD-10 code")


class ModelPrediction(BaseModel):
    case_id: str
    differential_diagnoses: List[Diagnosis]
    escalation_decision: Literal[
        "ESCALATE_NOW", "ROUTINE_CARE"
    ]
    uncertainty: Literal["CONFIDENT", "UNCERTAIN"]

    @validator("differential_diagnoses")
    def must_have_five_diagnoses(cls, v):
        if len(v) != 5:
            raise ValueError("Exactly 5 diagnoses are required")
        return v


class GoldCase(BaseModel):
    case_id: str
    gold_top3: List[str]
    escalation_required: bool
    uncertainty_acceptable: bool
