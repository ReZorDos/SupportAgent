from typing import Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field, create_model


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Route(StrictModel):
    category: Literal["upload", "service", "access", "login", "materials", "grades", "other"]
    reason: str = Field(max_length=500)

class Decision(StrictModel):
    action: Literal["get_upload_rules", "check_file", "get_service_status", "check_course_access", "check_demo_account", "check_material_status", "check_grade_status", "finish"]
    reason: str = Field(max_length=500)

def decision_schema(allowed):
    if not allowed:
        raise ValueError("Пустой список допустимых действий")
    return create_model("StepDecision", __base__=Decision,
                        action=(Literal[tuple(allowed)], ...))

class Draft(StrictModel):
    mode: Literal["solution", "clarification", "escalation"]
    answer: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str]

class Verdict(StrictModel):
    approved: bool
    issues: list[str]

class ClarificationReview(StrictModel):
    requests_missing_information: bool = Field(description="Does the candidate answer ask the user to provide currently unknown diagnostic information? The user need not have supplied it yet.")
    unsupported_claims: list[str] = Field(description="Unsupported factual claims in the candidate ANSWER only; an unknown file size is not a claim.")
    inappropriate_requests: list[str] = Field(description="Requests for secrets or irrelevant, already known information. Requests for the missing format and size are appropriate.")

class SolutionDraft(Draft):
    mode: Literal["solution"]

class ClarificationDraft(Draft):
    mode: Literal["clarification"]

class State(TypedDict, total=False):
    message: str
    history: list[dict]
    memory_context: dict
    fallback_kind: str | None
    course_code: str
    demo_account_code: str
    material_code: str
    submission_code: str
    filename: str
    size_mb: float | None
    category: str
    evidence: list[dict]
    action: str
    tool_calls: int
    draft: dict
    evidence_binding: dict
    verdict: dict
    revisions: int
    final: str
    status: str
