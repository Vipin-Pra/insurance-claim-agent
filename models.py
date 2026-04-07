from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, Literal

class ActionSchema(BaseModel):
    action_type: Literal["SearchPolicy", "RequestDocument", "AnalyzeFraud", "ApproveClaim", "RejectClaim"]
    policy_id: Optional[str] = Field(None, description="Required if action_type is SearchPolicy")
    document_type: Optional[str] = Field(None, description="Required if action_type is RequestDocument (e.g. 'Police Report')")
    claim_id: Optional[str] = Field(None, description="Required if action_type is AnalyzeFraud")
    reason: Optional[str] = Field(None, description="Required if action_type is ApproveClaim or RejectClaim")

    @model_validator(mode="after")
    def validate_action_fields(self):
        required_fields = {
            "SearchPolicy": "policy_id",
            "RequestDocument": "document_type",
            "AnalyzeFraud": "claim_id",
            "ApproveClaim": "reason",
            "RejectClaim": "reason",
        }

        required_field = required_fields[self.action_type]
        if not getattr(self, required_field):
            raise ValueError(f"{required_field} is required for action_type '{self.action_type}'.")

        return self

class ObservationSchema(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    done: bool = False
    info: Optional[Dict[str, Any]] = None


class RewardSchema(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    details: Optional[Dict[str, Any]] = None
