from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

class ActionSchema(BaseModel):
    action_type: Literal["SearchPolicy", "RequestDocument", "AnalyzeFraud", "ApproveClaim", "RejectClaim"]
    policy_id: Optional[str] = Field(None, description="Required if action_type is SearchPolicy")
    document_type: Optional[str] = Field(None, description="Required if action_type is RequestDocument (e.g. 'Police Report')")
    claim_id: Optional[str] = Field(None, description="Required if action_type is AnalyzeFraud")
    reason: Optional[str] = Field(None, description="Required if action_type is ApproveClaim or RejectClaim")

class ObservationSchema(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    done: bool = False
