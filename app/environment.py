import random
from typing import Dict, Any, Optional
from .models import ActionSchema, ObservationSchema
from .tasks import TASKS, grade_task
from .data_loader import load_claims_dataset

class InsuranceEnvironment:
    def __init__(self, use_real_data: bool = False, data_path: Optional[str] = None):
        self.use_real_data = use_real_data
        self.real_claims = []
        if self.use_real_data:
            if not data_path:
                raise ValueError("data_path is required when use_real_data=True")
            self.real_claims = load_claims_dataset(data_path)
        self.reset("easy")

    def _select_real_claim(self, task_name: str) -> Dict[str, Any]:
        if not self.real_claims:
            raise ValueError("No real claim records loaded.")

        if task_name in {"easy", "medium", "hard"}:
            matching = [record for record in self.real_claims if record.get("difficulty") == task_name]
            if matching:
                return random.choice(matching).copy()

        return random.choice(self.real_claims).copy()

    def reset(self, task_name: str = "easy") -> ObservationSchema:
        if self.use_real_data:
            self.task_data = self._select_real_claim(task_name)
            self.task_name = self.task_data.get("difficulty", "real")
        else:
            if task_name not in TASKS:
                raise ValueError(f"Task {task_name} not found.")

            self.task_name = task_name
            self.task_data = TASKS[task_name].copy()
        
        self.current_state = {
            "claim_id": self.task_data["claim_id"],
            "description": self.task_data["description"],
            "documents_received": self.task_data["documents_provided"].copy(),
            "policy_verified": False,
            "fraud_analyzed": False,
            "missing_documents_requested": False,
            "last_action_type": None,
        }

        self.action_history = []
        
        self.done = False
        self.step_count = 0
        self.max_steps = 10
        self.reward = 0.0

        return ObservationSchema(
            message=f"Environment reset to task '{self.task_name}'. You have a new claim to process. Claim ID: {self.current_state['claim_id']}. Description: {self.current_state['description']}",
            data={"claim_id": self.current_state["claim_id"], "documents_received": self.current_state["documents_received"]},
            reward=0.0,
            done=False,
            info={"task": self.task_name, "grader_score": 0.0, "mode": "real" if self.use_real_data else "synthetic"},
        )

    def step(self, action: ActionSchema) -> ObservationSchema:
        if self.done:
            return ObservationSchema(message="Episode is already done. Please reset.", data=self.current_state, reward=0.0, done=True)
            
        self.step_count += 1
        self.current_state["last_action_type"] = action.action_type
        self.action_history.append(action.action_type)
        reward = 0.0
        message = ""
        done = False
        
        # Action Logic
        if action.action_type == "SearchPolicy":
            if action.policy_id == self.task_data["policy_id"]:
                self.current_state["policy_verified"] = True
                message = f"Policy {action.policy_id} is {self.task_data['policy_status']} with coverage: {', '.join(self.task_data['policy_coverage'])}"
                reward += 0.1 # Partial reward
            else:
                message = f"Policy {action.policy_id} not found."
                reward -= 0.1
                
        elif action.action_type == "RequestDocument":
            if action.document_type in self.task_data["required_documents"] and action.document_type not in self.current_state["documents_received"]:
                self.current_state["missing_documents_requested"] = True
                message = f"Document '{action.document_type}' requested successfully and has been uploaded to the claim."
                self.current_state["documents_received"].append(action.document_type)
                reward += 0.2 # Good partial reward
            elif action.document_type in self.current_state["documents_received"]:
                message = f"Document '{action.document_type}' is already on file."
            else:
                message = f"Document '{action.document_type}' is not a valid document or cannot be retrieved."
                reward -= 0.05
                
        elif action.action_type == "AnalyzeFraud":
            if action.claim_id == self.task_data["claim_id"]:
                self.current_state["fraud_analyzed"] = True
                message = self.task_data["fraud_analysis_result"]
                reward += 0.1
            else:
                message = f"Claim {action.claim_id} not found."
                
        elif action.action_type == "ApproveClaim":
            done = True
            missing_docs = set(self.task_data["required_documents"]) - set(self.current_state["documents_received"])
            
            if not self.current_state["policy_verified"]:
                message = "CRITICAL ERROR: Claim approved without verifying the policy. Claim processed incorrectly."
                reward += 0.0
            elif len(missing_docs) > 0:
                message = f"CRITICAL ERROR: Claim approved while missing required documents: {missing_docs}. Claim processed incorrectly."
                reward += 0.0
            elif self.task_data["fraud_flags"]:
                message = "CRITICAL ERROR: Fraudulent claim was approved. Major financial loss."
                reward += 0.0
            else:
                message = "SUCCESS: Claim approved correctly. All checks passed."
                reward += 1.0 # Max reward
                
        elif action.action_type == "RejectClaim":
            done = True
            if self.task_data["fraud_flags"]:
                if self.current_state["fraud_analyzed"]:
                    message = "SUCCESS: Fraudulent claim correctly identified and rejected."
                    reward += 1.0
                else:
                    message = "LUCKY SUCCESS: Claim rejected correctly, but fraud analysis was never run. Bad practice."
                    reward += 0.5
            else:
                # Rejecting a valid claim
                message = "ERROR: Valid claim was rejected without cause."
                reward += 0.0

        if self.step_count >= self.max_steps and not done:
            done = True
            message = "Max steps reached without a final decision."

        grader_score = grade_task(
            self.task_name,
            self.task_data,
            self.current_state,
            self.action_history,
            done,
        )

        if done:
            reward = grader_score

        self.done = done
        self.reward = max(0.0, min(1.0, reward))
        
        return ObservationSchema(
            message=message,
            data=self.current_state,
            reward=reward,
            done=done,
            info={
                "task": self.task_name,
                "grader_score": grader_score,
                "steps_used": self.step_count,
            },
        )

    def state(self) -> ObservationSchema:
        return ObservationSchema(
            message="Current environment state",
            data=self.current_state,
            reward=self.reward,
            done=self.done,
            info={
                "task": self.task_name,
                "steps_used": self.step_count,
                "history": self.action_history,
            },
        )

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_data": self.task_data,
            "current_state": self.current_state,
            "done": self.done,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "reward": self.reward,
            "action_history": self.action_history,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "InsuranceEnvironment":
        env = cls()
        env.task_name = snapshot["task_name"]
        env.task_data = snapshot["task_data"]
        env.current_state = snapshot["current_state"]
        env.done = snapshot["done"]
        env.step_count = snapshot["step_count"]
        env.max_steps = snapshot["max_steps"]
        env.reward = snapshot["reward"]
        env.action_history = snapshot.get("action_history", [])
        return env
