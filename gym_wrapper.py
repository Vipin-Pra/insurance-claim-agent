import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional
from app.environment import InsuranceEnvironment
from app.models import ActionSchema

class InsuranceGymWrapper(gym.Env):
    """
    Custom Environment that follows gym interface to wrap the OpenEnv.
    This simplifies the state and action spaces to be compatible with SB3 PPO.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, use_real_data: bool = False, data_path: Optional[str] = None):
        super(InsuranceGymWrapper, self).__init__()
        self.use_real_data = use_real_data
        self.env = InsuranceEnvironment(use_real_data=use_real_data, data_path=data_path)
        
        # Simplified Action Space
        # 0: SearchPolicy
        # 1: RequestDocument (Police Report)
        # 2: RequestDocument (Purchase Receipt)
        # 3: RequestDocument (Fire Department Report)
        # 4: AnalyzeFraud
        # 5: ApproveClaim
        # 6: RejectClaim
        self.action_space = spaces.Discrete(7)
        
        # Simplified Observation Space (Discrete features)
        # [ policy_verified (0/1), missing_documents_requested (0/1), fraud_analyzed (0/1), has_police_report (0/1), has_receipt (0/1), has_fire_report (0/1) ]
        self.observation_space = spaces.MultiBinary(6)

    def _get_obs(self):
        state = self.env.current_state
        docs = state["documents_received"]
        return np.array([
            int(state["policy_verified"]),
            int(state["missing_documents_requested"]),
            int(state["fraud_analyzed"]),
            int("Police Report" in docs),
            int("Purchase Receipt" in docs),
            int("Fire Department Report" in docs)
        ], dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize task for RL training
        tasks = ["easy", "medium", "hard"]
        task_name = random.choice(tasks)
        self.env.reset(task_name)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        map_action = {
            0: ActionSchema(action_type="SearchPolicy", policy_id=self.env.task_data["policy_id"]),
            1: ActionSchema(action_type="RequestDocument", document_type="Police Report"),
            2: ActionSchema(action_type="RequestDocument", document_type="Purchase Receipt"),
            3: ActionSchema(action_type="RequestDocument", document_type="Fire Department Report"),
            4: ActionSchema(action_type="AnalyzeFraud", claim_id=self.env.task_data["claim_id"]),
            5: ActionSchema(action_type="ApproveClaim", reason="Looks good"),
            6: ActionSchema(action_type="RejectClaim", reason="Risk identified")
        }
        
        act_schema = map_action[int(action)]
        result = self.env.step(act_schema)
        
        obs = self._get_obs()
        reward = result.reward
        done = result.done or self.env.step_count >= self.env.max_steps
        
        # Encourage efficiency
        if not done:
            reward -= 0.01
            
        return obs, reward, done, False, {}

    def render(self):
        print(f"State: {self.env.current_state}")
