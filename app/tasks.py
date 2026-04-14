TASKS = {
    "easy": {
        "claim_id": "CLM-1001",
        "policy_id": "POL-1001",
        "description": "Minor fender bender. Car was rear-ended at a stoplight. All information attached.",
        "policy_status": "Active",
        "policy_coverage": ["Collision", "Comprehensive"],
        "required_documents": ["Police Report", "Photos"],
        "documents_provided": ["Police Report", "Photos"],
        "fraud_flags": False,
        "fraud_analysis_result": "No anomalies detected."
    },
    "medium": {
        "claim_id": "CLM-2002",
        "policy_id": "POL-2002",
        "description": "Stolen laptop from the back seat of the car. Need to claim the value.",
        "policy_status": "Active",
        "policy_coverage": ["Comprehensive", "Personal Property"],
        "required_documents": ["Police Report", "Purchase Receipt"],
        "documents_provided": ["Purchase Receipt"],
        "fraud_flags": False,
        "fraud_analysis_result": "No anomalies detected."
    },
    "hard": {
        "claim_id": "CLM-3003",
        "policy_id": "POL-3003",
        "description": "Kitchen fire caused by unattended stove, significant smoke damage throughout house.",
        "policy_status": "Active",
        "policy_coverage": ["Homeowners", "Fire"],
        "required_documents": ["Fire Department Report", "Photos", "Inventory List"],
        "documents_provided": ["Fire Department Report", "Photos", "Inventory List"],
        "fraud_flags": True,
        "fraud_analysis_result": "HIGH RISK: The fire department report states no stove damage, fire started in the attic. Claim description contradicts official report."
    }
}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def grade_easy(task_data, state, action_history, done: bool) -> float:
    required_docs = set(task_data["required_documents"])
    have_docs = set(state.get("documents_received", []))
    all_docs_present = required_docs.issubset(have_docs)
    approved = state.get("last_action_type") == "ApproveClaim"

    if done:
        if approved and state.get("policy_verified") and all_docs_present and not task_data["fraud_flags"]:
            return 1.0
        return 0.0

    score = 0.0
    score += 0.4 if state.get("policy_verified") else 0.0
    score += 0.6 * _safe_ratio(len(required_docs & have_docs), len(required_docs))
    return max(0.0, min(1.0, score))


def grade_medium(task_data, state, action_history, done: bool) -> float:
    required_docs = set(task_data["required_documents"])
    have_docs = set(state.get("documents_received", []))
    all_docs_present = required_docs.issubset(have_docs)
    approved = state.get("last_action_type") == "ApproveClaim"
    requested_missing_docs = state.get("missing_documents_requested", False)

    if done:
        if approved and state.get("policy_verified") and all_docs_present and requested_missing_docs:
            return 1.0
        return 0.0

    score = 0.0
    score += 0.3 if state.get("policy_verified") else 0.0
    score += 0.4 * _safe_ratio(len(required_docs & have_docs), len(required_docs))
    score += 0.3 if requested_missing_docs else 0.0
    return max(0.0, min(1.0, score))


def grade_hard(task_data, state, action_history, done: bool) -> float:
    rejected = state.get("last_action_type") == "RejectClaim"
    analyzed = state.get("fraud_analyzed", False)

    if done:
        if task_data["fraud_flags"] and rejected and analyzed:
            return 1.0
        if task_data["fraud_flags"] and rejected and not analyzed:
            return 0.6
        return 0.0

    score = 0.0
    score += 0.5 if analyzed else 0.0
    score += 0.2 if state.get("policy_verified") else 0.0
    score += 0.3 if "AnalyzeFraud" in action_history else 0.0
    return max(0.0, min(1.0, score))


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_task(task_name: str, task_data, state, action_history, done: bool) -> float:
    score = TASK_GRADERS[task_name](task_data, state, action_history, done)
    return max(0.01, min(0.99, score))
