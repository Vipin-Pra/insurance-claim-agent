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
