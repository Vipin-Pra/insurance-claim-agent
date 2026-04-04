import json
from typing import Any, Dict, List


REQUIRED_FIELDS = [
    "claim_id",
    "policy_id",
    "description",
    "policy_status",
    "policy_coverage",
    "required_documents",
    "documents_provided",
    "fraud_flags",
    "fraud_analysis_result",
]


def _validate_record(record: Dict[str, Any], line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing fields in record line {line_number}: {missing}")


def load_claims_dataset(path: str) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            _validate_record(record, idx)
            claims.append(record)

    if not claims:
        raise ValueError(f"No valid claim records found in {path}")

    return claims
