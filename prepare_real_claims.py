import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

REQUIRED_OUTPUT_FIELDS = [
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


def parse_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_list(value: str, separator: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]


def normalize_row(row: Dict[str, str], separator: str) -> Dict[str, object]:
    record: Dict[str, object] = {
        "claim_id": row.get("claim_id", "").strip(),
        "policy_id": row.get("policy_id", "").strip(),
        "description": row.get("description", "").strip(),
        "policy_status": row.get("policy_status", "Active").strip() or "Active",
        "policy_coverage": parse_list(row.get("policy_coverage", ""), separator),
        "required_documents": parse_list(row.get("required_documents", ""), separator),
        "documents_provided": parse_list(row.get("documents_provided", ""), separator),
        "fraud_flags": parse_bool(row.get("fraud_flags", "false")),
        "fraud_analysis_result": row.get("fraud_analysis_result", "No anomalies detected.").strip()
        or "No anomalies detected.",
    }

    difficulty = row.get("difficulty", "").strip().lower()
    if difficulty in {"easy", "medium", "hard"}:
        record["difficulty"] = difficulty

    missing = [field for field in REQUIRED_OUTPUT_FIELDS if not record.get(field) and record.get(field) != False]
    if missing:
        raise ValueError(f"Missing required output values: {missing}")

    return record


def convert_csv_to_jsonl(input_csv: Path, output_jsonl: Path, separator: str) -> None:
    written = 0
    skipped = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile, output_jsonl.open(
        "w", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        required_input_columns = [
            "claim_id",
            "policy_id",
            "description",
            "policy_coverage",
            "required_documents",
            "documents_provided",
            "fraud_flags",
        ]

        missing_input = [col for col in required_input_columns if col not in (reader.fieldnames or [])]
        if missing_input:
            raise ValueError(f"Input CSV is missing required columns: {missing_input}")

        for row_number, row in enumerate(reader, start=2):
            try:
                record = normalize_row(row, separator)
                outfile.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"Skipping row {row_number}: {exc}")

    print(f"Conversion complete. Wrote {written} records to {output_jsonl}.")
    if skipped:
        print(f"Skipped {skipped} invalid rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert claim CSV exports into JSONL format for real-data RL.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--output-jsonl", default="data/real_claims.jsonl", help="Path to output JSONL file.")
    parser.add_argument(
        "--list-separator",
        default=";",
        help="Separator for list-like CSV fields such as policy_coverage and required_documents.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_jsonl = Path(args.output_jsonl)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    convert_csv_to_jsonl(input_csv, output_jsonl, args.list_separator)


if __name__ == "__main__":
    main()
