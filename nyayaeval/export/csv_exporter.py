"""
nyayaeval.export.csv_exporter — CSV Serialization
====================================================

Exports verified pipeline data as CSV for spreadsheet analysis,
reporting, and compatibility with tools that don't support JSONL.

The CSV format flattens the hierarchical entity structure into a
tabular layout suitable for pandas/Excel consumption.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import structlog

from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)

# Column headers for the flattened CSV export
CSV_HEADERS = [
    "document_id", "source_language", "case_number", "court_name",
    "adapted_text_preview", "faithfulness", "context_recall",
    "legal_consistency", "is_verified", "retry_count",
]


async def export_to_csv(state: NyayaEvalState, output_path: Path) -> Path:
    """
    Serialize a verified pipeline state to a CSV file.

    Appends a row to the output file. Creates the file with headers
    if it doesn't exist.

    Args:
        state: Verified pipeline state to export.
        output_path: Path to the output .csv file.

    Returns:
        The output file path.

    TODO (Phase 2):
        - Extract case_number from entities
        - Serialize full evaluation scores
    """
    document_id = state.get("document_id", "unknown")
    logger.info("export.csv.start", document_id=document_id, output=str(output_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()

        adapted_text = state.get("adapted_text", "")
        row: dict[str, Any] = {
            "document_id": document_id,
            "source_language": state.get("source_language", ""),
            "case_number": "",  # Phase 2: extract from entities
            "court_name": "",
            "adapted_text_preview": adapted_text[:200] if adapted_text else "",
            "faithfulness": 0.0,
            "context_recall": 0.0,
            "legal_consistency": 0.0,
            "is_verified": state.get("is_verified", False),
            "retry_count": state.get("retry_count", 0),
        }
        writer.writerow(row)

    logger.info("export.csv.complete", document_id=document_id)
    return output_path
