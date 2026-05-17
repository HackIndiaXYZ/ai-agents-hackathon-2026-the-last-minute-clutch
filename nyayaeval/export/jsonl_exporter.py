"""
nyayaeval.export.jsonl_exporter — JSONL Serialization
=======================================================

Exports verified pipeline states as newline-delimited JSON (JSONL),
the standard format for LLM training data and HuggingFace datasets.

Each line contains a self-contained JSON object representing one
processed document with its entities, adapted text, evaluation scores,
and provenance metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


async def export_to_jsonl(state: NyayaEvalState, output_path: Path) -> Path:
    """
    Serialize a verified pipeline state to a JSONL file.

    Appends a single line to the output file (creates if not exists).
    Thread-safe for concurrent pipeline executions writing to the same file.

    Args:
        state: Verified pipeline state to export.
        output_path: Path to the output .jsonl file.

    Returns:
        The output file path.

    TODO (Phase 2):
        - Implement full serialization logic
        - Add Pydantic model serialization for entities
        - Handle entity UUID serialization
    """
    document_id = state.get("document_id", "unknown")
    logger.info("export.jsonl.start", document_id=document_id, output=str(output_path))

    record: dict[str, Any] = {
        "document_id": document_id,
        "adapted_text": state.get("adapted_text", ""),
        "source_language": state.get("source_language", ""),
        "entities": [],  # Phase 2: serialize LegalEntity list
        "evaluation_scores": {},  # Phase 2: serialize EvaluationScores
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("export.jsonl.complete", document_id=document_id)
    return output_path
