"""
nyayaeval.agents.export_node — Export Agent Node
==================================================

Terminal success node in the LangGraph pipeline. Serializes verified
pipeline state into JSONL/CSV formats and, critically for the hackathon,
uploads the final dataset to the Adaption platform for evaluation and
publication to HuggingFace/Kaggle.

LangGraph node contract:
    Reads:  document_id, adapted_text, entities, evaluation_scores,
            source_language, document_metadata, adaptation_metadata
    Writes: current_phase, execution_logs
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from nyayaeval.connectors.registry import get_adaption
from nyayaeval.core.state import NyayaEvalState
from nyayaeval.export.csv_exporter import export_to_csv
from nyayaeval.export.jsonl_exporter import export_to_jsonl

logger = structlog.get_logger(__name__)

# Default output directory
_OUTPUT_DIR = Path("output")


async def export_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Serialize the verified pipeline state to JSONL, CSV, and Adaption platform.

    Flow:
        1. Export to local JSONL (append mode)
        2. Export to local CSV (append mode)
        3. Upload final dataset to Adaption for evaluation + HF publishing
        4. Log export paths and Adaption dataset ID

    Returns:
        Partial state update with current_phase and execution_logs.
    """
    document_id = state.get("document_id", "unknown")
    start_time = time.monotonic()

    logger.info("agent.export.start", document_id=document_id)

    export_paths: list[str] = []
    errors: list[str] = []

    # ── Local JSONL Export ────────────────────────────────────────────────
    try:
        jsonl_path = await export_to_jsonl(state, _OUTPUT_DIR / "nyayaeval_output.jsonl")
        export_paths.append(str(jsonl_path))
        logger.info("agent.export.jsonl_complete", path=str(jsonl_path))
    except Exception as exc:
        errors.append(f"JSONL export failed: {exc}")
        logger.error("agent.export.jsonl_failed", error=str(exc))

    # ── Local CSV Export ─────────────────────────────────────────────────
    try:
        csv_path = await export_to_csv(state, _OUTPUT_DIR / "nyayaeval_output.csv")
        export_paths.append(str(csv_path))
        logger.info("agent.export.csv_complete", path=str(csv_path))
    except Exception as exc:
        errors.append(f"CSV export failed: {exc}")
        logger.error("agent.export.csv_failed", error=str(exc))

    # ── Adaption Platform Upload ──────────────────────────────────────────
    # Upload the verified output as a dataset to Adaption for evaluation
    # and potential HuggingFace/Kaggle publishing (hackathon requirement)
    adaption_dataset_id = None
    try:
        adaption_client = get_adaption()

        # Build export record for Adaption
        entities = state.get("entities", [])
        entity_summaries = []
        for e in entities[:50]:  # Cap to avoid oversized datasets
            if hasattr(e, "model_dump"):
                d = e.model_dump(exclude={"raw_text_span"})
                # Convert non-serializable types
                for k, v in d.items():
                    if hasattr(v, "hex"):  # UUID
                        d[k] = str(v)
                entity_summaries.append(d)

        scores = state.get("evaluation_scores")
        scores_dict = {}
        if scores:
            scores_dict = {
                "faithfulness": scores.faithfulness,
                "context_recall": scores.context_recall,
                "legal_consistency": scores.legal_consistency,
            }

        export_record = {
            "document_id": document_id,
            "source_language": state.get("source_language", ""),
            "source_text": state.get("raw_text", "")[:5000],
            "adapted_text": state.get("adapted_text", "")[:5000],
            "entities_count": len(entities),
            "entities": json.dumps(entity_summaries, ensure_ascii=False, default=str),
            "evaluation_scores": json.dumps(scores_dict),
            "is_verified": state.get("is_verified", False),
            "retry_count": state.get("retry_count", 0),
        }

        upload_result = adaption_client.upload_from_records(
            [export_record],
            name=f"nyayaeval_verified_{document_id}",
        )
        adaption_dataset_id = upload_result.get("dataset_id")
        export_paths.append(f"adaption:{adaption_dataset_id}")
        logger.info(
            "agent.export.adaption_uploaded",
            dataset_id=adaption_dataset_id,
        )

    except RuntimeError:
        logger.info("agent.export.adaption_not_configured")
    except Exception as exc:
        errors.append(f"Adaption upload failed: {exc}")
        logger.warning("agent.export.adaption_failed", error=str(exc))

    duration_s = time.monotonic() - start_time
    logger.info(
        "agent.export.complete",
        document_id=document_id,
        exports=len(export_paths),
        errors=len(errors),
        adaption_dataset_id=adaption_dataset_id,
        duration_s=round(duration_s, 2),
    )

    status = "completed" if not errors else "completed_with_errors"
    log_msg = (
        f"[export] {document_id}: exported to {len(export_paths)} targets "
        f"({', '.join(export_paths)}) in {duration_s:.1f}s"
    )
    if errors:
        log_msg += f" | Errors: {'; '.join(errors)}"

    return {
        "current_phase": status,
        "execution_logs": [log_msg],
    }
