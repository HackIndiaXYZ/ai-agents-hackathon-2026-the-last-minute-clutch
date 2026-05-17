"""
nyayaeval.agents.adaptation — Adaptive Data Translation Agent Node
====================================================================

Integrates with the Adaption Adaptive Data platform to translate
extracted text from regional Indian languages into standardized
English legal terminology.

Uses the official ``adaption`` SDK lifecycle:
    1. Upload page text as a JSONL dataset
    2. Run the adaptation job
    3. Wait for completion
    4. Download the adapted dataset
    5. Retrieve quality evaluation metrics

Also supports Redis caching to avoid re-uploading identical content,
and an English passthrough for documents already in English.

LangGraph node contract:
    Reads:  raw_text, raw_pages, source_language, document_id
    Writes: adapted_text, adapted_segments, adaptation_metadata,
            current_phase, execution_logs
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from nyayaeval.connectors.registry import get_adaption, get_redis
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


def _content_hash(text: str) -> str:
    """Generate a short hash of text content for cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def adaptation_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Translate and standardize the raw extracted text via Adaptive Data.

    Flow:
        1. English passthrough if source language is 'en'
        2. Check Redis for cached adaptation result
        3. Upload pages as JSONL to Adaption platform
        4. Run adaptation job and wait for completion
        5. Download adapted dataset
        6. Retrieve Adaption evaluation metrics
        7. Assemble adapted_text from results

    Returns:
        Partial state update with adapted_text, adapted_segments, and logs.
    """
    document_id = state.get("document_id", "unknown")
    source_lang = state.get("source_language", "unknown")
    raw_text = state.get("raw_text", "")
    raw_pages = state.get("raw_pages", [raw_text] if raw_text else [])
    start_time = time.monotonic()

    logger.info(
        "agent.adaptation.start",
        document_id=document_id,
        source_language=source_lang,
        pages=len(raw_pages),
    )

    # ── Pass-through for English text ─────────────────────────────────────
    if source_lang.lower() in ("en", "eng", "english"):
        logger.info("agent.adaptation.english_passthrough", document_id=document_id)
        segments = [
            {"source": page, "target": page, "language": "en"}
            for page in raw_pages
        ]
        return {
            "adapted_text": raw_text,
            "adapted_segments": segments,
            "adaptation_metadata": {"method": "passthrough", "source_language": "en"},
            "current_phase": "adaptation",
            "execution_logs": [
                f"[adaptation] English passthrough for document {document_id} "
                f"({len(raw_pages)} pages)"
            ],
        }

    # ── Check Redis cache ─────────────────────────────────────────────────
    content_hash = _content_hash(raw_text)
    cache_key = f"adaptation:{content_hash}:{source_lang}"

    try:
        redis_client = get_redis()
        cached = await redis_client.get(cache_key)
        if cached is not None:
            cached_data = json.loads(cached)
            logger.info("agent.adaptation.cache_hit", document_id=document_id)
            return {
                "adapted_text": cached_data["adapted_text"],
                "adapted_segments": cached_data["adapted_segments"],
                "adaptation_metadata": {
                    **cached_data.get("metadata", {}),
                    "method": "cache_hit",
                },
                "current_phase": "adaptation",
                "execution_logs": [
                    f"[adaptation] Cache hit for document {document_id}"
                ],
            }
    except RuntimeError:
        redis_client = None
    except Exception as exc:
        redis_client = None
        logger.warning("agent.adaptation.cache_error", error=str(exc))

    # ── Use Adaption SDK ──────────────────────────────────────────────────
    try:
        adaption_client = get_adaption()
    except RuntimeError:
        # Fallback: no Adaption client configured → passthrough with warning
        logger.warning("agent.adaptation.no_adaption_client", document_id=document_id)
        segments = [
            {"source": page, "target": page, "language": source_lang}
            for page in raw_pages
        ]
        return {
            "adapted_text": raw_text,
            "adapted_segments": segments,
            "adaptation_metadata": {
                "method": "passthrough_no_client",
                "warning": "Adaption SDK not configured — text passed through unchanged",
            },
            "current_phase": "adaptation",
            "execution_logs": [
                f"[adaptation] WARNING: No Adaption client — passthrough for {document_id}"
            ],
        }

    # ── Step 1: Build JSONL records from pages ────────────────────────────
    records = []
    for i, page in enumerate(raw_pages):
        if page.strip():
            records.append({
                "page_number": i + 1,
                "source_text": page,
                "source_language": source_lang,
                "document_id": document_id,
            })

    if not records:
        return {
            "adapted_text": "",
            "adapted_segments": [],
            "adaptation_metadata": {"method": "empty_input"},
            "current_phase": "adaptation",
            "execution_logs": [f"[adaptation] No text to adapt for {document_id}"],
        }

    # ── Step 2: Upload to Adaption platform ───────────────────────────────
    dataset_name = f"nyayaeval_{document_id}_{content_hash[:8]}"
    adaption_metadata: dict[str, Any] = {"method": "adaption_sdk", "source_language": source_lang}

    try:
        upload_result = adaption_client.upload_from_records(records, name=dataset_name)
        dataset_id = upload_result["dataset_id"]
        adaption_metadata["dataset_id"] = dataset_id
        logger.info(
            "agent.adaptation.uploaded",
            document_id=document_id,
            dataset_id=dataset_id,
            records=len(records),
        )

        # ── Step 3: Run adaptation job ────────────────────────────────────
        run_result = adaption_client.run_adaptation(
            dataset_id=dataset_id,
            source_column="source_text",
        )
        adaption_metadata["run_status"] = run_result.get("status")

        # ── Step 4: Wait for completion ───────────────────────────────────
        completion = adaption_client.wait_for_completion(
            dataset_id=dataset_id,
            timeout=300,  # 5 min max for a single document
        )
        adaption_metadata["final_status"] = completion.get("status")

        # ── Step 5: Get evaluation metrics ────────────────────────────────
        eval_result = adaption_client.get_evaluation(dataset_id)
        adaption_metadata["evaluation"] = eval_result.get("evaluation")

        # ── Step 6: Download adapted dataset ──────────────────────────────
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, dir="."
        ) as tmp:
            tmp_path = tmp.name

        try:
            adaption_client.download_dataset(dataset_id, tmp_path, file_format="jsonl")

            # Parse the downloaded JSONL
            adapted_segments: list[dict[str, str]] = []
            with open(tmp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        adapted_segments.append({
                            "source": row.get("source_text", ""),
                            "target": row.get("adapted_text", row.get("completion", "")),
                            "language": source_lang,
                        })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        adapted_text = "\n\n".join(seg["target"] for seg in adapted_segments)

    except Exception as exc:
        # Adaption SDK failed — fall back to passthrough
        logger.error(
            "agent.adaptation.sdk_failed",
            document_id=document_id,
            error=str(exc),
        )
        adapted_segments = [
            {"source": page, "target": page, "language": source_lang}
            for page in raw_pages
        ]
        adapted_text = raw_text
        adaption_metadata["error"] = str(exc)
        adaption_metadata["method"] = "passthrough_sdk_error"

    # ── Cache the result ──────────────────────────────────────────────────
    if redis_client is not None:
        try:
            cache_data = json.dumps({
                "adapted_text": adapted_text,
                "adapted_segments": adapted_segments,
                "metadata": adaption_metadata,
            }, ensure_ascii=False)
            await redis_client.set(cache_key, cache_data, ttl=86400)
        except Exception as exc:
            logger.warning("agent.adaptation.cache_write_failed", error=str(exc))

    duration_s = time.monotonic() - start_time
    adaption_metadata["duration_s"] = round(duration_s, 2)

    logger.info(
        "agent.adaptation.complete",
        document_id=document_id,
        segments=len(adapted_segments),
        method=adaption_metadata.get("method", "adaption_sdk"),
        duration_s=round(duration_s, 2),
    )

    return {
        "adapted_text": adapted_text,
        "adapted_segments": adapted_segments,
        "adaptation_metadata": adaption_metadata,
        "current_phase": "adaptation",
        "execution_logs": [
            f"[adaptation] {adaption_metadata.get('method', 'adaption_sdk')}: "
            f"{source_lang}→en for {document_id}, "
            f"{len(adapted_segments)} segments in {duration_s:.1f}s"
        ],
    }
