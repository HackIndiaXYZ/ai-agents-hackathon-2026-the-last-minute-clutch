"""
nyayaeval.agents.ingestion — Document Ingestion Agent Node
============================================================

Responsible for extracting raw text from multi-page Indian district court
PDFs and performing LLM-powered entity extraction.

Two extraction paths:
    1. **PDF path**: If ``document_metadata["file_path"]`` is set, reads the
       PDF via PyMuPDF (fitz), extracts text per page.
    2. **Text path**: If ``raw_text`` is already populated in the state
       (e.g., passed directly via the API), skips PDF extraction.

Entity extraction uses the LLM with structured output — we send page text
to the model with our Pydantic entity schemas and get back typed entities.
This is a pragmatic choice for a hackathon: no custom NER training required,
and the LLM handles multilingual legal text well.

LangGraph node contract:
    Reads:  document_metadata, raw_text (optional), document_id, source_language
    Writes: raw_text, raw_pages, token_boundaries, entities,
            entity_extraction_metadata, current_phase, execution_logs
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from nyayaeval.connectors.llm_provider import get_llm
from nyayaeval.core.models import (
    Case,
    CaseStatus,
    Judge,
    Precedent,
    Section,
    Statute,
    TokenBoundary,
)
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


# ─── Structured Output Schema for LLM Entity Extraction ──────────────────────


class ExtractedEntities(BaseModel):
    """Schema the LLM fills when extracting entities from legal text."""

    cases: list[_CaseExtract] = Field(default_factory=list)
    sections: list[_SectionExtract] = Field(default_factory=list)
    statutes: list[_StatuteExtract] = Field(default_factory=list)
    judges: list[_JudgeExtract] = Field(default_factory=list)
    precedents: list[_PrecedentExtract] = Field(default_factory=list)


class _CaseExtract(BaseModel):
    case_number: str
    court_name: str
    case_title: str | None = None
    status: str | None = None
    jurisdiction: str | None = None


class _SectionExtract(BaseModel):
    section_number: str
    parent_statute: str | None = None
    title: str | None = None


class _StatuteExtract(BaseModel):
    statute_name: str
    abbreviation: str | None = None
    year_enacted: int | None = None


class _JudgeExtract(BaseModel):
    full_name: str
    designation: str | None = None
    court_name: str | None = None


class _PrecedentExtract(BaseModel):
    citation: str
    case_name: str | None = None
    court_level: str | None = None
    year: int | None = None
    relevance_summary: str | None = None


# ─── Reorder to satisfy forward refs ─────────────────────────────────────────
ExtractedEntities.model_rebuild()


# ─── Constants ────────────────────────────────────────────────────────────────

_ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a legal document analysis expert specializing in Indian district court records.

Given text from a court document, extract ALL legal entities you can identify. Be thorough and precise.

Entity types to extract:
1. **Cases**: Case numbers (e.g., CRL.A.123/2024), court names, parties, status
2. **Sections**: Legal section references (e.g., Section 302 IPC, Section 498A IPC, Order XXI Rule 97 CPC)
3. **Statutes**: Legislative acts (e.g., Indian Penal Code, Code of Criminal Procedure, Hindu Marriage Act 1955)
4. **Judges**: Names and designations of judicial officers
5. **Precedents**: Cited case law with full citations (e.g., AIR 1978 SC 1457)

Rules:
- Extract ONLY entities explicitly mentioned in the text
- Do NOT hallucinate or infer entities not present in the text
- For sections, always identify the parent statute if mentioned
- For precedents, capture the full citation string
- If a field is not mentioned, leave it as null"""


# ─── PDF Extraction ──────────────────────────────────────────────────────────


def _extract_pdf_pages(file_path: str) -> list[str]:
    """
    Extract text from each page of a PDF using PyMuPDF.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        List of strings, one per page.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        RuntimeError: If PyMuPDF fails to parse the document.
    """
    import fitz  # PyMuPDF — imported here to keep it optional at module level

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        doc = fitz.open(str(path))
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            # Clean up common PDF extraction artifacts
            text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse excessive newlines
            text = text.strip()
            if text:
                pages.append(text)
        doc.close()
        return pages
    except Exception as exc:
        raise RuntimeError(f"Failed to extract PDF '{file_path}': {exc}") from exc


# ─── Token Boundary Detection ────────────────────────────────────────────────


def _find_token_boundaries(raw_text: str, entities: list[Any]) -> list[TokenBoundary]:
    """
    Find character offsets where extracted entities appear in the raw text.

    Performs simple string matching — finds the first occurrence of each
    entity's identifying text (case_number, citation, full_name, etc.)
    in the raw text and records the span.
    """
    boundaries: list[TokenBoundary] = []

    for entity in entities:
        # Determine the searchable text for each entity type
        search_terms: list[tuple[str, str]] = []

        if hasattr(entity, "case_number"):
            search_terms.append((entity.case_number, "CASE"))
        if hasattr(entity, "citation"):
            search_terms.append((entity.citation, "PRECEDENT"))
        if hasattr(entity, "full_name"):
            search_terms.append((entity.full_name, "JUDGE"))
        if hasattr(entity, "section_number"):
            term = f"Section {entity.section_number}"
            if hasattr(entity, "parent_statute") and entity.parent_statute:
                term += f" {entity.parent_statute}"
            search_terms.append((term, "SECTION"))
        if hasattr(entity, "statute_name"):
            search_terms.append((entity.statute_name, "STATUTE"))

        for term, label in search_terms:
            idx = raw_text.find(term)
            if idx >= 0:
                boundaries.append(
                    TokenBoundary(
                        start=idx,
                        end=idx + len(term),
                        label=label,
                        confidence=entity.extraction_confidence
                        if hasattr(entity, "extraction_confidence")
                        else 1.0,
                    )
                )

    return boundaries


# ─── Entity Conversion ───────────────────────────────────────────────────────


def _convert_extracted_entities(
    extracted: ExtractedEntities, document_id: str
) -> list[Any]:
    """Convert LLM-extracted entity structs into domain model instances."""
    entities: list[Any] = []

    for c in extracted.cases:
        status_map = {
            "pending": CaseStatus.PENDING,
            "disposed": CaseStatus.DISPOSED,
            "appealed": CaseStatus.APPEALED,
            "transferred": CaseStatus.TRANSFERRED,
        }
        entities.append(
            Case(
                case_number=c.case_number,
                court_name=c.court_name,
                case_title=c.case_title,
                status=status_map.get((c.status or "").lower(), CaseStatus.UNKNOWN),
                jurisdiction=c.jurisdiction,
                source_document_id=document_id,
            )
        )

    for s in extracted.sections:
        entities.append(
            Section(
                section_number=s.section_number,
                parent_statute=s.parent_statute,
                title=s.title,
                source_document_id=document_id,
            )
        )

    for st in extracted.statutes:
        entities.append(
            Statute(
                statute_name=st.statute_name,
                abbreviation=st.abbreviation,
                year_enacted=st.year_enacted,
                source_document_id=document_id,
            )
        )

    for j in extracted.judges:
        entities.append(
            Judge(
                full_name=j.full_name,
                designation=j.designation,
                court_name=j.court_name,
                source_document_id=document_id,
            )
        )

    for p in extracted.precedents:
        entities.append(
            Precedent(
                citation=p.citation,
                case_name=p.case_name,
                court_level=p.court_level,
                year=p.year,
                relevance_summary=p.relevance_summary,
                source_document_id=document_id,
            )
        )

    return entities


# ─── Agent Node ───────────────────────────────────────────────────────────────


async def ingestion_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Extract text from the source document and identify legal entities.

    Flow:
        1. Extract text from PDF (if file_path provided) or use existing raw_text
        2. Run LLM-powered entity extraction on the text
        3. Detect token boundaries for entity spans
        4. Return extracted data as partial state update

    Returns:
        Partial state update with raw_text, raw_pages, entities, token_boundaries.
    """
    document_id = state.get("document_id", "unknown")
    metadata = state.get("document_metadata", {})
    start_time = time.monotonic()

    logger.info("agent.ingestion.start", document_id=document_id)

    # ── Step 1: Text Extraction ───────────────────────────────────────────
    raw_text = state.get("raw_text", "")
    raw_pages: list[str] = state.get("raw_pages", [])

    file_path = metadata.get("file_path")
    if file_path and not raw_text:
        try:
            raw_pages = _extract_pdf_pages(file_path)
            raw_text = "\n\n".join(raw_pages)
            logger.info(
                "agent.ingestion.pdf_extracted",
                document_id=document_id,
                pages=len(raw_pages),
                chars=len(raw_text),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("agent.ingestion.pdf_failed", document_id=document_id, error=str(exc))
            return {
                "error": f"Ingestion failed: {exc}",
                "current_phase": "failed",
                "execution_logs": [f"[ingestion] FAILED: {exc}"],
            }

    if not raw_text:
        return {
            "error": "No text to process: raw_text is empty and no file_path provided",
            "current_phase": "failed",
            "execution_logs": ["[ingestion] FAILED: No input text"],
        }

    # Split into pages if not already done
    if not raw_pages:
        # Heuristic: split on double newlines or form feeds
        raw_pages = [p.strip() for p in re.split(r"\f|\n{3,}", raw_text) if p.strip()]
        if not raw_pages:
            raw_pages = [raw_text]

    # ── Step 2: LLM Entity Extraction ─────────────────────────────────────
    entities: list[Any] = []
    extraction_meta: dict[str, Any] = {"method": "llm_structured_output", "pages_processed": 0}

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ExtractedEntities)

        # Process in chunks to avoid token limits — concatenate pages up to ~4000 chars
        chunks: list[str] = []
        current_chunk = ""
        for page in raw_pages:
            if len(current_chunk) + len(page) > 4000 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = page
            else:
                current_chunk += ("\n\n" + page) if current_chunk else page
        if current_chunk:
            chunks.append(current_chunk)

        for i, chunk in enumerate(chunks):
            try:
                result: ExtractedEntities = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=_ENTITY_EXTRACTION_SYSTEM_PROMPT),
                        HumanMessage(content=f"Extract all legal entities from this court document text:\n\n{chunk}"),
                    ]
                )
                chunk_entities = _convert_extracted_entities(result, document_id)
                entities.extend(chunk_entities)
                extraction_meta["pages_processed"] = i + 1
                logger.debug(
                    "agent.ingestion.chunk_extracted",
                    document_id=document_id,
                    chunk=i + 1,
                    entities_found=len(chunk_entities),
                )
            except Exception as exc:
                logger.warning(
                    "agent.ingestion.chunk_failed",
                    document_id=document_id,
                    chunk=i + 1,
                    error=str(exc),
                )

        extraction_meta["total_entities"] = len(entities)
        extraction_meta["entity_breakdown"] = {
            "cases": sum(1 for e in entities if hasattr(e, "case_number")),
            "sections": sum(1 for e in entities if hasattr(e, "section_number")),
            "statutes": sum(1 for e in entities if hasattr(e, "statute_name")),
            "judges": sum(1 for e in entities if hasattr(e, "full_name") and not hasattr(e, "case_number")),
            "precedents": sum(1 for e in entities if hasattr(e, "citation")),
        }

    except Exception as exc:
        logger.error("agent.ingestion.entity_extraction_failed", error=str(exc))
        extraction_meta["error"] = str(exc)

    # ── Step 3: Token Boundary Detection ──────────────────────────────────
    token_boundaries = _find_token_boundaries(raw_text, entities)

    duration_s = time.monotonic() - start_time
    logger.info(
        "agent.ingestion.complete",
        document_id=document_id,
        pages=len(raw_pages),
        entities=len(entities),
        boundaries=len(token_boundaries),
        duration_s=round(duration_s, 2),
    )

    return {
        "raw_text": raw_text,
        "raw_pages": raw_pages,
        "entities": entities,
        "token_boundaries": token_boundaries,
        "entity_extraction_metadata": extraction_meta,
        "current_phase": "ingestion",
        "execution_logs": [
            f"[ingestion] Extracted {len(raw_pages)} pages, {len(raw_text)} chars, "
            f"{len(entities)} entities in {duration_s:.1f}s"
        ],
    }
