"""
nyayaeval.core.models — Legal Domain Entities
===============================================

Pydantic models representing the core legal concepts extracted from Indian
district court documents. These entities serve dual purpose:

    1. **Boundary Validation** — strict schema enforcement when ingesting
       raw extracted data or receiving API responses.
    2. **Knowledge Graph Nodes** — each model maps to a Neo4j node label,
       with relationships defined at the graph-builder agent layer.

Design decisions:
    - All models inherit from a common ``LegalEntityBase`` to enforce shared
      metadata (source document ID, extraction confidence, timestamps).
    - ``LegalEntity`` is a discriminated union (tagged via ``entity_type``)
      so that heterogeneous entity lists can be deserialized polymorphically.
    - ``TokenBoundary`` is deliberately kept outside the entity hierarchy —
      it's a positional annotation, not a domain concept.

Neo4j label mapping (implemented in agents/graph_builder.py):
    Case       → (:Case)
    Section    → (:Section)
    Statute    → (:Statute)
    Judge      → (:Judge)
    Precedent  → (:Precedent)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────


class EntityType(StrEnum):
    """Discriminator for the LegalEntity tagged union."""

    CASE = "case"
    SECTION = "section"
    STATUTE = "statute"
    JUDGE = "judge"
    PRECEDENT = "precedent"


class CaseStatus(StrEnum):
    """Lifecycle status of a district court case."""

    PENDING = "pending"
    DISPOSED = "disposed"
    APPEALED = "appealed"
    TRANSFERRED = "transferred"
    UNKNOWN = "unknown"


# ─── Token Boundary ──────────────────────────────────────────────────────────


class TokenBoundary(BaseModel):
    """
    Positional marker for a token span within extracted document text.

    Used by the ingestion agent to track where entities were found in the
    raw text, enabling downstream agents to perform targeted re-extraction
    or correction on specific spans rather than re-processing entire pages.
    """

    start: int = Field(..., ge=0, description="Start character offset (inclusive)")
    end: int = Field(..., gt=0, description="End character offset (exclusive)")
    label: str = Field(..., description="Entity label or annotation tag")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence score"
    )
    page_number: int | None = Field(default=None, ge=1, description="Source page number")


# ─── Base Entity ──────────────────────────────────────────────────────────────


class LegalEntityBase(BaseModel):
    """
    Common metadata shared by all legal entities.

    Every entity carries provenance information (which document it came from,
    when it was extracted, and how confident the extraction was). This enables
    the evaluation loop to trace hallucinations back to their source.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique entity identifier")
    source_document_id: str | None = Field(
        default=None, description="ID of the originating document"
    )
    extraction_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Model confidence in this extraction"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Extraction timestamp"
    )
    raw_text_span: str | None = Field(
        default=None,
        description="Original text span from which this entity was extracted",
    )


# ─── Domain Entities ─────────────────────────────────────────────────────────


class Case(LegalEntityBase):
    """
    A district court case — the primary document-level entity.

    Relationships (in Neo4j):
        (Case)-[:CITES]->(Precedent)
        (Case)-[:GOVERNED_BY]->(Statute)
        (Case)-[:PRESIDED_BY]->(Judge)
        (Case)-[:CONTAINS_SECTION]->(Section)
    """

    entity_type: Literal[EntityType.CASE] = EntityType.CASE
    case_number: str = Field(..., description="Official case number (e.g., 'CRL.A.123/2024')")
    court_name: str = Field(..., description="Name of the district court")
    case_title: str | None = Field(default=None, description="Case title / parties")
    filing_date: date | None = Field(default=None, description="Date case was filed")
    decision_date: date | None = Field(default=None, description="Date of judgment")
    status: CaseStatus = Field(default=CaseStatus.UNKNOWN, description="Current case status")
    jurisdiction: str | None = Field(default=None, description="Jurisdictional district/state")
    original_language: str | None = Field(
        default=None, description="ISO 639-1 language code of the source document"
    )
    summary: str | None = Field(default=None, description="AI-generated case summary")


class Section(LegalEntityBase):
    """
    A section or sub-section of a legal statute referenced in a case.

    Examples: Section 302 IPC, Section 498A IPC, Order XXI Rule 97 CPC.
    """

    entity_type: Literal[EntityType.SECTION] = EntityType.SECTION
    section_number: str = Field(..., description="Section identifier (e.g., '302', '498A')")
    parent_statute: str | None = Field(
        default=None, description="Parent statute abbreviation (e.g., 'IPC', 'CPC')"
    )
    title: str | None = Field(default=None, description="Section title or heading")
    description: str | None = Field(default=None, description="Section content summary")


class Statute(LegalEntityBase):
    """
    A legislative act or code referenced in court documents.

    Examples: Indian Penal Code (IPC), Code of Criminal Procedure (CrPC),
    Hindu Marriage Act, 1955.
    """

    entity_type: Literal[EntityType.STATUTE] = EntityType.STATUTE
    statute_name: str = Field(..., description="Full name of the statute")
    abbreviation: str | None = Field(default=None, description="Common abbreviation (e.g., 'IPC')")
    year_enacted: int | None = Field(default=None, description="Year the statute was enacted")
    is_active: bool = Field(default=True, description="Whether the statute is currently in force")


class Judge(LegalEntityBase):
    """
    A judicial officer who presided over or is referenced in a case.
    """

    entity_type: Literal[EntityType.JUDGE] = EntityType.JUDGE
    full_name: str = Field(..., description="Full name of the judge")
    designation: str | None = Field(
        default=None,
        description="Judicial designation (e.g., 'District Judge', 'Additional Sessions Judge')",
    )
    court_name: str | None = Field(default=None, description="Court of assignment")


class Precedent(LegalEntityBase):
    """
    A cited precedent (prior case law) referenced in the current case.

    Precedents are critical for the Knowledge Graph — they form the backbone
    of legal reasoning chains. The evaluation loop specifically validates
    that cited precedents actually exist and are correctly attributed.
    """

    entity_type: Literal[EntityType.PRECEDENT] = EntityType.PRECEDENT
    citation: str = Field(
        ..., description="Full legal citation (e.g., 'AIR 1978 SC 1457')"
    )
    case_name: str | None = Field(default=None, description="Name of the cited case")
    court_level: str | None = Field(
        default=None,
        description="Level of court (e.g., 'Supreme Court', 'High Court')",
    )
    year: int | None = Field(default=None, description="Year of the cited judgment")
    relevance_summary: str | None = Field(
        default=None, description="Why this precedent was cited"
    )


# ─── Discriminated Union ─────────────────────────────────────────────────────

LegalEntity = Annotated[
    Union[Case, Section, Statute, Judge, Precedent],
    Field(discriminator="entity_type"),
]
"""
Polymorphic legal entity type.

Using a discriminated union (tagged by ``entity_type``) allows us to
deserialize heterogeneous entity lists from JSON without ambiguity.
The discriminator field is automatically checked by Pydantic to select
the correct model class.

Usage:
    from pydantic import TypeAdapter
    adapter = TypeAdapter(list[LegalEntity])
    entities = adapter.validate_json(raw_json)
"""
