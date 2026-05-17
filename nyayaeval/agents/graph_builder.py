"""
nyayaeval.agents.graph_builder — Knowledge Graph Construction Agent Node
=========================================================================

Maps extracted entities and relationships into the Neo4j Knowledge Graph.

Strategy:
    - Uses MERGE (not CREATE) for idempotent node creation — running the
      pipeline twice on the same document won't create duplicates.
    - Entities are batched by type and written via UNWIND for efficiency.
    - Relationships are created AFTER all nodes exist to avoid reference errors.
    - After writes, queries the subgraph to populate graph_context for the evaluator.

LangGraph node contract:
    Reads:  entities, adapted_text, document_id
    Writes: graph_context, graph_node_ids, current_phase, execution_logs
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

import structlog

from nyayaeval.connectors.registry import get_neo4j
from nyayaeval.core.models import Case, Judge, Precedent, Section, Statute
from nyayaeval.core.state import NyayaEvalState

logger = structlog.get_logger(__name__)


# ─── Cypher Templates ────────────────────────────────────────────────────────

_MERGE_CASE = """
UNWIND $batch AS row
MERGE (c:Case {case_number: row.case_number})
SET c.court_name = row.court_name,
    c.case_title = row.case_title,
    c.status = row.status,
    c.jurisdiction = row.jurisdiction,
    c.original_language = row.original_language,
    c.summary = row.summary,
    c.source_document_id = row.source_document_id,
    c.entity_id = row.entity_id
RETURN elementId(c) AS node_id
"""

_MERGE_SECTION = """
UNWIND $batch AS row
MERGE (s:Section {section_number: row.section_number, parent_statute: coalesce(row.parent_statute, 'UNKNOWN')})
SET s.title = row.title,
    s.description = row.description,
    s.source_document_id = row.source_document_id,
    s.entity_id = row.entity_id
RETURN elementId(s) AS node_id
"""

_MERGE_STATUTE = """
UNWIND $batch AS row
MERGE (st:Statute {statute_name: row.statute_name})
SET st.abbreviation = row.abbreviation,
    st.year_enacted = row.year_enacted,
    st.is_active = row.is_active,
    st.source_document_id = row.source_document_id,
    st.entity_id = row.entity_id
RETURN elementId(st) AS node_id
"""

_MERGE_JUDGE = """
UNWIND $batch AS row
MERGE (j:Judge {full_name: row.full_name})
SET j.designation = row.designation,
    j.court_name = row.court_name,
    j.source_document_id = row.source_document_id,
    j.entity_id = row.entity_id
RETURN elementId(j) AS node_id
"""

_MERGE_PRECEDENT = """
UNWIND $batch AS row
MERGE (p:Precedent {citation: row.citation})
SET p.case_name = row.case_name,
    p.court_level = row.court_level,
    p.year = row.year,
    p.relevance_summary = row.relevance_summary,
    p.source_document_id = row.source_document_id,
    p.entity_id = row.entity_id
RETURN elementId(p) AS node_id
"""

# ─── Relationship Queries ────────────────────────────────────────────────────

_REL_CASE_CITES_PRECEDENT = """
UNWIND $batch AS row
MATCH (c:Case {case_number: row.case_number})
MATCH (p:Precedent {citation: row.citation})
MERGE (c)-[:CITES]->(p)
"""

_REL_CASE_GOVERNED_BY_STATUTE = """
UNWIND $batch AS row
MATCH (c:Case {case_number: row.case_number})
MATCH (st:Statute {statute_name: row.statute_name})
MERGE (c)-[:GOVERNED_BY]->(st)
"""

_REL_CASE_PRESIDED_BY_JUDGE = """
UNWIND $batch AS row
MATCH (c:Case {case_number: row.case_number})
MATCH (j:Judge {full_name: row.judge_name})
MERGE (c)-[:PRESIDED_BY]->(j)
"""

_REL_CASE_CONTAINS_SECTION = """
UNWIND $batch AS row
MATCH (c:Case {case_number: row.case_number})
MATCH (s:Section {section_number: row.section_number, parent_statute: coalesce(row.parent_statute, 'UNKNOWN')})
MERGE (c)-[:CONTAINS_SECTION]->(s)
"""

_REL_SECTION_PART_OF_STATUTE = """
UNWIND $batch AS row
MATCH (s:Section {section_number: row.section_number, parent_statute: row.abbreviation})
MATCH (st:Statute {statute_name: row.statute_name})
MERGE (s)-[:PART_OF]->(st)
"""

# ─── Context Query ────────────────────────────────────────────────────────────

_SUBGRAPH_CONTEXT_QUERY = """
MATCH (c:Case {source_document_id: $doc_id})
OPTIONAL MATCH (c)-[:CITES]->(p:Precedent)
OPTIONAL MATCH (c)-[:GOVERNED_BY]->(st:Statute)
OPTIONAL MATCH (c)-[:PRESIDED_BY]->(j:Judge)
OPTIONAL MATCH (c)-[:CONTAINS_SECTION]->(s:Section)
RETURN c.case_number AS case_number,
       collect(DISTINCT p.citation) AS precedents_cited,
       collect(DISTINCT st.statute_name) AS statutes_referenced,
       collect(DISTINCT j.full_name) AS judges,
       collect(DISTINCT s.section_number) AS sections
"""


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _serialize_uuid(val: Any) -> str | None:
    """Convert UUID to string for Neo4j storage."""
    if isinstance(val, UUID):
        return str(val)
    return val


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    """Convert a Pydantic entity to a Neo4j-compatible flat dict."""
    data = entity.model_dump(exclude={"id", "created_at", "extraction_confidence", "raw_text_span", "entity_type"})
    data["entity_id"] = str(entity.id)
    # Convert any remaining non-primitive types
    for k, v in data.items():
        if isinstance(v, UUID):
            data[k] = str(v)
        elif v is None:
            data[k] = None
    return data


# ─── Agent Node ───────────────────────────────────────────────────────────────


async def graph_builder_node(state: NyayaEvalState) -> dict[str, Any]:
    """
    Construct the legal Knowledge Graph from extracted entities.

    Flow:
        1. Group entities by type
        2. Batch-MERGE nodes for each type
        3. Create relationships between nodes
        4. Query subgraph for context data
        5. Return graph_context and node_ids

    Returns:
        Partial state update with graph_context, graph_node_ids, and logs.
    """
    document_id = state.get("document_id", "unknown")
    entities = state.get("entities", [])
    start_time = time.monotonic()

    logger.info("agent.graph_builder.start", document_id=document_id, entities=len(entities))

    if not entities:
        return {
            "graph_context": {"message": "No entities to process"},
            "graph_node_ids": [],
            "current_phase": "graph_building",
            "execution_logs": [f"[graph_builder] No entities for document {document_id}"],
        }

    # ── Get Neo4j client ──────────────────────────────────────────────────
    try:
        neo4j = get_neo4j()
    except RuntimeError:
        logger.warning("agent.graph_builder.no_neo4j", document_id=document_id)
        return {
            "graph_context": {"warning": "Neo4j not configured"},
            "graph_node_ids": [],
            "current_phase": "graph_building",
            "execution_logs": [
                f"[graph_builder] WARNING: Neo4j not configured — skipping KG for {document_id}"
            ],
        }

    # ── Group entities by type ────────────────────────────────────────────
    cases: list[Case] = [e for e in entities if isinstance(e, Case)]
    sections: list[Section] = [e for e in entities if isinstance(e, Section)]
    statutes: list[Statute] = [e for e in entities if isinstance(e, Statute)]
    judges: list[Judge] = [e for e in entities if isinstance(e, Judge)]
    precedents: list[Precedent] = [e for e in entities if isinstance(e, Precedent)]

    all_node_ids: list[str] = []
    stats: dict[str, int] = {}

    # ── Step 1: MERGE Nodes ───────────────────────────────────────────────
    type_batches: list[tuple[str, str, list[Any]]] = [
        ("cases", _MERGE_CASE, cases),
        ("sections", _MERGE_SECTION, sections),
        ("statutes", _MERGE_STATUTE, statutes),
        ("judges", _MERGE_JUDGE, judges),
        ("precedents", _MERGE_PRECEDENT, precedents),
    ]

    for type_name, query, entity_list in type_batches:
        if not entity_list:
            stats[type_name] = 0
            continue

        batch_data = [_entity_to_dict(e) for e in entity_list]
        try:
            records = await neo4j.execute_write_batch(query, batch_data)
            node_ids = [r.get("node_id", "") for r in records if r.get("node_id")]
            all_node_ids.extend(node_ids)
            stats[type_name] = len(node_ids)
            logger.debug(
                "agent.graph_builder.nodes_created",
                type=type_name,
                count=len(node_ids),
            )
        except Exception as exc:
            logger.error(
                "agent.graph_builder.merge_failed",
                type=type_name,
                error=str(exc),
            )
            stats[type_name] = 0

    # ── Step 2: Create Relationships ──────────────────────────────────────
    rel_count = 0

    if cases and precedents:
        for case in cases:
            rels = [{"case_number": case.case_number, "citation": p.citation} for p in precedents]
            try:
                await neo4j.execute_write_batch(_REL_CASE_CITES_PRECEDENT, rels)
                rel_count += len(rels)
            except Exception as exc:
                logger.warning("agent.graph_builder.rel_failed", rel="CITES", error=str(exc))

    if cases and statutes:
        for case in cases:
            rels = [{"case_number": case.case_number, "statute_name": s.statute_name} for s in statutes]
            try:
                await neo4j.execute_write_batch(_REL_CASE_GOVERNED_BY_STATUTE, rels)
                rel_count += len(rels)
            except Exception as exc:
                logger.warning("agent.graph_builder.rel_failed", rel="GOVERNED_BY", error=str(exc))

    if cases and judges:
        for case in cases:
            rels = [{"case_number": case.case_number, "judge_name": j.full_name} for j in judges]
            try:
                await neo4j.execute_write_batch(_REL_CASE_PRESIDED_BY_JUDGE, rels)
                rel_count += len(rels)
            except Exception as exc:
                logger.warning("agent.graph_builder.rel_failed", rel="PRESIDED_BY", error=str(exc))

    if cases and sections:
        for case in cases:
            rels = [
                {
                    "case_number": case.case_number,
                    "section_number": s.section_number,
                    "parent_statute": s.parent_statute or "UNKNOWN",
                }
                for s in sections
            ]
            try:
                await neo4j.execute_write_batch(_REL_CASE_CONTAINS_SECTION, rels)
                rel_count += len(rels)
            except Exception as exc:
                logger.warning("agent.graph_builder.rel_failed", rel="CONTAINS_SECTION", error=str(exc))

    if sections and statutes:
        rels = []
        for sec in sections:
            for stat in statutes:
                if sec.parent_statute and (
                    sec.parent_statute == stat.abbreviation
                    or sec.parent_statute in (stat.statute_name or "")
                ):
                    rels.append({
                        "section_number": sec.section_number,
                        "abbreviation": sec.parent_statute,
                        "statute_name": stat.statute_name,
                    })
        if rels:
            try:
                await neo4j.execute_write_batch(_REL_SECTION_PART_OF_STATUTE, rels)
                rel_count += len(rels)
            except Exception as exc:
                logger.warning("agent.graph_builder.rel_failed", rel="PART_OF", error=str(exc))

    # ── Step 3: Query Subgraph Context ────────────────────────────────────
    graph_context: dict[str, Any] = {
        "node_stats": stats,
        "relationship_count": rel_count,
        "subgraph": [],
    }

    try:
        subgraph_records = await neo4j.execute_read(
            _SUBGRAPH_CONTEXT_QUERY, parameters={"doc_id": document_id}
        )
        graph_context["subgraph"] = subgraph_records
    except Exception as exc:
        logger.warning("agent.graph_builder.context_query_failed", error=str(exc))

    duration_s = time.monotonic() - start_time
    logger.info(
        "agent.graph_builder.complete",
        document_id=document_id,
        nodes=sum(stats.values()),
        relationships=rel_count,
        duration_s=round(duration_s, 2),
    )

    return {
        "graph_context": graph_context,
        "graph_node_ids": all_node_ids,
        "current_phase": "graph_building",
        "execution_logs": [
            f"[graph_builder] Created {sum(stats.values())} nodes, {rel_count} relationships "
            f"for document {document_id} in {duration_s:.1f}s"
        ],
    }
