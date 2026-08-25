"""Bounded, read-only tools for an agent using campaign memory."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from llama_index.core.tools import FunctionTool

from settings import (
    CAMPAIGN_DB_FILE_NAME,
    LORE_EXCERPT_MAX_CHARS,
    LORE_EXCERPT_TOP_K,
    LORE_RETRIEVER_TOP_K,
    TOOL_DEFAULT_LIMIT,
    TOOL_MAX_LIMIT,
)

from .database import connect
from .queries import get_active_quests, get_recent_events, get_relevant_events


logger = logging.getLogger(__name__)

DEFAULT_LIMIT = TOOL_DEFAULT_LIMIT
MAX_LIMIT = TOOL_MAX_LIMIT


def _limit(value: int) -> int:
    return max(1, min(value, MAX_LIMIT))


def _event_result(row: Any) -> dict[str, Any]:
    return {
        "section": row["session_title"],
        "sequence": row["sequence"],
        "summary": row["summary"],
        "status": row["status"],
    }


def get_latest_campaign_events(
    database_path: str = CAMPAIGN_DB_FILE_NAME,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Return the newest imported campaign events, newest first."""
    started = time.perf_counter()
    safe_limit = _limit(limit)
    logger.info("campaign tool=get_latest_campaign_events limit=%d", safe_limit)
    connection = connect(Path(database_path))
    try:
        results = [_event_result(row) for row in get_recent_events(connection, safe_limit)]
        logger.info(
            "campaign tool=get_latest_campaign_events results=%d duration_ms=%.1f",
            len(results),
            (time.perf_counter() - started) * 1000,
        )
        return results
    finally:
        connection.close()


def search_campaign_events(
    terms: list[str] | str,
    database_path: str = CAMPAIGN_DB_FILE_NAME,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Search campaign events using agent-selected terms and fixed SQL retrieval."""
    if isinstance(terms, str):
        raw_terms = [part for part in terms.replace(",", " ").split(" ") if part]
    else:
        raw_terms = [str(term) for term in terms]

    cleaned_terms = [term.strip() for term in raw_terms if term and term.strip()][:6]
    if not cleaned_terms:
        return []

    started = time.perf_counter()
    safe_limit = _limit(limit)
    logger.info("campaign tool=search_campaign_events terms=%s limit=%d", cleaned_terms, safe_limit)
    connection = connect(Path(database_path))
    try:
        results = [
            _event_result(row)
            for row in get_relevant_events(connection, " ".join(cleaned_terms), safe_limit)
        ]
        logger.info(
            "campaign tool=search_campaign_events results=%d duration_ms=%.1f",
            len(results),
            (time.perf_counter() - started) * 1000,
        )
        return results
    finally:
        connection.close()


def get_active_campaign_quests(
    location: str = "",
    quest_type: str = "",
    database_path: str = CAMPAIGN_DB_FILE_NAME,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Return active or uncertain quests, optionally restricted to a location."""
    normalized_location = location.strip().lower()
    normalized_type = quest_type.strip().lower()
    if normalized_type not in {"all", "main", "side"}:
        normalized_type = ""
    started = time.perf_counter()
    safe_limit = _limit(limit)
    logger.info(
        "campaign tool=get_active_campaign_quests location=%r quest_type=%r limit=%d",
        location[:100],
        normalized_type,
        safe_limit,
    )
    connection = connect(Path(database_path))
    try:
        rows = get_active_quests(
            connection,
            normalized_type if normalized_type in {"main", "side"} else None,
        )
        if normalized_location:
            rows = [
                row
                for row in rows
                if normalized_location in (row["location"] or "").lower()
            ]
        results = [
            {
                "title": row["title"],
                "location": row["location"],
                "quest_type": row["quest_type"],
                "status": row["status"],
                "content": row["content"][:500],
            }
            for row in rows[:safe_limit]
        ]
        logger.info(
            "campaign tool=get_active_campaign_quests results=%d duration_ms=%.1f",
            len(results),
            (time.perf_counter() - started) * 1000,
        )
        return results
    finally:
        connection.close()


def get_lore_search_tool(index: Any) -> FunctionTool:
    """Wrap the existing lore index as a read-only agent tool."""
    retriever = index.as_retriever(similarity_top_k=LORE_RETRIEVER_TOP_K)

    def search_lore(query: str) -> str:
        """Search stable Aglar lore and return retrieved source excerpts."""
        cleaned_query = query.strip()[:1000]
        if not cleaned_query:
            return "Keine Lore-Suchanfrage angegeben."

        nodes = retriever.retrieve(cleaned_query)
        if not nodes:
            return "Keine passenden Lore-Treffer gefunden."

        excerpts: list[str] = []
        for idx, node in enumerate(nodes[:LORE_EXCERPT_TOP_K], start=1):
            source = node.metadata.get("file_name") or node.metadata.get("file_path") or "unbekannte Quelle"
            text = node.get_content().strip().replace("\n\n", "\n")
            excerpt = text[:LORE_EXCERPT_MAX_CHARS]
            excerpts.append(f"[{idx}] Quelle: {source}\n{excerpt}")

        return "\n\n".join(excerpts)

    return FunctionTool.from_defaults(
        fn=search_lore,
        name="search_lore",
        description="Search stable Aglar world lore, rules, history, and calendar information.",
    )


def get_campaign_tools(database_path: str = CAMPAIGN_DB_FILE_NAME) -> list[FunctionTool]:
    """Create the read-only campaign tools exposed to an agent."""
    def _search_tool_adapter(terms: list[str] | str = "", query: str = "", limit: int = DEFAULT_LIMIT):
        search_input: list[str] | str = terms if terms else query
        return search_campaign_events(search_input, database_path, limit)

    return [
        FunctionTool.from_defaults(
            fn=lambda limit=DEFAULT_LIMIT: get_latest_campaign_events(database_path, limit),
            name="get_latest_campaign_events",
            description="Get the newest campaign events in chronological reverse order.",
        ),
        FunctionTool.from_defaults(
            fn=_search_tool_adapter,
            name="search_campaign_events",
            description="Search imported campaign events for specific names or topics.",
        ),
        FunctionTool.from_defaults(
            fn=lambda location="", quest_type="", limit=DEFAULT_LIMIT: get_active_campaign_quests(
                location, quest_type, database_path, limit
            ),
            name="get_active_campaign_quests",
            description="List active or uncertain campaign quests. Use quest_type='all' for both categories, or 'main'/'side' for one category.",
        ),
    ]