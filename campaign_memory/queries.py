"""Read-only query helpers for the structured campaign memory."""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path

from settings import CONTEXT_RECENT_EVENTS_LIMIT, QUERY_MIN_TERM_LEN, QUERY_TERM_LIMIT, TOOL_DEFAULT_LIMIT


STOPWORDS = {
    "aber", "alle", "auch", "dass", "eine", "einer", "eines", "für",
    "haben", "hier", "ich", "ist", "mit", "nach", "oder", "sein", "sich",
    "sind", "über", "und", "von", "war", "was", "welche", "welcher", "wie",
    "wurde", "the", "what", "when", "where", "which", "with", "have", "this",
    "that", "from", "were", "they", "them", "about", "after", "before",
}


def get_sources(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT id, source_type, path, content_hash, imported_at FROM sources ORDER BY id"
    ).fetchall()


def get_recent_events(connection: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT e.id, e.session_id, s.title AS session_title, e.sequence, e.summary, e.status
        FROM events e
        JOIN sessions s ON s.id = e.session_id
        ORDER BY s.order_index DESC, e.sequence DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def get_relevant_events(
    connection: sqlite3.Connection,
    query: str,
    limit: int = TOOL_DEFAULT_LIMIT,
) -> list[sqlite3.Row]:
    """Find imported events containing meaningful words from a user question."""
    terms = [
        term
        for term in dict.fromkeys(re.findall(rf"[\wÄÖÜäöüß]{{{QUERY_MIN_TERM_LEN},}}", query.lower()))
        if term not in STOPWORDS
    ][:QUERY_TERM_LIMIT]
    if not terms:
        return []

    term_predicates = [
        "(LOWER(e.summary) LIKE ? OR LOWER(e.source_text) LIKE ? OR LOWER(s.title) LIKE ?)"
        for _ in terms
    ]
    predicates = " OR ".join(term_predicates)
    score = " + ".join(
        f"(CASE WHEN {predicate} THEN 1 ELSE 0 END)" for predicate in term_predicates
    )
    parameters = [value for term in terms for value in (f"%{term}%",) * 3]
    return connection.execute(
        f"""
        SELECT e.id, e.session_id, s.title AS session_title, e.sequence, e.summary, e.status,
               {score} AS match_score
        FROM events e
        JOIN sessions s ON s.id = e.session_id
        WHERE {predicates}
        ORDER BY match_score DESC, s.order_index DESC, e.sequence DESC
        LIMIT ?
        """,
        (*parameters, *parameters, limit),
    ).fetchall()


def get_recent_sections(connection: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT s.id, s.title, s.order_index,
               COALESCE((SELECT GROUP_CONCAT(e.summary, '\n')
                         FROM events e
                         WHERE e.session_id = s.id
                         ORDER BY e.sequence ASC), '') AS summary
        FROM sessions s
        ORDER BY s.order_index DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def get_session_events(connection: sqlite3.Connection, session_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT e.id, e.sequence, e.summary, e.status, e.source_text
        FROM events e
        WHERE e.session_id = ?
        ORDER BY e.sequence ASC
        """,
        (session_id,),
    ).fetchall()


def get_active_quests(
    connection: sqlite3.Connection,
    quest_type: str | None = None,
) -> list[sqlite3.Row]:
    query = """
        SELECT q.id, q.title, q.location, q.quest_type, q.status, q.content
        FROM quests q
        WHERE q.status IN ('open', 'in_progress', 'unknown', 'active', 'pending')
    """
    parameters: tuple[str, ...] = ()
    if quest_type in {"main", "side"}:
        query += " AND q.quest_type = ?"
        parameters = (quest_type,)
    query += " ORDER BY q.location, q.title"
    return connection.execute(query, parameters).fetchall()


def get_latest_session(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT id, source_id, session_number, title, order_index FROM sessions ORDER BY order_index DESC LIMIT 1"
    ).fetchone()


def get_campaign_memory_context(
    connection: sqlite3.Connection,
    recent_limit: int = 5,
    user_query: str = "",
) -> str:
    """Return a concise German summary for the current prompt from the campaign DB."""
    recent_sections = get_recent_sections(connection, limit=recent_limit)
    recent_events = get_recent_events(connection, limit=CONTEXT_RECENT_EVENTS_LIMIT)
    relevant_events = get_relevant_events(connection, user_query)
    active_quests = get_active_quests(connection)

    parts: list[str] = []

    if recent_sections:
        section_lines = [f"- {row['title']}" for row in recent_sections]
        parts.append("Kampagnen-Memory (neueste Abschnitte):\n" + "\n".join(section_lines))

    if recent_events:
        event_lines = [
            f"- {row['session_title']}, Ereignis {row['sequence']}: {row['summary'][:300]}"
            for row in recent_events
        ]
        parts.append("Zuletzt importierte Ereignisse:\n" + "\n".join(event_lines))

    if relevant_events:
        relevant_lines = [
            f"- {row['session_title']}, Ereignis {row['sequence']}: {row['summary'][:300]}"
            for row in relevant_events
        ]
        parts.append("Thematisch passende Ereignisse:\n" + "\n".join(relevant_lines))

    if active_quests:
        main_lines: list[str] = []
        side_lines: list[str] = []
        other_lines: list[str] = []
        for row in active_quests:
            location = row["location"] or "Unbekannt"
            line = f"- {location}: {row['title']} [{row['status']}]"
            if row["quest_type"] == "main":
                main_lines.append(line)
            elif row["quest_type"] == "side":
                side_lines.append(line)
            else:
                other_lines.append(line)
        quest_parts = []
        if main_lines:
            quest_parts.append("Hauptquests:\n" + "\n".join(main_lines))
        if side_lines:
            quest_parts.append("Nebenquests:\n" + "\n".join(side_lines))
        if other_lines:
            quest_parts.append("Nicht kategorisierte Quests:\n" + "\n".join(other_lines))
        parts.append("Aktive Quests:\n" + "\n\n".join(quest_parts))

    return "\n\n".join(parts)
