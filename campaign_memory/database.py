"""SQLite persistence for immutable campaign sources and derived records."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_metadata (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL UNIQUE REFERENCES sources(id),
    title TEXT,
    description TEXT,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    session_number INTEGER,
    title TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    UNIQUE(source_id, order_index)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    sequence INTEGER NOT NULL,
    summary TEXT NOT NULL,
    source_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'imported',
    UNIQUE(session_id, sequence)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    description TEXT,
    UNIQUE(name, entity_type)
);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id INTEGER NOT NULL REFERENCES events(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    role TEXT NOT NULL DEFAULT 'mentioned',
    PRIMARY KEY(event_id, entity_id, role)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uncertain',
    source_event_id INTEGER REFERENCES events(id),
    source_text TEXT NOT NULL,
    context_summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS quests (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    title TEXT NOT NULL,
    location TEXT,
    quest_type TEXT NOT NULL DEFAULT 'unknown',
    status TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(source_id, title, location)
);

CREATE INDEX IF NOT EXISTS idx_events_order ON events(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_quests_status ON quests(status);
"""


def _backfill_session_content_hashes(connection: sqlite3.Connection) -> int:
    """Populate session hashes for legacy databases so incremental diffing works."""
    legacy_rows = connection.execute(
        """
        SELECT id, title
        FROM sessions
        WHERE content_hash IS NULL OR content_hash = ''
        ORDER BY order_index ASC
        """,
    ).fetchall()
    if not legacy_rows:
        return 0

    updated = 0
    for row in legacy_rows:
        session_id = int(row["id"])
        event_rows = connection.execute(
            "SELECT summary FROM events WHERE session_id = ? ORDER BY sequence ASC",
            (session_id,),
        ).fetchall()
        section_text = "\n".join(str(event["summary"]) for event in event_rows)
        content_hash = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
        connection.execute(
            "UPDATE sessions SET content_hash = ? WHERE id = ?",
            (content_hash, session_id),
        )
        updated += 1
    return updated


def connect(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    session_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
    }
    if "content_hash" not in session_columns:
        connection.execute(
            "ALTER TABLE sessions ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
        )
    _backfill_session_content_hashes(connection)
    quest_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(quests)").fetchall()
    }
    if "quest_type" not in quest_columns:
        connection.execute(
            "ALTER TABLE quests ADD COLUMN quest_type TEXT NOT NULL DEFAULT 'unknown'"
        )
    claim_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(claims)").fetchall()
    }
    if "context_summary" not in claim_columns:
        connection.execute(
            "ALTER TABLE claims ADD COLUMN context_summary TEXT NOT NULL DEFAULT ''"
        )
    return connection


def upsert_source(
    connection: sqlite3.Connection,
    source_type: str,
    path: str,
    content_hash: str,
    content: str,
) -> int:
    connection.execute(
        """
        INSERT INTO sources (source_type, path, content_hash, content)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            source_type = excluded.source_type,
            content_hash = excluded.content_hash,
            content = excluded.content
        """,
        (source_type, path, content_hash, content),
    )
    row = connection.execute("SELECT id FROM sources WHERE path = ?", (path,)).fetchone()
    if row is None:
        raise RuntimeError(f"Could not store source: {path}")
    return int(row["id"])
