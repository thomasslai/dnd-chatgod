"""Importers for campaign sources with LLM-based entity extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from settings import (
    CAMPAIGN_ENTITY_API_KEY_ENV,
    CAMPAIGN_ENTITY_MAX_CLAIMS,
    CAMPAIGN_ENTITY_MAX_ENTITIES,
    CAMPAIGN_ENTITY_MODEL_DEFAULT,
    CAMPAIGN_ENTITY_MODEL_ENV,
    ROOT,
)

from .database import upsert_source


SESSION_RE = re.compile(r"^\s*(?:SESSION|Session)\s+(\d+)\b", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+\d+\b.*$", re.IGNORECASE)
DAY_RE = re.compile(r"(?:tag|zeit).*a\.p\.?", re.IGNORECASE)
QUEST_RE = re.compile(r"^###\s+(?:!?\[([^]]+)\]\s*)?(.+?)\s*$")
LOCATION_RE = re.compile(r"^##\s+(.+?)\s*$")

logger = logging.getLogger(__name__)

load_dotenv(ROOT / ".env")

CALENDAR_MONTHS = (
    "diamant",
    "amethyst",
    "crystal",
    "topaz",
    "emerald",
    "sapphire",
    "ruby",
    "jade",
    "copper",
    "silver",
    "gold",
    "platin",
)

ENTITY_MODEL_DEFAULT = CAMPAIGN_ENTITY_MODEL_DEFAULT
ALLOWED_ENTITY_TYPES = {
    "character",
    "group",
    "organization",
    "location",
    "deity",
    "item",
    "event",
    "concept",
    "unknown",
}
ALLOWED_CLAIM_STATUS = {"confirmed", "uncertain", "rumor", "retcon", "contradicted"}
RELATION_TOKEN_RE = re.compile(r"^[a-z0-9_]{3,64}$")


def is_day_heading(line: str) -> bool:
    """Recognize in-world date headings like '12ter Tag des Diamanten 205 a.P.'"""
    if not line or len(line) > 220:
        return False
    lowered = line.lower()
    if "a.p" not in lowered:
        return False
    if "tag" not in lowered and "zeit" not in lowered:
        return False
    if not re.search(r"\d{1,4}", line):
        return False
    if any(month in lowered for month in CALENDAR_MONTHS):
        return True
    if re.search(r"\b(?:tag|zeit)\b", lowered):
        return True
    return False


def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def quest_status(marker: str | None, path: Path) -> str:
    """Normalize quest markers and snapshot-folder status into queryable values."""
    if marker:
        normalized = marker.strip().lower()
        if normalized in {"done", "completed", "complete", "x"}:
            return "done"
        if normalized in {"failed", "failure"}:
            return "failed"
        return normalized

    filename = path.name.lower()
    if "offen" in filename:
        return "open"
    if "abgeschlossen" in filename:
        return "done"
    return "unknown"


def quest_type(path: Path) -> str:
    """Infer main versus side quest from the snapshot filename."""
    filename = path.name.lower()
    if "main" in filename:
        return "main"
    if "side" in filename:
        return "side"
    return "unknown"


def read_docx(path: Path) -> str:
    logger.debug("Reading DOCX file: %s", path)
    document = Document(path)
    content = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    logger.debug("Loaded DOCX content lines=%d file=%s", len(content.splitlines()), path)
    return content


def normalize_entity_type(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    return normalized if normalized in ALLOWED_ENTITY_TYPES else "unknown"


def upsert_entity(connection, name: str, entity_type: str = "unknown") -> int:
    """Create or reuse an entity id."""
    connection.execute(
        """
        INSERT INTO entities (name, entity_type)
        VALUES (?, ?)
        ON CONFLICT(name, entity_type) DO NOTHING
        """,
        (name, entity_type),
    )
    row = connection.execute(
        "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
        (name, entity_type),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not upsert entity: {name}")
    return int(row["id"])


def normalize_claim_status(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in ALLOWED_CLAIM_STATUS else "uncertain"


def normalize_relation_token(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    if RELATION_TOKEN_RE.fullmatch(token):
        return token
    return "related_to"


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(text[start : end + 1])


def extract_entities_with_llm(
    client: OpenAI,
    model: str,
    event_text: str,
    max_entities: int = CAMPAIGN_ENTITY_MAX_ENTITIES,
    max_claims: int = CAMPAIGN_ENTITY_MAX_CLAIMS,
) -> dict:
    """Extract entities, mentions, and claims from one event line using an LLM."""
    logger.debug(
        "LLM extraction request model=%s text_len=%d max_entities=%d max_claims=%d",
        model,
        len(event_text),
        max_entities,
        max_claims,
    )
    system_prompt = (
        "Du extrahierst strukturierte Kampagnenfakten aus DnD-Sessiontext. "
        "Gib NUR valides JSON ohne Markdown zurueck. "
        "Ignoriere generische Nomen wie Angriff, Mangel, weniger, Gruppe, Dorf, Koenigreich, "
        "falls sie keine konkrete benannte Entitaet sind. "
        "Extrahiere nur explizit genannte Entitaeten und klar belegte Beziehungen."
    )
    user_prompt = (
        "Analysiere den folgenden Event-Text und gib JSON in genau diesem Schema zurueck:\n"
        "{\n"
        '  "entities": [{"name": "...", "entity_type": "character|group|organization|location|deity|item|event|concept|unknown", "description": "..."}],\n'
        '  "mentions": [{"name": "...", "role": "mentioned|speaker|target|ally|enemy"}],\n'
        '  "claims": [{"subject": "...", "predicate": "snake_case_relation", "value": "...", "status": "confirmed|uncertain|rumor|retcon|contradicted", "evidence": "..."}]\n'
        "}\n"
        f"Grenzen: max {max_entities} entities, max {max_claims} claims.\n"
        "Wenn unklar, lass Eintraege weg statt zu raten.\n\n"
        f"EVENT:\n{event_text}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = _extract_json_object(content)
    logger.debug("LLM extraction response size=%d", len(content))
    return payload if isinstance(payload, dict) else {}


def index_entities_and_claims_for_source(
    connection,
    source_id: int,
    entity_model: str | None = None,
) -> None:
    """Populate entities and claims for imported events using an LLM extractor."""
    event_rows = connection.execute(
        """
        SELECT e.id, e.summary, e.source_text
        FROM events e
        JOIN sessions s ON s.id = e.session_id
        WHERE s.source_id = ?
        ORDER BY s.order_index ASC, e.sequence ASC
        """,
        (source_id,),
    ).fetchall()

    connection.execute(
        """
        DELETE FROM claims
        WHERE source_event_id IN (
            SELECT e.id
            FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.source_id = ?
        )
        """,
        (source_id,),
    )
    connection.execute(
        """
        DELETE FROM event_entities
        WHERE event_id IN (
            SELECT e.id
            FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.source_id = ?
        )
        """,
        (source_id,),
    )

    api_key = os.getenv(CAMPAIGN_ENTITY_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"{CAMPAIGN_ENTITY_API_KEY_ENV} is required for LLM entity extraction.")

    model = (entity_model or os.getenv(CAMPAIGN_ENTITY_MODEL_ENV) or ENTITY_MODEL_DEFAULT).strip()
    client = OpenAI(api_key=api_key)
    logger.info(
        "Starting LLM entity extraction source_id=%s model=%s events=%d",
        source_id,
        model,
        len(event_rows),
    )

    extracted_events = 0
    failed_events = 0
    empty_payload_events = 0
    inserted_entities = 0
    inserted_mentions = 0
    inserted_claims = 0

    for row in event_rows:
        event_id = int(row["id"])
        source_text = row["source_text"] or row["summary"]

        try:
            payload = extract_entities_with_llm(client, model, source_text)
        except Exception as exc:
            failed_events += 1
            logger.warning("llm extraction failed for event_id=%s: %s", event_id, exc)
            continue

        entities_raw = payload.get("entities") if isinstance(payload, dict) else []
        mentions_raw = payload.get("mentions") if isinstance(payload, dict) else []
        claims_raw = payload.get("claims") if isinstance(payload, dict) else []

        if not isinstance(entities_raw, list):
            entities_raw = []
        if not isinstance(mentions_raw, list):
            mentions_raw = []
        if not isinstance(claims_raw, list):
            claims_raw = []

        entity_ids: dict[str, int] = {}

        for entity in entities_raw[:20]:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name or len(name) > 160:
                continue
            entity_type = normalize_entity_type(str(entity.get("entity_type", "unknown")))
            entity_id = upsert_entity(connection, name, entity_type)
            entity_ids[name] = entity_id
            inserted_entities += 1

            description = str(entity.get("description", "")).strip()
            if description:
                connection.execute(
                    """
                    UPDATE entities
                    SET description = CASE
                        WHEN description IS NULL OR description = '' THEN ?
                        ELSE description
                    END
                    WHERE id = ?
                    """,
                    (description[:600], entity_id),
                )

        if not entity_ids and not mentions_raw and not claims_raw:
            empty_payload_events += 1
            logger.debug("No entities/mentions/claims extracted event_id=%s", event_id)
            continue

        seen_event_entity_keys: set[tuple[int, str]] = set()
        for mention in mentions_raw[:30]:
            if not isinstance(mention, dict):
                continue
            name = str(mention.get("name", "")).strip()
            if not name:
                continue
            role = str(mention.get("role", "mentioned")).strip().lower().replace(" ", "_") or "mentioned"
            role = role[:40]
            entity_id = entity_ids.get(name)
            if entity_id is None:
                entity_id = upsert_entity(connection, name, "unknown")
                entity_ids[name] = entity_id
            key = (entity_id, role)
            if key in seen_event_entity_keys:
                continue
            seen_event_entity_keys.add(key)
            connection.execute(
                """
                INSERT OR IGNORE INTO event_entities (event_id, entity_id, role)
                VALUES (?, ?, ?)
                """,
                (event_id, entity_id, role),
            )
            inserted_mentions += 1

        for claim in claims_raw[:30]:
            if not isinstance(claim, dict):
                continue
            subject = str(claim.get("subject", "")).strip()
            predicate = normalize_relation_token(str(claim.get("predicate", "related_to")))
            value = str(claim.get("value", "")).strip()
            status = normalize_claim_status(str(claim.get("status", "uncertain")))
            evidence = str(claim.get("evidence", "")).strip() or source_text

            if not subject or not value:
                continue

            subject_id = entity_ids.get(subject)
            if subject_id is None:
                subject_id = upsert_entity(connection, subject, "unknown")
                entity_ids[subject] = subject_id

            connection.execute(
                """
                INSERT INTO claims (entity_id, predicate, value, status, source_event_id, source_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (subject_id, predicate, value[:500], status, event_id, evidence[:1000]),
            )
            inserted_claims += 1

        extracted_events += 1

    logger.info(
        "llm entity extraction completed source_id=%s events=%d extracted=%d failed=%d model=%s",
        source_id,
        len(event_rows),
        extracted_events,
        failed_events,
        model,
    )
    logger.info(
        "llm extraction details source_id=%s empty_events=%d entities=%d mentions=%d claims=%d",
        source_id,
        empty_payload_events,
        inserted_entities,
        inserted_mentions,
        inserted_claims,
    )


def split_transcript(content: str) -> list[tuple[int | None, str, str]]:
    """Split a transcript into chapter/day sections, preserving each section verbatim."""
    sections: list[tuple[int | None, str, str]] = []
    current_number: int | None = None
    current_title = "Opening"
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            sections.append((current_number, current_title, "\n".join(current_lines).strip()))

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        session_match = SESSION_RE.match(line)
        if session_match:
            flush()
            current_number = int(session_match.group(1))
            current_title = line
            current_lines = []
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            flush()
            current_number = None
            current_title = line
            current_lines = []
            continue

        day_match = is_day_heading(line)
        if day_match:
            flush()
            current_number = None
            current_title = line
            current_lines = []
            continue

        current_lines.append(line)
    flush()
    logger.debug("Split transcript into %d sections", len(sections))
    return sections


def preview_sections(path: Path, limit: int = 10) -> list[tuple[str, int]]:
    """Return a compact preview of the identified transcript sections."""
    content = read_docx(path)
    sections = split_transcript(content)
    previews: list[tuple[str, int]] = []
    for _, title, section in sections[:limit]:
        previews.append((title, len(section.splitlines())))
    return previews


def import_transcript(connection, path: Path, entity_model: str | None = None) -> int:
    logger.info("Importing transcript: %s", path)
    content = read_docx(path)
    source_id = upsert_source(connection, "reliable_transcript", str(path), file_hash(content), content)
    connection.execute(
        "DELETE FROM events WHERE session_id IN (SELECT id FROM sessions WHERE source_id = ?)",
        (source_id,),
    )
    connection.execute("DELETE FROM sessions WHERE source_id = ?", (source_id,))

    sections = split_transcript(content)
    logger.info("Transcript sections detected: %d", len(sections))

    imported_events = 0
    for order_index, (session_number, title, section) in enumerate(sections):
        session_cursor = connection.execute(
            """
            INSERT INTO sessions (source_id, session_number, title, order_index)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, session_number, title, order_index),
        )
        session_id = session_cursor.lastrowid
        event_lines = [line.strip() for line in section.splitlines() if line.strip()]
        logger.debug(
            "Importing section order_index=%d title=%s session_number=%s events=%d",
            order_index,
            title,
            session_number,
            len(event_lines),
        )
        for sequence, summary in enumerate(event_lines, start=1):
            connection.execute(
                """
                INSERT INTO events (session_id, sequence, summary, source_text)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, sequence, summary, summary),
            )
            imported_events += 1
    logger.info("Transcript import complete source_id=%s events=%d", source_id, imported_events)
    index_entities_and_claims_for_source(connection, source_id, entity_model=entity_model)
    return source_id


def import_quests(connection, quest_directory: Path) -> int:
    logger.info("Importing quests from directory: %s", quest_directory)
    imported = 0
    source_files = 0
    for path in sorted(quest_directory.rglob("*.md")):
        source_files += 1
        content = path.read_text(encoding="utf-8")
        source_id = upsert_source(connection, "quest_snapshot", str(path), file_hash(content), content)
        connection.execute("DELETE FROM quests WHERE source_id = ?", (source_id,))
        location = None
        current_title = None
        current_status = "unknown"
        current_type = quest_type(path)
        current_lines: list[str] = []

        def flush_quest() -> None:
            nonlocal imported, current_title, current_lines
            if current_title is None:
                return
            connection.execute(
                """
                INSERT INTO quests (source_id, title, location, quest_type, status, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    current_title,
                    location,
                    current_type,
                    current_status,
                    "\n".join(current_lines).strip(),
                ),
            )
            imported += 1
            current_title = None
            current_lines = []

        for line in content.splitlines():
            location_match = LOCATION_RE.match(line)
            quest_match = QUEST_RE.match(line)
            if location_match and current_type == "main":
                flush_quest()
                location = None
                current_status = quest_status(None, path)
                current_title = location_match.group(1).strip()
            elif location_match:
                flush_quest()
                location = location_match.group(1).strip()
            elif quest_match:
                flush_quest()
                current_status = quest_status(quest_match.group(1), path)
                current_title = quest_match.group(2).strip()
            elif current_title is not None:
                current_lines.append(line)
        flush_quest()
        logger.debug("Imported quest snapshot file=%s", path)
    logger.info("Quest import complete files=%d quests=%d", source_files, imported)
    return imported
