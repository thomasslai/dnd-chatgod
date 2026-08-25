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
) -> dict:
    """Extract entities, mentions, and claims from one event line using an LLM."""
    logger.debug("LLM extraction request model=%s text_len=%d", model, len(event_text))
    system_prompt = (
        "Du extrahierst strukturierte Kampagnenfakten aus DnD-Sessiontext. "
        "Gib NUR ein valides JSON-Objekt zurueck, ohne Markdown oder Erklaerungen. "
        "Beschreibungen sollen konkrete Informationen enthalten: Wer/was ist es, wo ist es, was macht es, "
        "welche Rolle hat es im Setting, und welche Beziehung besteht zu bekannten Personen oder Gruppen. "
        "Vermeide generische Fragmente wie 'Reich', 'Ort', 'Gruppe', 'Dorf' ohne zusätzliche kontextuelle Details. "
        "Benenne Orte, Situationen, Gruppen, Charaktere, Gegenstände, Zeitangaben, usw. immer explizit, wenn sie im Text vorkommen. "
        "Vermeide vage oder unbestimmte Begriffe, wie 'jemand', 'er/sie/es', 'dort', 'dieser Ort', 'diese Gruppe', usw. "
        "Die Informationen müssen immer eigentständig und ohne Kontext verständlich sein. "
        "Extrahiere nur explizit genannte Entitaeten und klar belegte Beziehungen."
    )
    user_prompt = (
        "Analysiere den folgenden Event-Text und gib ein JSON-Objekt in genau diesem Schema zurueck:\n"
        "{\n"
        '  "entities": [{"name": "...", "entity_type": "character|group|organization|location|deity|item|event|concept|unknown", "description": "..."}],\n'
        '  "mentions": [{"name": "...", "role": "mentioned|speaker|target|ally|enemy"}],\n'
        '  "claims": [{"subject": "...", "predicate": "snake_case_relation", "value": "...", "status": "confirmed|uncertain|rumor|retcon|contradicted", "evidence": "...", "context_summary": "..."}]\n'
        "}\n"
        "Fuer jeden Claim: context_summary muss in zwei bis drei kurzen Saetzen erklaeren, in welcher Situation, an welchem Ort, waehrend welcher Auseinandersetzung oder mit welcher Rolle dieser Claim relevant ist. Verwende explizit Namen, vermeide Verweise wie 'er', 'sie', 'es', 'dieser Ort', 'diese Gruppe', usw. "
        "Beziehe nur Kontext ein, der im Event-Text belegt ist. Wenn kein zusaetzlicher Kontext vorhanden ist, beschreibe knapp die Situation aus dem Event-Text.\n"
        "Wenn unklar, lass Eintraege weg statt zu raten.\n\n"
        f"EVENT:\n{event_text}"
    )
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="low",
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


def summarize_entity_description_with_llm(
    client: OpenAI,
    model: str,
    entity_name: str,
    context_lines: list[str],
    existing_description: str | None = None,
) -> str:
    """Derive a durable, canonical entity description from all available context."""
    if not context_lines:
        return existing_description or ""

    compact_context = "\n".join(f"- {line.strip()}" for line in context_lines if line and line.strip())
    system_prompt = (
        "Du schreibst eine dauerhafte, fundierte Beschreibung einer Kampagnen-Entitaet. "
        "Gib NUR ein valides JSON-Objekt zurueck, ohne Markdown oder Erklaerungen. "
        "Ziel ist die Beschreibung der grundlegenden Identitaet, Rolle und Bedeutung im Setting. "
        "Nicht der letzte Plot-Status, sondern das, was langfristig wahr bleibt. "
        "Nutze nur Informationen aus dem vorliegenden Kontext. "
        "Wenn der Kontext schwach ist, gib eine leere Zeichenkette zurueck statt zu raten."
    )
    user_prompt = (
        "Erstelle eine kurze, aber informative Beschreibung der Entitaet und gib ein JSON-Objekt zurueck. "
        "Das JSON-Objekt muss genau dieses Schema haben: {\"description\": \"...\"}. "
        "Die Beschreibung soll eine stabile Identitaet beschreiben: Wer/was ist die Entitaet, welche Herkunft hat sie, "
        "welche Rolle spielt sie, und welche groessere Bedeutung hat sie im Setting. "
        "Vermeide temporaere Zustandsaenderungen, aktuelle Lage, Geruechte, oder einzelne Plot-Fragmente. "
        "Wenn der Kontext keine klaren dauerhaften Fakten liefert, gib {\"description\": \"\"} zurueck.\n\n"
        f"Entitaet: {entity_name}\n"
        f"Vorhandene Beschreibung: {existing_description or 'keine'}\n\n"
        "Kontext:\n"
        f"{compact_context}"
    )
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = _extract_json_object(content)
    if not isinstance(payload, dict):
        return existing_description or ""
    value = str(payload.get("description") or payload.get("summary") or "").strip()
    return value


def finalize_entity_descriptions(
    connection,
    source_id: int,
    entity_model: str | None = None,
    entity_ids: set[int] | None = None,
) -> None:
    """Rewrite entity descriptions as stable identity summaries based on all mentions in the source.

    When `entity_ids` is given, only those entities are re-summarized (e.g. the ones
    touched by an incremental update), instead of every entity in the source.
    """
    if entity_ids is not None and not entity_ids:
        logger.info("Skipping final entity description pass source_id=%s: no touched entities.", source_id)
        return

    query = """
        SELECT DISTINCT e.id, e.name, e.description
        FROM entities e
        JOIN event_entities ee ON ee.entity_id = e.id
        JOIN events ev ON ev.id = ee.event_id
        JOIN sessions s ON s.id = ev.session_id
        WHERE s.source_id = ?
    """
    parameters: tuple = (source_id,)
    if entity_ids is not None:
        placeholders = ",".join("?" for _ in entity_ids)
        query += f" AND e.id IN ({placeholders})"
        parameters = (source_id, *entity_ids)
    query += " ORDER BY e.name"
    entity_rows = connection.execute(query, parameters).fetchall()

    api_key = os.getenv(CAMPAIGN_ENTITY_API_KEY_ENV, "").strip()
    if not api_key:
        logger.warning("Skipping final entity description pass: %s is not configured.", CAMPAIGN_ENTITY_API_KEY_ENV)
        return

    model = (entity_model or os.getenv(CAMPAIGN_ENTITY_MODEL_ENV) or ENTITY_MODEL_DEFAULT).strip()
    client = OpenAI(api_key=api_key)
    logger.info(
        "Starting final entity description pass source_id=%s model=%s entities=%d",
        source_id,
        model,
        len(entity_rows),
    )

    processed = 0
    updated = 0
    skipped = 0

    for row in entity_rows:
        entity_id = int(row["id"])
        entity_name = str(row["name"])
        existing_description = row["description"]
        processed += 1

        related_sessions = connection.execute(
            """
            SELECT DISTINCT s.id, s.title
            FROM event_entities ee
            JOIN events ev ON ev.id = ee.event_id
            JOIN sessions s ON s.id = ev.session_id
            WHERE ee.entity_id = ? AND s.source_id = ?
            ORDER BY s.order_index ASC
            """,
            (entity_id, source_id),
        ).fetchall()

        if not related_sessions:
            skipped += 1
            continue

        context_lines: list[str] = []
        for session_row in related_sessions:
            session_id = int(session_row["id"])
            session_title = session_row["title"]
            session_events = connection.execute(
                """
                SELECT ev.summary, ev.source_text
                FROM events ev
                WHERE ev.session_id = ?
                ORDER BY ev.sequence ASC
                """,
                (session_id,),
            ).fetchall()
            if session_title:
                context_lines.append(f"Session: {session_title}")
            for event in session_events:
                text = (event["source_text"] or event["summary"] or "").strip()
                if text:
                    context_lines.append(text)

        if not context_lines:
            skipped += 1
            continue

        canonical = summarize_entity_description_with_llm(
            client,
            model,
            entity_name,
            context_lines,
            existing_description=existing_description,
        )
        if not canonical:
            logger.debug("Final description skipped for entity_id=%s name=%s; no stable context available.", entity_id, entity_name)
            skipped += 1
            continue

        if canonical != (existing_description or ""):
            connection.execute(
                "UPDATE entities SET description = ? WHERE id = ?",
                (canonical, entity_id),
            )
            updated += 1
            logger.debug(
                "Updated final description entity_id=%s name=%s old_len=%s new_len=%s",
                entity_id,
                entity_name,
                len(existing_description or ""),
                len(canonical),
            )

    logger.info(
        "Final entity description pass complete source_id=%s processed=%d updated=%d skipped=%d",
        source_id,
        processed,
        updated,
        skipped,
    )


def index_entities_and_claims_for_events(
    connection,
    event_ids: list[int],
    entity_model: str | None = None,
) -> set[int]:
    """Populate entities and claims for a specific set of (new/changed) events only.

    Only these events pay the LLM extraction cost, so unchanged sessions from a
    previous import are left untouched. Returns the set of touched entity ids.
    """
    if not event_ids:
        return set()

    placeholders = ",".join("?" for _ in event_ids)
    event_rows = connection.execute(
        f"""
        SELECT e.id, e.summary, e.source_text
        FROM events e
        WHERE e.id IN ({placeholders})
        ORDER BY e.session_id ASC, e.sequence ASC
        """,
        event_ids,
    ).fetchall()

    connection.execute(
        f"DELETE FROM claims WHERE source_event_id IN ({placeholders})",
        event_ids,
    )
    connection.execute(
        f"DELETE FROM event_entities WHERE event_id IN ({placeholders})",
        event_ids,
    )

    api_key = os.getenv(CAMPAIGN_ENTITY_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"{CAMPAIGN_ENTITY_API_KEY_ENV} is required for LLM entity extraction.")

    model = (entity_model or os.getenv(CAMPAIGN_ENTITY_MODEL_ENV) or ENTITY_MODEL_DEFAULT).strip()
    client = OpenAI(api_key=api_key)
    logger.info(
        "Starting LLM entity extraction model=%s events=%d",
        model,
        len(event_rows),
    )

    extracted_events = 0
    failed_events = 0
    empty_payload_events = 0
    inserted_entities = 0
    inserted_mentions = 0
    inserted_claims = 0
    touched_entity_ids: set[int] = set()

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

        for entity in entities_raw:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_type = normalize_entity_type(str(entity.get("entity_type", "unknown")))
            entity_id = upsert_entity(connection, name, entity_type)
            entity_ids[name] = entity_id
            touched_entity_ids.add(entity_id)
            inserted_entities += 1

            connection.execute(
                """
                INSERT OR IGNORE INTO event_entities (event_id, entity_id, role)
                VALUES (?, ?, 'mentioned')
                """,
                (event_id, entity_id),
            )

        if not entity_ids and not mentions_raw and not claims_raw:
            empty_payload_events += 1
            logger.debug("No entities/mentions/claims extracted event_id=%s", event_id)
            continue

        seen_event_entity_keys: set[tuple[int, str]] = set()
        for mention in mentions_raw:
            if not isinstance(mention, dict):
                continue
            name = str(mention.get("name", "")).strip()
            if not name:
                continue
            role = str(mention.get("role", "mentioned")).strip().lower().replace(" ", "_") or "mentioned"
            entity_id = entity_ids.get(name)
            if entity_id is None:
                entity_id = upsert_entity(connection, name, "unknown")
                entity_ids[name] = entity_id
            touched_entity_ids.add(entity_id)
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

        for claim in claims_raw:
            if not isinstance(claim, dict):
                continue
            subject = str(claim.get("subject", "")).strip()
            predicate = normalize_relation_token(str(claim.get("predicate", "related_to")))
            value = str(claim.get("value", "")).strip()
            status = normalize_claim_status(str(claim.get("status", "uncertain")))
            evidence = str(claim.get("evidence", "")).strip() or source_text
            context_summary = (
                str(claim.get("context_summary", "")).strip()
                or str(claim.get("context", "")).strip()
                or evidence
            )

            if not subject or not value:
                continue

            subject_id = entity_ids.get(subject)
            if subject_id is None:
                subject_id = upsert_entity(connection, subject, "unknown")
                entity_ids[subject] = subject_id
            touched_entity_ids.add(subject_id)

            connection.execute(
                """
                INSERT INTO claims (entity_id, predicate, value, status, source_event_id, source_text, context_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (subject_id, predicate, value, status, event_id, evidence, context_summary),
            )
            inserted_claims += 1

        extracted_events += 1

    logger.info(
        "llm entity extraction completed events=%d extracted=%d failed=%d model=%s",
        len(event_rows),
        extracted_events,
        failed_events,
        model,
    )
    logger.info(
        "llm extraction details empty_events=%d entities=%d mentions=%d claims=%d",
        empty_payload_events,
        inserted_entities,
        inserted_mentions,
        inserted_claims,
    )
    return touched_entity_ids


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
    """Import the transcript incrementally: only changed/new chapters pay the LLM extraction cost.

    Chapters are matched to existing sessions by title. A section whose content hash
    matches the stored session is left untouched (events, entities, claims all kept).
    Changed or new sections are reimported and re-extracted; removed sections are deleted.
    """
    logger.info("Importing transcript: %s", path)
    content = read_docx(path)
    source_id = upsert_source(connection, "reliable_transcript", str(path), file_hash(content), content)

    existing_sessions = {
        row["title"]: row
        for row in connection.execute(
            "SELECT id, title, content_hash FROM sessions WHERE source_id = ?",
            (source_id,),
        ).fetchall()
    }

    sections = split_transcript(content)
    logger.info("Transcript sections detected: %d", len(sections))

    matched_session_ids: set[int] = set()
    changed_event_ids: list[int] = []
    imported_events = 0
    unchanged_sections = 0

    for order_index, (session_number, title, section) in enumerate(sections):
        section_hash = file_hash(section)
        existing = existing_sessions.get(title)

        if existing is not None and existing["content_hash"] == section_hash:
            session_id = int(existing["id"])
            connection.execute(
                "UPDATE sessions SET order_index = ?, session_number = ? WHERE id = ?",
                (order_index, session_number, session_id),
            )
            matched_session_ids.add(session_id)
            unchanged_sections += 1
            continue

        if existing is not None:
            session_id = int(existing["id"])
            connection.execute(
                """
                DELETE FROM claims WHERE source_event_id IN (
                    SELECT id FROM events WHERE session_id = ?
                )
                """,
                (session_id,),
            )
            connection.execute(
                """
                DELETE FROM event_entities WHERE event_id IN (
                    SELECT id FROM events WHERE session_id = ?
                )
                """,
                (session_id,),
            )
            connection.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            connection.execute(
                "UPDATE sessions SET order_index = ?, session_number = ?, content_hash = ? WHERE id = ?",
                (order_index, session_number, section_hash, session_id),
            )
        else:
            session_cursor = connection.execute(
                """
                INSERT INTO sessions (source_id, session_number, title, order_index, content_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, session_number, title, order_index, section_hash),
            )
            session_id = session_cursor.lastrowid

        matched_session_ids.add(session_id)
        event_lines = [line.strip() for line in section.splitlines() if line.strip()]
        logger.debug(
            "Importing changed section order_index=%d title=%s session_number=%s events=%d",
            order_index,
            title,
            session_number,
            len(event_lines),
        )
        for sequence, summary in enumerate(event_lines, start=1):
            event_cursor = connection.execute(
                """
                INSERT INTO events (session_id, sequence, summary, source_text)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, sequence, summary, summary),
            )
            changed_event_ids.append(int(event_cursor.lastrowid))
            imported_events += 1

    removed_session_ids = [row["id"] for row in existing_sessions.values() if row["id"] not in matched_session_ids]
    for session_id in removed_session_ids:
        connection.execute(
            """
            DELETE FROM claims WHERE source_event_id IN (
                SELECT id FROM events WHERE session_id = ?
            )
            """,
            (session_id,),
        )
        connection.execute(
            """
            DELETE FROM event_entities WHERE event_id IN (
                SELECT id FROM events WHERE session_id = ?
            )
            """,
            (session_id,),
        )
        connection.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    logger.info(
        "Transcript import complete source_id=%s new_events=%d unchanged_sections=%d removed_sections=%d",
        source_id,
        imported_events,
        unchanged_sections,
        len(removed_session_ids),
    )

    if changed_event_ids:
        touched_entity_ids = index_entities_and_claims_for_events(connection, changed_event_ids, entity_model=entity_model)
        finalize_entity_descriptions(connection, source_id, entity_model=entity_model, entity_ids=touched_entity_ids)
    else:
        logger.info("No transcript changes detected source_id=%s; skipping LLM extraction.", source_id)

    return source_id



def import_quests(connection, quest_directory: Path) -> int:
    logger.info("Importing quests from directory: %s", quest_directory)
    imported = 0
    source_files = 0
    skipped_files = 0
    for path in sorted(quest_directory.rglob("*.md")):
        source_files += 1
        content = path.read_text(encoding="utf-8")
        new_hash = file_hash(content)
        existing = connection.execute(
            "SELECT id, content_hash FROM sources WHERE path = ?",
            (str(path),),
        ).fetchone()
        if existing is not None and existing["content_hash"] == new_hash:
            imported += connection.execute(
                "SELECT COUNT(*) FROM quests WHERE source_id = ?", (existing["id"],)
            ).fetchone()[0]
            skipped_files += 1
            logger.debug("Quest snapshot unchanged, skipping reparse file=%s", path)
            continue

        source_id = upsert_source(connection, "quest_snapshot", str(path), new_hash, content)
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
    logger.info(
        "Quest import complete files=%d quests=%d skipped_unchanged=%d",
        source_files,
        imported,
        skipped_files,
    )
    return imported
