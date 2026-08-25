"""Build the structured campaign-memory database from sorted campaign data.

Run from the repository root:
    python ingest_campaign.py
    python ingest_campaign.py --reset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from campaign_memory.database import connect
from campaign_memory.importers import import_quests, import_transcript
from secure_artifacts import (
    DEFAULT_SQLITE_ENC,
    DEFAULT_VECTOR_DIR,
    DEFAULT_VECTOR_ENC,
    run_encrypt,
)
from settings import (
    CAMPAIGN_DB_FILE_NAME,
    CAMPAIGN_DIR,
    CAMPAIGN_QUESTS_DIR_NAME,
    CAMPAIGN_TRANSCRIPT_FILE_NAME,
    ENCRYPTION_DEFAULT_PASSPHRASE_ENV,
    ROOT,
)


DATABASE_PATH = ROOT / CAMPAIGN_DB_FILE_NAME

# Ensure OPENAI_API_KEY and other local settings are available for CLI ingestion.
load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the structured campaign memory database.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing SQLite file before importing.")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="SQLite database path to write into.")
    parser.add_argument("--preview", action="store_true", help="Print the first transcript sections without importing.")
    parser.add_argument(
        "--entity-model",
        type=str,
        default=None,
        help="OpenAI model for entity extraction (default: CAMPAIGN_ENTITY_MODEL or gpt-4o-mini).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for ingestion output.",
    )
    parser.add_argument(
        "--skip-encrypt",
        action="store_true",
        help="Skip automatic encryption after successful ingestion.",
    )
    parser.add_argument(
        "--encryption-passphrase-env",
        type=str,
        default=ENCRYPTION_DEFAULT_PASSPHRASE_ENV,
        help="Environment variable containing the encryption passphrase.",
    )
    parser.add_argument(
        "--remove-plain-after-encrypt",
        action="store_true",
        help="Delete plaintext sqlite/vector artifacts after encryption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    transcript_path = CAMPAIGN_DIR / CAMPAIGN_TRANSCRIPT_FILE_NAME
    quest_directory = CAMPAIGN_DIR / CAMPAIGN_QUESTS_DIR_NAME
    logger.info(
        "Starting campaign ingestion database=%s reset=%s preview=%s entity_model=%s",
        args.database,
        args.reset,
        args.preview,
        args.entity_model or "default",
    )
    logger.debug("Transcript path: %s", transcript_path)
    logger.debug("Quest directory: %s", quest_directory)

    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    if args.preview:
        from campaign_memory.importers import preview_sections

        sections = preview_sections(transcript_path, limit=12)
        logger.info("Preview mode: %d sections found (showing up to 12)", len(sections))
        for idx, (title, line_count) in enumerate(sections, start=1):
            print(f"{idx}. {title} ({line_count} lines)")
        return

    if not quest_directory.exists():
        raise FileNotFoundError(quest_directory)

    if args.reset and args.database.exists():
        args.database.unlink()
        print(f"Removed existing database: {args.database}")
        logger.info("Removed existing database: %s", args.database)

    logger.info("Connecting to database: %s", args.database)
    connection = connect(args.database)
    try:
        logger.info("Importing transcript sections and events")
        import_transcript(connection, transcript_path, entity_model=args.entity_model)
        logger.info("Importing quest snapshots")
        quest_count = import_quests(connection, quest_directory)
        logger.info("Committing transaction")
        connection.commit()
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        session_count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        source_count = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        entity_count = connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        mention_count = connection.execute("SELECT COUNT(*) FROM event_entities").fetchone()[0]
        logger.info(
            "Import complete sources=%s sessions=%s events=%s quests=%s entities=%s claims=%s mentions=%s",
            source_count,
            session_count,
            event_count,
            quest_count,
            entity_count,
            claim_count,
            mention_count,
        )
        print(f"Imported {source_count} sources, {session_count} transcript sections, {event_count} events, and {quest_count} quest records.")
        print(f"Database: {args.database}")

        if args.skip_encrypt:
            logger.info("Skipping post-ingestion encryption due to --skip-encrypt")
        else:
            logger.info(
                "Running post-ingestion encryption sqlite=%s vector_dir=%s passphrase_env=%s remove_plain=%s",
                args.database,
                DEFAULT_VECTOR_DIR,
                args.encryption_passphrase_env,
                args.remove_plain_after_encrypt,
            )
            encrypt_args = argparse.Namespace(
                passphrase_env=args.encryption_passphrase_env,
                sqlite=Path(args.database),
                vector_dir=DEFAULT_VECTOR_DIR,
                sqlite_out=DEFAULT_SQLITE_ENC,
                vector_out=DEFAULT_VECTOR_ENC,
                remove_plain=args.remove_plain_after_encrypt,
            )
            run_encrypt(encrypt_args)
            logger.info("Post-ingestion encryption finished")
    finally:
        connection.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    main()
