"""Centralized project settings for app runtime, ingestion, and encryption."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data/artifact paths (relative to repository root)
VECTOR_DB_DIR_NAME = "vector_db"
CAMPAIGN_DB_FILE_NAME = "campaign_memory.sqlite3"
CAMPAIGN_DB_ENC_FILE_NAME = "campaign_memory.sqlite3.enc"
VECTOR_DB_ENC_FILE_NAME = "vector_db.tar.gz.enc"

# Streamlit / app settings
# CHAT_* controls the fallback chat engine over lore index.
CHAT_EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6-luna"
CHAT_TEMPERATURE = 0.3
CHAT_SIMILARITY_TOP_K = 15
# AGENT_* controls the FunctionAgent used when use_agent=true.
AGENT_MODEL = "gpt-5.6-luna"
AGENT_TEMPERATURE = 0.3
AGENT_REASONING_EFFORT = "none"
AGENT_MEMORY_TOKEN_LIMIT = 4000
AGENT_DEBUG_MAX_CHARS = 1500

# Agent workflow settings (execution bounds and safety)
AGENT_MAX_ITERATIONS = 6
AGENT_TIMEOUT_SECONDS = 30
AGENT_ALLOW_PARALLEL_TOOL_CALLS = False
AGENT_EARLY_STOPPING_METHOD = "generate"

# Tool/query settings
# TOOL_* bounds per-tool output size.
TOOL_DEFAULT_LIMIT = 16
TOOL_MAX_LIMIT = 24
# OVERVIEW_* bounds the whole-campaign summary tool (covers every chapter, not just recent/matched events).
OVERVIEW_SESSION_LIMIT = 150
OVERVIEW_CHARS_PER_SESSION = 10000
# LORE_* controls how much raw lore retrieval context is returned to the agent.
LORE_RETRIEVER_TOP_K = 14
LORE_EXCERPT_TOP_K = 10
LORE_EXCERPT_MAX_CHARS = 900
# QUERY_* controls keyword extraction in SQL relevance search.
QUERY_TERM_LIMIT = 6
QUERY_MIN_TERM_LEN = 4
# CONTEXT_* controls compact memory text sent into fallback prompts.
CONTEXT_RECENT_LIMIT = 5
CONTEXT_RECENT_EVENTS_LIMIT = 8

# Lore ingestion settings (vector index build)
LORE_DEFAULT_FOLDER = ROOT / "data" / "sorted_data" / "lore"
LORE_PERSIST_DIR = ROOT / VECTOR_DB_DIR_NAME
LORE_SUPPORTED_EXTENSIONS = {".docx", ".txt", ".pdf", ".odt"}
LORE_CAMPAIGN_PATH_MARKERS = {"campaign_memory", "quests"}
# Chunking values passed to LlamaIndex SimpleNodeParser.
LORE_CHUNK_SIZE = 512
LORE_CHUNK_OVERLAP = 64

# Campaign ingestion settings (SQLite memory build)
CAMPAIGN_DIR = ROOT / "data" / "sorted_data" / "campaign_memory"
CAMPAIGN_TRANSCRIPT_FILE_NAME = "Timeline von Mountain Madness.docx"
CAMPAIGN_QUESTS_DIR_NAME = "quests"
# LLM extractor defaults for entity/claims enrichment.
CAMPAIGN_ENTITY_MODEL_DEFAULT = "gpt-5.6-luna"
# Environment variable names used by ingestion.
CAMPAIGN_ENTITY_MODEL_ENV = "CAMPAIGN_ENTITY_MODEL"
CAMPAIGN_ENTITY_API_KEY_ENV = "OPENAI_API_KEY"

# Encryption settings
# Binary envelope + KDF parameters for secure_artifacts.py.
ENCRYPTION_MAGIC = b"DNDSEC1"
ENCRYPTION_SALT_SIZE = 16
ENCRYPTION_NONCE_SIZE = 12
ENCRYPTION_PBKDF2_ITERATIONS = 390000
ENCRYPTION_DEFAULT_PASSPHRASE_ENV = "DATA_ENCRYPTION_PASSPHRASE"

# Secret names (Streamlit secrets.toml keys)
SECRET_OPENAI_API_KEY = "OPENAI_API_KEY"
SECRET_DATA_PASSPHRASE = "DATA_ENCRYPTION_PASSPHRASE"
SECRET_PASSWORD = "password"
SECRET_SYSTEM_PROMPT = "system_prompt"
SECRET_USE_AGENT = "use_agent"
SECRET_SHOW_AGENT_DEBUG = "show_agent_debug"
