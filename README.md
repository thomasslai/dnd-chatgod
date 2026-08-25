# Aglar DnD Chatgod

Streamlit chatbot for the Aglar campaign with two data domains:

- Stable lore retrieval from a LlamaIndex vector store
- Campaign chronology, quests, entities, and claims in SQLite

The app is configured for encrypted data artifacts in runtime/deployment.

## Architecture

- App entry: `app_streamlit.py`
- Campaign DB pipeline: `ingest_campaign.py` + `campaign_memory/`
- Lore vector pipeline: `ingest.py`
- Encryption/decryption utility: `secure_artifacts.py`
- Centralized parameters: `settings.py`

Data flow:

1. Lore docs are parsed/chunked and embedded into `vector_db/`
2. Campaign transcript/quest snapshots are imported into `campaign_memory.sqlite3`
3. Artifacts are encrypted to:
	- `campaign_memory.sqlite3.enc`
	- `vector_db.tar.gz.enc`
4. App startup decrypts artifacts using Streamlit secret `DATA_ENCRYPTION_PASSPHRASE`

## Prerequisites

- Python `>=3.13`
- `uv`
- OpenAI API key

Install dependencies:

```powershell
uv sync
```

## Configuration

### Local `.env`

Create a root `.env` file with at least:

```dotenv
OPENAI_API_KEY=...
DATA_ENCRYPTION_PASSPHRASE=...
CAMPAIGN_ENTITY_MODEL=gpt-4o-mini
```

Notes:

- `OPENAI_API_KEY` is used by ingestion and LLM extraction
- `DATA_ENCRYPTION_PASSPHRASE` is used by encryption CLI utilities

### Streamlit secrets

Create `.streamlit/secrets.toml` for local app usage (and mirror in Streamlit Cloud):

```toml
use_agent = true
show_agent_debug = false
password = "..."
OPENAI_API_KEY = "..."
DATA_ENCRYPTION_PASSPHRASE = "..."
system_prompt = """..."""
```

## Run the app

```powershell
uv run streamlit run app_streamlit.py
```

## Ingestion workflows

### 1) Build lore vector store

```powershell
uv run python ingest.py
```

This reads lore input (default: `data/sorted_data/lore`) and writes `vector_db/`.

### 2) Build campaign memory DB

```powershell
uv run python ingest_campaign.py --reset
```

Important:

- LLM-based entity extraction is enabled in campaign ingestion
- Logging level can be changed with `--log-level DEBUG|INFO|WARNING|ERROR`
- By default, ingestion encrypts artifacts after successful import

Useful flags:

```powershell
uv run python ingest_campaign.py --reset --log-level DEBUG
uv run python ingest_campaign.py --skip-encrypt
uv run python ingest_campaign.py --remove-plain-after-encrypt
uv run python ingest_campaign.py --entity-model gpt-4o-mini
```

## Encryption and decryption

Encrypt artifacts:

```powershell
uv run python secure_artifacts.py encrypt --passphrase-env DATA_ENCRYPTION_PASSPHRASE
```

Encrypt and delete plaintext artifacts:

```powershell
uv run python secure_artifacts.py encrypt --passphrase-env DATA_ENCRYPTION_PASSPHRASE --remove-plain
```

Manual decrypt (local utility):

```powershell
uv run python secure_artifacts.py decrypt --passphrase-env DATA_ENCRYPTION_PASSPHRASE
```

## Encrypted-only runtime behavior

`app_streamlit.py` enforces encrypted artifacts:

- Requires:
  - `campaign_memory.sqlite3.enc`
  - `vector_db.tar.gz.enc`
  - Streamlit secret `DATA_ENCRYPTION_PASSPHRASE`
- Decrypts at startup via `decrypt_artifacts_for_runtime(...)`

If encrypted files or passphrase are missing, startup fails fast.

## Deployment (Streamlit Cloud)

1. Commit code and encrypted artifacts:
	- `campaign_memory.sqlite3.enc`
	- `vector_db.tar.gz.enc`
2. Do not commit plaintext artifacts:
	- `campaign_memory.sqlite3`
	- `vector_db/`
3. Configure Streamlit secrets:
	- `OPENAI_API_KEY`
	- `DATA_ENCRYPTION_PASSPHRASE`
	- `password`
	- `system_prompt`

## Troubleshooting

### `search_campaign_events` returns empty

The tool supports both:

- `terms=["kyrr"]`
- `query="kyrr"`

If still empty, confirm campaign DB was rebuilt and decrypted successfully.

### Entity extraction fails

Check:

- `OPENAI_API_KEY` availability
- model availability (`CAMPAIGN_ENTITY_MODEL` / `--entity-model`)
- ingestion logs with `--log-level DEBUG`

### App fails on startup with encryption error

Check:

- `.enc` files exist in repo root
- `DATA_ENCRYPTION_PASSPHRASE` exists in Streamlit secrets

## Security checklist

- Rotate any exposed API keys/passwords before publish
- Keep `.env` and `.streamlit/secrets.toml` out of Git
- Prefer `--remove-plain-after-encrypt` for release artifacts
