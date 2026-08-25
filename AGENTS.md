# Aglar DnD Chatgod

## Project map

- `app_streamlit.py` is the user-facing Streamlit chat app. It loads the persisted LlamaIndex index from `vector_db/`, creates the OpenAI chat and embedding models, and stores chat state in `st.session_state`.
- `parse_docs.py` converts PDF, ODT, DOCX, and TXT files to parsed TXT files while preserving the input folder structure.
- `ingest.py` reads the parsed-document folder, chunks documents, creates OpenAI embeddings, and persists a new index to `vector_db/`.
- `docs/` contains the current lore and rules source material; `data/` contains source/archive material. `notebooks/` is exploratory and is not part of the normal runtime.
- `vector_db/` is generated index state. Rebuild it through `ingest.py` rather than editing its JSON files manually.

## Run and rebuild

- Install the pinned dependencies with `python -m pip install -r requirements.txt`.
- Start the app with `streamlit run app_streamlit.py` or use the existing `.vscode/launch.json` configuration.
- Set `OPENAI_API_KEY`, `DOCS_RAW_FOLDER`, and `DOCS_PARSED_FOLDER` in the dotenv environment before running the document pipeline.
- Run `python parse_docs.py` to regenerate parsed text, then `python ingest.py` to rebuild `vector_db/`. These scripts perform work at module execution time.
- The Streamlit app expects `password`, `OPENAI_API_KEY`, and `system_prompt` in `.streamlit/secrets.toml`, plus a valid existing `vector_db/`.
- No automated test suite is configured. For Python-only changes, use `python -m compileall app_streamlit.py ingest.py parse_docs.py`; for app changes, run the Streamlit app and exercise authentication, chat, streaming, and reset behavior.

## Change guidance

- Keep the three responsibilities separate: parsing produces text, ingestion produces the persisted index, and the Streamlit app consumes the index.
- Treat the German roleplay text and system prompt as product behavior. Preserve the established German UI and character voice unless the task explicitly changes them.
- Keep paths and configuration explicit. The scripts currently use paths relative to the working directory, so run commands from the repository root.
- Be careful importing `parse_docs.py`: it currently runs the batch parser as a side effect of import. Do not import it from application code unless that behavior is intentionally changed.
- Do not print, copy, or commit values from `.env` or `.streamlit/secrets.toml`. Never put API keys or passwords in source, tests, documentation, or agent responses; rotate exposed credentials before treating them as secure.
- Avoid hand-editing generated data under `vector_db/` and avoid committing notebook outputs or temporary caches.
- Preserve the pinned dependency versions in `requirements.txt` unless a dependency upgrade is part of the task and the complete pipeline is revalidated.

## Documentation

- Use the source material in `docs/` and `data/` as the authority for lore and campaign facts.
- Add project-specific process documentation only when it cannot be represented by a focused code change or a link to the relevant source file.
