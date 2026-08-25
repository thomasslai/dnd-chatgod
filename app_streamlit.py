"""
Aglar DnD Chatgod - Streamlit Web App

This app provides a password-protected chatbot interface for DnD lore and game mechanics using OpenAI models and llama_index.

Features:
- Loads a vector database for document retrieval
- Uses OpenAI embeddings and chat models
- Password authentication for access
- Maintains and displays chat history
- Allows resetting chat history
- Streams responses from the chat engine
"""

from pathlib import Path
import logging
import re
import json

import streamlit as st
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import llama_index

from campaign_memory.agent import AgentStream, ToolCall, ToolCallResult, create_campaign_agent, run_agent
from campaign_memory.database import connect
from campaign_memory.queries import get_campaign_memory_context as query_campaign_memory_context
from secure_artifacts import decrypt_artifacts_for_runtime
from settings import (
    AGENT_DEBUG_MAX_CHARS,
    AGENT_MEMORY_TOKEN_LIMIT,
    AGENT_MODEL,
    AGENT_REASONING_EFFORT,
    AGENT_TEMPERATURE,
    CAMPAIGN_DB_ENC_FILE_NAME,
    CAMPAIGN_DB_FILE_NAME,
    CHAT_EMBED_MODEL,
    CHAT_MODEL,
    CHAT_SIMILARITY_TOP_K,
    CHAT_TEMPERATURE,
    CONTEXT_RECENT_LIMIT,
    SECRET_DATA_PASSPHRASE,
    SECRET_OPENAI_API_KEY,
    SECRET_PASSWORD,
    SECRET_SHOW_AGENT_DEBUG,
    SECRET_SYSTEM_PROMPT,
    SECRET_USE_AGENT,
    VECTOR_DB_DIR_NAME,
    VECTOR_DB_ENC_FILE_NAME,
)

persist_dir = VECTOR_DB_DIR_NAME
CAMPAIGN_DB_PATH = Path(__file__).resolve().parent / CAMPAIGN_DB_FILE_NAME
CAMPAIGN_DB_ENC_PATH = Path(__file__).resolve().parent / CAMPAIGN_DB_ENC_FILE_NAME
VECTOR_DB_ENC_PATH = Path(__file__).resolve().parent / VECTOR_DB_ENC_FILE_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_agent_response(response: str) -> str:
    """Remove a role label emitted at the start of an agent response."""
    return re.sub(r"^\s*assistant\s*:\s*", "", response, count=1, flags=re.IGNORECASE)


def ensure_runtime_data_ready() -> None:
    """Require encrypted artifacts and ensure decrypted runtime files are available."""
    root = Path(__file__).resolve().parent
    has_encrypted = CAMPAIGN_DB_ENC_PATH.exists() and VECTOR_DB_ENC_PATH.exists()

    if not has_encrypted:
        raise RuntimeError(
            "Encrypted artifacts are required. Missing one or both files: "
            "campaign_memory.sqlite3.enc, vector_db.tar.gz.enc"
        )

    passphrase = st.secrets.get(SECRET_DATA_PASSPHRASE, "")
    if not passphrase:
        raise RuntimeError(
            "DATA_ENCRYPTION_PASSPHRASE is required in Streamlit secrets to decrypt data artifacts."
        )

    decrypt_artifacts_for_runtime(
        passphrase=passphrase,
        sqlite_enc=CAMPAIGN_DB_ENC_PATH,
        vector_enc=VECTOR_DB_ENC_PATH,
        sqlite_out=CAMPAIGN_DB_PATH,
        vector_parent=root,
    )
    logger.info("Encrypted artifacts were decrypted for runtime access.")


def _short_debug_text(value: object, max_chars: int = AGENT_DEBUG_MAX_CHARS) -> str:
    """Format debug payloads into readable, bounded text for the UI."""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[gekürzt]"


def _extract_tool_result_text(event: ToolCallResult) -> str:
    """Best-effort extraction of tool result payload from LlamaIndex events."""
    for attr in ("tool_output", "tool_result", "result", "raw_output", "output"):
        if hasattr(event, attr):
            value = getattr(event, attr)
            if value is not None:
                return _short_debug_text(value)
    # Fallback: show full event representation if no known payload field exists.
    return _short_debug_text(event)

def get_index():
    """
    Load the vector index from persistent storage.
    Returns:
        index: The loaded vector index for document retrieval.
    """
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = llama_index.core.load_index_from_storage(storage_context)
    return index


def get_chat_engine():
    """
    Initialize and cache the chat engine in Streamlit session state.
    Uses OpenAI API key and system prompt from Streamlit secrets.
    Returns:
        chat_engine: The chat engine instance for conversation.
    """
    if "chat_engine" not in st.session_state:
        api_key = st.secrets[SECRET_OPENAI_API_KEY]
        embed_model = OpenAIEmbedding(model=CHAT_EMBED_MODEL, api_key=api_key)
        llm = OpenAI(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE, api_key=api_key)
        system_prompt = st.secrets[SECRET_SYSTEM_PROMPT]
        index = get_index()
        st.session_state["chat_engine"] = index.as_chat_engine(
            llm=llm,
            embed_model=embed_model,
            similarity_top_k=CHAT_SIMILARITY_TOP_K,
            chat_mode="best",
            system_prompt=system_prompt,
            streaming=True
        )
    return st.session_state["chat_engine"]


def get_agent():
    """Initialize the unified lore and campaign agent when explicitly enabled."""
    if "agent" not in st.session_state:
        api_key = st.secrets[SECRET_OPENAI_API_KEY]
        llm = OpenAI(
            model=AGENT_MODEL,
            reasoning_effort=AGENT_REASONING_EFFORT,
            api_key=api_key,
        )
        st.session_state["agent"] = create_campaign_agent(
            llm,
            database_path=CAMPAIGN_DB_PATH,
            lore_index=get_index(),
        )
    return st.session_state["agent"] 


def get_agent_memory() -> ChatMemoryBuffer:
    """Return the bounded conversation memory for the current Streamlit session."""
    if "agent_memory" not in st.session_state:
        st.session_state["agent_memory"] = ChatMemoryBuffer.from_defaults(token_limit=AGENT_MEMORY_TOKEN_LIMIT)
    return st.session_state["agent_memory"]


def get_campaign_memory_context(user_query: str = "") -> str:
    """Return a compact, read-only campaign memory summary for the current prompt."""
    if not CAMPAIGN_DB_PATH.exists():
        return ""

    connection = connect(CAMPAIGN_DB_PATH)
    try:
        return query_campaign_memory_context(connection, recent_limit=CONTEXT_RECENT_LIMIT, user_query=user_query)
    finally:
        connection.close()

# --- Conversation Chatbot UI ---

def main():
    """
    Main function to run the Streamlit app UI.
    Handles authentication, chat history, and user interaction.
    """
    st.title('Aglar DnD Chatgod')
    ensure_runtime_data_ready()

    # --- Simple password authentication ---
    PASSWORD = st.secrets[SECRET_PASSWORD]  # Password for access
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Check authentication status
    if not st.session_state["authenticated"]:
        st.subheader("🔒 Zugang geschützt")
        password_input = st.text_input("Bitte gib das Passwort ein:", type="password")
        if st.button("Login"):
            if password_input == PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Zugang gewährt!")
                st.rerun()
            else:
                st.error("Falsches Passwort.")
        st.stop()

    # --- Chatbot UI (only shown if authenticated) ---
    use_agent = st.secrets.get(SECRET_USE_AGENT, False)
    show_agent_debug = st.secrets.get(SECRET_SHOW_AGENT_DEBUG, False)
    chat_engine = None if use_agent else get_chat_engine()
    agent = get_agent() if use_agent else None
    agent_memory = get_agent_memory() if use_agent else None

    with st.sidebar:
        # Button to clear history and reset chat
        if st.button('Clear History & Reset Chat'):
            if chat_engine is not None:
                chat_engine.reset()
            if agent is not None:
                st.session_state.pop("agent", None)
                st.session_state.pop("agent_memory", None)
            st.session_state['messages'] = []
            st.success('Chat history cleared and chat reset.')

    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response streaming
    prompt = st.chat_input("Sprich mein Kind.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if agent is not None:
                with st.status("Denkt nach...", expanded=show_agent_debug) as agent_status:
                    def on_agent_event(event):
                        if isinstance(event, ToolCall):
                            message = f"Verwende Tool: `{event.tool_name}`"
                            logger.info("agent tool_call=%s", event.tool_name)
                            agent_status.write(message)
                        elif isinstance(event, ToolCallResult):
                            logger.info("agent tool_result=%s", event.tool_name)
                            agent_status.write(f"Tool abgeschlossen: `{event.tool_name}`")
                            if show_agent_debug:
                                agent_status.code(
                                    _extract_tool_result_text(event),
                                    language="json",
                                )
                        elif isinstance(event, AgentStream) and event.response:
                            message_placeholder.markdown(clean_agent_response(event.response) + "▌")

                    try:
                        full_response = run_agent(
                            agent,
                            prompt,
                            memory=agent_memory,
                            on_event=on_agent_event,
                        )
                    except Exception:
                        logger.exception("Agent request failed; using chat-engine fallback")
                        agent_status.write("Agent fehlgeschlagen, verwende Standard-Chat.")
                        fallback_engine = get_chat_engine()
                        context = get_campaign_memory_context(prompt)
                        effective_prompt = prompt
                        if context:
                            effective_prompt = (
                                "Nutze die folgende Kampagnen-Memory nur, wenn sie für die Antwort relevant ist.\n\n"
                                f"{context}\n\nUser-Frage: {prompt}"
                            )
                        streaming_response = fallback_engine.stream_chat(effective_prompt)
                        full_response = ""
                        for token in streaming_response.response_gen:
                            full_response += token
                            message_placeholder.markdown(clean_agent_response(full_response) + "▌")
                        full_response = clean_agent_response(full_response)
                    agent_status.update(label="Fertig", state="complete")
                message_placeholder.markdown(full_response)
            else:
                context = get_campaign_memory_context(prompt)
                effective_prompt = prompt
                if context:
                    effective_prompt = (
                        "Nutze die folgende Kampagnen-Memory nur, wenn sie für die Antwort relevant ist.\n\n"
                        f"{context}\n\nUser-Frage: {prompt}"
                    )
                streaming_response = chat_engine.stream_chat(effective_prompt)
                full_response = ""
                for token in streaming_response.response_gen:
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()


