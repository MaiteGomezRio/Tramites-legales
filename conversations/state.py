import time
import uuid

import streamlit as st

from config import INITIAL_ASSISTANT_GREETING
from core.helpers import now_label


def new_conversation(seed_assistant: bool = True):
    cid = str(uuid.uuid4())
    convo = {
        "id": cid,
        "title": "Nuevo chat",
        "messages": [],
        "identified_tramite": None,
        "last_ms": None,
        "closed": False,
        "created_at": time.time(),
        "pending_question": None,
        "pending_hits": None,
        "desambiguacion_intentos": 0,
        "relevancia_reintentos": 0,
    }

    if seed_assistant:
        convo["messages"].append(
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_GREETING,
                "t": now_label(),
            }
        )
    return convo


def init_conversations() -> None:
    if "conversations" not in st.session_state:
        first = new_conversation(seed_assistant=True)
        st.session_state.conversations = [first]
        st.session_state.active_id = first["id"]


def get_active():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c

    c = new_conversation(seed_assistant=True)
    st.session_state.conversations.insert(0, c)
    st.session_state.active_id = c["id"]
    return c


def set_active(cid: str) -> None:
    st.session_state.active_id = cid


def delete_conversation(cid: str) -> None:
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if not st.session_state.conversations:
        c = new_conversation(seed_assistant=True)
        st.session_state.conversations = [c]
        st.session_state.active_id = c["id"]
    elif st.session_state.active_id == cid:
        st.session_state.active_id = st.session_state.conversations[0]["id"]


def create_and_switch() -> None:
    c = new_conversation(seed_assistant=True)
    st.session_state.conversations.insert(0, c)
    st.session_state.active_id = c["id"]
