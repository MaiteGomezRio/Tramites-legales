import time
import uuid

import streamlit as st

from core.helpers import now_label

AUTO_CLOSE_STATUSES = {"ok", "no_match"}


def new_conversation(seed_assistant=True):
    cid = str(uuid.uuid4())
    convo = {
        "id": cid,
        "title": "Nuevo chat",
        "messages": [],
        "identified_tramite": None,
        "last_ms": None,
        "closed": False,
        "created_at": time.time(),
    }
    if seed_assistant:
        convo["messages"].append(
            {
                "role": "assistant",
                "content": "Hola 👋 Cuéntame tu caso y te digo el trámite del catálogo. Si me falta info, te haré preguntas para afinar.",
                "t": now_label(),
            }
        )
    return convo


def ensure_session_state():
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


def set_active(cid: str):
    st.session_state.active_id = cid


def delete_conversation(cid: str):
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if not st.session_state.conversations:
        c = new_conversation(seed_assistant=True)
        st.session_state.conversations = [c]
        st.session_state.active_id = c["id"]
    elif st.session_state.active_id == cid:
        st.session_state.active_id = st.session_state.conversations[0]["id"]


def create_and_switch():
    c = new_conversation(seed_assistant=True)
    st.session_state.conversations.insert(0, c)
    st.session_state.active_id = c["id"]
