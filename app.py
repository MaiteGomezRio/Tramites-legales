import streamlit as st

from agents.client import load_agents
from agents.services import (
    call_agente_clarificador,
    call_asistente_tramites,
    clasificar_con_desambiguacion,
)
from config import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    ASSISTANT_AVATAR,
    CSS_PATH,
    MENSAJE_FUERA_DE_AMBITO,
    MENSAJE_SCORE_BAJO,
    UMBRAL_AMBIGUEDAD,
    USER_AVATAR,
)
from conversations.state import (
    create_and_switch,
    delete_conversation,
    ensure_session_state,
    get_active,
    new_conversation,
    set_active,
)
from core.helpers import make_title_from_text, normalize_role, now_label
from core.retrieval import load_catalog_and_embeddings, render_debug_info
from ui.styles import load_css

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)
load_css(CSS_PATH)

catalog_df, embeddings = load_catalog_and_embeddings()
agents = load_agents()

ensure_session_state()

with st.sidebar:
    st.image("assets/logo_junta_fondo_verde.jpeg", width="stretch")
    st.markdown("## Conversaciones")

    if st.button("➕ Nuevo chat", use_container_width=True):
        create_and_switch()
        st.rerun()

    st.write("")

    for c in sorted(st.session_state.conversations, key=lambda x: x["created_at"], reverse=True):
        is_active = c["id"] == st.session_state.active_id
        title = c["title"] or "Chat"
        suffix = " ✅" if c.get("closed") else ""
        label = f"{'👉 ' if is_active else ''}{title}{suffix}"
        if st.button(label, key=f"open_{c['id']}", use_container_width=True):
            set_active(c["id"])
            st.rerun()

    st.write("")
    st.markdown("---")
    active = get_active()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Borrar chat", use_container_width=True):
            delete_conversation(active["id"])
            st.rerun()
    with col_b:
        if st.button("Borrar todo", use_container_width=True):
            st.session_state.conversations = [new_conversation(seed_assistant=True)]
            st.session_state.active_id = st.session_state.conversations[0]["id"]
            st.rerun()

active = get_active()

st.title(APP_TITLE)

for msg in active["messages"]:
    role = normalize_role(msg.get("role", "user"))
    avatar = ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR

    with st.chat_message(role, avatar=avatar):
        st.write(msg["content"])
        if msg.get("t"):
            st.caption(msg["t"])

        if role == "assistant" and "debug" in msg:
            with st.expander("ℹ️ Ver detalles de clasificación"):
                render_debug_info(
                    hits=msg["debug"]["hits"],
                    umbral_ambiguedad=msg["debug"]["umbral"],
                )

user_text = st.chat_input("Escribe tu consulta…")

if user_text:
    if active.get("closed"):
        create_and_switch()
        active = get_active()

    if active["title"] == "Nuevo chat":
        active["title"] = make_title_from_text(user_text)

    user_msg = {
        "role": "user",
        "content": user_text,
        "t": now_label(),
    }
    active["messages"].append(user_msg)

    if active.get("pending_question"):
        with st.spinner("Analizando tu respuesta…"):
            salida, _ = call_agente_clarificador(
                pregunta_sistema=active["pending_question"],
                respuesta_usuario=user_text,
                hits=active.get("pending_hits"),
                agents=agents,
            )

            if salida.startswith("ACLARACION:"):
                texto = salida.replace("ACLARACION:", "").strip()
                active["messages"].append(
                    {
                        "role": "assistant",
                        "content": texto,
                        "t": now_label(),
                        "meta": "clarificador",
                    }
                )
                st.rerun()

            if salida.startswith("RESPUESTA_VALIDA:"):
                info = salida.replace("RESPUESTA_VALIDA:", "").strip()
                user_text = f"{active['pending_question']}\n\nRespuesta del ciudadano: {info}"
                active.pop("pending_question", None)
                active.pop("pending_hits", None)

    with st.spinner("Analizando…"):
        try:
            resultado = clasificar_con_desambiguacion(
                message=user_text,
                catalog_df=catalog_df,
                embeddings=embeddings,
                agents=agents,
            )
        except Exception as e:
            resultado = {
                "tipo": "error",
                "mensaje": f"Error del sistema: {e}",
            }

    tipo = resultado.get("tipo")
    print(f"Resultado clasificación: {tipo}")

    if tipo in ("directo", "desambiguado"):
        tramite = resultado["tramite"]
        clasificador_text = f"✅ Identificado: **{tramite}**"

        active["messages"].append(
            {
                "role": "assistant",
                "content": clasificador_text,
                "t": now_label(),
                "debug": {
                    "hits": resultado["hits"],
                    "umbral": UMBRAL_AMBIGUEDAD,
                },
            }
        )

        active["identified_tramite"] = tramite
        active["closed"] = True
        active.pop("pending_question", None)
        active.pop("pending_hits", None)

        try:
            rag_text, _ = call_asistente_tramites(tramite, agents=agents)
        except Exception as e:
            rag_text = f"No se ha podido recuperar la información del trámite. {e}"

        active["messages"].append(
            {
                "role": "assistant",
                "content": rag_text,
                "t": now_label(),
            }
        )

    elif tipo == "necesita_info":
        pregunta = resultado["pregunta"]
        active["pending_question"] = pregunta
        active["pending_hits"] = resultado["hits"]

        active["messages"].append(
            {
                "role": "assistant",
                "content": pregunta,
                "t": now_label(),
                "debug": {
                    "hits": resultado["hits"],
                    "umbral": UMBRAL_AMBIGUEDAD,
                },
            }
        )

    elif tipo == "fuera_de_ambito":
        active["messages"].append(
            {
                "role": "assistant",
                "content": MENSAJE_FUERA_DE_AMBITO,
                "t": now_label(),
            }
        )

    elif tipo in ("no_clasificable", "score_bajo"):
        mensaje = resultado.get("mensaje", MENSAJE_SCORE_BAJO)
        active["messages"].append(
            {
                "role": "assistant",
                "content": mensaje,
                "t": now_label(),
            }
        )

    else:
        mensaje = resultado.get("mensaje", "Ha ocurrido un error inesperado.")
        active["messages"].append(
            {
                "role": "assistant",
                "content": mensaje,
                "t": now_label(),
            }
        )

    st.rerun()
