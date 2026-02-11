import streamlit as st

from agents.client import load_agents
from agents.services import (
    call_agente_clarificador,
    call_agente_desambiguador,
    call_agente_relevancia,
    call_asistente_tramites,
    call_agente_preguntas
)
from config import (
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    ASSISTANT_AVATAR,
    CSS_PATH,
    CLIENT,
    MENSAJE_FUERA_DE_AMBITO,
    MENSAJE_SCORE_BAJO,
    MENSAJE_ERROR_RELEVANCIA,
    UMBRAL_AMBIGUEDAD,
    UMBRAL_SCORE_MINIMO,
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
from core.retrieval import load_catalog_and_embeddings, render_debug_info, retrieve_top_k
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

# ============================================
# Inicialización de estado de desambiguación
# ============================================

if "flow_state" not in active:
    active["flow_state"] = "idle"

if "ambiguous_hits" not in active:
    active["ambiguous_hits"] = None

if "ambig_iter" not in active:
    active["ambig_iter"] = 0

if "ultima_pregunta" not in active:
    active["ultima_pregunta"] = None

if "mensaje_base" not in active:
    active["mensaje_base"] = None

# ==========================================================
# 🔵 BLOQUE A — EMBEDDINGS + SCORE + RELEVANCIA
# ==========================================================

user_text = st.chat_input("Escribe tu consulta…", key="main_chat_input")

if user_text:

    # ==========================================================
    # Render inmediato del mensaje del usuario
    # ==========================================================
    active["messages"].append({
        "role": "user",
        "content": user_text,
        "t": now_label(),
    })

    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(user_text)

    # ==========================================================
    # CASO 1 — Estamos esperando respuesta a una pregunta
    # ==========================================================
    if active["flow_state"] == "esperando_respuesta":

        with st.spinner("Analizando tu respuesta…"):
            tipo_respuesta, _ = call_agente_preguntas(
                pregunta_sistema=active["ultima_pregunta"],
                mensaje_usuario=user_text,
                agents=agents,
            )

        # ------------------------------------------------------
        # Usuario está preguntando (clarificación)
        # ------------------------------------------------------
        if tipo_respuesta == "ES_PREGUNTA":

            with st.spinner("Analizando tu pregunta"):
                salida_clarificador, _ = call_agente_clarificador(
                    pregunta_sistema=active["ultima_pregunta"],
                    respuesta_usuario=user_text,
                    hits=active["ambiguous_hits"],
                    agents=agents
                )

            active["messages"].append({
                "role": "assistant",
                "content": salida_clarificador,
                "t": now_label(),
                "debug": {
                    "hits": active["ambiguous_hits"],
                    "umbral": UMBRAL_AMBIGUEDAD
                }
            })

            st.rerun()

        # ------------------------------------------------------
        # Usuario ha respondido
        # ------------------------------------------------------
        elif tipo_respuesta == "ES_RESPUESTA":

            mensaje_actual = (
                f"{active['mensaje_base']}\n\n"
                f"Respuesta del ciudadano: {user_text}"
            )

            with st.spinner("Identificando posibles trámites…"):
                salida_desambiguador, _ = call_agente_desambiguador(
                    message=mensaje_actual,
                    ambiguous_hits=active["ambiguous_hits"],
                    agents=agents
                )

            # ---- Trámite identificado ----
            if salida_desambiguador.startswith("CLASIFICACION:"):

                tramite = salida_desambiguador.replace("CLASIFICACION:", "").strip()

                active["flow_state"] = "idle"
                active["ambiguous_hits"] = None
                active["ambig_iter"] = 0
                active["ultima_pregunta"] = None
                active["mensaje_base"] = None

                with st.spinner("Generando información del trámite..."):
                    try:
                        rag_text, _ = call_asistente_tramites(tramite, agents=agents)
                    except Exception as e:
                        rag_text = f"No se ha podido recuperar la información del trámite. {e}"

                with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                    st.write(rag_text)

                # Guardar el mensaje final en historial
                active["messages"].append({
                    "role": "assistant",
                    "content": rag_text,
                    "t": now_label(),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD,
                    }
                })

                active["flow_state"] = "idle"
                active["closed"] = True

            # ---- Sigue habiendo ambigüedad ----
            elif salida_desambiguador.startswith("PREGUNTA:"):

                if active["ambig_iter"] >= 2:

                    active["messages"].append({
                        "role": "assistant",
                        "content": "No ha sido posible determinar el trámite con la información proporcionada.",
                        "t": now_label(),
                        "debug": {
                            "hits": active["ambiguous_hits"],
                            "umbral": UMBRAL_AMBIGUEDAD
                        }
                    })

                    active["flow_state"] = "idle"
                    active["ambiguous_hits"] = None
                    active["ambig_iter"] = 0
                    active["ultima_pregunta"] = None
                    active["mensaje_base"] = None

                    st.rerun()

                pregunta = salida_desambiguador.replace("PREGUNTA:", "").strip()

                active["ambig_iter"] += 1
                active["ultima_pregunta"] = pregunta

                active["messages"].append({
                    "role": "assistant",
                    "content": pregunta,
                    "t": now_label(),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD
                    }
                })

                st.rerun()

    # ==========================================================
    # CASO 2 — Flujo normal (estado idle)
    # ==========================================================
    else:

        # ------------------------------------------------------
        # BLOQUE A — Retrieval
        # ------------------------------------------------------
        with st.spinner("Analizando…"):
            hits = retrieve_top_k(
                message=user_text,
                df=catalog_df,
                client=CLIENT,
                embeddings=embeddings,
            )

        top_score = hits["score"].values[0]

        # ------------------------------------------------------
        # BLOQUE B — Umbral mínimo
        # ------------------------------------------------------
        if top_score < UMBRAL_SCORE_MINIMO:

            with st.spinner("Analizando consulta…"):
                relevancia, _ = call_agente_relevancia(
                    message=user_text,
                    agents=agents
                )

            if relevancia == "NO_RELEVANTE":

                active["messages"].append({
                    "role": "assistant",
                    "content": MENSAJE_FUERA_DE_AMBITO,
                    "t": now_label(),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD
                    }
                })

                st.rerun()

            active["messages"].append({
                "role": "assistant",
                "content": MENSAJE_SCORE_BAJO,
                "t": now_label(),
            })

            st.rerun()

        # ------------------------------------------------------
        # BLOQUE C — Ambigüedad
        # ------------------------------------------------------
        scores = hits["score"].values
        ambiguous_mask = (scores[0] - scores) < UMBRAL_AMBIGUEDAD
        n_ambiguous = ambiguous_mask.sum()

        # ---- Sin ambigüedad ----
        if n_ambiguous <= 1:

            tramite = hits.iloc[0]["Trámite"]

            with st.spinner("Generando información del trámite..."):
                try:
                    rag_text, _ = call_asistente_tramites(tramite, agents=agents)
                except Exception as e:
                    rag_text = f"No se ha podido recuperar la información del trámite. {e}"

            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.write(rag_text)

            # Guardar el mensaje final en historial
            active["messages"].append({
                "role": "assistant",
                "content": rag_text,
                "t": now_label(),
                "debug": {
                    "hits": hits,
                    "umbral": UMBRAL_AMBIGUEDAD,
                }
            })

            active["flow_state"] = "idle"
            active["closed"] = True

        # ---- Hay ambigüedad ----
        else:

            ambiguous_hits = hits[ambiguous_mask]

            with st.spinner("Identificando posibles trámites…"):
                salida_desambiguador, _ = call_agente_desambiguador(
                    message=user_text,
                    ambiguous_hits=ambiguous_hits,
                    agents=agents
                )

            if salida_desambiguador.startswith("CLASIFICACION:"):

                tramite = salida_desambiguador.replace("CLASIFICACION:", "").strip()

                with st.spinner("Generando información del trámite..."):
                    try:
                        rag_text, _ = call_asistente_tramites(tramite, agents=agents)
                    except Exception as e:
                        rag_text = f"No se ha podido recuperar la información del trámite. {e}"

                with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                    st.write(rag_text)

                # Guardar el mensaje final en historial
                active["messages"].append({
                    "role": "assistant",
                    "content": rag_text,
                    "t": now_label(),
                    "debug": {
                        "hits": hits,
                        "umbral": UMBRAL_AMBIGUEDAD,
                    }
                })

                active["flow_state"] = "idle"
                active["closed"] = True

            elif salida_desambiguador.startswith("PREGUNTA:"):

                pregunta = salida_desambiguador.replace("PREGUNTA:", "").strip()

                active["flow_state"] = "esperando_respuesta"
                active["ambiguous_hits"] = ambiguous_hits
                active["ambig_iter"] = 1
                active["ultima_pregunta"] = pregunta
                active["mensaje_base"] = user_text

                active["messages"].append({
                    "role": "assistant",
                    "content": pregunta,
                    "t": now_label(),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD
                    }
                })

                st.rerun()