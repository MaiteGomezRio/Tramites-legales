import pandas as pd
import streamlit as st

from agents import (
    call_agente_clarificador,
    call_asistente_tramites,
    clasificar_con_desambiguacion,
    detectar_tipo_mensaje_pendiente,
    load_agents,
    safe_agente_relevancia,
)
from config import (
    ASSISTANT_AVATAR,
    CSS_PATH,
    MAX_INTENTOS_DESAMBIGUACION,
    MAX_REINTENTOS_RELEVANCIA,
    MENSAJE_FUERA_DE_AMBITO,
    MENSAJE_REFORMULAR,
    MENSAJE_SCORE_BAJO,
    PAGE_ICON,
    PAGE_LAYOUT,
    PAGE_TITLE,
    UMBRAL_AMBIGUEDAD,
    USER_AVATAR,
)
from conversations import create_and_switch, get_active, init_conversations
from core import (
    attach_token_trace_to_last_assistant_message,
    extract_prefixed_value,
    format_no_clasificable_msg,
    fuera_de_ambito_rapido,
    is_no_relevante,
    is_prompt_leak_attempt,
    looks_like_injection,
    make_title_from_text,
    new_query_token_trace,
    now_label,
)
from core.retrieval import load_catalog
from ui import load_css, render_chat_messages, render_sidebar


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
load_css(CSS_PATH)

# Calentamos cache del catálogo (útil para diagnóstico y validación de dataset)
load_catalog()

init_conversations()
render_sidebar()

AGENTS = load_agents()

active = get_active()
st.title(PAGE_TITLE)
render_chat_messages(active, assistant_avatar=ASSISTANT_AVATAR, user_avatar=USER_AVATAR)

user_text = st.chat_input("Escribe tu consulta…")

if user_text:
    query_token_trace = new_query_token_trace()

    if active.get("closed"):
        create_and_switch()
        active = get_active()

    if active["title"] == "Nuevo chat":
        active["title"] = make_title_from_text(user_text)

    active["messages"].append(
        {
            "role": "user",
            "content": user_text,
            "t": now_label(),
        }
    )

    # =========================
    # PREGUNTAS + CLARIFICADOR
    # =========================
    if active.get("pending_question"):
        pregunta_sistema = active.get("pending_question") or ""
        pending_hits = active.get("pending_hits")

        with st.spinner("Analizando tu respuesta…"):
            tipo_msg, contenido = detectar_tipo_mensaje_pendiente(
                agents=AGENTS,
                pregunta_sistema=pregunta_sistema,
                mensaje_usuario=user_text,
                token_trace=query_token_trace,
            )

        if tipo_msg == "pregunta":
            try:
                salida, _ = call_agente_clarificador(
                    agents=AGENTS,
                    pregunta_sistema=pregunta_sistema,
                    respuesta_usuario=contenido,
                    hits=pending_hits if pending_hits is not None else pd.DataFrame(),
                    token_trace=query_token_trace,
                )
                texto = extract_prefixed_value(salida, "ACLARACION:") or salida.strip()
            except Exception:
                texto = (
                    "Claro. Necesito una respuesta concreta para poder clasificar tu trámite.\n\n"
                    f"{pregunta_sistema}"
                )

            active["messages"].append(
                {
                    "role": "assistant",
                    "content": texto,
                    "t": now_label(),
                    "meta": "clarificador",
                }
            )
            attach_token_trace_to_last_assistant_message(active, query_token_trace)
            st.rerun()

        info = (contenido or "").strip() or user_text
        user_text = f"{pregunta_sistema}\n\nRespuesta del ciudadano: {info}"
        active["pending_question"] = None
        active["pending_hits"] = None

    # =========================
    # SEGURIDAD + FUERA DE ÁMBITO (pre-clasificación)
    # =========================
    if looks_like_injection(user_text) or is_prompt_leak_attempt(user_text):
        active["messages"].append(
            {
                "role": "assistant",
                "content": MENSAJE_FUERA_DE_AMBITO,
                "t": now_label(),
            }
        )
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    if fuera_de_ambito_rapido(user_text):
        active["messages"].append(
            {
                "role": "assistant",
                "content": MENSAJE_FUERA_DE_AMBITO,
                "t": now_label(),
            }
        )
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    relevancia_precheck = safe_agente_relevancia(AGENTS, user_text, token_trace=query_token_trace)
    if is_no_relevante(relevancia_precheck):
        active["messages"].append(
            {
                "role": "assistant",
                "content": MENSAJE_FUERA_DE_AMBITO,
                "t": now_label(),
            }
        )
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    # =========================
    # CLASIFICACIÓN
    # =========================
    with st.spinner("Clasificando consulta…"):
        try:
            resultado = clasificar_con_desambiguacion(
                message=user_text,
                agents=AGENTS,
                token_trace=query_token_trace,
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

        active["messages"].append(
            {
                "role": "assistant",
                "content": f"✅ Identificado: **{tramite}**",
                "t": now_label(),
                "debug": {
                    "hits": resultado["hits"],
                    "umbral": UMBRAL_AMBIGUEDAD,
                },
            }
        )

        active["identified_tramite"] = tramite
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        active["desambiguacion_intentos"] = 0
        active["relevancia_reintentos"] = 0

        try:
            rag_text, _ = call_asistente_tramites(
                agents=AGENTS,
                tramite_nombre=tramite,
                token_trace=query_token_trace,
            )
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
        active["relevancia_reintentos"] = 0
        intentos = int(active.get("desambiguacion_intentos", 0)) + 1
        active["desambiguacion_intentos"] = intentos

        if intentos > MAX_INTENTOS_DESAMBIGUACION:
            mensaje = format_no_clasificable_msg(resultado.get("ambiguous_hits", resultado.get("hits")))
            active["pending_question"] = None
            active["pending_hits"] = None
            active["desambiguacion_intentos"] = 0
            active["messages"].append(
                {
                    "role": "assistant",
                    "content": mensaje,
                    "t": now_label(),
                }
            )
        else:
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
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None

    elif tipo == "score_bajo":
        active["desambiguacion_intentos"] = 0
        active["pending_question"] = None
        active["pending_hits"] = None

        if is_no_relevante(relevancia_precheck):
            mensaje = MENSAJE_FUERA_DE_AMBITO
            active["closed"] = True
        else:
            reintentos = int(active.get("relevancia_reintentos", 0)) + 1
            active["relevancia_reintentos"] = reintentos
            if reintentos < MAX_REINTENTOS_RELEVANCIA:
                mensaje = MENSAJE_REFORMULAR
            else:
                mensaje = MENSAJE_SCORE_BAJO
                active["relevancia_reintentos"] = 0

        active["messages"].append(
            {
                "role": "assistant",
                "content": mensaje,
                "t": now_label(),
            }
        )

    elif tipo == "no_clasificable":
        active["messages"].append(
            {
                "role": "assistant",
                "content": resultado.get("mensaje", MENSAJE_SCORE_BAJO),
                "t": now_label(),
            }
        )

    else:
        active["messages"].append(
            {
                "role": "assistant",
                "content": resultado.get("mensaje", "Ha ocurrido un error inesperado."),
                "t": now_label(),
            }
        )

    attach_token_trace_to_last_assistant_message(active, query_token_trace)
    st.rerun()
