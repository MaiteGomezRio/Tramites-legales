"""
LexAssist — Clasificador de Trámites Judiciales.
Lógica de agentes Foundry + diseño LexAssist.

Ejecutar:
  streamlit run app_redesign.py

IMPORTANTE: la carpeta .streamlit/config.toml debe estar junto a este archivo.
"""

import streamlit as st

from agents.client import load_agents
from agents.services import (
    call_agente_clarificador,
    call_agente_desambiguador,
    call_agente_relevancia,
    call_asistente_tramites,
    call_agente_preguntas,
)
from config import (
    APP_ICON,
    APP_TITLE,
    ASSISTANT_AVATAR,
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
from core.helpers import (
    build_usage_summary,
    make_title_from_text,
    normalize_role,
    now_label,
    render_token_usage_expander,
)
from core.retrieval import load_catalog_and_embeddings, render_debug_info, retrieve_top_k


# ─── Configuración de página ───

st.set_page_config(
    page_title="LexAssist · Clasificador de Trámites",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ─── CSS (diseño LexAssist — NO se carga el CSS original) ───

st.markdown("""
<style>
    /* Ocultar elementos de Streamlit */
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

    /* Sidebar oscuro */
    [data-testid="stSidebar"] { background-color: #1B2A3D !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #C8D6E5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #C8D6E5 !important;
        border-radius: 8px !important;
        font-size: 13.5px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.25) !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cargar recursos ───

catalog_df, embeddings = load_catalog_and_embeddings()
agents = load_agents()
ensure_session_state()


# ─── Helpers de renderizado ───

def render_result_html(label, label_class, tramite=None, message="", hits=None):
    """Tarjeta de resultado con puntuaciones."""
    label_colors = {
        "success": "#2E7D5B",
        "warning": "#B8860B",
        "error": "#C0392B",
        "info": "#2D5F8A",
    }
    border_color = label_colors.get(label_class, "#2D5F8A")

    tramite_html = ""
    if tramite:
        tramite_html = f'<div style="font-size:18px;font-weight:600;color:#1A1A2E;margin:6px 0 10px;">{tramite}</div>'

    scores_html = ""
    if hits is not None:
        try:
            rows_html = ""
            for _, row in hits.head(3).iterrows():
                name = str(row.get("Trámite", ""))
                score = float(row.get("score", 0))
                pct = round(score * 100)
                rows_html += f'''
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12.5px;">
                    <span style="flex:1;color:#5A5A72;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                          title="{name}">{name}</span>
                    <div style="width:100px;height:5px;background:#E2DDD5;border-radius:3px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:#2D5F8A;border-radius:3px;"></div>
                    </div>
                    <span style="width:34px;text-align:right;color:#8E8EA0;font-size:11.5px;font-weight:500;">{pct}%</span>
                </div>'''
            if rows_html:
                scores_html = f'''
                <div style="margin-top:12px;padding-top:12px;border-top:1px solid #E2DDD5;">
                    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:#8E8EA0;margin-bottom:8px;">
                        Puntuaciones de similitud
                    </div>
                    {rows_html}
                </div>'''
        except Exception:
            pass

    return f'''
    <div style="background:white;border-radius:12px;padding:18px 20px;border:1px solid #E2DDD5;
                border-left:3.5px solid {border_color};box-shadow:0 1px 4px rgba(0,0,0,0.04);margin:4px 0;">
        <div style="font-size:10px;font-weight:700;letter-spacing:0.07em;color:{border_color};margin-bottom:4px;">
            {label}
        </div>
        {tramite_html}
        <div style="font-size:13.5px;color:#5A5A72;line-height:1.65;">{message}</div>
        {scores_html}
    </div>'''


# ═══════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚖️ LexAssist")
    st.caption("Clasificador de trámites judiciales")
    st.divider()

    if st.button("➕ Nuevo chat", use_container_width=True):
        create_and_switch()
        st.rerun()

    st.markdown("")

    for c in sorted(
        st.session_state.conversations,
        key=lambda x: x["created_at"],
        reverse=True,
    ):
        is_active = c["id"] == st.session_state.active_id
        title = c["title"] or "Chat"
        suffix = " ✅" if c.get("closed") else ""
        label = f"{'👉 ' if is_active else ''}{title}{suffix}"
        if st.button(label, key=f"open_{c['id']}", use_container_width=True):
            set_active(c["id"])
            st.rerun()

    st.divider()

    active = get_active()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Borrar", use_container_width=True):
            delete_conversation(active["id"])
            st.rerun()
    with col_b:
        if st.button("🗑 Todo", use_container_width=True):
            st.session_state.conversations = [new_conversation(seed_assistant=True)]
            st.session_state.active_id = st.session_state.conversations[0]["id"]
            st.rerun()

    st.divider()
    st.success("Sistema activo", icon="✅")
    st.divider()
    st.caption("Prototipo de demostración · v1.0")


# ═══════════════════════════════════════
#  ÁREA PRINCIPAL
# ═══════════════════════════════════════

active = get_active()

# ─── Pantalla de bienvenida (si no hay mensajes del usuario) ───

has_user_messages = any(
    normalize_role(msg.get("role", "user")) == "user"
    for msg in active["messages"]
)

if not has_user_messages:
    st.markdown("")
    st.markdown(
        '<div style="text-align:center;padding:20px 0 8px;">'
        '<div style="width:56px;height:56px;background:#2D5F8A;border-radius:14px;'
        'display:inline-flex;align-items:center;justify-content:center;'
        'box-shadow:0 4px 16px rgba(45,95,138,0.25);font-size:26px;">⚖️</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2 style="text-align:center;font-weight:600;margin-bottom:4px;">Asistente de Trámites Judiciales</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:#5A5A72;font-size:15px;max-width:500px;margin:0 auto 28px;line-height:1.6;">'
        'Describa su situación legal y nuestro sistema identificará automáticamente '
        'el trámite judicial correspondiente. Puede escribir con sus propias palabras.'
        '</p>',
        unsafe_allow_html=True,
    )

    examples = [
        "Quiero reclamar una deuda que no me pagan",
        "Me han despedido y quiero impugnarlo",
        "Necesito tramitar una herencia",
        "Quiero divorciarme de mutuo acuerdo",
    ]

    col1, col2 = st.columns(2)
    for i, ex in enumerate(examples):
        with (col1 if i % 2 == 0 else col2):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state._pending_example = ex
                st.rerun()

    st.markdown("")

else:
    # ─── Historial de mensajes ───
    for msg in active["messages"]:
        role = normalize_role(msg.get("role", "user"))
        avatar = "⚖️" if role == "assistant" else "👤"

        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])

            if msg.get("t"):
                st.caption(msg["t"])

            if role == "assistant" and msg.get("usage"):
                render_token_usage_expander(msg.get("usage"))

            if role == "assistant" and "debug" in msg:
                with st.expander("ℹ️ Ver detalles de clasificación"):
                    render_debug_info(
                        hits=msg["debug"]["hits"],
                        umbral_ambiguedad=msg["debug"]["umbral"],
                    )


# ═══════════════════════════════════════
#  ESTADO DE DESAMBIGUACIÓN
# ═══════════════════════════════════════

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


# ═══════════════════════════════════════
#  PROCESAR EJEMPLO PENDIENTE
# ═══════════════════════════════════════

if hasattr(st.session_state, "_pending_example") and st.session_state._pending_example:
    _ex = st.session_state._pending_example
    st.session_state._pending_example = None

    active["messages"].append({
        "role": "user",
        "content": _ex,
        "t": now_label(),
    })
    st.rerun()


# ═══════════════════════════════════════
#  PROCESAMIENTO DE INPUT
# ═══════════════════════════════════════

user_text = st.chat_input("Escriba su consulta legal…", key="main_chat_input")

if user_text:

    # ── Guardar mensaje del usuario ──
    active["messages"].append({
        "role": "user",
        "content": user_text,
        "t": now_label(),
    })
    with st.chat_message("user", avatar="👤"):
        st.write(user_text)

    # ══════════════════════════════════════
    #  CASO 1 — Esperando respuesta a pregunta de desambiguación
    # ══════════════════════════════════════

    if active["flow_state"] == "esperando_respuesta":

        with st.spinner("Analizando su respuesta…"):
            tipo_respuesta, _, preguntas_usage = call_agente_preguntas(
                pregunta_sistema=active["ultima_pregunta"],
                mensaje_usuario=user_text,
                agents=agents,
            )
        usage_steps = [("preguntas", preguntas_usage)]

        # ── Usuario está preguntando (clarificación) ──
        if tipo_respuesta == "ES_PREGUNTA":

            with st.spinner("Preparando aclaración…"):
                salida_clarificador, _, clarificador_usage = call_agente_clarificador(
                    pregunta_sistema=active["ultima_pregunta"],
                    respuesta_usuario=user_text,
                    hits=active["ambiguous_hits"],
                    agents=agents,
                )
            usage_steps.append(("clarificador", clarificador_usage))

            active["messages"].append({
                "role": "assistant",
                "content": salida_clarificador,
                "t": now_label(),
                "usage": build_usage_summary(usage_steps),
                "debug": {
                    "hits": active["ambiguous_hits"],
                    "umbral": UMBRAL_AMBIGUEDAD,
                },
            })
            st.rerun()

        # ── Usuario ha respondido ──
        elif tipo_respuesta == "ES_RESPUESTA":

            mensaje_actual = (
                f"{active['mensaje_base']}\n\n"
                f"Respuesta del ciudadano: {user_text}"
            )

            with st.spinner("Identificando posibles trámites…"):
                salida_desambiguador, _, desambiguador_usage = call_agente_desambiguador(
                    message=mensaje_actual,
                    ambiguous_hits=active["ambiguous_hits"],
                    agents=agents,
                )
            usage_steps.append(("desambiguador", desambiguador_usage))

            # ── Trámite identificado ──
            if salida_desambiguador.startswith("CLASIFICACION:"):

                tramite = salida_desambiguador.replace("CLASIFICACION:", "").strip()

                active["flow_state"] = "idle"
                active["ambiguous_hits"] = None
                active["ambig_iter"] = 0
                active["ultima_pregunta"] = None
                active["mensaje_base"] = None

                with st.spinner("Generando información del trámite…"):
                    try:
                        rag_text, _, asistente_usage = call_asistente_tramites(
                            tramite, agents=agents
                        )
                    except Exception as e:
                        rag_text = f"No se ha podido recuperar la información del trámite. {e}"
                        asistente_usage = None

                usage_steps.append(("asistente", asistente_usage))
                usage_summary = build_usage_summary(usage_steps)

                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(
                        render_result_html(
                            label="✓ TRÁMITE IDENTIFICADO",
                            label_class="success",
                            tramite=tramite,
                            message=rag_text,
                        ),
                        unsafe_allow_html=True,
                    )
                    render_token_usage_expander(usage_summary)

                active["messages"].append({
                    "role": "assistant",
                    "content": rag_text,
                    "t": now_label(),
                    "usage": usage_summary,
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD,
                    },
                })

                active["flow_state"] = "idle"
                active["closed"] = True

            # ── Sigue habiendo ambigüedad ──
            elif salida_desambiguador.startswith("PREGUNTA:"):

                if active["ambig_iter"] >= 2:
                    active["messages"].append({
                        "role": "assistant",
                        "content": "No ha sido posible determinar el trámite con la información proporcionada.",
                        "t": now_label(),
                        "usage": build_usage_summary(usage_steps),
                        "debug": {
                            "hits": active["ambiguous_hits"],
                            "umbral": UMBRAL_AMBIGUEDAD,
                        },
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
                    "usage": build_usage_summary(usage_steps),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD,
                    },
                })
                st.rerun()

    # ══════════════════════════════════════
    #  CASO 2 — Flujo normal (idle)
    # ══════════════════════════════════════

    else:

        # ── Retrieval ──
        with st.spinner("Analizando consulta…"):
            hits = retrieve_top_k(message=user_text)

        top_score = hits["score"].values[0]

        # ── Umbral mínimo ──
        if top_score < UMBRAL_SCORE_MINIMO:

            with st.spinner("Verificando relevancia…"):
                relevancia, _, relevancia_usage = call_agente_relevancia(
                    message=user_text, agents=agents
                )
            usage_summary = build_usage_summary([("relevancia", relevancia_usage)])

            if relevancia == "NO_RELEVANTE":
                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(
                        render_result_html(
                            label="✗ FUERA DE ÁMBITO",
                            label_class="error",
                            message=MENSAJE_FUERA_DE_AMBITO,
                        ),
                        unsafe_allow_html=True,
                    )

                active["messages"].append({
                    "role": "assistant",
                    "content": MENSAJE_FUERA_DE_AMBITO,
                    "t": now_label(),
                    "usage": usage_summary,
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD,
                    },
                })
                st.rerun()

            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(
                    render_result_html(
                        label="⚠ NO CLASIFICABLE",
                        label_class="warning",
                        message=MENSAJE_SCORE_BAJO,
                    ),
                    unsafe_allow_html=True,
                )

            active["messages"].append({
                "role": "assistant",
                "content": MENSAJE_SCORE_BAJO,
                "t": now_label(),
                "usage": usage_summary,
            })
            st.rerun()

        # ── Ambigüedad ──
        scores = hits["score"].values
        ambiguous_mask = (scores[0] - scores) < UMBRAL_AMBIGUEDAD
        n_ambiguous = ambiguous_mask.sum()

        # ── Sin ambigüedad ──
        if n_ambiguous <= 1:

            tramite = hits.iloc[0]["Trámite"]

            with st.spinner("Generando información del trámite…"):
                try:
                    rag_text, _, asistente_usage = call_asistente_tramites(
                        tramite, agents=agents
                    )
                except Exception as e:
                    rag_text = f"No se ha podido recuperar la información del trámite. {e}"
                    asistente_usage = None

            usage_summary = build_usage_summary([("asistente", asistente_usage)])

            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(
                    render_result_html(
                        label="✓ TRÁMITE IDENTIFICADO",
                        label_class="success",
                        tramite=tramite,
                        message=rag_text,
                        hits=hits,
                    ),
                    unsafe_allow_html=True,
                )
                render_token_usage_expander(usage_summary)

            active["messages"].append({
                "role": "assistant",
                "content": rag_text,
                "t": now_label(),
                "usage": usage_summary,
                "debug": {
                    "hits": hits,
                    "umbral": UMBRAL_AMBIGUEDAD,
                },
            })

            active["flow_state"] = "idle"
            active["closed"] = True

        # ── Hay ambigüedad ──
        else:

            ambiguous_hits = hits[ambiguous_mask]

            with st.spinner("Identificando posibles trámites…"):
                salida_desambiguador, _, desambiguador_usage = call_agente_desambiguador(
                    message=user_text,
                    ambiguous_hits=ambiguous_hits,
                    agents=agents,
                )
            usage_steps = [("desambiguador", desambiguador_usage)]

            if salida_desambiguador.startswith("CLASIFICACION:"):

                tramite = salida_desambiguador.replace("CLASIFICACION:", "").strip()

                with st.spinner("Generando información del trámite…"):
                    try:
                        rag_text, _, asistente_usage = call_asistente_tramites(
                            tramite, agents=agents
                        )
                    except Exception as e:
                        rag_text = f"No se ha podido recuperar la información del trámite. {e}"
                        asistente_usage = None

                usage_steps.append(("asistente", asistente_usage))
                usage_summary = build_usage_summary(usage_steps)

                with st.chat_message("assistant", avatar="⚖️"):
                    st.markdown(
                        render_result_html(
                            label="✓ TRÁMITE IDENTIFICADO",
                            label_class="success",
                            tramite=tramite,
                            message=rag_text,
                            hits=hits,
                        ),
                        unsafe_allow_html=True,
                    )
                    render_token_usage_expander(usage_summary)

                active["messages"].append({
                    "role": "assistant",
                    "content": rag_text,
                    "t": now_label(),
                    "usage": usage_summary,
                    "debug": {
                        "hits": hits,
                        "umbral": UMBRAL_AMBIGUEDAD,
                    },
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
                    "usage": build_usage_summary(usage_steps),
                    "debug": {
                        "hits": active["ambiguous_hits"],
                        "umbral": UMBRAL_AMBIGUEDAD,
                    },
                })
                st.rerun()
