import os
import time
import json
import uuid
import datetime as dt
import streamlit as st
import pandas as pd
from openai import OpenAI
 
from dotenv import load_dotenv
load_dotenv()
 
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
 
# =========================
# PAGE
# =========================
st.set_page_config(page_title="Asistente de Trámites", page_icon="🧾", layout="wide")

# =========================
# CONFIG PRODUCCIÓN (AGENTES)
# =========================
AIPROJECT_ENDPOINT = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT", "")

CLASIFICADOR_AGENT_NAME = os.getenv(
    "AZURE_CLASIFICADOR_AGENT_NAME",
    "ClasificadorTramites"
)

ASISTENTE_AGENT_NAME = os.getenv(
    "AZURE_ASISTENTE_AGENT_NAME",
    "AsistenteTramites"
)

DESAMBIGUADOR_AGENT_NAME = os.getenv(
    "AZURE_DESAMBIGUADOR_AGENT_NAME",
    "DesambiguadorTramites"
)

RELEVANCIA_AGENT_NAME = os.getenv(
    "AZURE_RELEVANCIA_AGENT_NAME",
    "RelevanciaRespuestaTramites"
)

CLARIFICADOR_AGENT_NAME = os.getenv(
    "AZURE_CLARIFICADOR_AGENT_NAME",
    "ClarificadorTramites"
)

PREGUNTAS_AGENT_NAME = os.getenv(
    "AZURE_PREGUNTAS_AGENT_NAME",
    "PreguntasTramites"
)

ASSISTANT_AVATAR = "assets/logo_junta_avatar.png"
USER_AVATAR = "assets/user_avatar.png"

UMBRAL_RETRIEVAL = 0.14  # gap máximo respecto al mejor score para incluir un candidato
UMBRAL_AMBIGUEDAD = 0.05
UMBRAL_SCORE_MINIMO = 0.35

MENSAJE_NO_CLASIFICABLE = (
    "Lamentamos no haber podido clasificar su consulta de forma automática. "
    "No ha sido posible diferenciar con una precisión razonable entre las categorías "
    "{categorias}. "
    "Le facilitamos el siguiente correo electrónico para que pueda derivar su consulta "
    "a un profesional especializado: consulta.tramites@ejemplo.com"
)

MENSAJE_SCORE_BAJO = (
    "No hemos encontrado ninguna categoría que se ajuste a su consulta. "
    "Le facilitamos el siguiente correo electrónico para que pueda derivar su consulta "
    "a un profesional especializado: consulta.tramites@ejemplo.com"
)

MENSAJE_FUERA_DE_AMBITO = (
    "Este es un asistente de clasificación de trámites judiciales. "
    "Solo podemos ayudarle con consultas relacionadas con procedimientos legales y judiciales. "
    "Si tiene alguna consulta de este tipo, por favor reformule su mensaje."
)

MENSAJE_REFORMULAR = (
    "No he podido identificar el trámite con suficiente precisión. "
    "Por favor, describe tu caso con más detalle: qué pasó, contra quién, "
    "y qué resultado necesitas."
)

MAX_INTENTOS_DESAMBIGUACION = 2
MAX_REINTENTOS_RELEVANCIA = 2

# Frases típicas de inyección / exfiltración
SUSPECT_PATTERNS = [
    "ignore", "ignora", "olvida", "disregard", "system prompt", "prompt del sistema", "developer message",
    "revela", "reveal", "muestra tu prompt", "show your prompt", "dame el prompt", "prompt interno",
    "api key", "clave api", "key1", "key2", "subscription key", "token", "secreto", "secret",
    "haz esto primero", "actúa como", "actua como", "jailbreak", "bypass", "override",
    "ejecuta", "run", "shell", "terminal", "powershell", "cmd.exe", "curl", "wget",
    "sql injection", "inyección sql", "command injection", "inyeccion de comandos",
    "roba", "robame", "extrae", "exfiltra", "filtra", "dump", "leak",
]

PROMPT_LEAK_PATTERNS = [
    "prompt", "system prompt", "developer message", "dime el prompt", "muéstrame el prompt",
    "reveal", "revela", "mostrar instrucciones", "instrucciones internas",
    "api key", "clave", "token", "secreto", "secret"
]
 
# =========================
# TODO: DEBUG REMOVE
# =========================

CLIENT = OpenAI(
    api_key = "6TZDD5vTve7kB5DVixnKnoGUtYUahlVxuEzuuU3KWXu9lTagcLWYJQQJ99CBACHYHv6XJ3w3AAABACOGj8AC",
    base_url = "https://azure-openai-tramites-legales.openai.azure.com/openai/v1/" 
)

EMB_MODEL = "text-embedding-3-small"
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY", "")
INDEX_NAME = os.getenv("AZURE_INDEX_NAME", "tramites-index")
SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "embedding")
SEARCH_TRAMITE_FIELD = os.getenv("AZURE_SEARCH_TRAMITE_FIELD", "tramite")
SEARCH_DESCRIPCION_FIELD = os.getenv("AZURE_SEARCH_DESCRIPCION_FIELD", "descripcion")

# =========================
# STYLES
# =========================
 
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Asistente de Trámites",
    page_icon="🧾",
    layout="wide"
)

load_css("styles/theme_accenture.css")

# =========================
# HELPERS
# =========================
def _try_parse_json(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(text[a:b+1])
        except Exception:
            return None
    return None

def _extract_prefixed_value(text: str, prefix: str):
    raw = (text or "").strip()
    if raw.upper().startswith(prefix.upper()):
        return raw[len(prefix):].strip()
    return None

def _is_no_relevante(raw_text: str):
    t = (raw_text or "").strip().upper()
    return "NO_RELEVANTE" in t

def looks_like_injection(text: str):
    t = (text or "").lower()
    return any(p in t for p in SUSPECT_PATTERNS)

def is_prompt_leak_attempt(text: str):
    t = (text or "").lower()
    return any(p in t for p in PROMPT_LEAK_PATTERNS)

def fuera_de_ambito_rapido(text: str):
    t = (text or "").lower()
    legal_hints = [
        "juicio", "juzgado", "demanda", "abogado", "herencia", "deuda", "contrato",
        "procedimiento", "tramite", "trámite", "notario", "ejecución", "monitorio",
        "conciliación", "perito", "embargo", "reclamar", "reclamación", "desahucio",
        "divorcio", "custodia", "incapacidad", "curatela", "tutela", "defensor judicial",
    ]
    nonlegal_hints = [
        "receta", "tortilla", "cocinar", "bacon", "queso", "comida", "gym", "entrenar",
        "película", "serie", "música", "canción", "viaje", "hotel", "restaurante"
    ]
    has_legal = any(w in t for w in legal_hints)
    has_nonlegal = any(w in t for w in nonlegal_hints)
    return has_nonlegal and not has_legal

def _format_no_clasificable_msg(hits: pd.DataFrame):
    if hits is None or hits.empty:
        return MENSAJE_SCORE_BAJO

    nombres = [
        str(row["Trámite"])
        for _, row in hits.head(3).iterrows()
        if pd.notna(row.get("Trámite"))
    ]
    if not nombres:
        return MENSAJE_SCORE_BAJO

    if len(nombres) == 1:
        lista = nombres[0]
    elif len(nombres) == 2:
        lista = f"{nombres[0]} y {nombres[1]}"
    else:
        lista = f"{', '.join(nombres[:-1])} y {nombres[-1]}"
    return MENSAJE_NO_CLASIFICABLE.format(categorias=lista)

def _usage_get(usage, key: str):
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)

def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0

def _extract_usage_totals(usage):
    prompt_tokens = _safe_int(
        _usage_get(usage, "prompt_tokens")
        or _usage_get(usage, "input_tokens")
    )
    completion_tokens = _safe_int(
        _usage_get(usage, "completion_tokens")
        or _usage_get(usage, "output_tokens")
    )
    total_tokens = _safe_int(_usage_get(usage, "total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def _new_query_token_trace():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": [],
    }

def _track_usage(trace, label: str, usage):
    if trace is None or usage is None:
        return
    totals = _extract_usage_totals(usage)
    if totals["prompt_tokens"] == 0 and totals["completion_tokens"] == 0 and totals["total_tokens"] == 0:
        return
    trace["prompt_tokens"] += totals["prompt_tokens"]
    trace["completion_tokens"] += totals["completion_tokens"]
    trace["total_tokens"] += totals["total_tokens"]
    trace["calls"].append({
        "label": label,
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["total_tokens"],
    })

def _attach_token_trace_to_last_assistant_message(conversation, trace):
    if not trace:
        return
    if trace.get("total_tokens", 0) == 0 and not trace.get("calls"):
        return
    for msg in reversed(conversation.get("messages", [])):
        if msg.get("role") == "assistant":
            msg["token_usage"] = {
                "prompt_tokens": trace.get("prompt_tokens", 0),
                "completion_tokens": trace.get("completion_tokens", 0),
                "total_tokens": trace.get("total_tokens", 0),
                "calls": list(trace.get("calls", [])),
            }
            return
 
def badge_for_conf(conf):
    if conf is None:
        return '<span class="badge badge-no">sin confianza</span>'
    if conf >= 0.75:
        return f'<span class="badge badge-ok">confianza alta · {conf:.2f}</span>'
    if conf >= 0.55:
        return f'<span class="badge badge-warn">confianza media · {conf:.2f}</span>'
    return f'<span class="badge badge-no">confianza baja · {conf:.2f}</span>'
 
def now_label():
    return dt.datetime.now().strftime("%H:%M")
 
def make_title_from_text(text: str, max_len=34):
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[:max_len-1] + "…"

@st.cache_resource(show_spinner=False)
def get_search_client():
    if not SEARCH_ENDPOINT or not SEARCH_QUERY_KEY or not INDEX_NAME:
        raise RuntimeError(
            "Faltan variables de entorno para Azure AI Search: "
            "AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_QUERY_KEY o AZURE_INDEX_NAME."
        )
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_QUERY_KEY),
    )

def retrieve_top_k(message, k=8, gap_max=UMBRAL_RETRIEVAL, token_trace=None):
    q_resp = CLIENT.embeddings.create(model=EMB_MODEL, input=message)
    _track_usage(token_trace, "retrieval_query_embedding", getattr(q_resp, "usage", None))
    q_vec = q_resp.data[0].embedding

    vq = VectorizedQuery(
        vector=q_vec,
        fields=SEARCH_VECTOR_FIELD,
    )

    results = get_search_client().search(
        search_text="",
        vector_queries=[vq],
        select=[SEARCH_TRAMITE_FIELD, SEARCH_DESCRIPCION_FIELD],
        top=k,
    )

    rows = []
    for r in results:
        try:
            score = float(r.get("@search.score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        rows.append({
            "Trámite": r.get(SEARCH_TRAMITE_FIELD),
            "Descripción": r.get(SEARCH_DESCRIPCION_FIELD),
            "score": score,
        })

    hits = pd.DataFrame(rows, columns=["Trámite", "Descripción", "score"])
    if hits.empty:
        return hits

    hits = hits.sort_values("score", ascending=False).reset_index(drop=True)
    hits = (
        hits.dropna(subset=["Trámite"])
        .drop_duplicates(subset=["Trámite"], keep="first")
        .reset_index(drop=True)
    )
    if hits.empty:
        return pd.DataFrame(columns=["Trámite", "Descripción", "score"])

    top_score = float(hits.loc[0, "score"])
    filtered = hits[(top_score - hits["score"].astype(float)) < gap_max].copy()

    if len(filtered) < 3:
        filtered = hits.head(min(3, len(hits))).copy()
    elif len(filtered) > k:
        filtered = filtered.head(k).copy()

    return filtered

@st.cache_resource(show_spinner="Cargando catálogo de trámites…")
def load_catalog():
    catalog_df = pd.read_excel("descripcion-tramites.xlsx")
    catalog_df = catalog_df.dropna(subset=["Trámite"]).reset_index(drop=True)
    return catalog_df

def render_debug_info(hits, umbral_ambiguedad):
    top_score = hits.iloc[0]["score"]

    st.markdown("**Candidatos evaluados:**")
    for _, row in hits.iterrows():
        gap = top_score - row["score"]
        marker = " ⚠️ ambiguo" if gap < umbral_ambiguedad else ""
        st.markdown(
            f"- `{row['score']:.4f}` · **{row['Trámite']}**{marker}"
        )

# Load catalog at startup (cached)
catalog_df = load_catalog()

# =========================
# AGENT CLIENT
# =========================
@st.cache_resource(show_spinner=False)
def get_project_client():
    if not AIPROJECT_ENDPOINT:
        raise RuntimeError("Falta AZURE_EXISTING_AIPROJECT_ENDPOINT.")
    return AIProjectClient(
        endpoint=AIPROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

@st.cache_resource(show_spinner=False)
def get_agent(agent_name: str):
    project_client = get_project_client()
    
    try:
        print(f"Obteniendo agente '{agent_name}' de Foundry...")
        agent = project_client.agents.get(agent_name=agent_name)
    except Exception as e:
        print(f" Error al obtener agente '{agent_name}': {e}")

    openai_client = project_client.get_openai_client()

    return openai_client

@st.cache_resource(show_spinner="Cargando agente…")
def load_agents():
    return {
        "clasificador": get_agent(CLASIFICADOR_AGENT_NAME),
        "asistente": get_agent(ASISTENTE_AGENT_NAME),
        "desambiguador": get_agent(DESAMBIGUADOR_AGENT_NAME),
        "clarificador": get_agent(CLARIFICADOR_AGENT_NAME),
        "relevancia": get_agent(RELEVANCIA_AGENT_NAME),
        "preguntas": get_agent(PREGUNTAS_AGENT_NAME),
    }

def call_clasificador(messages):
    """
    messages: historial del chat
    Devuelve:
      - ("identified", payload_dict, ms)  → JSON válido
      - ("text", texto_plano, ms)         → aclaración / no_match / out of scope
    """
    start = time.time()

    response = AGENTS["clasificador"].responses.create(
        input=messages,
        extra_body={"agent": {"name": CLASIFICADOR_AGENT_NAME, "type": "agent_reference"}},
    )

    raw_text = getattr(response, "output_text", "") or ""
    parsed = _try_parse_json(raw_text)

    ms = int((time.time() - start) * 1000)

    # JSON SOLO cuando está identificado
    if isinstance(parsed, dict) and parsed.get("status") == "identified":
        return "identified", parsed, ms

    # En cualquier otro caso → texto plano
    return "text", raw_text.strip(), ms

def call_asistente_tramites(tramite_nombre: str, token_trace=None):
    start = time.time()

    prompt = (
        f"El trámite identificado es: {tramite_nombre}.\n"
        "Proporciona información clara y orientativa para un ciudadano."
    )

    response = AGENTS["asistente"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={"agent": {"name": ASISTENTE_AGENT_NAME, "type": "agent_reference"}},
    )
    _track_usage(token_trace, "chat_asistente_tramites", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms

def call_agente_desambiguador(message: str, ambiguous_hits: pd.DataFrame, token_trace=None):
    start = time.time()

    # Construir el input que se le pasa al agente
    candidates_detail = "\n\n".join(
        f"- {row['Trámite']} (score: {row['score']:.4f})\n"
        f"  Descripción: {row['Descripción']}"
        for _, row in ambiguous_hits.iterrows()
    )

    prompt = (
        "MENSAJE DEL CIUDADANO:\n"
        f"\"\"\"{message}\"\"\"\n\n"
        "CATEGORÍAS AMBIGUAS:\n"
        f"{candidates_detail}"
    )

    # Llamada al agente
    response = AGENTS["desambiguador"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": DESAMBIGUADOR_AGENT_NAME,
                "type": "agent_reference"
            }
        },
    )
    _track_usage(token_trace, "chat_desambiguador", getattr(response, "usage", None))

    # Extraer salida
    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms

def clasificar_con_desambiguacion(
    message,
    k=8,
    gap_max=UMBRAL_RETRIEVAL,
    umbral=UMBRAL_AMBIGUEDAD,
    umbral_minimo=UMBRAL_SCORE_MINIMO,
    token_trace=None,
):
    # 1️⃣ Retrieval semántico
    hits = retrieve_top_k(
        message=message,
        k=k,
        gap_max=gap_max,
        token_trace=token_trace,
    )
    scores = hits["score"].values

    if len(scores) == 0:
        return {
            "tramite": None,
            "tipo": "score_bajo",
            "hits": hits
        }

    # 2️⃣ Score mínimo
    if scores[0] < umbral_minimo:
        return {
            "tramite": None,
            "tipo": "score_bajo",
            "hits": hits
        }

    # 3️⃣ Detección de ambigüedad
    ambiguous_mask = (scores[0] - scores) < umbral
    n_ambiguous = ambiguous_mask.sum()

    if n_ambiguous > 1:
        ambiguous_hits = hits[ambiguous_mask]

        print(f"  Ambigüedad detectada: {n_ambiguous} candidatos con diferencia < {umbral}")
        for _, row in ambiguous_hits.iterrows():
            print(f"   {row['score']:.4f} | {row['Trámite']}")

        # 4️⃣ Llamada al agente de desambiguación en Foundry
        resultado, ms = call_agente_desambiguador(
            message=message,
            ambiguous_hits=ambiguous_hits,
            token_trace=token_trace,
        )

        print(f"  Resultado del agente de desambiguación (en {ms} ms): {resultado}")

        # 5️⃣ Interpretar respuesta del agente
        if resultado.startswith("CLASIFICACION:"):
            tramite = resultado.replace("CLASIFICACION:", "").strip()
            return {
                "tramite": tramite,
                "tipo": "desambiguado",
                "hits": hits
            }

        elif resultado.startswith("PREGUNTA:"):
            pregunta = resultado.replace("PREGUNTA:", "").strip()
            return {
                "tramite": None,
                "pregunta": pregunta,
                "tipo": "necesita_info",
                "hits": hits,
                "ambiguous_hits": ambiguous_hits
            }

        # Fallback defensivo (no debería ocurrir)
        return {
            "tramite": None,
            "tipo": "error_desambiguador",
            "raw_output": resultado,
            "hits": hits
        }

    # 6️⃣ Clasificación directa (sin ambigüedad)
    tramite = hits.iloc[0]["Trámite"]
    return {
        "tramite": tramite,
        "tipo": "directo",
        "hits": hits
    }

def call_agente_relevancia(message: str, token_trace=None):
    start = time.time()


    prompt = (
        "MENSAJE DEL USUARIO:\n"
        f"\"\"\"{message}\"\"\""
    )

    response = AGENTS["relevancia"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": RELEVANCIA_AGENT_NAME,
                "type": "agent_reference"
            }
        },
    )
    _track_usage(token_trace, "chat_relevancia", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms

def safe_agente_relevancia(message: str, token_trace=None):
    """
    Wrapper seguro:
    - Si detecta inyección/exfiltración -> NO_RELEVANTE
    - Si el agente de relevancia falla -> NO_RELEVANTE
    """
    if looks_like_injection(message) or is_prompt_leak_attempt(message):
        return "NO_RELEVANTE"
    try:
        text, _ = call_agente_relevancia(message, token_trace=token_trace)
        return text
    except Exception as e:
        print(f"[WARN] agente_relevancia falló ({type(e).__name__}). Se trata como NO_RELEVANTE.")
        return "NO_RELEVANTE"

def call_agente_preguntas(
    pregunta_sistema: str,
    mensaje_usuario: str,
    token_trace=None,
):
    start = time.time()

    prompt = (
        "PREGUNTA DEL SISTEMA:\n"
        f"\"\"\"{pregunta_sistema}\"\"\"\n\n"
        "NUEVO MENSAJE DEL USUARIO:\n"
        f"\"\"\"{mensaje_usuario}\"\"\""
    )

    response = AGENTS["preguntas"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": PREGUNTAS_AGENT_NAME,
                "type": "agent_reference"
            }
        },
    )
    _track_usage(token_trace, "chat_preguntas", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms

def call_agente_clarificador(
    pregunta_sistema: str,
    respuesta_usuario: str,
    hits: pd.DataFrame,
    token_trace=None,
):
    start = time.time()

    # Construir contexto de trámites
    tramites_contexto = "\n\n".join(
        f"- {row['Trámite']}\n  {row['Descripción']}"
        for _, row in hits.iterrows()
    )

    prompt = (
        "PREGUNTA DEL SISTEMA:\n"
        f"\"\"\"{pregunta_sistema}\"\"\"\n\n"
        "RESPUESTA DEL CIUDADANO:\n"
        f"\"\"\"{respuesta_usuario}\"\"\"\n\n"
        "TRÁMITES RELACIONADOS:\n"
        f"{tramites_contexto}"
    )

    response = AGENTS["clarificador"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": CLARIFICADOR_AGENT_NAME,
                "type": "agent_reference"
            }
        },
    )
    _track_usage(token_trace, "chat_clarificador", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms

def detectar_tipo_mensaje_pendiente(
    pregunta_sistema: str,
    mensaje_usuario: str,
    token_trace=None,
):
    """
    Devuelve:
      - ("respuesta", texto_normalizado)
      - ("pregunta", texto_pregunta_o_aclaracion)
    """
    try:
        salida, _ = call_agente_preguntas(
            pregunta_sistema=pregunta_sistema,
            mensaje_usuario=mensaje_usuario,
            token_trace=token_trace,
        )
    except Exception:
        salida = ""

    respuesta_val = (
        _extract_prefixed_value(salida, "RESPUESTA:")
        or _extract_prefixed_value(salida, "RESPUESTA_VALIDA:")
    )
    if respuesta_val:
        return "respuesta", respuesta_val

    pregunta_val = (
        _extract_prefixed_value(salida, "PREGUNTA:")
        or _extract_prefixed_value(salida, "ACLARACION:")
    )
    if pregunta_val:
        return "pregunta", pregunta_val

    # Fallback heurístico defensivo
    if "?" in (mensaje_usuario or ""):
        return "pregunta", mensaje_usuario
    return "respuesta", mensaje_usuario
 
# =========================
# MULTI-CHAT STATE
# =========================
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
        "pending_question": None,
        "pending_hits": None,
        "desambiguacion_intentos": 0,
        "relevancia_reintentos": 0,
    }
    if seed_assistant:
        convo["messages"].append({
            "role": "assistant",
            "content": "Hola 👋 Cuéntame tu caso y te digo el trámite del catálogo. Si me falta info, te haré preguntas para afinar.",
            "t": now_label()
        })
    return convo
 
if "conversations" not in st.session_state:
    first = new_conversation(seed_assistant=True)
    st.session_state.conversations = [first]
    st.session_state.active_id = first["id"]
 
def get_active():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    # fallback
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
 
# =========================
# SIDEBAR (lista de chats)
# =========================
with st.sidebar:
    st.image("assets/logo_junta_fondo_verde.jpeg", width="stretch")
    st.markdown("## Conversaciones")
 
    if st.button("➕ Nuevo chat", use_container_width=True):
        create_and_switch()
        st.rerun()
 
    st.write("")
 
    # Listado (más reciente arriba)
    # Guardamos sin tabla, estilo “lista”
    for c in sorted(st.session_state.conversations, key=lambda x: x["created_at"], reverse=True):
        is_active = (c["id"] == st.session_state.active_id)
        title = c["title"] or "Chat"
        suffix = " ✅" if c.get("closed") else ""
        label = f"{'👉 ' if is_active else ''}{title}{suffix}"
        if st.button(label, key=f"open_{c['id']}", use_container_width=True):
            set_active(c["id"])
            st.rerun()
 
    st.write("")
    st.markdown("---")
    active = get_active()
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Borrar chat", use_container_width=True):
            delete_conversation(active["id"])
            st.rerun()
    with colB:
        if st.button("Borrar todo", use_container_width=True):
            st.session_state.conversations = [new_conversation(seed_assistant=True)]
            st.session_state.active_id = st.session_state.conversations[0]["id"]
            st.rerun()

# =========================
# Agent clients
# =========================

AGENTS = load_agents()
 
# =========================
# MAIN UI (chat)
# =========================
active = get_active()

st.title("Asistente de Trámites")

# =========================
# Render mensajes del chat activo
# =========================

def normalize_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"

for msg in active["messages"]:
    role = normalize_role(msg.get("role", "user"))
    avatar = ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR

    with st.chat_message(role, avatar=avatar):
        st.write(msg["content"])
        if msg.get("t"):
            st.caption(msg["t"])

        if role == "assistant" and msg.get("token_usage"):
            tu = msg["token_usage"]
            st.caption(
                f"Tokens consulta: prompt={tu.get('prompt_tokens', 0)} | "
                f"completion={tu.get('completion_tokens', 0)} | "
                f"total={tu.get('total_tokens', 0)}"
            )
            calls = tu.get("calls") or []
            if calls:
                with st.expander("Ver tokens por llamada"):
                    for call in calls:
                        st.write(
                            f"{call.get('label', 'llamada')}: "
                            f"prompt={call.get('prompt_tokens', 0)} | "
                            f"completion={call.get('completion_tokens', 0)} | "
                            f"total={call.get('total_tokens', 0)}"
                        )

        # 🔍 Renderizado estable del debug
        if role == "assistant" and "debug" in msg:
            with st.expander("ℹ️ Ver detalles de clasificación"):
                render_debug_info(
                    hits=msg["debug"]["hits"],
                    umbral_ambiguedad=msg["debug"]["umbral"]
                )

# =========================
# Input
# =========================
user_text = st.chat_input("Escribe tu consulta…")

if user_text:
    query_token_trace = _new_query_token_trace()

    # Si este chat está cerrado (ya identificado antes), abrimos uno nuevo
    if active.get("closed"):
        create_and_switch()
        active = get_active()

    # Primer mensaje de usuario → título del chat
    if active["title"] == "Nuevo chat":
        active["title"] = make_title_from_text(user_text)

    # =========================
    # 1️⃣ Añadir y RENDERIZAR INMEDIATAMENTE el mensaje del usuario
    # =========================
    user_msg = {
        "role": "user",
        "content": user_text,
        "t": now_label()
    }
    active["messages"].append(user_msg)

    # =========================
    # PREGUNTAS + CLARIFICADOR
    # =========================
    if active.get("pending_question"):
        pregunta_sistema = active.get("pending_question") or ""
        pending_hits = active.get("pending_hits")

        with st.spinner("Analizando tu respuesta…"):
            tipo_msg, contenido = detectar_tipo_mensaje_pendiente(
                pregunta_sistema=pregunta_sistema,
                mensaje_usuario=user_text,
                token_trace=query_token_trace,
            )

        if tipo_msg == "pregunta":
            try:
                salida, _ = call_agente_clarificador(
                    pregunta_sistema=pregunta_sistema,
                    respuesta_usuario=contenido,
                    hits=pending_hits if pending_hits is not None else pd.DataFrame(),
                    token_trace=query_token_trace,
                )
                texto = _extract_prefixed_value(salida, "ACLARACION:") or salida.strip()
            except Exception:
                texto = (
                    "Claro. Necesito una respuesta concreta para poder clasificar tu trámite.\n\n"
                    f"{pregunta_sistema}"
                )

            active["messages"].append({
                "role": "assistant",
                "content": texto,
                "t": now_label(),
                "meta": "clarificador"
            })
            _attach_token_trace_to_last_assistant_message(active, query_token_trace)
            st.rerun()

        info = (contenido or "").strip() or user_text
        user_text = (
            f"{pregunta_sistema}\n\n"
            f"Respuesta del ciudadano: {info}"
        )
        active["pending_question"] = None
        active["pending_hits"] = None

    # =========================
    # SEGURIDAD + FUERA DE ÁMBITO (pre-clasificación)
    # =========================
    if looks_like_injection(user_text) or is_prompt_leak_attempt(user_text):
        active["messages"].append({
            "role": "assistant",
            "content": MENSAJE_FUERA_DE_AMBITO,
            "t": now_label()
        })
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        _attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    if fuera_de_ambito_rapido(user_text):
        active["messages"].append({
            "role": "assistant",
            "content": MENSAJE_FUERA_DE_AMBITO,
            "t": now_label()
        })
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        _attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    relevancia_precheck = safe_agente_relevancia(user_text, token_trace=query_token_trace)
    if _is_no_relevante(relevancia_precheck):
        active["messages"].append({
            "role": "assistant",
            "content": MENSAJE_FUERA_DE_AMBITO,
            "t": now_label()
        })
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        _attach_token_trace_to_last_assistant_message(active, query_token_trace)
        st.rerun()

    # =========================
    # Nuevo flujo de CLASIFICACIÓN (V2)
    # =========================
    with st.spinner("Clasificando consulta…"):
        try:
            resultado = clasificar_con_desambiguacion(
                message=user_text,
                token_trace=query_token_trace,
            )
        except Exception as e:
            resultado = {
                "tipo": "error",
                "mensaje": f"Error del sistema: {e}"
            }

    # =========================
    # RESPUESTA DEL CLASIFICADOR V2
    # =========================
    tipo = resultado.get("tipo")

    print(f"Resultado clasificación: {tipo}")

    # ─────────────────────────
    # ✅ CASO 1: CLASIFICACIÓN DIRECTA O DESAMBIGUADA
    # ─────────────────────────
    if tipo in ("directo", "desambiguado"):
        tramite = resultado["tramite"]

        clasificador_text = f"✅ Identificado: **{tramite}**"

        active["messages"].append({
            "role": "assistant",
            "content": clasificador_text,
            "t": now_label(),
            "debug": {
                "hits": resultado["hits"],
                "umbral": UMBRAL_AMBIGUEDAD
            }
        })

        # Cerramos clasificación
        active["identified_tramite"] = tramite
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None
        active["desambiguacion_intentos"] = 0
        active["relevancia_reintentos"] = 0

        # =========================
        # Llamada al AGENTE RAG
        # =========================
        try:
            rag_text, rag_ms = call_asistente_tramites(tramite, token_trace=query_token_trace)
        except Exception as e:
            rag_text = f"No se ha podido recuperar la información del trámite. {e}"

        # =========================
        # Sustituir placeholder por respuesta real
        # =========================

        active["messages"].append({
            "role": "assistant",
            "content": rag_text,
            "t": now_label()
        })

    # ─────────────────────────
    # ❓ CASO 2: NECESITA MÁS INFORMACIÓN (DESAMBIGUACIÓN)
    # ─────────────────────────
    elif tipo == "necesita_info":
        active["relevancia_reintentos"] = 0
        intentos = int(active.get("desambiguacion_intentos", 0)) + 1
        active["desambiguacion_intentos"] = intentos

        if intentos > MAX_INTENTOS_DESAMBIGUACION:
            mensaje = _format_no_clasificable_msg(resultado.get("ambiguous_hits", resultado.get("hits")))
            active["pending_question"] = None
            active["pending_hits"] = None
            active["desambiguacion_intentos"] = 0
            active["messages"].append({
                "role": "assistant",
                "content": mensaje,
                "t": now_label()
            })
        else:
            pregunta = resultado["pregunta"]
            active["pending_question"] = pregunta
            active["pending_hits"] = resultado["hits"]

            active["messages"].append({
                "role": "assistant",
                "content": pregunta,
                "t": now_label(),
                "debug": {
                    "hits": resultado["hits"],
                    "umbral": UMBRAL_AMBIGUEDAD
                }
            })

    # ─────────────────────────
    # ⚠️ CASO 3: FUERA DE ÁMBITO
    # ─────────────────────────
    elif tipo == "fuera_de_ambito":
        mensaje = MENSAJE_FUERA_DE_AMBITO

        active["messages"].append({
            "role": "assistant",
            "content": mensaje,
            "t": now_label()
        })
        active["closed"] = True
        active["pending_question"] = None
        active["pending_hits"] = None

    # ─────────────────────────
    # ❌ CASO 4: NO CLASIFICABLE / SCORE BAJO
    # ─────────────────────────
    elif tipo == "score_bajo":
        active["desambiguacion_intentos"] = 0
        active["pending_question"] = None
        active["pending_hits"] = None

        relevancia_text = relevancia_precheck

        if _is_no_relevante(relevancia_text):
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

        active["messages"].append({
            "role": "assistant",
            "content": mensaje,
            "t": now_label()
        })

    elif tipo == "no_clasificable":
        mensaje = resultado.get("mensaje", MENSAJE_SCORE_BAJO)
        active["messages"].append({
            "role": "assistant",
            "content": mensaje,
            "t": now_label()
        })

    # ─────────────────────────
    # ❗ CASO 5: ERROR
    # ─────────────────────────
    else:
        mensaje = resultado.get("mensaje", "Ha ocurrido un error inesperado!")

        active["messages"].append({
            "role": "assistant",
            "content": mensaje,
            "t": now_label()
        })

    _attach_token_trace_to_last_assistant_message(active, query_token_trace)
    st.rerun()

