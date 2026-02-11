import time
from typing import Dict, Tuple

import pandas as pd

from config import (
    ASISTENTE_AGENT_NAME,
    CLARIFICADOR_AGENT_NAME,
    DESAMBIGUADOR_AGENT_NAME,
    PREGUNTAS_AGENT_NAME,
    RELEVANCIA_AGENT_NAME,
    UMBRAL_AMBIGUEDAD,
    UMBRAL_RETRIEVAL,
    UMBRAL_SCORE_MINIMO,
)
from core.helpers import (
    extract_prefixed_value,
    is_prompt_leak_attempt,
    looks_like_injection,
    track_usage,
)
from core.retrieval import retrieve_top_k


def _agent_client(agents: Dict, key: str):
    entry = agents.get(key)
    if isinstance(entry, dict):
        return entry.get("client")
    return entry


def _agent_name(agents: Dict, key: str, default_name: str) -> str:
    entry = agents.get(key)
    if isinstance(entry, dict):
        return (entry.get("name") or default_name).strip()
    return default_name


def call_asistente_tramites(agents: Dict, tramite_nombre: str, token_trace=None) -> Tuple[str, int]:
    start = time.time()

    prompt = (
        f"El trámite identificado es: {tramite_nombre}.\n"
        "Proporciona información clara y orientativa para un ciudadano."
    )

    response = _agent_client(agents, "asistente").responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": _agent_name(agents, "asistente", ASISTENTE_AGENT_NAME),
                "type": "agent_reference",
            }
        },
    )
    track_usage(token_trace, "chat_asistente_tramites", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def call_agente_desambiguador(agents: Dict, message: str, ambiguous_hits: pd.DataFrame, token_trace=None) -> Tuple[str, int]:
    start = time.time()

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

    response = _agent_client(agents, "desambiguador").responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": _agent_name(agents, "desambiguador", DESAMBIGUADOR_AGENT_NAME),
                "type": "agent_reference",
            }
        },
    )
    track_usage(token_trace, "chat_desambiguador", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def clasificar_con_desambiguacion(
    message: str,
    agents: Dict,
    k: int = 8,
    gap_max: float = UMBRAL_RETRIEVAL,
    umbral: float = UMBRAL_AMBIGUEDAD,
    umbral_minimo: float = UMBRAL_SCORE_MINIMO,
    token_trace=None,
):
    hits = retrieve_top_k(
        message=message,
        k=k,
        gap_max=gap_max,
        token_trace=token_trace,
    )
    scores = hits["score"].values

    if len(scores) == 0:
        return {"tramite": None, "tipo": "score_bajo", "hits": hits}

    if scores[0] < umbral_minimo:
        return {"tramite": None, "tipo": "score_bajo", "hits": hits}

    ambiguous_mask = (scores[0] - scores) < umbral
    n_ambiguous = ambiguous_mask.sum()

    if n_ambiguous > 1:
        ambiguous_hits = hits[ambiguous_mask]

        print(f"  Ambigüedad detectada: {n_ambiguous} candidatos con diferencia < {umbral}")
        for _, row in ambiguous_hits.iterrows():
            print(f"   {row['score']:.4f} | {row['Trámite']}")

        resultado, ms = call_agente_desambiguador(
            agents=agents,
            message=message,
            ambiguous_hits=ambiguous_hits,
            token_trace=token_trace,
        )

        print(f"  Resultado del agente de desambiguación (en {ms} ms): {resultado}")

        if resultado.startswith("CLASIFICACION:"):
            tramite = resultado.replace("CLASIFICACION:", "").strip()
            return {
                "tramite": tramite,
                "tipo": "desambiguado",
                "hits": hits,
            }

        if resultado.startswith("PREGUNTA:"):
            pregunta = resultado.replace("PREGUNTA:", "").strip()
            return {
                "tramite": None,
                "pregunta": pregunta,
                "tipo": "necesita_info",
                "hits": hits,
                "ambiguous_hits": ambiguous_hits,
            }

        return {
            "tramite": None,
            "tipo": "error_desambiguador",
            "raw_output": resultado,
            "hits": hits,
        }

    tramite = hits.iloc[0]["Trámite"]
    return {
        "tramite": tramite,
        "tipo": "directo",
        "hits": hits,
    }


def call_agente_relevancia(agents: Dict, message: str, token_trace=None) -> Tuple[str, int]:
    start = time.time()

    prompt = (
        "MENSAJE DEL USUARIO:\n"
        f"\"\"\"{message}\"\"\""
    )

    response = _agent_client(agents, "relevancia").responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": _agent_name(agents, "relevancia", RELEVANCIA_AGENT_NAME),
                "type": "agent_reference",
            }
        },
    )
    track_usage(token_trace, "chat_relevancia", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def safe_agente_relevancia(agents: Dict, message: str, token_trace=None) -> str:
    if looks_like_injection(message) or is_prompt_leak_attempt(message):
        return "NO_RELEVANTE"
    try:
        text, _ = call_agente_relevancia(agents=agents, message=message, token_trace=token_trace)
        return text
    except Exception as e:
        print(f"[WARN] agente_relevancia falló ({type(e).__name__}). Se trata como NO_RELEVANTE.")
        return "NO_RELEVANTE"


def call_agente_preguntas(agents: Dict, pregunta_sistema: str, mensaje_usuario: str, token_trace=None) -> Tuple[str, int]:
    start = time.time()

    prompt = (
        "PREGUNTA DEL SISTEMA:\n"
        f"\"\"\"{pregunta_sistema}\"\"\"\n\n"
        "NUEVO MENSAJE DEL USUARIO:\n"
        f"\"\"\"{mensaje_usuario}\"\"\""
    )

    response = _agent_client(agents, "preguntas").responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": _agent_name(agents, "preguntas", PREGUNTAS_AGENT_NAME),
                "type": "agent_reference",
            }
        },
    )
    track_usage(token_trace, "chat_preguntas", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def call_agente_clarificador(
    agents: Dict,
    pregunta_sistema: str,
    respuesta_usuario: str,
    hits: pd.DataFrame,
    token_trace=None,
) -> Tuple[str, int]:
    start = time.time()

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

    response = _agent_client(agents, "clarificador").responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": _agent_name(agents, "clarificador", CLARIFICADOR_AGENT_NAME),
                "type": "agent_reference",
            }
        },
    )
    track_usage(token_trace, "chat_clarificador", getattr(response, "usage", None))

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def detectar_tipo_mensaje_pendiente(agents: Dict, pregunta_sistema: str, mensaje_usuario: str, token_trace=None):
    try:
        salida, _ = call_agente_preguntas(
            agents=agents,
            pregunta_sistema=pregunta_sistema,
            mensaje_usuario=mensaje_usuario,
            token_trace=token_trace,
        )
    except Exception:
        salida = ""

    respuesta_val = extract_prefixed_value(salida, "RESPUESTA:") or extract_prefixed_value(
        salida, "RESPUESTA_VALIDA:"
    )
    if respuesta_val:
        return "respuesta", respuesta_val

    pregunta_val = extract_prefixed_value(salida, "PREGUNTA:") or extract_prefixed_value(
        salida, "ACLARACION:"
    )
    if pregunta_val:
        return "pregunta", pregunta_val

    if "?" in (mensaje_usuario or ""):
        return "pregunta", mensaje_usuario
    return "respuesta", mensaje_usuario
