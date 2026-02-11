import time

import pandas as pd

from config import (
    ASISTENTE_AGENT_NAME,
    CLARIFICADOR_AGENT_NAME,
    CLIENT,
    CLASIFICADOR_AGENT_NAME,
    DESAMBIGUADOR_AGENT_NAME,
    RELEVANCIA_AGENT_NAME,
    UMBRAL_AMBIGUEDAD,
    UMBRAL_RETRIEVAL,
    UMBRAL_SCORE_MINIMO,
)
from core.helpers import try_parse_json
from core.retrieval import retrieve_top_k


def call_clasificador(messages, agents):
    start = time.time()

    response = agents["clasificador"].responses.create(
        input=messages,
        extra_body={"agent": {"name": CLASIFICADOR_AGENT_NAME, "type": "agent_reference"}},
    )

    raw_text = getattr(response, "output_text", "") or ""
    parsed = try_parse_json(raw_text)
    ms = int((time.time() - start) * 1000)

    if isinstance(parsed, dict) and parsed.get("status") == "identified":
        return "identified", parsed, ms

    return "text", raw_text.strip(), ms


def call_asistente_tramites(tramite_nombre: str, agents):
    start = time.time()

    prompt = (
        f"El trámite identificado es: {tramite_nombre}.\n"
        "Proporciona información clara y orientativa para un ciudadano."
    )

    response = agents["asistente"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={"agent": {"name": ASISTENTE_AGENT_NAME, "type": "agent_reference"}},
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    return text.strip(), ms


def call_agente_desambiguador(message: str, ambiguous_hits: pd.DataFrame, agents):
    start = time.time()

    candidates_detail = "\n\n".join(
        f"- {row['Trámite']} (score: {row['score']:.4f})\n"
        f"  Descripción: {row['Descripción']}"
        for _, row in ambiguous_hits.iterrows()
    )

    prompt = (
        "MENSAJE DEL CIUDADANO:\n"
        f'"""{message}"""\n\n'
        "CATEGORÍAS AMBIGUAS:\n"
        f"{candidates_detail}"
    )

    response = agents["desambiguador"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": DESAMBIGUADOR_AGENT_NAME,
                "type": "agent_reference",
            }
        },
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms


def clasificar_con_desambiguacion(
    message,
    catalog_df,
    embeddings,
    agents,
    k=8,
    gap_max=UMBRAL_RETRIEVAL,
    umbral=UMBRAL_AMBIGUEDAD,
    umbral_minimo=UMBRAL_SCORE_MINIMO,
):
    hits = retrieve_top_k(message, catalog_df, embeddings, CLIENT, k=k, gap_max=gap_max)
    scores = hits["score"].values

    if scores[0] < umbral_minimo:
        return {
            "tramite": None,
            "tipo": "score_bajo",
            "hits": hits,
        }

    ambiguous_mask = (scores[0] - scores) < umbral
    n_ambiguous = ambiguous_mask.sum()

    if n_ambiguous > 1:
        ambiguous_hits = hits[ambiguous_mask]

        print(f"Ambigüedad detectada: {n_ambiguous} candidatos con diferencia < {umbral}")
        for _, row in ambiguous_hits.iterrows():
            print(f"{row['score']:.4f} | {row['Trámite']}")

        resultado, ms = call_agente_desambiguador(message=message, ambiguous_hits=ambiguous_hits, agents=agents)

        print(f"Resultado del agente de desambiguación (en {ms} ms): {resultado}")

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


def call_agente_relevancia(message: str, agents):
    start = time.time()

    prompt = "MENSAJE DEL USUARIO:\n" f'"""{message}"""'

    response = agents["relevancia"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": RELEVANCIA_AGENT_NAME,
                "type": "agent_reference",
            }
        },
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms


def call_agente_clarificador(pregunta_sistema: str, respuesta_usuario: str, hits: pd.DataFrame, agents):
    start = time.time()

    tramites_contexto = "\n\n".join(
        f"- {row['Trámite']}\n  {row['Descripción']}" for _, row in hits.iterrows()
    )

    prompt = (
        "PREGUNTA DEL SISTEMA:\n"
        f'"""{pregunta_sistema}"""\n\n'
        "RESPUESTA DEL CIUDADANO:\n"
        f'"""{respuesta_usuario}"""\n\n'
        "TRÁMITES RELACIONADOS:\n"
        f"{tramites_contexto}"
    )

    response = agents["clarificador"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": CLARIFICADOR_AGENT_NAME,
                "type": "agent_reference",
            }
        },
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)

    return text.strip(), ms
