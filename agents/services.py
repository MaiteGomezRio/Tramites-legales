import time
import re
from typing import Optional

import pandas as pd

from config import (
    ASISTENTE_LOCAL_MAX_CHARS,
    ASISTENTE_MODE,
    ASISTENTE_AGENT_NAME,
    CLARIFICADOR_AGENT_NAME,
    CLIENT,
    CLASIFICADOR_AGENT_NAME,
    DESAMBIGUADOR_MAX_CANDIDATOS,
    DESAMBIGUADOR_MAX_DESC_CHARS,
    DESAMBIGUADOR_AGENT_NAME,
    RELEVANCIA_AGENT_NAME,
    PREGUNTAS_AGENT_NAME,
    UMBRAL_AMBIGUEDAD,
    UMBRAL_RETRIEVAL,
    UMBRAL_SCORE_MINIMO,
)
from core.helpers import try_parse_json
from core.retrieval import retrieve_top_k


def _normalize_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        total_tokens = usage.get("total_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_clasificador(messages, agents):
    start = time.time()

    response = agents["clasificador"].responses.create(
        input=messages,
        extra_body={"agent": {"name": CLASIFICADOR_AGENT_NAME, "type": "agent_reference"}},
    )

    raw_text = getattr(response, "output_text", "") or ""
    parsed = try_parse_json(raw_text)
    ms = int((time.time() - start) * 1000)
    usage = _normalize_usage(response)

    if isinstance(parsed, dict) and parsed.get("status") == "identified":
        return "identified", parsed, ms, usage

    return "text", raw_text.strip(), ms, usage


def call_asistente_tramites(
    tramite_nombre: str,
    agents,
    descripcion: Optional[str] = None,
):
    start = time.time()

    if ASISTENTE_MODE != "agent":
        text = build_local_asistente_answer(tramite_nombre, descripcion=descripcion or "")
        ms = int((time.time() - start) * 1000)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return text.strip(), ms, usage

    prompt = (
        f"El trámite identificado es: {tramite_nombre}.\n"
        f"Resumen del catálogo: {_shorten_text(descripcion or '', 1200)}\n\n"
        "Proporciona información clara y orientativa para un ciudadano."
    )

    assistant_client = agents.get("asistente")
    if assistant_client is None:
        # Puede faltar por caché de Streamlit tras cambiar ASISTENTE_MODE en caliente.
        from agents.client import get_agent

        assistant_client = get_agent(ASISTENTE_AGENT_NAME)
        agents["asistente"] = assistant_client

    response = assistant_client.responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={"agent": {"name": ASISTENTE_AGENT_NAME, "type": "agent_reference"}},
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    usage = _normalize_usage(response)
    return text.strip(), ms, usage


def _split_fragments(text: str) -> list[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    return [f.strip(" -•\t") for f in re.split(r"(?<=[\.;:!?])\s+|\n+", compact) if f.strip()]


def _score_fragment(fragment: str, index: int) -> int:
    low = fragment.lower()
    keywords = (
        "euros", "cuant", "deuda", "laboral", "civil", "abogado", "procurador",
        "document", "demanda", "juzgado", "plazo", "notific", "oposic",
        "desahucio", "alquiler", "monitorio", "verbal",
    )
    score = sum(1 for kw in keywords if kw in low)
    if any(ch.isdigit() for ch in low):
        score += 2
    if index < 4:
        score += 1
    if len(fragment) < 45:
        score -= 1
    return score


def build_local_asistente_answer(tramite_nombre: str, descripcion: str) -> str:
    fragments = _split_fragments(descripcion)
    picked = []

    if fragments:
        ranked = []
        for idx, fragment in enumerate(fragments):
            ranked.append((idx, _score_fragment(fragment, idx), fragment))
        ranked.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
        selected = sorted(ranked[:5], key=lambda x: x[0])

        total_chars = 0
        for _, _, fragment in selected:
            clean = _shorten_text(fragment, 300)
            if not clean:
                continue
            if total_chars + len(clean) > ASISTENTE_LOCAL_MAX_CHARS:
                continue
            picked.append(clean)
            total_chars += len(clean)
            if len(picked) >= 4:
                break

    if not picked:
        fallback = _shorten_text(descripcion or "", min(ASISTENTE_LOCAL_MAX_CHARS, 700))
        picked = [fallback] if fallback else ["No dispongo de más detalle para este trámite en el catálogo."]

    lines = [
        f"He identificado el trámite: {tramite_nombre}.",
        "",
        "Información orientativa:",
    ]
    lines.extend([f"- {p}" for p in picked])
    lines.extend([
        "",
        "Si quieres, te indico los pasos concretos para tu caso (documentos, órgano y presentación).",
    ])
    return "\n".join(lines)


def _normalize_binary_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    normalized = " ".join(str(text).strip().lower().split())
    yes_values = {"si", "sí", "s", "afirmativo", "correcto", "exacto", "yes"}
    no_values = {"no", "n", "negativo"}

    if normalized in yes_values:
        return "SI"
    if normalized in no_values:
        return "NO"
    return None


_STOPWORDS_ES = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "en", "con", "por", "para", "del", "al", "que", "se", "su", "sus", "es",
    "son", "como", "sobre", "mi", "tu", "si", "sí", "no", "me", "te", "le",
}


def _shorten_text(text: str, max_chars: int) -> str:
    max_chars = max(80, int(max_chars or 0))
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _keyword_set(*texts: Optional[str]) -> set[str]:
    tokens = []
    for text in texts:
        if not text:
            continue
        tokens.extend(re.findall(r"[a-záéíóúñü0-9]{3,}", str(text).lower()))
    return {t for t in tokens if t not in _STOPWORDS_ES}


def _relevant_desc_snippet(description: str, keywords: set[str], max_chars: int) -> str:
    compact = " ".join(str(description or "").split())
    if len(compact) <= max_chars:
        return compact

    fragments = [f.strip() for f in re.split(r"(?<=[\.;:])\s+|\n+", compact) if f.strip()]
    if not fragments:
        return _shorten_text(compact, max_chars)

    scored = []
    for idx, fragment in enumerate(fragments):
        low = fragment.lower()
        score = sum(1 for kw in keywords if kw in low)
        if any(ch.isdigit() for ch in low):
            score += 1
        scored.append((idx, score, fragment))

    scored.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
    selected = sorted(scored[:2], key=lambda x: x[0])
    snippet = " ".join(item[2] for item in selected).strip()
    return _shorten_text(snippet or compact, max_chars)


def _build_desambiguador_candidates(
    ambiguous_hits: pd.DataFrame,
    message: str,
    pregunta_sistema: Optional[str],
    respuesta_usuario: Optional[str],
) -> str:
    rows = (
        ambiguous_hits.sort_values("score", ascending=False)
        .head(max(1, DESAMBIGUADOR_MAX_CANDIDATOS))
        .reset_index(drop=True)
    )
    keywords = _keyword_set(message, pregunta_sistema, respuesta_usuario)

    lines = []
    for idx, row in rows.iterrows():
        tramite = str(row.get("Trámite", "") or "").strip()
        score = float(row.get("score", 0.0))
        resumen = _relevant_desc_snippet(
            row.get("Descripción", ""),
            keywords=keywords,
            max_chars=DESAMBIGUADOR_MAX_DESC_CHARS,
        )
        lines.append(f"{idx + 1}. {tramite} | score={score:.4f} | resumen={resumen}")
    return "\n".join(lines)


def call_agente_desambiguador(
    message: str,
    ambiguous_hits: pd.DataFrame,
    agents,
    pregunta_sistema: Optional[str] = None,
    respuesta_usuario: Optional[str] = None,
):
    start = time.time()

    candidates_detail = _build_desambiguador_candidates(
        ambiguous_hits=ambiguous_hits,
        message=message,
        pregunta_sistema=pregunta_sistema,
        respuesta_usuario=respuesta_usuario,
    )

    prompt_sections = [
        "Devuelve SOLO uno de estos formatos:\n"
        "- CLASIFICACION:<tramite exacto>\n"
        "- PREGUNTA:<pregunta concreta para desambiguar>",
        "CONSULTA ORIGINAL DEL CIUDADANO:\n"
        f'"""{message}"""',
    ]

    if pregunta_sistema and respuesta_usuario:
        respuesta_normalizada = _normalize_binary_answer(respuesta_usuario)
        qa_block = (
            "PREGUNTA HECHA AL CIUDADANO:\n"
            f'"""{pregunta_sistema}"""\n\n'
            "RESPUESTA DEL CIUDADANO:\n"
            f'"""{respuesta_usuario}"""'
        )
        if respuesta_normalizada:
            qa_block += f"\n\nRESPUESTA_BINARIA_NORMALIZADA: {respuesta_normalizada}"
        prompt_sections.append(qa_block)

    prompt_sections.append("CATEGORÍAS AMBIGUAS:\n" f"{candidates_detail}")
    prompt = "\n\n".join(prompt_sections)

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
    usage = _normalize_usage(response)

    return text.strip(), ms, usage


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

        resultado, ms, _ = call_agente_desambiguador(
            message=message,
            ambiguous_hits=ambiguous_hits,
            agents=agents,
        )

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
    usage = _normalize_usage(response)

    return text.strip(), ms, usage

def call_agente_preguntas(pregunta_sistema: str, mensaje_usuario: str, agents):
    start = time.time()

    prompt = (
        "PREGUNTA DEL SISTEMA:\n"
        f"\"\"\"{pregunta_sistema}\"\"\"\n\n"
        "NUEVO MENSAJE DEL USUARIO:\n"
        f"\"\"\"{mensaje_usuario}\"\"\""
    )

    response = agents["preguntas"].responses.create(
        input=[{"role": "user", "content": prompt}],
        extra_body={
            "agent": {
                "name": PREGUNTAS_AGENT_NAME,
                "type": "agent_reference"
            }
        },
    )

    text = getattr(response, "output_text", "") or ""
    ms = int((time.time() - start) * 1000)
    usage = _normalize_usage(response)

    return text.strip(), ms, usage

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
    usage = _normalize_usage(response)

    return text.strip(), ms, usage
