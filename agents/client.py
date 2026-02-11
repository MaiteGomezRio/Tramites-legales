import streamlit as st
from openai import OpenAI

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import (
    AIPROJECT_ENDPOINT,
    ASISTENTE_AGENT_NAME,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    CLARIFICADOR_AGENT_NAME,
    CLASIFICADOR_AGENT_NAME,
    DESAMBIGUADOR_AGENT_NAME,
    PREGUNTAS_AGENT_NAME,
    RELEVANCIA_AGENT_NAME,
)


@st.cache_resource(show_spinner=False)
def get_embedding_client() -> OpenAI:
    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_BASE_URL,
    )


@st.cache_resource(show_spinner=False)
def get_project_client() -> AIProjectClient:
    if not AIPROJECT_ENDPOINT:
        raise RuntimeError("Falta AZURE_EXISTING_AIPROJECT_ENDPOINT.")
    return AIProjectClient(
        endpoint=AIPROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )


@st.cache_resource(show_spinner=False)
def get_agent_runtime(agent_name: str):
    requested = (agent_name or "").strip().strip("\"").strip("'")
    project_client = get_project_client()

    candidates = [requested]
    if ":" in requested:
        base_name = requested.split(":", 1)[0].strip()
        if base_name and base_name not in candidates:
            candidates.append(base_name)

    resolved_name = requested
    last_error = None
    for candidate in candidates:
        try:
            print(f"Obteniendo agente '{candidate}' de Foundry...")
            project_client.agents.get(agent_name=candidate)
            resolved_name = candidate
            if candidate != requested:
                print(f"Usando fallback de agente: '{requested}' -> '{candidate}'")
            break
        except Exception as e:
            print(f"Error al obtener agente '{candidate}': {e}")
            last_error = e

    if last_error is not None and resolved_name == requested and len(candidates) > 1:
        print(
            f"No se pudo resolver fallback para '{requested}'. "
            f"Se mantiene el nombre original."
        )

    return {
        "client": project_client.get_openai_client(),
        "name": resolved_name,
    }


@st.cache_resource(show_spinner="Cargando agente…")
def load_agents():
    return {
        "clasificador": get_agent_runtime(CLASIFICADOR_AGENT_NAME),
        "asistente": get_agent_runtime(ASISTENTE_AGENT_NAME),
        "desambiguador": get_agent_runtime(DESAMBIGUADOR_AGENT_NAME),
        "clarificador": get_agent_runtime(CLARIFICADOR_AGENT_NAME),
        "relevancia": get_agent_runtime(RELEVANCIA_AGENT_NAME),
        "preguntas": get_agent_runtime(PREGUNTAS_AGENT_NAME),
    }
