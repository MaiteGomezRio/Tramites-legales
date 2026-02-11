import streamlit as st
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import (
    AIPROJECT_ENDPOINT,
    ASISTENTE_AGENT_NAME,
    CLARIFICADOR_AGENT_NAME,
    CLASIFICADOR_AGENT_NAME,
    DESAMBIGUADOR_AGENT_NAME,
    RELEVANCIA_AGENT_NAME,
    PREGUNTAS_AGENT_NAME
)


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
        project_client.agents.get(agent_name=agent_name)
    except Exception as e:
        print(f"Error al obtener agente '{agent_name}': {e}")

    openai_client = project_client.get_openai_client()
    return openai_client


@st.cache_resource(show_spinner="Cargando agente…")
def load_agents():
    return {
        # "clasificador": get_agent(CLASIFICADOR_AGENT_NAME),
        "asistente": get_agent(ASISTENTE_AGENT_NAME),
        "desambiguador": get_agent(DESAMBIGUADOR_AGENT_NAME),
        "clarificador": get_agent(CLARIFICADOR_AGENT_NAME),
        "relevancia": get_agent(RELEVANCIA_AGENT_NAME),
        "preguntas": get_agent(PREGUNTAS_AGENT_NAME),
    }
