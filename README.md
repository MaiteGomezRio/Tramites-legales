# Asistente de Trámites Legales

Aplicación web en Streamlit para clasificar consultas ciudadanas sobre trámites legales y devolver información orientativa del trámite identificado.

El flujo combina:
- `embeddings` sobre un catálogo local (`descripcion-tramites.xlsx`)
- desambiguación con agentes cuando hay candidatos similares
- un asistente final que explica el trámite detectado

## Funcionalidades

- Clasificación semántica de consultas con `text-embedding-3-small`.
- Desambiguación guiada cuando hay varios trámites cercanos.
- Preguntas de clarificación en caso de información insuficiente.
- Control de ámbito (consultas fuera de trámites judiciales).
- Historial de conversaciones en sesión de Streamlit.
- Panel de debug con scores de clasificación.

## Stack técnico

- Python 3.10+
- Streamlit
- OpenAI Python SDK
- Azure AI Projects (`azure-ai-projects`, `azure-identity`)
- Pandas, NumPy, OpenPyXL
- python-dotenv

## Estructura del proyecto

- `app.py`: entry point de Streamlit.
- `config.py`: configuración global, umbrales y clientes.
- `descripcion-tramites.xlsx`: catálogo de trámites.
- `agents/client.py`: carga clientes/agentes de Azure.
- `agents/services.py`: llamadas a clasificador/asistente/desambiguador.
- `core/retrieval.py`: embeddings, similitud y top-k.
- `core/helpers.py`: utilidades varias.
- `conversations/state.py`: estado multi-chat en sesión.
- `ui/styles.py`: carga de CSS.
- `styles/theme_accenture.css`: tema visual.

## Requisitos previos

1. Tener Python 3.10 o superior.
2. Tener acceso a Azure AI Project con los agentes configurados.
3. Tener credenciales de Azure disponibles para `DefaultAzureCredential` (por ejemplo, `az login` en local).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit openai azure-ai-projects azure-identity pandas numpy openpyxl python-dotenv
```

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Azure AI Project
AZURE_EXISTING_AIPROJECT_ENDPOINT=https://<tu-proyecto>.services.ai.azure.com/api/projects/<nombre-proyecto>

# Nombres de agentes (opcionales si usas los defaults del código)
AZURE_CLASIFICADOR_AGENT_NAME=ClasificadorTramites
AZURE_ASISTENTE_AGENT_NAME=AsistenteTramites
AZURE_DESAMBIGUADOR_AGENT_NAME=DesambiguadorTramites
AZURE_RELEVANCIA_AGENT_NAME=RelevanciaRespuestaTramites
AZURE_CLARIFICADOR_AGENT_NAME=ClarificadorTramites

# Modo de respuesta final
# agent = usa el agente AsistenteTramites de Foundry
# local = sin llamada al agente asistente (tokens muy bajos)
ASISTENTE_MODE=agent
ASISTENTE_LOCAL_MAX_CHARS=1600

# Optimización de tokens en desambiguación (opcionales)
DESAMBIGUADOR_MAX_CANDIDATOS=3
DESAMBIGUADOR_MAX_DESC_CHARS=420

# Azure OpenAI (embeddings)
AZURE_OPENAI_API_KEY=<tu_api_key>
AZURE_OPENAI_BASE_URL=https://<tu-recurso>.openai.azure.com/openai/v1/
```

## Ejecución

```bash
streamlit run app.py
```

Abre en el navegador la URL que muestra Streamlit (normalmente `http://localhost:8501`).

## Flujo de clasificación (resumen)

1. Se embebe la consulta del usuario.
2. Se compara contra el catálogo de trámites.
3. Si hay un trámite dominante, se clasifica directamente.
4. Si hay ambigüedad, actúa el agente desambiguador:
   - `CLASIFICACION:<tramite>` o
   - `PREGUNTA:<pregunta>`
5. Si hay pregunta, se usa el agente clarificador con la respuesta del usuario.
6. Una vez identificado el trámite, el asistente devuelve información orientativa.

## Troubleshooting rápido

- Error `Falta AZURE_EXISTING_AIPROJECT_ENDPOINT`:
  - Define esa variable en `.env`.
- Error de autenticación con Azure:
  - Ejecuta `az login` o configura credenciales de servicio.
- No aparece el trámite esperado:
  - Revisa `descripcion-tramites.xlsx` y los umbrales en `config.py`.
- Arranque lento en la primera carga:
  - Es normal; se calculan embeddings del catálogo y luego quedan cacheados.
