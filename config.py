import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "Asistente de Trámites"
APP_ICON = "🧾"
APP_LAYOUT = "wide"

AIPROJECT_ENDPOINT = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT", "")

CLASIFICADOR_AGENT_NAME = os.getenv("AZURE_CLASIFICADOR_AGENT_NAME", "ClasificadorTramites")
ASISTENTE_AGENT_NAME = os.getenv("AZURE_ASISTENTE_AGENT_NAME", "AsistenteTramites")
DESAMBIGUADOR_AGENT_NAME = os.getenv("AZURE_DESAMBIGUADOR_AGENT_NAME", "DesambiguadorTramites")
RELEVANCIA_AGENT_NAME = os.getenv("AZURE_RELEVANCIA_AGENT_NAME", "RelevanciaRespuestaTramites")
CLARIFICADOR_AGENT_NAME = os.getenv("AZURE_CLARIFICADOR_AGENT_NAME", "ClarificadorTramites")
PREGUNTAS_AGENT_NAME = os.getenv("AZURE_PREGUNTAS_AGENT_NAME", "PreguntasTramites")

ASSISTANT_AVATAR = "assets/logo_junta_avatar.png"
USER_AVATAR = "assets/user_avatar.png"

CSS_PATH = "styles/theme_accenture.css"
CATALOG_PATH = "descripcion-tramites.xlsx"

UMBRAL_RETRIEVAL = 0.14
UMBRAL_AMBIGUEDAD = 0.05
UMBRAL_SCORE_MINIMO = 0.35
MAX_INTENTOS_DESAMBIGUACION = 2

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

MENSAJE_ERROR_RELEVANCIA = (
    "He tenido dificultades para evaluar la relevancia de tu consulta. "
    "¿Podrías reformularla o aportar más detalles?"
)

EMB_MODEL = "text-embedding-3-small"

CLIENT = OpenAI(
    api_key=os.getenv(
        "AZURE_OPENAI_API_KEY",
        "6TZDD5vTve7kB5DVixnKnoGUtYUahlVxuEzuuU3KWXu9lTagcLWYJQQJ99CBACHYHv6XJ3w3AAABACOGj8AC",
    ),
    base_url=os.getenv(
        "AZURE_OPENAI_BASE_URL",
        "https://azure-openai-tramites-legales.openai.azure.com/openai/v1/",
    ),
)

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY", "")
INDEX_NAME = os.getenv("AZURE_INDEX_NAME", "tramites-index")
SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "embedding")
SEARCH_TRAMITE_FIELD = os.getenv("AZURE_SEARCH_TRAMITE_FIELD", "tramite")
SEARCH_DESCRIPCION_FIELD = os.getenv("AZURE_SEARCH_DESCRIPCION_FIELD", "descripcion")