import os
from dotenv import load_dotenv

load_dotenv()


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


# =========================
# App / UI
# =========================
PAGE_TITLE = "Asistente de Trámites"
PAGE_ICON = "🧾"
PAGE_LAYOUT = "wide"

LOGO_SIDEBAR = "assets/logo_junta_fondo_verde.jpeg"
ASSISTANT_AVATAR = "assets/logo_junta_avatar.png"
USER_AVATAR = "assets/user_avatar.png"
CSS_PATH = "styles/theme_accenture.css"
CATALOG_XLSX = "descripcion-tramites.xlsx"


# =========================
# Azure AI Project / Agents
# =========================
AIPROJECT_ENDPOINT = _env("AZURE_EXISTING_AIPROJECT_ENDPOINT", default="")

CLASIFICADOR_AGENT_NAME = _env("AZURE_CLASIFICADOR_AGENT_NAME", "CLASIFICADOR_AGENT_NAME", default="ClasificadorTramites")
ASISTENTE_AGENT_NAME = _env("AZURE_ASISTENTE_AGENT_NAME", "ASISTENTE_AGENT_NAME", default="AsistenteTramites")
DESAMBIGUADOR_AGENT_NAME = _env("AZURE_DESAMBIGUADOR_AGENT_NAME", "DESAMBIGUADOR_AGENT_NAME", default="DesambiguadorTramites")
RELEVANCIA_AGENT_NAME = _env("AZURE_RELEVANCIA_AGENT_NAME", "RELEVANCIA_AGENT_NAME", default="RelevanciaRespuestaTramites")
CLARIFICADOR_AGENT_NAME = _env("AZURE_CLARIFICADOR_AGENT_NAME", "CLARIFICADOR_AGENT_NAME", default="ClarificadorTramites")
PREGUNTAS_AGENT_NAME = _env("AZURE_PREGUNTAS_AGENT_NAME", "PREGUNTAS_AGENT_NAME", default="PreguntasTramites")


# =========================
# Embeddings + Search
# =========================
AZURE_OPENAI_API_KEY = _env(
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
    default="6TZDD5vTve7kB5DVixnKnoGUtYUahlVxuEzuuU3KWXu9lTagcLWYJQQJ99CBACHYHv6XJ3w3AAABACOGj8AC",
)
AZURE_OPENAI_BASE_URL = _env(
    "AZURE_OPENAI_BASE_URL",
    default="https://azure-openai-tramites-legales.openai.azure.com/openai/v1/",
)

EMB_MODEL = _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "EMB_MODEL", default="text-embedding-3-small")

SEARCH_ENDPOINT = _env("AZURE_SEARCH_ENDPOINT", default="")
SEARCH_QUERY_KEY = _env("AZURE_SEARCH_QUERY_KEY", default="")
INDEX_NAME = _env("AZURE_INDEX_NAME", default="tramites-index")
SEARCH_VECTOR_FIELD = _env("AZURE_SEARCH_VECTOR_FIELD", default="embedding")
SEARCH_TRAMITE_FIELD = _env("AZURE_SEARCH_TRAMITE_FIELD", default="tramite")
SEARCH_DESCRIPCION_FIELD = _env("AZURE_SEARCH_DESCRIPCION_FIELD", default="descripcion")


# =========================
# Umbrales
# =========================
UMBRAL_RETRIEVAL = 0.14
UMBRAL_AMBIGUEDAD = 0.05
UMBRAL_SCORE_MINIMO = 0.35
MAX_INTENTOS_DESAMBIGUACION = 2
MAX_REINTENTOS_RELEVANCIA = 2


# =========================
# Mensajes
# =========================
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

INITIAL_ASSISTANT_GREETING = (
    "Hola 👋 Cuéntame tu caso y te digo el trámite del catálogo. "
    "Si me falta info, te haré preguntas para afinar."
)


# =========================
# Seguridad / fuera de ámbito
# =========================
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
    "api key", "clave", "token", "secreto", "secret",
]

LEGAL_HINTS = [
    "juicio", "juzgado", "demanda", "abogado", "herencia", "deuda", "contrato",
    "procedimiento", "tramite", "trámite", "notario", "ejecución", "monitorio",
    "conciliación", "perito", "embargo", "reclamar", "reclamación", "desahucio",
    "divorcio", "custodia", "incapacidad", "curatela", "tutela", "defensor judicial",
]

NONLEGAL_HINTS = [
    "receta", "tortilla", "cocinar", "bacon", "queso", "comida", "gym", "entrenar",
    "película", "serie", "música", "canción", "viaje", "hotel", "restaurante",
]
