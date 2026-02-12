from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import pandas as pd
import uuid
import os
import math

print("CWD:", os.getcwd())
print("Files:", os.listdir())

# ========= CONFIG =========
# Azure AI Search (vector index)
SEARCH_ENDPOINT = "https://aisearch-tramiteslegales.search.windows.net"
SEARCH_ADMIN_KEY = "gRqAnh2TdSTUz88lDlE6KMb13XnaaYh5sDmApmvuklAzSeAtM7xx"  # Azure AI Search -> Keys -> Admin key
INDEX_NAME = "tramites-index"

# Azure OpenAI (embeddings)
AZURE_OPENAI_KEY = "8Y5B2R4KWyJbuDs6cRfNfyBryGHYVcIBrdEag6aveNCBiKBmjreoJQQJ99CBACYeBjFXJ3w3AAABACOGczBJ"        # Azure OpenAI -> Keys and Endpoint -> Key1 o Key2
AZURE_OPENAI_ENDPOINT = "https://openai-tramites-legales.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2023-05-15"

# IMPORTANT: esto es el NOMBRE DEL DEPLOYMENT (según tu captura es igual)
EMB_DEPLOYMENT = "text-embedding-3-small"

# CSV local
CSV_PATH = "descripcion_tramites.csv"
# ==========================
def clean_vector(vec):
    out = []
    for x in vec:
        xf = float(x)
        out.append(xf if math.isfinite(xf) else 0.0)
    return out


def clean_text(x) -> str:
    # Evita NaN/None (que rompen JSON) y bytes nulos
    if pd.isna(x):
        return ""
    s = str(x)
    return s.replace("\x00", "").strip()


def validate_doc(doc):
    emb = doc.get("embedding")
    if not isinstance(emb, list):
        raise TypeError(f"embedding no es list: {type(emb)}")
    if len(emb) != 1536:
        raise ValueError(f"embedding longitud inválida: {len(emb)} (esperado 1536)")
    for x in emb[:50]:
        if not isinstance(x, float) or (not math.isfinite(x)):
            raise ValueError("embedding contiene valores no válidos (NaN/Inf o no-float)")

    # Asegura strings (nunca NaN/None)
    if not isinstance(doc.get("tramite"), str):
        raise TypeError(f"tramite no es str: {type(doc.get('tramite'))}")
    if not isinstance(doc.get("descripcion"), str):
        raise TypeError(f"descripcion no es str: {type(doc.get('descripcion'))}")


openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
)

print("Testing embeddings...")
test = openai_client.embeddings.create(model=EMB_DEPLOYMENT, input="hola")
print("OK:", test.data[0].embedding[:5])

df = pd.read_csv(CSV_PATH)

required_cols = {"Trámite", "Descripción"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"El CSV debe tener columnas {required_cols}. Columnas actuales: {list(df.columns)}")

# Extra: evita NaNs en descripción y trámite antes del loop
df["Trámite"] = df["Trámite"].apply(clean_text)
df["Descripción"] = df["Descripción"].apply(clean_text)
df = df[df["Trámite"] != ""].reset_index(drop=True)

docs = []
for _, row in df.iterrows():
    tramite = row["Trámite"]
    descripcion = row["Descripción"]

    text = f"{tramite}\n{descripcion}".strip()

    raw_emb = openai_client.embeddings.create(
        model=EMB_DEPLOYMENT,
        input=text
    ).data[0].embedding

    emb = clean_vector(raw_emb)

    doc = {
        "id": str(uuid.uuid4()),
        "tramite": tramite,
        "descripcion": descripcion,
        "embedding": emb
    }

    validate_doc(doc)
    docs.append(doc)

BATCH_SIZE = 10
for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    try:
        result = search_client.upload_documents(batch)
    except Exception as e:
        print(f"❌ Error subiendo batch {i//BATCH_SIZE + 1} (docs {i}..{i+len(batch)-1}): {e}")
        print("IDs batch:", [d["id"] for d in batch])
        print("Ejemplo doc (1):", {k: (type(batch[0][k]).__name__) for k in ["tramite","descripcion","embedding"]})
        raise

    failed = [r for r in result if not r.succeeded]
    if failed:
        print(f"❌ Fallos en batch {i//BATCH_SIZE + 1}:", failed)
    else:
        print(f"✅ Batch {i//BATCH_SIZE + 1} subido ({len(batch)} docs)")

print(f"✅ TOTAL: {len(docs)} trámites indexados en '{INDEX_NAME}'")