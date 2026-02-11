import pandas as pd
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from agents.client import get_embedding_client
from config import (
    CATALOG_XLSX,
    EMB_MODEL,
    INDEX_NAME,
    SEARCH_DESCRIPCION_FIELD,
    SEARCH_ENDPOINT,
    SEARCH_QUERY_KEY,
    SEARCH_TRAMITE_FIELD,
    SEARCH_VECTOR_FIELD,
    UMBRAL_RETRIEVAL,
)
from core.helpers import track_usage


@st.cache_resource(show_spinner=False)
def get_search_client() -> SearchClient:
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


@st.cache_resource(show_spinner="Cargando catálogo de trámites…")
def load_catalog() -> pd.DataFrame:
    catalog_df = pd.read_excel(CATALOG_XLSX)
    catalog_df = catalog_df.dropna(subset=["Trámite"]).reset_index(drop=True)
    return catalog_df


def retrieve_top_k(message: str, k: int = 8, gap_max: float = UMBRAL_RETRIEVAL, token_trace=None) -> pd.DataFrame:
    emb_client = get_embedding_client()

    q_resp = emb_client.embeddings.create(model=EMB_MODEL, input=message)
    track_usage(token_trace, "retrieval_query_embedding", getattr(q_resp, "usage", None))
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

        rows.append(
            {
                "Trámite": r.get(SEARCH_TRAMITE_FIELD),
                "Descripción": r.get(SEARCH_DESCRIPCION_FIELD),
                "score": score,
            }
        )

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
