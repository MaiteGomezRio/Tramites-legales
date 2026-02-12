from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CATALOG_PATH, 
    CLIENT, 
    EMB_MODEL, 
    UMBRAL_RETRIEVAL,
    SEARCH_VECTOR_FIELD,
    SEARCH_TRAMITE_FIELD,
    SEARCH_DESCRIPCION_FIELD
)

from azure.search.documents.models import VectorizedQuery
from agents.client import get_search_client


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


def cosine_sim_matrix(query_vec, mat):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q


def retrieve_top_k(message, k=8, gap_max=UMBRAL_RETRIEVAL, token_trace=None):
    q_resp = CLIENT.embeddings.create(model=EMB_MODEL, input=message)
    if token_trace is not None:
        token_trace.append(_normalize_usage(q_resp))
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
        rows.append({
            "Trámite": r.get(SEARCH_TRAMITE_FIELD),
            "Descripción": r.get(SEARCH_DESCRIPCION_FIELD),
            "score": score,
        })

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


def embed_texts(texts, client, model=EMB_MODEL, batch_size=100):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)


@st.cache_resource(show_spinner="Cargando catálogo de trámites…")
def load_catalog_and_embeddings():
    catalog_df = pd.read_excel(CATALOG_PATH)
    catalog_df = catalog_df.dropna(subset=["Trámite"]).reset_index(drop=True)

    catalog_df["embed_text"] = catalog_df.apply(
        lambda r: f"{r['Trámite']}\n{r['Descripción']}".strip(),
        axis=1,
    )

    embeddings = embed_texts(catalog_df["embed_text"].tolist(), CLIENT)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-12)

    sim_matrix = emb_norm @ emb_norm.T

    pairs = []
    n = len(catalog_df)
    for i, j in combinations(range(n), 2):
        pairs.append(
            {
                "Trámite_A": catalog_df.iloc[i]["Trámite"],
                "Trámite_B": catalog_df.iloc[j]["Trámite"],
                "similitud": sim_matrix[i, j],
            }
        )

    pairs_df = pd.DataFrame(pairs).sort_values("similitud", ascending=False)

    print("=== TOP 5 PARES MÁS SIMILARES ===\n")
    for _, row in pairs_df.head(5).iterrows():
        print(f"  {row['similitud']:.4f}  |  {row['Trámite_A']}")
        print(f"           |  {row['Trámite_B']}")
        print()

    return catalog_df, emb_norm


def render_debug_info(hits, umbral_ambiguedad):

    if hits is None or len(hits) == 0:
        st.write("Sin candidatos.")
        return

    # Ordenar por score descendente
    hits = hits.sort_values("score", ascending=False).reset_index(drop=True)

    st.markdown("### 🔎 Candidatos evaluados")

    html_parts = []

    for idx, row in hits.iterrows():
        score = float(row["score"])

        gap = hits.iloc[0]["score"] - row["score"]
        marker = "⚠️ ambiguo" if gap < umbral_ambiguedad else ""
        name = f"- {row['score']:.4f} · {row['Trámite']} {marker}"

        # Color
        color = "#388443" 

        width_pct = max(min(score, 1.0), 0.0) * 100  # escala 0-1

        html_parts.append(f"""
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                <span style="color:{color}; font-weight:600;">
                    {idx+1}. {name}
                </span>
            </div>
            <div style="background:#1f2937; height:8px; border-radius:4px;">
                <div style="
                    width:{width_pct:.2f}%;
                    background:{color};
                    height:100%;
                    border-radius:4px;
                    transition:width 0.4s ease;
                "></div>
            </div>
        </div>
        """)

    st.markdown("".join(html_parts), unsafe_allow_html=True)
