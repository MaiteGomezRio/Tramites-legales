from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from config import CATALOG_PATH, CLIENT, EMB_MODEL, UMBRAL_RETRIEVAL


def cosine_sim_matrix(query_vec, mat):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q


def retrieve_top_k(message, df, embeddings, client, k=8, gap_max=UMBRAL_RETRIEVAL):
    q_resp = client.embeddings.create(model=EMB_MODEL, input=message)
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    sims = cosine_sim_matrix(q_vec, embeddings)
    idx_sorted = np.argsort(-sims)

    top_score = sims[idx_sorted[0]]
    mask = (top_score - sims[idx_sorted]) < gap_max
    idx = idx_sorted[mask]

    if len(idx) < 3:
        idx = idx_sorted[:3]
    elif len(idx) > k:
        idx = idx[:k]

    hits = df.iloc[idx].copy()
    hits["score"] = sims[idx]
    return hits


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
    top_score = hits.iloc[0]["score"]

    st.markdown("**Candidatos evaluados:**")
    for _, row in hits.iterrows():
        gap = top_score - row["score"]
        marker = " ⚠️ ambiguo" if gap < umbral_ambiguedad else ""
        st.markdown(f"- `{row['score']:.4f}` · **{row['Trámite']}**{marker}")
