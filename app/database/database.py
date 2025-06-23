import faiss
import numpy as np
import pickle
import sqlite3
import streamlit as st


@st.cache_resource(show_spinner="Loading FAISS index and embeddings...")
def load_index_and_ids(db_path="labeled_news.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM news")
    ids, embeddings = [], []
    for row in c.fetchall():
        ids.append(row[0])
        embeddings.append(pickle.loads(row[1]))
    conn.close()
    embeddings = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    index_to_dbid = dict(enumerate(ids))
    return faiss_index, index_to_dbid