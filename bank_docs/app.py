import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                text = f.read()

                # ✅ Split into smaller chunks (IMPORTANT FIX)
                for chunk in text.split("\n\n"):
                    if chunk.strip():
                        docs.append(chunk.strip())

    return docs


def build_index(folder_path):
    docs = load_documents(folder_path)

    if len(docs) == 0:
        st.error("No documents found")
        st.stop()

    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs


def search(query, index, docs, k=2):  # ✅ reduced results
    query_vector = model.encode([query])
    _, indices = index.search(np.array(query_vector), k)
    return [docs[i] for i in indices[0]]


st.title("🏦 Bank Bot")

folder_path = "bank_docs"

index, docs = build_index(folder_path)

query = st.text_input("Ask about bank")

if query:
    results = search(query, index, docs)

    st.write("🤖 Bot:")

    # ✅ Show only best answer instead of all
    if results:
        st.success(results[0])   # best match only
