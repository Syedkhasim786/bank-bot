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

                # ✅ Smart splitting (KEEP ONLY THIS)
                lines = text.split("\n")

                for line in lines:
                    line = line.strip()

                    if line and not line.endswith(":"):
                        docs.append(line)

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


def search(query, index, docs, k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    best_distance = distances[0][0]
    best_doc = docs[indices[0][0]]

    # ✅ Slightly flexible threshold
    if best_distance > 1.2:
        return None

    return best_doc


st.title("🏦 Bank Bot")

folder_path = "bank_docs"

index, docs = build_index(folder_path)

query = st.text_input("Ask about bank")

if query:
    result = search(query, index, docs)

    st.write("🤖 Bot:")

    # ✅ FIXED OUTPUT
    if result:
        st.success(result)
    else:
        st.warning("Please ask a more specific question.")
