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

                chunks = text.split("\n\n")

                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk:
                        docs.append(chunk)

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

    if result:
        # ✅ FIXED: clean output without breaking words
        if "Account Types" in result:
            st.success("Here are the account types:")
            st.write("• Savings Account")
            st.write("• Current Account")
        else:
            st.success(result)

    else:
        st.warning("Please ask a more specific question.")
