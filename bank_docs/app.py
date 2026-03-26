import streamlit as st
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# EMI Function
# -------------------------------
def calculate_emi(P, annual_rate, months):
    r = annual_rate / (12 * 100)
    emi = (P * r * (1 + r)**months) / ((1 + r)**months - 1)
    return round(emi, 2)

# -------------------------------
# Loan Eligibility Function
# -------------------------------
def check_loan_eligibility(salary, age, existing_emi):
    max_emi = salary * 0.4

    if age < 21 or age > 60:
        return "❌ Not eligible due to age criteria."

    if existing_emi > max_emi:
        return "❌ Not eligible due to high existing EMI."

    eligible_loan = (max_emi - existing_emi) * 60
    return f"✅ You are eligible for loan up to ₹{int(eligible_loan)}"

# -------------------------------
# Load docs
# -------------------------------
def load_documents(folder_path):
    docs = []
    metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = text.split("\n\n")

                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk:
                        docs.append(chunk)
                        metadata.append({"source": file})

    return docs, metadata

# -------------------------------
# Build index
# -------------------------------
def build_index(folder_path):
    docs, metadata = load_documents(folder_path)

    if len(docs) == 0:
        st.error("No documents found")
        st.stop()

    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs, metadata

# -------------------------------
# Search
# -------------------------------
def search(query, index, docs, metadata, k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < 1.2:
            results.append({
                "text": docs[idx],
                "source": metadata[idx]["source"]
            })

    return results if results else None

# -------------------------------
# UI
# -------------------------------
st.title("🏦 Bank Bot")

folder_path = "bank_docs"
index, docs, metadata = build_index(folder_path)

# -------------------------------
# Session State
# -------------------------------
if "loan_step" not in st.session_state:
    st.session_state.loan_step = 0

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hey there! How can I help you today?"}
    ]

for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# -------------------------------
# Input
# -------------------------------
query_input = st.text_input("Ask about bank")
query = query_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"**You:** {query}")

    query_lower = query.lower()

    # -------------------------------
    # 🔥 FIXED LOAN FLOW (PRIORITY)
    # -------------------------------
    if st.session_state.loan_step > 0 or "loan eligibility" in query_lower:

        if st.session_state.loan_step == 0:
            st.session_state.loan_step = 1
            response = "Enter your monthly salary:"
            results = None

        elif st.session_state.loan_step == 1:
            try:
                st.session_state.salary = int(query)
                st.session_state.loan_step = 2
                response = "Enter your age:"
            except:
                response = "❗ Please enter a valid salary (number only)"
            results = None

        elif st.session_state.loan_step == 2:
            try:
                st.session_state.age = int(query)
                st.session_state.loan_step = 3
                response = "Enter your existing EMI (if none, type 0):"
            except:
                response = "❗ Please enter a valid age"
            results = None

        elif st.session_state.loan_step == 3:
            try:
                existing_emi = int(query)

                response = check_loan_eligibility(
                    st.session_state.salary,
                    st.session_state.age,
                    existing_emi
                )

                st.session_state.loan_step = 0
            except:
                response = "❗ Please enter a valid EMI"
            results = None

    # -------------------------------
    # EMI
    # -------------------------------
    elif "emi" in query_lower:
        numbers = re.findall(r"\d+", query)

        if len(numbers) >= 3:
            P = int(numbers[0])
            rate = float(numbers[1])
            months = int(numbers[2])
            response = f"💰 Your EMI is ₹{calculate_emi(P, rate, months)}"
            results = None
        else:
            response = "Please enter: amount rate months"
            results = None

    # -------------------------------
    # Search
    # -------------------------------
    else:
        results = search(query, index, docs, metadata)

        if results:
            response = results[0]["text"]
        else:
            response = "Please ask a more specific question."

    # -------------------------------
    # Output
    # -------------------------------
    st.markdown(f"**Bot:** {response}")

    if results:
        st.markdown("📄 **Source:**")
        for res in results:
            st.markdown(f"- {res['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
