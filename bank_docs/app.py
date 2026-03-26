import streamlit as st
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# EMI Function
# -------------------------------
def calculate_emi(P, annual_rate, months):
    try:
        r = annual_rate / (12 * 100)
        emi = (P * r * (1 + r)**months) / ((1 + r)**months - 1)
        return round(emi, 2)
    except:
        return "Invalid input"

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
# Load Documents
# -------------------------------
def load_documents(folder_path):
    docs = []
    metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                text = f.read()

                chunks = [text[i:i+300] for i in range(0, len(text), 300)]

                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk:
                        docs.append(chunk)
                        metadata.append({"source": file})

    return docs, metadata

# -------------------------------
# Build Index
# -------------------------------
@st.cache_resource
def build_index(folder_path):
    docs, metadata = load_documents(folder_path)

    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs, metadata

# -------------------------------
# Search
# -------------------------------
def search(query, index, docs, metadata, k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": docs[idx],
            "source": metadata[idx]["source"],
            "score": distances[0][i]
        })

    return results

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AI Bank Assistant", page_icon="🏦")

st.title("🏦 AI Bank Assistant")
st.markdown("💬 Ask about loans, EMI, FD, cards, accounts and more!")

folder_path = "bank_docs"
index, docs, metadata = build_index(folder_path)

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! How can I assist you today?"}
    ]

if "loan_step" not in st.session_state:
    st.session_state.loan_step = 0

# -------------------------------
# Chat History
# -------------------------------
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# -------------------------------
# Quick Actions
# -------------------------------
st.markdown("### ⚡ Quick Actions")

quick_query = None

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💰 EMI"):
        quick_query = "calculate emi"

with col2:
    if st.button("🏦 Loans"):
        quick_query = "loan information"

with col3:
    if st.button("💳 Cards"):
        quick_query = "credit card details"

with col4:
    if st.button("🏧 ATM"):
        quick_query = "atm charges"

col5, col6, col7 = st.columns(3)

with col5:
    if st.button("💵 Balance"):
        quick_query = "minimum balance"

with col6:
    if st.button("📈 FD"):
        quick_query = "fixed deposit interest"

with col7:
    if st.button("📊 Accounts"):
        quick_query = "account types"

# -------------------------------
# Input
# -------------------------------
query_input = st.text_input("Ask your question...")
query = quick_query if quick_query else query_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"**🧑 You:** {query}")

    query_lower = query.lower()
    response = ""
    results = None

    # -------------------------------
    # Loan Eligibility Flow
    # -------------------------------
    if st.session_state.loan_step > 0 or "loan eligibility" in query_lower:

        if st.session_state.loan_step == 0:
            st.session_state.loan_step = 1
            response = "💰 Enter your monthly salary:"

        elif st.session_state.loan_step == 1:
            try:
                st.session_state.salary = int(query)
                st.session_state.loan_step = 2
                response = "🎂 Enter your age:"
            except:
                response = "❗ Enter valid salary"

        elif st.session_state.loan_step == 2:
            try:
                st.session_state.age = int(query)
                st.session_state.loan_step = 3
                response = "💳 Enter existing EMI (0 if none):"
            except:
                response = "❗ Enter valid age"

        elif st.session_state.loan_step == 3:
            try:
                emi = int(query)
                response = check_loan_eligibility(
                    st.session_state.salary,
                    st.session_state.age,
                    emi
                )
                st.session_state.loan_step = 0
            except:
                response = "❗ Enter valid EMI"

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
        else:
            response = "👉 Example: 500000 8 60"

    # -------------------------------
    # Smart Responses (fallback)
    # -------------------------------
    elif "fd" in query_lower:
        response = "📈 FD interest ranges from 6% - 7.5% depending on tenure."

    elif "loan" in query_lower:
        response = "🏦 We offer Home Loan, Personal Loan, Car Loan."

    elif "atm" in query_lower:
        response = "🏧 5 free transactions per month, then charges apply."

    elif "credit" in query_lower:
        response = "💳 We offer cashback, rewards and travel credit cards."

    elif "balance" in query_lower:
        response = "💵 Savings: ₹1000 | Current: ₹5000"

    elif "account" in query_lower:
        response = "📊 Savings Account, Current Account"

    # -------------------------------
    # FAISS Search
    # -------------------------------
    else:
        results = search(query, index, docs, metadata)

        if results:
            response = results[0]["text"]
        else:
            response = "🤖 I can help with loans, EMI, FD, cards, accounts. Try asking those!"

    # -------------------------------
    # Output
    # -------------------------------
    st.markdown(f"**🤖 Bot:** {response}")

    if results:
        st.markdown("📄 Sources:")
        for r in results:
            st.markdown(f"- {r['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
