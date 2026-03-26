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
# Load docs (WITH SOURCE)
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
                        metadata.append({"source": file})  # ✅ source added

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
# Search (TOP-K + SOURCE)
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hey there! How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# -------------------------------
# ⚡ Quick Actions
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
        st.session_state.show_balance_options = True
        quick_query = None

with col6:
    if st.button("📈 FD"):
        quick_query = "fd interest"

with col7:
    if st.button("📊 Accounts"):
        quick_query = "account types"

# -------------------------------
# Balance Options
# -------------------------------
if "show_balance_options" not in st.session_state:
    st.session_state.show_balance_options = False

if st.session_state.show_balance_options:
    st.markdown("### 💵 Select Account Type")

    colA, colB = st.columns(2)

    if colA.button("Savings Account"):
        quick_query = "savings minimum balance"
        st.session_state.show_balance_options = False

    if colB.button("Current Account"):
        quick_query = "current minimum balance"
        st.session_state.show_balance_options = False

# -------------------------------
# User input
# -------------------------------
query_input = st.text_input("Ask about bank")
query = quick_query if quick_query else query_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"**You:** {query}")

    query_lower = query.lower()

    # -------------------------------
    # EMI logic
    # -------------------------------
    if "emi" in query_lower:
        numbers = re.findall(r"\d+", query)

        if len(numbers) >= 3:
            P = int(numbers[0])
            rate = float(numbers[1])
            months = int(numbers[2])
            response = f"💰 Your EMI is ₹{calculate_emi(P, rate, months)}"
            results = None
        else:
            st.subheader("💰 EMI Calculator")

            col1, col2, col3 = st.columns(3)

            with col1:
                amount = st.number_input("Loan Amount", value=500000)

            with col2:
                rate = st.number_input("Interest Rate (%)", value=8.0)

            with col3:
                months = st.number_input("Tenure (Months)", value=60)

            if st.button("Calculate EMI"):
                response = f"💰 Your EMI is ₹{calculate_emi(amount, rate, months)}"
            else:
                response = "Please enter loan details."
            results = None

    # -------------------------------
    # Search logic
    # -------------------------------
    else:
        results = search(query, index, docs, metadata)

        if results:
            result = results[0]["text"]

            if "savings" in query_lower and "balance" in query_lower:
                response = "💰 Savings Account Minimum Balance: ₹1000"

            elif "current" in query_lower and "balance" in query_lower:
                response = "💰 Current Account Minimum Balance: ₹5000"

            elif "account types" in query_lower:
                response = "💳 Savings Account, Current Account"

            else:
                response = result
        else:
            response = "Please ask a more specific question."

    # -------------------------------
    # Show response
    # -------------------------------
    st.markdown(f"**Bot:** {response}")

    # ✅ SOURCE DISPLAY (NEW FEATURE)
    if results:
        st.markdown("📄 **Source:**")
        for res in results:
            st.markdown(f"- {res['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
