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
# Load docs
# -------------------------------
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

# -------------------------------
# Build index
# -------------------------------
def build_index(folder_path):
    docs = load_documents(folder_path)
    if len(docs) == 0:
        st.error("No documents found")
        st.stop()
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, docs

# -------------------------------
# Search
# -------------------------------
def search(query, index, docs, k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    best_distance = distances[0][0]
    best_doc = docs[indices[0][0]]
    if best_distance > 1.2:
        return None
    return best_doc

# -------------------------------
# UI
# -------------------------------
st.title("🏦 Bank Bot")

folder_path = "bank_docs"
index, docs = build_index(folder_path)

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
# ⚡ Quick Actions (UPDATED)
# -------------------------------
st.markdown("### ⚡ Quick Actions")

quick_query = None

# Row 1
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

# Row 2
col5, col6, col7 = st.columns(3)

with col5:
    if st.button("💵 Balance"):
        quick_query = "minimum balance"

with col6:
    if st.button("📈 FD"):
        quick_query = "fd interest"

with col7:
    if st.button("📊 Accounts"):
        quick_query = "account types"

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
        else:
            st.subheader("💰 EMI Calculator")

            col1, col2, col3 = st.columns(3)

            with col1:
                amount = st.number_input("Loan Amount", value=500000, key="emi_amount")

            with col2:
                rate = st.number_input("Interest Rate (%)", value=8.0, key="emi_rate")

            with col3:
                months = st.number_input("Tenure (Months)", value=60, key="emi_months")

            if st.button("Calculate EMI", key="emi_calc_btn"):
                response = f"💰 Your EMI is ₹{calculate_emi(amount, rate, months)}"
            else:
                response = "Please enter loan details to calculate EMI."

    # -------------------------------
    # Bank search logic
    # -------------------------------
    else:
        result = search(query, index, docs)

        if result:
            if "home" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Home Loan" in line), result)

            elif "personal" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Personal Loan" in line), result)

            elif "credit" in query_lower:
                response = "\n".join([line for line in result.split("\n") if "credit" in line.lower()])

            elif "atm" in query_lower:
                response = "\n".join([line for line in result.split("\n") if "atm" in line.lower() or "transaction" in line.lower()])

            elif "minimum balance" in query_lower:
                response = "\n".join([line for line in result.split("\n") if "balance" in line.lower()])

            elif "fd" in query_lower or "fixed deposit" in query_lower:
                response = "\n".join([line for line in result.split("\n") if "year" in line.lower() or "interest" in line.lower()])

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
    st.session_state.messages.append({"role": "assistant", "content": response})
