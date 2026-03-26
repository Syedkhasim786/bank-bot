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
# Load Documents (Better Chunking)
# -------------------------------
def load_documents(folder_path):
    docs = []
    metadata = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                text = f.read()

                # 🔥 Better chunking
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

    if len(docs) == 0:
        st.error("No documents found")
        st.stop()

    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs, metadata

# -------------------------------
# Search Function (Top-K + Source)
# -------------------------------
def search(query, index, docs, metadata, k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    if distances[0][0] > 1.2:
        return None

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
st.set_page_config(page_title="Bank Bot", page_icon="🏦")

st.title("🏦 AI Bank Assistant")
st.markdown("💬 Ask about loans, EMI, cards, accounts, FD and more!")

folder_path = "bank_docs"
index, docs, metadata = build_index(folder_path)

# -------------------------------
# Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome! How can I assist you today?"}
    ]

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
        st.session_state.show_balance_options = True

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
# User Input
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
    # EMI Logic
    # -------------------------------
    if "emi" in query_lower:
        numbers = re.findall(r"\d+", query)

        if len(numbers) >= 3:
            try:
                P = int(numbers[0])
                rate = float(numbers[1])
                months = int(numbers[2])

                emi = calculate_emi(P, rate, months)
                response = f"💰 **Your EMI is ₹{emi}**"
            except:
                response = "❗ Please enter valid values (amount rate months)"
        else:
            response = "👉 Example: 500000 8 60"

    # -------------------------------
    # Search Logic
    # -------------------------------
    else:
        results = search(query, index, docs, metadata)

        if results:
            result = results[0]["text"]

            if "savings" in query_lower and "balance" in query_lower:
                response = "💰 **Savings Account Minimum Balance: ₹1000**"

            elif "current" in query_lower and "balance" in query_lower:
                response = "💰 **Current Account Minimum Balance: ₹5000**"

            elif "account types" in query_lower:
                response = "💳 **Savings Account, Current Account**"

            else:
                response = result
        else:
            response = "❗ Please ask a more specific banking question."

    # -------------------------------
    # Output
    # -------------------------------
    st.markdown(f"**🤖 Bot:** {response}")

    # ✅ Source Display (PRO FEATURE)
    if results:
        st.markdown("📄 **Sources:**")
        for res in results:
            st.markdown(f"- {res['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
