import streamlit as st
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load Model
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
# Load Docs
# -------------------------------
def load_documents(folder_path):
    docs, metadata = [], []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        docs.append(line)
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
def search(query, index, docs, metadata):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), 1)

    if distances[0][0] > 1.2:
        return None

    idx = indices[0][0]
    return {"text": docs[idx], "source": metadata[idx]["source"]}

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

if "loan_mode" not in st.session_state:
    st.session_state.loan_mode = False

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
    result = None

    # -------------------------------
    # NEW LOAN FORM (FIXED)
    # -------------------------------
    if "loan eligibility" in query_lower:
        st.session_state.loan_mode = True
        response = "📋 Fill the form below 👇"

    # -------------------------------
    # SHOW FORM
    # -------------------------------
    if st.session_state.loan_mode:
        st.markdown("### 🏦 Loan Eligibility Form")

        salary = st.number_input("💰 Monthly Salary (₹)", min_value=0, step=1000)
        age = st.number_input("🎂 Age", min_value=18, max_value=100)
        emi_existing = st.number_input("💳 Existing EMI (₹)", min_value=0, step=500)

        if st.button("Check Eligibility"):
            if salary <= 0:
                response = "❗ Enter valid salary"
            elif age < 21 or age > 60:
                response = "❌ Age must be between 21 and 60"
            else:
                response = check_loan_eligibility(salary, age, emi_existing)

            st.session_state.loan_mode = False

    # -------------------------------
    # EMI
    # -------------------------------
    elif "emi" in query_lower:
        numbers = re.findall(r"\d+", query)

        if len(numbers) >= 3:
            P, rate, months = int(numbers[0]), float(numbers[1]), int(numbers[2])
            response = f"💰 Your EMI is ₹{calculate_emi(P, rate, months)}"
        else:
            response = "👉 Example: 500000 8 60"

    # -------------------------------
    # Fixed Responses
    # -------------------------------
    elif "fd" in query_lower:
        response = "📈 FD Interest Rates:\n• 1 year - 6%\n• 3 years - 7%\n• 5 years - 7.5%"

    elif "loan" in query_lower:
        response = "🏦 We offer Home Loan, Personal Loan, and Car Loan."

    elif "atm" in query_lower:
        response = "🏧 5 free transactions per month. ₹20 extra."

    elif "credit" in query_lower:
        response = "💳 Cashback & Travel Credit Cards available."

    elif "balance" in query_lower:
        response = "💵 Savings: ₹1000 | Current: ₹5000"

    elif "account" in query_lower:
        response = "📊 Savings & Current Accounts available."

    # -------------------------------
    # FAISS Search
    # -------------------------------
    else:
        result = search(query, index, docs, metadata)

        if result:
            response = result["text"]
        else:
            response = "🤖 I can help with loans, EMI, FD, cards, and accounts."

    # -------------------------------
    # Output
    # -------------------------------
    st.markdown(f"**🤖 Bot:** {response}")

    if result:
        st.markdown(f"📄 Source: {result['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response})
