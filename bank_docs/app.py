import streamlit as st
import os
import faiss
import numpy as np
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
        if annual_rate == 0:
            return round(P / months, 2)

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

    if age < 18:
        return "❌ You must be at least 18 years old."

    if 18 <= age < 21:
        return "⚠️ You are legally eligible, but most banks require minimum age of 21 for loans."

    if age > 60:
        return "❌ Not eligible due to age criteria (above 60)."

    if existing_emi > max_emi:
        return "❌ Not eligible due to high existing EMI."

    eligible_loan = (max_emi - existing_emi) * 60
    return f"✅ You are eligible for loan up to ₹{int(eligible_loan)}"

# -------------------------------
# Load Docs
# -------------------------------
def load_documents(folder_path):
    docs, metadata = [], []

    if not os.path.exists(folder_path):
        return docs, metadata

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

    if len(docs) == 0:
        return None, [], []

    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, docs, metadata

# -------------------------------
# Search
# -------------------------------
def search(query, index, docs, metadata):
    if index is None or len(docs) == 0:
        return None

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

if "show_loan_button" not in st.session_state:
    st.session_state.show_loan_button = False

if "emi_mode" not in st.session_state:
    st.session_state.emi_mode = False

if "emi_result" not in st.session_state:
    st.session_state.emi_result = None

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
        st.session_state.emi_mode = True

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
# EMI CALCULATOR
# -------------------------------
if st.session_state.emi_mode:
    st.markdown("### 💰 EMI Calculator")

    P = st.number_input("💵 Loan Amount (₹)", min_value=0)
    rate = st.number_input("📊 Interest Rate (%)", min_value=0.0)
    months = st.number_input("📅 Tenure (months)", min_value=1)

    if st.button("Calculate EMI", key="emi_btn"):
        st.session_state.emi_result = calculate_emi(P, rate, months)

    if st.session_state.emi_result is not None:
        st.success(f"💰 Your EMI is ₹{st.session_state.emi_result}")

# -------------------------------
# INPUT
# -------------------------------
query_input = st.text_input("Ask your question...")
query = quick_query if quick_query else query_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"**🧑 You:** {query}")

    query_lower = query.lower()
    response = ""
    result = None

    if "loan eligibility" in query_lower:
        st.session_state.loan_mode = True
        response = "📋 Fill the form below 👇"

    elif "loan" in query_lower:
        response = "🏦 We offer Home Loan, Personal Loan, and Car Loan."
        st.session_state.show_loan_button = True

    elif "fd" in query_lower:
        response = "📈 FD Interest Rates:\n• 1 year - 6%\n• 3 years - 7%\n• 5 years - 7.5%"

    elif "atm" in query_lower:
        response = "🏧 5 free transactions per month. ₹20 extra."

    elif "credit" in query_lower:
        response = "💳 Cashback & Travel Credit Cards available."

    elif "balance" in query_lower:
        response = "💵 Savings: ₹1000 | Current: ₹5000"

    elif "account" in query_lower:
        response = "📊 Savings & Current Accounts available."

    else:
        result = search(query, index, docs, metadata)

        if result:
            response = result["text"]
        else:
            response = "🤖 I can help with loans, EMI, FD, cards, and accounts."

    st.markdown(f"**🤖 Bot:** {response}")

    st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------
# SAFE LOAN BUTTON (OUTSIDE CHAT)
# -------------------------------
if st.session_state.show_loan_button:
    st.markdown("👉 Click below to check your loan eligibility")

    if st.button("Check Loan Eligibility", key="loan_check_btn"):
        st.session_state.loan_mode = True
        st.session_state.show_loan_button = False
        st.rerun()

# -------------------------------
# LOAN FORM
# -------------------------------
if st.session_state.loan_mode:
    st.markdown("### 🏦 Loan Eligibility Form")

    salary = st.number_input("💰 Monthly Salary (₹)", min_value=0, step=1000)
    age = st.number_input("🎂 Age", min_value=18, max_value=100)
    emi_existing = st.number_input("💳 Existing EMI (₹)", min_value=0, step=500)

    if st.button("Check Eligibility", key="loan_btn"):
        result = check_loan_eligibility(salary, age, emi_existing)
        st.success(result)
        st.session_state.loan_mode = False
