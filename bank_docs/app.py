
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
        return "⚠️ Banks usually require minimum age of 21."

    if age > 60:
        return "❌ Not eligible due to age criteria."

    if existing_emi > max_emi:
        return "❌ Not eligible due to high existing EMI."

    eligible_loan = (max_emi - existing_emi) * 60
    return f"✅ Eligible loan amount: ₹{int(eligible_loan)}"

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

if "atm_mode" not in st.session_state:
    st.session_state.atm_mode = False

# ✅ FIX STATE
if "show_atm_calc" not in st.session_state:
    st.session_state.show_atm_calc = False

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
        st.session_state.atm_mode = True

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
        response = "🏦 We offer Home, Personal & Car Loans."
        st.session_state.show_loan_button = True

    elif "fd" in query_lower:
        response = "📈 FD Rates:\n• 1yr: 6%\n• 3yr: 7%\n• 5yr: 7.5%"

    elif "atm" in query_lower:
        response = "🏧 Showing ATM services below 👇"
        st.session_state.atm_mode = True

    elif "credit" in query_lower:
        response = "💳 Cashback & Travel Cards available."

    elif "balance" in query_lower:
        response = "💵 Savings: ₹1000 | Current: ₹5000"

    elif "account" in query_lower:
        response = "📊 Savings & Current Accounts available."

    else:
        result = search(query, index, docs, metadata)

        if result:
            response = result["text"]
        else:
            response = "🤖 Ask me about loans, EMI, ATM, FD, cards."

    st.markdown(f"**🤖 Bot:** {response}")
    st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------
# LOAN BUTTON
# -------------------------------
if st.session_state.show_loan_button:
    if st.button("Check Loan Eligibility"):
        st.session_state.loan_mode = True
        st.session_state.show_loan_button = False
        st.rerun()

# -------------------------------
# LOAN FORM
# -------------------------------
if st.session_state.loan_mode:
    st.markdown("### 🏦 Loan Eligibility Form")

    salary = st.number_input("💰 Salary", min_value=0)
    age = st.number_input("🎂 Age", min_value=18, max_value=100)
    emi_existing = st.number_input("💳 Existing EMI", min_value=0)

    if st.button("Check Eligibility"):
        result = check_loan_eligibility(salary, age, emi_existing)
        st.success(result)
        st.session_state.loan_mode = False

# -------------------------------
# ✅ ATM FEATURE (FINAL FIXED)
# -------------------------------
if st.session_state.atm_mode:
    st.markdown("### 🏧 ATM Services")

    st.info("📍 ATM Info")
    st.write("• 5 free transactions/month")
    st.write("• ₹20 per extra transaction")

    if st.button("💰 Check Charges", key="atm_btn"):
        st.session_state.show_atm_calc = True

    if st.session_state.show_atm_calc:
        transactions = st.number_input(
            "Enter number of transactions:",
            min_value=0,
            step=1,
            key="atm_input"
        )

        free_limit = 5
        charge = 20

        if transactions <= free_limit:
            st.success("✅ No charges")
        else:
            extra = transactions - free_limit
            total = extra * charge

            st.warning(f"⚠️ Charges: ₹{total}")
            st.write(f"Extra Transactions: {extra}")
```
