# -------------------------------
# Chat input
# -------------------------------
query = st.chat_input("Ask about bank")

if query:
    st.session_state.messages.append({"role": "user", "content": query"})

    with st.chat_message("user"):
        st.write(query)

    query_lower = query.lower()

    # -------------------------------
    # EMI detection (show only if user asks)
    # -------------------------------
    if "emi" in query_lower:
        # Option 1: automatic calculation from numbers in query
        import re
        numbers = re.findall(r"\d+", query)

        if len(numbers) >= 3:
            P = int(numbers[0])
            rate = float(numbers[1])
            months = int(numbers[2])

            emi = calculate_emi(P, rate, months)
            response = f"💰 Your EMI is ₹{emi}"

        else:
            # Option 2: show input fields if user didn't provide numbers
            st.subheader("💰 EMI Calculator")
            col1, col2, col3 = st.columns(3)

            with col1:
                amount = st.number_input("Loan Amount", value=500000, key="emi_amount")

            with col2:
                rate = st.number_input("Interest Rate (%)", value=8.0, key="emi_rate")

            with col3:
                months = st.number_input("Tenure (Months)", value=60, key="emi_months")

            if st.button("Calculate EMI", key="emi_calc_btn"):
                emi = calculate_emi(amount, rate, months)
                response = f"💰 Your EMI is ₹{emi}"
            else:
                response = "Please enter loan details to calculate EMI."

    else:
        # Normal bank search
        result = search(query, index, docs)
        if result:
            if "home" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Home Loan" in line), result)

            elif "personal" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Personal Loan" in line), result)

            else:
                response = result
        else:
            response = "Please ask a more specific question."

    # Show bot message
    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
