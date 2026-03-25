    # -------------------------------
    # Normal bank search
    # -------------------------------
    else:
        result = search(query, index, docs)
        if result:
            if "home" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Home Loan" in line), result)

            elif "personal" in query_lower and "loan" in query_lower:
                response = next((line for line in result.split("\n") if "Personal Loan" in line), result)

            # ✅ NEW FEATURES ADDED
            elif "credit" in query_lower:
                response = next((line for line in result.split("\n") if "Credit Card" in line), result)

            elif "atm" in query_lower:
                response = next((line for line in result.split("\n") if "ATM" in line), result)

            elif "minimum balance" in query_lower:
                response = next((line for line in result.split("\n") if "Minimum balance" in line), result)

            elif "fd" in query_lower or "fixed deposit" in query_lower:
                response = next((line for line in result.split("\n") if "year" in line or "FD" in line), result)

            # existing
            elif "account types" in query_lower:
                response = "💳 Savings Account, Current Account"

            elif "loan" in query_lower:
                response = result

            else:
                response = result
        else:
            response = "Please ask a more specific question."
