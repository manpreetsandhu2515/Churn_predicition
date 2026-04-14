# =============================================================================
# groq_explainer.py — All Groq AI Features for Churn Prediction
# =============================================================================
# Features:
#   1. explain_prediction()     — Why will this customer churn?
#   2. generate_retention_email() — Write a personalized retention email
#   3. whatif_analysis()        — What changes if customer upgrades X?
#   4. chat_with_data()         — Ask free-form questions about customer data
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.1-8b-instant"


def _get_groq_client():
    """Returns an authenticated Groq client using the key from .env"""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY missing. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def _call_groq(system_msg: str, user_msg: str, max_tokens: int = 600) -> str:
    """
    Central helper that calls the Groq API and returns the response text.
    All 4 features use this — keeps error handling in one place.
    """
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except ValueError as e:
        return f"⚠️ Configuration error: {e}"
    except Exception as e:
        return f"⚠️ Groq API error ({type(e).__name__}): {e}"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: Prediction Explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(customer_data: dict, prediction: int, probability: float) -> str:
    """
    Explains WHY the ML model predicted churn or no-churn for this customer.

    Returns 3 sections:
      - Why they might leave/stay
      - Key factors
      - Recommended retention actions
    """
    churn_label = "WILL CHURN" if prediction == 1 else "WILL NOT CHURN"
    customer_lines = "\n".join(f"  - {k}: {v}" for k, v in customer_data.items())

    system = (
        "You are a customer retention analyst at a telecom company. "
        "Explain ML churn predictions in plain English for business users."
    )

    user = f"""A machine learning model predicted:

PREDICTION: {churn_label}
CHURN PROBABILITY: {probability:.1%}

CUSTOMER PROFILE:
{customer_lines}

Provide exactly 3 sections:

**Why this customer might {'leave' if prediction == 1 else 'stay'}:**
(2-3 bullet points interpreting the key signals)

**Key factors influencing this prediction:**
(2-3 bullet points on the most important features)

**Recommended retention actions:**
(2-3 specific, actionable business suggestions)

Be concise and professional. Interpret the data — don't just repeat it."""

    return _call_groq(system, user, max_tokens=600)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: Retention Email Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_retention_email(customer_data: dict, probability: float) -> str:
    """
    Generates a personalized retention email for a customer at risk of churning.

    The email is tailored to the customer's specific profile — contract type,
    services used, tenure, charges — and offers relevant incentives.
    """
    customer_lines = "\n".join(f"  - {k}: {v}" for k, v in customer_data.items())

    system = (
        "You are a customer success manager at a telecom company. "
        "Write warm, professional, personalized retention emails."
    )

    user = f"""Write a personalized retention email for this at-risk customer.

CHURN PROBABILITY: {probability:.1%}

CUSTOMER PROFILE:
{customer_lines}

Requirements:
- Subject line that feels personal, not generic
- Warm opening that acknowledges their loyalty (based on tenure)
- 2-3 specific offers tailored to their profile (e.g. if month-to-month → offer contract discount)
- Clear call to action
- Professional sign-off from "The Customer Success Team"

Format:
Subject: [subject line]

[email body]

Keep it under 200 words. Sound human, not like a marketing template."""

    return _call_groq(system, user, max_tokens=500)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: What-If Analysis
# ─────────────────────────────────────────────────────────────────────────────

def whatif_analysis(
    customer_data: dict,
    current_prediction: int,
    current_probability: float,
    change_field: str,
    change_value: str,
    new_prediction: int,
    new_probability: float,
) -> str:
    """
    Analyses the impact of changing one customer attribute on churn risk.

    For example: "What if this customer upgrades from Month-to-month to One year contract?"
    Shows the before/after and explains why the change matters.

    Args:
        customer_data       : original customer profile dict
        current_prediction  : original ML prediction (0 or 1)
        current_probability : original churn probability
        change_field        : the field being changed (e.g. "Contract")
        change_value        : the new value (e.g. "One year")
        new_prediction      : ML prediction after the change
        new_probability     : churn probability after the change
    """
    customer_lines = "\n".join(f"  - {k}: {v}" for k, v in customer_data.items())
    direction = "decreased" if new_probability < current_probability else "increased"
    delta = abs(new_probability - current_probability)

    system = (
        "You are a data analyst explaining the business impact of customer profile changes "
        "on churn risk predictions."
    )

    user = f"""Analyse the impact of this customer profile change on churn risk.

ORIGINAL PROFILE:
{customer_lines}

CHANGE MADE: {change_field} changed from "{customer_data.get(change_field, 'unknown')}" → "{change_value}"

BEFORE: {current_probability:.1%} churn probability ({'likely to churn' if current_prediction == 1 else 'likely to stay'})
AFTER:  {new_probability:.1%} churn probability ({'likely to churn' if new_prediction == 1 else 'likely to stay'})
IMPACT: Churn risk {direction} by {delta:.1%}

Provide:

**What changed and why it matters:**
(Explain the business significance of this specific change)

**Impact on churn risk:**
(Explain why this change {direction} the churn probability)

**Business recommendation:**
(Should the company incentivise this change? How?)

Be specific and concise."""

    return _call_groq(system, user, max_tokens=500)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: Chat With Your Data
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_data(question: str, data_summary: str, chat_history: list) -> str:
    """
    Answers free-form questions about the customer dataset using Groq.

    Maintains conversation history so follow-up questions work naturally.

    Args:
        question     : the user's question (e.g. "Which contract type churns most?")
        data_summary : a text summary of the dataset stats passed as context
        chat_history : list of {"role": "user"/"assistant", "content": "..."} dicts
                       for multi-turn conversation memory

    Returns:
        The LLM's answer as a string.
    """
    try:
        client = _get_groq_client()

        system = f"""You are a data analyst assistant for a telecom customer churn project.
You have access to the following dataset summary:

{data_summary}

Answer questions about this data clearly and concisely.
Use bullet points where helpful. If asked for numbers, be specific.
If you don't know something from the data provided, say so honestly."""

        # Build messages: system + full chat history + new question
        messages = [{"role": "system", "content": system}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    except ValueError as e:
        return f"⚠️ Configuration error: {e}"
    except Exception as e:
        return f"⚠️ Groq API error ({type(e).__name__}): {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Quick test — python groq_explainer.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "No",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "TechSupport": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.40, "TotalCharges": 286.20,
    }

    print("=" * 60)
    print("FEATURE 1: Prediction Explanation")
    print("=" * 60)
    print(explain_prediction(sample, 1, 0.82))

    print("\n" + "=" * 60)
    print("FEATURE 2: Retention Email")
    print("=" * 60)
    print(generate_retention_email(sample, 0.82))

    print("\n" + "=" * 60)
    print("FEATURE 3: What-If Analysis")
    print("=" * 60)
    print(whatif_analysis(sample, 1, 0.82, "Contract", "One year", 0, 0.21))

    print("\n" + "=" * 60)
    print("FEATURE 4: Chat With Data")
    print("=" * 60)
    summary = "Dataset: 3000 customers, 576 churners (19.2%). Month-to-month contracts have 35% churn rate. Fiber optic users churn more. Average tenure: 32 months."
    print(chat_with_data("Which customers are most at risk?", summary, []))
