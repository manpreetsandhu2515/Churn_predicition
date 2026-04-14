# =============================================================================
# app.py — Customer Churn Predictor (Premium Dark UI)
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from groq_explainer import (
    explain_prediction,
    generate_retention_email,
    whatif_analysis,
    chat_with_data,
)

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global background ── */
.stApp {
    background: #0d0d14;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13131f;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin-top: 0.5rem;
}

/* ── Sidebar section label ── */
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4a5568 !important;
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e1e30;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.8rem 2rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.page-header:hover {
    border-color: #2d5a8f;
    box-shadow: 0 8px 32px rgba(15,52,96,0.4);
    transform: translateY(-1px);
}
.page-header-left h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.page-header-left p {
    color: #7f9cf5;
    margin: 0.3rem 0 0 0;
    font-size: 0.88rem;
    font-weight: 400;
}
.header-badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* ── Risk card hover ── */
.risk-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}
.risk-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.risk-card-high {
    background: linear-gradient(135deg, #2d1515 0%, #1a0a0a 100%);
    border-color: #742a2a;
}
.risk-card-medium {
    background: linear-gradient(135deg, #2d2415 0%, #1a1505 100%);
    border-color: #744210;
}
.risk-card-low {
    background: linear-gradient(135deg, #0f2d1a 0%, #071a0f 100%);
    border-color: #1a4731;
}
.risk-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.risk-label-high  { color: #fc8181; }
.risk-label-medium{ color: #f6ad55; }
.risk-label-low   { color: #68d391; }
.risk-percent {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.risk-percent-high   { color: #fc8181; }
.risk-percent-medium { color: #f6ad55; }
.risk-percent-low    { color: #68d391; }
.risk-sublabel {
    font-size: 0.82rem;
    color: #718096;
}

/* ── Stat card ── */
.stat-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}
.stat-card:hover {
    border-color: #4f46e5;
    background: #16162a;
    transform: translateX(4px);
    box-shadow: -3px 0 0 #4f46e5, 0 4px 16px rgba(99,102,241,0.12);
}
.stat-card-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4a5568;
    margin-bottom: 0.3rem;
}
.stat-card-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4a5568;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e30;
}

/* ── AI output box ── */
.ai-output {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    color: #cbd5e0;
    line-height: 1.7;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.ai-output:hover {
    border-left-color: #818cf8;
    background: #16162a;
    box-shadow: 0 4px 20px rgba(99,102,241,0.1);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #13131f;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1e1e30;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.55rem 1.4rem;
    font-size: 0.88rem;
    font-weight: 500;
    color: #718096;
    background: transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: #1a1a2e !important;
    color: #a5b4fc !important;
    font-weight: 600;
    border: 1px solid #2d2d50 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.65rem 1.5rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: 0.3px;
    position: relative;
    overflow: hidden;
}
.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #4f46e5);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4), 0 2px 8px rgba(99,102,241,0.2);
}
.stButton > button:hover::after { opacity: 1; }
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}
[data-testid="stMetric"]:hover {
    border-color: #4f46e5;
    background: #16162a;
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
}
[data-testid="stMetricLabel"] { color: #718096 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.4rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e30 !important;
    border-radius: 12px !important;
    overflow: hidden;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stDataFrame"]:hover {
    border-color: #2d2d50 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

/* ── Alerts ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
[data-baseweb="notification"] {
    background: #13131f !important;
    border-radius: 10px !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    color: #cbd5e0 !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* ── Select / input ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #68d391, #f6ad55, #fc8181) !important;
    border-radius: 4px;
}
.stProgress > div > div {
    background: #1e1e30 !important;
    border-radius: 4px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem;
}

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background: #13131f !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr { border-color: #1e1e30 !important; margin: 1.5rem 0; }

/* ── Suggestion buttons ── */
.suggest-btn > button {
    background: #13131f !important;
    border: 1px solid #2d2d44 !important;
    color: #a0aec0 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transform: none !important;
    transition: all 0.2s ease !important;
}
.suggest-btn > button:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.08) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.15) !important;
}

/* ── Compare box hover ── */
.compare-box {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}
.compare-box:hover {
    border-color: #4f46e5;
    background: #16162a;
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.15);
}
.compare-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4a5568;
    margin-bottom: 0.4rem;
}
.compare-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
}
.compare-sub {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 0.2rem;
}

/* ── Email card ── */
.email-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    line-height: 1.8;
    color: #cbd5e0;
    font-size: 0.92rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.email-card:hover {
    border-color: #2d2d50;
    background: #16162a;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.email-subject {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6366f1;
    margin-bottom: 0.4rem;
}
.email-subject-text {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1e1e30;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Info/success/error ── */
.stSuccess { background: #0f2d1a !important; border-left: 3px solid #68d391 !important; }
.stError   { background: #2d1515 !important; border-left: 3px solid #fc8181 !important; }
.stInfo    { background: #0f1a2d !important; border-left: 3px solid #63b3ed !important; }
</style>
""", unsafe_allow_html=True)

# ── Encoding mappings ─────────────────────────────────────────────────────────
MAPPINGS = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

# ── Load & train model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Use the directory where app.py lives so the path works regardless
    # of which directory Streamlit is launched from
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "telco_churn.csv")

    # If dataset is missing, auto-generate it instead of crashing
    if not os.path.exists(DATA_PATH):
        import numpy as np
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        n = 3000
        np.random.seed(42)
        df_gen = pd.DataFrame({
            "customerID":       [f"C{i:05d}" for i in range(n)],
            "gender":           np.random.choice(["Male", "Female"], n),
            "SeniorCitizen":    np.random.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner":          np.random.choice(["Yes", "No"], n),
            "Dependents":       np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure":           np.random.randint(1, 72, n),
            "PhoneService":     np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService":  np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
            "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract":         np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
            "PaymentMethod":    np.random.choice(["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"], n),
            "MonthlyCharges":   np.round(np.random.uniform(18, 120, n), 2),
        })
        df_gen["TotalCharges"] = np.round(
            df_gen["tenure"] * df_gen["MonthlyCharges"] + np.random.normal(0, 50, n), 2
        ).clip(0)
        churn_prob = (
            0.05
            + 0.30 * (df_gen["Contract"] == "Month-to-month").astype(int)
            + 0.10 * (df_gen["MonthlyCharges"] > 70).astype(int)
            - 0.15 * (df_gen["tenure"] > 24).astype(int)
            + 0.08 * (df_gen["SeniorCitizen"] == 1).astype(int)
        ).clip(0, 1)
        df_gen["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")
        df_gen.to_csv(DATA_PATH, index=False)
        st.toast("Dataset auto-generated successfully.")
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    X_train, _, y_train, _ = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mdl = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X_train_scaled, y_train)

    raw = pd.read_csv(DATA_PATH)
    churn_rate = (raw["Churn"] == "Yes").mean() if "Churn" in raw.columns else 0
    summary = f"""Dataset: {len(raw)} customers, {int(churn_rate * len(raw))} churners ({churn_rate:.1%} churn rate).
Contract churn rates: Month-to-month ~42%, One year ~11%, Two year ~3%.
Fiber optic customers churn more (~30%) vs DSL (~20%).
Senior citizens churn at ~41% vs non-seniors (~26%).
No online security or tech support correlates with higher churn.
Average monthly charges: churners ~$74 vs non-churners ~$61.
Customers with tenure under 12 months have the highest churn risk."""

    return mdl, scaler, feature_cols, summary, churn_rate, len(raw)


model, scaler, feature_cols, data_summary, overall_churn_rate, total_customers = load_model()


def encode_input(raw: dict) -> pd.DataFrame:
    encoded = {}
    for col in feature_cols:
        if col in MAPPINGS:
            encoded[col] = MAPPINGS[col].get(str(raw.get(col, "")), 0)
        else:
            encoded[col] = raw.get(col, 0)
    return pd.DataFrame([encoded])[feature_cols]


def predict(raw: dict):
    df_enc = encode_input(raw)
    df_scaled = scaler.transform(df_enc)
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]
    return int(pred), float(prob)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Churn Intelligence</div>
    <div style="font-size:0.75rem; color:#4a5568; margin-top:2px;">Customer Profile Editor</div>
</div>
""", unsafe_allow_html=True)

def sidebar_inputs():
    st.sidebar.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    gender   = st.sidebar.selectbox("Gender", ["Male", "Female"], label_visibility="visible")
    senior   = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner  = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    deps     = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    tenure   = st.sidebar.slider("Tenure (months)", 1, 72, 12)

    st.sidebar.markdown('<div class="sidebar-section">Services</div>', unsafe_allow_html=True)
    phone    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    lines    = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    support  = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    st.sidebar.markdown('<div class="sidebar-section">Account</div>', unsafe_allow_html=True)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless= st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment  = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly  = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total    = st.sidebar.number_input("Total Charges ($)", min_value=0.0,
                                       value=float(tenure * monthly))
    return {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": deps, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security,
        "TechSupport": support, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }


raw_input = sidebar_inputs()
prediction, probability = predict(raw_input)

# Live risk card in sidebar
st.sidebar.markdown('<div class="sidebar-section">Live Risk Score</div>', unsafe_allow_html=True)

if probability > 0.5:
    risk_class = "high"
    risk_text  = "High Risk"
    verdict    = "Likely to churn"
elif probability > 0.3:
    risk_class = "medium"
    risk_text  = "Medium Risk"
    verdict    = "Monitor closely"
else:
    risk_class = "low"
    risk_text  = "Low Risk"
    verdict    = "Likely to stay"

st.sidebar.markdown(f"""
<div class="risk-card risk-card-{risk_class}">
    <div class="risk-label risk-label-{risk_class}">{risk_text}</div>
    <div class="risk-percent risk-percent-{risk_class}">{probability:.1%}</div>
    <div class="risk-sublabel">{verdict}</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.progress(probability)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="page-header-left">
        <h1>Churn Intelligence</h1>
        <p>Predict, explain, and act on customer churn risk using ML and Groq AI</p>
    </div>
    <div>
        <span class="header-badge">Random Forest + Groq llama-3.1-8b-instant</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top stats bar ─────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Customers", f"{total_customers:,}")
s2.metric("Overall Churn Rate", f"{overall_churn_rate:.1%}")
s3.metric("Current Risk Score", f"{probability:.1%}")
s4.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Predict and Explain  ",
    "  Retention Email  ",
    "  What-If Analysis  ",
    "  Chat With Data  ",
])


# =============================================================================
# TAB 1 — Predict and Explain
# =============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Risk card
        st.markdown(f"""
        <div class="risk-card risk-card-{risk_class}" style="margin-bottom:1rem;">
            <div class="risk-label risk-label-{risk_class}">Churn Prediction</div>
            <div class="risk-percent risk-percent-{risk_class}">{probability:.1%}</div>
            <div class="risk-sublabel" style="margin-top:0.5rem; font-size:1rem; font-weight:600; color:#e2e8f0;">
                {"This customer is likely to churn" if prediction == 1 else "This customer is likely to stay"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key stats
        st.markdown('<div class="section-heading">Key Attributes</div>', unsafe_allow_html=True)
        ka1, ka2 = st.columns(2)
        ka1.metric("Contract", raw_input["Contract"])
        ka2.metric("Tenure", f"{raw_input['tenure']} mo")
        ka3, ka4 = st.columns(2)
        ka3.metric("Monthly Charges", f"${raw_input['MonthlyCharges']:.0f}")
        ka4.metric("Internet", raw_input["InternetService"])

    with col_right:
        st.markdown('<div class="section-heading">Full Customer Profile</div>', unsafe_allow_html=True)
        display = {}
        labels = {
            "gender": "Gender", "Partner": "Partner", "Dependents": "Dependents",
            "tenure": "Tenure (mo)", "PhoneService": "Phone", "MultipleLines": "Multi Lines",
            "InternetService": "Internet", "OnlineSecurity": "Security",
            "TechSupport": "Tech Support", "Contract": "Contract",
            "PaperlessBilling": "Paperless", "PaymentMethod": "Payment",
            "MonthlyCharges": "Monthly ($)", "TotalCharges": "Total ($)",
        }
        for k, label in labels.items():
            display[label] = raw_input.get(k, "")
        df_display = pd.DataFrame(list(display.items()), columns=["Attribute", "Value"])
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=380)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">AI Explanation</div>', unsafe_allow_html=True)
    st.caption("Groq analyses the customer profile and explains the prediction with key factors and retention actions.")

    if st.button("Generate AI Explanation", key="explain_btn", use_container_width=True):
        with st.spinner("Analysing with Groq AI..."):
            result = explain_prediction(raw_input, prediction, probability)
        st.markdown(f'<div class="ai-output">{result}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — Retention Email
# =============================================================================
with tab2:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-heading">Customer Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="risk-card risk-card-{risk_class}">
            <div class="risk-label risk-label-{risk_class}">Churn Risk</div>
            <div class="risk-percent risk-percent-{risk_class}">{probability:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Contract</div>
            <div class="stat-card-value">{raw_input["Contract"]}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Tenure</div>
            <div class="stat-card-value">{raw_input["tenure"]} months</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Monthly Charges</div>
            <div class="stat-card-value">${raw_input["MonthlyCharges"]:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Payment Method</div>
            <div class="stat-card-value" style="font-size:0.95rem;">{raw_input["PaymentMethod"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-heading">Generated Email</div>', unsafe_allow_html=True)
        st.caption("Groq writes a personalized retention email tailored to this customer's profile and risk level.")

        if st.button("Generate Retention Email", use_container_width=True):
            if probability < 0.2:
                st.info("Low churn risk — email generated anyway for proactive outreach.")
            with st.spinner("Writing personalized email via Groq..."):
                email = generate_retention_email(raw_input, probability)

            lines = email.strip().split("\n")
            subject_line = ""
            body_lines = []
            for line in lines:
                if line.lower().startswith("subject:"):
                    subject_line = line.replace("Subject:", "").replace("subject:", "").strip()
                else:
                    body_lines.append(line)

            body_text = "\n".join(body_lines).strip()

            st.markdown(f"""
            <div class="email-card">
                <div class="email-subject">Subject Line</div>
                <div class="email-subject-text">{subject_line if subject_line else "—"}</div>
                {body_text.replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.text_area("Copy plain text", value=email, height=200, label_visibility="collapsed")


# =============================================================================
# TAB 3 — What-If Analysis
# =============================================================================
with tab3:
    st.markdown('<div class="section-heading">Scenario Setup</div>', unsafe_allow_html=True)
    st.caption("Change one attribute and see how the churn probability shifts. Groq explains the business impact.")

    WHATIF_OPTIONS = {
        "Contract":         ["Month-to-month", "One year", "Two year"],
        "InternetService":  ["DSL", "Fiber optic", "No"],
        "OnlineSecurity":   ["Yes", "No", "No internet service"],
        "TechSupport":      ["Yes", "No", "No internet service"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod":    [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ],
    }

    col1, col2 = st.columns(2, gap="large")
    with col1:
        change_field = st.selectbox("Attribute to change", list(WHATIF_OPTIONS.keys()))
        current_val  = raw_input.get(change_field, "")
        available    = [v for v in WHATIF_OPTIONS[change_field] if str(v) != str(current_val)]
        change_value = st.selectbox("New value", available)

    with col2:
        st.markdown('<div class="section-heading">Current State</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Current {change_field}</div>
            <div class="stat-card-value">{current_val}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Current Churn Probability</div>
            <div class="stat-card-value">{probability:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Run What-If Analysis", use_container_width=True):
        modified = raw_input.copy()
        modified[change_field] = change_value
        new_pred, new_prob = predict(modified)
        delta = new_prob - probability

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3, gap="medium")
        with r1:
            st.markdown(f"""
            <div class="compare-box">
                <div class="compare-label">Before</div>
                <div class="compare-value" style="color:#a0aec0;">{probability:.1%}</div>
                <div class="compare-sub">{current_val}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            arrow_color = "#68d391" if delta < 0 else "#fc8181"
            arrow = "▼" if delta < 0 else "▲"
            st.markdown(f"""
            <div class="compare-box">
                <div class="compare-label">After</div>
                <div class="compare-value" style="color:{arrow_color};">{new_prob:.1%}</div>
                <div class="compare-sub">{change_value}</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="compare-box">
                <div class="compare-label">Risk Change</div>
                <div class="compare-value" style="color:{arrow_color};">{arrow} {abs(delta):.1%}</div>
                <div class="compare-sub">{"Lower risk" if delta < 0 else "Higher risk"}</div>
            </div>
            """, unsafe_allow_html=True)

        if new_pred != prediction:
            if new_pred == 0:
                st.success("Prediction flipped — this change would likely retain the customer.")
            else:
                st.error("Prediction flipped — this change would push the customer toward churning.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">AI Impact Analysis</div>', unsafe_allow_html=True)
        with st.spinner("Analysing business impact via Groq..."):
            analysis = whatif_analysis(
                raw_input, prediction, probability,
                change_field, change_value, new_pred, new_prob
            )
        st.markdown(f'<div class="ai-output">{analysis}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4 — Chat With Data
# =============================================================================
with tab4:
    st.markdown('<div class="section-heading">Ask About Your Customer Data</div>', unsafe_allow_html=True)
    st.caption("The AI has full context of your dataset — churn rates, patterns, segments, and risk factors.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggestion pills
    suggestions = [
        "Which customers are most at risk?",
        "Churn rate by contract type?",
        "How does tenure affect churn?",
        "Top 3 factors driving churn?",
        "Which payment method churns most?",
    ]
    s_cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with s_cols[i]:
            st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = s
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask anything about your customer data...")

    if "pending_question" in st.session_state:
        user_question = st.session_state.pop("pending_question")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_with_data(
                    question=user_question,
                    data_summary=data_summary,
                    chat_history=st.session_state.chat_history,
                )
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "user",      "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#4a5568; font-size:0.78rem;">
    <span>Churn Intelligence — Customer Retention Platform</span>
    <span>Random Forest  ·  Groq llama-3.1-8b-instant  ·  Streamlit</span>
</div>
""", unsafe_allow_html=True)
