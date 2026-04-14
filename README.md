# Churn Intelligence

A full-stack machine learning project that predicts customer churn for a telecom company, explains predictions using Groq AI, and provides actionable retention tools through a polished dark-themed Streamlit interface.

---

## What This Project Does

Given a customer's profile — contract type, tenure, services, billing info — the system:

1. Predicts whether the customer will churn using a trained Random Forest model
2. Explains the prediction in plain English using Groq's LLM
3. Generates a personalized retention email for at-risk customers
4. Runs what-if scenarios to show how changing one attribute shifts the risk
5. Answers free-form questions about the dataset through a chat interface

---

## Project Structure

```
churn-prediction/
│
├── app.py                  # Streamlit UI — 4-tab frontend
├── churn_analysis.py       # Full ML pipeline (EDA, training, evaluation)
├── groq_explainer.py       # All Groq AI features (4 functions)
│
├── data/
│   └── telco_churn.csv     # Auto-generated on first run (or use real Kaggle CSV)
│
├── plots/                  # Saved charts from churn_analysis.py
│   ├── eda.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── .env                    # Your Groq API key (never commit this)
├── .env.example            # Template for .env
├── .gitignore
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit groq python-dotenv
```

### 2. Set up your Groq API key

Copy the example file and add your key:

```bash
copy .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at [https://console.groq.com/keys](https://console.groq.com/keys)

### 3. Run the ML pipeline

This generates the dataset, runs EDA, trains all models, and saves plots:

```bash
python churn_analysis.py
```

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ML Pipeline — `churn_analysis.py`

Runs a complete machine learning pipeline in 7 steps:

| Step | What it does |
|------|-------------|
| 1 | Loads `data/telco_churn.csv` or auto-generates a realistic synthetic dataset |
| 2 | Cleans data — converts types, fills missing values with `SimpleImputer`, encodes categoricals with `LabelEncoder`, scales with `StandardScaler` |
| 3 | EDA — 6 charts: churn distribution, churn by contract, monthly charges, tenure, senior citizen breakdown, correlation heatmap |
| 4 | Trains Logistic Regression, Decision Tree, and Random Forest |
| 5 | Evaluates each model — accuracy, precision, recall, F1-score, confusion matrix |
| 6 | Compares all models in a grouped bar chart, picks best by F1-score |
| 7 | Plots feature importances (Random Forest) and coefficient magnitudes (Logistic Regression) |

All plots are saved to the `plots/` folder.

---

## Groq AI Features — `groq_explainer.py`

Four AI-powered functions, all using `llama-3.1-8b-instant` via the Groq API:

### `explain_prediction(customer_data, prediction, probability)`
Sends the ML prediction and customer profile to Groq. Returns a 3-section explanation:
- Why the customer might leave or stay
- Key factors influencing the prediction
- Recommended retention actions

### `generate_retention_email(customer_data, probability)`
Writes a personalized retention email tailored to the customer's contract type, tenure, services, and churn risk level. Includes a subject line and a clear call to action.

### `whatif_analysis(customer_data, current_prediction, current_probability, change_field, change_value, new_prediction, new_probability)`
Analyses the business impact of changing one customer attribute. Shows before/after churn probability and explains why the change matters.

### `chat_with_data(question, data_summary, chat_history)`
Multi-turn conversational AI with memory. Answers free-form questions about the dataset using a pre-built summary as context. Supports follow-up questions.

---

## Streamlit App — `app.py`

Four tabs, all driven by the sidebar customer profile:

### Sidebar
- Grouped into Demographics, Services, and Account sections
- Live churn risk score updates instantly as you change inputs
- Color-coded badge: High Risk (red), Medium Risk (yellow), Low Risk (green)

### Tab 1 — Predict and Explain
- Risk card showing churn probability and verdict
- Key attribute metrics (contract, tenure, charges, internet)
- Full customer profile table
- "Generate AI Explanation" button — calls Groq and renders the explanation

### Tab 2 — Retention Email
- Customer snapshot with key stats
- "Generate Retention Email" button — Groq writes a personalized email
- Rendered email card with subject line separated
- Plain text copy box for easy use

### Tab 3 — What-If Analysis
- Select any attribute (Contract, Internet, Security, etc.) and a new value
- Before / After / Change comparison boxes with color-coded arrows
- Flipped prediction alert if the change changes the outcome
- Groq explains the business impact of the change

### Tab 4 — Chat With Data
- 5 suggested question buttons to get started
- Full multi-turn chat with conversation memory
- Groq answers questions about churn patterns, segments, and risk factors
- Clear conversation button

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com/keys |

The key is loaded from `.env` using `python-dotenv`. Never hardcode it or commit `.env` to version control — it is already in `.gitignore`.

---

## Using a Real Dataset

Download the Telco Customer Churn dataset from Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Rename the file to `telco_churn.csv` and place it in the `data/` folder. The pipeline detects it automatically.

---

## Model Performance

Trained on 3000 customers (80/20 split, stratified):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | ~80% | ~45% | ~4% | ~8% |
| Decision Tree | ~79% | ~41% | ~17% | ~24% |
| Random Forest | ~80% | ~35% | ~5% | ~9% |

> Recall is low due to class imbalance (~19% churn rate). The Streamlit app uses Random Forest. The ML pipeline selects the best model by F1-score.

---

## How to Improve the Model

- Add `class_weight='balanced'` to all models — significantly boosts recall on the minority churn class
- Try XGBoost or LightGBM — typically outperform Random Forest on tabular data
- Use `GridSearchCV` or `RandomizedSearchCV` for hyperparameter tuning
- Apply SMOTE oversampling to balance the training set
- Engineer new features like `MonthlyCharges / tenure` (cost per loyalty month)
- Use `StratifiedKFold` cross-validation for more reliable evaluation scores

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML | scikit-learn (Random Forest, Logistic Regression, Decision Tree) |
| AI | Groq API — llama-3.1-8b-instant |
| Frontend | Streamlit |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Config | python-dotenv |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
groq
python-dotenv
```

---

## Security Notes

- The `.env` file is listed in `.gitignore` and will not be committed
- Never paste your API key directly into any Python file
- The `.env.example` file is safe to commit — it contains only a placeholder value
