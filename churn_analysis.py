# =============================================================================
# Customer Churn Prediction - Full ML Pipeline
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# All files live flat inside the churn-prediction folder
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "telco_churn.csv")
PLOT_DIR  = BASE_DIR


# ─────────────────────────────────────────────
# STEP 1: Load or Generate Dataset
# ─────────────────────────────────────────────

def load_or_generate_data():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("[INFO] No CSV found — generating synthetic Telco Churn dataset...")
    np.random.seed(42)
    n = 3000

    df = pd.DataFrame({
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
        "PaymentMethod":    np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges":   np.round(np.random.uniform(18, 120, n), 2),
        "TotalCharges":     "",
    })

    df["TotalCharges"] = np.round(
        df["tenure"] * df["MonthlyCharges"] + np.random.normal(0, 50, n), 2
    ).clip(0)

    churn_prob = (
        0.05
        + 0.30 * (df["Contract"] == "Month-to-month").astype(int)
        + 0.10 * (df["MonthlyCharges"] > 70).astype(int)
        - 0.15 * (df["tenure"] > 24).astype(int)
        + 0.08 * (df["SeniorCitizen"] == 1).astype(int)
    ).clip(0, 1)

    df["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    missing_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved to {DATA_PATH}")
    return df


# ─────────────────────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n[STEP 2] Preprocessing...")
    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"  → Encoding {len(cat_cols)} categorical columns: {cat_cols}")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
    print(f"  → Imputed {X.isna().sum().sum()} missing values")

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"  → Train: {X_train_scaled.shape[0]}  Test: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, feature_names


# ─────────────────────────────────────────────
# STEP 3: EDA
# ─────────────────────────────────────────────

def run_eda(df):
    print("\n[STEP 3] Running EDA...")

    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Customer Churn — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    churn_counts = df["Churn"].value_counts()
    axes[0, 0].bar(churn_counts.index, churn_counts.values, color=["steelblue", "tomato"])
    axes[0, 0].set_title("Churn Distribution")
    axes[0, 0].set_xlabel("Churn")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(churn_counts.values):
        axes[0, 0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_churn.plot(kind="bar", ax=axes[0, 1], color=["steelblue", "tomato"], rot=15)
    axes[0, 1].set_title("Churn by Contract Type")
    axes[0, 1].legend(["No Churn", "Churn"])

    for label, color in [("No", "steelblue"), ("Yes", "tomato")]:
        axes[0, 2].hist(df[df["Churn"] == label]["MonthlyCharges"], bins=30, alpha=0.6, label=label, color=color)
    axes[0, 2].set_title("Monthly Charges by Churn")
    axes[0, 2].legend(title="Churn")

    for label, color in [("No", "steelblue"), ("Yes", "tomato")]:
        axes[1, 0].hist(df[df["Churn"] == label]["tenure"], bins=30, alpha=0.6, label=label, color=color)
    axes[1, 0].set_title("Tenure by Churn")
    axes[1, 0].legend(title="Churn")

    senior_churn = df.groupby(["SeniorCitizen", "Churn"]).size().unstack(fill_value=0)
    senior_churn.index = ["Non-Senior", "Senior"]
    senior_churn.plot(kind="bar", ax=axes[1, 1], color=["steelblue", "tomato"], rot=0)
    axes[1, 1].set_title("Senior Citizen vs Churn")
    axes[1, 1].legend(["No Churn", "Churn"])

    num_df = df.copy()
    for col in num_df.select_dtypes(include="object").columns:
        num_df[col] = LabelEncoder().fit_transform(num_df[col].astype(str))
    num_df["Churn"] = (num_df["Churn"] == "Yes").astype(int)
    sns.heatmap(num_df.corr(), ax=axes[1, 2], cmap="coolwarm", annot=False, linewidths=0.5)
    axes[1, 2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "eda.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → EDA plots saved to {out}")


# ─────────────────────────────────────────────
# STEP 4 & 5: Train & Evaluate
# class_weight='balanced' fixes the low recall on the minority churn class
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    print("\n[STEP 4 & 5] Training and evaluating models...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    }

    results = {}
    trained_models = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        trained_models[name] = model

        print(f"\n  [{name}]")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
        axes[idx].set_title(name)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  → Confusion matrices saved to {out}")
    return results, trained_models


# ─────────────────────────────────────────────
# STEP 6: Compare Models
# ─────────────────────────────────────────────

def compare_models(results):
    print("\n[STEP 6] Comparing models...")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())

    ax = results_df.plot(kind="bar", figsize=(10, 6), rot=15, colormap="Set2", edgecolor="black")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → Model comparison chart saved to {out}")

    best_model = results_df["F1"].idxmax()
    print(f"\n  Best model by F1-score: {best_model} ({results_df.loc[best_model, 'F1']:.4f})")
    return best_model


# ─────────────────────────────────────────────
# STEP 7: Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(trained_models, feature_names):
    print("\n[STEP 7] Plotting feature importance...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    rf = trained_models["Random Forest"]
    pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True).plot(
        kind="barh", ax=axes[0], color="steelblue", edgecolor="black"
    )
    axes[0].set_title("Random Forest — Feature Importances")
    axes[0].set_xlabel("Importance Score")

    lr = trained_models["Logistic Regression"]
    pd.Series(np.abs(lr.coef_[0]), index=feature_names).sort_values(ascending=True).plot(
        kind="barh", ax=axes[1], color="tomato", edgecolor="black"
    )
    axes[1].set_title("Logistic Regression — |Coefficients|")
    axes[1].set_xlabel("Absolute Coefficient")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → Feature importance plots saved to {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_or_generate_data()
    print(f"\nDataset shape: {df.shape}")
    print(df.head(3).to_string())
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")

    run_eda(df)

    X_train, X_test, y_train, y_test, feature_names, scaler, col_names = preprocess(df)
    results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
    best = compare_models(results)
    plot_feature_importance(trained_models, feature_names)

    print(f"\n[DONE] All plots saved in {PLOT_DIR}")
    print("       Run: python -m streamlit run app.py")
