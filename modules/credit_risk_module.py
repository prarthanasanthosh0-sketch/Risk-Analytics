"""
Module 7: Credit Risk Modeling
Logistic regression PD model trained on synthetic data.
Credit score gauge, confusion matrix, PD trend.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from modules.data_utils import fetch_stock_info


def generate_synthetic_training_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training dataset of companies with financial features.

    Features:
    - Debt-to-Equity ratio
    - Interest Coverage Ratio
    - Current Ratio
    - Return on Equity (ROE)
    - Net Profit Margin

    Args:
        n: Number of synthetic companies
        seed: Random seed

    Returns:
        DataFrame with features and binary Default label.
    """
    np.random.seed(seed)
    # Non-default companies: healthier financials
    nd_size = int(n * 0.85)
    d_size = n - nd_size

    non_default = pd.DataFrame({
        "DebtToEquity": np.random.lognormal(0.5, 0.5, nd_size),
        "InterestCoverage": np.random.lognormal(2.0, 0.7, nd_size),
        "CurrentRatio": np.random.normal(1.8, 0.5, nd_size).clip(0.5, 5),
        "ROE": np.random.normal(0.15, 0.08, nd_size),
        "NetProfitMargin": np.random.normal(0.12, 0.06, nd_size),
        "Default": 0
    })
    default = pd.DataFrame({
        "DebtToEquity": np.random.lognormal(1.5, 0.6, d_size),
        "InterestCoverage": np.random.lognormal(0.3, 0.5, d_size).clip(0.1, 3),
        "CurrentRatio": np.random.normal(0.9, 0.3, d_size).clip(0.2, 1.8),
        "ROE": np.random.normal(-0.05, 0.12, d_size),
        "NetProfitMargin": np.random.normal(-0.02, 0.08, d_size),
        "Default": 1
    })
    return pd.concat([non_default, default], ignore_index=True).sample(frac=1, random_state=seed)


def train_pd_model(df: pd.DataFrame):
    """
    Train logistic regression PD model.

    Args:
        df: Training DataFrame with features and Default column

    Returns:
        Tuple of (fitted model, scaler, metrics dict)
    """
    features = ["DebtToEquity", "InterestCoverage", "CurrentRatio", "ROE", "NetProfitMargin"]
    X = df[features].values
    y = df["Default"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, {"auc": round(auc, 4), "accuracy": round(acc * 100, 2), "cm": cm}


def predict_pd(model, scaler, ticker: str) -> float:
    """
    Predict Probability of Default for the selected stock using yfinance ratios.

    Args:
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        ticker: Stock ticker symbol

    Returns:
        PD as a percentage (0–100).
    """
    try:
        info = fetch_stock_info(ticker)
        de = info.get("debtToEquity", 50) or 50
        ic = info.get("ebitda", 1) / max(info.get("totalDebt", 1), 1) if info.get("totalDebt") else 3.0
        cr = info.get("currentRatio", 1.5) or 1.5
        roe = info.get("returnOnEquity", 0.1) or 0.1
        npm = info.get("profitMargins", 0.1) or 0.1

        # Normalise D/E: yfinance gives it as percentage sometimes
        if de > 100:
            de = de / 100

        features = np.array([[de, ic, cr, roe, npm]])
        features_s = scaler.transform(features)
        pd_prob = model.predict_proba(features_s)[0][1]
        return pd_prob * 100
    except Exception:
        return 5.0  # Default fallback


def pd_to_credit_score(pd_pct: float) -> int:
    """
    Map PD percentage to credit score on 300–850 scale.
    Lower PD = higher score.

    Args:
        pd_pct: PD as percentage

    Returns:
        Integer credit score.
    """
    # Linear mapping: 0% PD → 850, 100% PD → 300
    score = int(850 - (pd_pct / 100) * 550)
    return max(300, min(850, score))


def pd_to_risk_grade(pd_pct: float) -> str:
    """
    Map PD to credit risk grade.

    Args:
        pd_pct: PD as percentage

    Returns:
        Risk grade string.
    """
    if pd_pct < 0.5:
        return "AAA"
    elif pd_pct < 1.0:
        return "AA"
    elif pd_pct < 2.0:
        return "A"
    elif pd_pct < 5.0:
        return "BBB"
    elif pd_pct < 10.0:
        return "BB"
    elif pd_pct < 20.0:
        return "B"
    elif pd_pct < 40.0:
        return "CCC"
    else:
        return "D"


def render_credit_risk_module(ticker: str):
    """
    Render the Credit Risk Modeling module in Streamlit.

    Args:
        ticker: Stock ticker symbol
    """
    st.markdown("### 🏦 Module 7: Credit Risk Modeling")

    with st.spinner("Training PD model..."):
        df = generate_synthetic_training_data(500)
        model, scaler, metrics = train_pd_model(df)
        pd_pct = predict_pd(model, scaler, ticker)

    credit_score = pd_to_credit_score(pd_pct)
    risk_grade = pd_to_risk_grade(pd_pct)
    risk_level = "Low" if pd_pct < 5 else ("High" if pd_pct > 15 else "Medium")
    risk_color = "#06d6a0" if risk_level == "Low" else ("#ef233c" if risk_level == "High" else "#ffd166")

    col_gauge, col_metrics = st.columns([1, 1.2])

    with col_gauge:
        # Semicircular gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            number={"font": {"size": 40, "color": "white"}},
            title={"text": "Credit Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [300, 850], "tickwidth": 1},
                "bar": {"color": "#00b4d8"},
                "steps": [
                    {"range": [300, 500], "color": "#ef233c"},
                    {"range": [500, 650], "color": "#ffd166"},
                    {"range": [650, 750], "color": "#90be6d"},
                    {"range": [750, 850], "color": "#06d6a0"},
                ],
                "threshold": {"value": credit_score, "line": {"color": "white", "width": 3}}
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        kc1, kc2 = st.columns(2)
        kc1.metric("PD (%)", f"{pd_pct:.2f}%")
        kc2.metric("Risk Grade", risk_grade)
        st.markdown(
            f"<div style='background:{risk_color};padding:4px 10px;border-radius:8px;"
            f"font-weight:700;color:#0d1117;text-align:center'>Risk Level: {risk_level}</div>",
            unsafe_allow_html=True
        )

    with col_metrics:
        st.markdown("**Model Performance**")
        mc1, mc2 = st.columns(2)
        mc1.metric("AUC-ROC", metrics["auc"])
        mc2.metric("Accuracy", f"{metrics['accuracy']:.1f}%")

        # Confusion matrix
        st.markdown("**Confusion Matrix**")
        cm = metrics["cm"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Non-Default", "Actual: Default"],
            columns=["Pred: Non-Default", "Pred: Default"]
        )
        st.dataframe(cm_df, use_container_width=True)

    # --- PD Trend (15-month simulated) ---
    st.markdown("**📉 15-Month PD Trend**")
    months = list(range(-14, 1))
    np.random.seed(17)
    pd_trend = pd_pct + np.cumsum(np.random.normal(0, 0.3, 15))[::-1]
    pd_trend = np.clip(pd_trend, 0.1, 60)

    fig_trend = go.Figure(go.Scatter(
        x=months, y=pd_trend, mode="lines+markers",
        line=dict(color="#f77f00", width=2),
        marker=dict(size=5)
    ))
    fig_trend.update_layout(
        template="plotly_dark", height=220,
        title="Probability of Default Trend (Simulated 15 Months)",
        xaxis_title="Months Ago → Now", yaxis_title="PD (%)",
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    return pd_pct, credit_score, risk_grade
