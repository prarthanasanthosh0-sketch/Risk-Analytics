"""
Module 9: Stress Testing & Scenario Analysis
Module 10: Correlation Heatmap
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# MODULE 9: STRESS TESTING
# ─────────────────────────────────────────────

PREDEFINED_SCENARIOS = [
    {"name": "Market Crash -20%", "market_shock": -0.20, "rate_shock": 0.0, "oil_shock": 0.0},
    {"name": "Interest Rate Hike +2%", "market_shock": -0.05, "rate_shock": 0.02, "oil_shock": 0.0},
    {"name": "Recession Scenario", "market_shock": -0.15, "rate_shock": 0.01, "oil_shock": -0.10},
    {"name": "Oil Price Shock +25%", "market_shock": -0.03, "rate_shock": 0.0, "oil_shock": 0.25},
    {"name": "Best Case Scenario +15%", "market_shock": 0.15, "rate_shock": -0.005, "oil_shock": -0.05},
]


def compute_factor_betas(log_returns: pd.Series, market_returns: pd.Series) -> dict:
    """
    Compute factor betas via OLS regression against market returns.
    Rate and oil betas are estimated heuristically.

    Args:
        log_returns: Stock log return series
        market_returns: Market (Nifty proxy) log return series

    Returns:
        Dictionary of factor betas.
    """
    try:
        aligned = pd.concat([log_returns, market_returns], axis=1).dropna()
        aligned.columns = ["stock", "market"]
        cov = np.cov(aligned["stock"], aligned["market"])
        market_beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        # Heuristic: rate beta (negative for equity), oil beta sector-dependent
        rate_beta = -market_beta * 0.5
        oil_beta = market_beta * 0.2
        return {"market": round(market_beta, 3), "rate": round(rate_beta, 3), "oil": round(oil_beta, 3)}
    except Exception:
        return {"market": 1.0, "rate": -0.5, "oil": 0.2}


def compute_scenario_impacts(betas: dict, scenarios: list) -> pd.DataFrame:
    """
    Compute portfolio P&L impact for each stress scenario.

    Args:
        betas: Factor beta dictionary
        scenarios: List of scenario dictionaries

    Returns:
        DataFrame with scenario names and impact percentages.
    """
    rows = []
    for s in scenarios:
        impact = (
            betas["market"] * s["market_shock"] * 100 +
            betas["rate"] * s["rate_shock"] * 100 +
            betas["oil"] * s["oil_shock"] * 100
        )
        rows.append({"Scenario": s["name"], "Portfolio Impact (%)": round(impact, 2)})
    return pd.DataFrame(rows)


def render_stress_testing_module(log_returns: pd.Series, ticker: str):
    """
    Render the Stress Testing & Scenario Analysis module.

    Args:
        log_returns: Stock log return series
        ticker: Ticker symbol
    """
    st.markdown("### 🔥 Module 9: Stress Testing & Scenario Analysis")

    # Simulate market returns as proxy (Nifty-like)
    np.random.seed(11)
    market_ret = log_returns + np.random.normal(0, 0.002, len(log_returns))
    market_ret.index = log_returns.index

    betas = compute_factor_betas(log_returns, market_ret)

    # Custom scenario
    st.markdown("**Custom Scenario**")
    custom_col1, custom_col2, custom_col3, custom_col4 = st.columns(4)
    with custom_col1:
        custom_name = st.text_input("Scenario Name", "My Custom Scenario", key=f"st_name_{ticker}")
    with custom_col2:
        mkt_shock = st.slider("Market Shock (%)", -30, 30, 0, key=f"st_mkt_{ticker}") / 100
    with custom_col3:
        rate_shock = st.slider("Rate Shock (%)", -5, 5, 0, key=f"st_rate_{ticker}") / 100
    with custom_col4:
        oil_shock = st.slider("Oil Shock (%)", -30, 30, 0, key=f"st_oil_{ticker}") / 100

    all_scenarios = PREDEFINED_SCENARIOS + [{
        "name": custom_name,
        "market_shock": mkt_shock,
        "rate_shock": rate_shock,
        "oil_shock": oil_shock
    }]

    impact_df = compute_scenario_impacts(betas, all_scenarios)

    # Factor betas info
    beta_col1, beta_col2, beta_col3 = st.columns(3)
    beta_col1.metric("Market Beta", betas["market"])
    beta_col2.metric("Rate Beta", betas["rate"])
    beta_col3.metric("Oil Beta", betas["oil"])

    # --- Table ---
    def colour_impact(val):
        try:
            v = float(str(val).replace("%", ""))
            if v < 0:
                return "color:#ef233c"
            elif v > 0:
                return "color:#06d6a0"
        except Exception:
            pass
        return ""

    st.dataframe(impact_df, use_container_width=True, hide_index=True)

    # --- Horizontal Bar Chart ---
    colors = ["#ef233c" if v < 0 else "#06d6a0" for v in impact_df["Portfolio Impact (%)"]]
    fig = go.Figure(go.Bar(
        x=impact_df["Portfolio Impact (%)"],
        y=impact_df["Scenario"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in impact_df["Portfolio Impact (%)"]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=360,
        title="Scenario Impact Analysis",
        xaxis=dict(title="Portfolio Impact (%)", range=[-30, 30]),
        margin=dict(l=180, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Risk narrative ---
    worst_idx = impact_df["Portfolio Impact (%)"].idxmin()
    worst_scenario = impact_df.loc[worst_idx, "Scenario"]
    worst_val = impact_df.loc[worst_idx, "Portfolio Impact (%)"]
    dom_factor = max(betas, key=lambda k: abs(betas[k]))
    hedge_recs = {
        "market": "Consider put options or Nifty index shorts to hedge market risk.",
        "rate": "Consider duration-matching or interest rate swaps to hedge rate sensitivity.",
        "oil": "Consider commodity derivatives or sector rotation to hedge oil price exposure."
    }
    hedge = hedge_recs.get(dom_factor, "Review portfolio diversification.")

    st.info(
        f"⚠️ **Risk Narrative:** The highest-risk scenario is **{worst_scenario}** "
        f"with a projected portfolio impact of **{worst_val:.2f}%**. "
        f"The most significant factor exposure is **{dom_factor} beta = {betas[dom_factor]}**. "
        f"Recommendation: {hedge}"
    )


# ─────────────────────────────────────────────
# MODULE 10: CORRELATION HEATMAP
# ─────────────────────────────────────────────

def render_correlation_heatmap(all_data: dict, ticker_names: list, start: str, end: str):
    """
    Render the Correlation Heatmap for all portfolio assets.

    Args:
        all_data: Dictionary of ticker → DataFrame with Close prices
        ticker_names: List of selected tickers
        start: Start date string
        end: End date string
    """
    st.markdown("### 🔲 Module 10: Correlation Heatmap")

    available = [t for t in ticker_names if t in all_data]

    close_prices = {}
    for t in available:
        close_prices[t] = all_data[t]["Close"]

    # Add bond and gold
    from modules.data_utils import get_bond_proxy, fetch_gold
    bond_df = get_bond_proxy(start, end)
    gold_df = fetch_gold(start, end)
    close_prices["Bond"] = bond_df["Close"]
    close_prices["Gold"] = gold_df["Close"]

    prices_df = pd.DataFrame(close_prices).dropna()
    if prices_df.empty:
        st.warning("No data available for correlation analysis.")
        return

    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    corr = log_ret.corr()

    assets = corr.columns.tolist()
    z = corr.values

    # Build annotation text (with ⚠️ for |corr| > 0.70)
    annotations = []
    for i in range(len(assets)):
        for j in range(len(assets)):
            v = z[i, j]
            text = f"{v:.2f}"
            if i != j and abs(v) > 0.70:
                text += " ⚠️"
            annotations.append(
                dict(x=j, y=i, text=text, showarrow=False,
                     font=dict(size=10, color="white" if abs(v) < 0.3 else "black"))
            )

    fig = go.Figure(go.Heatmap(
        z=z, x=assets, y=assets,
        colorscale=[
            [0.0, "#00b4d8"],
            [0.5, "#ffffff"],
            [1.0, "#ef233c"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation"),
        showscale=True,
    ))
    fig.update_layout(
        template="plotly_dark", height=420,
        title="Asset Correlation Heatmap (Daily Log Returns)",
        margin=dict(l=80, r=20, t=60, b=80),
        annotations=annotations,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Diversification insights ---
    corr_pairs = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            corr_pairs.append((assets[i], assets[j], z[i, j]))

    corr_pairs.sort(key=lambda x: x[2])
    most_diversifying = corr_pairs[0]
    most_redundant = corr_pairs[-1]

    col_d, col_r = st.columns(2)
    col_d.success(
        f"🟢 **Most Diversifying Pair:** {most_diversifying[0]} & {most_diversifying[1]} "
        f"(correlation: {most_diversifying[2]:.3f})"
    )
    col_r.warning(
        f"🔴 **Most Redundant Pair:** {most_redundant[0]} & {most_redundant[1]} "
        f"(correlation: {most_redundant[2]:.3f})"
    )
