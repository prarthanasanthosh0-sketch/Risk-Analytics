"""
Module 8: Portfolio Optimization
Efficient frontier via 5,000 random portfolios.
Max-Sharpe optimization, pie chart, interactive asset toggle.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize


def compute_portfolio_metrics(weights: np.ndarray, mean_returns: np.ndarray,
                               cov_matrix: np.ndarray, rf: float = 0.065) -> tuple:
    """
    Compute expected return, volatility, and Sharpe ratio for a given weight vector.

    Args:
        weights: Asset weight array (sums to 1)
        mean_returns: Annualised mean returns per asset
        cov_matrix: Annualised covariance matrix
        rf: Risk-free rate (default 6.5% for India)

    Returns:
        Tuple of (expected return, volatility, sharpe ratio) all as fractions.
    """
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe


def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        n_portfolios: int = 5000, rf: float = 0.065) -> pd.DataFrame:
    """
    Generate efficient frontier by random portfolio sampling.

    Args:
        mean_returns: Annualised mean returns per asset
        cov_matrix: Annualised covariance matrix
        n_portfolios: Number of random portfolios
        rf: Risk-free rate

    Returns:
        DataFrame with columns: Return, Volatility, Sharpe, Weights.
    """
    n_assets = len(mean_returns)
    np.random.seed(42)
    results = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = compute_portfolio_metrics(w, mean_returns, cov_matrix, rf)
        results.append({
            "Return": ret * 100,
            "Volatility": vol * 100,
            "Sharpe": sharpe,
            "Weights": w.tolist()
        })
    return pd.DataFrame(results)


def max_sharpe_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                          rf: float = 0.065) -> np.ndarray:
    """
    Find Maximum Sharpe Ratio portfolio using scipy minimization.

    Args:
        mean_returns: Annualised mean returns per asset
        cov_matrix: Annualised covariance matrix
        rf: Risk-free rate

    Returns:
        Optimal weight array.
    """
    n = len(mean_returns)

    def neg_sharpe(w):
        ret, vol, _ = compute_portfolio_metrics(w, mean_returns, cov_matrix, rf)
        return -(ret - rf) / vol if vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.6)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x if result.success else x0


def render_portfolio_module(all_data: dict, ticker_names: list, start: str, end: str):
    """
    Render the Portfolio Optimization module in Streamlit.

    Args:
        all_data: Dictionary of ticker → DataFrame
        ticker_names: List of ticker symbols
        start: Start date string
        end: End date string
    """
    st.markdown("### 📦 Module 8: Portfolio Optimization")

    # Asset toggle
    available = [t for t in ticker_names if t in all_data]
    selected = st.multiselect(
        "Toggle Assets (min 2):",
        options=available + ["BOND_PROXY", "GOLD"],
        default=available[:5],
        key=f"port_assets_{start}"
    )
    if len(selected) < 2:
        st.warning("Please select at least 2 assets.")
        return

    # Build combined returns DataFrame
    close_prices = {}
    for t in selected:
        if t == "BOND_PROXY":
            from modules.data_utils import get_bond_proxy
            df = get_bond_proxy(start, end)
            close_prices[t] = df["Close"]
        elif t == "GOLD":
            from modules.data_utils import fetch_gold
            df = fetch_gold(start, end)
            close_prices[t] = df["Close"]
        elif t in all_data:
            close_prices[t] = all_data[t]["Close"]

    prices_df = pd.DataFrame(close_prices).dropna()
    if prices_df.empty or len(prices_df) < 50:
        st.warning("Insufficient data for optimization.")
        return

    log_ret = np.log(prices_df / prices_df.shift(1)).dropna()
    mean_ret = log_ret.mean() * 252
    cov_mat = log_ret.cov() * 252

    with st.spinner("Computing efficient frontier (5,000 portfolios)..."):
        ef_df = efficient_frontier(mean_ret.values, cov_mat.values, 5000)
        opt_weights = max_sharpe_portfolio(mean_ret.values, cov_mat.values)

    opt_ret, opt_vol, opt_sharpe = compute_portfolio_metrics(
        opt_weights, mean_ret.values, cov_mat.values
    )

    # Max Sharpe row from random portfolios (for comparison)
    max_row = ef_df.loc[ef_df["Sharpe"].idxmax()]

    # --- Efficient Frontier Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ef_df["Volatility"], y=ef_df["Return"],
        mode="markers",
        marker=dict(
            color=ef_df["Sharpe"], colorscale="Viridis",
            size=3, opacity=0.6,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name="Portfolios"
    ))
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100],
        mode="markers+text",
        marker=dict(color="#ef233c", size=14, symbol="star"),
        text=["Max Sharpe"], textposition="top right",
        name="Optimal Portfolio"
    ))
    fig.update_layout(
        template="plotly_dark", height=400,
        title="Efficient Frontier (5,000 Random Portfolios)",
        xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Pie + Metrics ---
    col_pie, col_kpi = st.columns([1.2, 1])
    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=selected,
            values=opt_weights,
            hole=0.35,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            template="plotly_dark", height=320,
            title="Optimal Portfolio Allocation",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_kpi:
        st.metric("Expected Return", f"{opt_ret*100:.2f}%")
        st.metric("Portfolio Volatility", f"{opt_vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
        wdf = pd.DataFrame({
            "Asset": selected,
            "Weight (%)": [f"{w*100:.1f}%" for w in opt_weights]
        })
        st.dataframe(wdf, use_container_width=True, hide_index=True)

    return opt_ret * 100, opt_vol * 100, opt_sharpe
