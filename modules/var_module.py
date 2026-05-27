"""
Module 6: Value at Risk (VaR)
Three methods: Historical Simulation, Parametric Normal, Monte Carlo.
CVaR, loss distribution chart, and Kupiec backtesting.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical Simulation VaR: empirical percentile of daily returns.

    Args:
        returns: Array of daily log returns
        confidence: Confidence level (e.g., 0.95)

    Returns:
        VaR as a positive percentage (represents loss).
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Parametric Normal VaR using mean and standard deviation.

    Args:
        returns: Array of daily log returns
        confidence: Confidence level

    Returns:
        VaR as a positive percentage.
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    return -(mu + stats.norm.ppf(1 - confidence) * sigma)


def var_monte_carlo(returns: np.ndarray, confidence: float = 0.95,
                    n_paths: int = 10000) -> float:
    """
    Monte Carlo VaR from simulated 1-day P&L distribution.

    Args:
        returns: Historical log returns for parameter estimation
        confidence: Confidence level
        n_paths: Number of simulation paths

    Returns:
        VaR as a positive percentage.
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    np.random.seed(42)
    sim_returns = np.random.normal(mu, sigma, n_paths)
    return -np.percentile(sim_returns, (1 - confidence) * 100)


def cvar(returns: np.ndarray, var_95: float) -> float:
    """
    Expected Shortfall (CVaR) at 95%: mean of losses beyond VaR threshold.

    Args:
        returns: Array of daily log returns
        var_95: VaR threshold (positive value representing loss)

    Returns:
        CVaR as a positive percentage.
    """
    tail_losses = returns[returns < -var_95]
    if len(tail_losses) == 0:
        return var_95 * 1.2
    return -np.mean(tail_losses)


def kupiec_test(returns: np.ndarray, var_95: float, alpha: float = 0.05) -> dict:
    """
    Kupiec Proportion of Failures (POF) backtesting test.

    Args:
        returns: Array of daily returns (last 252 days)
        var_95: Historical VaR threshold at 95%
        alpha: Significance level for test

    Returns:
        Dictionary with exceptions, expected, p_value, verdict.
    """
    test_window = returns[-252:] if len(returns) >= 252 else returns
    T = len(test_window)
    exceptions = np.sum(test_window < -var_95)
    expected = T * (1 - 0.95)

    p_hat = exceptions / T if T > 0 else 0.05
    p_null = 0.05

    # Likelihood ratio test
    if p_hat == 0:
        lr = 2 * T * np.log(1 - p_null) - 2 * T * np.log(1 - p_null)
        p_val = 1.0
    elif p_hat == 1:
        lr = 1000.0
        p_val = 0.0
    else:
        lr = -2 * (
            exceptions * np.log(p_null / p_hat) +
            (T - exceptions) * np.log((1 - p_null) / (1 - p_hat))
        )
        p_val = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "exceptions": int(exceptions),
        "expected": round(expected, 1),
        "p_value": round(p_val, 4),
        "verdict": "✅ Valid" if p_val > alpha else "❌ Invalid",
    }


def render_var_module(log_returns: pd.Series, current_price: float, ticker: str):
    """
    Render the Value at Risk module in Streamlit.

    Args:
        log_returns: Series of daily log returns
        current_price: Current stock price
        ticker: Ticker symbol
    """
    st.markdown("### ⚠️ Module 6: Value at Risk (VaR)")

    returns = log_returns.dropna().values

    # Compute all VaR values
    var_h_95 = var_historical(returns, 0.95)
    var_h_99 = var_historical(returns, 0.99)
    var_p_95 = var_parametric(returns, 0.95)
    var_p_99 = var_parametric(returns, 0.99)
    var_mc_95 = var_monte_carlo(returns, 0.95, 10000)
    var_mc_99 = var_monte_carlo(returns, 0.99, 10000)

    cvar_95 = cvar(returns, var_h_95)

    # --- VaR Comparison Table ---
    var_df = pd.DataFrame({
        "Method": ["Historical Simulation", "Parametric Normal", "Monte Carlo"],
        "VaR 95% (%)": [f"-{var_h_95*100:.3f}%", f"-{var_p_95*100:.3f}%", f"-{var_mc_95*100:.3f}%"],
        "VaR 99% (%)": [f"-{var_h_99*100:.3f}%", f"-{var_p_99*100:.3f}%", f"-{var_mc_99*100:.3f}%"],
    })

    col_table, col_cvar = st.columns([1.5, 1])
    with col_table:
        st.markdown("**VaR Comparison Table**")
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    with col_cvar:
        st.metric("Expected Shortfall (CVaR 95%)", f"-{cvar_95*100:.3f}%")
        st.caption("CVaR is the mean loss beyond the VaR threshold — "
                   "it exceeds VaR as it captures the average severity of tail losses.")

    # --- Loss Distribution Chart (Monte Carlo) ---
    np.random.seed(42)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    sim_pnl = np.random.normal(mu, sigma, 10000) * 100  # in %

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim_pnl, nbinsx=100,
        marker_color="rgba(0,180,216,0.6)",
        name="P&L Distribution"
    ))
    # Shade tail
    tail_x = sim_pnl[sim_pnl < -var_mc_95 * 100]
    fig.add_trace(go.Histogram(
        x=tail_x, nbinsx=40,
        marker_color="rgba(239,35,60,0.7)",
        name="Tail (>VaR 95%)"
    ))
    fig.add_vline(
        x=-var_mc_95 * 100,
        line_dash="dash", line_color="#ef233c", line_width=2,
        annotation_text=f"VaR 95%: {-var_mc_95*100:.3f}%",
        annotation_position="top left",
        annotation_font_color="#ef233c"
    )
    fig.update_layout(
        template="plotly_dark", height=360,
        title=f"Loss Distribution (Monte Carlo 1-Day P&L) — {ticker}",
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        barmode="overlay", legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Kupiec Backtesting ---
    kupiec = kupiec_test(returns, var_h_95)
    st.markdown("**📊 Kupiec Backtesting (252 Trading Days)**")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Exceptions", kupiec["exceptions"])
    kc2.metric("Expected", kupiec["expected"])
    kc3.metric("p-value", kupiec["p_value"])
    kc4.metric("Model Status", kupiec["verdict"])

    return -var_h_95 * 100  # Return VaR 95% as negative % for use in signal logic
