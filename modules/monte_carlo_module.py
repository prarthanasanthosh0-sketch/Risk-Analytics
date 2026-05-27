"""
Module 5: Monte Carlo Simulation
GBM-based price path simulation with interactive sliders,
colour-coded paths, and probability summary.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import time


def run_gbm_simulation(
    current_price: float,
    mu: float,
    sigma: float,
    n_simulations: int = 1000,
    horizon: int = 252,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion.

    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Args:
        current_price: Starting price (S0)
        mu: Annualised mean log return
        sigma: Annualised volatility
        n_simulations: Number of simulation paths
        horizon: Time horizon in trading days
        seed: Random seed for reproducibility

    Returns:
        2D numpy array (n_simulations x horizon+1) of simulated prices.
    """
    np.random.seed(seed)
    dt = 1 / 252
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.standard_normal((n_simulations, horizon))
    log_returns = drift + diffusion * Z

    paths = np.zeros((n_simulations, horizon + 1))
    paths[:, 0] = current_price
    for t in range(1, horizon + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(log_returns[:, t - 1])

    return paths


def compute_probability_summary(paths: np.ndarray, current_price: float) -> dict:
    """
    Compute summary statistics at the end of the simulation horizon.

    Args:
        paths: 2D array of simulated price paths
        current_price: Starting/current price

    Returns:
        Dictionary of probability statistics.
    """
    end_prices = paths[:, -1]
    return {
        "Expected Price": np.mean(end_prices),
        "Median Price": np.median(end_prices),
        "Best Case (95th %ile)": np.percentile(end_prices, 95),
        "Worst Case (5th %ile)": np.percentile(end_prices, 5),
        "P(Price > 1.10x)": np.mean(end_prices > current_price * 1.10) * 100,
        "P(Price < 0.90x)": np.mean(end_prices < current_price * 0.90) * 100,
    }


def render_monte_carlo_module(prices: pd.Series, log_returns: pd.Series, ticker: str):
    """
    Render the Monte Carlo Simulation module in Streamlit.

    Args:
        prices: Historical close prices
        log_returns: Historical log returns
        ticker: Ticker symbol
    """
    st.markdown("### 🎲 Module 5: Monte Carlo Simulation")

    # Interactive sliders
    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.select_slider(
            "Number of Simulations",
            options=list(range(500, 10001, 500)),
            value=1000,
            key=f"mc_nsims_{ticker}"
        )
    with col2:
        horizon = st.select_slider(
            "Time Horizon (Trading Days)",
            options=list(range(63, 253, 21)),
            value=252,
            key=f"mc_horizon_{ticker}"
        )

    current_price = prices.iloc[-1]
    mu = log_returns.mean() * 252
    sigma = log_returns.std() * np.sqrt(252)

    start_time = time.time()
    paths = run_gbm_simulation(current_price, mu, sigma, n_sims, horizon)
    elapsed = time.time() - start_time

    end_prices = paths[:, -1]
    p5 = np.percentile(end_prices, 5)
    p95 = np.percentile(end_prices, 95)

    # --- Chart ---
    fig = go.Figure()
    x_axis = list(range(horizon + 1))

    # Plot all paths in batches for performance
    for i in range(min(n_sims, 300)):
        ep = paths[i, -1]
        if ep >= p95:
            color = "rgba(6,214,160,0.15)"
        elif ep <= p5:
            color = "rgba(239,35,60,0.15)"
        else:
            color = "rgba(0,180,216,0.06)"
        fig.add_trace(go.Scatter(
            x=x_axis, y=paths[i],
            mode="lines", line=dict(color=color, width=0.5),
            showlegend=False
        ))

    # Add percentile lines
    median_path = np.median(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=x_axis, y=median_path,
        mode="lines", line=dict(color="white", width=2),
        name="Median Path"
    ))

    fig.update_layout(
        template="plotly_dark", height=400,
        title=f"Monte Carlo GBM — {n_sims:,} Paths over {horizon} Days ({ticker})",
        xaxis_title="Trading Days", yaxis_title="Price (INR)",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"⏱ Computation time: {elapsed:.2f}s")

    # --- Probability Summary ---
    summary = compute_probability_summary(paths, current_price)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📋 Summary (1 Year)**")
        rows = [
            {"Metric": k, "Value": f"₹{v:,.2f}" if "Price" in k else f"{v:.2f}%"}
            for k, v in summary.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with col_b:
        c1, c2 = st.columns(2)
        c1.metric("Expected Price", f"₹{summary['Expected Price']:,.2f}")
        c2.metric("Median Price", f"₹{summary['Median Price']:,.2f}")
        c1.metric("Best Case (95th)", f"₹{summary['Best Case (95th %ile)']:,.2f}")
        c2.metric("Worst Case (5th)", f"₹{summary['Worst Case (5th %ile)']:,.2f}")
        c1.metric("P(>+10%)", f"{summary['P(Price > 1.10x)']:.1f}%")
        c2.metric("P(<-10%)", f"{summary['P(Price < 0.90x)']:.1f}%")

    return paths
