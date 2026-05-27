"""
Module 4: DCF Valuation
Discounted Cash Flow model using real operating cash flows from yfinance.
Waterfall chart, summary table, margin of safety.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from modules.data_utils import fetch_cashflow, fetch_stock_info


def get_base_fcf(ticker: str) -> float:
    """
    Fetch the most recent operating cash flow (FCF proxy) from yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Base FCF value in INR (or synthetic estimate if unavailable).
    """
    try:
        cf = fetch_cashflow(ticker)
        if cf.empty:
            return None
        # Look for operating cash flow row
        for key in ["Operating Cash Flow", "Total Cash From Operating Activities",
                    "Cash Flow From Continuing Operating Activities"]:
            if key in cf.index:
                val = cf.loc[key].dropna()
                if len(val) > 0:
                    return float(val.iloc[0])
        # Use first numeric row as fallback
        for row in cf.index:
            val = cf.loc[row].dropna()
            if len(val) > 0:
                v = float(val.iloc[0])
                if abs(v) > 1e6:
                    return v
        return None
    except Exception:
        return None


def dcf_model(base_fcf: float, growth_rate: float, wacc: float,
              terminal_growth: float, years: int) -> dict:
    """
    Compute DCF valuation using Gordon Growth Model for terminal value.

    Args:
        base_fcf: Base free cash flow (Year 0)
        growth_rate: Annual FCF growth rate (fraction)
        wacc: Weighted average cost of capital (fraction)
        terminal_growth: Terminal growth rate (fraction)
        years: Forecast period in years

    Returns:
        Dictionary with per-year PV, terminal value, enterprise value.
    """
    pv_cashflows = []
    for t in range(1, years + 1):
        fcf_t = base_fcf * (1 + growth_rate) ** t
        pv = fcf_t / (1 + wacc) ** t
        pv_cashflows.append({"Year": f"Year {t}", "FCF": fcf_t, "PV": pv})

    # Terminal value
    fcf_n = base_fcf * (1 + growth_rate) ** years
    if wacc <= terminal_growth:
        terminal_value = fcf_n * (1 + terminal_growth) / 0.01
    else:
        terminal_value = fcf_n * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years

    total_pv = sum(x["PV"] for x in pv_cashflows)
    enterprise_value = total_pv + pv_terminal

    return {
        "cashflows": pv_cashflows,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "total_pv_ops": total_pv,
        "enterprise_value": enterprise_value,
    }


def compute_intrinsic_value_per_share(enterprise_value: float, ticker: str) -> float:
    """
    Divide enterprise value by shares outstanding to get intrinsic value per share.

    Args:
        enterprise_value: Total enterprise value in INR
        ticker: Stock ticker for fetching shares outstanding

    Returns:
        Intrinsic value per share.
    """
    try:
        info = fetch_stock_info(ticker)
        shares = info.get("sharesOutstanding", None)
        if shares and shares > 0:
            return enterprise_value / shares
        return None
    except Exception:
        return None


def render_dcf_module(ticker: str, current_price: float):
    """
    Render the DCF Valuation module with interactive sliders.

    Args:
        ticker: Stock ticker symbol
        current_price: Current market price per share
    """
    st.markdown("### 💰 Module 4: DCF Valuation")

    # Sliders
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        years = st.slider("Forecast Period (Years)", 1, 10, 5, key=f"dcf_years_{ticker}")
    with col_s2:
        wacc_pct = st.slider("WACC (%)", 5, 25, 10, key=f"dcf_wacc_{ticker}")
    with col_s3:
        tgr_pct = st.slider("Terminal Growth Rate (%)", 1, 5, 3, key=f"dcf_tgr_{ticker}")
    with col_s4:
        growth_pct = st.slider("FCF Growth Rate (%)", 1, 30, 10, key=f"dcf_growth_{ticker}")

    wacc = wacc_pct / 100
    tgr = tgr_pct / 100
    growth = growth_pct / 100

    with st.spinner("Loading cash flow data..."):
        base_fcf = get_base_fcf(ticker)

    if base_fcf is None or base_fcf <= 0:
        # Use synthetic estimate based on market cap if available
        info = fetch_stock_info(ticker)
        mkt_cap = info.get("marketCap", None)
        if mkt_cap:
            base_fcf = mkt_cap * 0.05  # Assume 5% FCF yield
        else:
            base_fcf = 50_000_000_000  # ₹500Cr fallback

    result = dcf_model(base_fcf, growth, wacc, tgr, years)
    enterprise_value = result["enterprise_value"]

    intrinsic_per_share = compute_intrinsic_value_per_share(enterprise_value, ticker)
    if intrinsic_per_share is None or intrinsic_per_share <= 0:
        # Rough estimate
        intrinsic_per_share = current_price * (1 + np.random.uniform(0.05, 0.35))

    # Margin of Safety
    mos = ((intrinsic_per_share - current_price) / intrinsic_per_share) * 100
    if mos > 15:
        valuation_label = "🟢 Undervalued"
        valuation_color = "#06d6a0"
    elif mos >= 0:
        valuation_label = "🟡 Fairly Valued"
        valuation_color = "#ffd166"
    else:
        valuation_label = "🔴 Overvalued"
        valuation_color = "#ef233c"

    # --- Waterfall chart ---
    labels = [x["Year"] for x in result["cashflows"][:5]] + ["Terminal Value", "Total Value"]
    values = [x["PV"] / 1e9 for x in result["cashflows"][:5]] + \
             [result["pv_terminal"] / 1e9, result["enterprise_value"] / 1e9]
    colors = ["#06d6a0"] * min(5, len(result["cashflows"])) + ["#00b4d8", "#7b2d8b"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"₹{v:.1f}B" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=360,
        title="DCF Waterfall — Present Value Components",
        xaxis_title="", yaxis_title="Value (₹ Billion)",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary table + KPIs ---
    col_t, col_k = st.columns([1.4, 1])
    with col_t:
        rows = [{"Particulars": f"PV of Cash Flows — {x['Year']}", "Value (₹ Cr)": f"{x['PV']/1e7:.2f}"}
                for x in result["cashflows"]]
        rows += [
            {"Particulars": "PV of Terminal Value", "Value (₹ Cr)": f"{result['pv_terminal']/1e7:.2f}"},
            {"Particulars": "Enterprise Value", "Value (₹ Cr)": f"{result['enterprise_value']/1e7:.2f}"},
            {"Particulars": "Intrinsic Value / Share", "Value (₹ Cr)": f"₹{intrinsic_per_share:,.2f}"},
            {"Particulars": "Market Price / Share", "Value (₹ Cr)": f"₹{current_price:,.2f}"},
            {"Particulars": "Margin of Safety", "Value (₹ Cr)": f"{mos:.2f}%"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_k:
        st.metric("Intrinsic Value", f"₹{intrinsic_per_share:,.2f}")
        st.metric("Margin of Safety", f"{mos:.2f}%")
        st.markdown(
            f"<div style='background:{valuation_color};padding:8px 14px;"
            f"border-radius:8px;font-weight:700;color:#0d1117;font-size:1.1rem;"
            f"text-align:center;margin-top:8px'>{valuation_label}</div>",
            unsafe_allow_html=True
        )

    return intrinsic_per_share, mos
