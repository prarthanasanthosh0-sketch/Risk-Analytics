"""
Module 3: GARCH Volatility Modeling
Fits GARCH(1,1) on log returns, plots conditional vs rolling volatility,
detects regime, annotates highest spike.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def fit_garch(log_returns: pd.Series):
    """
    Fit GARCH(1,1) model on log return series.

    Args:
        log_returns: Series of daily log returns (as fractions)

    Returns:
        Fitted model result object, or None if unavailable.
    """
    if not ARCH_AVAILABLE:
        return None

    try:
        # Scale returns for numerical stability
        scaled = log_returns * 100

        model = arch_model(
            scaled,
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )

        result = model.fit(disp="off", show_warning=False)

        return result

    except Exception as e:
        st.warning(f"GARCH fitting error: {e}")
        return None


def get_conditional_volatility(result, log_returns: pd.Series) -> pd.Series:
    """
    Extract annualised conditional volatility from GARCH result.

    Args:
        result: Fitted arch model result
        log_returns: Original log returns (for index alignment)

    Returns:
        Series of annualised conditional volatility (%)
    """
    try:
        # Unscale volatility
        cond_vol = result.conditional_volatility / 100

        # Annualise
        cond_vol_ann = cond_vol * np.sqrt(252) * 100

        # Align index
        cond_vol_ann.index = log_returns.index[-len(cond_vol_ann):]

        return cond_vol_ann

    except Exception:
        return pd.Series(dtype=float)


def detect_regime(current_vol: float, historical_vol: pd.Series) -> tuple:
    """
    Determine volatility regime based on percentile thresholds.

    Args:
        current_vol: Current volatility value
        historical_vol: Full historical volatility series

    Returns:
        Tuple of (regime string, colour string)
    """

    hist = historical_vol.dropna()

    if len(hist) == 0:
        return "Unknown", "#999999"

    p25 = np.percentile(hist, 25)
    p75 = np.percentile(hist, 75)

    if current_vol > p75:
        return "High", "#ef233c"

    elif current_vol < p25:
        return "Low", "#06d6a0"

    else:
        return "Moderate", "#ffd166"


def render_garch_module(
    log_returns: pd.Series,
    prices: pd.Series,
    ticker: str
):
    """
    Render the GARCH Volatility Modeling module in Streamlit.

    Args:
        log_returns: Series of daily log returns
        prices: Series of closing prices
        ticker: Ticker symbol
    """

    st.markdown("### 📊 Module 3: GARCH Volatility Modeling")

    # ----------------------------
    # Rolling Volatility
    # ----------------------------
    rolling_vol = (
        log_returns.rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    # ----------------------------
    # Fit GARCH
    # ----------------------------
    with st.spinner("Fitting GARCH(1,1) model..."):

        result = fit_garch(log_returns)

    # ----------------------------
    # Conditional Volatility
    # ----------------------------
    if result is not None:

        cond_vol = get_conditional_volatility(
            result,
            log_returns
        )

    else:
        # Fallback to EWMA
        cond_vol = (
            log_returns.ewm(span=20).std()
            * np.sqrt(252)
            * 100
        )

        st.info(
            "Using EWMA volatility proxy "
            "(arch library not available)."
        )

    # Remove NaNs
    cond_vol = cond_vol.dropna()
    rolling_vol = rolling_vol.dropna()

    # ----------------------------
    # Spike Detection
    # ----------------------------
    if len(cond_vol) > 0:

        spike_idx = cond_vol.idxmax()
        spike_val = cond_vol.max()

        current_vol = cond_vol.iloc[-1]
        long_term_avg = cond_vol.mean()

        regime, regime_color = detect_regime(
            current_vol,
            cond_vol
        )

    else:
        spike_idx = None
        spike_val = 0
        current_vol = 0
        long_term_avg = 0

        regime = "Unknown"
        regime_color = "#999999"

    # ----------------------------
    # Plotly Figure
    # ----------------------------
    fig = go.Figure()

    # GARCH Volatility
    fig.add_trace(
        go.Scatter(
            x=cond_vol.index,
            y=cond_vol.values,
            name="Conditional Volatility (GARCH)",
            line=dict(
                color="#f77f00",
                width=1.8
            )
        )
    )

    # Rolling Volatility
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            name="20-Day Rolling Volatility",
            line=dict(
                color="white",
                width=1.2,
                dash="dot"
            )
        )
    )

    # ----------------------------
    # Spike Marker
    # ----------------------------
    if spike_idx is not None:

        # Vertical line ONLY
        fig.add_vline(
            x=spike_idx,
            line_dash="dash",
            line_color="#ef233c",
            line_width=2
        )

        # Separate annotation
        # (avoids pandas Timestamp bug)
        fig.add_annotation(
            x=spike_idx,
            y=1,
            yref="paper",
            text=f"Peak: {spike_idx.strftime('%d %b %Y')}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(
                color="#ef233c",
                size=12
            ),
            bgcolor="rgba(255,255,255,0.7)"
        )

    # ----------------------------
    # Layout
    # ----------------------------
    fig.update_layout(
        template="plotly_dark",
        height=380,
        title=f"GARCH(1,1) Conditional Volatility — {ticker}",
        xaxis_title="Date",
        yaxis_title="Annualised Volatility (%)",
        legend=dict(
            orientation="h",
            y=1.08
        ),
        margin=dict(
            l=40,
            r=20,
            t=60,
            b=40
        ),
    )

    # Render chart
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------
    # Summary Statistics
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Volatility",
        f"{current_vol:.2f}%"
    )

    c2.metric(
        "Long-Term Average",
        f"{long_term_avg:.2f}%"
    )

    c3.metric(
        "Last Spike Date",
        spike_idx.strftime("%d %b %Y")
        if spike_idx is not None
        else "N/A"
    )

    # Regime badge
    with c4:

        st.markdown("**Volatility Regime**")

        st.markdown(
            f"""
            <span style="
                background:{regime_color};
                padding:4px 12px;
                border-radius:12px;
                font-weight:700;
                color:#0d1117;
            ">
                {regime}
            </span>
            """,
            unsafe_allow_html=True
        )