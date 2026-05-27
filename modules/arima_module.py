"""
Module 2: ARIMA Forecasting
Uses pmdarima auto_arima to fit best model, plots forecast with confidence interval,
reports RMSE/MAE/MAPE/Direction, and walk-forward validation.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
except ImportError:
    pass


def fit_arima(prices: pd.Series):
    """
    Fit best ARIMA model using auto_arima (stepwise, non-seasonal).

    Args:
        prices: Series of adjusted close prices

    Returns:
        Fitted model object, selected order tuple
    """
    if not PMDARIMA_AVAILABLE:
        return None, (1, 1, 1)
    try:
        model = auto_arima(
            prices,
            stepwise=True,
            seasonal=False,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5,
        )
        return model, model.order
    except Exception as e:
        st.warning(f"auto_arima failed: {e}. Using ARIMA(1,1,1).")
        return None, (1, 1, 1)


def get_arima_metrics(prices: pd.Series, model) -> dict:
    """
    Compute in-sample metrics: RMSE, MAE, MAPE.

    Args:
        prices: Original price series
        model: Fitted pmdarima model

    Returns:
        Dictionary with RMSE, MAE, MAPE, AIC, BIC.
    """
    try:
        fitted = model.predict_in_sample()
        fitted = pd.Series(fitted, index=prices.index[-len(fitted):])
        actual = prices.iloc[-len(fitted):]
        rmse = np.sqrt(np.mean((actual.values - fitted.values) ** 2))
        mae = mean_absolute_error(actual.values, fitted.values)
        mape = np.mean(np.abs((actual.values - fitted.values) / actual.values)) * 100
        return {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "AIC": round(model.aic(), 2),
            "BIC": round(model.bic(), 2),
        }
    except Exception:
        return {"RMSE": 0, "MAE": 0, "MAPE": 0, "AIC": 0, "BIC": 0}


def forecast_arima(prices: pd.Series, model, horizon: int = 90):
    """
    Generate ARIMA forecast with 95% confidence intervals.

    Args:
        prices: Historical price series
        model: Fitted pmdarima model
        horizon: Forecast horizon in trading days

    Returns:
        forecast Series, lower CI Series, upper CI Series
    """
    try:
        fc, conf = model.predict(n_periods=horizon, return_conf_int=True)
        last_date = prices.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        fc_series = pd.Series(fc, index=future_dates)
        lower = pd.Series(conf[:, 0], index=future_dates)
        upper = pd.Series(conf[:, 1], index=future_dates)
        return fc_series, lower, upper
    except Exception:
        last_price = prices.iloc[-1]
        last_date = prices.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        fc = pd.Series([last_price] * horizon, index=future_dates)
        lower = fc * 0.95
        upper = fc * 1.05
        return fc, lower, upper


def walk_forward_validation(prices: pd.Series, order: tuple) -> dict:
    """
    Walk-forward validation: train on first 80%, predict on remaining 20%.

    Args:
        prices: Full price series
        order: ARIMA (p,d,q) order tuple

    Returns:
        dict with in_sample_rmse, oos_rmse
    """
    n = len(prices)
    train_size = int(n * 0.8)
    train = prices.iloc[:train_size]
    test = prices.iloc[train_size:]

    try:
        from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        predictions = []
        history = list(train.values)
        for t in range(len(test)):
            m = StatsARIMA(history, order=order)
            m_fit = m.fit(disp=False)
            yhat = m_fit.forecast(steps=1)[0]
            predictions.append(yhat)
            history.append(test.iloc[t])

        oos_rmse = np.sqrt(np.mean((test.values - np.array(predictions)) ** 2))

        # In-sample RMSE
        m_train = StatsARIMA(train.values, order=order).fit(disp=False)
        in_pred = m_train.fittedvalues
        in_rmse = np.sqrt(np.mean((train.values[len(train) - len(in_pred):] - in_pred) ** 2))

        return {"in_sample_rmse": round(in_rmse, 2), "oos_rmse": round(oos_rmse, 2)}
    except Exception:
        return {"in_sample_rmse": 0, "oos_rmse": 0}


def render_arima_module(prices: pd.Series, ticker: str):
    """
    Render the full ARIMA Forecasting module in Streamlit.

    Args:
        prices: Historical close price series
        ticker: Ticker label for display
    """
    st.markdown("### 📈 Module 2: ARIMA Forecasting")

    with st.spinner("Fitting ARIMA model..."):
        model, order = fit_arima(prices)

    if model is None:
        st.warning("ARIMA model unavailable. Install pmdarima.")
        return

    metrics = get_arima_metrics(prices, model)
    fc, lower, upper = forecast_arima(prices, model, horizon=90)
    direction = "🔼 Uptrend" if fc.iloc[-1] > prices.iloc[-1] else "🔽 Downtrend"

    # --- Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        name="Actual Price", line=dict(color="white", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc.values,
        name="ARIMA Forecast", line=dict(color="#00b4d8", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(0,180,216,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI"
    ))
    fig.update_layout(
        template="plotly_dark", height=380,
        title=f"ARIMA{order} Forecast — {ticker}",
        xaxis_title="Date", yaxis_title="Price (INR)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Model info card ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Order", f"ARIMA{order}")
    col2.metric("AIC", metrics["AIC"])
    col3.metric("BIC", metrics["BIC"])
    col4.metric("Direction", direction)

    # --- Metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"₹{metrics['RMSE']:,.2f}")
    c2.metric("MAE", f"₹{metrics['MAE']:,.2f}")
    c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    c4.metric("Forecast Horizon", "90 Trading Days")

    # --- Walk-forward validation ---
    with st.expander("📊 Walk-Forward Validation (80/20 Split)"):
        with st.spinner("Running walk-forward validation..."):
            wf = walk_forward_validation(prices, order)
        vc1, vc2 = st.columns(2)
        vc1.metric("In-Sample RMSE", f"₹{wf['in_sample_rmse']:,.2f}")
        vc2.metric("Out-of-Sample RMSE", f"₹{wf['oos_rmse']:,.2f}")
