"""
Data utilities module for Risk Analytics Dashboard.
Handles data fetching, cleaning, and preprocessing for all modules.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "WIPRO.NS": "Wipro",
}

TICKER_LIST = list(TICKERS.keys())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker and date range from Yahoo Finance.

    Args:
        ticker: NSE/BSE ticker symbol (e.g., 'RELIANCE.NS')
        start: Start date string 'YYYY-MM-DD'
        end: End date string 'YYYY-MM-DD'

    Returns:
        DataFrame with OHLCV columns, datetime index, cleaned of NaNs.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_tickers(tickers: list, start: str, end: str) -> dict:
    """
    Fetch price data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start: Start date string
        end: End date string

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    data = {}
    for t in tickers:
        df = fetch_price_data(t, start, end)
        if not df.empty:
            data[t] = df
    return data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns from a price series.

    Args:
        prices: Series of closing prices

    Returns:
        Series of log returns, NaNs dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_rolling_volatility(log_returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling annualised volatility.

    Args:
        log_returns: Series of log returns
        window: Rolling window in days (default 20)

    Returns:
        Series of rolling annualised volatility (%).
    """
    return log_returns.rolling(window).std() * np.sqrt(252) * 100


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    """
    Fetch stock fundamental info dict from yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary of fundamental data.
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cashflow(ticker: str) -> pd.DataFrame:
    """
    Fetch annual cash flow statement from yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with cash flow items as rows, years as columns.
    """
    try:
        stock = yf.Ticker(ticker)
        cf = stock.cashflow
        return cf
    except Exception:
        return pd.DataFrame()


def get_bond_proxy(start: str, end: str) -> pd.DataFrame:
    """
    Simulate a bond proxy as a low-volatility asset (e.g., 6% annual return with 2% vol).

    Args:
        start: Start date string
        end: End date string

    Returns:
        DataFrame with synthetic 'Close' prices.
    """
    date_range = pd.bdate_range(start=start, end=end)
    n = len(date_range)
    np.random.seed(42)
    daily_ret = 0.06 / 252
    daily_vol = 0.02 / np.sqrt(252)
    returns = np.random.normal(daily_ret, daily_vol, n)
    prices = 1000 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": prices}, index=date_range)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gold(start: str, end: str) -> pd.DataFrame:
    """
    Fetch Gold ETF proxy (GLD) from Yahoo Finance.

    Args:
        start: Start date string
        end: End date string

    Returns:
        DataFrame with Close prices.
    """
    df = fetch_price_data("GLD", start, end)
    if df.empty:
        # Fallback synthetic gold
        date_range = pd.bdate_range(start=start, end=end)
        n = len(date_range)
        np.random.seed(99)
        daily_ret = 0.08 / 252
        daily_vol = 0.12 / np.sqrt(252)
        returns = np.random.normal(daily_ret, daily_vol, n)
        prices = 5000 * np.cumprod(1 + returns)
        df = pd.DataFrame({"Close": prices}, index=date_range)
    return df
