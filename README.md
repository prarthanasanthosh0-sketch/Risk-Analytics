# 📊 Risk Analytics Dashboard
**MCA · Financial Analytics · Capstone Project**

A fully functional, interactive 10-module Risk Analytics Dashboard built with Python, Streamlit, and Plotly — covering NSE/BSE-listed stocks with real-time financial data.

---

## 🧩 Modules Overview

| # | Module | Key Deliverables | Marks |
|---|--------|-----------------|-------|
| 1 | **Executive Summary Panel** | KPI cards, sparklines, investment signal, risk level | 15 |
| 2 | **ARIMA Forecasting** | 90-day forecast chart, RMSE/MAE/MAPE, walk-forward validation | 15 |
| 3 | **GARCH Volatility Modeling** | Conditional volatility chart, regime detection, spike annotation | 12 |
| 4 | **DCF Valuation** | Waterfall chart, DCF table, margin of safety | 13 |
| 5 | **Monte Carlo Simulation** | GBM path chart, probability summary, interactive sliders | 13 |
| 6 | **Value at Risk (VaR)** | 3-method VaR table, loss distribution, Kupiec backtesting | 15 |
| 7 | **Credit Risk Modeling** | PD estimate, credit score gauge, confusion matrix, PD trend | 13 |
| 8 | **Portfolio Optimization** | Efficient frontier, max-Sharpe allocation, asset toggle | 12 |
| 9 | **Stress Testing** | 5+ scenarios, factor betas, horizontal bar chart, risk narrative | 10 |
| 10 | **Correlation Heatmap** | Heatmap with annotations, diversification insights | 8 |

**Base Total: 126 marks | Bonus (deployed): +20 marks**

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/risk-analytics-dashboard.git
cd risk-analytics-dashboard

pip install -r requirements.txt

streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deployment (Streamlit Cloud — Bonus +20 marks)

1. Push to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as entry point
4. Click **Deploy**

---

## 📁 Project Structure

```
risk_dashboard/
│
├── app.py                          # Main Streamlit application (Module 1 + orchestration)
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
│
└── modules/
    ├── __init__.py
    ├── data_utils.py               # Data fetching, caching, utilities
    ├── arima_module.py             # Module 2: ARIMA Forecasting
    ├── garch_module.py             # Module 3: GARCH Volatility
    ├── dcf_module.py               # Module 4: DCF Valuation
    ├── monte_carlo_module.py       # Module 5: Monte Carlo Simulation
    ├── var_module.py               # Module 6: Value at Risk
    ├── credit_risk_module.py       # Module 7: Credit Risk Modeling
    ├── portfolio_module.py         # Module 8: Portfolio Optimization
    └── stress_correlation_modules.py  # Modules 9 & 10
```

---

## 📊 Data Sources

| Source | Usage |
|--------|-------|
| **Yahoo Finance (yfinance)** | OHLCV prices, financial ratios, cash flows |
| **Synthetic (numpy)** | Bond proxy (low-vol asset), PD training data |
| **GLD ETF (Yahoo Finance)** | Gold proxy for portfolio optimization |

---

## 🔧 Key Technical Decisions

### ARIMA (Module 2)
- `pmdarima.auto_arima` with `stepwise=True, seasonal=False`
- Walk-forward validation: 80/20 train/test split
- Forecast horizon: 90 trading days

### GARCH (Module 3)
- `arch` library, `GARCH(1,1)` with normal distribution
- Annualised by multiplying `sqrt(252)`
- Regime detection: percentile-based (P25/P75 thresholds)

### DCF (Module 4)
- Real operating cash flows from `yfinance.Ticker.cashflow`
- Gordon Growth Model for terminal value: `TV = FCF_n × (1+g) / (WACC-g)`
- User-adjustable sliders: WACC, terminal growth rate, forecast period

### Monte Carlo (Module 5)
- GBM formula: `S(t+1) = S(t) × exp((μ - 0.5σ²)dt + σ√dt × Z)`
- Default 1,000 paths, 252-day horizon (adjustable to 10,000)

### VaR (Module 6)
- Three methods: Historical Simulation, Parametric Normal, Monte Carlo
- CVaR at 95% confidence
- Kupiec POF backtesting with χ² likelihood ratio test

### Credit Risk (Module 7)
- Logistic regression on synthetic 500-company dataset
- Features: D/E, Interest Coverage, Current Ratio, ROE, Net Profit Margin
- AUC-ROC target ≥ 0.70

### Portfolio Optimization (Module 8)
- 5,000 random portfolios for efficient frontier
- `scipy.optimize.minimize` for max-Sharpe weights
- Risk-free rate: 6.5% (India 10Y GSec proxy)

### Stress Testing (Module 9)
- Factor betas via OLS regression
- 5 predefined + 1 user-custom scenario
- Impact = Σ(beta × factor shock)

---

## ⚠️ Academic Integrity Declaration

- All code is original work by the student
- AI tools (Claude) were used for **scaffolding and code structure generation**
- The student has reviewed, understood, and can explain every line of code
- This declaration is made in accordance with the assignment's academic integrity policy

---

## 👨‍💻 Team Member

| Name | Roll No | Contributions |
|------|---------|--------------|
| [Your Name] | [Roll No] | All 10 modules, data pipeline, deployment |

---

## 📈 Limitations & Future Improvements

**Limitations:**
- ARIMA walk-forward validation can be slow on large datasets (>1000 rows)
- DCF model uses operating cash flow as FCF proxy — true FCF requires CapEx adjustment
- PD model trained on synthetic data; a real credit bureau dataset would improve accuracy
- Gold proxy (GLD) is USD-denominated; INR conversion adds minor basis risk

**Future Improvements:**
- Add real-time WebSocket streaming for live price updates
- Integrate Nifty50 index as benchmark for beta computation
- Add LSTM/Prophet as alternative forecasting models
- Implement proper Fama-French 3-factor model for expected returns
- Add PDF report export button
