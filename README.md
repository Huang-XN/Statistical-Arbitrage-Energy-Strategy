# Quant Finance Lab 🧬📈

Welcome to my personal quantitative finance laboratory. This repository documents my journey of transitioning from Bioinformatics to Quantitative Finance, focusing on reconstructing mathematical models and trading strategies from scratch.

## 📂 Project Overview

### 1. Warm Up (Trend Following Strategy, Moving Average)
* **File:** `01_warm_up.ipynb`
* **Objective:** To implement a classic "Golden Cross" strategy on AAPL stock using pure Python/Pandas logic.
* **Key Tech:** Data visualization, Signal generation using `.shift()` to avoid look-ahead bias, Backtesting framework.
* **Result:** Verified the strategy's Sharpe ratio vs. Buy & Hold, laying the foundation for vectorized backtesting.

### 2. Volatility Modeling (GARCH)
* **File:** `02_GARCH.ipynb`
* **Objective:** To model the "Volatility Clustering" phenomenon in financial time series, moving beyond constant volatility assumptions in Black-Scholes.
* **Key Tech:** Monte Carlo Simulation (Geometric Brownian Motion), GARCH(1,1) model implementation using `arch` library, Value-at-Risk (VaR) calculation.
* **Insight:** Identified statistical significance of volatility shocks and visualized the leverage effect in AAPL returns.

### 3. Statistical Arbitrage (Pairs Trading)
* **File:** `03_Pair_Trading.ipynb`
* **Objective:** To build a market-neutral strategy based on Cointegration theory rather than simple correlation.
* **Key Tech:** Augmented Dickey-Fuller (ADF) Test, OLS Regression for Hedge Ratio, Z-Score signal generation.
* **Case Study:**
    * **PEP vs KO:** Failed stationarity test (P-value > 0.05), proving structural breaks in consumer staples.
    * **XOM vs CVX:** Confirmed cointegration (P-value < 0.05).
* **Performance:** Achieved a total Log Return of ~1.09 over 5 years, despite significant drawdowns in 2022 due to geopolitical structural breaks.


### 7. Market Regime Detection (Hidden Markov Model) 🕵️‍♂️
* **File:** `07_HMM.ipynb`
* **Objective:** To switch from "micro-prediction" (forecasting daily returns) to "macro-identification" (detecting market states), applying the logic of **Cell State Classification** to financial markets.
* **Key Tech:** Gaussian HMM (Unsupervised Learning), Dynamic Regime Detection.
* **Insight:**
    * Successfully identified the **2022 Bear Market** (State 2: High Volatility, Negative Returns) without any labeled data.
    * Identified the **2024 Bull Run** (State 0/1: Low Volatility, Positive Returns).
    * **Visual Proof:**
    ![HMM Regimes](Figure_1_HMM_for_AAPL.jpg)
    *(Note: The model autonomously learned to flag high-risk periods in red)*

## 🚀 About Me
I am a **Ph.D. candidate in Biology** with a strong foundation in **Mathematics (B.S.)** and **Financial Mathematics (M.S.)**. 

For the past years, I have been working in biological research, specializing in extracting signals from high-dimensional, noisy biological sequencing data. I am now applying this "signal-from-noise" expertise to financial markets.
