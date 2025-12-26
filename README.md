# Quant Finance Lab 🧬📈

Welcome to my personal quantitative finance laboratory. This repository documents my journey of transitioning from **Computational Biology** to **Quantitative Finance**. 

I combine my background in **Mathematics** and **Ph.D. research in high-noise biological data processing** to reconstruct financial models. My goal is to treat the market not as a casino, but as a complex, adaptive system similar to biological organisms.

## 🧠 Philosophy
* **Signal from Noise:** Just as single-cell sequencing data requires denoising to find cell types, financial time series requires rigorous statistical filtering to find Alpha.
* **Regime Dependence:** Markets, like biological environments, have distinct "seasons" (Regimes). A strategy that survives in winter (Bear) may die in summer (Bull).

---

## 📂 Project Portfolio

### Part 1: Classical Quant Strategies (The Foundation)

#### 1. Trend Following (Moving Average)
* **File:** `1_warm_up.ipynb`
* **Objective:** To build a robust backtesting framework avoiding "Look-ahead Bias".
* **Key Tech:** Vectorized backtesting, Signal processing.
* **Result:** Demonstrated how a simple logic can reduce drawdown compared to Buy & Hold during bear markets.

#### 2. Volatility Modeling (GARCH)
* **File:** `GARCH.ipynb`
* **Objective:** To model "Volatility Clustering" — the financial equivalent of "bursty" biological signals.
* **Key Tech:** Monte Carlo Simulation (Geometric Brownian Motion), GARCH(1,1) with `arch` library, Value-at-Risk (VaR).
* **Insight:** Visualized the "Leverage Effect" (fear is stronger than greed) in AAPL returns.

#### 3. Statistical Arbitrage (Pairs Trading)
* **File:** `Pair_Trading.ipynb`
* **Objective:** To exploit Mean Reversion using Cointegration (ADF Test) rather than simple Correlation.
* **Case Study:** * **XOM vs CVX:** Confirmed stationarity (P-value < 0.05).
    * **Result:** Achieved ~1.09x Log Return over 5 years, but identified significant risks during structural breaks (e.g., 2022 geopolitical crisis).

---

### Part 2: Machine Learning & AI (The Advanced)

#### 4. Directional Prediction (Random Forest)
* **File:** `04_Machine_Learning_Prediction.ipynb`
* **Objective:** To treat stock price movement as a **Binary Classification** problem (Up/Down).
* **Feature Engineering:**
    * **Lagged Returns:** Capturing Momentum (similar to autoregulation in biology).
    * **Macro Context:** Added `SPY` (S&P 500) returns to capture systematic risk.
    * **RSI:** Added Relative Strength Index to capture mean-reversion boundaries.
* **Performance:** Improved accuracy from **54.03%** (Baseline) to **56.85%** (with SPY & RSI context).

#### 5. The Overfitting Experiment (Hyperparameter Tuning)
* **File:** `5_Hyperparameter_Optimization/05_Hyperparameter_Tuning.ipynb`
            `6_Ensemble/6_VotingClassifier.ipynb`
* **Objective:** To optimize the model using Grid Search and Ensemble Learning (Voting Classifier).
* **Critical Finding (Shadow Work):**
    * **Optimization Trap:** Grid Search improved Training Accuracy to **95.28%** but dropped Test Accuracy to **51.41%**.
    * **Lesson Learned:** Financial data suffers from severe **Data Drift**. Complex models tend to memorize history rather than learning laws. Simplicity often outperforms complexity out-of-sample.

#### 6. Market Regime Detection (Hidden Markov Model)
* **File:** `7_HMM.ipynb`
* **Objective:** Unsupervised learning to decode the market's "Hidden States" (Bull, Bear, Sideways).
* **Key Tech:** Gaussian HMM with `hmmlearn`.
* **Result:** Successfully identified the 2022 Bear Market as a distinct high-volatility state (State 2) without human labeling.
* **Visual:**
  ![Market Regimes](./7_HMM/Figure_1_HMM_for_AAPL.png)
  *(Figure: Unsupervised clustering of market regimes. Red dots indicate high-volatility "Bear" states identified by the model.)*
