# AI-Driven Quant Research and Predictive Trading Platform

This repository is the foundation for an institutional-style quantitative research platform. We will build it step by step, from clean research infrastructure to feature engineering, forecasting models, reinforcement learning, regime detection, portfolio simulation, live signals, monitoring, and risk controls.

You listed 19 major build steps in the brief. We will proceed sequentially, starting with Step 1 and building each layer into a reusable research platform.

## Step 1: Project Setup and Architecture

Professional quant research fails quickly when code, data, experiments, and results are mixed together. The goal of this first step is to separate responsibilities so every later component can be tested, reused, and audited.

### Why Architecture Matters in Finance

Financial ML is unusually sensitive to small mistakes:

- A feature computed with future data can create fake alpha.
- A random train/test split can leak market regimes from the future into the past.
- A backtest without transaction costs can turn a losing strategy into a beautiful lie.
- A notebook-only workflow is hard to reproduce for investment committees, risk teams, or production deployment.

This project uses a modular structure so research code can graduate into production code without being rewritten from scratch.

## Repository Structure

```text
.
|-- config/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- external/
|   `-- live/
|-- evaluation/
|-- explainability/
|-- features/
|-- live/
|-- models/
|-- notebooks/
|-- optimization/
|-- results/
|-- rl/
|-- scripts/
|-- tests/
|-- visualization/
|-- README.md
|-- requirements.txt
`-- pyproject.toml
```

## Module Roles in a Quant Research Pipeline

### `data/`

Stores all market and alternative data.

- `data/raw/`: original vendor/API downloads. This should be treated as immutable evidence.
- `data/processed/`: cleaned, aligned, point-in-time safe datasets.
- `data/external/`: macro, sentiment, Fama-French, VIX, Google Trends, or other non-price data.
- `data/live/`: rolling buffers and temporary real-time market snapshots.

In finance, preserving raw data matters because backtest results must be reproducible and auditable.

### `features/`

Contains feature engineering logic such as returns, momentum, volatility, volume, market structure, sentiment features, and rolling normalization.

This is where raw prices become candidate alpha signals. Feature code must avoid lookahead bias, meaning every feature at time `t` must use only information available at or before time `t`.

### `models/`

Contains supervised learning models such as XGBoost, LSTM, linear baselines, ensemble models, and model serialization utilities.

The model layer should not know how to download data or execute trades. Its job is to learn the relationship between features and targets.

### `evaluation/`

Contains metrics and backtesting utilities:

- Information Coefficient
- Information Ratio
- Sharpe Ratio
- Max Drawdown
- CAGR
- Calmar Ratio
- Profit Factor
- Win Rate

Prediction accuracy alone is not enough in trading. A model can be right often and still lose money if losses are larger than wins, costs are high, or signals arrive in poor regimes.

### `optimization/`

Contains Optuna studies and hyperparameter tuning logic.

Quant models are fragile when overtuned. This module will use time-series cross-validation, purging, and embargo logic instead of random cross-validation.

### `rl/`

Contains reinforcement learning trading environments and agents.

RL is useful when the problem is not just predicting returns, but deciding sequential actions under transaction costs, drawdown constraints, changing positions, and delayed rewards.

### `explainability/`

Contains SHAP-based model interpretation tools.

In institutional finance, risk committees and model validation teams need to understand why a model is taking risk. Explainability is also important for model governance frameworks such as SR 11-7 and regulatory expectations around model oversight.

### `live/`

Contains real-time market data ingestion, rolling feature updates, live inference, alerting hooks, and risk checks.

Research code becomes dangerous when deployed without live-specific safeguards. This module will handle streaming data, reconnection logic, confidence-weighted signals, and kill-switch controls.

### `visualization/`

Contains Plotly and Streamlit dashboard code.

Good visualization helps researchers see alpha decay, drawdowns, regime changes, feature drift, and model behavior before those issues become capital losses.

### `notebooks/`

Contains exploratory research notebooks.

Notebooks are useful for discovery, but final reusable logic should move into importable modules.

### `tests/`

Contains unit and integration tests.

In quant finance, tests are not only about software correctness. They protect against silent research errors such as future leakage, timestamp misalignment, and incorrect return calculations.

### `results/`

Stores generated artifacts such as trained models, backtest reports, plots, MLflow runs, and optimization studies.

Large generated files should usually stay out of Git unless they are small and intentionally versioned.

## Installation

Create and activate a Python 3.10+ virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the first smoke test:

```bash
pytest
```

## Configuration

The main configuration file is:

```text
config/config.yaml
```

It centralizes:

- Data paths and symbols
- Preprocessing rules
- Feature windows
- Target horizons
- Model defaults
- Evaluation settings
- Live trading controls
- MLflow tracking settings

Centralized configuration matters because research needs to be reproducible. If a backtest result depends on hidden notebook variables, it is not production-ready.

## Step 2: Data Collection and Preprocessing

Step 2 adds reusable data collectors and preprocessing utilities:

- Historical OHLCV from yfinance
- Binance REST historical crypto data
- UTC timestamp alignment
- Missing value handling
- Intraday gap forward-fill
- 3-sigma return outlier clipping
- Rolling z-score normalization with no lookahead
- Beginner-friendly explanations of lookahead bias, survivorship bias, data leakage, and point-in-time data

### Files Added

```text
data/
|-- __init__.py
|-- market_data.py
`-- preprocessing.py
scripts/
`-- download_data.py
tests/
`-- test_preprocessing.py
```

### Downloading Historical Data

Use the command-line downloader for Yahoo Finance symbols:

```bash
python scripts/download_data.py --symbols SPY AAPL GC=F EURUSD=X BTC-USD --start 2020-01-01 --interval 1d
```

This saves raw files into `data/raw/` and cleaned files into `data/processed/`.

You can also use the collectors directly:

```python
from data.market_data import BinanceHistoricalCollector, YahooFinanceCollector

yahoo = YahooFinanceCollector()
spy = yahoo.download_symbol("SPY", start="2020-01-01", interval="1d", asset_class="stocks")

binance = BinanceHistoricalCollector()
btc = binance.download_klines("BTCUSDT", interval="1h", start="1 Jan, 2022")
```

### Preprocessing Pipeline

```python
from data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    missing_value_method="ffill",
    outlier_clip_sigma=3.0,
    rolling_zscore_window=60,
    min_periods=20,
)

clean = preprocessor.clean_ohlcv(spy)
```

The default preprocessing chain:

1. Standardizes OHLCV column names.
2. Converts timestamps to UTC.
3. Optionally forward-fills intraday gaps.
4. Fills missing values using past values.
5. Computes log returns.
6. Clips extreme return outliers using shifted expanding statistics.
7. Adds rolling z-score normalized columns using shifted rolling windows.

### Why UTC Alignment Matters

Global assets trade in different sessions and time zones:

- US equities use New York exchange hours.
- Forex trades nearly 24 hours across global sessions.
- Crypto trades continuously.
- Gold futures have their own exchange schedule.

If timestamps are not aligned to UTC, a model may accidentally compare prices from different real-world moments. That can create false relationships between assets.

### Lookahead Bias

Lookahead bias happens when a feature at time `t` uses information that was only known after time `t`.

Example: normalizing today's return using the full-sample mean and standard deviation leaks future volatility into the past. This project avoids that by shifting rolling and expanding statistics by one bar.

In finance, lookahead bias is deadly because it makes a strategy look profitable in research and fail in live trading.

### Survivorship Bias

Survivorship bias happens when the dataset includes only assets that still exist today.

Example: training only on the current S&P 500 ignores companies that were removed after poor performance, bankruptcy, or acquisition. That makes historical performance look better than it really was.

For production equity research, use point-in-time index membership and delisted securities.

### Data Leakage

Data leakage is any path where future information enters training data, validation data, features, labels, or preprocessing.

Common leakage examples:

- Randomly splitting time-series rows.
- Scaling features using the entire dataset before train/test split.
- Joining macro data by report period instead of release timestamp.
- Using revised economic data instead of the values known at the time.

This is why preprocessing functions in this repo use past-only rolling statistics.

### Point-in-Time Data

Point-in-time data means the dataset reflects what a trader could have known at that exact moment.

This matters because financial datasets are revised:

- Economic indicators can be revised after release.
- Company fundamentals can be restated.
- Index constituents change over time.
- Corporate actions can alter historical prices.

Institutional-grade research must preserve the historical information set, not the cleaned-up version visible today.

## Step 3: Feature Engineering Engine

Step 3 turns cleaned OHLCV data into candidate alpha features.

### Files Added

```text
features/
|-- __init__.py
`-- technical.py
scripts/
`-- build_features.py
tests/
`-- test_features.py
```

### Running Feature Generation

After Step 2 has created a processed OHLCV file, build features with:

```bash
python scripts/build_features.py --input data/processed/SPY.csv --output data/processed/SPY_features.csv
```

Or use the feature engine directly:

```python
from features import FeatureEngineer

engineer = FeatureEngineer()
feature_frame = engineer.build_all_features(clean_ohlcv_frame)
model_matrix = feature_frame[engineer.feature_columns(feature_frame)]
```

### Return Features

The engine creates:

- `log_return`
- `return_5`, `return_10`, `return_20`, `return_60`
- `return_autocorr_lag_1` through `return_autocorr_lag_5`

Why this matters: returns are the basic language of quant finance. Rolling returns capture intermediate-term momentum or reversal. Autocorrelation asks whether recent returns persist, mean-revert, or behave like noise.

Market inefficiency intuition: investor underreaction can create momentum, while overreaction and liquidity shocks can create short-term reversal.

### Momentum Features

The engine creates:

- `rsi_14`
- `ema_8`, `ema_33`
- `ema_cross`, `ema_cross_signal`
- `macd`, `macd_signal`, `macd_histogram`
- `roc_5`, `roc_10`, `roc_20`, `roc_60`

Why this matters: momentum indicators compress trend, acceleration, and exhaustion into model-readable variables.

Market inefficiency intuition: large institutions often adjust positions gradually, creating persistent trends. RSI and MACD can also detect stretched conditions where the next marginal buyer or seller may be weaker.

### Volatility Features

The engine creates:

- `volatility_10`, `volatility_20`
- `atr_14`
- `volatility_regime`

Why this matters: volatility controls risk, position sizing, stop distance, and model stability. The same signal can behave very differently in calm versus stressed markets.

Market inefficiency intuition: volatility clustering is one of the strongest empirical facts in markets. Regime-aware models can avoid treating calm drift and panic liquidation as the same environment.

### Volume Features

The engine creates:

- `obv`
- `volume_ratio_20`
- `volume_zscore_20`

Why this matters: price movement with unusual volume carries different information than price movement on thin volume.

Market inefficiency intuition: volume can reveal participation. A breakout with strong volume may indicate institutional demand, while a low-volume breakout may be easier to fade.

### Market Structure Features

The engine creates:

- `fractal_high_confirmed`
- `fractal_low_confirmed`
- `bullish_liquidity_sweep`
- `bearish_liquidity_sweep`
- `higher_high`
- `lower_low`
- `trend_structure`
- `vwap`
- `vwap_deviation`

Why this matters: market structure features describe where price is relative to recent swing points, liquidity pools, and volume-weighted fair value.

Market inefficiency intuition: stops and resting liquidity often cluster near recent highs/lows. A sweep followed by rejection can signal forced liquidation rather than genuine trend continuation. VWAP deviation helps identify whether price is stretched versus the average execution level.

### No-Lookahead Fractals

Classic fractal indicators often look into future candles to decide whether a past candle was a swing high or low. That is fine for chart annotation but dangerous for model training.

This project uses confirmed fractals. If a swing high needs two bars on the right to be confirmed, the feature appears two bars later, when the information would have been known in live trading.

## Step 4: Target Variable and Research Design

Step 4 defines what the model is trying to predict and how we evaluate that prediction without time-series leakage.

### Files Added

```text
features/
`-- targets.py
evaluation/
|-- metrics.py
`-- validation.py
tests/
`-- test_targets_and_validation.py
```

### Regression Target

For horizon `N`, the regression label is:

```text
target_return_N[t] = log(close[t + N] / close[t])
```

The project creates `target_return_1` and `target_return_5` by default.

Why this matters: return prediction is the cleanest way to train a ranking or sizing model. It preserves magnitude, so later portfolio logic can distinguish a weak signal from a strong signal.

### Classification Targets

The project also creates:

- `target_direction_N`: `1` for non-negative future return and `-1` for negative future return.
- `target_tertile_N`: `-1` bottom return bucket, `0` middle bucket, `1` top bucket.

Why this matters: classification can be more stable than direct return prediction when returns are noisy. Direction labels support BUY/SELL/HOLD logic, while tertile labels support ranking assets into weak, neutral, and strong groups.

Tertile labels use shifted expanding quantiles by default. That avoids defining classes with the full future return distribution.

### Building a Supervised Dataset

```python
from features import FeatureEngineer, TargetBuilder, build_supervised_frame

feature_frame = FeatureEngineer().build_all_features(clean_ohlcv_frame)

X, y = build_supervised_frame(
    feature_frame,
    target_builder=TargetBuilder(horizons=(1, 5)),
    target_column="target_return_1",
)
```

Rows with missing features or missing labels are dropped together so timestamps remain aligned.

### Why Random Split Is Wrong in Finance

Random train/test split assumes observations are independent and identically distributed. Markets are not.

Financial data has:

- Serial correlation
- Volatility clustering
- Regime shifts
- Overlapping labels
- Time-varying liquidity

If we randomly mix 2020, 2021, 2022, and 2023 rows, the model can learn future regimes while pretending to predict the past. That makes backtests look smarter than they would be in live trading.

Use chronological splits:

```python
from evaluation import temporal_train_test_split

train_idx, test_idx = temporal_train_test_split(
    X,
    test_size=0.2,
    purge_bars=5,
    embargo_bars=5,
)
```

### Expanding Window Cross-Validation

Expanding windows simulate a research desk that learns more history over time:

```python
from evaluation import ExpandingWindowSplit

splitter = ExpandingWindowSplit(
    initial_train_size=504,
    test_size=63,
    purge_bars=5,
    embargo_bars=5,
)

for train_idx, test_idx in splitter.split(X):
    ...
```

Why this matters: it mimics a production model that is periodically retrained as more market history becomes available.

### Rolling Window Cross-Validation

Rolling windows keep a fixed training length:

```python
from evaluation import RollingWindowSplit

splitter = RollingWindowSplit(
    train_size=504,
    test_size=63,
    purge_bars=5,
    embargo_bars=5,
)
```

Why this matters: older data can become stale. A rolling window adapts faster when regimes change.

### Purging and Embargo

If the target is a 5-day forward return, labels near a split boundary overlap across train and test. A training row just before the split may include returns from inside the test period.

Purging removes training rows near the test boundary. Embargo skips test rows immediately after the train boundary.

Why this matters: it prevents the model from seeing outcome information that is too close to the test window.

### Metrics

Step 4 adds:

- Information Coefficient: correlation between predictions and realized returns.
- Rolling IC: rolling signal quality over time.
- Information Ratio: mean IC divided by IC volatility, annualized.
- Hit rate: fraction of correct directional calls.

Why this matters: IC tells us whether a signal ranks future returns well. A strategy can have mediocre accuracy but strong IC if it correctly ranks the biggest winners and losers.

## Step 5: XGBoost Factor Model

Step 5 adds the first supervised predictive model layer: XGBoost factor models for tabular quant features.

### Files Added

```text
models/
|-- factor_dataset.py
`-- xgboost_factor.py
evaluation/
`-- performance.py
scripts/
`-- train_xgboost.py
tests/
|-- test_factor_dataset.py
`-- test_xgboost_factor.py
```

### Why XGBoost Dominates Tabular Quant Finance

XGBoost is often a strong first institutional model because market features are usually tabular:

- Momentum indicators
- Volatility estimates
- Volume features
- Cross-sectional rankings
- Regime labels
- Macro and sentiment features

Linear models assume mostly straight-line relationships. Markets often have thresholds and interactions instead:

- Momentum may work only when volatility is low.
- Volume spikes may matter only near recent highs or lows.
- RSI can mean different things in trending and ranging regimes.

Gradient-boosted trees capture nonlinear rules and feature interactions without needing a deep neural network.

### Regression Model

Use regression when the target is future return magnitude:

```python
from models import XGBoostFactorModel, XGBoostModelConfig

model = XGBoostFactorModel(
    task="regression",
    config=XGBoostModelConfig(n_estimators=500, max_depth=4),
)

model.fit(train_x, train_y, valid_x, valid_y)
predicted_returns = model.predict(test_x)
```

Regression is useful for ranking assets and sizing positions by signal strength.

### Classification Model

Use classification when the target is direction:

```python
model = XGBoostFactorModel(task="classification")
model.fit(train_x, train_direction, valid_x, valid_direction)
up_probability = model.predict(test_x)
direction = model.predict_direction(test_x)
```

Classification is useful for BUY/SELL/HOLD logic when exact return magnitude is too noisy.

### Timestamp-Aligned Feature Matrix

```python
from models import build_factor_dataset

dataset = build_factor_dataset(
    feature_frame,
    target_column="target_return_1",
)

X = dataset.x
y = dataset.y
realized_returns = dataset.realized_returns
```

The helper drops missing rows only after features and targets are joined, preserving timestamp alignment.

### Early Stopping on Validation IC

The XGBoost wrapper uses native XGBoost training with a custom validation Information Coefficient metric. Early stopping selects the boosting round with the best validation rank IC.

Why this matters: RMSE can improve while trading signal quality gets worse. In quant research, we care whether predictions rank future returns correctly.

### Evaluation

The model evaluates:

- RMSE for regression
- Accuracy for classification
- Information Coefficient
- Hit rate
- Annualized Sharpe from the derived signal strategy
- Max Drawdown from the derived signal strategy

Example:

```python
evaluation = model.evaluate(
    test_x,
    test_y,
    realized_returns=test_returns,
    signal_threshold=0.0,
    transaction_cost_bps=2.0,
)
```

### Command-Line Training

After Step 3 has generated a feature CSV:

```bash
python scripts/train_xgboost.py --input data/processed/SPY_features.csv --task regression
```

For direction classification:

```bash
python scripts/train_xgboost.py --input data/processed/SPY_features.csv --task classification
```

The script saves a model and metrics JSON under `results/xgboost/`.

## Step 6: LSTM Time-Series Model

Step 6 adds a sequence model for forecasting future returns from ordered feature histories.

### Files Added

```text
models/
`-- lstm_timeseries.py
scripts/
`-- train_lstm.py
tests/
`-- test_lstm_timeseries.py
```

### Sliding Window Sequences

The LSTM input shape is:

```text
samples x timesteps x features
```

By default:

```text
T = 60 timesteps
N = number of engineered features
```

The sample ending at timestamp `t` uses feature rows:

```text
[t - 59, ..., t]
```

and predicts the target at `t`, such as `target_return_1`.

Why this matters: the model sees the path of features through time, not just the current snapshot. That can help when the order of events matters, such as volatility expansion after a slow trend or momentum weakening before reversal.

```python
from models import create_lstm_sequences

sequence_dataset = create_lstm_sequences(
    features=X,
    target=y,
    sequence_length=60,
)
```

### Architecture

The default LSTM model uses:

- Two stacked LSTM layers
- 128 hidden units per LSTM layer
- Dropout `0.2` after each LSTM layer
- Dense output layer for return prediction
- Adam optimizer
- Mean squared error loss
- ReduceLROnPlateau
- EarlyStopping with restored best weights

```python
from models import LSTMModelConfig, LSTMTimeSeriesModel

config = LSTMModelConfig(
    sequence_length=60,
    hidden_units=128,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
)

model = LSTMTimeSeriesModel(config)
model.fit(train_sequences, validation=validation_sequences)
predictions = model.predict(test_sequences)
```

### TimeSeriesSplit Cross-Validation

The sequence split helper wraps scikit-learn `TimeSeriesSplit`:

```python
from models import time_series_sequence_splits

splits = time_series_sequence_splits(
    sequence_dataset,
    n_splits=5,
    test_size=63,
)
```

Each split is chronological: training sequence indices are always before test sequence indices. That protects against future leakage.

### Command-Line Training

After Step 3 has generated a feature CSV:

```bash
python scripts/train_lstm.py --input data/processed/SPY_features.csv --sequence-length 60
```

The script saves:

- `results/lstm/lstm_return_forecaster.keras`
- `results/lstm/lstm_metrics.json`
- `results/lstm/lstm_features.json`

### XGBoost vs LSTM

XGBoost is usually the first model to try in quant finance:

- Faster to train
- Strong on tabular engineered features
- Easier to debug and explain
- Works well on smaller datasets

LSTM can help when:

- Feature order matters
- Patterns unfold across many bars
- Sequential state contains information not captured by current indicators
- You have enough data to avoid overfitting

The tradeoff: LSTMs are slower, less interpretable, and easier to overfit. In an institutional workflow, XGBoost is often the benchmark that an LSTM must beat after costs and walk-forward validation.

## Step 7: SHAP Explainability Engine

Step 7 adds model interpretability tooling for research review, risk committees, and production model governance.

### Files Added

```text
explainability/
|-- __init__.py
`-- shap_engine.py
scripts/
`-- explain_xgboost.py
tests/
`-- test_shap_engine.py
```

### Why Explainability Matters in Finance

A profitable backtest is not enough for institutional deployment. A model also needs a credible answer to:

- What features drive risk?
- Did the model learn real market structure or noisy artifacts?
- Does behavior change across regimes?
- Are predictions explainable to investment, risk, and compliance teams?
- Can model validation challenge the assumptions?

SHAP helps decompose predictions into feature contributions. That is useful for internal model governance frameworks such as SR 11-7 and for explainability expectations connected to regulations such as MiFID II.

### XGBoost TreeExplainer

TreeExplainer is fast and accurate for tree models:

```python
from explainability import ShapExplainer, global_feature_importance

explainer = ShapExplainer()
explanation = explainer.explain_xgboost(model, test_x)

importance = global_feature_importance(
    explanation.values,
    explanation.feature_names,
)
```

The output ranks features by `mean(abs(SHAP))`, which answers: which variables most changed model predictions on average?

### LSTM DeepExplainer

DeepExplainer can explain TensorFlow/Keras sequence models:

```python
explanation = explainer.explain_lstm(
    model=lstm_model,
    background_sequences=train_sequences.x[:100],
    sample_sequences=test_sequences.x[:50],
    feature_names=test_sequences.feature_names,
)
```

For LSTM SHAP arrays shaped `samples x timesteps x features`, global importance averages across both samples and timesteps.

### Force Plot Data

A force plot decomposes one prediction:

```python
explainer.save_force_plot(
    explanation,
    row=0,
    output_path="results/explainability/force_plot.html",
)
```

Why this matters: local explanations help answer why one BUY/SELL/HOLD signal happened at a specific timestamp.

### Dependence Plot Data

Dependence analysis shows how feature values relate to SHAP impact:

```python
from explainability import dependence_frame

ema_dependence = dependence_frame(
    explanation.values,
    test_x,
    feature="ema_cross",
    interaction_feature="volume_zscore_20",
)
```

Why this matters: if `ema_cross` contributes positively only when volume is elevated, the model may have learned a trend-confirmation interaction rather than simple trend chasing.

### Feature Interaction Detection

For XGBoost, the engine can rank SHAP interaction values:

```python
interactions = explainer.xgboost_interactions(model, test_x)
```

To focus on EMA and volume interactions:

```python
from explainability import analyze_ema_volume_interactions

ema_volume = analyze_ema_volume_interactions(
    shap_interaction_values,
    feature_names,
)
```

Why this matters: interactions often reveal conditional alpha. For example, an EMA crossover may be weak alone but meaningful when paired with abnormal volume.

### Command-Line XGBoost Explanation

Training now saves feature names next to the model:

```bash
python scripts/train_xgboost.py --input data/processed/SPY_features.csv --task regression
```

Then generate SHAP artifacts:

```bash
python scripts/explain_xgboost.py \
  --model results/xgboost/xgboost_regression.json \
  --features-json results/xgboost/xgboost_regression_features.json \
  --input data/processed/SPY_features.csv
```

This writes:

- `results/explainability/xgboost_global_importance.csv`
- `results/explainability/xgboost_top_dependence.csv`
- `results/explainability/xgboost_interactions.csv`

## Step 8: Reinforcement Learning Trading

Step 8 adds a Gym-compatible reinforcement learning environment and Stable-Baselines3 agent helpers.

### Files Added

```text
rl/
|-- __init__.py
|-- agents.py
`-- trading_env.py
scripts/
`-- train_rl_agent.py
tests/
`-- test_rl_trading_env.py
```

### Environment State

The default state includes market features plus portfolio state:

```text
[open, high, low, close, volume, RSI, EMA crossover, MACD, ATR, portfolio_value_ratio, position]
```

Why this matters: a trading decision depends on both the market and the current portfolio. Buying while flat is different from buying when already long.

### Action Space

The environment uses `Discrete(3)`:

```text
0 = Hold current position
1 = Buy / target long
2 = Sell / target short
```

If shorting is disabled, `Sell` targets flat instead of short.

### Reward Function

The reward is:

```text
(step_return / rolling_volatility) - drawdown_penalty - transaction_cost
```

where:

- `step_return` is the next-bar return from the chosen position after costs.
- `rolling_volatility` normalizes rewards so the agent does not prefer noisy profits blindly.
- `drawdown_penalty` discourages paths that damage capital.
- `transaction_cost` prevents unrealistic overtrading.

Why this matters: RL agents optimize the reward literally. A poorly designed reward can create a model that trades too often, takes hidden tail risk, or maximizes short-term gains while accepting unacceptable drawdowns.

### Create the Environment

```python
from rl import TradingEnvironment, TradingEnvironmentConfig

env = TradingEnvironment(
    feature_frame,
    config=TradingEnvironmentConfig(
        initial_cash=100_000,
        transaction_cost_bps=2.0,
        drawdown_penalty=0.1,
    ),
)
```

### Train DQN, PPO, or A2C

```python
from rl import RLTrainingConfig, train_agent

agent = train_agent(
    env,
    RLTrainingConfig(
        agent="PPO",
        total_timesteps=10_000,
    ),
)
```

Supported agents:

- `DQN`
- `PPO`
- `A2C`

### Command-Line Training

```bash
python scripts/train_rl_agent.py --input data/processed/SPY_features.csv --agent PPO --timesteps 10000
```

This saves:

- `results/rl/ppo_trading_agent.zip`
- `results/rl/ppo_metrics.json`
- `results/rl/ppo_returns.csv`

### Compare RL vs XGBoost Signal Strategy

```python
from rl import compare_rl_to_signal_strategy

comparison = compare_rl_to_signal_strategy(
    rl_returns=rl_returns,
    signal_returns=xgboost_signal_returns,
)
```

Why this matters: an RL agent should not be judged in isolation. It must beat simpler baselines such as XGBoost signal strategies after transaction costs and drawdowns.

### Sequential Decision-Making

Supervised models answer: what is the next likely return?

RL models answer: what action should I take now, given market state, current position, costs, and risk?

That distinction matters because profitable trading is sequential. Today’s action changes tomorrow’s state through position, cash, drawdown, and risk limits.

### Exploration vs Exploitation

RL must balance:

- Exploration: try actions to discover what works.
- Exploitation: use actions currently believed to be profitable.

In markets, exploration is expensive because bad trades lose capital. That is why RL is usually developed in simulation first and benchmarked heavily before live use.

## Step 9: Regime Detection and Regime-Aware ML

Step 9 adds market-state detection and dynamic model routing.

### Files Added

```text
features/
`-- regimes.py
models/
`-- regime_aware.py
scripts/
`-- train_regime_xgboost.py
tests/
`-- test_regimes.py
```

### Why One Model Fails Across All Conditions

Markets are not stationary. A model trained across all periods may average together incompatible behaviors:

- Momentum can work in persistent trend regimes.
- Mean reversion can work in range-bound regimes.
- Breakouts can fail during low-liquidity chop.
- Volatility spikes can change transaction costs, stop distances, and signal reliability.

Regime-aware ML asks: what kind of market are we in, and which model is appropriate for that state?

### HMM Regime Detection

Hidden Markov Models infer latent states from returns:

```python
from features import RegimeDetector

returns = feature_frame["close"].pct_change()
hmm_labels = RegimeDetector().hmm_regime(returns)
```

Why this matters: HMMs model markets as switching hidden states, often corresponding to calm, stressed, and transition periods.

`hmmlearn` is lazy-loaded, so install it before using HMM:

```bash
pip install hmmlearn
```

### K-Means Regime Detection

K-Means clusters:

```text
[rolling_vol, trend_strength, volume_ratio]
```

```python
kmeans_labels = RegimeDetector().kmeans_regime(feature_frame)
```

Why this matters: clustering gives a simple, auditable way to group similar market states without target labels.

### ADX Trend/Range Classifier

ADX is used as a transparent rule-based regime classifier:

```python
adx_labels = RegimeDetector().adx_regime(feature_frame)
```

Default rules:

```text
ADX > 25 = trending
ADX < 20 = ranging
otherwise = transition
```

Why this matters: ADX does not predict direction. It measures trend strength. That makes it useful for deciding whether trend-following or mean-reversion logic should dominate.

### Add All Regimes

```python
from features import RegimeDetector

regime_frame = RegimeDetector().add_all_regimes(
    feature_frame,
    include_hmm=False,
)
```

This adds:

- `regime_rolling_vol`
- `regime_trend_strength`
- `regime_volume_ratio`
- `kmeans_regime`
- `adx`
- `adx_regime`

### Regime-Aware XGBoost

The router trains:

- A global fallback model
- One XGBoost model per regime when enough samples exist

```python
from models import RegimeAwareXGBoostModel

router = RegimeAwareXGBoostModel(
    task="regression",
    min_regime_samples=30,
)

router.fit(
    x_train,
    y_train,
    train_regimes=regime_labels_train,
    x_valid=x_valid,
    y_valid=y_valid,
    valid_regimes=regime_labels_valid,
)

predictions = router.predict(test_x, test_regimes)
```

At inference time:

```text
detect regime -> route row to matching model -> fallback if unseen regime
```

### Command-Line Training

```bash
python scripts/train_regime_xgboost.py \
  --input data/processed/SPY_features.csv \
  --regime-column adx_regime
```

If `adx_regime` is missing, the script creates regime labels first.

### Production Intuition

Regime-aware modeling is not magic. It can overfit if regimes are unstable or too finely sliced. The goal is not to create dozens of tiny models; the goal is to stop forcing one model to explain market states with genuinely different mechanics.

## Step 10: Alternative Data Engine

Step 10 adds optional non-price data sources and point-in-time merging.

### Files Added

```text
data/
`-- alternative_data.py
scripts/
`-- merge_alternative_data.py
tests/
`-- test_alternative_data.py
```

### Why Alternative Data Matters

Prices and volume tell us what traded. Alternative data can help explain why behavior is changing:

- News sentiment can capture institutional narrative shifts.
- Reddit sentiment can proxy retail attention and crowding.
- Google Trends can proxy search interest and retail momentum.
- Macro data can capture inflation, growth, policy, and volatility conditions.

The danger: alternative data is easy to leak. Every item must be aligned by the timestamp when it was actually available.

### FinBERT News Sentiment

```python
from data import FinBERTSentimentScorer

scored_news = FinBERTSentimentScorer().score_frame(
    news_frame,
    text_column="headline",
)
```

The scorer returns:

- `finbert_label`
- `finbert_confidence`
- `finbert_score`

Why this matters: FinBERT is trained for financial language, where words like "liability," "beat," or "downgrade" carry domain-specific meaning.

### Reddit Sentiment

```python
from data import RedditSentimentProcessor

reddit_scores = RedditSentimentProcessor().score_vader(reddit_frame)
```

The processor combines title/comment text and can score with:

- VADER for lightweight social sentiment
- FinBERT for finance-aware text classification

Why this matters: retail attention can affect short-horizon flows, especially in crowded single names and crypto.

### Google Trends

```python
from data import GoogleTrendsCollector

trends = GoogleTrendsCollector().fetch_interest(["bitcoin", "ethereum"])
```

Why this matters: search interest can proxy retail attention before or during momentum bursts.

### Macro Features

```python
from data import MacroDataCollector

macro = MacroDataCollector(api_key="YOUR_FRED_KEY").fetch_fred_series(
    start="2018-01-01",
)
```

Default FRED series:

- `macro_cpi`
- `macro_pmi`
- `macro_fed_funds`
- `macro_vix`

Why this matters: factor behavior changes with inflation, growth, rates, and market stress.

### Timestamp-Safe Merge

```python
from data import AlternativeDataMerger

merged = AlternativeDataMerger().merge_many(
    market_frame,
    [news_sentiment, macro, trends],
    tolerance="7D",
)
```

The merge uses backward `merge_asof`: each market bar receives the latest alternative observation available at or before the bar timestamp.

### 3-Day Sentiment Momentum

The merger creates rolling sentiment momentum features:

```text
finbert_score_momentum_3d
vader_compound_momentum_3d
```

This captures whether sentiment is improving or deteriorating, not just whether it is positive or negative.

### Command-Line Merge

```bash
python scripts/merge_alternative_data.py \
  --market data/processed/SPY_features.csv \
  --alternative data/external/news_sentiment.csv data/external/macro.csv \
  --output data/processed/SPY_features_alt.csv \
  --tolerance 7D
```

### Verify Contribution with SHAP

After training a model with alternative features:

```python
from data import verify_alternative_data_contribution

alt_importance = verify_alternative_data_contribution(shap_importance)
```

Why this matters: alternative data is expensive and noisy. If SHAP shows that sentiment, trends, or macro features contribute nothing out of sample, they should not stay in the production model.

## Step 11: Bayesian Optimization Engine

Step 11 adds Optuna-based hyperparameter optimization for XGBoost and LSTM models.

### Files Added

```text
optimization/
|-- __init__.py
`-- optuna_tuning.py
scripts/
|-- optimize_xgboost.py
`-- optimize_lstm.py
tests/
`-- test_optuna_tuning.py
```

### Why Bayesian Optimization Matters

Grid search wastes trials because it tests every combination mechanically. Financial ML search spaces are large:

- Tree depth
- Learning rate
- Regularization
- Sampling ratios
- LSTM hidden size
- Dropout
- Batch size

Bayesian optimization uses previous trial results to choose the next promising region. Optuna's TPE sampler is often much more efficient than grid search when each trial is expensive.

### Objective: Validation IC

The optimization objective is mean validation Information Coefficient:

```text
objective = mean(IC(predictions, realized_returns)) across TimeSeriesSplit folds
```

Why this matters: a trading model should rank future returns well. Random CV error is the wrong target because it leaks regimes and optimizes a statistical score that may not map to alpha.

### XGBoost Search Space

The project searches:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

```python
from optimization import run_xgboost_study

study, result = run_xgboost_study(
    x=X,
    y=y,
    realized_returns=realized_returns,
    n_trials=100,
    objective_kwargs={
        "n_splits": 5,
        "purge_bars": 5,
        "embargo_bars": 5,
    },
)
```

### LSTM Search Space

The project searches:

- `num_layers`
- `hidden_size`
- `dropout`
- `learning_rate`
- `batch_size`

```python
from optimization import run_lstm_study

study, result = run_lstm_study(
    x=X,
    y=y,
    n_trials=100,
    objective_kwargs={
        "sequence_length": 60,
        "n_splits": 5,
        "epochs": 100,
    },
)
```

### MedianPruner

Optuna's `MedianPruner` stops weak trials early when intermediate fold IC is poor.

Why this matters: in quant research, one careless optimization run can burn hours. Pruning lets the platform spend more time on promising configurations.

### Command-Line XGBoost Optimization

```bash
python scripts/optimize_xgboost.py \
  --input data/processed/SPY_features_alt.csv \
  --trials 100 \
  --n-splits 5 \
  --purge-bars 5 \
  --embargo-bars 5
```

### Command-Line LSTM Optimization

```bash
python scripts/optimize_lstm.py \
  --input data/processed/SPY_features_alt.csv \
  --trials 100 \
  --sequence-length 60 \
  --epochs 100
```

### Important Guardrail

Optimization is where overfitting often sneaks in wearing a nice suit. Use TimeSeriesSplit, purging, embargo, and out-of-sample walk-forward validation. A tuned model is not trustworthy until it survives Step 13.

## Step 12: Ensemble Framework

Step 12 combines model predictions into a more robust signal.

### Files Added

```text
models/
`-- ensemble.py
scripts/
`-- run_ensemble.py
tests/
`-- test_ensemble.py
```

### Why Ensembles Matter

No single model owns every market condition. XGBoost may dominate tabular nonlinear relationships, LSTM may capture sequence patterns, and a linear baseline can remain surprisingly hard to beat when markets are noisy.

An ensemble asks:

```text
Which model has been useful recently, and how should its signal be weighted?
```

### Linear Baseline

```python
from models import LinearRegressionBaseline

baseline = LinearRegressionBaseline().fit(train_x, train_y)
linear_predictions = baseline.predict(test_x)
```

Why this matters: every complex ML model should beat a simple statistical baseline after costs. If it cannot, complexity is not justified.

### IC-Weighted Ensemble

```python
from models import ic_weighted_ensemble

predictions = pd.DataFrame({
    "xgboost": xgb_predictions,
    "lstm": lstm_predictions,
    "linear_baseline": linear_predictions,
})

ensemble, weights = ic_weighted_ensemble(
    predictions,
    realized_returns,
    window=20,
)
```

Weights are based on shifted rolling 20-period validation IC. The shift matters: the ensemble weight at time `t` uses only prior realized outcomes.

### Ridge Stacking

```python
from models import RidgeStackingEnsemble

stacker = RidgeStackingEnsemble(alpha=1.0)
stacker.fit(validation_predictions, validation_returns)
stacked_predictions = stacker.predict(test_predictions)
```

Why this matters: stacking learns a meta-model over base predictions. Ridge regularization helps avoid unstable weights when base models are correlated.

### Regime-Conditional Weights

```python
from models import regime_conditional_weights, regime_conditional_ensemble

weights = regime_conditional_weights(
    predictions,
    realized_returns,
    regimes,
)

ensemble = regime_conditional_ensemble(
    predictions,
    regimes,
    weights,
)
```

Why this matters: a model that works in trending markets may fail in ranging markets. Regime-conditional weights let the ensemble adapt without retraining every base model.

### Evaluate Ensemble vs Individual Models

```python
from models import evaluate_prediction_signals

evaluation = evaluate_prediction_signals(
    predictions,
    realized_returns,
    ensemble,
    ensemble_name="ic_weighted_ensemble",
    transaction_cost_bps=2.0,
)
```

The evaluation table includes:

- Individual model IC
- Ensemble IC
- Individual model Sharpe
- Ensemble Sharpe
- Max Drawdown

### Command-Line Ensemble

```bash
python scripts/run_ensemble.py \
  --input results/predictions.csv \
  --prediction-columns xgboost lstm linear_baseline \
  --realized-column realized_return \
  --method ic_weighted \
  --output results/ensemble/ensemble_predictions.csv \
  --metrics-output results/ensemble/ensemble_metrics.csv
```

For regime-conditional weights:

```bash
python scripts/run_ensemble.py \
  --input results/predictions.csv \
  --prediction-columns xgboost lstm linear_baseline \
  --realized-column realized_return \
  --method regime_conditional \
  --regime-column adx_regime \
  --output results/ensemble/regime_ensemble_predictions.csv
```

### Important Guardrail

Ensembles can overfit too. IC weights, stacking weights, and regime weights should be learned on validation or walk-forward windows, then evaluated on out-of-sample data only.

## Step 13: Walk-Forward Research Pipeline

Step 13 adds realistic out-of-sample model evaluation.

### Files Added

```text
evaluation/
`-- walk_forward.py
scripts/
`-- walk_forward_xgboost.py
tests/
`-- test_walk_forward.py
```

### Why Walk-Forward Is the Real Test

A single train/test split can accidentally flatter a model. Walk-forward evaluation simulates how a research desk would operate:

1. Train on historical data available so far.
2. Predict the next out-of-sample window.
3. Slide forward.
4. Repeat until the dataset ends.
5. Evaluate only the concatenated out-of-sample predictions.

That final point is the key. Metrics are computed on OOS predictions only, not on training or validation windows.

### Calendar Setup

Default setup:

```text
initial train = 2 years
test window = 3 months
```

```python
from evaluation import WalkForwardConfig, run_walk_forward_model

config = WalkForwardConfig(
    initial_train_years=2,
    test_months=3,
    validation_fraction=0.2,
)
```

### Generic Walk-Forward Runner

```python
from models import XGBoostFactorModel

result = run_walk_forward_model(
    x=X,
    y=y,
    model_factory=lambda: XGBoostFactorModel("regression"),
    config=config,
    realized_returns=realized_returns,
    rolling_ic_window=20,
)
```

The result contains:

- `predictions`: concatenated OOS predictions
- `metrics`: OOS IC, Sharpe, Max Drawdown
- `rolling_ic`: rolling OOS IC
- `windows`: train/test window metadata

### Command-Line XGBoost Walk-Forward

```bash
python scripts/walk_forward_xgboost.py \
  --input data/processed/SPY_features_alt.csv \
  --initial-train-years 2 \
  --test-months 3
```

Outputs:

- `results/walk_forward/oos_predictions.csv`
- `results/walk_forward/windows.csv`
- `results/walk_forward/metrics.json`
- `results/walk_forward/rolling_ic.csv`
- `results/walk_forward/retrain_triggers.csv`
- `results/walk_forward/rolling_ic.html` if Plotly is installed

### Rolling IC and Alpha Decay

```python
from evaluation import plot_rolling_ic

figure = plot_rolling_ic(result.rolling_ic, threshold=0.02)
```

Why this matters: alpha decays. A model that worked six months ago may stop working when positioning, volatility, or market structure changes.

### Feature Drift Detection

```python
from evaluation import feature_drift_report

drift = feature_drift_report(
    train_features,
    live_or_test_features,
    psi_threshold=0.2,
)
```

The report uses Population Stability Index. A high PSI means the feature distribution has shifted away from what the model saw during training.

### Auto-Retrain Trigger

```python
from evaluation import retrain_triggers

triggers = retrain_triggers(
    result.rolling_ic,
    ic_threshold=0.02,
    consecutive_periods=2,
)
```

Why this matters: retraining should be governed by evidence, not anxiety. Rolling IC decay below a threshold is a clean retraining signal.

### Important Guardrail

Walk-forward validation is still research, not live proof. It is the closest offline approximation of production behavior, but Step 16 and Step 19 will add live monitoring so the model can be judged after deployment too.

## Step 14 - Factor Risk Modeling

Step 14 asks a harder question than raw performance: what risks produced the returns?
Institutional investors need to know whether a strategy is true alpha or just disguised exposure to equity beta, value, size, momentum, volatility, or correlated positions.

### Core API

```python
from evaluation import (
    beta_to_benchmark,
    covariance_matrix,
    fama_french_exposures,
    jensens_alpha,
    return_attribution,
    volatility_factor_correlation,
)

beta = beta_to_benchmark(strategy_returns, spy_returns)
alpha = jensens_alpha(strategy_returns, spy_returns)
exposures = fama_french_exposures(strategy_returns, factor_returns)
attribution = return_attribution(strategy_returns, factor_returns)
vix_corr = volatility_factor_correlation(strategy_returns, vix_levels)
covariance = covariance_matrix(asset_returns)
```

### Command-Line Factor Risk Report

```bash
python scripts/factor_risk_report.py \
  --input results/walk_forward/oos_predictions.csv \
  --strategy-column strategy_return \
  --benchmark-column SPY \
  --factor-columns MKT SMB HML UMD \
  --vix-column VIX \
  --asset-return-columns SPY QQQ GLD BTC \
  --output-dir results/factor_risk
```

Outputs:

- `summary.json`: benchmark beta, Jensen's alpha, R-squared, VIX sensitivity
- `factor_exposures.csv`: regression betas and correlations to MKT, SMB, HML, UMD
- `return_attribution.csv`: factor contribution vs alpha plus residual contribution
- `factor_residuals.csv`: unexplained return stream after factor adjustment
- `covariance_matrix.csv`: annualized covariance matrix for portfolio risk

### Why It Matters

Beta measures market exposure. Jensen's alpha estimates return after benchmark adjustment. Fama-French-style exposures show whether the strategy is loading on known compensated factors rather than discovering new alpha. VIX correlation reveals hidden short-volatility behavior. Covariance tells the portfolio engine how positions interact instead of treating each signal as isolated.

## Step 15 - Trading Integration and Portfolio Evaluation

Step 15 converts model scores into tradable portfolio behavior. This matters because prediction accuracy alone is not a trading result: a model can be directionally right but still lose money after costs, oversized losses, weak payoff asymmetry, or excessive turnover.

### Core API

```python
from evaluation import (
    ExecutionCostConfig,
    atr_position_size,
    backtest_signal_portfolio,
    prediction_to_signal,
    signal_strength_weights,
)

signals = prediction_to_signal(predictions["SPY"], buy_threshold=0.001)
units = atr_position_size(
    account_value=100000,
    atr=ohlcv["atr_14"],
    price=ohlcv["close"],
    risk_per_trade=0.01,
)
weights = signal_strength_weights(prediction_matrix, max_position_weight=0.05)
result = backtest_signal_portfolio(
    prediction_matrix,
    realized_return_matrix,
    threshold=0.001,
    cost_config=ExecutionCostConfig(slippage_bps=2.0, spread_bps=1.0),
)
```

### Command-Line Portfolio Backtest

```bash
python scripts/backtest_portfolio.py \
  --input results/predictions.csv \
  --prediction-columns SPY_pred QQQ_pred GLD_pred \
  --return-columns SPY QQQ GLD \
  --asset-names SPY QQQ GLD \
  --benchmark-column SPY \
  --threshold 0.001 \
  --max-position-weight 0.05 \
  --max-gross-leverage 1.0 \
  --slippage-bps 2 \
  --spread-bps 1 \
  --output-dir results/portfolio
```

Outputs:

- `portfolio_returns.csv`: gross return, execution cost, net strategy return, equity curve, turnover
- `target_weights.csv`: signal-strength-weighted portfolio allocations
- `metrics.json`: Sharpe, Max Drawdown, CAGR, Calmar, Information Ratio, IC, Profit Factor, Win Rate
- `<asset>_signals.csv`: BUY/SELL/HOLD labels for the first asset

### Finance Intuition

Signal thresholds reduce churn from weak predictions. ATR sizing ties trade size to stop distance, so volatile assets receive smaller unit exposure. Signal-strength weighting gives more capital to stronger forecasts while max-position and leverage caps prevent a single attractive score from dominating the book. Slippage and spread costs make the backtest closer to executable trading instead of a frictionless research fantasy.

## Step 16 - Real-Time Market Analysis and Live Signal Engine

Step 16 introduces the live market-data path. The goal is not just to receive prices, but to keep a clean rolling state, recompute point-in-time features, evaluate multi-timeframe context, and route the latest feature vector through trained models.

### Core API

```python
from live import (
    BinanceKlineStreamer,
    LiveFeatureEngine,
    LiveInferenceEngine,
    LiveSignalConfig,
    MultiTimeframeConfluence,
)

feature_engine = LiveFeatureEngine.create(max_bars=200)
confluence_engine = MultiTimeframeConfluence()
inference_engine = LiveInferenceEngine(
    xgboost_model=xgb_model,
    lstm_model=lstm_model,
    regime_detector=regime_model,
    config=LiveSignalConfig(buy_threshold=0.001, confidence_scale=0.01),
)
```

### Command-Line Live Runner

```bash
python scripts/run_live_signals.py \
  --symbols BTCUSDT ETHUSDT \
  --intervals 1h 15m 5m \
  --xgboost-model results/models/xgboost.pkl \
  --lstm-model results/models/lstm.keras \
  --regime-model results/models/regime.pkl \
  --entry-interval 5m \
  --output-jsonl data/live/signals.jsonl
```

When models are supplied, the runner emits JSON lines with:

- `symbol`
- `timestamp`
- `signal`: BUY, SELL, or HOLD
- `confidence`
- `prediction`
- `regime`
- per-model predictions
- H1/M15/M5 confluence votes

### Components

- `KlineBar`: normalized Binance OHLCV event
- `RollingOHLCVBuffer`: capped 200-bar buffer per symbol and timeframe
- `LiveFeatureEngine`: incremental feature refresh using the Step 3 feature engine
- `MultiTimeframeConfluence`: H1 EMA trend, M15 RSI/MACD confirmation, M5 volume/MACD entry
- `LiveInferenceEngine`: combines XGBoost, LSTM, and regime filters into a confidence-weighted signal
- `BinanceKlineStreamer`: async Binance WebSocket client with reconnection logic

### Finance Intuition

Low-latency trading systems are stateful. A single incoming bar only matters because it updates rolling volatility, momentum, volume pressure, and regime context. Multi-timeframe confluence reduces noisy entries by requiring higher-timeframe trend and lower-timeframe confirmation to agree. Regime filters are a final guardrail: some markets are tradable, and some are conditions where the best signal is no signal.

## Step 17 - Execution Engine and Institutional Risk Management

Step 17 adds the final gate before a signal becomes an order. A professional trading system must be able to reject trades, resize trades, halt trading, and flatten exposure when risk limits are breached.

### Core API

```python
from live import (
    ExecutionConfig,
    ExecutionSimulator,
    InstitutionalRiskManager,
    Order,
    OrderSide,
    OrderType,
)

simulator = ExecutionSimulator(
    ExecutionConfig(slippage_bps=2.0, market_impact_bps=1.0)
)
risk_manager = InstitutionalRiskManager(
    ExecutionConfig(max_daily_drawdown=-0.03, max_position_weight=0.05, max_leverage=2.0)
)

decision = risk_manager.evaluate_order(
    account_equity=100000,
    proposed_notional=10000,
    confidence=0.75,
    current_volatility=0.04,
    reference_volatility=0.02,
)
```

### Order Simulation

Supported order types:

- `MARKET`
- `LIMIT`
- `STOP_LOSS`
- `TAKE_PROFIT`
- `TRAILING_STOP`

Market fills occur at the bar open with adverse slippage and market impact. Limit and stop orders fill only when the bar trades through the trigger. Trailing stops update their stop level as price moves favorably, then trigger when price reverses.

### Command-Line Execution Simulation

```bash
python scripts/simulate_execution.py \
  --bars data/processed/BTCUSDT_5m.csv \
  --orders results/orders.csv \
  --slippage-bps 2 \
  --market-impact-bps 1 \
  --output-fills results/execution/fills.csv \
  --output-positions results/execution/positions.csv
```

Order CSV columns:

- `timestamp`
- `order_id`
- `symbol`
- `side`: BUY or SELL
- `order_type`: MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP
- `quantity`
- optional: `limit_price`, `stop_price`, `take_profit_price`, `trailing_distance`

### Risk Controls

- Volatility-adjusted exposure scales down in high-volatility regimes.
- Confidence-weighted exposure reduces size when ensemble agreement is weak.
- Max daily drawdown halts new orders after a configured loss threshold.
- Max position weight prevents concentration in one symbol.
- Max leverage prevents gross exposure from exceeding institutional limits.
- Kill-switch halts all new orders and can be paired with `close_all_positions`.
- Correlation checks reject new positions that duplicate existing risk.

### Finance Intuition

Execution quality can decide whether a strategy survives. A good model with bad order handling can bleed alpha through slippage, spread, and market impact. Risk controls are not decoration; they are the difference between a research signal and a managed trading business.

## Step 18 - Market Microstructure and Order-Flow Analytics

Step 18 adds features that look beneath OHLCV bars. Instead of only asking where price closed, these analytics ask who was aggressive, where visible liquidity leaned, whether price swept prior liquidity, and how far price stretched from VWAP.

### Core API

```python
from features import (
    aggregate_order_book_imbalance,
    build_microstructure_features,
    cumulative_delta,
    delta_divergence,
    liquidity_sweep_detection,
    order_flow_pressure_signal,
    volume_delta,
    vwap_standard_deviation_bands,
)

imbalance = aggregate_order_book_imbalance(order_book_snapshots)
pressure = order_flow_pressure_signal(imbalance)
delta = volume_delta(trades)
cum_delta = cumulative_delta(trades)
divergence = delta_divergence(ohlcv["close"], cum_delta)
sweeps = liquidity_sweep_detection(ohlcv)
bands = vwap_standard_deviation_bands(ohlcv)
features = build_microstructure_features(ohlcv, trades=trades, order_book=order_book)
```

### Command-Line Feature Build

```bash
python scripts/build_microstructure_features.py \
  --ohlcv data/processed/BTCUSDT_5m.csv \
  --trades data/live/BTCUSDT_trades.csv \
  --order-book data/live/BTCUSDT_book.csv \
  --lookback 20 \
  --output data/processed/BTCUSDT_microstructure.csv
```

Expected trade columns:

- `timestamp`
- `price`
- `bid`
- `ask`
- `quantity`

Expected order book columns:

- `timestamp`
- one or more `bid_qty*` columns
- one or more `ask_qty*` columns

### Signals

- Order book imbalance: `(bid_qty - ask_qty) / (bid_qty + ask_qty)`
- Volume delta: buy-initiated volume minus sell-initiated volume
- Cumulative delta: running sum of volume delta by session
- Bullish sweep: price breaks a recent low, closes back above it, and volume spikes
- Bearish sweep: price breaks a recent high, closes back below it, and volume spikes
- VWAP deviation: `(price - VWAP) / ATR`
- VWAP bands: 1, 2, and 3 standard-deviation reference levels

### Finance Intuition

Order book imbalance above `0.6` suggests strong visible bid pressure. Delta divergence is more subtle: if price makes a new high while cumulative delta fails to confirm, aggressive buying may be weakening. VWAP bands give an institutional reference point because many execution desks judge whether price is rich or cheap relative to volume-weighted participation.

## Step 19 - Dashboard, Alerting, and Model Monitoring

Step 19 turns the platform into an operating system for live model oversight. A deployed model needs more than predictions: it needs signal visibility, rolling performance checks, drift alerts, retraining triggers, and an audit trail.

### Core Monitoring API

```python
from evaluation import build_monitoring_snapshot

snapshot = build_monitoring_snapshot(
    live_predictions,
    reference_features=train_features,
    current_features=live_features,
    rolling_ic_window=20,
    ic_threshold=0.02,
    psi_threshold=0.2,
)

snapshot.metrics
snapshot.alerts
snapshot.drift_report
snapshot.retrain_report
```

### Monitoring Report

```bash
python scripts/monitor_live_model.py \
  --live-predictions data/live/predictions.csv \
  --reference-features data/processed/train_features.csv \
  --current-features data/live/current_features.csv \
  --rolling-ic-window 20 \
  --ic-threshold 0.02 \
  --psi-threshold 0.2 \
  --output-dir results/monitoring
```

Outputs:

- `rolling_ic.csv`
- `drift_report.csv`
- `retrain_report.csv`
- `metrics.json`
- `alerts.txt`

Add `--log-mlflow` to log monitoring metrics and alert counts to the configured MLflow experiment.

### Streamlit Dashboard

```bash
streamlit run scripts/run_dashboard.py
```

Dashboard panels include:

- Latest live signal and confidence
- Portfolio equity curve
- Drawdown chart
- Rolling IC with retrain threshold
- SHAP top-5 feature importance panel

### Alerts

```python
from live import DiscordWebhookAlerter, TelegramAlerter, signal_alert_message

message = signal_alert_message(live_signal, risk_info="max position 5%, leverage OK")
TelegramAlerter(bot_token, chat_id).send(message)
DiscordWebhookAlerter(webhook_url).send(message)
```

Telegram and Discord calls are lazy-loaded and can be disabled in config. This keeps offline research and tests independent of credentials.

### Finance Intuition

Rolling IC monitors whether alpha is still ranking returns correctly. PSI drift detects whether live feature distributions have moved away from the training regime. A retraining trigger should fire when IC decays or drift becomes material, not just because the model had a bad day. SHAP top features help operators see whether live decisions are being driven by sensible signals or by suspicious feature behavior.

## Project Status

Steps 1 through 19 are now implemented as a complete institutional-style research, trading, execution, monitoring, and dashboard foundation. The remaining work in a real deployment would be connecting broker/exchange credentials, filling production data stores, hardening infrastructure, and validating the system in paper trading before capital is at risk.
