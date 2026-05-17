import numpy as np
import pandas as pd

from evaluation import (
    ExecutionCostConfig,
    active_information_ratio,
    atr_position_size,
    backtest_signal_portfolio,
    execution_costs,
    portfolio_performance_summary,
    prediction_to_position,
    prediction_to_signal,
    signal_strength_weights,
)


def sample_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC", name="timestamp")


def test_prediction_to_signal_and_position_maps_thresholds() -> None:
    predictions = pd.Series([-0.02, -0.001, 0.0, 0.003, 0.02], index=range(5))

    signals = prediction_to_signal(predictions, buy_threshold=0.002)
    positions = prediction_to_position(predictions, threshold=0.002)

    assert signals.tolist() == ["SELL", "HOLD", "HOLD", "BUY", "BUY"]
    assert positions.tolist() == [-1.0, 0.0, 0.0, 1.0, 1.0]


def test_atr_position_size_respects_stop_risk_and_position_cap() -> None:
    index = sample_index()[:2]
    atr = pd.Series([2.0, 0.5], index=index)
    price = pd.Series([100.0, 100.0], index=index)

    units = atr_position_size(
        account_value=100_000.0,
        atr=atr,
        price=price,
        risk_per_trade=0.01,
        atr_multiplier=2.0,
        max_position_weight=0.05,
    )

    assert units.iloc[0] == 50.0
    assert units.iloc[1] == 50.0


def test_signal_strength_weights_caps_position_and_leverage() -> None:
    index = sample_index()[:2]
    predictions = pd.DataFrame(
        {
            "A": [0.03, 0.0],
            "B": [-0.01, 0.04],
            "C": [0.02, -0.03],
        },
        index=index,
    )

    weights = signal_strength_weights(
        predictions,
        threshold=0.005,
        max_position_weight=0.2,
        max_gross_leverage=0.5,
    )

    assert weights.abs().max().max() <= 0.2
    assert (weights.abs().sum(axis=1) <= 0.5).all()
    assert weights.loc[index[0], "B"] < 0.0


def test_execution_costs_charges_turnover() -> None:
    turnover = pd.Series([0.1, 0.0, 0.3])
    costs = execution_costs(turnover, ExecutionCostConfig(slippage_bps=2.0, spread_bps=2.0))

    assert np.isclose(costs.iloc[0], 0.1 * 3.0 / 10_000)
    assert costs.iloc[1] == 0.0


def test_backtest_signal_portfolio_outputs_returns_weights_and_metrics() -> None:
    index = sample_index()
    predictions = pd.DataFrame(
        {
            "A": np.linspace(0.01, 0.03, len(index)),
            "B": np.linspace(-0.02, -0.01, len(index)),
        },
        index=index,
    )
    returns = pd.DataFrame(
        {
            "A": np.full(len(index), 0.01),
            "B": np.full(len(index), -0.005),
        },
        index=index,
    )

    result = backtest_signal_portfolio(
        predictions,
        returns,
        threshold=0.001,
        max_position_weight=0.25,
        max_gross_leverage=0.5,
        cost_config=ExecutionCostConfig(slippage_bps=0.0, spread_bps=0.0),
    )

    assert result.returns.name == "strategy_return"
    assert result.target_weights.shape == predictions.shape
    assert result.metrics["final_equity"] > 1.0
    assert "sharpe_ratio" in result.metrics


def test_portfolio_summary_includes_active_information_ratio_and_ic() -> None:
    index = sample_index()
    returns = pd.Series(np.linspace(0.001, 0.003, len(index)), index=index)
    benchmark = pd.Series(np.linspace(0.0, 0.001, len(index)), index=index)
    predictions = pd.Series(np.arange(len(index)), index=index)
    realized = pd.Series(np.arange(len(index)) * 0.001, index=index)

    summary = portfolio_performance_summary(
        returns,
        benchmark_returns=benchmark,
        predictions=predictions,
        realized_returns=realized,
    )

    assert active_information_ratio(returns, benchmark) > 0
    assert summary["information_ratio"] > 0
    assert summary["information_coefficient"] > 0.99
