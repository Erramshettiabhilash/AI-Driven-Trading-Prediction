import numpy as np
import pandas as pd

from live import (
    ExecutionConfig,
    ExecutionSimulator,
    InstitutionalRiskManager,
    Order,
    OrderSide,
    OrderType,
    Position,
    apply_slippage,
    confidence_weighted_notional,
    daily_drawdown,
    max_abs_correlation,
    passes_correlation_check,
    volatility_adjusted_notional,
)


def bar(
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 101.0,
) -> pd.Series:
    return pd.Series(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": 1000.0},
        name=pd.Timestamp("2024-01-01 10:00", tz="UTC"),
    )


def test_apply_slippage_is_adverse_by_side() -> None:
    assert apply_slippage(100.0, OrderSide.BUY, 10.0) == 100.1
    assert apply_slippage(100.0, OrderSide.SELL, 10.0) == 99.9


def test_market_order_fills_at_open_with_slippage_and_updates_position() -> None:
    simulator = ExecutionSimulator(ExecutionConfig(slippage_bps=2.0, market_impact_bps=1.0))
    order = Order(
        order_id="m1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2.0,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
    )

    simulator.submit_order(order)
    fills = simulator.process_bar("BTCUSDT", bar(open_price=100.0))

    assert len(fills) == 1
    assert np.isclose(fills[0].price, 100.03)
    assert simulator.positions["BTCUSDT"].quantity == 2.0


def test_limit_order_fills_only_when_bar_trades_through_limit() -> None:
    simulator = ExecutionSimulator(ExecutionConfig(slippage_bps=0.0, market_impact_bps=0.0))
    order = Order(
        order_id="l1",
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        limit_price=98.0,
    )

    simulator.submit_order(order)
    no_fill = simulator.process_bar("ETHUSDT", bar(low=99.0))
    fill = simulator.process_bar("ETHUSDT", bar(open_price=100.0, low=97.0))

    assert no_fill == []
    assert fill[0].price == 98.0


def test_stop_loss_take_profit_and_trailing_stop_fill() -> None:
    simulator = ExecutionSimulator(ExecutionConfig(slippage_bps=0.0, market_impact_bps=0.0))
    simulator.submit_order(
        Order(
            order_id="s1",
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=1.0,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            stop_price=97.0,
        ),
    )
    simulator.submit_order(
        Order(
            order_id="tp1",
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT,
            quantity=1.0,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            take_profit_price=104.0,
        ),
    )
    simulator.submit_order(
        Order(
            order_id="tr1",
            symbol="SPY",
            side=OrderSide.SELL,
            order_type=OrderType.TRAILING_STOP,
            quantity=1.0,
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            trailing_distance=2.0,
        ),
    )

    fills = simulator.process_bar("SPY", bar(high=105.0, low=96.5, close=101.0))
    fill_prices = {fill.order_id: fill.price for fill in fills}

    assert fill_prices["s1"] == 97.0
    assert fill_prices["tp1"] == 104.0
    assert fill_prices["tr1"] == 103.0


def test_close_all_positions_flattens_book() -> None:
    simulator = ExecutionSimulator(ExecutionConfig(slippage_bps=0.0, market_impact_bps=0.0))
    simulator.positions["AAPL"] = Position("AAPL", quantity=10.0, average_price=100.0, last_price=110.0)
    simulator.positions["MSFT"] = Position("MSFT", quantity=-5.0, average_price=200.0, last_price=190.0)

    fills = simulator.close_all_positions({"AAPL": 111.0, "MSFT": 189.0}, pd.Timestamp("2024-01-02", tz="UTC"))

    assert len(fills) == 2
    assert simulator.positions["AAPL"].quantity == 0.0
    assert simulator.positions["MSFT"].quantity == 0.0


def test_volatility_and_confidence_scaling() -> None:
    assert volatility_adjusted_notional(10_000.0, current_volatility=0.04, reference_volatility=0.02) == 5_000.0
    assert volatility_adjusted_notional(10_000.0, current_volatility=0.01, reference_volatility=0.02) == 10_000.0
    assert confidence_weighted_notional(10_000.0, confidence=0.6) == 6_000.0
    assert confidence_weighted_notional(10_000.0, confidence=0.2, min_confidence=0.5) == 0.0


def test_daily_drawdown_uses_current_session_start() -> None:
    index = pd.to_datetime(
        ["2024-01-01 15:00", "2024-01-02 09:30", "2024-01-02 10:00"],
        utc=True,
    )
    equity = pd.Series([100_000.0, 101_000.0, 97_970.0], index=index)

    assert np.isclose(daily_drawdown(equity), -0.03)


def test_risk_manager_rejects_kill_switch_drawdown_leverage_and_correlation() -> None:
    manager = InstitutionalRiskManager(
        ExecutionConfig(max_position_weight=0.05, max_leverage=2.0, max_daily_drawdown=-0.03),
    )
    positions = {"SPY": Position("SPY", quantity=199.6, average_price=100.0, last_price=100.0)}
    equity = pd.Series(
        [100_000.0, 96_900.0],
        index=pd.to_datetime(["2024-01-01 09:30", "2024-01-01 10:00"], utc=True),
    )

    drawdown_decision = manager.evaluate_order(100_000.0, 1_000.0, equity_curve=equity)
    assert not drawdown_decision.approved
    assert drawdown_decision.reason == "daily_drawdown_halt"

    manager.trigger_kill_switch()
    kill_decision = manager.evaluate_order(100_000.0, 1_000.0)
    assert not kill_decision.approved
    assert kill_decision.reason == "kill_switch_active"
    manager.reset_kill_switch()

    leverage_decision = manager.evaluate_order(10_000.0, 1_000.0, current_positions=positions)
    assert not leverage_decision.approved
    assert leverage_decision.reason == "max_leverage_exceeded"

    candidate = pd.Series([1, 2, 3, 4, 5], dtype=float)
    existing = pd.DataFrame({"SPY": [1, 2, 3, 4, 5]}, dtype=float)
    corr_decision = manager.evaluate_order(
        100_000.0,
        1_000.0,
        candidate_returns=candidate,
        existing_returns=existing,
    )
    assert not corr_decision.approved
    assert corr_decision.reason == "correlation_limit_exceeded"


def test_risk_manager_approves_and_resizes_by_confidence_vol_and_position_cap() -> None:
    manager = InstitutionalRiskManager(
        ExecutionConfig(max_position_weight=0.05, high_volatility_scale=0.5),
    )

    decision = manager.evaluate_order(
        account_equity=100_000.0,
        proposed_notional=20_000.0,
        confidence=0.8,
        current_volatility=0.04,
        reference_volatility=0.02,
    )

    assert decision.approved
    assert decision.reason == "approved"
    assert decision.adjusted_notional == 5_000.0


def test_correlation_helpers_measure_existing_overlap() -> None:
    candidate = pd.Series([1, 2, 3, 4, 5], dtype=float)
    existing = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 1, 3, 2, 4],
        },
        dtype=float,
    )

    assert max_abs_correlation(candidate, existing) > 0.99
    assert not passes_correlation_check(candidate, existing, threshold=0.8)
