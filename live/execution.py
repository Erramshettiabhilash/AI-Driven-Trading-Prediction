"""Execution simulation and institutional risk controls.

The classes in this module are intentionally deterministic. They model the
decision checks an institutional trading stack performs before an order is
allowed to leave the strategy layer: order type simulation, slippage, volatility
scaling, confidence scaling, leverage limits, drawdown halts, kill-switches, and
correlation checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd


class OrderSide(str, Enum):
    """Supported order sides."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported simulated order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(str, Enum):
    """Lifecycle states for simulated orders."""

    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Order:
    """Execution instruction submitted by the strategy or risk engine."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    timestamp: pd.Timestamp
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    trailing_distance: float | None = None
    status: OrderStatus = OrderStatus.OPEN


@dataclass(frozen=True)
class Fill:
    """Executed order fill after simulated slippage and impact."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: pd.Timestamp
    slippage_bps: float
    notional: float


@dataclass
class Position:
    """Open position state for a single symbol."""

    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Return signed market value."""
        return self.quantity * self.last_price

    @property
    def gross_market_value(self) -> float:
        """Return absolute market value."""
        return abs(self.market_value)


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution friction and risk defaults."""

    slippage_bps: float = 2.0
    market_impact_bps: float = 1.0
    max_position_weight: float = 0.05
    max_leverage: float = 2.0
    max_daily_drawdown: float = -0.03
    high_volatility_scale: float = 0.5
    correlation_threshold: float = 0.8
    min_confidence: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    """Result of pre-trade risk approval."""

    approved: bool
    reason: str
    adjusted_notional: float = 0.0


def apply_slippage(price: float, side: OrderSide, slippage_bps: float) -> float:
    """Apply adverse slippage to a fill price."""
    multiplier = 1 + slippage_bps / 10_000 if side == OrderSide.BUY else 1 - slippage_bps / 10_000
    return float(price * multiplier)


def volatility_adjusted_notional(
    base_notional: float,
    current_volatility: float,
    reference_volatility: float,
    high_volatility_scale: float = 0.5,
) -> float:
    """Scale down target notional when volatility is above reference."""
    if reference_volatility <= 0 or not np.isfinite(reference_volatility):
        return float(base_notional)
    if current_volatility <= reference_volatility:
        return float(base_notional)
    return float(base_notional * high_volatility_scale)


def confidence_weighted_notional(
    base_notional: float,
    confidence: float,
    min_confidence: float = 0.0,
) -> float:
    """Scale exposure by ensemble agreement or signal confidence."""
    bounded_confidence = float(np.clip(confidence, 0.0, 1.0))
    if bounded_confidence < min_confidence:
        return 0.0
    return float(base_notional * bounded_confidence)


def daily_drawdown(equity_curve: pd.Series) -> float:
    """Return current daily drawdown from intraday equity observations."""
    if equity_curve.empty:
        return 0.0
    today = equity_curve.index[-1].date() if isinstance(equity_curve.index, pd.DatetimeIndex) else None
    if today is not None:
        todays_equity = equity_curve[equity_curve.index.date == today]
    else:
        todays_equity = equity_curve
    if todays_equity.empty:
        return 0.0
    start_equity = float(todays_equity.iloc[0])
    current_equity = float(todays_equity.iloc[-1])
    return current_equity / start_equity - 1 if start_equity else 0.0


def max_abs_correlation(
    candidate_returns: pd.Series,
    existing_returns: pd.DataFrame,
) -> float:
    """Return max absolute correlation to existing positions."""
    if existing_returns.empty:
        return 0.0
    aligned = existing_returns.join(candidate_returns.rename("candidate"), how="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    correlations = aligned[existing_returns.columns].corrwith(aligned["candidate"]).abs().dropna()
    return float(correlations.max()) if not correlations.empty else 0.0


def passes_correlation_check(
    candidate_returns: pd.Series,
    existing_returns: pd.DataFrame,
    threshold: float = 0.8,
) -> bool:
    """Return whether a new position is sufficiently diversified."""
    return max_abs_correlation(candidate_returns, existing_returns) < threshold


class ExecutionSimulator:
    """Simulate order fills against OHLCV bars."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
        self.open_orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}

    def submit_order(self, order: Order) -> None:
        """Store an open order for later bar processing."""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        self.open_orders[order.order_id] = order

    def process_bar(self, symbol: str, bar: pd.Series) -> list[Fill]:
        """Process all open orders for ``symbol`` against one OHLCV bar."""
        fills: list[Fill] = []
        for order in list(self.open_orders.values()):
            if order.symbol.upper() != symbol.upper() or order.status != OrderStatus.OPEN:
                continue
            updated_order, fill_price = self._fill_price(order, bar)
            self.open_orders[order.order_id] = updated_order
            if fill_price is None:
                continue
            fill = self._create_fill(updated_order, fill_price, pd.Timestamp(bar.name))
            fills.append(fill)
            self._apply_fill(fill)
            self.open_orders.pop(order.order_id, None)
        return fills

    def close_all_positions(self, prices: dict[str, float], timestamp: pd.Timestamp) -> list[Fill]:
        """Create market fills that flatten every open position."""
        fills: list[Fill] = []
        for symbol, position in list(self.positions.items()):
            if position.quantity == 0 or symbol not in prices:
                continue
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            fill_price = apply_slippage(prices[symbol], side, self.config.slippage_bps)
            fill = Fill(
                order_id=f"kill_{symbol}_{timestamp.value}",
                symbol=symbol,
                side=side,
                quantity=abs(position.quantity),
                price=fill_price,
                timestamp=timestamp,
                slippage_bps=self.config.slippage_bps,
                notional=abs(position.quantity) * fill_price,
            )
            fills.append(fill)
            self._apply_fill(fill)
        return fills

    def _fill_price(self, order: Order, bar: pd.Series) -> tuple[Order, float | None]:
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if order.order_type == OrderType.MARKET:
            return order, open_price
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("Limit orders require limit_price.")
            if order.side == OrderSide.BUY and low <= order.limit_price:
                return order, min(open_price, order.limit_price)
            if order.side == OrderSide.SELL and high >= order.limit_price:
                return order, max(open_price, order.limit_price)
            return order, None
        if order.order_type in {OrderType.STOP_LOSS, OrderType.TAKE_PROFIT}:
            trigger = order.stop_price if order.order_type == OrderType.STOP_LOSS else order.take_profit_price
            if trigger is None:
                raise ValueError(f"{order.order_type.value} orders require a trigger price.")
            if order.side == OrderSide.SELL and low <= trigger <= high:
                return order, trigger
            if order.side == OrderSide.BUY and low <= trigger <= high:
                return order, trigger
            return order, None
        if order.order_type == OrderType.TRAILING_STOP:
            return self._trailing_stop_fill(order, high=high, low=low, close=close)
        return order, None

    def _trailing_stop_fill(self, order: Order, high: float, low: float, close: float) -> tuple[Order, float | None]:
        if order.trailing_distance is None or order.trailing_distance <= 0:
            raise ValueError("Trailing stop orders require a positive trailing_distance.")
        if order.side == OrderSide.SELL:
            prior_stop = order.stop_price if order.stop_price is not None else close - order.trailing_distance
            new_stop = max(prior_stop, high - order.trailing_distance)
            updated = replace(order, stop_price=new_stop)
            return (updated, new_stop) if low <= new_stop else (updated, None)
        prior_stop = order.stop_price if order.stop_price is not None else close + order.trailing_distance
        new_stop = min(prior_stop, low + order.trailing_distance)
        updated = replace(order, stop_price=new_stop)
        return (updated, new_stop) if high >= new_stop else (updated, None)

    def _create_fill(self, order: Order, raw_price: float, timestamp: pd.Timestamp) -> Fill:
        total_slippage = self.config.slippage_bps + self.config.market_impact_bps
        fill_price = apply_slippage(raw_price, order.side, total_slippage)
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol.upper(),
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=timestamp,
            slippage_bps=total_slippage,
            notional=order.quantity * fill_price,
        )

    def _apply_fill(self, fill: Fill) -> None:
        signed_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        position = self.positions.get(fill.symbol, Position(symbol=fill.symbol))
        new_quantity = position.quantity + signed_quantity
        if new_quantity == 0:
            self.positions[fill.symbol] = Position(fill.symbol, quantity=0.0, average_price=0.0, last_price=fill.price)
            return
        if np.sign(position.quantity) == np.sign(signed_quantity) or position.quantity == 0:
            old_notional = abs(position.quantity) * position.average_price
            new_notional = fill.quantity * fill.price
            average_price = (old_notional + new_notional) / abs(new_quantity)
        else:
            average_price = position.average_price if np.sign(new_quantity) == np.sign(position.quantity) else fill.price
        self.positions[fill.symbol] = Position(
            fill.symbol,
            quantity=new_quantity,
            average_price=float(average_price),
            last_price=fill.price,
        )


@dataclass
class InstitutionalRiskManager:
    """Pre-trade risk manager with drawdown, leverage, and kill-switch checks."""

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    kill_switch_active: bool = False

    def trigger_kill_switch(self) -> None:
        """Halt new orders immediately."""
        self.kill_switch_active = True

    def reset_kill_switch(self) -> None:
        """Allow trading after a manual operational reset."""
        self.kill_switch_active = False

    def evaluate_order(
        self,
        account_equity: float,
        proposed_notional: float,
        current_positions: dict[str, Position] | None = None,
        equity_curve: pd.Series | None = None,
        confidence: float = 1.0,
        current_volatility: float | None = None,
        reference_volatility: float | None = None,
        candidate_returns: pd.Series | None = None,
        existing_returns: pd.DataFrame | None = None,
    ) -> RiskDecision:
        """Approve, resize, or reject a proposed order."""
        if self.kill_switch_active:
            return RiskDecision(False, "kill_switch_active")
        if account_equity <= 0:
            return RiskDecision(False, "invalid_account_equity")
        if equity_curve is not None and daily_drawdown(equity_curve) <= self.config.max_daily_drawdown:
            return RiskDecision(False, "daily_drawdown_halt")

        adjusted_notional = confidence_weighted_notional(
            abs(proposed_notional),
            confidence=confidence,
            min_confidence=self.config.min_confidence,
        )
        if current_volatility is not None and reference_volatility is not None:
            adjusted_notional = volatility_adjusted_notional(
                adjusted_notional,
                current_volatility=current_volatility,
                reference_volatility=reference_volatility,
                high_volatility_scale=self.config.high_volatility_scale,
            )

        max_position_notional = account_equity * self.config.max_position_weight
        adjusted_notional = min(adjusted_notional, max_position_notional)
        if adjusted_notional <= 0:
            return RiskDecision(False, "zero_after_confidence_or_limits")

        current_gross = sum(position.gross_market_value for position in (current_positions or {}).values())
        if current_gross + adjusted_notional > account_equity * self.config.max_leverage:
            return RiskDecision(False, "max_leverage_exceeded", adjusted_notional=adjusted_notional)

        if candidate_returns is not None and existing_returns is not None:
            if not passes_correlation_check(
                candidate_returns,
                existing_returns,
                threshold=self.config.correlation_threshold,
            ):
                return RiskDecision(False, "correlation_limit_exceeded", adjusted_notional=adjusted_notional)

        return RiskDecision(True, "approved", adjusted_notional=adjusted_notional)
