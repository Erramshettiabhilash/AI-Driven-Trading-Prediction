"""Market microstructure and order-flow analytics.

These features explain what happened inside the candle: whether liquidity was
leaning bid or ask, whether aggressive buyers or sellers dominated trades, and
whether price stretched away from VWAP. They are most useful on intraday data,
but the functions are deterministic and work with any timestamped frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VWAPBands:
    """VWAP and standard-deviation bands."""

    vwap: pd.Series
    upper_1: pd.Series
    lower_1: pd.Series
    upper_2: pd.Series
    lower_2: pd.Series
    upper_3: pd.Series
    lower_3: pd.Series


def order_book_imbalance(
    bid_quantity: pd.Series | np.ndarray | float,
    ask_quantity: pd.Series | np.ndarray | float,
) -> pd.Series:
    """Return order book imbalance: ``(bid_qty - ask_qty) / (bid_qty + ask_qty)``.

    Positive values mean visible bid-side depth is larger than ask-side depth.
    This is not guaranteed alpha, but it is a useful short-horizon pressure
    proxy when combined with trade flow and price response.
    """
    bid = pd.Series(bid_quantity, dtype=float)
    ask = pd.Series(ask_quantity, index=bid.index, dtype=float)
    denominator = (bid + ask).replace(0, np.nan)
    return ((bid - ask) / denominator).fillna(0.0).rename("order_book_imbalance")


def aggregate_order_book_imbalance(
    snapshots: pd.DataFrame,
    bid_quantity_columns: list[str] | None = None,
    ask_quantity_columns: list[str] | None = None,
) -> pd.Series:
    """Aggregate multi-level order book quantities into one imbalance series."""
    bid_columns = bid_quantity_columns or [column for column in snapshots.columns if column.startswith("bid_qty")]
    ask_columns = ask_quantity_columns or [column for column in snapshots.columns if column.startswith("ask_qty")]
    if not bid_columns or not ask_columns:
        raise KeyError("Order book snapshots require bid_qty and ask_qty columns.")
    bid_total = snapshots[bid_columns].astype(float).sum(axis=1)
    ask_total = snapshots[ask_columns].astype(float).sum(axis=1)
    return order_book_imbalance(bid_total, ask_total)


def classify_trade_direction(
    trades: pd.DataFrame,
    price_column: str = "price",
    bid_column: str = "bid",
    ask_column: str = "ask",
) -> pd.Series:
    """Classify trades as buy-initiated ``1`` or sell-initiated ``-1``.

    Trades at or above the ask are treated as buyer-initiated. Trades at or
    below the bid are treated as seller-initiated. Mid-market prints fall back
    to a simple tick rule based on the previous trade price.
    """
    required = {price_column, bid_column, ask_column}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing trade columns: {sorted(missing)}")

    price = trades[price_column].astype(float)
    bid = trades[bid_column].astype(float)
    ask = trades[ask_column].astype(float)
    tick_rule = np.sign(price.diff()).replace(0, np.nan).ffill().fillna(0.0)
    direction = pd.Series(
        np.select([price >= ask, price <= bid], [1.0, -1.0], default=tick_rule),
        index=trades.index,
        name="trade_direction",
    )
    return direction.astype(float)


def volume_delta(
    trades: pd.DataFrame,
    quantity_column: str = "quantity",
    price_column: str = "price",
    bid_column: str = "bid",
    ask_column: str = "ask",
) -> pd.Series:
    """Return signed volume delta for each trade."""
    if quantity_column not in trades:
        raise KeyError(f"Missing trade quantity column: {quantity_column}")
    direction = classify_trade_direction(trades, price_column, bid_column, ask_column)
    return (direction * trades[quantity_column].astype(float)).rename("volume_delta")


def cumulative_delta(
    trades: pd.DataFrame,
    quantity_column: str = "quantity",
    session_column: str | None = None,
) -> pd.Series:
    """Return running cumulative volume delta, optionally reset by session."""
    delta = volume_delta(trades, quantity_column=quantity_column)
    if session_column is not None:
        if session_column not in trades:
            raise KeyError(f"Missing session column: {session_column}")
        return delta.groupby(trades[session_column]).cumsum().rename("cumulative_delta")
    if isinstance(trades.index, pd.DatetimeIndex):
        sessions = trades.index.tz_convert("UTC").date if trades.index.tz is not None else trades.index.date
        return delta.groupby(sessions).cumsum().rename("cumulative_delta")
    return delta.cumsum().rename("cumulative_delta")


def liquidity_sweep_detection(
    ohlcv: pd.DataFrame,
    lookback: int = 20,
    volume_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Detect high/low sweeps with volume confirmation.

    A bullish sweep breaks a recent low and closes back above it. A bearish
    sweep breaks a recent high and closes back below it. Volume confirmation
    filters out many random wick events.
    """
    required = {"high", "low", "close", "volume"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise KeyError(f"Missing OHLCV columns: {sorted(missing)}")
    previous_high = ohlcv["high"].rolling(lookback).max().shift(1)
    previous_low = ohlcv["low"].rolling(lookback).min().shift(1)
    volume_mean = ohlcv["volume"].rolling(lookback).mean().shift(1)
    volume_spike = ohlcv["volume"] > volume_mean * volume_multiplier
    return pd.DataFrame(
        {
            "bullish_liquidity_sweep": (
                (ohlcv["low"] < previous_low) & (ohlcv["close"] > previous_low) & volume_spike
            ).astype(int),
            "bearish_liquidity_sweep": (
                (ohlcv["high"] > previous_high) & (ohlcv["close"] < previous_high) & volume_spike
            ).astype(int),
        },
        index=ohlcv.index,
    )


def anchored_vwap(
    ohlcv: pd.DataFrame,
    price_column: str = "close",
    volume_column: str = "volume",
) -> pd.Series:
    """Return session-anchored VWAP from price and volume."""
    if price_column not in ohlcv or volume_column not in ohlcv:
        raise KeyError(f"Missing {price_column} or {volume_column} for VWAP.")
    price = ohlcv[price_column].astype(float)
    volume = ohlcv[volume_column].astype(float)
    dollar_volume = price * volume
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        sessions = ohlcv.index.tz_convert("UTC").date if ohlcv.index.tz is not None else ohlcv.index.date
        vwap = dollar_volume.groupby(sessions).cumsum() / volume.groupby(sessions).cumsum().replace(0, np.nan)
    else:
        vwap = dollar_volume.cumsum() / volume.cumsum().replace(0, np.nan)
    return vwap.rename("micro_vwap")


def vwap_deviation(
    price: pd.Series,
    vwap: pd.Series,
    atr: pd.Series,
) -> pd.Series:
    """Return ``(price - VWAP) / ATR`` as an institutional reference distance."""
    aligned = pd.concat(
        [price.rename("price"), vwap.rename("vwap"), atr.rename("atr")],
        axis=1,
    )
    return ((aligned["price"] - aligned["vwap"]) / aligned["atr"].replace(0, np.nan)).rename("vwap_deviation_atr")


def vwap_standard_deviation_bands(
    ohlcv: pd.DataFrame,
    window: int = 20,
    price_column: str = "close",
    volume_column: str = "volume",
) -> VWAPBands:
    """Return VWAP plus 1, 2, and 3 standard-deviation bands."""
    vwap = anchored_vwap(ohlcv, price_column=price_column, volume_column=volume_column)
    std = ohlcv[price_column].astype(float).rolling(window, min_periods=2).std(ddof=0)
    return VWAPBands(
        vwap=vwap,
        upper_1=(vwap + std).rename("vwap_upper_1"),
        lower_1=(vwap - std).rename("vwap_lower_1"),
        upper_2=(vwap + 2 * std).rename("vwap_upper_2"),
        lower_2=(vwap - 2 * std).rename("vwap_lower_2"),
        upper_3=(vwap + 3 * std).rename("vwap_upper_3"),
        lower_3=(vwap - 3 * std).rename("vwap_lower_3"),
    )


def delta_divergence(
    close: pd.Series,
    cumulative_delta_series: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Detect price/delta divergence.

    ``-1`` means price is pushing a lookback high while cumulative delta is not
    confirming. ``1`` means price is pushing a lookback low while cumulative
    delta is holding above its prior low.
    """
    prior_high = close.rolling(lookback).max().shift(1)
    prior_low = close.rolling(lookback).min().shift(1)
    prior_delta_high = cumulative_delta_series.rolling(lookback).max().shift(1)
    prior_delta_low = cumulative_delta_series.rolling(lookback).min().shift(1)
    bearish = (close > prior_high) & (cumulative_delta_series < prior_delta_high)
    bullish = (close < prior_low) & (cumulative_delta_series > prior_delta_low)
    return pd.Series(np.select([bullish, bearish], [1, -1], default=0), index=close.index, name="delta_divergence")


def order_flow_pressure_signal(
    imbalance: pd.Series,
    bullish_threshold: float = 0.6,
    bearish_threshold: float = -0.6,
) -> pd.Series:
    """Translate order book imbalance into bullish, bearish, or neutral pressure."""
    return pd.Series(
        np.select([imbalance > bullish_threshold, imbalance < bearish_threshold], [1, -1], default=0),
        index=imbalance.index,
        name="order_flow_pressure",
    )


def build_microstructure_features(
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    order_book: pd.DataFrame | None = None,
    atr_column: str = "atr_14",
    lookback: int = 20,
    volume_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Build Step 18 microstructure features from OHLCV plus optional rich data."""
    output = ohlcv.copy().sort_index()
    bands = vwap_standard_deviation_bands(output, window=lookback)
    output["micro_vwap"] = bands.vwap
    output["vwap_upper_1"] = bands.upper_1
    output["vwap_lower_1"] = bands.lower_1
    output["vwap_upper_2"] = bands.upper_2
    output["vwap_lower_2"] = bands.lower_2
    output["vwap_upper_3"] = bands.upper_3
    output["vwap_lower_3"] = bands.lower_3

    if atr_column in output:
        output["micro_vwap_deviation"] = vwap_deviation(output["close"], output["micro_vwap"], output[atr_column])
    sweeps = liquidity_sweep_detection(output, lookback=lookback, volume_multiplier=volume_multiplier)
    output = output.join(sweeps, rsuffix="_micro")

    if trades is not None and not trades.empty:
        trade_features = pd.DataFrame(
            {
                "volume_delta": volume_delta(trades),
                "cumulative_delta": cumulative_delta(trades),
            },
            index=trades.index,
        )
        resampled = trade_features.resample(_infer_resample_rule(output)).last().reindex(output.index, method="ffill")
        output = output.join(resampled)
        output["delta_divergence"] = delta_divergence(output["close"], output["cumulative_delta"], lookback=lookback)

    if order_book is not None and not order_book.empty:
        imbalance = aggregate_order_book_imbalance(order_book)
        aligned_imbalance = imbalance.resample(_infer_resample_rule(output)).last().reindex(output.index, method="ffill")
        output["order_book_imbalance"] = aligned_imbalance
        output["order_flow_pressure"] = order_flow_pressure_signal(aligned_imbalance)

    return output


def _infer_resample_rule(frame: pd.DataFrame) -> str:
    """Infer a reasonable pandas resample rule from the OHLCV index."""
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        return "1min"
    delta = frame.index.to_series().diff().dropna().median()
    seconds = max(int(delta.total_seconds()), 1)
    return f"{seconds}s"
