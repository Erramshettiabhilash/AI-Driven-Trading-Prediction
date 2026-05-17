"""Async Binance kline streaming and rolling OHLCV buffers.

The networking class is intentionally thin and lazy-imports ``python-binance``
so tests can exercise all parsing and buffer logic without opening sockets.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class KlineBar:
    """Normalized OHLCV bar from an exchange kline stream."""

    symbol: str
    interval: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


def parse_binance_kline_message(message: dict) -> KlineBar:
    """Parse a Binance kline WebSocket message into a normalized bar."""
    payload = message.get("data", message)
    if "k" not in payload:
        raise KeyError("Binance kline message is missing the 'k' payload.")
    kline = payload["k"]
    return KlineBar(
        symbol=str(kline["s"]).upper(),
        interval=str(kline["i"]),
        open_time=pd.to_datetime(kline["t"], unit="ms", utc=True),
        close_time=pd.to_datetime(kline["T"], unit="ms", utc=True),
        open=float(kline["o"]),
        high=float(kline["h"]),
        low=float(kline["l"]),
        close=float(kline["c"]),
        volume=float(kline["v"]),
        is_closed=bool(kline["x"]),
    )


class RollingOHLCVBuffer:
    """Maintain a capped OHLCV buffer per ``(symbol, interval)`` pair."""

    def __init__(self, max_bars: int = 200) -> None:
        self.max_bars = max_bars
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    def update(self, bar: KlineBar) -> pd.DataFrame:
        """Insert or replace a bar and return the updated capped frame."""
        key = (bar.symbol.upper(), bar.interval)
        row = pd.DataFrame(
            {
                "open": [bar.open],
                "high": [bar.high],
                "low": [bar.low],
                "close": [bar.close],
                "volume": [bar.volume],
                "close_time": [bar.close_time],
                "is_closed": [bar.is_closed],
            },
            index=pd.DatetimeIndex([bar.open_time], name="timestamp"),
        )
        frame = pd.concat([self._frames.get(key, pd.DataFrame()), row])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index().tail(self.max_bars)
        self._frames[key] = frame
        return frame.copy()

    def get(self, symbol: str, interval: str) -> pd.DataFrame:
        """Return a copy of the current buffer for a symbol and interval."""
        return self._frames.get((symbol.upper(), interval), pd.DataFrame()).copy()

    def latest(self, symbol: str, interval: str) -> pd.Series:
        """Return the latest buffered bar for a symbol and interval."""
        frame = self.get(symbol, interval)
        if frame.empty:
            raise KeyError(f"No buffered bars for {symbol.upper()} {interval}.")
        return frame.iloc[-1]


BarCallback = Callable[[KlineBar], Awaitable[None] | None]


@dataclass
class BinanceKlineStreamer:
    """Subscribe to Binance kline streams with simple reconnection logic."""

    symbols: list[str]
    intervals: list[str]
    on_bar: BarCallback
    reconnect_backoff_seconds: float = 5.0
    max_reconnect_attempts: int | None = None

    async def run(self) -> None:
        """Run the multiplexed kline stream until cancelled or attempts expire."""
        try:
            from binance import AsyncClient, BinanceSocketManager
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("python-binance is required for live Binance streaming.") from exc

        attempts = 0
        while self.max_reconnect_attempts is None or attempts <= self.max_reconnect_attempts:
            client = await AsyncClient.create()
            try:
                streams = [
                    f"{symbol.lower()}@kline_{interval}"
                    for symbol in self.symbols
                    for interval in self.intervals
                ]
                socket_manager = BinanceSocketManager(client)
                async with socket_manager.multiplex_socket(streams) as stream:
                    attempts = 0
                    while True:
                        message = await stream.recv()
                        bar = parse_binance_kline_message(message)
                        result = self.on_bar(bar)
                        if result is not None:
                            await result
            except asyncio.CancelledError:
                raise
            except Exception:
                attempts += 1
                if self.max_reconnect_attempts is not None and attempts > self.max_reconnect_attempts:
                    raise
                await asyncio.sleep(self.reconnect_backoff_seconds)
            finally:
                await client.close_connection()
