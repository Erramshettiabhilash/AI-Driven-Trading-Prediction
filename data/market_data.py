"""Market data collectors for historical OHLCV research datasets.

The classes in this module intentionally focus on data access only. Cleaning,
normalization, outlier handling, and timestamp alignment live in
``data.preprocessing`` so raw vendor downloads can be preserved separately from
research-ready datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
YFINANCE_COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "volume": "volume",
}


class MarketDataError(RuntimeError):
    """Raised when a market data provider returns unusable data."""


def _safe_symbol(symbol: str) -> str:
    """Return a filesystem-safe symbol name for saving local files."""
    return symbol.replace("/", "_").replace("=", "_").replace("-", "_").upper()


def _ensure_utc_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a sorted UTC ``DatetimeIndex``."""
    output = frame.copy()

    if not isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, utc=True)
    elif output.index.tz is None:
        output.index = output.index.tz_localize("UTC")
    else:
        output.index = output.index.tz_convert("UTC")

    output.index.name = "timestamp"
    return output.sort_index()


@dataclass(frozen=True)
class YahooFinanceCollector:
    """Download historical OHLCV data from Yahoo Finance through ``yfinance``.

    Yahoo Finance is convenient for research across equities, ETFs, futures,
    forex pairs, and crypto. It is not a perfect institutional data source, but
    it is useful for learning and for building the platform plumbing before
    connecting premium point-in-time data.
    """

    auto_adjust: bool = False
    actions: bool = False
    progress: bool = False

    def download_symbol(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
        asset_class: str | None = None,
    ) -> pd.DataFrame:
        """Download one symbol and return standardized UTC-indexed OHLCV data.

        Args:
            symbol: Yahoo Finance ticker, e.g. ``SPY``, ``GC=F``,
                ``EURUSD=X``, or ``BTC-USD``.
            start: Inclusive start date accepted by yfinance.
            end: Optional exclusive end date accepted by yfinance.
            interval: Bar interval such as ``1d``, ``1h``, or ``15m``.
            asset_class: Optional asset class label stored with the data.

        Returns:
            A DataFrame indexed by UTC timestamp with lowercase OHLCV columns
            and metadata columns ``symbol``, ``source``, and ``asset_class``.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MarketDataError("Install yfinance to download Yahoo Finance data.") from exc

        frame = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            actions=self.actions,
            progress=self.progress,
        )

        if frame.empty:
            raise MarketDataError(f"Yahoo Finance returned no data for {symbol}.")

        return self._standardize_yfinance_frame(frame, symbol=symbol, asset_class=asset_class)

    def download_many(
        self,
        symbols: Iterable[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
        asset_class: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Download several Yahoo Finance symbols into a symbol-to-frame mapping."""
        return {
            symbol: self.download_symbol(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                asset_class=asset_class,
            )
            for symbol in symbols
        }

    def _standardize_yfinance_frame(
        self,
        frame: pd.DataFrame,
        symbol: str,
        asset_class: str | None,
    ) -> pd.DataFrame:
        """Convert a yfinance response into the platform's canonical schema."""
        output = frame.copy()

        if isinstance(output.columns, pd.MultiIndex):
            output.columns = output.columns.get_level_values(0)

        output.columns = [
            YFINANCE_COLUMN_MAP.get(str(column).strip().lower(), str(column).strip().lower())
            for column in output.columns
        ]

        missing_columns = [column for column in OHLCV_COLUMNS if column not in output.columns]
        if missing_columns:
            raise MarketDataError(f"{symbol} is missing required columns: {missing_columns}")

        output = _ensure_utc_datetime_index(output)
        output["symbol"] = symbol
        output["source"] = "yfinance"
        output["asset_class"] = asset_class or "unknown"

        numeric_columns = [column for column in ["open", "high", "low", "close", "adj_close", "volume"] if column in output]
        output[numeric_columns] = output[numeric_columns].apply(pd.to_numeric, errors="coerce")
        return output


@dataclass(frozen=True)
class BinanceHistoricalCollector:
    """Download historical crypto klines from Binance REST via ``python-binance``."""

    api_key: str | None = None
    api_secret: str | None = None

    def download_klines(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Download Binance historical klines and return standardized OHLCV data.

        Args:
            symbol: Binance symbol such as ``BTCUSDT``.
            interval: Binance interval such as ``1h`` or ``1d``.
            start: Start time string accepted by python-binance.
            end: Optional end time string accepted by python-binance.

        Returns:
            A UTC-indexed DataFrame with OHLCV and Binance metadata columns.
        """
        try:
            from binance.client import Client
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MarketDataError("Install python-binance to download Binance data.") from exc

        client = Client(api_key=self.api_key, api_secret=self.api_secret)
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start,
            end_str=end,
        )

        if not klines:
            raise MarketDataError(f"Binance returned no data for {symbol}.")

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        output = pd.DataFrame(klines, columns=columns)
        output["timestamp"] = pd.to_datetime(output["open_time"], unit="ms", utc=True)
        output = output.set_index("timestamp").sort_index()

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        output[numeric_columns] = output[numeric_columns].apply(pd.to_numeric, errors="coerce")
        output["symbol"] = symbol
        output["source"] = "binance"
        output["asset_class"] = "crypto"

        return output.drop(columns=["ignore"])


def save_market_data(frames: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Save market data frames as CSV files and return the created paths.

    CSV is used at this stage to keep the project dependency-light. Later steps
    can switch to Parquet once the data lake interface is formalized.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for symbol, frame in frames.items():
        output_path = root / f"{_safe_symbol(symbol)}.csv"
        frame.to_csv(output_path)
        saved_paths.append(output_path)

    return saved_paths
