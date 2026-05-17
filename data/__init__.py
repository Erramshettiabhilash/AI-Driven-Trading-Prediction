"""Data collection and preprocessing utilities for the quant research platform."""

from data.alternative_data import (
    AlternativeDataMerger,
    FinBERTSentimentScorer,
    GoogleTrendsCollector,
    MacroDataCollector,
    RedditSentimentProcessor,
    VaderSentimentScorer,
    verify_alternative_data_contribution,
)
from data.market_data import BinanceHistoricalCollector, YahooFinanceCollector
from data.preprocessing import DataPreprocessor

__all__ = [
    "AlternativeDataMerger",
    "BinanceHistoricalCollector",
    "DataPreprocessor",
    "FinBERTSentimentScorer",
    "GoogleTrendsCollector",
    "MacroDataCollector",
    "RedditSentimentProcessor",
    "VaderSentimentScorer",
    "YahooFinanceCollector",
    "verify_alternative_data_contribution",
]
