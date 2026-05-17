"""Alternative data collection, sentiment scoring, and timestamp-safe merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FINBERT_LABEL_MAP = {
    "positive": 1.0,
    "bullish": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
    "bearish": -1.0,
}

DEFAULT_MACRO_SERIES = {
    "cpi": "CPIAUCSL",
    "pmi": "NAPM",
    "fed_funds": "FEDFUNDS",
    "vix": "VIXCLS",
}


def ensure_utc_timestamp_column(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Return a copy with a UTC timestamp column sorted in ascending order."""
    output = frame.copy()
    if timestamp_column not in output:
        if isinstance(output.index, pd.DatetimeIndex):
            output[timestamp_column] = output.index
        else:
            raise KeyError(f"Missing timestamp column: {timestamp_column}")

    output[timestamp_column] = pd.to_datetime(output[timestamp_column], utc=True)
    return output.sort_values(timestamp_column).reset_index(drop=True)


@dataclass(frozen=True)
class FinBERTSentimentScorer:
    """Score financial text using a HuggingFace FinBERT pipeline."""

    model_name: str = "ProsusAI/finbert"
    max_length: int = 512

    def score_texts(self, texts: Iterable[str]) -> pd.DataFrame:
        """Return FinBERT label, confidence, and numeric sentiment score."""
        pipeline = self._pipeline()
        text_list = ["" if text is None else str(text) for text in texts]
        predictions = pipeline(
            text_list,
            truncation=True,
            max_length=self.max_length,
        )
        rows: list[dict[str, float | str]] = []
        for prediction in predictions:
            label = str(prediction["label"]).lower()
            confidence = float(prediction["score"])
            rows.append(
                {
                    "finbert_label": label,
                    "finbert_confidence": confidence,
                    "finbert_score": FINBERT_LABEL_MAP.get(label, 0.0) * confidence,
                },
            )
        return pd.DataFrame(rows)

    def score_frame(
        self,
        frame: pd.DataFrame,
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Append FinBERT sentiment columns to a text DataFrame."""
        if text_column not in frame:
            raise KeyError(f"Missing text column: {text_column}")
        scores = self.score_texts(frame[text_column])
        return pd.concat([frame.reset_index(drop=True), scores], axis=1)

    def _pipeline(self):
        """Create a HuggingFace sentiment pipeline with a clear setup error."""
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install transformers and torch to use FinBERT sentiment.") from exc
        return pipeline("sentiment-analysis", model=self.model_name)


@dataclass(frozen=True)
class VaderSentimentScorer:
    """Score social text with VADER sentiment."""

    def score_texts(self, texts: Iterable[str]) -> pd.DataFrame:
        """Return VADER compound/positive/neutral/negative scores."""
        analyzer = self._analyzer()
        rows = []
        for text in texts:
            scores = analyzer.polarity_scores("" if text is None else str(text))
            rows.append(
                {
                    "vader_negative": float(scores["neg"]),
                    "vader_neutral": float(scores["neu"]),
                    "vader_positive": float(scores["pos"]),
                    "vader_compound": float(scores["compound"]),
                },
            )
        return pd.DataFrame(rows)

    def score_frame(self, frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Append VADER scores to a DataFrame."""
        if text_column not in frame:
            raise KeyError(f"Missing text column: {text_column}")
        scores = self.score_texts(frame[text_column])
        return pd.concat([frame.reset_index(drop=True), scores], axis=1)

    @staticmethod
    def _analyzer():
        """Create a VADER analyzer with a clear setup error."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install vaderSentiment to use VADER sentiment.") from exc
        return SentimentIntensityAnalyzer()


@dataclass(frozen=True)
class RedditSentimentProcessor:
    """Score Reddit titles/comments already collected from a subreddit."""

    text_columns: tuple[str, ...] = ("title", "comment")

    def prepare_text(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Combine Reddit title/comment fields into a single text column."""
        output = frame.copy()
        available_columns = [column for column in self.text_columns if column in output]
        if not available_columns:
            raise KeyError(f"Expected at least one Reddit text column from {self.text_columns}.")
        output["text"] = output[available_columns].fillna("").agg(" ".join, axis=1).str.strip()
        return output

    def score_vader(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append VADER sentiment to Reddit title/comment data."""
        prepared = self.prepare_text(frame)
        return VaderSentimentScorer().score_frame(prepared, text_column="text")

    def score_finbert(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append FinBERT sentiment to Reddit title/comment data."""
        prepared = self.prepare_text(frame)
        return FinBERTSentimentScorer().score_frame(prepared, text_column="text")


@dataclass(frozen=True)
class GoogleTrendsCollector:
    """Collect Google Trends interest as a retail attention proxy."""

    timeframe: str = "today 5-y"
    geo: str = ""

    def fetch_interest(self, keywords: list[str]) -> pd.DataFrame:
        """Fetch Google Trends interest over time for keywords."""
        try:
            from pytrends.request import TrendReq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install pytrends to collect Google Trends data.") from exc

        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload(keywords, timeframe=self.timeframe, geo=self.geo)
        trends = pytrends.interest_over_time()
        if "isPartial" in trends:
            trends = trends.drop(columns=["isPartial"])
        trends.index = pd.to_datetime(trends.index, utc=True)
        trends.index.name = "timestamp"
        return trends.add_prefix("google_trends_")


@dataclass(frozen=True)
class MacroDataCollector:
    """Collect macroeconomic series from FRED."""

    api_key: str | None = None
    series_map: dict[str, str] | None = None

    def fetch_fred_series(
        self,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch configured macro series from FRED."""
        try:
            from fredapi import Fred
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install fredapi to collect macro data.") from exc

        fred = Fred(api_key=self.api_key)
        series_map = self.series_map or DEFAULT_MACRO_SERIES
        rows: list[pd.Series] = []
        for name, series_id in series_map.items():
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            series.name = f"macro_{name}"
            rows.append(series)

        output = pd.concat(rows, axis=1)
        output.index = pd.to_datetime(output.index, utc=True)
        output.index.name = "timestamp"
        return output.sort_index()


@dataclass(frozen=True)
class AlternativeDataMerger:
    """Merge alternative data into market bars with point-in-time alignment."""

    timestamp_column: str = "timestamp"
    rolling_sentiment_window: int = 3

    def aggregate_sentiment(
        self,
        events: pd.DataFrame,
        score_columns: list[str],
        frequency: str = "1D",
    ) -> pd.DataFrame:
        """Aggregate event-level sentiment into timestamp buckets."""
        event_frame = ensure_utc_timestamp_column(events, self.timestamp_column)
        missing = [column for column in score_columns if column not in event_frame]
        if missing:
            raise KeyError(f"Missing sentiment score columns: {missing}")

        indexed = event_frame.set_index(self.timestamp_column)
        aggregated = indexed[score_columns].resample(frequency).mean()
        aggregated["sentiment_event_count"] = indexed[score_columns[0]].resample(frequency).count()
        aggregated.index.name = "timestamp"
        return aggregated

    def add_sentiment_momentum(
        self,
        frame: pd.DataFrame,
        score_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add rolling sentiment momentum features."""
        output = frame.copy()
        candidate_columns = score_columns or [
            column
            for column in output.columns
            if "sentiment" in column or "finbert_score" in column or "vader_compound" in column
        ]
        for column in candidate_columns:
            if column not in output:
                continue
            output[f"{column}_momentum_{self.rolling_sentiment_window}d"] = (
                output[column].rolling(self.rolling_sentiment_window).mean()
                - output[column].rolling(self.rolling_sentiment_window).mean().shift(self.rolling_sentiment_window)
            )
        return output

    def merge_asof_features(
        self,
        market_frame: pd.DataFrame,
        alternative_frame: pd.DataFrame,
        tolerance: str | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Point-in-time merge alternative features into market bars.

        Each market timestamp receives the latest alternative-data observation
        available at or before that timestamp.
        """
        market = ensure_utc_timestamp_column(market_frame, self.timestamp_column)
        alternative = ensure_utc_timestamp_column(alternative_frame, self.timestamp_column)
        merged = pd.merge_asof(
            market,
            alternative,
            on=self.timestamp_column,
            direction="backward",
            tolerance=pd.Timedelta(tolerance) if tolerance else None,
        )
        return merged.set_index(self.timestamp_column).sort_index()

    def merge_many(
        self,
        market_frame: pd.DataFrame,
        alternative_frames: Iterable[pd.DataFrame],
        tolerance: str | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Merge several alternative feature frames into one market frame."""
        output = market_frame.copy()
        for alternative_frame in alternative_frames:
            output = self.merge_asof_features(output, alternative_frame, tolerance=tolerance)
        return self.add_sentiment_momentum(output)


def verify_alternative_data_contribution(
    shap_importance: pd.DataFrame,
    alternative_prefixes: tuple[str, ...] = (
        "finbert",
        "vader",
        "reddit",
        "google_trends",
        "macro",
        "sentiment",
    ),
) -> pd.DataFrame:
    """Filter a SHAP importance table to alternative-data features."""
    if "feature" not in shap_importance:
        raise KeyError("SHAP importance table must include a feature column.")
    mask = shap_importance["feature"].str.lower().apply(
        lambda feature: any(feature.startswith(prefix) or prefix in feature for prefix in alternative_prefixes),
    )
    return shap_importance.loc[mask].reset_index(drop=True)
