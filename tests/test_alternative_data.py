import numpy as np
import pandas as pd

from data import (
    AlternativeDataMerger,
    RedditSentimentProcessor,
    verify_alternative_data_contribution,
)
from data.alternative_data import ensure_utc_timestamp_column


def test_ensure_utc_timestamp_column_from_index() -> None:
    frame = pd.DataFrame({"value": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))

    output = ensure_utc_timestamp_column(frame)

    assert str(output["timestamp"].dt.tz) == "UTC"
    assert output["timestamp"].is_monotonic_increasing


def test_reddit_prepare_text_combines_title_and_comment() -> None:
    frame = pd.DataFrame(
        {
            "title": ["AAPL breakout"],
            "comment": ["strong call flow"],
        },
    )

    output = RedditSentimentProcessor().prepare_text(frame)

    assert output.loc[0, "text"] == "AAPL breakout strong call flow"


def test_aggregate_sentiment_resamples_event_scores() -> None:
    events = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 10:00Z", "2024-01-01 12:00Z", "2024-01-02 09:00Z"],
            "finbert_score": [1.0, -0.5, 0.25],
        },
    )

    output = AlternativeDataMerger().aggregate_sentiment(events, ["finbert_score"])

    assert np.isclose(output.loc[pd.Timestamp("2024-01-01", tz="UTC"), "finbert_score"], 0.25)
    assert output.loc[pd.Timestamp("2024-01-01", tz="UTC"), "sentiment_event_count"] == 2


def test_merge_asof_features_uses_latest_available_alternative_data() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "close": [100.0, 101.0, 102.0],
        },
    )
    alternative = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-12-31 23:00Z", "2024-01-02 12:00Z"]),
            "macro_cpi": [3.0, 3.2],
        },
    )

    merged = AlternativeDataMerger().merge_asof_features(market, alternative)

    assert merged.loc[pd.Timestamp("2024-01-01", tz="UTC"), "macro_cpi"] == 3.0
    assert merged.loc[pd.Timestamp("2024-01-03", tz="UTC"), "macro_cpi"] == 3.2


def test_add_sentiment_momentum_creates_3d_feature() -> None:
    frame = pd.DataFrame(
        {"finbert_score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]},
        index=pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"),
    )

    output = AlternativeDataMerger(rolling_sentiment_window=3).add_sentiment_momentum(frame)

    expected = np.mean([0.7, 0.8, 0.9]) - np.mean([0.1, 0.2, 0.3])
    assert np.isclose(output["finbert_score_momentum_3d"].iloc[-1], expected)


def test_verify_alternative_data_contribution_filters_shap_table() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["ema_cross", "finbert_score_momentum_3d", "macro_vix"],
            "mean_abs_shap": [0.2, 0.4, 0.1],
        },
    )

    output = verify_alternative_data_contribution(importance)

    assert output["feature"].tolist() == ["finbert_score_momentum_3d", "macro_vix"]
