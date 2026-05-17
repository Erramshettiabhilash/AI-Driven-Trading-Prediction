import numpy as np
import pandas as pd

from live import DiscordWebhookAlerter, LiveSignal, TelegramAlerter, signal_alert_message
from visualization import drawdown_series, equity_curve_from_returns, shap_top_features


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


def test_signal_alert_message_contains_risk_context() -> None:
    signal = LiveSignal(
        symbol="BTCUSDT",
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        signal="BUY",
        confidence=0.75,
        prediction=0.004,
        regime="trending",
    )

    message = signal_alert_message(signal, risk_info="max position 5%")

    assert "BTCUSDT" in message.title
    assert "Confidence: 75.00%" in message.body
    assert "max position 5%" in message.body
    assert message.severity == "warning"


def test_telegram_and_discord_alerters_post_expected_payloads() -> None:
    calls = []

    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    message = signal_alert_message(
        LiveSignal("ETHUSDT", pd.Timestamp("2024-01-01", tz="UTC"), "HOLD", 0.2, 0.0),
    )
    TelegramAlerter("token", "chat", post=fake_post).send(message)
    DiscordWebhookAlerter("https://discord.example/webhook", post=fake_post).send(message)

    assert "sendMessage" in calls[0]["url"]
    assert calls[0]["json"]["chat_id"] == "chat"
    assert "content" in calls[1]["json"]


def test_dashboard_equity_drawdown_and_shap_helpers() -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    equity = equity_curve_from_returns(returns)
    drawdown = drawdown_series(equity)
    shap = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [0.2, 0.5, 0.1],
        },
    )

    top = shap_top_features(shap, top_n=2)

    assert np.isclose(equity.iloc[0], 1.01)
    assert drawdown.iloc[1] < 0
    assert top["feature"].tolist() == ["b", "a"]
