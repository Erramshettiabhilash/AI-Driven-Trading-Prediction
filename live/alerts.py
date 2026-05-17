"""Telegram and Discord alert helpers for trading signals and monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from live.signal_engine import LiveSignal


PostFunction = Callable[..., object]


def _default_post(*args, **kwargs) -> object:
    """Import requests only when a real outbound alert is sent."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install requests to send Telegram or Discord alerts.") from exc
    return requests.post(*args, **kwargs)


@dataclass(frozen=True)
class AlertMessage:
    """Human-readable alert payload."""

    title: str
    body: str
    severity: str = "info"

    def as_text(self) -> str:
        """Return a compact plain-text alert."""
        return f"[{self.severity.upper()}] {self.title}\n{self.body}"


def signal_alert_message(signal: LiveSignal, risk_info: str | None = None) -> AlertMessage:
    """Create a BUY/SELL/HOLD alert from a live signal."""
    regime = signal.regime or "unknown"
    risk_suffix = f"\nRisk: {risk_info}" if risk_info else ""
    body = (
        f"Symbol: {signal.symbol}\n"
        f"Signal: {signal.signal}\n"
        f"Confidence: {signal.confidence:.2%}\n"
        f"Prediction: {signal.prediction:.6f}\n"
        f"Regime: {regime}"
        f"{risk_suffix}"
    )
    severity = "warning" if signal.signal in {"BUY", "SELL"} else "info"
    return AlertMessage(title=f"{signal.symbol} {signal.signal}", body=body, severity=severity)


class TelegramAlerter:
    """Send alerts to a Telegram bot chat."""

    def __init__(self, bot_token: str, chat_id: str, post: PostFunction | None = None) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.post = post or _default_post

    def send(self, message: AlertMessage) -> None:
        """Send a Telegram text alert."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        response = self.post(
            url,
            json={"chat_id": self.chat_id, "text": message.as_text()},
            timeout=10,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()


class DiscordWebhookAlerter:
    """Send alerts to a Discord webhook."""

    def __init__(self, webhook_url: str, post: PostFunction | None = None) -> None:
        self.webhook_url = webhook_url
        self.post = post or _default_post

    def send(self, message: AlertMessage) -> None:
        """Send a Discord webhook alert."""
        response = self.post(
            self.webhook_url,
            json={"content": message.as_text()},
            timeout=10,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
