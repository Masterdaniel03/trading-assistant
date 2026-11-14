"""Alerts: tell the user when something needs attention.

    raise_alert(conn, "error", "Daily run failed", "fetch prices: exit code 1", source="daily")

Every alert is stored in the database (the dashboard shows it until it is
marked as read) and appended to logs/alerts.log. It is then sent to each
channel that is switched on:

  desktop  a Windows notification on this PC          (config: alerts.desktop)
  ntfy     a push message to a phone, via ntfy.sh     (.env: NTFY_TOPIC)
  discord  a message to a Discord channel             (.env: DISCORD_WEBHOOK_URL)

Sending is best effort. A channel that fails is recorded on the alert and
never stops the run that raised it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from tradebot import db

LEVELS = ("info", "warning", "error")
NTFY_PRIORITY = {"info": "default", "warning": "high", "error": "urgent"}
NTFY_TAGS = {"info": "white_check_mark", "warning": "warning", "error": "rotating_light"}

# Shows the notification under PowerShell's own app id, so nothing has to be installed.
_TOAST_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$title = [System.Security.SecurityElement]::Escape($env:TRADEBOT_ALERT_TITLE)
$body = [System.Security.SecurityElement]::Escape($env:TRADEBOT_ALERT_BODY)
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$title</text><text>$body</text></binding></visual></toast>")
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show($toast)
"""


def send_desktop(title: str, body: str, level: str = "info") -> None:
    """A Windows notification. It stays in the notification centre until dismissed."""
    if sys.platform != "win32":
        raise RuntimeError("desktop notifications are only set up for Windows")
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", _TOAST_SCRIPT],
        env={**os.environ, "TRADEBOT_ALERT_TITLE": title, "TRADEBOT_ALERT_BODY": body[:300]},
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip().splitlines()[0][:200])


def send_ntfy(topic: str, title: str, body: str, level: str = "info",
              server: str = "https://ntfy.sh") -> None:
    """A push message to every phone subscribed to `topic` in the ntfy app."""
    import requests
    reply = requests.post(f"{server.rstrip('/')}/{topic}", data=(body or title).encode("utf-8"),
                          headers={"Title": title.encode("ascii", "replace").decode(),
                                   "Priority": NTFY_PRIORITY[level], "Tags": NTFY_TAGS[level]},
                          timeout=15)
    reply.raise_for_status()


def send_discord(url: str, title: str, body: str, level: str = "info") -> None:
    import requests
    reply = requests.post(url, json={"content": f"**{title}**\n{body}"[:1900]}, timeout=15)
    reply.raise_for_status()


def configured_channels(cfg: dict | None, env=None) -> dict[str, Callable[[str, str, str], None]]:
    """The channels switched on right now: name -> function(title, body, level)."""
    env = os.environ if env is None else env
    if env.get("TRADEBOT_ALERTS", "").lower() == "off":      # the test suite sets this
        return {}
    settings = (cfg or {}).get("alerts", {}) or {}
    channels: dict[str, Callable[[str, str, str], None]] = {}
    if settings.get("desktop", True) and sys.platform == "win32":
        channels["desktop"] = send_desktop
    topic = (env.get("NTFY_TOPIC") or "").strip()
    if topic:
        server = (env.get("NTFY_SERVER") or "https://ntfy.sh").strip()
        channels["ntfy"] = lambda title, body, level: send_ntfy(topic, title, body, level, server)
    hook = (env.get("DISCORD_WEBHOOK_URL") or "").strip()
    if hook:
        channels["discord"] = lambda title, body, level: send_discord(hook, title, body, level)
    return channels


def raise_alert(conn, level: str, title: str, detail: str = "", source: str = "",
                cfg: dict | None = None, channels: dict | None = None,
                log_path: str | Path | None = None) -> int:
    """Store the alert, write it to the alert log, and send it to every channel.

    `channels` replaces the configured ones (tests pass fakes). Returns the alert's id.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown alert level {level!r}. Options: {LEVELS}")
    channels = configured_channels(cfg) if channels is None else channels
    delivered, failed = [], []
    for name, send in channels.items():
        try:
            send(title, detail, level)
            delivered.append(name)
        except Exception as error:  # noqa: BLE001 - an alert that can't be sent must not stop a run
            failed.append(f"{name}: {type(error).__name__}: {str(error)[:120]}")
    alert_id = db.add_alert(conn, level, title, detail, source, ", ".join(delivered) or None,
                            "; ".join(failed) or None)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            sent = ", ".join(delivered) or "nowhere (no channel is switched on)"
            fh.write(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  {level.upper():<7} {title}\n")
            for line in (detail or "").splitlines():
                fh.write(f"    {line}\n")
            fh.write(f"    sent to: {sent}" + (f"; not sent: {'; '.join(failed)}" if failed else "") + "\n")
    return alert_id
