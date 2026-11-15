"""Load config.yaml and secrets from .env."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # make db path absolute relative to project root
    cfg["db_path"] = str((ROOT / cfg["db_path"]).resolve())
    # A universe may be a list, or "file:<path>" naming a file with one ticker per line.
    for name, value in (cfg.get("universes") or {}).items():
        if isinstance(value, str) and value.startswith("file:"):
            cfg["universes"][name] = read_tickers(ROOT / value[5:])
    if cfg.get("model"):
        cfg["model"]["prices_path"] = str((ROOT / cfg["model"]["prices_path"]).resolve())
    return cfg


def read_tickers(path: str | Path) -> list[str]:
    """Tickers from a text file: one per line, blank lines and # comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _real_key(value: str | None) -> str | None:
    """None for blanks, template placeholders, or an endpoint URL pasted by mistake."""
    if not value or value.startswith(("your_", "http://", "https://")):
        return None
    return value


def load_secrets() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return {
        "alpaca_key": _real_key(os.getenv("ALPACA_API_KEY")),
        "alpaca_secret": _real_key(os.getenv("ALPACA_SECRET_KEY")),
        "alpaca_paper": os.getenv("ALPACA_PAPER", "true").lower() != "false",
        "gemini_key": _real_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    }
