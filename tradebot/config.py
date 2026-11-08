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
    return cfg


def load_secrets() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return {
        "alpaca_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_secret": os.getenv("ALPACA_SECRET_KEY"),
        "alpaca_paper": os.getenv("ALPACA_PAPER", "true").lower() != "false",
        "anthropic_key": os.getenv("ANTHROPIC_API_KEY"),
    }
