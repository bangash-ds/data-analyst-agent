"""
Centralized application configuration.

Loads environment variable from .env, validates required settings
and defines project directories paths. All directories are created 
automatically on import
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)


BASE_DIR: Path = Path(__file__).resolve().parents[2]

# Environment variables
_env_file = BASE_DIR / ".env"

if not _env_file.exists():
    log.warning(
        ".env file not found at %s. Using system environment variables",
        _env_file
    )

load_dotenv(_env_file)

# API Configuration for Groq
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    log.error(
        "GROQ_API_KEY is not set. Agent will fail on LLM calls. "
        "Add it to .env or export as environment variable."
    )

# Project directories

DATA_DIR: Path = BASE_DIR / "data"
REPORT_DIR: Path = BASE_DIR / "reports"
CHART_DIR: Path = BASE_DIR / "charts"

for _dir in (DATA_DIR, CHART_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
    log.debug("Ensured directory exists: %s", _dir)

    