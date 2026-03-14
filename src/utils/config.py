import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: str = "configs/settings.yaml") -> dict:
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    finnhub_key = os.getenv("FINNHUB_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_KEY")

    if not finnhub_key:
        logger.warning("FINNHUB_API_KEY is not set. News fetching from Finnhub will be disabled.")
    if not newsapi_key:
        logger.warning("NEWSAPI_KEY is not set. News fetching from NewsAPI will be disabled.")

    config["api_keys"] = {
        "finnhub": finnhub_key,
        "newsapi": newsapi_key,
    }

    return config
