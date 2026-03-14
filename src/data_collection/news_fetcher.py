import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_finnhub_news(
    symbol: str,
    api_key: str,
    lookback_days: int = 365,
    max_articles: int = 500,
    calls_per_minute: int = 55,
) -> list[dict]:
    try:
        import finnhub
    except ImportError:
        logger.error("finnhub-python is not installed")
        return []

    client = finnhub.Client(api_key=api_key)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    all_news: list[dict] = []
    delay = 60.0 / calls_per_minute

    chunk_days = 30
    current_start = start_date
    while current_start < end_date and len(all_news) < max_articles:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        try:
            news = client.company_news(
                symbol,
                _from=current_start.strftime("%Y-%m-%d"),
                to=current_end.strftime("%Y-%m-%d"),
            )
            for article in news:
                if len(all_news) >= max_articles:
                    break
                all_news.append({
                    "datetime": datetime.fromtimestamp(article.get("datetime", 0)).isoformat() + "Z",
                    "symbol": symbol,
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", ""),
                    "source": "finnhub",
                    "url": article.get("url", ""),
                })
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Finnhub error for {symbol} ({current_start.date()} - {current_end.date()}): {e}")
            time.sleep(delay)

        current_start = current_end + timedelta(days=1)

    logger.info(f"Finnhub: fetched {len(all_news)} articles for {symbol}")
    return all_news


def fetch_newsapi_news(
    symbol: str,
    api_key: str,
    lookback_days: int = 30,
    max_articles: int = 100,
) -> list[dict]:
    all_news: list[dict] = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(lookback_days, 30))

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),
            "apiKey": api_key,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        for article in data.get("articles", []):
            all_news.append({
                "datetime": article.get("publishedAt", ""),
                "symbol": symbol,
                "headline": article.get("title", ""),
                "summary": article.get("description", "") or "",
                "source": "newsapi",
                "url": article.get("url", ""),
            })
    except Exception as e:
        logger.error(f"NewsAPI error for {symbol}: {e}")

    logger.info(f"NewsAPI: fetched {len(all_news)} articles for {symbol}")
    return all_news


def deduplicate_news(news_list: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for article in news_list:
        url = article.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(article)
    return unique


def fetch_all_news(
    symbols: list[str],
    finnhub_api_key: str | None = None,
    newsapi_key: str | None = None,
    lookback_days: int = 365,
    max_articles_per_symbol: int = 500,
    calls_per_minute: int = 55,
    save_dir: str = "data/raw/news",
) -> dict[str, list[dict]]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if not finnhub_api_key and not newsapi_key:
        logger.warning("No API keys provided. Skipping news fetching.")
        return {}

    results: dict[str, list[dict]] = {}

    for i, symbol in enumerate(symbols):
        logger.info(f"Fetching news for {symbol} ({i + 1}/{len(symbols)})...")
        all_articles: list[dict] = []

        if finnhub_api_key:
            finnhub_news = fetch_finnhub_news(
                symbol=symbol,
                api_key=finnhub_api_key,
                lookback_days=lookback_days,
                max_articles=max_articles_per_symbol,
                calls_per_minute=calls_per_minute,
            )
            all_articles.extend(finnhub_news)

        if newsapi_key:
            newsapi_news = fetch_newsapi_news(
                symbol=symbol,
                api_key=newsapi_key,
                lookback_days=min(lookback_days, 30),
                max_articles=100,
            )
            all_articles.extend(newsapi_news)

        all_articles = deduplicate_news(all_articles)
        results[symbol] = all_articles

        jsonl_path = save_path / f"{symbol}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for article in all_articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(all_articles)} articles for {symbol}")

    return results
