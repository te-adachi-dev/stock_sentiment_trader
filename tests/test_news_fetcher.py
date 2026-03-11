import os

import pytest

from src.data_collection.news_fetcher import deduplicate_news, fetch_finnhub_news


class TestNewsFetcher:
    @pytest.fixture
    def finnhub_api_key(self) -> str | None:
        return os.getenv("FINNHUB_API_KEY")

    def test_finnhub_fetch(self, finnhub_api_key: str | None, tmp_path: str) -> None:
        if not finnhub_api_key:
            pytest.skip("FINNHUB_API_KEY not set")
        news = fetch_finnhub_news(
            symbol="AAPL",
            api_key=finnhub_api_key,
            lookback_days=7,
            max_articles=10,
        )
        assert isinstance(news, list)
        if news:
            article = news[0]
            assert "datetime" in article
            assert "symbol" in article
            assert "headline" in article
            assert "summary" in article
            assert "source" in article
            assert "url" in article

    def test_unified_format(self, finnhub_api_key: str | None) -> None:
        if not finnhub_api_key:
            pytest.skip("FINNHUB_API_KEY not set")
        news = fetch_finnhub_news(
            symbol="MSFT",
            api_key=finnhub_api_key,
            lookback_days=7,
            max_articles=5,
        )
        required_keys = {"datetime", "symbol", "headline", "summary", "source", "url"}
        for article in news:
            assert required_keys.issubset(article.keys())
            assert article["source"] == "finnhub"
            assert article["symbol"] == "MSFT"

    def test_deduplication(self) -> None:
        news = [
            {"url": "https://example.com/1", "headline": "GPU sales surge", "symbol": "NVDA"},
            {"url": "https://example.com/2", "headline": "CPU demand rises", "symbol": "INTC"},
            {"url": "https://example.com/1", "headline": "GPU sales surge (duplicate)", "symbol": "NVDA"},
            {"url": "https://example.com/3", "headline": "Sedan market grows", "symbol": "TSLA"},
        ]
        unique = deduplicate_news(news)
        assert len(unique) == 3
        urls = [a["url"] for a in unique]
        assert len(set(urls)) == 3
