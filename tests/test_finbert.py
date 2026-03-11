import pytest

from src.sentiment.finbert_analyzer import analyze_sentiment


class TestFinBERT:
    def test_model_loads_and_runs(self) -> None:
        results = analyze_sentiment(
            texts=["Stock price surged today"],
            device="cpu",
        )
        assert len(results) == 1
        assert "label" in results[0]
        assert "score" in results[0]
        assert "positive" in results[0]
        assert "negative" in results[0]
        assert "neutral" in results[0]

    def test_positive_sentiment(self) -> None:
        results = analyze_sentiment(
            texts=["Stock price surged today with record profits"],
            device="cpu",
        )
        assert results[0]["label"] == "positive"
        assert results[0]["positive"] > results[0]["negative"]

    def test_batch_processing(self) -> None:
        texts = [
            "Revenue exceeded expectations",
            "Company filed for bankruptcy",
            "Quarterly earnings were flat",
            "New product launch drives growth",
        ]
        results = analyze_sentiment(texts=texts, device="cpu", batch_size=2)
        assert len(results) == 4
        for result in results:
            assert result["label"] in {"positive", "negative", "neutral"}
            assert 0.0 <= result["score"] <= 1.0
            total = result["positive"] + result["negative"] + result["neutral"]
            assert abs(total - 1.0) < 0.01
