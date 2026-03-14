from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_model = None
_tokenizer = None
_device = None


def _load_model(model_name: str = "ProsusAI/finbert", device: str = "cuda") -> None:
    global _model, _tokenizer, _device

    if _model is not None:
        return

    _device = device if torch.cuda.is_available() else "cpu"
    if _device != device:
        logger.warning(f"CUDA not available, falling back to CPU")

    logger.info(f"Loading FinBERT model ({model_name}) on {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()
    logger.info("FinBERT model loaded successfully")


def analyze_sentiment(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
) -> list[dict]:
    _load_model(model_name=model_name, device=device)

    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    all_results: list[dict] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = _tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for j in range(len(batch_texts)):
            prob_values = probs[j].cpu().tolist()
            predicted_label_idx = probs[j].argmax().item()
            all_results.append({
                "label": label_map[predicted_label_idx],
                "score": prob_values[predicted_label_idx],
                "positive": prob_values[0],
                "negative": prob_values[1],
                "neutral": prob_values[2],
            })

    return all_results


def analyze_news_sentiment(
    news_data: dict[str, list[dict]],
    model_name: str = "ProsusAI/finbert",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
    save_dir: str = "data/processed/sentiment",
) -> dict[str, pd.DataFrame]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for symbol, articles in news_data.items():
        if not articles:
            logger.warning(f"No articles for {symbol}, skipping sentiment analysis")
            continue

        logger.info(f"Analyzing sentiment for {symbol} ({len(articles)} articles)...")

        texts = []
        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            combined = f"{headline}. {summary}".strip()
            if combined == ".":
                combined = "neutral"
            texts.append(combined)

        sentiments = analyze_sentiment(
            texts=texts,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        rows = []
        for article, sentiment in zip(articles, sentiments):
            rows.append({
                "datetime": article.get("datetime", ""),
                "symbol": symbol,
                "headline": article.get("headline", ""),
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "positive": sentiment["positive"],
                "negative": sentiment["negative"],
                "neutral": sentiment["neutral"],
            })

        df = pd.DataFrame(rows)
        csv_path = save_path / f"{symbol}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        results[symbol] = df
        logger.info(f"Sentiment analysis complete for {symbol}: {len(df)} rows")

    return results
