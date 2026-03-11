# Stock Sentiment Trader

News sentiment analysis and technical indicator ensemble system for generating day-trading signals on US large-cap stocks. The system uses FinBERT for financial sentiment analysis on news articles and validates the predictive relationship between sentiment and stock returns through Granger causality testing.

## Setup

```bash
cd ~/stock_sentiment_trader
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variables

Copy the example env file and set your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `FINNHUB_API_KEY` - Required for news fetching (free tier: 60 calls/min)
- `NEWSAPI_KEY` - Optional, supplements Finnhub news data (free tier: 100 requests/day)

### FinBERT Model

The FinBERT model (~400MB) is automatically downloaded from HuggingFace on first run. Ensure you have sufficient disk space and internet connectivity.

## Phase 1: Validation

Run the Phase 1 validation pipeline to test whether news sentiment has predictive power for stock returns:

```bash
python scripts/run_phase1_validation.py
```

This will:
1. Fetch 1 year of daily stock data for 10 US large-cap stocks
2. Collect news articles from Finnhub (and NewsAPI if configured)
3. Run FinBERT sentiment analysis on all collected news
4. Perform Granger causality tests (sentiment -> returns and returns -> sentiment)
5. Generate correlation analysis with visualizations
6. Output a Go/No-Go judgment for Phase 2

Results are saved to `data/results/`.

## Running Tests

```bash
pytest tests/
```

Tests for news fetching require `FINNHUB_API_KEY` to be set and will be skipped otherwise.

## Directory Structure

```
stock_sentiment_trader/
├── configs/settings.yaml          # Target symbols, API config, thresholds
├── src/
│   ├── data_collection/           # Stock price and news fetching
│   ├── sentiment/                 # FinBERT sentiment analysis
│   ├── analysis/                  # Granger causality and correlation
│   ├── indicators/                # Technical indicators (RSI, MACD, etc.)
│   └── utils/                     # Logger, config loader
├── scripts/                       # Pipeline execution scripts
├── tests/                         # Unit tests
├── data/                          # Raw/processed data and results (gitignored)
├── logs/                          # Application logs (gitignored)
└── models/                        # Saved models (gitignored)
```

## Phase Roadmap

- **Phase 1**: Skeleton and validation - Verify that news sentiment Granger-causes stock returns
- **Phase 2**: Backtesting - Build trading strategy and evaluate historical performance
- **Phase 3**: Paper trading - Live signal generation without real money
