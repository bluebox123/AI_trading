# NSE News Sentiment Analysis Pipeline

A comprehensive, automated pipeline for real-time sentiment analysis of NSE (National Stock Exchange) stock news using Google News RSS feeds and FinBERT sentiment analysis.

## 🎯 Overview

This pipeline automatically:
- Fetches news articles for NSE tickers every 10 minutes from Google News RSS feeds
- Analyzes sentiment using the pre-trained FinBERT model
- Stores timestamped results for further analysis
- Provides deduplication to avoid processing duplicate articles
- Includes comprehensive error handling and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Fetcher  │    │ Sentiment       │    │ Data            │
│                 │───▶│ Analyzer        │───▶│ Persistence     │
│ (Google RSS)    │    │ (FinBERT)       │    │ (CSV/SQLite)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Scheduler     │
                    │ (APScheduler)   │
                    └─────────────────┘
```

## 🚀 Features

### Core Functionality
- **Real-time News Fetching**: Polls Google News RSS feeds every 10 minutes
- **Smart Deduplication**: Tracks seen articles to avoid reprocessing
- **Batch Processing**: Efficient sentiment analysis using FinBERT
- **Flexible Storage**: Supports both CSV and SQLite storage
- **Rate Limiting**: Respects Google's rate limits with intelligent delays

### Reliability & Monitoring
- **Health Checks**: Continuous monitoring of all components
- **Error Recovery**: Automatic retry with exponential backoff
- **Failure Handling**: Cooldown mode after consecutive failures
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

### Configuration
- **30 NSE Tickers**: Pre-configured with major NSE stocks
- **Customizable Settings**: Easy configuration via config files
- **Multiple Run Modes**: Scheduled, single run, test, and status modes
- **Debug Support**: Dry-run mode and verbose logging

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for FinBERT model)
- Internet connection for RSS feeds and model download

### Python Dependencies
```
feedparser==6.0.10
apscheduler==3.10.4
transformers==4.36.0
torch==2.1.1
pandas==2.1.4
numpy==1.24.3
requests==2.31.0
beautifulsoup4==4.12.2
```

## 🛠️ Installation

1. **Clone or navigate to the news-sentiment-service directory**
   ```bash
   cd trading-signals-web/news-sentiment-service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories** (automatically created on first run)
   ```bash
   mkdir -p data logs
   ```

## 🎮 Usage

### Scheduled Mode (Continuous Operation)
Run the pipeline continuously with automatic scheduling:
```bash
python main.py
# or explicitly
python main.py --mode scheduled
```

### Single Run Mode
Execute the pipeline once and exit:
```bash
# Process all configured tickers
python main.py --mode single

# Process specific tickers
python main.py --mode single --tickers RELIANCE.NS TCS.NS INFY.NS

# Save results to file
python main.py --mode single --output results.json
```

### Test Mode
Quick test with a subset of tickers:
```bash
python main.py --mode test
```

### Status Mode
Check current pipeline status and recent activity:
```bash
python main.py --mode status
```

### Advanced Options
```bash
# Enable verbose logging
python main.py --verbose

# Dry run (don't save data)
python main.py --dry-run

# Debug mode
python main.py --debug

# Custom log level
python main.py --log-level DEBUG
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_pipeline.py
```

This will test:
- News fetcher functionality
- Sentiment analyzer performance
- Data persistence operations
- End-to-end pipeline execution
- Health check systems

## 📊 Configuration

### Main Configuration (`config/config.py`)

Key settings you can modify:

```python
# Polling frequency
POLLING_INTERVAL_MINUTES = 10

# Tickers to monitor
NSE_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', ...]

# Sentiment analysis
MIN_CONFIDENCE_THRESHOLD = 0.6
BATCH_SIZE = 16

# Rate limiting
MIN_REQUEST_INTERVAL = 2  # seconds between requests
```

### Ticker Configuration
The pipeline monitors 30 major NSE stocks by default. You can modify the `NSE_TICKERS` list in `config/config.py` to add or remove tickers.

### Company Name Mapping
For better news relevance, company names are mapped to tickers in `TICKER_SEARCH_NAMES`. This improves search results on Google News.

## 📁 Data Output

### CSV Format
Results are saved to `data/nse_news_sentiment.csv` with columns:
- `ticker`: Stock ticker (e.g., RELIANCE.NS)
- `headline`: News headline
- `link`: Article URL
- `pub_date`: Publication date
- `sentiment`: Predicted sentiment (positive/neutral/negative)
- `sentiment_confidence`: Confidence score (0-1)
- `sentiment_probabilities`: Full probability distribution
- `fetch_timestamp`: When the article was fetched
- `sentiment_timestamp`: When sentiment was analyzed

### Database Option
Set `USE_DATABASE = True` in config to use SQLite instead of CSV.

## 📈 Monitoring & Logging

### Log Files
- `logs/info.log`: General operation logs
- `logs/error.log`: Error and warning logs
- `logs/scheduler.log`: Scheduler-specific logs (when running in scheduled mode)

### Health Monitoring
The pipeline includes built-in health checks that monitor:
- News fetcher connectivity
- Sentiment analyzer model status
- Data persistence functionality
- Consecutive failure counts

### Performance Metrics
- Articles fetched per run
- Sentiment analysis throughput
- Success/failure rates
- Average confidence scores
- Processing duration

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```
   Error: Failed to load FinBERT model
   ```
   - Ensure stable internet connection
   - Check if Hugging Face transformers can access the model
   - Try running with `--verbose` for detailed error messages

2. **No News Articles Found**
   ```
   Warning: No entries found in RSS feed
   ```
   - Google News RSS may have rate limits
   - Try reducing polling frequency
   - Check if specific tickers have recent news

3. **Memory Issues**
   ```
   Error: CUDA out of memory / RAM insufficient
   ```
   - Reduce `BATCH_SIZE` in config
   - Use CPU-only mode (model will auto-detect)
   - Consider running fewer tickers simultaneously

4. **Permission Errors**
   ```
   Error: Permission denied writing to data/
   ```
   - Ensure write permissions for data/ and logs/ directories
   - Check disk space availability

### Debug Mode
Enable debug mode for detailed troubleshooting:
```bash
python main.py --mode test --debug --verbose
```

## 🚦 Rate Limiting & Best Practices

### Google News RSS Limits
- Maximum 64 entries per RSS query
- Unofficial rate limiting (avoid requests faster than every 15 minutes)
- Pipeline defaults to 10-minute intervals with 2-second delays between tickers

### Performance Optimization
- Uses batch processing for sentiment analysis
- Implements smart caching for seen articles
- Automatically cleans up old data to prevent storage bloat

### Ethical Usage
- Respects Google's robots.txt and rate limits
- Only processes publicly available RSS feeds
- Includes proper error handling to avoid overwhelming services

## 📄 License

This project is part of the larger trading signals web application. Please refer to the main project license.

## 🤝 Contributing

1. Test your changes thoroughly using `test_pipeline.py`
2. Follow the existing code style and logging patterns
3. Update configuration documentation for any new settings
4. Ensure error handling is comprehensive

## 📞 Support

For issues specific to the news sentiment pipeline:
1. Check the logs in `logs/` directory
2. Run the test suite to isolate issues
3. Use `--mode status` to check component health
4. Enable debug mode for detailed error information

---

## 🎯 Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the pipeline
python main.py --mode test

# 3. Run once to see results
python main.py --mode single --tickers RELIANCE.NS

# 4. Start continuous monitoring
python main.py --mode scheduled
```

This will start fetching and analyzing news sentiment for NSE stocks every 10 minutes, saving results to CSV files for further analysis. 