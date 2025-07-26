# Intraday Sentiment Analysis System

## Overview

This system provides real-time sentiment analysis for 117 NSE stocks during market hours (9:15 AM - 3:30 PM IST). It monitors multiple news sources, analyzes sentiment using FinBERT, and provides alerts for significant sentiment changes.

## Features

- **Real-time Monitoring**: Continuously monitors 117 NSE stocks during market hours
- **Multiple News Sources**: Fetches news from MoneyControl, Economic Times, LiveMint, Business Standard, and NDTV Business
- **Advanced Sentiment Analysis**: Uses FinBERT (financial BERT) for accurate financial sentiment analysis
- **Intraday Tracking**: Stores sentiment data with timestamps for historical analysis
- **Smart Alerts**: Generates alerts for strong positive/negative sentiment (threshold: ±0.6)
- **Market Hours Detection**: Only runs during NSE market hours
- **Database Storage**: SQLite database for persistent storage and analysis
- **Export Capabilities**: Export data to CSV for further analysis

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r intraday_requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import transformers, torch, feedparser; print('✅ All dependencies installed')"
   ```

## Usage

### Quick Start

1. **Start Intraday Monitoring**:
   ```bash
   python intraday_control.py start
   ```

2. **Check System Status**:
   ```bash
   python intraday_control.py status
   ```

3. **View Recent Alerts**:
   ```bash
   python intraday_control.py alerts
   ```

### Control Commands

| Command | Description | Options |
|---------|-------------|---------|
| `start` | Start intraday monitoring | None |
| `status` | Show system status and recent activity | None |
| `alerts` | Show recent sentiment alerts | `--hours` (default: 24) |
| `stock` | Show sentiment for specific stock | `--symbol`, `--hours` |
| `export` | Export data to CSV | `--output` |
| `summary` | Show comprehensive summary | None |

### Examples

**Start monitoring during market hours**:
```bash
python intraday_control.py start
```

**Check current status**:
```bash
python intraday_control.py status
```

**View alerts from last 6 hours**:
```bash
python intraday_control.py alerts --hours 6
```

**Check sentiment for RELIANCE**:
```bash
python intraday_control.py stock --symbol RELIANCE.NSE --hours 12
```

**Export all data**:
```bash
python intraday_control.py export --output sentiment_data.csv
```

**Get comprehensive summary**:
```bash
python intraday_control.py summary
```

## System Architecture

### Components

1. **News Fetcher**: Fetches RSS feeds from multiple financial news sources
2. **Stock Mention Extractor**: Identifies which stocks are mentioned in news articles
3. **Sentiment Analyzer**: Uses FinBERT for financial sentiment analysis
4. **Database Manager**: Stores sentiment data and alerts in SQLite
5. **Alert System**: Monitors for significant sentiment changes
6. **Scheduler**: Runs updates every 2 minutes during market hours

### Data Flow

```
News Sources → RSS Feeds → Stock Mention Detection → Sentiment Analysis → Database Storage → Alerts
```

### Database Schema

**intraday_sentiment table**:
- `id`: Primary key
- `timestamp`: ISO timestamp
- `symbol`: Stock symbol (e.g., RELIANCE.NSE)
- `sentiment_score`: Score from -1 to 1
- `sentiment_label`: positive/negative/neutral
- `news_count`: Number of news articles analyzed
- `source`: Data source
- `confidence`: Model confidence score
- `volume_change`: Optional volume change data
- `price_change`: Optional price change data

**sentiment_alerts table**:
- `id`: Primary key
- `timestamp`: Alert timestamp
- `symbol`: Stock symbol
- `alert_type`: strong_positive/strong_negative
- `sentiment_score`: Sentiment score that triggered alert
- `message`: Alert message

## Configuration

### Market Hours
- **Open**: 9:15 AM IST
- **Close**: 3:30 PM IST
- **Timezone**: Asia/Kolkata

### Update Intervals
- **News Fetch**: Every 2 minutes
- **Sentiment Update**: Every 5 minutes
- **Alert Threshold**: ±0.6

### News Sources
- MoneyControl RSS
- Economic Times RSS
- LiveMint RSS
- Business Standard RSS
- NDTV Business RSS

## Monitoring 117 Stocks

The system monitors all stocks from your NSE symbols file:

**Large Cap (50 stocks)**:
- ADANIENT.NSE, ADANIPORTS.NSE, APOLLOHOSP.NSE, ASIANPAINT.NSE, AXISBANK.NSE
- BAJAJ-AUTO.NSE, BAJFINANCE.NSE, BHARTIARTL.NSE, BPCL.NSE, BRITANNIA.NSE
- And 40 more...

**Mid Cap (47 stocks)**:
- ALKEM.NSE, AMBER.NSE, AUBANK.NSE, AUROPHARMA.NSE, BANDHANBNK.NSE
- BANKBARODA.NSE, BATAINDIA.NSE, BERGEPAINT.NSE, BIOCON.NSE, BOSCHLTD.NSE
- And 37 more...

**Additional Large Cap (10 stocks)**:
- ACC.NSE, ABCAPITAL.NSE, ABFRL.NSE, IRCTC.NSE, MINDAIND.NSE
- NYKAA.NSE, PAYTM.NSE, POLICYBZR.NSE, STAR.NSE, ZOMATO.NSE

**Additional Mid Cap (10 stocks)**:
- ASTRAL.NSE, BALKRISIND.NSE, CUMMINSIND.NSE, EXIDEIND.NSE, HONAUT.NSE
- IPCALAB.NSE, LICHSGFIN.NSE, MAXHEALTH.NSE, OBEROIRLTY.NSE, TATACHEM.NSE

## Output Examples

### Status Output
```
=== Intraday Sentiment System Status ===
📊 Recent Activity (Last Hour): 234 sentiment records
📈 Total Records: 1,247
🚨 Recent Alerts (24h): 12

📋 Sentiment Distribution (Last Hour):
  Positive: 89
  Neutral: 98
  Negative: 47

🔥 Most Active Stocks (Last Hour):
  RELIANCE.NSE: 15 mentions
  TCS.NSE: 12 mentions
  HDFCBANK.NSE: 10 mentions
```

### Alert Output
```
=== Recent Sentiment Alerts (Last 24 hours) ===
🟢 2025-01-07T14:30:15 | RELIANCE.NSE | strong_positive | Score: 0.723
   Strong positive sentiment detected for RELIANCE.NSE: 0.723

🔴 2025-01-07T13:45:22 | TCS.NSE | strong_negative | Score: -0.689
   Strong negative sentiment detected for TCS.NSE: -0.689
```

## Troubleshooting

### Common Issues

1. **FinBERT Model Loading Error**:
   ```bash
   # Check internet connection and try again
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')"
   ```

2. **Database Lock Error**:
   ```bash
   # Remove database and restart
   rm intraday_sentiment.db
   python intraday_control.py start
   ```

3. **No News Data**:
   - Check internet connection
   - Verify RSS feed URLs are accessible
   - Check if outside market hours

### Log Files

- `intraday_sentiment.log`: Main system log
- `intraday_sentiment.db`: SQLite database
- `intraday_sentiment_export_*.csv`: Exported data files

## Performance

- **Processing Speed**: ~10-15 seconds per update cycle
- **Memory Usage**: ~2-3 GB (mainly for FinBERT model)
- **Storage**: ~1-2 MB per day of sentiment data
- **Concurrent Processing**: Up to 10 stocks analyzed simultaneously

## Integration

This system can be integrated with:
- Trading algorithms
- Portfolio management systems
- Risk management tools
- Dashboard applications
- Alert systems (email, SMS, webhooks)

## License

This system is part of the trading signals project and follows the same licensing terms. 