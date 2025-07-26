# Live Trading Signals Implementation

## Overview

The Live Trading Signals page integrates your v5 and v4 AI models with real-time EODHD intraday sentiment analysis to generate comprehensive trading signals for 117 NSE stocks.

## Key Features

### 1. **Multi-Model Integration**
- **v5 Model**: Enhanced model trained on all 117 stocks with better sentiment data
- **v4 Model**: Temporal causality model for ~97 supported stocks
- **Weighted Scoring**: v5 has 50% weight, v4 has 30% weight (when available)

### 2. **Real-Time Sentiment Analysis**
- **EODHD Intraday Sentiment**: Primary sentiment source with 80% weight
- **CSV Momentum Analysis**: Uses `stock_sentiment_dataset_month.csv` for momentum checks (25% weight)
- **Contradiction Detection**: Identifies and handles conflicting signals between sources

### 3. **Technical Indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA 20/50 (Simple Moving Averages)
- Volume Ratio Analysis
- Market Regime Detection (Bullish/Bearish/Neutral)

### 4. **Signal Generation Logic**
```python
# Weighted Score Calculation
if weighted_score > 0.15:
    signal = 'BUY'
elif weighted_score < -0.15:
    signal = 'SELL'
else:
    signal = 'HOLD'
```

### 5. **Risk Management**
- Dynamic risk scoring based on RSI extremes, volatility, and volume anomalies
- Price targets calculated using confidence and volatility
- Stop-loss levels adjusted for risk

## Frontend Features

### Live Signals Page (`/signals`)
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Advanced Filtering**: By signal type, confidence level, and sorting options
- **Market Overview**: Signal distribution, average confidence, market sentiment
- **Signal Cards**: Detailed view with:
  - Current price, target, and stop-loss
  - Model scores (v5, v4)
  - Sentiment analysis and momentum
  - Technical indicators
  - Risk assessment
  - Key drivers

## Backend Architecture

### Enhanced Signal Generator (`src/models/enhanced_signal_generator.py`)
- Loads and manages v5 and v4 models
- Integrates EODHD sentiment analyzer
- Processes CSV sentiment data for momentum
- Handles parallel signal generation
- Manages contradiction detection

### API Endpoint (`/api/signals/live`)
```json
POST /api/signals/live
{
  "use_v5": true,
  "use_v4": true,
  "use_intraday_sentiment": true,
  "include_momentum_check": true
}
```

**Response:**
```json
{
  "success": true,
  "signals": [
    {
      "symbol": "RELIANCE.NSE",
      "signal": "BUY",
      "confidence": 0.782,
      "current_price": 2450.50,
      "price_target": 2548.52,
      "stop_loss": 2401.49,
      "v5_score": 0.723,
      "v4_score": 0.654,
      "intraday_sentiment": 0.342,
      "sentiment_momentum": 0.125,
      "risk_score": 0.35,
      "technical_indicators": {
        "rsi": 58.3,
        "macd": 12.5,
        "volume_ratio": 1.25
      },
      "key_drivers": ["Positive momentum", "Bullish trend"]
    }
  ],
  "stats": {
    "total_signals": 117,
    "buy_signals": 42,
    "sell_signals": 28,
    "hold_signals": 47,
    "avg_confidence": 0.68,
    "market_sentiment": 0.15
  }
}
```

## How to Use

### 1. Start the API Server
```bash
python scripts/start_enhanced_server.py
```

### 2. Test the API
```bash
python scripts/test_live_signals.py
```

### 3. Access the Frontend
Navigate to `http://localhost:3000/signals` after starting the Next.js dev server.

### 4. Monitor Performance
- Check console logs for model loading status
- Monitor API response times
- Review signal quality metrics

## Signal Interpretation

### Confidence Levels
- **High (≥75%)**: Strong signal with multiple confirming factors
- **Medium (60-75%)**: Moderate signal, consider position sizing
- **Low (<60%)**: Weak signal, additional analysis recommended

### Contradiction Handling
When EODHD sentiment contradicts CSV momentum:
- Confidence is reduced by 15%
- "Signal Contradiction" appears in key drivers
- Consider waiting for alignment

### Risk Scores
- **Low (<30%)**: Normal market conditions
- **Medium (30-60%)**: Elevated volatility or technical extremes
- **High (>60%)**: High risk, reduce position size

## Model Details

### v5 Model Features
- Price change patterns
- Volume analysis
- Technical indicators (RSI, MACD)
- Sentiment scores
- News count impact
- Momentum factors

### v4 Model (Temporal Causality)
- Sequential pattern recognition
- Technical indicator adjustments
- Momentum-based predictions
- Available for ~97 stocks only

## Data Sources Priority

1. **EODHD Intraday Data** (Primary)
   - Real-time prices
   - Intraday sentiment scores
   - Technical indicators
   - Market regime detection

2. **CSV Sentiment Data** (Momentum Check)
   - Historical sentiment trends
   - Momentum calculation
   - Volatility assessment
   - Trend identification

3. **Model Predictions** (AI Analysis)
   - v5 comprehensive analysis
   - v4 temporal patterns
   - Ensemble scoring

## Best Practices

1. **Signal Validation**
   - Check for contradiction warnings
   - Verify technical indicator alignment
   - Consider market regime context

2. **Risk Management**
   - Use provided stop-loss levels
   - Size positions based on confidence
   - Monitor risk scores

3. **Timing**
   - Best results during market hours
   - Check sentiment momentum for entry timing
   - Watch for volume confirmations

## Troubleshooting

### Common Issues

1. **"Model not found" errors**
   - Ensure model files exist in `data/models/`
   - Check file paths in enhanced_signal_generator.py

2. **EODHD sentiment failures**
   - Verify API key in environment variables
   - Check rate limits
   - Ensure symbols are formatted correctly (e.g., "RELIANCE.NSE")

3. **Slow performance**
   - Reduce max_workers for parallel processing
   - Check EODHD API response times
   - Consider caching frequently accessed data

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **WebSocket Support**: Real-time signal updates
2. **Alert System**: Push notifications for high-confidence signals
3. **Backtesting Integration**: Historical performance analysis
4. **Portfolio Optimization**: Multi-asset allocation based on signals
5. **Custom Watchlists**: User-defined stock groups

## API Rate Limits

- EODHD: 20 requests per second
- Consider implementing caching for production use
- Batch requests when possible

## Security Notes

- API keys should be stored securely
- Implement authentication for production
- Use HTTPS for all API calls
- Validate all user inputs 