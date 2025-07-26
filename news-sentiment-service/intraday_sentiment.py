#!/usr/bin/env python3
"""
Intraday Sentiment Analysis System for NSE Stocks
=================================================

This system provides real-time sentiment analysis for 117 NSE stocks during market hours.
Features:
- Real-time news monitoring from multiple sources
- Intraday sentiment tracking with timestamps
- Market hours detection (9:15 AM - 3:30 PM IST)
- Sentiment alerts and notifications
- Historical intraday sentiment storage
- Performance analytics and reporting
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import schedule

# News and sentiment imports
import feedparser
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import re

# Configuration
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
TIMEZONE = "Asia/Kolkata"
SENTIMENT_UPDATE_INTERVAL = 5  # minutes
NEWS_FETCH_INTERVAL = 2  # minutes
ALERT_THRESHOLD = 0.6  # Strong sentiment threshold

@dataclass
class IntradaySentiment:
    """Data class for intraday sentiment data"""
    timestamp: str
    symbol: str
    sentiment_score: float
    sentiment_label: str
    news_count: int
    source: str
    confidence: float
    volume_change: Optional[float] = None
    price_change: Optional[float] = None

class IntradaySentimentAnalyzer:
    """Real-time intraday sentiment analysis for NSE stocks"""
    
    def __init__(self, db_path: str = "intraday_sentiment.db"):
        self.db_path = db_path
        self.logger = self._setup_logging()
        self.stocks = self._load_stocks()
        self.sentiment_model = None
        self.tokenizer = None
        self._load_models()
        self.sentiment_queue = Queue()
        self.running = False
        self.current_sentiments = {}
        self.alert_history = []
        
        # Initialize database
        self._init_database()
        
        # News sources for intraday monitoring
        self.news_sources = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/',
            'economictimes': 'https://economictimes.indiatimes.com/rss.cms',
            'livemint': 'https://www.livemint.com/rss/',
            'business_standard': 'https://www.business-standard.com/rss/',
            'ndtv_business': 'https://feeds.feedburner.com/ndtvbusiness-top-stories'
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('intraday_sentiment.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_stocks(self) -> List[str]:
        """Load the 117 NSE stock symbols"""
        stocks = []
        try:
            with open('../../nse_stock_symbols_complete.txt', 'r') as f:
                content = f.read()
                # Extract stock symbols using regex
                import re
                pattern = r'([A-Z]+\.NSE)'
                stocks = re.findall(pattern, content)
                self.logger.info(f"Loaded {len(stocks)} stock symbols")
        except FileNotFoundError:
            self.logger.error("Stock symbols file not found, using default list")
            # Fallback to a subset for testing
            stocks = ['RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE', 'INFY.NSE', 'ICICIBANK.NSE']
        
        return stocks
    
    def _load_models(self):
        """Load FinBERT model for sentiment analysis"""
        try:
            self.logger.info("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _init_database(self):
        """Initialize SQLite database for intraday sentiment storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create intraday sentiment table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    news_count INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    volume_change REAL,
                    price_change REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON intraday_sentiment(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_symbol ON sentiment_alerts(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours (9:15 AM - 3:30 PM IST)"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _extract_stock_mentions(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        mentions = []
        for stock in self.stocks:
            # Remove .NSE suffix for matching
            stock_name = stock.replace('.NSE', '')
            if stock_name.lower() in text.lower():
                mentions.append(stock)
        return mentions
    
    def _analyze_sentiment_finbert(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using FinBERT"""
        if not self.model or not self.tokenizer:
            return 0.0, "neutral", 0.5
        
        try:
            # Clean and prepare text
            text = text[:512]  # Limit length for model
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: negative, neutral, positive
            labels = ['negative', 'neutral', 'positive']
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert to sentiment score (-1 to 1)
            if predicted_class == 0:  # negative
                sentiment_score = -confidence
            elif predicted_class == 2:  # positive
                sentiment_score = confidence
            else:  # neutral
                sentiment_score = 0.0
            
            return sentiment_score, labels[predicted_class], confidence
            
        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {e}")
            return 0.0, "neutral", 0.5
    
    def _analyze_sentiment_textblob(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using TextBlob as fallback"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            confidence = abs(blob.sentiment.subjectivity)
            
            if sentiment_score > 0.1:
                label = "positive"
            elif sentiment_score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return sentiment_score, label, confidence
            
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {e}")
            return 0.0, "neutral", 0.5
    
    def _fetch_intraday_news(self) -> List[Dict]:
        """Fetch real-time news from multiple sources"""
        all_news = []
        
        for source_name, rss_url in self.news_sources.items():
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:10]:  # Limit to recent entries
                    news_item = {
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_name,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Extract stock mentions
                    full_text = f"{news_item['title']} {news_item['description']}"
                    mentions = self._extract_stock_mentions(full_text)
                    
                    if mentions:
                        news_item['stock_mentions'] = mentions
                        all_news.append(news_item)
                
                self.logger.info(f"Fetched {len(feed.entries)} news items from {source_name}")
                
            except Exception as e:
                self.logger.error(f"Error fetching news from {source_name}: {e}")
        
        return all_news
    
    def _analyze_stock_sentiment(self, stock: str, news_items: List[Dict]) -> IntradaySentiment:
        """Analyze sentiment for a specific stock from news items"""
        stock_news = [item for item in news_items if stock in item.get('stock_mentions', [])]
        
        if not stock_news:
            return IntradaySentiment(
                timestamp=datetime.now().isoformat(),
                symbol=stock,
                sentiment_score=0.0,
                sentiment_label="neutral",
                news_count=0,
                source="no_news",
                confidence=0.5
            )
        
        # Combine all news text for analysis
        combined_text = " ".join([
            f"{item['title']} {item['description']}" 
            for item in stock_news
        ])
        
        # Analyze sentiment
        sentiment_score, sentiment_label, confidence = self._analyze_sentiment_finbert(combined_text)
        
        # If FinBERT fails, use TextBlob
        if sentiment_score == 0.0 and confidence == 0.5:
            sentiment_score, sentiment_label, confidence = self._analyze_sentiment_textblob(combined_text)
        
        return IntradaySentiment(
            timestamp=datetime.now().isoformat(),
            symbol=stock,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            news_count=len(stock_news),
            source="news_analysis",
            confidence=confidence
        )
    
    def _save_sentiment_data(self, sentiment: IntradaySentiment):
        """Save sentiment data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO intraday_sentiment 
                (timestamp, symbol, sentiment_score, sentiment_label, news_count, source, confidence, volume_change, price_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sentiment.timestamp,
                sentiment.symbol,
                sentiment.sentiment_score,
                sentiment.sentiment_label,
                sentiment.news_count,
                sentiment.source,
                sentiment.confidence,
                sentiment.volume_change,
                sentiment.price_change
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")
    
    def _check_alerts(self, sentiment: IntradaySentiment):
        """Check for sentiment alerts and save if threshold exceeded"""
        if abs(sentiment.sentiment_score) >= ALERT_THRESHOLD:
            alert_type = "strong_positive" if sentiment.sentiment_score > 0 else "strong_negative"
            message = f"Strong {sentiment.sentiment_label} sentiment detected for {sentiment.symbol}: {sentiment.sentiment_score:.3f}"
            
            # Save alert
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO sentiment_alerts 
                    (timestamp, symbol, alert_type, sentiment_score, message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    sentiment.timestamp,
                    sentiment.symbol,
                    alert_type,
                    sentiment.sentiment_score,
                    message
                ))
                
                conn.commit()
                conn.close()
                
                self.alert_history.append({
                    'timestamp': sentiment.timestamp,
                    'symbol': sentiment.symbol,
                    'alert_type': alert_type,
                    'message': message
                })
                
                self.logger.warning(f"ALERT: {message}")
                
            except Exception as e:
                self.logger.error(f"Error saving alert: {e}")
    
    def _update_sentiment_for_all_stocks(self):
        """Update sentiment for all stocks"""
        if not self._is_market_hours():
            self.logger.info("Outside market hours, skipping sentiment update")
            return
        
        try:
            # Fetch latest news
            news_items = self._fetch_intraday_news()
            self.logger.info(f"Fetched {len(news_items)} news items with stock mentions")
            
            # Analyze sentiment for each stock
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for stock in self.stocks:
                    future = executor.submit(self._analyze_stock_sentiment, stock, news_items)
                    futures.append((stock, future))
                
                # Collect results
                for stock, future in futures:
                    try:
                        sentiment = future.result(timeout=30)
                        self.current_sentiments[stock] = sentiment
                        self._save_sentiment_data(sentiment)
                        self._check_alerts(sentiment)
                        
                    except Exception as e:
                        self.logger.error(f"Error analyzing sentiment for {stock}: {e}")
            
            self.logger.info(f"Updated sentiment for {len(self.current_sentiments)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error in sentiment update: {e}")
    
    def start_intraday_monitoring(self):
        """Start the intraday sentiment monitoring system"""
        self.running = True
        self.logger.info("Starting intraday sentiment monitoring...")
        
        # Schedule regular updates
        schedule.every(NEWS_FETCH_INTERVAL).minutes.do(self._update_sentiment_for_all_stocks)
        
        # Main monitoring loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Stopping intraday monitoring...")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_current_sentiments(self) -> Dict[str, IntradaySentiment]:
        """Get current sentiment data for all stocks"""
        return self.current_sentiments
    
    def get_sentiment_summary(self) -> Dict:
        """Get summary of current sentiment distribution"""
        if not self.current_sentiments:
            return {"message": "No sentiment data available"}
        
        positive_count = sum(1 for s in self.current_sentiments.values() if s.sentiment_score > 0.1)
        negative_count = sum(1 for s in self.current_sentiments.values() if s.sentiment_score < -0.1)
        neutral_count = len(self.current_sentiments) - positive_count - negative_count
        
        avg_sentiment = np.mean([s.sentiment_score for s in self.current_sentiments.values()])
        
        return {
            "total_stocks": len(self.current_sentiments),
            "positive_sentiment": positive_count,
            "negative_sentiment": negative_count,
            "neutral_sentiment": neutral_count,
            "average_sentiment": avg_sentiment,
            "strong_positive": sum(1 for s in self.current_sentiments.values() if s.sentiment_score > ALERT_THRESHOLD),
            "strong_negative": sum(1 for s in self.current_sentiments.values() if s.sentiment_score < -ALERT_THRESHOLD)
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent sentiment alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, symbol, alert_type, sentiment_score, message
                FROM sentiment_alerts
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (cutoff_time,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'timestamp': row[0],
                    'symbol': row[1],
                    'alert_type': row[2],
                    'sentiment_score': row[3],
                    'message': row[4]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error fetching alerts: {e}")
            return []
    
    def export_sentiment_data(self, output_file: str = "intraday_sentiment_export.csv"):
        """Export sentiment data to CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM intraday_sentiment 
                ORDER BY timestamp DESC
            ''', conn)
            conn.close()
            
            df.to_csv(output_file, index=False)
            self.logger.info(f"Exported sentiment data to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")

def main():
    """Main function to run intraday sentiment analysis"""
    analyzer = IntradaySentimentAnalyzer()
    
    print("=== Intraday Sentiment Analysis System ===")
    print(f"Monitoring {len(analyzer.stocks)} NSE stocks")
    print(f"Market hours: {MARKET_OPEN_TIME} - {MARKET_CLOSE_TIME} IST")
    print(f"Update interval: {NEWS_FETCH_INTERVAL} minutes")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        analyzer.start_intraday_monitoring()
    except KeyboardInterrupt:
        print("\nStopping intraday sentiment analysis...")
        
        # Show final summary
        summary = analyzer.get_sentiment_summary()
        print("\n=== Final Sentiment Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Export data
        analyzer.export_sentiment_data()

if __name__ == "__main__":
    main() 