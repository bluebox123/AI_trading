#!/usr/bin/env python3
"""
Integrated Sentiment Service for Signal Generation
=================================================

This service provides comprehensive sentiment analysis for the trading signals system.
It runs comprehensive analysis on startup and then provides cached sentiment data
for fast signal generation.
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import the comprehensive analyzer
from comprehensive_sentiment_analyzer import ComprehensiveSentimentAnalyzer

class IntegratedSentimentService:
    """
    Integrated sentiment service for signal generation
    - Runs comprehensive analysis periodically (every hour)
    - Provides cached sentiment data for fast signal generation
    - Supports both batch and individual stock queries
    """
    
    def __init__(self, update_interval_hours: int = 1):
        self.logger = self._setup_logging()
        self.update_interval_hours = update_interval_hours
        self.sentiment_cache = {}
        self.last_update = None
        self.is_updating = False
        self.analyzer = None
        self.cache_file = "sentiment_cache.json"
        
        # Initialize the comprehensive analyzer
        self._initialize_analyzer()
        
        # Load existing cache if available
        self._load_cache()
        
        # Start background update task
        self._start_background_updates()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [SENTIMENT] %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_analyzer(self):
        """Initialize the comprehensive sentiment analyzer"""
        try:
            self.analyzer = ComprehensiveSentimentAnalyzer()
            self.logger.info("Comprehensive sentiment analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize analyzer: {e}")
            self.analyzer = None
    
    def _load_cache(self):
        """Load existing sentiment cache"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.sentiment_cache = cache_data.get('sentiment_data', {})
                    self.last_update = cache_data.get('last_update')
                    self.logger.info(f"Loaded sentiment cache with {len(self.sentiment_cache)} stocks")
                    if self.last_update:
                        self.logger.info(f"Last update: {self.last_update}")
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self.sentiment_cache = {}
    
    def _save_cache(self):
        """Save sentiment cache to file"""
        try:
            cache_data = {
                'sentiment_data': self.sentiment_cache,
                'last_update': self.last_update,
                'update_interval_hours': self.update_interval_hours
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info(f"Sentiment cache saved with {len(self.sentiment_cache)} stocks")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _start_background_updates(self):
        """Start background thread for periodic updates"""
        def update_loop():
            while True:
                try:
                    # Check if update is needed
                    if self._should_update():
                        self.logger.info("Starting background sentiment update...")
                        self._update_sentiment_data()
                    
                    # Wait before next check (check every 10 minutes)
                    time.sleep(600)
                    
                except Exception as e:
                    self.logger.error(f"Error in background update loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        # Start background thread
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        self.logger.info("Background sentiment update thread started")
    
    def _should_update(self) -> bool:
        """Check if sentiment data should be updated"""
        if not self.last_update:
            return True
        
        if self.is_updating:
            return False
        
        try:
            last_update_time = datetime.fromisoformat(self.last_update)
            time_since_update = datetime.now() - last_update_time
            return time_since_update.total_seconds() > (self.update_interval_hours * 3600)
        except:
            return True
    
    def _update_sentiment_data(self):
        """Update sentiment data using comprehensive analysis"""
        if self.is_updating:
            self.logger.info("Update already in progress, skipping...")
            return
        
        if not self.analyzer:
            self.logger.error("Analyzer not available, skipping update")
            return
        
        try:
            self.is_updating = True
            self.logger.info("Starting comprehensive sentiment analysis...")
            
            # Run comprehensive analysis
            results = self.analyzer.analyze_all_stocks_comprehensive()
            
            # Convert to cache format
            new_cache = {}
            for result in results:
                symbol = result['symbol']
                new_cache[symbol] = {
                    'sentiment_score': result['sentiment_score'],
                    'sentiment_label': result['sentiment_label'],
                    'confidence': result['confidence'],
                    'news_count': result['news_count'],
                    'source': result['source'],
                    'sample_headlines': result.get('sample_headlines', []),
                    'sector': result.get('sector', 'Other'),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update cache
            self.sentiment_cache = new_cache
            self.last_update = datetime.now().isoformat()
            
            # Save to file
            self._save_cache()
            
            self.logger.info(f"Sentiment data updated for {len(new_cache)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment data: {e}")
        finally:
            self.is_updating = False
    
    def get_sentiment(self, symbol: str) -> Dict:
        """Get sentiment data for a single stock"""
        if symbol in self.sentiment_cache:
            return self.sentiment_cache[symbol]
        
        # Return neutral sentiment if not found
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.3,
            'news_count': 0,
            'source': 'default',
            'sample_headlines': [],
            'sector': 'Other',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_sentiment_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get sentiment data for multiple stocks"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_sentiment(symbol)
        return results
    
    def get_all_sentiment(self) -> Dict[str, Dict]:
        """Get sentiment data for all stocks"""
        return self.sentiment_cache.copy()
    
    def get_market_sentiment_summary(self) -> Dict:
        """Get overall market sentiment summary"""
        if not self.sentiment_cache:
            return {
                'total_stocks': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_sentiment': 0.0,
                'high_confidence_count': 0,
                'last_update': None
            }
        
        sentiments = list(self.sentiment_cache.values())
        sentiment_scores = [s['sentiment_score'] for s in sentiments]
        
        positive_count = sum(1 for s in sentiment_scores if s > 0.1)
        negative_count = sum(1 for s in sentiment_scores if s < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        high_confidence_count = sum(1 for s in sentiments if s['confidence'] > 0.7)
        
        return {
            'total_stocks': len(sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'average_sentiment': round(np.mean(sentiment_scores), 4),
            'high_confidence_count': high_confidence_count,
            'last_update': self.last_update,
            'positive_percentage': round(positive_count / len(sentiments) * 100, 1),
            'negative_percentage': round(negative_count / len(sentiments) * 100, 1),
            'neutral_percentage': round(neutral_count / len(sentiments) * 100, 1)
        }
    
    def get_sector_sentiment(self) -> Dict[str, Dict]:
        """Get sentiment summary by sector"""
        if not self.sentiment_cache:
            return {}
        
        sector_data = {}
        for symbol, data in self.sentiment_cache.items():
            sector = data.get('sector', 'Other')
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(data['sentiment_score'])
        
        sector_summary = {}
        for sector, scores in sector_data.items():
            sector_summary[sector] = {
                'average_sentiment': round(np.mean(scores), 4),
                'stock_count': len(scores),
                'positive_count': sum(1 for s in scores if s > 0.1),
                'negative_count': sum(1 for s in scores if s < -0.1),
                'neutral_count': len(scores) - sum(1 for s in scores if abs(s) > 0.1)
            }
        
        return sector_summary
    
    def force_update(self) -> bool:
        """Force an immediate update of sentiment data"""
        if self.is_updating:
            self.logger.info("Update already in progress")
            return False
        
        self.logger.info("Forcing sentiment data update...")
        self._update_sentiment_data()
        return True
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'is_updating': self.is_updating,
            'last_update': self.last_update,
            'cached_stocks': len(self.sentiment_cache),
            'update_interval_hours': self.update_interval_hours,
            'analyzer_available': self.analyzer is not None,
            'next_update_due': self._get_next_update_time()
        }
    
    def _get_next_update_time(self) -> Optional[str]:
        """Get the next scheduled update time"""
        if not self.last_update:
            return "ASAP"
        
        try:
            last_update_time = datetime.fromisoformat(self.last_update)
            next_update = last_update_time + timedelta(hours=self.update_interval_hours)
            return next_update.isoformat()
        except:
            return "Unknown"

# Global instance
sentiment_service = None

def get_sentiment_service() -> IntegratedSentimentService:
    """Get the global sentiment service instance"""
    global sentiment_service
    if sentiment_service is None:
        sentiment_service = IntegratedSentimentService()
    return sentiment_service

def initialize_sentiment_service(update_interval_hours: int = 1) -> IntegratedSentimentService:
    """Initialize the global sentiment service"""
    global sentiment_service
    sentiment_service = IntegratedSentimentService(update_interval_hours)
    return sentiment_service

if __name__ == "__main__":
    # Test the service
    print("🚀 Testing Integrated Sentiment Service...")
    
    service = IntegratedSentimentService(update_interval_hours=1)
    
    # Force an update
    print("📊 Forcing sentiment update...")
    service.force_update()
    
    # Get some sample data
    print("\n📈 Sample sentiment data:")
    sample_stocks = ['RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE']
    for stock in sample_stocks:
        sentiment = service.get_sentiment(stock)
        print(f"{stock}: {sentiment['sentiment_score']:.4f} ({sentiment['sentiment_label']})")
    
    # Get market summary
    print("\n🌍 Market sentiment summary:")
    summary = service.get_market_sentiment_summary()
    print(f"Total stocks: {summary['total_stocks']}")
    print(f"Positive: {summary['positive_count']} ({summary['positive_percentage']}%)")
    print(f"Negative: {summary['negative_count']} ({summary['negative_percentage']}%)")
    print(f"Neutral: {summary['neutral_count']} ({summary['neutral_percentage']}%)")
    print(f"Average sentiment: {summary['average_sentiment']}")
    
    print("\n✅ Service test completed!") 