#!/usr/bin/env python3
"""
Intraday Sentiment Aggregation System
====================================

This module provides intraday aggregation of sentiment data into different
time buckets (15-min, hourly, etc.) with comprehensive statistical measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, time
import logging
from pathlib import Path
import json
from collections import defaultdict

@dataclass
class IntradayBucket:
    """Data class for intraday sentiment bucket"""
    start_time: datetime
    end_time: datetime
    bucket_type: str
    sentiment_scores: List[float]
    confidences: List[float]
    article_count: int
    avg_sentiment: float
    median_sentiment: float
    sentiment_std: float
    weighted_sentiment: float
    confidence_weighted_sentiment: float
    positive_count: int
    negative_count: int
    neutral_count: int
    volume_score: float
    momentum_score: float
    volatility_score: float
    articles: List[dict] = None

class IntradayAggregator:
    """Intraday sentiment aggregation system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the intraday aggregator"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.aggregation_cache = {}
        
        self.logger.info("Intraday Aggregator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the aggregator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for intraday aggregation"""
        default_config = {
            "bucket_types": {
                "15min": {"minutes": 15, "enabled": True},
                "30min": {"minutes": 30, "enabled": True},
                "1hour": {"minutes": 60, "enabled": True},
                "2hour": {"minutes": 120, "enabled": True},
                "4hour": {"minutes": 240, "enabled": True},
                "daily": {"minutes": 1440, "enabled": True}
            },
            "trading_hours": {
                "start": "09:15",
                "end": "15:30",
                "timezone": "Asia/Kolkata"
            },
            "statistical_measures": {
                "include_momentum": True,
                "include_volatility": True,
                "include_volume": True,
                "confidence_weighting": True
            },
            "caching": {
                "enable_cache": True,
                "cache_duration_hours": 24
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def create_time_buckets(self, start_time: datetime, end_time: datetime, 
                           bucket_type: str) -> List[Tuple[datetime, datetime]]:
        """Create time buckets for aggregation"""
        bucket_config = self.config["bucket_types"].get(bucket_type)
        if not bucket_config or not bucket_config["enabled"]:
            return []
        
        minutes = bucket_config["minutes"]
        buckets = []
        current_time = start_time
        
        while current_time < end_time:
            bucket_start = current_time
            bucket_end = current_time + timedelta(minutes=minutes)
            
            # Ensure bucket doesn't exceed end_time
            if bucket_end > end_time:
                bucket_end = end_time
            
            buckets.append((bucket_start, bucket_end))
            current_time = bucket_end
        
        return buckets
    
    def is_trading_hour(self, dt: datetime) -> bool:
        """Check if datetime is within trading hours"""
        trading_start = time(9, 15)  # 9:15 AM
        trading_end = time(15, 30)   # 3:30 PM
        
        current_time = dt.time()
        return trading_start <= current_time <= trading_end
    
    def aggregate_sentiment_data(self, 
                               sentiment_data: List[Dict],
                               bucket_type: str = "1hour",
                               stock_symbol: str = None) -> List[IntradayBucket]:
        """Aggregate sentiment data into time buckets"""
        
        if not sentiment_data:
            return []
        
        # Convert sentiment data to DataFrame for easier processing
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Get time range
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        
        # Create time buckets
        buckets = self.create_time_buckets(start_time, end_time, bucket_type)
        
        aggregated_buckets = []
        
        for bucket_start, bucket_end in buckets:
            # Filter data for this bucket
            mask = (df['timestamp'] >= bucket_start) & (df['timestamp'] < bucket_end)
            bucket_data = df[mask]
            
            if len(bucket_data) == 0:
                continue
            
            # Extract sentiment scores and confidences
            sentiment_scores = bucket_data['sentiment_score'].tolist()
            confidences = bucket_data.get('confidence', [0.5] * len(bucket_data)).tolist()
            
            # Calculate basic statistics
            avg_sentiment = np.mean(sentiment_scores)
            median_sentiment = np.median(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            # Calculate weighted sentiment
            if self.config["statistical_measures"]["confidence_weighting"]:
                weighted_sentiment = np.average(sentiment_scores, weights=confidences)
                confidence_weighted_sentiment = weighted_sentiment
            else:
                weighted_sentiment = avg_sentiment
                confidence_weighted_sentiment = avg_sentiment
            
            # Count sentiment categories
            positive_count = sum(1 for s in sentiment_scores if s > 0.1)
            negative_count = sum(1 for s in sentiment_scores if s < -0.1)
            neutral_count = sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1)
            
            # Calculate advanced metrics
            volume_score = self._calculate_volume_score(bucket_data)
            momentum_score = self._calculate_momentum_score(sentiment_scores)
            volatility_score = self._calculate_volatility_score(sentiment_scores)
            
            bucket = IntradayBucket(
                start_time=bucket_start,
                end_time=bucket_end,
                bucket_type=bucket_type,
                sentiment_scores=sentiment_scores,
                confidences=confidences,
                article_count=len(bucket_data),
                avg_sentiment=avg_sentiment,
                median_sentiment=median_sentiment,
                sentiment_std=sentiment_std,
                weighted_sentiment=weighted_sentiment,
                confidence_weighted_sentiment=confidence_weighted_sentiment,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                volume_score=volume_score,
                momentum_score=momentum_score,
                volatility_score=volatility_score,
                articles=bucket_data.to_dict('records')
            )
            
            aggregated_buckets.append(bucket)
        
        return aggregated_buckets
    
    def _calculate_volume_score(self, bucket_data: pd.DataFrame) -> float:
        """Calculate volume score based on article count and engagement"""
        if len(bucket_data) == 0:
            return 0.0
        
        # Base volume on article count
        article_count = len(bucket_data)
        
        # Normalize to 0-1 scale (assuming max 100 articles per bucket is high volume)
        volume_score = min(article_count / 100.0, 1.0)
        
        return volume_score
    
    def _calculate_momentum_score(self, sentiment_scores: List[float]) -> float:
        """Calculate momentum score based on sentiment trend"""
        if len(sentiment_scores) < 2:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(sentiment_scores))
        y = np.array(sentiment_scores)
        
        # Simple linear trend
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to -1 to 1 range
        momentum_score = np.tanh(slope * 10)  # Scale factor of 10
        
        return momentum_score
    
    def _calculate_volatility_score(self, sentiment_scores: List[float]) -> float:
        """Calculate volatility score based on sentiment variance"""
        if len(sentiment_scores) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_sentiment = np.mean(sentiment_scores)
        std_sentiment = np.std(sentiment_scores)
        
        if abs(mean_sentiment) < 1e-6:
            volatility_score = std_sentiment
        else:
            volatility_score = std_sentiment / abs(mean_sentiment)
        
        # Normalize to 0-1 range
        volatility_score = min(volatility_score, 1.0)
        
        return volatility_score
    
    def aggregate_multiple_buckets(self, 
                                 sentiment_data: List[Dict],
                                 bucket_types: List[str] = None) -> Dict[str, List[IntradayBucket]]:
        """Aggregate sentiment data into multiple bucket types"""
        
        if bucket_types is None:
            bucket_types = [k for k, v in self.config["bucket_types"].items() if v["enabled"]]
        
        results = {}
        
        for bucket_type in bucket_types:
            try:
                buckets = self.aggregate_sentiment_data(sentiment_data, bucket_type)
                results[bucket_type] = buckets
                self.logger.info(f"Aggregated {len(buckets)} {bucket_type} buckets")
            except Exception as e:
                self.logger.error(f"Error aggregating {bucket_type} buckets: {e}")
                results[bucket_type] = []
        
        return results
    
    def calculate_intraday_metrics(self, buckets: List[IntradayBucket]) -> Dict[str, float]:
        """Calculate comprehensive intraday metrics"""
        if not buckets:
            return {}
        
        # Extract metrics from buckets
        sentiments = [b.weighted_sentiment for b in buckets]
        volumes = [b.volume_score for b in buckets]
        momentums = [b.momentum_score for b in buckets]
        volatilities = [b.volatility_score for b in buckets]
        
        metrics = {
            'total_buckets': len(buckets),
            'avg_sentiment': np.mean(sentiments),
            'sentiment_trend': np.polyfit(range(len(sentiments)), sentiments, 1)[0],
            'sentiment_volatility': np.std(sentiments),
            'avg_volume': np.mean(volumes),
            'avg_momentum': np.mean(momentums),
            'avg_volatility': np.mean(volatilities),
            'peak_sentiment': max(sentiments),
            'trough_sentiment': min(sentiments),
            'sentiment_range': max(sentiments) - min(sentiments),
            'positive_buckets': sum(1 for s in sentiments if s > 0),
            'negative_buckets': sum(1 for s in sentiments if s < 0),
            'neutral_buckets': sum(1 for s in sentiments if abs(s) <= 0.1)
        }
        
        return metrics
    
    def generate_intraday_report(self, 
                               buckets: List[IntradayBucket],
                               stock_symbol: str = None) -> Dict[str, Any]:
        """Generate comprehensive intraday sentiment report"""
        
        if not buckets:
            return {"error": "No buckets to analyze"}
        
        # Calculate metrics
        metrics = self.calculate_intraday_metrics(buckets)
        
        # Generate time series data
        time_series = []
        for bucket in buckets:
            time_series.append({
                'timestamp': bucket.start_time.isoformat(),
                'sentiment': bucket.weighted_sentiment,
                'volume': bucket.volume_score,
                'momentum': bucket.momentum_score,
                'volatility': bucket.volatility_score,
                'article_count': bucket.article_count
            })
        
        # Identify key events
        key_events = self._identify_key_events(buckets)
        
        report = {
            'stock_symbol': stock_symbol,
            'bucket_type': buckets[0].bucket_type if buckets else None,
            'time_range': {
                'start': buckets[0].start_time.isoformat() if buckets else None,
                'end': buckets[-1].end_time.isoformat() if buckets else None
            },
            'metrics': metrics,
            'time_series': time_series,
            'key_events': key_events,
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _identify_key_events(self, buckets: List[IntradayBucket]) -> List[Dict]:
        """Identify key events in the intraday data"""
        events = []
        
        if len(buckets) < 2:
            return events
        
        # Find sentiment spikes
        sentiments = [b.weighted_sentiment for b in buckets]
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)
        
        for i, bucket in enumerate(buckets):
            sentiment = bucket.weighted_sentiment
            z_score = (sentiment - mean_sentiment) / std_sentiment if std_sentiment > 0 else 0
            
            if abs(z_score) > 2.0:  # Significant spike
                events.append({
                    'type': 'sentiment_spike',
                    'timestamp': bucket.start_time.isoformat(),
                    'sentiment': sentiment,
                    'z_score': z_score,
                    'direction': 'positive' if sentiment > mean_sentiment else 'negative'
                })
            
            # Volume spike
            if bucket.volume_score > 0.8:
                events.append({
                    'type': 'volume_spike',
                    'timestamp': bucket.start_time.isoformat(),
                    'volume_score': bucket.volume_score,
                    'article_count': bucket.article_count
                })
        
        return events
    
    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """Generate natural language summary of intraday metrics"""
        
        sentiment_trend = metrics.get('sentiment_trend', 0)
        avg_sentiment = metrics.get('avg_sentiment', 0)
        volatility = metrics.get('sentiment_volatility', 0)
        
        summary_parts = []
        
        # Overall sentiment
        if avg_sentiment > 0.1:
            summary_parts.append("Overall sentiment was positive")
        elif avg_sentiment < -0.1:
            summary_parts.append("Overall sentiment was negative")
        else:
            summary_parts.append("Overall sentiment was neutral")
        
        # Trend
        if sentiment_trend > 0.01:
            summary_parts.append("with an upward trend")
        elif sentiment_trend < -0.01:
            summary_parts.append("with a downward trend")
        else:
            summary_parts.append("with stable sentiment")
        
        # Volatility
        if volatility > 0.3:
            summary_parts.append("and high volatility")
        elif volatility > 0.1:
            summary_parts.append("and moderate volatility")
        else:
            summary_parts.append("and low volatility")
        
        return " ".join(summary_parts) + "."

def main():
    """Test the intraday aggregation system"""
    aggregator = IntradayAggregator()
    
    # Generate sample sentiment data
    base_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
    sentiment_data = []
    
    # Create sample data for 8 hours of trading
    for i in range(32):  # 32 15-minute intervals
        timestamp = base_time + timedelta(minutes=15 * i)
        
        # Simulate varying sentiment
        sentiment = 0.1 * np.sin(i * 0.5) + 0.05 * np.random.randn()
        confidence = 0.5 + 0.3 * np.random.random()
        
        sentiment_data.append({
            'timestamp': timestamp.isoformat(),
            'sentiment_score': sentiment,
            'confidence': confidence,
            'article_count': np.random.randint(1, 10)
        })
    
    print("Testing Intraday Aggregation System")
    print("=" * 50)
    
    # Test different bucket types
    bucket_types = ['15min', '1hour', '2hour']
    
    for bucket_type in bucket_types:
        print(f"\nAggregating {bucket_type} buckets...")
        
        buckets = aggregator.aggregate_sentiment_data(sentiment_data, bucket_type)
        
        if buckets:
            report = aggregator.generate_intraday_report(buckets, "RELIANCE.NSE")
            
            print(f"Generated {len(buckets)} {bucket_type} buckets")
            print(f"Average sentiment: {report['metrics']['avg_sentiment']:.3f}")
            print(f"Sentiment trend: {report['metrics']['sentiment_trend']:.3f}")
            print(f"Volatility: {report['metrics']['sentiment_volatility']:.3f}")
            print(f"Summary: {report['summary']}")

if __name__ == "__main__":
    main() 