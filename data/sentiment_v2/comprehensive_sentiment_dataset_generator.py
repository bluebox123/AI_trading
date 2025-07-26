#!/usr/bin/env python3
"""
Comprehensive Sentiment Dataset Generator with Advanced Features
=============================================================

This module generates comprehensive sentiment datasets for NSE stocks using
Google News RSS feeds with advanced features:
- Advanced NLP processing (language detection, NER, multi-model sentiment)
- Confidence calibration across all models
- Intraday aggregation with multiple time buckets
- Windowed boosting and custom decay profiles
- Temporal weighting with exponential decay
"""

import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import json
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import re
from urllib.parse import quote_plus
import hashlib
from dataclasses import dataclass
import math
from data_quality_manager import DataQualityManager
from backfill_manager import BackfillManager
from dateutil import parser as date_parser

# Import advanced modules
from advanced_nlp_processor import AdvancedNLPProcessor
from advanced_confidence_calibration import ConfidenceCalibrator, CalibratedConfidence
from intraday_aggregation import IntradayAggregator, IntradayBucket
from advanced_temporal_weighting import AdvancedTemporalWeighter, WeightedSentiment

@dataclass
class AdvancedSentimentResult:
    """Data class for advanced sentiment analysis result"""
    stock_symbol: str
    article_date: datetime
    sentiment_score: float
    confidence_score: float
    calibrated_confidence: CalibratedConfidence
    nlp_features: Dict[str, Any]
    temporal_weight: WeightedSentiment
    intraday_bucket: Optional[IntradayBucket]
    model_breakdown: Dict[str, float]
    language_detected: str
    entities_found: List[str]
    processed_text: str

class ComprehensiveSentimentGenerator:
    """Comprehensive sentiment dataset generator with advanced features"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the comprehensive sentiment generator"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize advanced components
        self.nlp_processor = AdvancedNLPProcessor()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.intraday_aggregator = IntradayAggregator()
        self.temporal_weighter = AdvancedTemporalWeighter()
        
        # Load stock symbols
        self.stock_symbols = self._load_stock_symbols()
        
        # Initialize caches
        self.sentiment_cache = {}
        self.aggregation_cache = {}
        
        self.data_quality = DataQualityManager(data_dir=".")
        
        self.logger.info("Comprehensive Sentiment Generator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the generator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for the generator"""
        default_config = {
            "google_news": {
                "base_url": "https://news.google.com/rss/search",
                "max_articles_per_stock": 100,
                "date_range_days": 1825,  # 5 years
                "rate_limit_delay": 1.0
            },
            "advanced_features": {
                "confidence_calibration": True,
                "intraday_aggregation": True,
                "windowed_boosting": True,
                "custom_decay_profiles": True,
                "multi_model_sentiment": True,
                "language_detection": True,
                "ner_extraction": True
            },
            "aggregation_settings": {
                "bucket_types": ["15min", "1hour", "2hour", "daily"],
                "trading_hours_only": True
            },
            "temporal_weighting": {
                "default_half_life_days": 7.0,
                "window_boost_enabled": True,
                "adaptive_weighting": True
            },
            "output_formats": {
                "raw_sentiment": True,
                "temporal_weighted": True,
                "intraday_aggregated": True,
                "calibrated_confidence": True
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
    
    def _load_stock_symbols(self) -> List[str]:
        """Load NSE stock symbols, skipping headers, dividers, and duplicates"""
        symbols_file = Path("../../nse_stock_symbols_complete.txt")
        if symbols_file.exists():
            seen = set()
            symbols = []
            with open(symbols_file, 'r') as f:
                for line in f:
                    symbol = line.strip()
                    if symbol.endswith('.NSE') and symbol not in seen:
                        symbols.append(symbol)
                        seen.add(symbol)
            self.logger.info(f"Loaded {len(symbols)} stock symbols")
            return symbols
        else:
            # Fallback to a subset for testing
            return ["RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE", "INFY.NSE", "ICICIBANK.NSE"]
    
    def fetch_news_for_stock(self, stock_symbol: str, start_date: date, end_date: date) -> List[Dict]:
        """Fetch news articles for a specific stock with rate limiting - ENHANCED VERSION"""
        
        # Clean stock symbol for search
        clean_symbol = stock_symbol.replace('.NSE', '').replace('.BSE', '')
        
        # Get company name variations for better search
        company_variations = {
            'ADANIENT': ['Adani Enterprises', 'Adani Group'],
            'RELIANCE': ['Reliance Industries', 'RIL', 'Reliance'],
            'TCS': ['Tata Consultancy Services', 'TCS'],
            'HDFCBANK': ['HDFC Bank', 'HDFC'],
            'INFY': ['Infosys', 'Infosys Limited'],
            'ICICIBANK': ['ICICI Bank', 'ICICI'],
            'HINDUNILVR': ['Hindustan Unilever', 'HUL'],
            'ITC': ['ITC Limited', 'ITC'],
            'SBIN': ['State Bank of India', 'SBI'],
            'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
            'KOTAKBANK': ['Kotak Mahindra Bank', 'Kotak Bank'],
            'LT': ['Larsen & Toubro', 'L&T'],
            'ASIANPAINT': ['Asian Paints'],
            'MARUTI': ['Maruti Suzuki', 'Maruti'],
            'BAJFINANCE': ['Bajaj Finance'],
            'HCLTECH': ['HCL Technologies', 'HCL Tech'],
            'AXISBANK': ['Axis Bank', 'Axis'],
            'ULTRACEMCO': ['UltraTech Cement'],
            'SUNPHARMA': ['Sun Pharmaceutical', 'Sun Pharma'],
            'TITAN': ['Titan Company', 'Titan'],
            'TECHM': ['Tech Mahindra'],
            'POWERGRID': ['Power Grid Corporation'],
            'NTPC': ['NTPC Limited'],
            'ONGC': ['Oil and Natural Gas Corporation'],
            'COALINDIA': ['Coal India'],
            'WIPRO': ['Wipro Limited', 'Wipro'],
            'TATAMOTORS': ['Tata Motors'],
            'TATASTEEL': ['Tata Steel'],
            'JSWSTEEL': ['JSW Steel'],
            'HINDALCO': ['Hindalco Industries'],
            'ADANIPORTS': ['Adani Ports'],
            'BPCL': ['Bharat Petroleum'],
            'BRITANNIA': ['Britannia Industries'],
            'CIPLA': ['Cipla Limited'],
            'DIVISLAB': ['Divi\'s Laboratories'],
            'DRREDDY': ['Dr. Reddy\'s Laboratories'],
            'EICHERMOT': ['Eicher Motors'],
            'GRASIM': ['Grasim Industries'],
            'HEROMOTOCO': ['Hero MotoCorp'],
            'INDUSINDBK': ['IndusInd Bank'],
            'NESTLEIND': ['Nestle India'],
            'SHREECEM': ['Shree Cement'],
            'UPL': ['UPL Limited'],
            'APOLLOHOSP': ['Apollo Hospitals'],
            'BAJAJ-AUTO': ['Bajaj Auto'],
            'GODREJCP': ['Godrej Consumer Products'],
            'PIDILITIND': ['Pidilite Industries'],
            'TATACONSUM': ['Tata Consumer Products'],
            'DMART': ['Avenue Supermarts', 'DMart']
        }
        
        # Get company names for this stock
        company_names = company_variations.get(clean_symbol, [clean_symbol])
        
        # Create date range
        current_date = start_date
        all_articles = []
        
        while current_date <= end_date:
            try:
                # Try multiple search queries for better coverage
                search_queries = []
                
                # Add company name searches
                for company_name in company_names[:2]:  # Limit to 2 company names
                    search_queries.extend([
                        f"{company_name} stock news {current_date.strftime('%Y-%m-%d')}",
                        f"{company_name} company news {current_date.strftime('%Y-%m-%d')}",
                        f"{company_name} financial news {current_date.strftime('%Y-%m-%d')}",
                        f"{company_name} {current_date.strftime('%Y-%m-%d')}"
                    ])
                
                # Add symbol-based searches
                search_queries.extend([
                    f"{clean_symbol} stock news {current_date.strftime('%Y-%m-%d')}",
                    f"{clean_symbol} NSE {current_date.strftime('%Y-%m-%d')}",
                    f"{clean_symbol} share price {current_date.strftime('%Y-%m-%d')}"
                ])
                
                articles_found = False
                
                for query in search_queries[:5]:  # Limit to 5 queries per day
                    try:
                        encoded_query = quote_plus(query)
                        
                        # Construct RSS URL - FIXED FORMAT
                        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
                        
                        # Fetch RSS feed
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        
                        # Parse RSS feed
                        feed = feedparser.parse(response.content)
                        
                        # Check if we got any entries
                        if feed.entries:
                            self.logger.info(f"Found {len(feed.entries)} articles for {stock_symbol} on {current_date} with query: {query}")
                            articles_found = True
                            
                            # Extract articles
                            for entry in feed.entries[:self.config['google_news']['max_articles_per_stock']]:
                                article = {
                                    'title': entry.get('title', ''),
                                    'link': entry.get('link', ''),
                                    'published': entry.get('published', ''),
                                    'summary': entry.get('summary', ''),
                                    'stock_symbol': stock_symbol,
                                    'search_date': current_date.isoformat(),
                                    'search_query': query
                                }
                                all_articles.append(article)
                            
                            # If we found articles, no need to try other queries
                            break
                        else:
                            self.logger.debug(f"No articles found for {stock_symbol} on {current_date} with query: {query}")
                    
                    except Exception as e:
                        self.logger.debug(f"Error with query '{query}' for {stock_symbol} on {current_date}: {e}")
                        continue
                
                if not articles_found:
                    self.logger.warning(f"No data for {stock_symbol} on {current_date}")
                
                # Rate limiting
                time.sleep(self.config['google_news']['rate_limit_delay'])
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                self.logger.error(f"Error fetching news for {stock_symbol} on {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        self.logger.info(f"Total articles found for {stock_symbol}: {len(all_articles)}")
        return all_articles
    
    def process_article_with_advanced_nlp(self, article: Dict) -> AdvancedSentimentResult:
        """Process article with advanced NLP features"""
        
        # Combine title and summary
        text = f"{article['title']} {article['summary']}"
        
        # Advanced NLP processing
        nlp_result = self.nlp_processor.process_text(text)
        
        # Get sentiment from multiple models
        sentiment_results, ensemble_score, confidence = self.nlp_processor.ensemble_sentiment_analysis(text)
        
        # Calibrate confidence
        calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
            sentiment_score=ensemble_score,
            raw_confidence=confidence,
            model_name="advanced_nlp_ensemble"
        )
        
        # Robust date parsing with timezone handling
        try:
            article_date = date_parser.parse(article['published'])
            # Ensure both dates are timezone-aware
            if article_date.tzinfo is None:
                article_date = article_date.replace(tzinfo=timezone.utc)
        except Exception:
            article_date = datetime.now(timezone.utc)
        
        target_date = datetime.now(timezone.utc)
        
        temporal_weight = self.temporal_weighter.calculate_temporal_weight(
            article_date=article_date,
            target_date=target_date,
            stock_symbol=article['stock_symbol'],
            confidence=calibrated_confidence.calibrated_confidence
        )
        
        # Set weighted sentiment
        temporal_weight.weighted_sentiment = ensemble_score * temporal_weight.final_weight
        
        return AdvancedSentimentResult(
            stock_symbol=article['stock_symbol'],
            article_date=article_date,
            sentiment_score=ensemble_score,
            confidence_score=confidence,
            calibrated_confidence=calibrated_confidence,
            nlp_features=nlp_result,
            temporal_weight=temporal_weight,
            intraday_bucket=None,  # Will be set during aggregation
            model_breakdown=sentiment_results,
            language_detected=nlp_result.get('language', 'unknown'),
            entities_found=nlp_result.get('entities', []),
            processed_text=nlp_result.get('processed_text', text)
        )
    
    def aggregate_intraday_data(self, sentiment_results: List[AdvancedSentimentResult], 
                               stock_symbol: str) -> Dict[str, List[IntradayBucket]]:
        """Aggregate sentiment data into intraday buckets with enhanced temporal features"""
        
        if not self.config['advanced_features']['intraday_aggregation']:
            return {}
        
        # Convert to format expected by aggregator with enhanced temporal weighting
        sentiment_data = []
        for result in sentiment_results:
            # Use temporal weighted sentiment for better intraday calculation
            weighted_sentiment = result.temporal_weight.weighted_sentiment
            temporal_confidence = result.temporal_weight.final_weight * result.calibrated_confidence.calibrated_confidence
            
            sentiment_data.append({
                'timestamp': result.article_date.isoformat(),
                'sentiment_score': weighted_sentiment,  # Use temporally weighted sentiment
                'confidence': temporal_confidence,  # Use temporally weighted confidence
                'stock_symbol': result.stock_symbol,
                'raw_sentiment': result.sentiment_score,  # Keep original for comparison
                'temporal_weight': result.temporal_weight.final_weight,
                'calibrated_confidence': result.calibrated_confidence.calibrated_confidence
            })
        
        # Enhanced aggregation with wider time windows for better intraday sentiment
        # Use multiple bucket types to capture different temporal patterns
        bucket_types = [
            'hourly',      # 1-hour buckets for immediate sentiment
            '4hour',       # 4-hour buckets for short-term patterns
            'daily',       # Daily buckets for overall sentiment
            'weekly',      # Weekly buckets for trend analysis
            'monthly'      # Monthly buckets for long-term patterns
        ]
        
        aggregated_buckets = self.intraday_aggregator.aggregate_multiple_buckets(
            sentiment_data, bucket_types
        )
        
        # Add temporal decay analysis to each bucket
        for bucket_type, buckets in aggregated_buckets.items():
            for bucket in buckets:
                # Calculate temporal decay metrics for this bucket
                bucket.temporal_decay_score = self._calculate_temporal_decay_score(bucket)
                bucket.sentiment_momentum = self._calculate_sentiment_momentum(bucket)
                bucket.confidence_trend = self._calculate_confidence_trend(bucket)
        
        return aggregated_buckets
    
    def _calculate_temporal_decay_score(self, bucket) -> float:
        """Calculate temporal decay score for a bucket based on article ages"""
        if not hasattr(bucket, 'articles') or not bucket.articles:
            return 1.0
        
        # Calculate average age of articles in this bucket
        current_time = datetime.now()
        ages = []
        
        for article in bucket.articles:
            try:
                article_time = datetime.fromisoformat(article['timestamp'].replace('Z', '+00:00'))
                age_hours = (current_time - article_time).total_seconds() / 3600
                ages.append(age_hours)
            except:
                continue
        
        if not ages:
            return 1.0
        
        avg_age_hours = sum(ages) / len(ages)
        
        # Apply exponential decay: newer articles have higher weight
        decay_factor = math.exp(-avg_age_hours / 24)  # 24-hour half-life
        return max(0.1, decay_factor)  # Minimum 0.1 weight
    
    def _calculate_sentiment_momentum(self, bucket) -> float:
        """Calculate sentiment momentum (trend direction) for a bucket"""
        if not hasattr(bucket, 'articles') or len(bucket.articles) < 2:
            return 0.0
        
        # Sort articles by timestamp
        sorted_articles = sorted(bucket.articles, key=lambda x: x['timestamp'])
        
        if len(sorted_articles) < 2:
            return 0.0
        
        # Calculate sentiment trend over time
        sentiments = [article['sentiment_score'] for article in sorted_articles]
        
        # Simple linear regression slope as momentum indicator
        n = len(sentiments)
        if n < 2:
            return 0.0
        
        x = list(range(n))
        y = sentiments
        
        # Calculate slope (momentum)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _calculate_confidence_trend(self, bucket) -> float:
        """Calculate confidence trend for a bucket"""
        if not hasattr(bucket, 'articles') or len(bucket.articles) < 2:
            return 0.0
        
        # Sort articles by timestamp
        sorted_articles = sorted(bucket.articles, key=lambda x: x['timestamp'])
        
        if len(sorted_articles) < 2:
            return 0.0
        
        # Calculate confidence trend over time
        confidences = [article['confidence'] for article in sorted_articles]
        
        # Simple linear regression slope as confidence trend
        n = len(confidences)
        if n < 2:
            return 0.0
        
        x = list(range(n))
        y = confidences
        
        # Calculate slope (confidence trend)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def generate_comprehensive_dataset(self, 
                                     start_date: date = None,
                                     end_date: date = None,
                                     stock_symbols: List[str] = None,
                                     output_dir: str = "comprehensive_datasets") -> Dict[str, Any]:
        """Generate comprehensive sentiment dataset with all advanced features"""
        
        if start_date is None:
            start_date = date.today() - timedelta(days=self.config['google_news']['date_range_days'])
        if end_date is None:
            end_date = date.today()
        if stock_symbols is None:
            stock_symbols = self.stock_symbols[:10]  # Limit for testing
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Starting comprehensive dataset generation for {len(stock_symbols)} stocks")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        all_results = {}
        progress_data = {
            'total_stocks': len(stock_symbols),
            'processed_stocks': 0,
            'total_articles': 0,
            'start_time': datetime.now()
        }
        
        # Process each stock
        for i, stock_symbol in enumerate(stock_symbols):
            try:
                self.logger.info(f"Processing {stock_symbol} ({i+1}/{len(stock_symbols)})")
                
                # Fetch news articles
                articles = self.fetch_news_for_stock(stock_symbol, start_date, end_date)
                
                if not articles:
                    self.logger.warning(f"No articles found for {stock_symbol}")
                    continue
                
                # Process articles with advanced NLP
                sentiment_results = []
                for article in articles:
                    try:
                        result = self.process_article_with_advanced_nlp(article)
                        sentiment_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing article for {stock_symbol}: {e}")
                        continue
                
                # Aggregate intraday data
                intraday_buckets = self.aggregate_intraday_data(sentiment_results, stock_symbol)
                
                # Store results
                all_results[stock_symbol] = {
                    'sentiment_results': sentiment_results,
                    'intraday_buckets': intraday_buckets,
                    'article_count': len(articles),
                    'processed_count': len(sentiment_results)
                }
                
                progress_data['processed_stocks'] += 1
                progress_data['total_articles'] += len(articles)
                
                # Save intermediate results
                self._save_stock_results(stock_symbol, all_results[stock_symbol], output_path)
                
                self.logger.info(f"Completed {stock_symbol}: {len(articles)} articles, {len(sentiment_results)} processed")
                
            except Exception as e:
                self.logger.error(f"Error processing {stock_symbol}: {e}")
                continue
        
        # Generate comprehensive reports
        comprehensive_report = self._generate_comprehensive_report(all_results, progress_data)
        
        # Save final dataset
        self._save_comprehensive_dataset(all_results, comprehensive_report, output_path)
        
        self.logger.info("Comprehensive dataset generation completed")
        return comprehensive_report
    
    def _save_stock_results(self, stock_symbol: str, results: Dict, output_path: Path):
        """Save results for a single stock"""
        
        stock_file = output_path / f"{stock_symbol}_advanced_sentiment.json"
        
        # Convert to serializable format
        serializable_results = {
            'stock_symbol': stock_symbol,
            'article_count': results['article_count'],
            'processed_count': results['processed_count'],
            'sentiment_data': []
        }
        
        for result in results['sentiment_results']:
            serializable_results['sentiment_data'].append({
                'article_date': result.article_date.isoformat(),
                'sentiment_score': result.sentiment_score,
                'confidence_score': result.confidence_score,
                'calibrated_confidence': result.calibrated_confidence.calibrated_confidence,
                'temporal_weight': result.temporal_weight.final_weight,
                'weighted_sentiment': result.temporal_weight.weighted_sentiment,
                'language_detected': result.language_detected,
                'entities_found': result.entities_found,
                'model_breakdown': result.model_breakdown
            })
        
        with open(stock_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _generate_comprehensive_report(self, all_results: Dict, progress_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive report with all advanced features"""
        
        total_articles = progress_data['total_articles']
        total_processed = sum(r['processed_count'] for r in all_results.values())
        
        # Calculate overall statistics
        all_sentiments = []
        all_confidences = []
        all_temporal_weights = []
        language_distribution = {}
        entity_frequency = {}
        
        for stock_data in all_results.values():
            for result in stock_data['sentiment_results']:
                all_sentiments.append(result.sentiment_score)
                all_confidences.append(result.calibrated_confidence.calibrated_confidence)
                all_temporal_weights.append(result.temporal_weight.final_weight)
                
                # Language distribution
                lang = result.language_detected
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                
                # Entity frequency
                for entity in result.entities_found:
                    entity_frequency[entity] = entity_frequency.get(entity, 0) + 1
        
        # Calculate intraday statistics
        intraday_stats = {}
        for stock_symbol, stock_data in all_results.items():
            for bucket_type, buckets in stock_data['intraday_buckets'].items():
                if bucket_type not in intraday_stats:
                    intraday_stats[bucket_type] = []
                intraday_stats[bucket_type].extend([
                    {
                        'stock_symbol': stock_symbol,
                        'bucket_count': len(buckets),
                        'avg_sentiment': np.mean([b.weighted_sentiment for b in buckets]) if buckets else 0
                    }
                ])
        
        report = {
            'generation_summary': {
                'total_stocks': len(all_results),
                'total_articles': total_articles,
                'total_processed': total_processed,
                'processing_rate': total_processed / total_articles if total_articles > 0 else 0,
                'start_time': progress_data['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': (datetime.now() - progress_data['start_time']).total_seconds() / 3600
            },
            'sentiment_statistics': {
                'mean_sentiment': np.mean(all_sentiments) if all_sentiments else 0,
                'median_sentiment': np.median(all_sentiments) if all_sentiments else 0,
                'sentiment_std': np.std(all_sentiments) if all_sentiments else 0,
                'sentiment_range': (min(all_sentiments), max(all_sentiments)) if all_sentiments else (0, 0)
            },
            'confidence_statistics': {
                'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
                'median_confidence': np.median(all_confidences) if all_confidences else 0,
                'confidence_std': np.std(all_confidences) if all_confidences else 0
            },
            'temporal_weighting_statistics': {
                'mean_temporal_weight': np.mean(all_temporal_weights) if all_temporal_weights else 0,
                'median_temporal_weight': np.median(all_temporal_weights) if all_temporal_weights else 0,
                'temporal_weight_std': np.std(all_temporal_weights) if all_temporal_weights else 0
            },
            'nlp_features': {
                'language_distribution': language_distribution,
                'top_entities': sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[:20],
                'total_entities_found': len(entity_frequency)
            },
            'intraday_aggregation': {
                'bucket_types_processed': list(intraday_stats.keys()),
                'bucket_statistics': intraday_stats
            },
            'advanced_features_used': {
                'confidence_calibration': self.config['advanced_features']['confidence_calibration'],
                'intraday_aggregation': self.config['advanced_features']['intraday_aggregation'],
                'windowed_boosting': self.config['advanced_features']['windowed_boosting'],
                'custom_decay_profiles': self.config['advanced_features']['custom_decay_profiles'],
                'multi_model_sentiment': self.config['advanced_features']['multi_model_sentiment'],
                'language_detection': self.config['advanced_features']['language_detection'],
                'ner_extraction': self.config['advanced_features']['ner_extraction']
            }
        }
        
        return report
    
    def _save_comprehensive_dataset(self, all_results: Dict, report: Dict, output_path: Path):
        """Save comprehensive dataset with all advanced features"""
        
        # Save comprehensive report
        report_file = output_path / "comprehensive_sentiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save aggregated dataset
        aggregated_data = []
        for stock_symbol, stock_data in all_results.items():
            for result in stock_data['sentiment_results']:
                aggregated_data.append({
                    'stock_symbol': stock_symbol,
                    'article_date': result.article_date.isoformat(),
                    'sentiment_score': result.sentiment_score,
                    'confidence_score': result.confidence_score,
                    'calibrated_confidence': result.calibrated_confidence.calibrated_confidence,
                    'temporal_weight': result.temporal_weight.final_weight,
                    'weighted_sentiment': result.temporal_weight.weighted_sentiment,
                    'language_detected': result.language_detected,
                    'entities_found': result.entities_found,
                    'model_breakdown': result.model_breakdown,
                    'nlp_features': result.nlp_features
                })
        
        # Save as CSV
        df = pd.DataFrame(aggregated_data)
        csv_file = output_path / "comprehensive_sentiment_dataset.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = output_path / "comprehensive_sentiment_dataset.json"
        with open(json_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        
        self.logger.info(f"Comprehensive dataset saved to {output_path}")
        self.logger.info(f"Files created: {list(output_path.glob('*'))}")

    def save_snapshot(self, file_list: List[str], tag: Optional[str] = None):
        return self.data_quality.save_snapshot(file_list, tag)

    def restore_snapshot(self, snapshot_name: str):
        return self.data_quality.restore_snapshot(snapshot_name)

    def rollup_aggregate(self, input_file: str, freq: str = "D", output_file: Optional[str] = None):
        return self.data_quality.rollup_aggregate(input_file, freq, output_file)

    def run_qc(self, input_file: str):
        return self.data_quality.generate_qc_report(input_file)

    def run_backfill(self, start_date, end_date, stocks=None):
        stocks = stocks or self.stock_symbols
        mgr = BackfillManager(self.fetch_news_for_stock, stocks)
        return mgr.backfill(start_date, end_date)

def cli():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="Comprehensive Sentiment Pipeline CLI")
    parser.add_argument("--snapshot", nargs='+', help="Files to snapshot (space separated)")
    parser.add_argument("--restore", type=str, help="Snapshot name to restore")
    parser.add_argument("--rollup", nargs=2, metavar=('INPUT_FILE', 'FREQ'), help="Roll-up aggregation (file, freq)")
    parser.add_argument("--qc", type=str, help="Run QC on file")
    parser.add_argument("--backfill", nargs=3, metavar=('START_DATE', 'END_DATE', 'STOCKS'), help="Backfill (YYYY-MM-DD YYYY-MM-DD STOCK1,STOCK2,...)")
    parser.add_argument("--schedule", action='store_true', help="Schedule daily QC/snapshot/rollup jobs")
    args = parser.parse_args()

    gen = ComprehensiveSentimentGenerator()

    if args.snapshot:
        print("Saving snapshot...")
        snap = gen.save_snapshot(args.snapshot)
        print(f"Snapshot saved: {snap}")
    if args.restore:
        print(f"Restoring snapshot {args.restore}...")
        gen.restore_snapshot(args.restore)
        print("Snapshot restored.")
    if args.rollup:
        print(f"Rolling up {args.rollup[0]} to {args.rollup[1]}...")
        out = gen.rollup_aggregate(args.rollup[0], args.rollup[1])
        print(f"Roll-up saved: {out}")
    if args.qc:
        print(f"Running QC on {args.qc}...")
        report = gen.run_qc(args.qc)
        print(f"QC report: {report}")
    if args.backfill:
        start = datetime.fromisoformat(args.backfill[0]).date()
        end = datetime.fromisoformat(args.backfill[1]).date()
        stocks = args.backfill[2].split(",")
        print(f"Backfilling {stocks} from {start} to {end}...")
        missing = gen.run_backfill(start, end, stocks)
        print(f"Backfill complete. Missing: {len(missing)} entries.")
    if args.schedule:
        import schedule, time as t
        print("Scheduling daily QC/snapshot/rollup jobs...")
        def daily_jobs():
            # Example: snapshot, rollup, QC on main dataset
            files = ["comprehensive_sentiment_dataset.csv"]
            gen.save_snapshot(files, tag="daily")
            gen.rollup_aggregate(files[0], "D")
            gen.run_qc(files[0])
        schedule.every().day.at("23:00").do(daily_jobs)
        while True:
            schedule.run_pending()
            t.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli()
    else:
        main() 