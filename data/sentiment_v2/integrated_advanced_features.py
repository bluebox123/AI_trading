#!/usr/bin/env python3
"""
Integrated Advanced Features for Sentiment Analysis
=================================================

This module integrates all advanced features:
- Confidence Calibration
- Intraday Aggregation  
- Windowed Boosting
- Custom Decay Profiles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Import advanced modules
from advanced_confidence_calibration import ConfidenceCalibrator, CalibratedConfidence
from intraday_aggregation import IntradayAggregator, IntradayBucket
from advanced_temporal_weighting import AdvancedTemporalWeighter, WeightedSentiment

@dataclass
class IntegratedSentimentResult:
    """Integrated sentiment result with all advanced features"""
    stock_symbol: str
    article_date: datetime
    sentiment_score: float
    raw_confidence: float
    calibrated_confidence: CalibratedConfidence
    temporal_weight: WeightedSentiment
    intraday_bucket: Optional[IntradayBucket]
    final_weighted_sentiment: float
    confidence_level: str
    reliability_score: float

class IntegratedAdvancedFeatures:
    """Integrated system for all advanced sentiment features"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the integrated advanced features system"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize all advanced components
        self.confidence_calibrator = ConfidenceCalibrator()
        self.intraday_aggregator = IntradayAggregator()
        self.temporal_weighter = AdvancedTemporalWeighter()
        
        self.logger.info("Integrated Advanced Features initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            "confidence_calibration": {
                "enabled": True,
                "methods": ["temperature_scaling", "platt_scaling", "ensemble"]
            },
            "intraday_aggregation": {
                "enabled": True,
                "bucket_types": ["15min", "1hour", "2hour", "daily"],
                "trading_hours_only": True
            },
            "windowed_boosting": {
                "enabled": True,
                "default_multiplier": 1.2,
                "max_multiplier": 2.0,
                "boost_hours": 1
            },
            "custom_decay_profiles": {
                "enabled": True,
                "sector_specific": True,
                "stock_specific": True
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
    
    def process_sentiment_with_advanced_features(self,
                                               stock_symbol: str,
                                               sentiment_score: float,
                                               confidence: float,
                                               article_date: datetime,
                                               target_date: datetime = None,
                                               sector: str = None) -> IntegratedSentimentResult:
        """Process sentiment with all advanced features integrated"""
        
        if target_date is None:
            target_date = datetime.now()
        
        # 1. Confidence Calibration
        calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
            sentiment_score=sentiment_score,
            raw_confidence=confidence,
            model_name="integrated_ensemble"
        )
        
        # 2. Temporal Weighting with Windowed Boosting
        temporal_weight = self.temporal_weighter.calculate_temporal_weight(
            article_date=article_date,
            target_date=target_date,
            stock_symbol=stock_symbol,
            sector=sector,
            confidence=calibrated_confidence.calibrated_confidence
        )
        
        # 3. Calculate final weighted sentiment
        final_weighted_sentiment = sentiment_score * temporal_weight.final_weight
        
        # 4. Get confidence level
        confidence_level = self.confidence_calibrator.get_confidence_level(
            calibrated_confidence.calibrated_confidence
        )
        
        return IntegratedSentimentResult(
            stock_symbol=stock_symbol,
            article_date=article_date,
            sentiment_score=sentiment_score,
            raw_confidence=confidence,
            calibrated_confidence=calibrated_confidence,
            temporal_weight=temporal_weight,
            intraday_bucket=None,  # Will be set during aggregation
            final_weighted_sentiment=final_weighted_sentiment,
            confidence_level=confidence_level,
            reliability_score=calibrated_confidence.reliability_score
        )
    
    def aggregate_intraday_data(self, 
                               sentiment_results: List[IntegratedSentimentResult],
                               stock_symbol: str) -> Dict[str, List[IntradayBucket]]:
        """Aggregate sentiment data into intraday buckets"""
        
        if not self.config['intraday_aggregation']['enabled']:
            return {}
        
        # Convert to format expected by aggregator
        sentiment_data = []
        for result in sentiment_results:
            sentiment_data.append({
                'timestamp': result.article_date.isoformat(),
                'sentiment_score': result.sentiment_score,
                'confidence': result.calibrated_confidence.calibrated_confidence,
                'stock_symbol': result.stock_symbol
            })
        
        # Aggregate into different bucket types
        bucket_types = self.config['intraday_aggregation']['bucket_types']
        aggregated_buckets = self.intraday_aggregator.aggregate_multiple_buckets(
            sentiment_data, bucket_types
        )
        
        return aggregated_buckets
    
    def generate_comprehensive_report(self, 
                                   sentiment_results: List[IntegratedSentimentResult],
                                   stock_symbol: str) -> Dict[str, Any]:
        """Generate comprehensive report with all advanced features"""
        
        if not sentiment_results:
            return {"error": "No sentiment results to analyze"}
        
        # Calculate statistics
        sentiments = [r.sentiment_score for r in sentiment_results]
        confidences = [r.calibrated_confidence.calibrated_confidence for r in sentiment_results]
        temporal_weights = [r.temporal_weight.final_weight for r in sentiment_results]
        weighted_sentiments = [r.final_weighted_sentiment for r in sentiment_results]
        
        # Confidence level distribution
        confidence_levels = [r.confidence_level for r in sentiment_results]
        level_distribution = {}
        for level in confidence_levels:
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Reliability statistics
        reliability_scores = [r.reliability_score for r in sentiment_results]
        
        report = {
            'stock_symbol': stock_symbol,
            'total_articles': len(sentiment_results),
            'date_range': {
                'start': min(r.article_date for r in sentiment_results).isoformat(),
                'end': max(r.article_date for r in sentiment_results).isoformat()
            },
            'sentiment_statistics': {
                'mean_sentiment': np.mean(sentiments),
                'median_sentiment': np.median(sentiments),
                'sentiment_std': np.std(sentiments),
                'sentiment_range': (min(sentiments), max(sentiments))
            },
            'confidence_statistics': {
                'mean_confidence': np.mean(confidences),
                'median_confidence': np.median(confidences),
                'confidence_std': np.std(confidences),
                'confidence_level_distribution': level_distribution
            },
            'temporal_weighting_statistics': {
                'mean_temporal_weight': np.mean(temporal_weights),
                'median_temporal_weight': np.median(temporal_weights),
                'temporal_weight_std': np.std(temporal_weights)
            },
            'weighted_sentiment_statistics': {
                'mean_weighted_sentiment': np.mean(weighted_sentiments),
                'median_weighted_sentiment': np.median(weighted_sentiments),
                'weighted_sentiment_std': np.std(weighted_sentiments)
            },
            'reliability_statistics': {
                'mean_reliability': np.mean(reliability_scores),
                'median_reliability': np.median(reliability_scores),
                'reliability_std': np.std(reliability_scores)
            },
            'advanced_features_summary': {
                'confidence_calibration_enabled': self.config['confidence_calibration']['enabled'],
                'intraday_aggregation_enabled': self.config['intraday_aggregation']['enabled'],
                'windowed_boosting_enabled': self.config['windowed_boosting']['enabled'],
                'custom_decay_profiles_enabled': self.config['custom_decay_profiles']['enabled']
            }
        }
        
        return report
    
    def compare_decay_profiles(self, stock_symbol: str = None, sector: str = None) -> Dict[str, List[float]]:
        """Compare different decay profiles for visualization"""
        return self.temporal_weighter.compare_decay_curves()
    
    def get_decay_statistics(self, profile_name: str = "default") -> Dict[str, float]:
        """Get statistics for a decay profile"""
        return self.temporal_weighter.get_decay_statistics(profile_name)
    
    def update_calibration_with_feedback(self,
                                       sentiment_score: float,
                                       raw_confidence: float,
                                       actual_sentiment: str,
                                       model_name: str = "integrated_ensemble"):
        """Update calibration with actual sentiment feedback"""
        self.confidence_calibrator.update_calibration_with_feedback(
            sentiment_score=sentiment_score,
            raw_confidence=raw_confidence,
            actual_sentiment=actual_sentiment,
            model_name=model_name
        )
    
    def create_custom_decay_profile(self, name: str, **kwargs):
        """Create a custom decay profile"""
        return self.temporal_weighter.create_custom_decay_profile(name, **kwargs)

def main():
    """Test the integrated advanced features system"""
    integrated_system = IntegratedAdvancedFeatures()
    
    print("Testing Integrated Advanced Features System")
    print("=" * 50)
    
    # Test with sample data
    stock_symbol = "RELIANCE.NSE"
    article_date = datetime.now() - timedelta(hours=2)
    
    test_cases = [
        (0.8, 0.9, "high confidence positive"),
        (-0.5, 0.6, "medium confidence negative"),
        (0.1, 0.3, "low confidence neutral")
    ]
    
    results = []
    
    for sentiment_score, confidence, description in test_cases:
        result = integrated_system.process_sentiment_with_advanced_features(
            stock_symbol=stock_symbol,
            sentiment_score=sentiment_score,
            confidence=confidence,
            article_date=article_date
        )
        
        results.append(result)
        
        print(f"\n{description}:")
        print(f"  Original sentiment: {result.sentiment_score:.3f}")
        print(f"  Raw confidence: {result.raw_confidence:.3f}")
        print(f"  Calibrated confidence: {result.calibrated_confidence.calibrated_confidence:.3f}")
        print(f"  Temporal weight: {result.temporal_weight.final_weight:.3f}")
        print(f"  Final weighted sentiment: {result.final_weighted_sentiment:.3f}")
        print(f"  Confidence level: {result.confidence_level}")
        print(f"  Reliability score: {result.reliability_score:.3f}")
    
    # Generate comprehensive report
    report = integrated_system.generate_comprehensive_report(results, stock_symbol)
    
    print(f"\nComprehensive Report Summary:")
    print(f"  Total articles: {report['total_articles']}")
    print(f"  Mean sentiment: {report['sentiment_statistics']['mean_sentiment']:.3f}")
    print(f"  Mean confidence: {report['confidence_statistics']['mean_confidence']:.3f}")
    print(f"  Mean temporal weight: {report['temporal_weighting_statistics']['mean_temporal_weight']:.3f}")
    print(f"  Mean reliability: {report['reliability_statistics']['mean_reliability']:.3f}")
    
    # Test decay profile comparison
    print(f"\nDecay Profile Comparison:")
    curves = integrated_system.compare_decay_profiles()
    for name, weights in curves.items():
        print(f"  {name}: 1d={weights[1]:.3f}, 7d={weights[7]:.3f}")

if __name__ == "__main__":
    main() 