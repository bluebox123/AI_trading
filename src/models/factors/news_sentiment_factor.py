#!/usr/bin/env python3
"""
News Sentiment Factor - Enhanced News Sentiment Analysis (25% weight)
====================================================================
Uses the existing comprehensive sentiment analysis system.
"""

import sys
from pathlib import Path
from typing import Dict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput
from src.models.enhanced_signal_generator_v2 import get_signal_generator_v2

logger = logging.getLogger(__name__)

class NewsSentimentFactor(BaseFactor):
    """Enhanced news sentiment analysis factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        self.signal_generator = None
        self._initialize_sentiment_service()
        
    def _initialize_sentiment_service(self):
        """Initialize the sentiment service"""
        try:
            self.signal_generator = get_signal_generator_v2()
            self.logger.info("Sentiment service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment service: {e}")
            self.signal_generator = None
            
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate news sentiment factor"""
        try:
            if not self.signal_generator:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'Sentiment service not available'},
                    sub_factors={}
                )
                
            # Get comprehensive sentiment data
            sentiment_data = self.signal_generator.get_comprehensive_sentiment(symbol)
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = sentiment_data['sentiment_score']
            confidence = sentiment_data['confidence']
            news_count = sentiment_data['news_count']
            
            # Adjust score based on news volume
            if news_count > 20:
                # High news volume - increase impact
                score_multiplier = 1.2
            elif news_count > 10:
                # Normal news volume
                score_multiplier = 1.0
            elif news_count > 3:
                # Low news volume - reduce impact
                score_multiplier = 0.8
            else:
                # Very low news volume - significantly reduce impact
                score_multiplier = 0.5
                confidence *= 0.7
                
            adjusted_score = sentiment_score * score_multiplier
            
            # Normalize to -1 to 1 range
            normalized_score = max(-1, min(1, adjusted_score))
            
            return FactorOutput(
                score=normalized_score,
                confidence=confidence,
                details={
                    'sentiment_label': sentiment_data['sentiment_label'],
                    'news_count': news_count,
                    'source': sentiment_data['source'],
                    'sample_headlines': sentiment_data.get('sample_headlines', [])[:3],
                    'sector': sentiment_data.get('sector', 'Other')
                },
                sub_factors={
                    'raw_sentiment': sentiment_score,
                    'news_volume_factor': score_multiplier,
                    'sentiment_confidence': sentiment_data['confidence']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating news sentiment factor for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            ) 