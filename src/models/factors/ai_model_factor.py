#!/usr/bin/env python3
"""
AI Model Factor - Core V5 Neural Network Prediction (50% weight)
================================================================
Uses the existing enhanced V5 model as the core prediction component.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput
from src.models.enhanced_signal_generator_v2 import get_signal_generator_v2

logger = logging.getLogger(__name__)

class AIModelFactor(BaseFactor):
    """Core V5 Neural Network prediction factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        self.signal_generator = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the V5 model"""
        try:
            self.signal_generator = get_signal_generator_v2()
            self.logger.info("V5 model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize V5 model: {e}")
            self.signal_generator = None
            
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate AI model prediction"""
        try:
            if not self.signal_generator:
                # Fallback if model not available
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'V5 model not available'},
                    sub_factors={}
                )
                
            # Get sentiment data (will be replaced by news sentiment factor)
            sentiment_data = self.signal_generator.get_comprehensive_sentiment(symbol)
            csv_data = self.signal_generator.csv_sentiment_data.get(symbol, {})
            
            # Prepare features for V5 model
            features = {
                'sentiment_score': sentiment_data['sentiment_score'],
                'confidence': sentiment_data['confidence'],
                'news_count': sentiment_data['news_count'],
                'momentum': csv_data.get('momentum', 0),
                'sector_score': 0.0,
                'sector': sentiment_data.get('sector', 'Other')
            }
            
            # Generate V5 prediction
            v5_score, v5_confidence = self.signal_generator.generate_v5_prediction(symbol, features)
            
            # Normalize score to -1 to 1 range
            normalized_score = max(-1, min(1, v5_score))
            
            return FactorOutput(
                score=normalized_score,
                confidence=v5_confidence,
                details={
                    'raw_v5_score': v5_score,
                    'model_version': 'enhanced_v5_20250703',
                    'features_used': features
                },
                sub_factors={
                    'neural_network_output': v5_score,
                    'feature_confidence': features['confidence']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating AI model factor for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            ) 