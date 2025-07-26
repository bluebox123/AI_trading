#!/usr/bin/env python3
"""
Modular Signal Generator
=======================
A comprehensive signal generation system with 8 weighted factors
that can be individually enabled/disabled.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

@dataclass
class FactorOutput:
    """Output from a factor calculation"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    details: Dict[str, Any] = field(default_factory=dict)
    sub_factors: Dict[str, float] = field(default_factory=dict)
    
class BaseFactor(ABC):
    """Abstract base class for all factors"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """
        Calculate factor for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NSE')
            market_data: Dictionary containing all relevant market data
            
        Returns:
            FactorOutput with score, confidence, and details
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if factor is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable this factor"""
        self.enabled = True
        self.logger.info(f"Factor {self.name} enabled")
        
    def disable(self):
        """Disable this factor"""
        self.enabled = False
        self.logger.info(f"Factor {self.name} disabled")
        
    def set_weight(self, weight: float):
        """Update factor weight"""
        self.weight = weight
        self.logger.info(f"Factor {self.name} weight updated to {weight}")

class ModularSignalGenerator:
    """
    Main signal generator that combines all factors with proper weighting
    """
    
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}
        self.eodhd_client = None
        self._initialize_factors()
        self._initialize_eodhd()
        
    def _initialize_factors(self):
        """Initialize all factor modules"""
        # Import factor modules
        from .factors.ai_model_factor import AIModelFactor
        from .factors.news_sentiment_factor import NewsSentimentFactor
        from .factors.technical_indicators_factor import TechnicalIndicatorsFactor
        from .factors.volatility_management_factor import VolatilityManagementFactor
        from .factors.order_flow_factor import OrderFlowFactor
        from .factors.macro_economic_factor import MacroEconomicFactor
        from .factors.market_regime_factor import MarketRegimeFactor
        from .factors.risk_management_factor import RiskManagementFactor
        
        # Initialize each factor with its weight
        self.factors = {
            'ai_model': AIModelFactor('AI Model Output', 0.50),
            'news_sentiment': NewsSentimentFactor('Enhanced News Sentiment', 0.25),
            'technical': TechnicalIndicatorsFactor('Advanced Technical Indicators', 0.22),
            'volatility': VolatilityManagementFactor('Dynamic Volatility Management', 0.20),
            'order_flow': OrderFlowFactor('Smart Order Flow Analysis', 0.12),
            'macro': MacroEconomicFactor('Macro Economic Intelligence', 0.08),
            'regime': MarketRegimeFactor('Market Regime Detector', 0.06),
            'risk': RiskManagementFactor('Risk Management Override', 0.04)
        }
        
        logger.info(f"Initialized {len(self.factors)} factors")
        
    def _initialize_eodhd(self):
        """Initialize EODHD client for real-time data"""
        try:
            from src.data.eodhd_v4_bridge import EODHDV4Bridge
            self.eodhd_client = EODHDV4Bridge()
            logger.info("EODHD client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EODHD client: {e}")
            self.eodhd_client = None
            
    def get_market_data(self, symbol: str) -> Dict:
        """
        Fetch comprehensive market data for a symbol
        
        Returns dictionary with:
        - current_price
        - historical_prices (DataFrame)
        - volume_data
        - technical_indicators
        - news_data
        - market_indices
        - etc.
        """
        market_data = {
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        try:
            if self.eodhd_client:
                # Get real-time quote
                quote = self.eodhd_client.get_real_time_quote(symbol)
                if quote:
                    market_data['current_price'] = quote.get('close', 0)
                    market_data['volume'] = quote.get('volume', 0)
                    market_data['change_percent'] = quote.get('change_p', 0)
                    
                # Get historical data for technical analysis
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                historical = self.eodhd_client.get_stock_data(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                if historical:
                    market_data['historical_prices'] = pd.DataFrame(historical)
                    
                # Get intraday data
                intraday = self.eodhd_client.get_intraday_data(symbol)
                if intraday:
                    market_data['intraday_data'] = pd.DataFrame(intraday)
                    
                # Get market indices (NIFTY, SENSEX)
                market_data['market_indices'] = {
                    'NIFTY': self.eodhd_client.get_real_time_quote('NIFTY50.INDX'),
                    'SENSEX': self.eodhd_client.get_real_time_quote('SENSEX.INDX')
                }
                
            else:
                # Fallback mock data
                logger.warning(f"Using mock data for {symbol}")
                market_data.update(self._get_mock_market_data(symbol))
                
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            market_data.update(self._get_mock_market_data(symbol))
            
        return market_data
    
    def _get_mock_market_data(self, symbol: str) -> Dict:
        """Generate mock market data for testing"""
        base_price = 1000 + (hash(symbol) % 3000)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate mock price data
        prices = []
        current = base_price
        for date in dates:
            change = np.random.normal(0, 0.02) * current
            current += change
            prices.append({
                'date': date,
                'open': current * 0.99,
                'high': current * 1.01,
                'low': current * 0.98,
                'close': current,
                'volume': np.random.randint(1000000, 5000000)
            })
            
        return {
            'current_price': current,
            'volume': prices[-1]['volume'],
            'change_percent': ((current - base_price) / base_price) * 100,
            'historical_prices': pd.DataFrame(prices),
            'intraday_data': pd.DataFrame(prices[-20:]),  # Last 20 days as intraday
            'market_indices': {
                'NIFTY': {'close': 22000, 'change_p': 0.5},
                'SENSEX': {'close': 72000, 'change_p': 0.6}
            }
        }
    
    def generate_signal(self, symbol: str, enabled_factors: Optional[List[str]] = None) -> Dict:
        """
        Generate trading signal for a symbol using enabled factors
        
        Args:
            symbol: Stock symbol
            enabled_factors: List of factor names to use (None = use all enabled)
            
        Returns:
            Dictionary with signal details
        """
        try:
            start_time = datetime.now()
            
            # Get market data
            market_data = self.get_market_data(symbol)
            
            # Calculate each factor
            factor_outputs = {}
            total_weight = 0
            
            for factor_name, factor in self.factors.items():
                # Check if factor should be used
                if enabled_factors is not None:
                    if factor_name not in enabled_factors:
                        continue
                elif not factor.is_enabled():
                    continue
                    
                try:
                    # Calculate factor
                    output = factor.calculate(symbol, market_data)
                    factor_outputs[factor_name] = output
                    total_weight += factor.weight
                    
                    logger.info(
                        f"[{factor_name}] {symbol}: Score={output.score:.3f}, "
                        f"Confidence={output.confidence:.3f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error calculating {factor_name} for {symbol}: {e}")
                    # Use neutral output on error
                    factor_outputs[factor_name] = FactorOutput(
                        score=0.0,
                        confidence=0.3,
                        details={'error': str(e)}
                    )
                    
            # Normalize weights to sum to 1.0
            if total_weight > 0:
                weight_scale = 1.0 / total_weight
            else:
                weight_scale = 0.0
                
            # Calculate weighted final score
            final_score = 0.0
            final_confidence = 0.0
            factor_contributions = {}
            
            for factor_name, output in factor_outputs.items():
                factor = self.factors[factor_name]
                normalized_weight = factor.weight * weight_scale
                
                contribution = output.score * normalized_weight
                final_score += contribution
                final_confidence += output.confidence * normalized_weight
                
                factor_contributions[factor_name] = {
                    'score': output.score,
                    'confidence': output.confidence,
                    'weight': normalized_weight,
                    'contribution': contribution,
                    'details': output.details,
                    'sub_factors': output.sub_factors
                }
                
            # Determine signal based on final score
            if final_score > 0.15:
                signal = 'BUY'
            elif final_score < -0.15:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                
            # Apply risk management override if enabled
            if 'risk' in factor_outputs:
                risk_output = factor_outputs['risk']
                if risk_output.details.get('override_signal'):
                    signal = risk_output.details['override_signal']
                    logger.warning(f"Risk override applied for {symbol}: {signal}")
                    
            # Calculate price targets
            current_price = market_data.get('current_price', 0)
            if signal == 'BUY':
                target_price = current_price * (1 + 0.03 + abs(final_score) * 0.02)
                stop_loss = current_price * (1 - 0.02)
            elif signal == 'SELL':
                target_price = current_price * (1 - 0.03 - abs(final_score) * 0.02)
                stop_loss = current_price * (1 + 0.02)
            else:
                target_price = current_price * 1.01
                stop_loss = current_price * 0.99
                
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'symbol': symbol,
                'signal': signal,
                'final_score': round(final_score, 4),
                'confidence': round(final_confidence, 3),
                'current_price': round(current_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(processing_time, 3),
                'factors_used': list(factor_outputs.keys()),
                'factor_contributions': factor_contributions,
                'market_data_summary': {
                    'volume': market_data.get('volume', 0),
                    'change_percent': market_data.get('change_percent', 0),
                    'market_sentiment': 'BULLISH' if final_score > 0 else 'BEARISH'
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def generate_bulk_signals(self, symbols: List[str], max_workers: int = 8,
                            enabled_factors: Optional[List[str]] = None) -> List[Dict]:
        """Generate signals for multiple symbols in parallel"""
        signals = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.generate_signal, symbol, enabled_factors): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result()
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
                    signals.append({
                        'symbol': symbol,
                        'signal': 'ERROR',
                        'error': str(e)
                    })
                    
        return signals
    
    def enable_factor(self, factor_name: str):
        """Enable a specific factor"""
        if factor_name in self.factors:
            self.factors[factor_name].enable()
        else:
            raise ValueError(f"Unknown factor: {factor_name}")
            
    def disable_factor(self, factor_name: str):
        """Disable a specific factor"""
        if factor_name in self.factors:
            self.factors[factor_name].disable()
        else:
            raise ValueError(f"Unknown factor: {factor_name}")
            
    def set_factor_weight(self, factor_name: str, weight: float):
        """Update weight for a specific factor"""
        if factor_name in self.factors:
            self.factors[factor_name].set_weight(weight)
        else:
            raise ValueError(f"Unknown factor: {factor_name}")
            
    def get_factor_status(self) -> Dict[str, Dict]:
        """Get status of all factors"""
        return {
            name: {
                'enabled': factor.is_enabled(),
                'weight': factor.weight,
                'class': factor.__class__.__name__
            }
            for name, factor in self.factors.items()
        }

# Singleton instance
_generator_instance = None

def get_modular_signal_generator() -> ModularSignalGenerator:
    """Get singleton instance of modular signal generator"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ModularSignalGenerator()
    return _generator_instance 