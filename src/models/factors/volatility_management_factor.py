#!/usr/bin/env python3
"""
Volatility Management Factor - Dynamic Risk Control (20% weight)
===============================================================
Implements VIX Fear Index, ATR Position Sizing, Volatility Regime Detection,
Correlation Analysis, and Drawdown Protection.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
import talib
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class VolatilityManagementFactor(BaseFactor):
    """Dynamic volatility management factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights
        self.sub_weights = {
            'vix_fear_index': 0.30,        # 6% of 20%
            'atr_position_sizing': 0.25,    # 5% of 20%
            'volatility_regime': 0.20,      # 4% of 20%
            'correlation_analysis': 0.15,   # 3% of 20%
            'drawdown_protection': 0.10     # 2% of 20%
        }
        
        # VIX thresholds
        self.vix_levels = {
            'low': 12,
            'normal': 20,
            'high': 30,
            'extreme': 40
        }
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate volatility management factor"""
        try:
            # Get required data
            historical_df = market_data.get('historical_prices')
            if historical_df is None or historical_df.empty:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'No historical data available'},
                    sub_factors={}
                )
                
            close_prices = historical_df['close'].values
            high_prices = historical_df['high'].values if 'high' in historical_df.columns else close_prices
            low_prices = historical_df['low'].values if 'low' in historical_df.columns else close_prices
            
            # Get market indices for VIX proxy
            market_indices = market_data.get('market_indices', {})
            
            # Calculate sub-factors
            vix_score = self._calculate_vix_fear_index(market_indices, close_prices)
            atr_score = self._calculate_atr_position_sizing(high_prices, low_prices, close_prices)
            regime_score = self._calculate_volatility_regime(close_prices)
            correlation_score = self._calculate_correlation_analysis(close_prices, market_indices)
            drawdown_score = self._calculate_drawdown_protection(close_prices)
            
            # Calculate weighted score
            sub_factors = {
                'vix_fear_index': vix_score,
                'atr_position_sizing': atr_score,
                'volatility_regime': regime_score,
                'correlation_analysis': correlation_score,
                'drawdown_protection': drawdown_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Calculate confidence based on volatility conditions
            current_volatility = self._calculate_current_volatility(close_prices)
            confidence = max(0.3, min(0.9, 1.0 - current_volatility))
            
            # Determine volatility state
            volatility_state = self._get_volatility_state(current_volatility, vix_score)
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'current_volatility': round(current_volatility, 4),
                    'volatility_state': volatility_state,
                    'risk_adjustment': self._get_risk_adjustment(volatility_state)
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility management for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _calculate_vix_fear_index(self, market_indices: Dict, close_prices: np.ndarray) -> float:
        """Calculate VIX fear index proxy score"""
        try:
            # Calculate realized volatility as VIX proxy
            returns = np.diff(close_prices) / close_prices[:-1]
            realized_vol = np.std(returns[-20:]) * np.sqrt(252) * 100  # Annualized
            
            # Get market fear level from indices
            nifty_data = market_indices.get('NIFTY', {})
            market_change = nifty_data.get('change_p', 0)
            
            # Combine volatility and market change
            if realized_vol < self.vix_levels['low']:
                vol_score = 0.3  # Low volatility - slightly bullish
            elif realized_vol < self.vix_levels['normal']:
                vol_score = 0.0  # Normal volatility - neutral
            elif realized_vol < self.vix_levels['high']:
                vol_score = -0.3  # High volatility - slightly bearish
            elif realized_vol < self.vix_levels['extreme']:
                vol_score = -0.6  # Very high volatility - bearish
            else:
                vol_score = -0.8  # Extreme volatility - very bearish
                
            # Adjust for market direction during high volatility
            if realized_vol > self.vix_levels['normal']:
                if market_change < -2:  # Market falling with high vol
                    vol_score -= 0.2
                elif market_change > 2:  # Market rising with high vol
                    vol_score += 0.1
                    
            return max(-1, min(1, vol_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating VIX fear index: {e}")
            return 0.0
            
    def _calculate_atr_position_sizing(self, high_prices: np.ndarray, 
                                     low_prices: np.ndarray, 
                                     close_prices: np.ndarray) -> float:
        """Calculate ATR-based position sizing score"""
        try:
            if len(close_prices) < 14:
                return 0.0
                
            # Calculate ATR
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            current_atr = atr[-1]
            avg_atr = np.mean(atr[-20:])
            
            if np.isnan(current_atr) or np.isnan(avg_atr) or avg_atr == 0:
                return 0.0
                
            # ATR relative to price
            atr_percentage = (current_atr / close_prices[-1]) * 100
            
            # Position sizing score based on ATR
            if atr_percentage < 1:  # Very low volatility
                size_score = 0.5  # Can take larger positions
            elif atr_percentage < 2:  # Low volatility
                size_score = 0.3
            elif atr_percentage < 3:  # Normal volatility
                size_score = 0.0
            elif atr_percentage < 5:  # High volatility
                size_score = -0.3  # Reduce position size
            else:  # Very high volatility
                size_score = -0.6  # Significantly reduce position size
                
            # Trend adjustment
            atr_trend = (current_atr - avg_atr) / avg_atr
            if atr_trend > 0.2:  # ATR expanding
                size_score -= 0.2
            elif atr_trend < -0.2:  # ATR contracting
                size_score += 0.1
                
            return max(-1, min(1, size_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR position sizing: {e}")
            return 0.0
            
    def _calculate_volatility_regime(self, close_prices: np.ndarray) -> float:
        """Detect and score volatility regime"""
        try:
            if len(close_prices) < 60:
                return 0.0
                
            # Calculate volatility over different periods
            returns = np.diff(close_prices) / close_prices[:-1]
            
            vol_5d = np.std(returns[-5:]) * np.sqrt(252)
            vol_20d = np.std(returns[-20:]) * np.sqrt(252)
            vol_60d = np.std(returns[-60:]) * np.sqrt(252)
            
            # Detect regime
            if vol_5d < vol_20d < vol_60d:
                # Volatility decreasing - bullish
                regime_score = 0.5
                regime = "decreasing_volatility"
            elif vol_5d > vol_20d > vol_60d:
                # Volatility increasing - bearish
                regime_score = -0.5
                regime = "increasing_volatility"
            elif vol_20d < 0.15:  # Low volatility regime
                regime_score = 0.3
                regime = "low_volatility"
            elif vol_20d > 0.30:  # High volatility regime
                regime_score = -0.3
                regime = "high_volatility"
            else:
                regime_score = 0.0
                regime = "normal_volatility"
                
            # Adjust for extreme changes
            vol_change = (vol_5d - vol_20d) / vol_20d
            if abs(vol_change) > 0.5:  # Rapid volatility change
                regime_score *= 1.5
                
            return max(-1, min(1, regime_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility regime: {e}")
            return 0.0
            
    def _calculate_correlation_analysis(self, close_prices: np.ndarray, 
                                      market_indices: Dict) -> float:
        """Analyze correlation with market indices"""
        try:
            if len(close_prices) < 20:
                return 0.0
                
            # Calculate stock returns
            stock_returns = np.diff(close_prices[-20:]) / close_prices[-21:-1]
            
            # Get market returns (using NIFTY as proxy)
            nifty_change = market_indices.get('NIFTY', {}).get('change_p', 0) / 100
            
            # Simple correlation approximation
            # In practice, would calculate rolling correlation with actual index data
            avg_stock_return = np.mean(stock_returns)
            
            # Score based on correlation behavior
            if avg_stock_return > 0 and nifty_change > 0:
                # Moving with market in uptrend - good
                corr_score = 0.3
            elif avg_stock_return < 0 and nifty_change < 0:
                # Moving with market in downtrend - neutral
                corr_score = 0.0
            elif avg_stock_return > 0 and nifty_change < 0:
                # Outperforming market - very good
                corr_score = 0.6
            elif avg_stock_return < 0 and nifty_change > 0:
                # Underperforming market - bad
                corr_score = -0.6
            else:
                corr_score = 0.0
                
            # Adjust for volatility
            stock_vol = np.std(stock_returns)
            if stock_vol > 0.03:  # High individual volatility
                corr_score *= 0.7
                
            return max(-1, min(1, corr_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation analysis: {e}")
            return 0.0
            
    def _calculate_drawdown_protection(self, close_prices: np.ndarray) -> float:
        """Calculate drawdown protection score"""
        try:
            if len(close_prices) < 20:
                return 0.0
                
            # Calculate running maximum
            running_max = np.maximum.accumulate(close_prices)
            
            # Calculate drawdown
            drawdown = (close_prices - running_max) / running_max
            current_drawdown = drawdown[-1]
            max_drawdown = np.min(drawdown[-20:])
            
            # Score based on drawdown levels
            if current_drawdown > -0.02:  # Less than 2% drawdown
                dd_score = 0.3
            elif current_drawdown > -0.05:  # 2-5% drawdown
                dd_score = 0.0
            elif current_drawdown > -0.10:  # 5-10% drawdown
                dd_score = -0.3
            elif current_drawdown > -0.15:  # 10-15% drawdown
                dd_score = -0.6
            else:  # More than 15% drawdown
                dd_score = -0.8
                
            # Check for recovery
            if len(close_prices) > 5:
                recent_trend = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
                if current_drawdown < -0.05 and recent_trend > 0.02:
                    # Recovering from drawdown
                    dd_score += 0.3
                    
            return max(-1, min(1, dd_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown protection: {e}")
            return 0.0
            
    def _calculate_current_volatility(self, close_prices: np.ndarray) -> float:
        """Calculate current volatility level"""
        try:
            if len(close_prices) < 20:
                return 0.5
                
            returns = np.diff(close_prices[-20:]) / close_prices[-21:-1]
            return np.std(returns) * np.sqrt(252)
            
        except Exception:
            return 0.5
            
    def _get_volatility_state(self, current_vol: float, vix_score: float) -> str:
        """Determine overall volatility state"""
        if current_vol < 0.15 and vix_score > 0:
            return "LOW_RISK"
        elif current_vol < 0.25 and vix_score >= -0.3:
            return "NORMAL_RISK"
        elif current_vol < 0.35 or vix_score >= -0.6:
            return "HIGH_RISK"
        else:
            return "EXTREME_RISK"
            
    def _get_risk_adjustment(self, volatility_state: str) -> float:
        """Get position size adjustment based on volatility state"""
        adjustments = {
            "LOW_RISK": 1.2,      # Can increase position size
            "NORMAL_RISK": 1.0,   # Normal position size
            "HIGH_RISK": 0.7,     # Reduce position size
            "EXTREME_RISK": 0.4   # Significantly reduce position size
        }
        return adjustments.get(volatility_state, 1.0) 