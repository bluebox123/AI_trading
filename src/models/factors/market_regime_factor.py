#!/usr/bin/env python3
"""
Market Regime Factor - Market State Detection (6% weight)
========================================================
Implements Trend Regime Classification, Volatility Regime State,
and Liquidity Regime Assessment.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import talib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class MarketRegimeFactor(BaseFactor):
    """Market regime detection factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights
        self.sub_weights = {
            'trend_regime': 0.50,       # 3% of 6%
            'volatility_regime': 0.33,  # 2% of 6%
            'liquidity_regime': 0.17    # 1% of 6%
        }
        
        # Regime thresholds
        self.regime_thresholds = {
            'trend_strength': 0.02,     # 2% for trend classification
            'volatility_low': 0.10,     # 10% annualized vol
            'volatility_high': 0.30,    # 30% annualized vol
            'liquidity_threshold': 0.7  # Liquidity score threshold
        }
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate market regime factor"""
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
                
            # Get market indices
            market_indices = market_data.get('market_indices', {})
            
            # Calculate sub-factors
            trend_score, trend_regime = self._classify_trend_regime(historical_df)
            vol_score, vol_regime = self._assess_volatility_regime(historical_df)
            liq_score, liq_regime = self._assess_liquidity_regime(historical_df, market_indices)
            
            # Calculate weighted score
            sub_factors = {
                'trend_regime': trend_score,
                'volatility_regime': vol_score,
                'liquidity_regime': liq_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Calculate confidence based on regime clarity
            regime_scores = [abs(score) for score in sub_factors.values()]
            avg_clarity = np.mean(regime_scores)
            confidence = max(0.3, min(0.9, 0.5 + avg_clarity * 0.5))
            
            # Determine overall market regime
            overall_regime = self._determine_overall_regime(trend_regime, vol_regime, liq_regime)
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'trend_regime': trend_regime,
                    'volatility_regime': vol_regime,
                    'liquidity_regime': liq_regime,
                    'overall_regime': overall_regime,
                    'regime_stability': self._assess_regime_stability(historical_df)
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _classify_trend_regime(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Classify trend regime"""
        try:
            if len(df) < 50:
                return 0.0, "INSUFFICIENT_DATA"
                
            close_prices = df['close'].values
            
            # Calculate multiple timeframe trends
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            sma_200 = talib.SMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
            
            current_price = close_prices[-1]
            
            # Short-term trend (20-day)
            short_trend = (current_price - sma_20[-1]) / sma_20[-1] if not np.isnan(sma_20[-1]) else 0
            
            # Medium-term trend (50-day)
            medium_trend = (current_price - sma_50[-1]) / sma_50[-1] if not np.isnan(sma_50[-1]) else 0
            
            # Long-term trend (200-day if available)
            if sma_200 is not None and not np.isnan(sma_200[-1]):
                long_trend = (current_price - sma_200[-1]) / sma_200[-1]
            else:
                long_trend = medium_trend  # Fallback to medium trend
                
            # ADX for trend strength
            if len(df) >= 14 and all(col in df.columns for col in ['high', 'low']):
                adx = talib.ADX(df['high'].values, df['low'].values, close_prices, timeperiod=14)
                trend_strength = adx[-1] if not np.isnan(adx[-1]) else 25
            else:
                trend_strength = 25  # Default moderate strength
                
            # Classify regime
            avg_trend = (short_trend + medium_trend + long_trend) / 3
            
            if trend_strength < 20:  # Weak trend
                regime = "RANGE_BOUND"
                score = 0.0
            elif avg_trend > self.regime_thresholds['trend_strength']:
                if trend_strength > 40:
                    regime = "STRONG_UPTREND"
                    score = 0.8
                else:
                    regime = "UPTREND"
                    score = 0.5
            elif avg_trend < -self.regime_thresholds['trend_strength']:
                if trend_strength > 40:
                    regime = "STRONG_DOWNTREND"
                    score = -0.8
                else:
                    regime = "DOWNTREND"
                    score = -0.5
            else:
                regime = "SIDEWAYS"
                score = 0.0
                
            # Adjust score for trend consistency
            if short_trend * medium_trend > 0:  # Same direction
                score *= 1.2
                
            return max(-1, min(1, score)), regime
            
        except Exception as e:
            self.logger.error(f"Error classifying trend regime: {e}")
            return 0.0, "ERROR"
            
    def _assess_volatility_regime(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Assess volatility regime"""
        try:
            if len(df) < 30:
                return 0.0, "INSUFFICIENT_DATA"
                
            close_prices = df['close'].values
            
            # Calculate volatility metrics
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Historical volatility (different periods)
            vol_10d = np.std(returns[-10:]) * np.sqrt(252)
            vol_30d = np.std(returns[-30:]) * np.sqrt(252)
            vol_60d = np.std(returns[-60:]) * np.sqrt(252) if len(returns) >= 60 else vol_30d
            
            # Current volatility level
            current_vol = vol_10d
            
            # Volatility trend
            vol_trend = (vol_10d - vol_30d) / vol_30d if vol_30d > 0 else 0
            
            # Classify regime
            if current_vol < self.regime_thresholds['volatility_low']:
                regime = "LOW_VOLATILITY"
                base_score = 0.3  # Low vol generally positive
            elif current_vol < self.regime_thresholds['volatility_high']:
                regime = "NORMAL_VOLATILITY"
                base_score = 0.0
            else:
                regime = "HIGH_VOLATILITY"
                base_score = -0.5  # High vol generally negative
                
            # Adjust for volatility trend
            if vol_trend > 0.2:  # Rising volatility
                regime = "RISING_" + regime
                base_score -= 0.2
            elif vol_trend < -0.2:  # Falling volatility
                regime = "FALLING_" + regime
                base_score += 0.2
                
            # Check for volatility clustering
            recent_large_moves = sum(abs(r) > 0.02 for r in returns[-10:])
            if recent_large_moves > 3:
                base_score -= 0.1  # Recent volatility clustering
                
            return max(-1, min(1, base_score)), regime
            
        except Exception as e:
            self.logger.error(f"Error assessing volatility regime: {e}")
            return 0.0, "ERROR"
            
    def _assess_liquidity_regime(self, df: pd.DataFrame, market_indices: Dict) -> Tuple[float, str]:
        """Assess liquidity regime"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.0, "INSUFFICIENT_DATA"
                
            volumes = df['volume'].values
            close_prices = df['close'].values
            
            # Volume-based liquidity metrics
            avg_volume_20d = np.mean(volumes[-20:])
            avg_volume_60d = np.mean(volumes[-60:]) if len(volumes) >= 60 else avg_volume_20d
            recent_volume = volumes[-1]
            
            # Volume trend
            volume_ratio = recent_volume / avg_volume_20d if avg_volume_20d > 0 else 1
            volume_trend = (avg_volume_20d - avg_volume_60d) / avg_volume_60d if avg_volume_60d > 0 else 0
            
            # Price impact metric (simplified)
            if len(df) >= 5:
                price_changes = np.abs(np.diff(close_prices[-5:]) / close_prices[-6:-1])
                volume_changes = volumes[-5:] / avg_volume_20d
                # High price change with low volume = low liquidity
                price_impact = np.mean(price_changes) / (np.mean(volume_changes) + 0.1)
            else:
                price_impact = 1.0
                
            # Market-wide liquidity (from indices)
            market_vol = 0
            if market_indices:
                for idx_name, idx_data in market_indices.items():
                    if idx_data and 'volume' in idx_data:
                        market_vol += idx_data.get('volume', 0)
                        
            # Calculate liquidity score
            liquidity_score = 0.0
            
            # Volume adequacy
            if volume_ratio > 1.5:
                liquidity_score += 0.3  # High volume = good liquidity
            elif volume_ratio < 0.5:
                liquidity_score -= 0.3  # Low volume = poor liquidity
                
            # Price impact
            if price_impact < 0.5:
                liquidity_score += 0.2  # Low impact = good liquidity
            elif price_impact > 2.0:
                liquidity_score -= 0.2  # High impact = poor liquidity
                
            # Volume trend
            if volume_trend > 0:
                liquidity_score += 0.1  # Improving liquidity
            elif volume_trend < -0.2:
                liquidity_score -= 0.1  # Deteriorating liquidity
                
            # Classify regime
            if liquidity_score > self.regime_thresholds['liquidity_threshold']:
                regime = "HIGH_LIQUIDITY"
            elif liquidity_score > 0:
                regime = "NORMAL_LIQUIDITY"
            elif liquidity_score > -self.regime_thresholds['liquidity_threshold']:
                regime = "LOW_LIQUIDITY"
            else:
                regime = "ILLIQUID"
                
            return max(-1, min(1, liquidity_score)), regime
            
        except Exception as e:
            self.logger.error(f"Error assessing liquidity regime: {e}")
            return 0.0, "ERROR"
            
    def _assess_regime_stability(self, df: pd.DataFrame) -> float:
        """Assess how stable the current regime is"""
        try:
            if len(df) < 20:
                return 0.5
                
            close_prices = df['close'].values[-20:]
            
            # Calculate regime changes
            sma_5 = talib.SMA(close_prices, timeperiod=5)
            sma_10 = talib.SMA(close_prices, timeperiod=10)
            
            # Count crossovers (regime changes)
            crossovers = 0
            for i in range(1, len(sma_5)):
                if not np.isnan(sma_5[i]) and not np.isnan(sma_10[i]):
                    if (sma_5[i-1] < sma_10[i-1] and sma_5[i] > sma_10[i]) or \
                       (sma_5[i-1] > sma_10[i-1] and sma_5[i] < sma_10[i]):
                        crossovers += 1
                        
            # More crossovers = less stable
            if crossovers == 0:
                return 1.0  # Very stable
            elif crossovers <= 2:
                return 0.7  # Stable
            elif crossovers <= 4:
                return 0.4  # Unstable
            else:
                return 0.2  # Very unstable
                
        except Exception:
            return 0.5
            
    def _determine_overall_regime(self, trend_regime: str, vol_regime: str, liq_regime: str) -> str:
        """Determine overall market regime"""
        # Combine regimes to determine market state
        if "STRONG_UPTREND" in trend_regime and "LOW_VOLATILITY" in vol_regime:
            return "BULL_QUIET"  # Best for longs
        elif "UPTREND" in trend_regime and "NORMAL" in vol_regime:
            return "BULL_NORMAL"
        elif "STRONG_DOWNTREND" in trend_regime:
            return "BEAR_MARKET"
        elif "HIGH_VOLATILITY" in vol_regime and "LOW_LIQUIDITY" in liq_regime:
            return "STRESSED"  # Avoid trading
        elif "RANGE_BOUND" in trend_regime or "SIDEWAYS" in trend_regime:
            return "CONSOLIDATION"
        else:
            return "TRANSITIONAL" 