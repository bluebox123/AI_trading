#!/usr/bin/env python3
"""
Technical Indicators Factor - Advanced Technical Analysis (22% weight)
====================================================================
Implements RSI, MACD, Moving Average Confluence, Bollinger Bands, 
Volume Profile Analysis, and Key Level Analysis.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging
import talib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class TechnicalIndicatorsFactor(BaseFactor):
    """Advanced technical indicators analysis factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights (must sum to 1.0)
        self.sub_weights = {
            'rsi_macd_momentum': 0.36,  # 8% of 22%
            'ma_confluence': 0.23,      # 5% of 22%
            'bollinger_volatility': 0.18,  # 4% of 22%
            'volume_profile': 0.14,     # 3% of 22%
            'key_levels': 0.09          # 2% of 22%
        }
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate technical indicators factor"""
        try:
            # Get historical prices
            historical_df = market_data.get('historical_prices')
            if historical_df is None or historical_df.empty:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'No historical data available'},
                    sub_factors={}
                )
                
            # Ensure we have required columns
            if 'close' not in historical_df.columns:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'Missing close price data'},
                    sub_factors={}
                )
                
            close_prices = historical_df['close'].values
            high_prices = historical_df['high'].values if 'high' in historical_df.columns else close_prices
            low_prices = historical_df['low'].values if 'low' in historical_df.columns else close_prices
            volume = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # Calculate sub-factors
            rsi_macd_score = self._calculate_rsi_macd_momentum(close_prices)
            ma_confluence_score = self._calculate_ma_confluence(close_prices)
            bollinger_score = self._calculate_bollinger_bands(close_prices)
            volume_score = self._calculate_volume_profile(close_prices, volume)
            key_levels_score = self._calculate_key_levels(high_prices, low_prices, close_prices)
            
            # Calculate weighted score
            sub_factors = {
                'rsi_macd_momentum': rsi_macd_score,
                'ma_confluence': ma_confluence_score,
                'bollinger_volatility': bollinger_score,
                'volume_profile': volume_score,
                'key_levels': key_levels_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Calculate confidence based on data quality and agreement
            scores = list(sub_factors.values())
            score_std = np.std(scores) if len(scores) > 1 else 0.5
            confidence = max(0.3, min(0.9, 1.0 - score_std))
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'current_price': close_prices[-1] if len(close_prices) > 0 else 0,
                    'price_change_pct': ((close_prices[-1] - close_prices[-2]) / close_prices[-2] * 100) if len(close_prices) > 1 else 0
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _calculate_rsi_macd_momentum(self, close_prices: np.ndarray) -> float:
        """Calculate RSI and MACD momentum score"""
        try:
            if len(close_prices) < 26:  # Need at least 26 periods for MACD
                return 0.0
                
            # Calculate RSI
            rsi = talib.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # RSI score: Oversold (< 30) = positive, Overbought (> 70) = negative
            if current_rsi < 30:
                rsi_score = (30 - current_rsi) / 30  # 0 to 1
            elif current_rsi > 70:
                rsi_score = (70 - current_rsi) / 30  # 0 to -1
            else:
                rsi_score = 0.0
                
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                                      fastperiod=12, 
                                                      slowperiod=26, 
                                                      signalperiod=9)
            
            # MACD score: Positive histogram = bullish, negative = bearish
            if not np.isnan(macd_hist[-1]):
                macd_score = np.tanh(macd_hist[-1] / (np.std(macd_hist[-20:]) + 1e-6))
            else:
                macd_score = 0.0
                
            # Combine RSI and MACD
            momentum_score = (rsi_score * 0.5 + macd_score * 0.5)
            
            return max(-1, min(1, momentum_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI/MACD: {e}")
            return 0.0
            
    def _calculate_ma_confluence(self, close_prices: np.ndarray) -> float:
        """Calculate moving average confluence score"""
        try:
            if len(close_prices) < 200:  # Need at least 200 periods
                return 0.0
                
            # Calculate multiple MAs
            ma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
            ma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
            ma_100 = talib.SMA(close_prices, timeperiod=100)[-1]
            ma_200 = talib.SMA(close_prices, timeperiod=200)[-1]
            
            current_price = close_prices[-1]
            
            # Check MA alignment
            mas = [ma_20, ma_50, ma_100, ma_200]
            if all(not np.isnan(ma) for ma in mas):
                # Bullish: Price > MA20 > MA50 > MA100 > MA200
                if current_price > ma_20 > ma_50 > ma_100 > ma_200:
                    alignment_score = 1.0
                # Bearish: Price < MA20 < MA50 < MA100 < MA200
                elif current_price < ma_20 < ma_50 < ma_100 < ma_200:
                    alignment_score = -1.0
                else:
                    # Partial alignment
                    bullish_count = sum([
                        current_price > ma_20,
                        ma_20 > ma_50,
                        ma_50 > ma_100,
                        ma_100 > ma_200
                    ])
                    alignment_score = (bullish_count - 2) / 2  # -1 to 1
                    
                # Distance from MAs
                avg_distance = np.mean([
                    (current_price - ma) / ma for ma in mas
                ])
                distance_score = np.tanh(avg_distance * 10)  # Normalize
                
                # Combine alignment and distance
                confluence_score = alignment_score * 0.7 + distance_score * 0.3
                
                return max(-1, min(1, confluence_score))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating MA confluence: {e}")
            return 0.0
            
    def _calculate_bollinger_bands(self, close_prices: np.ndarray) -> float:
        """Calculate Bollinger Bands volatility score"""
        try:
            if len(close_prices) < 20:
                return 0.0
                
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            current_price = close_prices[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]
            current_middle = middle[-1]
            
            if np.isnan(current_upper) or np.isnan(current_lower):
                return 0.0
                
            # Calculate position within bands
            band_width = current_upper - current_lower
            if band_width > 0:
                position = (current_price - current_lower) / band_width
                
                # Score based on position
                if position < 0.2:  # Near lower band - oversold
                    score = (0.2 - position) * 5  # 0 to 1
                elif position > 0.8:  # Near upper band - overbought
                    score = (0.8 - position) * 5  # 0 to -1
                else:
                    # Middle zone - check trend
                    if current_price > current_middle:
                        score = 0.1  # Slightly bullish
                    else:
                        score = -0.1  # Slightly bearish
                        
                # Adjust for band expansion/contraction
                avg_band_width = np.mean(upper[-20:] - lower[-20:])
                if band_width < avg_band_width * 0.8:  # Bands contracting
                    score *= 1.2  # Increase signal strength
                elif band_width > avg_band_width * 1.2:  # Bands expanding
                    score *= 0.8  # Decrease signal strength
                    
                return max(-1, min(1, score))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.0
            
    def _calculate_volume_profile(self, close_prices: np.ndarray, volume: np.ndarray) -> float:
        """Calculate volume profile analysis score"""
        try:
            if volume is None or len(volume) < 20:
                return 0.0
                
            # Calculate volume-weighted average price (VWAP)
            vwap = np.sum(close_prices[-20:] * volume[-20:]) / np.sum(volume[-20:])
            current_price = close_prices[-1]
            
            # Price vs VWAP
            vwap_score = (current_price - vwap) / vwap
            
            # Volume trend
            avg_volume = np.mean(volume[-20:])
            recent_volume = np.mean(volume[-5:])
            volume_trend = (recent_volume - avg_volume) / avg_volume
            
            # High volume on up days vs down days
            price_changes = np.diff(close_prices[-21:])
            up_volume = np.sum(volume[-20:][price_changes > 0])
            down_volume = np.sum(volume[-20:][price_changes < 0])
            
            if up_volume + down_volume > 0:
                volume_bias = (up_volume - down_volume) / (up_volume + down_volume)
            else:
                volume_bias = 0.0
                
            # Combine scores
            volume_score = (
                np.tanh(vwap_score * 10) * 0.4 +  # VWAP position
                np.tanh(volume_trend * 2) * 0.3 +  # Volume trend
                volume_bias * 0.3  # Volume direction bias
            )
            
            return max(-1, min(1, volume_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return 0.0
            
    def _calculate_key_levels(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                            close_prices: np.ndarray) -> float:
        """Calculate key support/resistance levels score"""
        try:
            if len(close_prices) < 50:
                return 0.0
                
            current_price = close_prices[-1]
            
            # Find recent highs and lows (peaks and troughs)
            # Using 5-period window for local extremes
            recent_highs = []
            recent_lows = []
            
            for i in range(5, len(high_prices) - 5):
                # Local high
                if high_prices[i] == max(high_prices[i-5:i+6]):
                    recent_highs.append(high_prices[i])
                # Local low
                if low_prices[i] == min(low_prices[i-5:i+6]):
                    recent_lows.append(low_prices[i])
                    
            if not recent_highs or not recent_lows:
                return 0.0
                
            # Find nearest resistance and support
            resistances = [h for h in recent_highs if h > current_price]
            supports = [l for l in recent_lows if l < current_price]
            
            nearest_resistance = min(resistances) if resistances else max(recent_highs)
            nearest_support = max(supports) if supports else min(recent_lows)
            
            # Calculate position relative to key levels
            range_size = nearest_resistance - nearest_support
            if range_size > 0:
                position = (current_price - nearest_support) / range_size
                
                # Score based on proximity to levels
                if position < 0.2:  # Near support
                    score = 0.5  # Bullish
                elif position > 0.8:  # Near resistance
                    score = -0.5  # Bearish
                else:
                    # Middle of range - neutral with slight trend bias
                    trend = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
                    score = np.tanh(trend * 50)
                    
                # Adjust for level strength (how many times tested)
                support_tests = sum(1 for l in low_prices[-20:] if abs(l - nearest_support) / nearest_support < 0.01)
                resistance_tests = sum(1 for h in high_prices[-20:] if abs(h - nearest_resistance) / nearest_resistance < 0.01)
                
                level_strength = (support_tests + resistance_tests) / 10
                score *= (1 + level_strength)
                
                return max(-1, min(1, score))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating key levels: {e}")
            return 0.0 