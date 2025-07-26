#!/usr/bin/env python3
"""
Order Flow Factor - Smart Money Movement Analysis (12% weight)
=============================================================
Implements Institutional Flow Detection, Volume-Weighted Signals,
Order Book Imbalance, and Dark Pool Activity detection.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class OrderFlowFactor(BaseFactor):
    """Smart order flow analysis factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights
        self.sub_weights = {
            'institutional_flow': 0.42,     # 5% of 12%
            'volume_weighted_signals': 0.25, # 3% of 12%
            'order_book_imbalance': 0.17,   # 2% of 12%
            'dark_pool_activity': 0.17       # 2% of 12%
        }
        
        # Thresholds for institutional activity
        self.institutional_thresholds = {
            'block_trade_size': 50000,      # Shares
            'large_order_value': 1000000,   # INR
            'unusual_volume_mult': 2.0      # Times average volume
        }
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate order flow factor"""
        try:
            # Get required data
            historical_df = market_data.get('historical_prices')
            intraday_df = market_data.get('intraday_data')
            
            if historical_df is None or historical_df.empty:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'No historical data available'},
                    sub_factors={}
                )
                
            # Use intraday data if available, otherwise use historical
            df = intraday_df if intraday_df is not None and not intraday_df.empty else historical_df
            
            # Calculate sub-factors
            institutional_score = self._detect_institutional_flow(df)
            volume_weighted_score = self._calculate_volume_weighted_signals(df)
            order_imbalance_score = self._calculate_order_book_imbalance(df)
            dark_pool_score = self._detect_dark_pool_activity(df)
            
            # Calculate weighted score
            sub_factors = {
                'institutional_flow': institutional_score,
                'volume_weighted_signals': volume_weighted_score,
                'order_book_imbalance': order_imbalance_score,
                'dark_pool_activity': dark_pool_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Calculate confidence based on volume quality
            volume_quality = self._assess_volume_quality(df)
            confidence = max(0.3, min(0.9, volume_quality))
            
            # Determine flow state
            flow_state = self._determine_flow_state(sub_factors)
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'flow_state': flow_state,
                    'volume_quality': round(volume_quality, 3),
                    'recent_volume': int(df['volume'].iloc[-1]) if 'volume' in df.columns else 0
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _detect_institutional_flow(self, df: pd.DataFrame) -> float:
        """Detect institutional trading activity"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.0
                
            close_prices = df['close'].values
            volumes = df['volume'].values
            
            # Calculate average volume and price
            avg_volume = np.mean(volumes[-20:])
            avg_price = np.mean(close_prices[-20:])
            recent_volume = volumes[-1]
            recent_price = close_prices[-1]
            
            # Detect unusual volume (potential institutional activity)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate trade value
            avg_trade_value = avg_volume * avg_price
            recent_trade_value = recent_volume * recent_price
            
            # Score based on institutional indicators
            institutional_score = 0.0
            
            # High volume spike
            if volume_ratio > self.institutional_thresholds['unusual_volume_mult']:
                if close_prices[-1] > close_prices[-2]:  # Buying pressure
                    institutional_score += 0.5
                else:  # Selling pressure
                    institutional_score -= 0.5
                    
            # Large trade values
            if recent_trade_value > self.institutional_thresholds['large_order_value']:
                price_impact = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
                if price_impact > 0.005:  # Positive price impact
                    institutional_score += 0.3
                elif price_impact < -0.005:  # Negative price impact
                    institutional_score -= 0.3
                    
            # Volume accumulation/distribution
            volume_trend = self._calculate_volume_trend(volumes[-10:])
            price_trend = self._calculate_price_trend(close_prices[-10:])
            
            if volume_trend > 0.2 and price_trend > 0:
                # Accumulation - institutional buying
                institutional_score += 0.3
            elif volume_trend > 0.2 and price_trend < 0:
                # Distribution - institutional selling
                institutional_score -= 0.3
                
            return max(-1, min(1, institutional_score))
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional flow: {e}")
            return 0.0
            
    def _calculate_volume_weighted_signals(self, df: pd.DataFrame) -> float:
        """Calculate volume-weighted price signals"""
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return 0.0
                
            close_prices = df['close'].values[-10:]
            volumes = df['volume'].values[-10:]
            
            # Calculate VWAP
            vwap = np.sum(close_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else close_prices[-1]
            current_price = close_prices[-1]
            
            # Price position relative to VWAP
            vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0
            
            # Volume-weighted momentum
            price_changes = np.diff(close_prices)
            volume_weights = volumes[1:] / np.sum(volumes[1:]) if np.sum(volumes[1:]) > 0 else np.ones_like(volumes[1:]) / len(volumes[1:])
            weighted_momentum = np.sum(price_changes * volume_weights)
            
            # Score calculation
            vwap_score = np.tanh(vwap_deviation * 100)  # Normalize deviation
            momentum_score = np.tanh(weighted_momentum * 10)  # Normalize momentum
            
            # Combine scores
            volume_weighted_score = vwap_score * 0.6 + momentum_score * 0.4
            
            return max(-1, min(1, volume_weighted_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-weighted signals: {e}")
            return 0.0
            
    def _calculate_order_book_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate order book imbalance (simulated from price/volume data)"""
        try:
            if len(df) < 5:
                return 0.0
                
            close_prices = df['close'].values[-5:]
            volumes = df['volume'].values[-5:]
            high_prices = df['high'].values[-5:] if 'high' in df.columns else close_prices
            low_prices = df['low'].values[-5:] if 'low' in df.columns else close_prices
            
            # Simulate order flow from price movement and volume
            buy_pressure = 0
            sell_pressure = 0
            
            for i in range(len(close_prices)):
                price_range = high_prices[i] - low_prices[i]
                if price_range > 0:
                    # Estimate buy/sell volume based on close position in range
                    close_position = (close_prices[i] - low_prices[i]) / price_range
                    buy_volume = volumes[i] * close_position
                    sell_volume = volumes[i] * (1 - close_position)
                    
                    buy_pressure += buy_volume
                    sell_pressure += sell_volume
                    
            # Calculate imbalance
            total_pressure = buy_pressure + sell_pressure
            if total_pressure > 0:
                imbalance = (buy_pressure - sell_pressure) / total_pressure
            else:
                imbalance = 0.0
                
            # Adjust for recent price action
            recent_trend = (close_prices[-1] - close_prices[0]) / close_prices[0] if close_prices[0] > 0 else 0
            imbalance_score = imbalance * 0.7 + np.tanh(recent_trend * 50) * 0.3
            
            return max(-1, min(1, imbalance_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating order book imbalance: {e}")
            return 0.0
            
    def _detect_dark_pool_activity(self, df: pd.DataFrame) -> float:
        """Detect potential dark pool activity"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.0
                
            close_prices = df['close'].values
            volumes = df['volume'].values
            
            # Dark pool indicators:
            # 1. Price movements without corresponding volume
            # 2. Sudden price gaps
            # 3. Volume spikes at specific price levels
            
            dark_pool_score = 0.0
            
            # Check for price movements without volume
            for i in range(1, min(5, len(close_prices))):
                price_change = abs(close_prices[-i] - close_prices[-i-1]) / close_prices[-i-1]
                volume_ratio = volumes[-i] / np.mean(volumes[-20:-i]) if np.mean(volumes[-20:-i]) > 0 else 1
                
                if price_change > 0.01 and volume_ratio < 0.5:
                    # Significant price move with low volume - potential dark pool
                    dark_pool_score += 0.2
                    
            # Check for price gaps
            if len(df) > 1:
                if 'open' in df.columns:
                    gap = (df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                    if abs(gap) > 0.005:  # 0.5% gap
                        dark_pool_score += 0.3 if gap > 0 else -0.3
                        
            # Check for volume clustering at round numbers
            if len(close_prices) >= 10:
                round_price_volume = 0
                total_volume = 0
                
                for i in range(-10, 0):
                    price = close_prices[i]
                    # Check if price is near round number (multiple of 10)
                    if abs(price - round(price / 10) * 10) / price < 0.002:
                        round_price_volume += volumes[i]
                    total_volume += volumes[i]
                    
                if total_volume > 0 and round_price_volume / total_volume > 0.4:
                    # High volume at round numbers - potential dark pool prints
                    dark_pool_score += 0.2
                    
            return max(-1, min(1, dark_pool_score))
            
        except Exception as e:
            self.logger.error(f"Error detecting dark pool activity: {e}")
            return 0.0
            
    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend"""
        try:
            if len(volumes) < 2:
                return 0.0
                
            # Linear regression slope
            x = np.arange(len(volumes))
            slope = np.polyfit(x, volumes, 1)[0]
            avg_volume = np.mean(volumes)
            
            return slope / avg_volume if avg_volume > 0 else 0.0
            
        except Exception:
            return 0.0
            
    def _calculate_price_trend(self, prices: np.ndarray) -> float:
        """Calculate price trend"""
        try:
            if len(prices) < 2:
                return 0.0
                
            return (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0
            
        except Exception:
            return 0.0
            
    def _assess_volume_quality(self, df: pd.DataFrame) -> float:
        """Assess quality of volume data"""
        try:
            if 'volume' not in df.columns:
                return 0.3
                
            volumes = df['volume'].values[-20:]
            
            # Check for consistent volume data
            zero_volume_days = np.sum(volumes == 0)
            if zero_volume_days > 5:
                return 0.3
                
            # Check for volume consistency
            volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1
            
            if volume_cv < 0.5:
                return 0.9
            elif volume_cv < 1.0:
                return 0.7
            elif volume_cv < 2.0:
                return 0.5
            else:
                return 0.4
                
        except Exception:
            return 0.3
            
    def _determine_flow_state(self, sub_factors: Dict[str, float]) -> str:
        """Determine overall order flow state"""
        avg_score = np.mean(list(sub_factors.values()))
        institutional = sub_factors.get('institutional_flow', 0)
        
        if avg_score > 0.3 and institutional > 0.3:
            return "STRONG_ACCUMULATION"
        elif avg_score > 0.1:
            return "ACCUMULATION"
        elif avg_score < -0.3 and institutional < -0.3:
            return "STRONG_DISTRIBUTION"
        elif avg_score < -0.1:
            return "DISTRIBUTION"
        else:
            return "NEUTRAL_FLOW" 