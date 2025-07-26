#!/usr/bin/env python3
"""
Risk Management Factor - Override Controls (4% weight)
=====================================================
Implements Maximum Drawdown Circuit, Correlation Spike Detector,
and Black Swan Protection.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class RiskManagementFactor(BaseFactor):
    """Risk management override factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights
        self.sub_weights = {
            'max_drawdown_circuit': 0.50,    # 2% of 4%
            'correlation_spike': 0.25,       # 1% of 4%
            'black_swan_protection': 0.25    # 1% of 4%
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_drawdown': 0.15,           # 15% drawdown threshold
            'severe_drawdown': 0.25,        # 25% severe drawdown
            'correlation_spike': 0.8,       # High correlation threshold
            'black_swan_threshold': 0.03,   # 3% single-day move
            'vix_panic': 35                 # VIX panic level
        }
        
        # Override signals
        self.override_active = False
        self.override_reason = None
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate risk management factor with override capability"""
        try:
            # Reset override status
            self.override_active = False
            self.override_reason = None
            
            # Get required data
            historical_df = market_data.get('historical_prices')
            if historical_df is None or historical_df.empty:
                return FactorOutput(
                    score=0.0,
                    confidence=0.3,
                    details={'error': 'No historical data available'},
                    sub_factors={}
                )
                
            # Get market indices for correlation
            market_indices = market_data.get('market_indices', {})
            
            # Calculate sub-factors
            drawdown_score, drawdown_active = self._check_max_drawdown_circuit(historical_df)
            correlation_score, correlation_active = self._detect_correlation_spike(historical_df, market_indices)
            black_swan_score, black_swan_active = self._check_black_swan_protection(historical_df, market_indices)
            
            # Calculate weighted score
            sub_factors = {
                'max_drawdown_circuit': drawdown_score,
                'correlation_spike': correlation_score,
                'black_swan_protection': black_swan_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Check for overrides
            override_signal = None
            if drawdown_active:
                self.override_active = True
                self.override_reason = "Maximum drawdown circuit breaker activated"
                override_signal = "SELL"  # Force sell on severe drawdown
            elif black_swan_active:
                self.override_active = True
                self.override_reason = "Black swan protection activated"
                override_signal = "HOLD"  # Force hold during extreme events
            elif correlation_active and total_score < -0.5:
                self.override_active = True
                self.override_reason = "High correlation spike detected"
                override_signal = "HOLD"  # Reduce exposure during correlation spikes
                
            # Calculate confidence based on risk conditions
            risk_level = self._assess_overall_risk_level(sub_factors)
            confidence = max(0.3, 1.0 - risk_level * 0.7)
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'override_active': self.override_active,
                    'override_signal': override_signal,
                    'override_reason': self.override_reason,
                    'risk_level': risk_level,
                    'circuit_breakers': {
                        'drawdown': drawdown_active,
                        'correlation': correlation_active,
                        'black_swan': black_swan_active
                    }
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk management for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _check_max_drawdown_circuit(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """Check maximum drawdown circuit breaker"""
        try:
            if len(df) < 20:
                return 0.0, False
                
            close_prices = df['close'].values
            
            # Calculate drawdown from recent peak
            recent_prices = close_prices[-60:] if len(close_prices) >= 60 else close_prices
            running_max = np.maximum.accumulate(recent_prices)
            drawdown = (recent_prices - running_max) / running_max
            current_drawdown = drawdown[-1]
            max_drawdown = np.min(drawdown)
            
            # Calculate drawdown velocity (speed of decline)
            if len(drawdown) >= 5:
                drawdown_velocity = (drawdown[-1] - drawdown[-5]) / 5
            else:
                drawdown_velocity = 0
                
            # Score calculation
            circuit_active = False
            
            if current_drawdown < -self.risk_thresholds['severe_drawdown']:
                # Severe drawdown - activate circuit breaker
                score = -1.0
                circuit_active = True
            elif current_drawdown < -self.risk_thresholds['max_drawdown']:
                # Significant drawdown
                score = -0.7
                # Activate if rapid decline
                if drawdown_velocity < -0.02:  # 2% decline per day
                    circuit_active = True
            elif current_drawdown < -0.10:
                # Moderate drawdown
                score = -0.4
            elif current_drawdown < -0.05:
                # Minor drawdown
                score = -0.2
            else:
                # No significant drawdown
                score = 0.1
                
            # Check for recovery
            if len(drawdown) >= 10:
                recent_recovery = drawdown[-1] - np.min(drawdown[-10:])
                if recent_recovery > 0.03:  # 3% recovery
                    score += 0.2
                    circuit_active = False  # Deactivate on recovery
                    
            return max(-1, min(1, score)), circuit_active
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown circuit: {e}")
            return 0.0, False
            
    def _detect_correlation_spike(self, df: pd.DataFrame, market_indices: Dict) -> Tuple[float, bool]:
        """Detect correlation spike with market"""
        try:
            if len(df) < 20:
                return 0.0, False
                
            close_prices = df['close'].values
            
            # Calculate returns
            stock_returns = np.diff(close_prices[-20:]) / close_prices[-21:-1]
            
            # Get market returns (simplified using index data)
            market_change = 0
            if market_indices and 'NIFTY' in market_indices:
                nifty_data = market_indices['NIFTY']
                if nifty_data and 'change_p' in nifty_data:
                    market_change = nifty_data['change_p'] / 100
                    
            # Calculate rolling correlation (simplified)
            # In production, would use actual rolling correlation
            avg_stock_return = np.mean(stock_returns)
            stock_volatility = np.std(stock_returns)
            
            # Correlation proxy based on directional alignment
            if market_change != 0:
                direction_alignment = np.sign(avg_stock_return) * np.sign(market_change)
                magnitude_ratio = min(abs(avg_stock_return), abs(market_change)) / max(abs(avg_stock_return), abs(market_change))
                correlation_proxy = direction_alignment * magnitude_ratio
            else:
                correlation_proxy = 0
                
            # Check for correlation spike
            spike_detected = False
            
            if abs(correlation_proxy) > self.risk_thresholds['correlation_spike']:
                spike_detected = True
                score = -0.5  # High correlation is risky
            elif abs(correlation_proxy) > 0.6:
                score = -0.3
            elif abs(correlation_proxy) > 0.4:
                score = -0.1
            else:
                score = 0.1  # Low correlation is good for diversification
                
            # Check for volatility clustering
            recent_large_moves = sum(abs(r) > 0.02 for r in stock_returns[-5:])
            if recent_large_moves >= 3 and abs(correlation_proxy) > 0.5:
                spike_detected = True
                score -= 0.2
                
            return max(-1, min(1, score)), spike_detected
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation spike: {e}")
            return 0.0, False
            
    def _check_black_swan_protection(self, df: pd.DataFrame, market_indices: Dict) -> Tuple[float, bool]:
        """Check for black swan events"""
        try:
            if len(df) < 5:
                return 0.0, False
                
            close_prices = df['close'].values
            
            # Check for extreme single-day moves
            if len(close_prices) >= 2:
                recent_return = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
                yesterday_return = (close_prices[-2] - close_prices[-3]) / close_prices[-3] if len(close_prices) >= 3 else 0
            else:
                recent_return = 0
                yesterday_return = 0
                
            # Check for extreme market moves
            market_stress = False
            if market_indices:
                for idx_name, idx_data in market_indices.items():
                    if idx_data and 'change_p' in idx_data:
                        if abs(idx_data['change_p']) > 3:  # 3% market move
                            market_stress = True
                            break
                            
            # Black swan detection
            black_swan_detected = False
            
            # Extreme single-day move
            if abs(recent_return) > self.risk_thresholds['black_swan_threshold']:
                black_swan_detected = True
                score = -0.8
            # Consecutive large moves
            elif abs(recent_return) > 0.02 and abs(yesterday_return) > 0.02:
                score = -0.5
                if np.sign(recent_return) != np.sign(yesterday_return):
                    # Whipsaw pattern - very dangerous
                    black_swan_detected = True
                    score = -0.9
            # Market stress
            elif market_stress:
                score = -0.4
                if abs(recent_return) > 0.015:
                    black_swan_detected = True
            else:
                score = 0.0
                
            # Check for gap moves
            if 'open' in df.columns and len(df) >= 2:
                gap = (df['open'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                if abs(gap) > 0.02:  # 2% gap
                    black_swan_detected = True
                    score = min(score, -0.6)
                    
            # VIX-like volatility check
            if len(close_prices) >= 20:
                returns = np.diff(close_prices[-20:]) / close_prices[-21:-1]
                realized_vol = np.std(returns) * np.sqrt(252) * 100
                if realized_vol > self.risk_thresholds['vix_panic']:
                    score -= 0.3
                    
            return max(-1, min(1, score)), black_swan_detected
            
        except Exception as e:
            self.logger.error(f"Error checking black swan protection: {e}")
            return 0.0, False
            
    def _assess_overall_risk_level(self, sub_factors: Dict[str, float]) -> float:
        """Assess overall risk level (0 to 1)"""
        try:
            # Calculate risk from negative scores
            risk_scores = [-score for score in sub_factors.values() if score < 0]
            
            if not risk_scores:
                return 0.1  # Low risk
                
            # Weighted average of risk scores
            avg_risk = np.mean(risk_scores)
            max_risk = max(risk_scores)
            
            # Overall risk is combination of average and maximum
            overall_risk = avg_risk * 0.7 + max_risk * 0.3
            
            return min(1.0, overall_risk)
            
        except Exception:
            return 0.5  # Medium risk on error 