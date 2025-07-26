#!/usr/bin/env python3
"""
Kelly Criterion Engine - Optimal Position Sizing
================================================
Implements the Kelly Criterion for optimal position sizing based on:
- Win rate (probability of winning)
- Average win size
- Average loss size
- Risk tolerance and portfolio constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class KellyCriterionEngine:
    """
    Kelly Criterion Engine for optimal position sizing
    """
    
    def __init__(self, risk_free_rate: float = 0.05, max_kelly_fraction: float = 0.25):
        """
        Initialize Kelly Criterion Engine
        
        Args:
            risk_free_rate: Risk-free rate (default 5%)
            max_kelly_fraction: Maximum Kelly fraction to use (default 25% for safety)
        """
        self.risk_free_rate = risk_free_rate
        self.max_kelly_fraction = max_kelly_fraction
        self.min_data_points = 20  # Minimum data points for reliable calculation
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                               avg_loss: float, volatility: float = None) -> Dict[str, float]:
        """
        Calculate Kelly Criterion fraction
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win size (as percentage or absolute)
            avg_loss: Average loss size (as percentage or absolute)
            volatility: Optional volatility for risk adjustment
            
        Returns:
            dict: Kelly fraction and related metrics
        """
        try:
            # Validate inputs
            if not (0 <= win_rate <= 1):
                raise ValueError("Win rate must be between 0 and 1")
            
            if avg_win <= 0 or avg_loss <= 0:
                raise ValueError("Average win and loss must be positive")
            
            # Calculate basic Kelly fraction
            # Kelly = (bp - q) / b
            # where: b = odds received on bet, p = probability of winning, q = probability of losing
            
            # Convert to Kelly formula format
            b = avg_win / avg_loss  # Odds received
            p = win_rate  # Probability of winning
            q = 1 - win_rate  # Probability of losing
            
            # Kelly fraction
            kelly_fraction = (b * p - q) / b
            
            # Calculate safe Kelly (fraction of full Kelly)
            safe_kelly = kelly_fraction * 0.25  # Use 25% of full Kelly for safety
            
            # Apply volatility adjustment if provided
            volatility_adjustment = 1.0
            if volatility is not None:
                # Reduce position size for high volatility
                if volatility > 0.3:  # High volatility
                    volatility_adjustment = 0.7
                elif volatility > 0.2:  # Medium volatility
                    volatility_adjustment = 0.85
                elif volatility < 0.1:  # Low volatility
                    volatility_adjustment = 1.1
                
                safe_kelly *= volatility_adjustment
            
            # Ensure Kelly fraction is within bounds
            safe_kelly = max(0, min(safe_kelly, self.max_kelly_fraction))
            
            # Calculate expected value and risk metrics
            expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            win_loss_ratio = avg_win / avg_loss
            
            # Calculate Sharpe ratio approximation
            if volatility and volatility > 0:
                sharpe_ratio = expected_value / volatility
            else:
                sharpe_ratio = expected_value / 0.2  # Default volatility
            
            return {
                'kelly_fraction': round(kelly_fraction, 4),
                'safe_kelly_fraction': round(safe_kelly, 4),
                'win_rate': round(win_rate, 4),
                'avg_win': round(avg_win, 4),
                'avg_loss': round(avg_loss, 4),
                'win_loss_ratio': round(win_loss_ratio, 4),
                'expected_value': round(expected_value, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'volatility_adjustment': round(volatility_adjustment, 4),
                'recommendation_strength': round(abs(safe_kelly) * 10, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return {
                'kelly_fraction': 0.0,
                'safe_kelly_fraction': 0.0,
                'win_rate': 0.5,
                'avg_win': 0.02,
                'avg_loss': 0.02,
                'win_loss_ratio': 1.0,
                'expected_value': 0.0,
                'sharpe_ratio': 0.0,
                'volatility_adjustment': 1.0,
                'recommendation_strength': 0.0
            }
    
    def calculate_historical_kelly(self, symbol: str, historical_data: pd.DataFrame, 
                                 lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate Kelly Criterion based on historical performance
        
        Args:
            symbol: Stock symbol
            historical_data: Historical price data
            lookback_days: Number of days to look back
            
        Returns:
            dict: Kelly metrics based on historical performance
        """
        try:
            if len(historical_data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(historical_data)} points")
                return self._get_default_kelly_metrics()
            
            # Calculate returns
            historical_data = historical_data.tail(lookback_days)
            returns = historical_data['close'].pct_change().dropna()
            
            if len(returns) < self.min_data_points:
                return self._get_default_kelly_metrics()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Define winning and losing trades
            # A "win" is a positive return, a "loss" is a negative return
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return self._get_default_kelly_metrics()
            
            # Calculate win rate and average win/loss
            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())  # Make positive
            
            # Calculate Kelly fraction
            kelly_metrics = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss, volatility)
            
            # Add historical context
            kelly_metrics.update({
                'symbol': symbol,
                'data_points': len(returns),
                'lookback_days': lookback_days,
                'volatility_annual': round(volatility, 4),
                'max_return': round(returns.max(), 4),
                'min_return': round(returns.min(), 4),
                'total_return': round(returns.sum(), 4),
                'positive_days': len(wins),
                'negative_days': len(losses),
                'neutral_days': len(returns) - len(wins) - len(losses)
            })
            
            return kelly_metrics
            
        except Exception as e:
            logger.error(f"Error calculating historical Kelly for {symbol}: {e}")
            return self._get_default_kelly_metrics()
    
    def calculate_signal_based_kelly(self, signal_confidence: float, signal_direction: str,
                                   volatility: float, base_kelly: float = 0.02) -> Dict[str, float]:
        """
        Calculate Kelly fraction based on signal confidence and direction
        
        Args:
            signal_confidence: Signal confidence (0-1)
            signal_direction: Signal direction ('BUY', 'SELL', 'HOLD')
            volatility: Asset volatility
            base_kelly: Base Kelly fraction for the asset
            
        Returns:
            dict: Adjusted Kelly metrics based on signal
        """
        try:
            # Base Kelly fraction
            kelly_fraction = base_kelly
            
            # Adjust based on signal confidence
            confidence_multiplier = 0.5 + (signal_confidence * 0.5)  # 0.5 to 1.0
            kelly_fraction *= confidence_multiplier
            
            # Adjust based on signal direction
            if signal_direction == 'BUY':
                direction_multiplier = 1.0
            elif signal_direction == 'SELL':
                direction_multiplier = 0.8  # Slightly more conservative for sells
            else:  # HOLD
                direction_multiplier = 0.3  # Much more conservative for holds
            
            kelly_fraction *= direction_multiplier
            
            # Adjust for volatility
            volatility_adjustment = 1.0
            if volatility > 0.3:
                volatility_adjustment = 0.7
            elif volatility > 0.2:
                volatility_adjustment = 0.85
            elif volatility < 0.1:
                volatility_adjustment = 1.1
            
            kelly_fraction *= volatility_adjustment
            
            # Calculate safe Kelly
            safe_kelly = min(kelly_fraction, self.max_kelly_fraction)
            
            return {
                'kelly_fraction': round(kelly_fraction, 4),
                'safe_kelly_fraction': round(safe_kelly, 4),
                'signal_confidence': round(signal_confidence, 4),
                'signal_direction': signal_direction,
                'volatility': round(volatility, 4),
                'confidence_multiplier': round(confidence_multiplier, 4),
                'direction_multiplier': round(direction_multiplier, 4),
                'volatility_adjustment': round(volatility_adjustment, 4),
                'recommendation_strength': round(abs(safe_kelly) * 10, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal-based Kelly: {e}")
            return {
                'kelly_fraction': 0.0,
                'safe_kelly_fraction': 0.0,
                'signal_confidence': signal_confidence,
                'signal_direction': signal_direction,
                'volatility': volatility,
                'confidence_multiplier': 1.0,
                'direction_multiplier': 1.0,
                'volatility_adjustment': 1.0,
                'recommendation_strength': 0.0
            }
    
    def calculate_portfolio_kelly(self, positions: Dict[str, Dict], 
                                portfolio_value: float) -> Dict[str, float]:
        """
        Calculate portfolio-level Kelly metrics
        
        Args:
            positions: Current portfolio positions
            portfolio_value: Total portfolio value
            
        Returns:
            dict: Portfolio Kelly metrics
        """
        try:
            if not positions:
                return {
                    'portfolio_kelly': 0.0,
                    'total_exposure': 0.0,
                    'diversification_score': 1.0,
                    'risk_concentration': 0.0
                }
            
            # Calculate total exposure
            total_exposure = sum(pos.get('value', 0) for pos in positions.values())
            exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate position weights
            weights = []
            for pos in positions.values():
                weight = pos.get('value', 0) / portfolio_value if portfolio_value > 0 else 0
                weights.append(weight)
            
            # Calculate diversification score (Herfindahl index)
            weights_array = np.array(weights)
            diversification_score = 1 - np.sum(weights_array ** 2)
            
            # Calculate risk concentration
            risk_concentration = max(weights) if weights else 0
            
            # Portfolio Kelly (weighted average of position Kelly fractions)
            portfolio_kelly = 0.0
            for pos in positions.values():
                kelly_fraction = pos.get('kelly_fraction', 0)
                weight = pos.get('value', 0) / portfolio_value if portfolio_value > 0 else 0
                portfolio_kelly += kelly_fraction * weight
            
            return {
                'portfolio_kelly': round(portfolio_kelly, 4),
                'total_exposure': round(exposure_ratio, 4),
                'diversification_score': round(diversification_score, 4),
                'risk_concentration': round(risk_concentration, 4),
                'position_count': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Kelly: {e}")
            return {
                'portfolio_kelly': 0.0,
                'total_exposure': 0.0,
                'diversification_score': 1.0,
                'risk_concentration': 0.0,
                'position_count': 0
            }
    
    def get_position_size_recommendation(self, symbol: str, kelly_metrics: Dict,
                                       portfolio_value: float, max_position_size: float = 0.1) -> Dict[str, float]:
        """
        Get position size recommendation based on Kelly metrics
        
        Args:
            symbol: Stock symbol
            kelly_metrics: Kelly Criterion metrics
            portfolio_value: Total portfolio value
            max_position_size: Maximum position size as fraction of portfolio
            
        Returns:
            dict: Position size recommendation
        """
        try:
            safe_kelly = kelly_metrics.get('safe_kelly_fraction', 0)
            
            # Calculate recommended position size
            recommended_size = safe_kelly * portfolio_value
            
            # Apply maximum position size limit
            max_position_value = portfolio_value * max_position_size
            final_position_value = min(recommended_size, max_position_value)
            
            # Calculate position size as percentage
            position_size_percent = (final_position_value / portfolio_value) * 100
            
            # Calculate risk metrics
            volatility = kelly_metrics.get('volatility_annual', 0.2)
            max_loss_percent = position_size_percent * volatility
            
            return {
                'symbol': symbol,
                'recommended_position_value': round(final_position_value, 2),
                'position_size_percent': round(position_size_percent, 2),
                'max_loss_percent': round(max_loss_percent, 2),
                'kelly_fraction_used': round(safe_kelly, 4),
                'max_position_limit': round(max_position_size * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return {
                'symbol': symbol,
                'recommended_position_value': 0.0,
                'position_size_percent': 0.0,
                'max_loss_percent': 0.0,
                'kelly_fraction_used': 0.0,
                'max_position_limit': max_position_size * 100
            }
    
    def _get_default_kelly_metrics(self) -> Dict[str, float]:
        """Get default Kelly metrics when insufficient data"""
        return {
            'kelly_fraction': 0.0,
            'safe_kelly_fraction': 0.0,
            'win_rate': 0.5,
            'avg_win': 0.02,
            'avg_loss': 0.02,
            'win_loss_ratio': 1.0,
            'expected_value': 0.0,
            'sharpe_ratio': 0.0,
            'volatility_adjustment': 1.0,
            'recommendation_strength': 0.0,
            'data_points': 0,
            'lookback_days': 0,
            'volatility_annual': 0.2,
            'max_return': 0.0,
            'min_return': 0.0,
            'total_return': 0.0,
            'positive_days': 0,
            'negative_days': 0,
            'neutral_days': 0
        }

# Global instance
kelly_engine = KellyCriterionEngine() 