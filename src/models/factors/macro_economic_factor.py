#!/usr/bin/env python3
"""
Macro Economic Factor - Global Economic Intelligence (8% weight)
===============================================================
Implements Interest Rate Environment, Economic Surprise Index,
Currency Strength Impact, and Inflation Expectations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.modular_signal_generator import BaseFactor, FactorOutput

logger = logging.getLogger(__name__)

class MacroEconomicFactor(BaseFactor):
    """Macro economic intelligence factor"""
    
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__(name, weight, enabled)
        
        # Sub-factor weights
        self.sub_weights = {
            'interest_rate_env': 0.375,      # 3% of 8%
            'economic_surprise': 0.25,       # 2% of 8%
            'currency_strength': 0.25,       # 2% of 8%
            'inflation_expectations': 0.125   # 1% of 8%
        }
        
        # Macro data (would be fetched from real sources in production)
        self.macro_data = {
            'repo_rate': 6.5,               # RBI repo rate
            'inflation_rate': 5.2,          # Current inflation
            'gdp_growth': 7.0,              # GDP growth rate
            'usd_inr': 83.0,                # USD/INR exchange rate
            'crude_oil': 85.0,              # Crude oil price
            '10y_yield': 7.2                # 10-year government bond yield
        }
        
        # Sector sensitivities to macro factors
        self.sector_sensitivities = {
            'Banking': {
                'interest_rate': 0.8,
                'currency': 0.3,
                'inflation': -0.2
            },
            'IT': {
                'interest_rate': -0.3,
                'currency': -0.7,  # Benefits from weak INR
                'inflation': -0.1
            },
            'Auto': {
                'interest_rate': -0.5,
                'currency': 0.4,
                'inflation': -0.3
            },
            'FMCG': {
                'interest_rate': -0.2,
                'currency': 0.2,
                'inflation': -0.4
            },
            'Energy': {
                'interest_rate': -0.2,
                'currency': 0.6,
                'inflation': 0.3
            }
        }
        
    def calculate(self, symbol: str, market_data: Dict) -> FactorOutput:
        """Calculate macro economic factor"""
        try:
            # Get sector information
            sector = self._get_sector(symbol, market_data)
            
            # Calculate sub-factors
            interest_rate_score = self._analyze_interest_rate_environment(sector)
            economic_surprise_score = self._calculate_economic_surprise_index()
            currency_score = self._analyze_currency_strength_impact(sector)
            inflation_score = self._analyze_inflation_expectations(sector)
            
            # Calculate weighted score
            sub_factors = {
                'interest_rate_env': interest_rate_score,
                'economic_surprise': economic_surprise_score,
                'currency_strength': currency_score,
                'inflation_expectations': inflation_score
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.sub_weights[factor] 
                for factor, score in sub_factors.items()
            )
            
            # Confidence based on sector clarity and data availability
            confidence = 0.7 if sector != 'Other' else 0.5
            
            # Determine macro environment
            macro_env = self._determine_macro_environment()
            
            return FactorOutput(
                score=total_score,
                confidence=confidence,
                details={
                    'sector': sector,
                    'macro_environment': macro_env,
                    'key_rates': {
                        'repo_rate': self.macro_data['repo_rate'],
                        '10y_yield': self.macro_data['10y_yield']
                    }
                },
                sub_factors=sub_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating macro economic factor for {symbol}: {e}")
            return FactorOutput(
                score=0.0,
                confidence=0.3,
                details={'error': str(e)},
                sub_factors={}
            )
            
    def _get_sector(self, symbol: str, market_data: Dict) -> str:
        """Determine stock sector"""
        # Simple sector mapping (would use real data in production)
        sector_map = {
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'AXISBANK': 'Banking',
            'KOTAKBANK': 'Banking',
            'TCS': 'IT',
            'INFY': 'IT',
            'WIPRO': 'IT',
            'HCLTECH': 'IT',
            'TECHM': 'IT',
            'MARUTI': 'Auto',
            'TATAMOTORS': 'Auto',
            'M&M': 'Auto',
            'HINDUNILVR': 'FMCG',
            'ITC': 'FMCG',
            'NESTLEIND': 'FMCG',
            'RELIANCE': 'Energy',
            'ONGC': 'Energy',
            'BPCL': 'Energy'
        }
        
        # Extract base symbol
        base_symbol = symbol.replace('.NSE', '')
        return sector_map.get(base_symbol, 'Other')
        
    def _analyze_interest_rate_environment(self, sector: str) -> float:
        """Analyze interest rate environment impact"""
        try:
            # Current vs historical rates
            current_rate = self.macro_data['repo_rate']
            historical_avg = 6.0  # Historical average repo rate
            rate_deviation = (current_rate - historical_avg) / historical_avg
            
            # Rate trend (simulated)
            rate_trend = 0.25 if current_rate > 6.25 else -0.25 if current_rate < 5.75 else 0
            
            # Base score from rate level and trend
            if current_rate < 5.5:  # Low rates
                base_score = 0.3  # Generally positive for growth
            elif current_rate < 6.5:  # Normal rates
                base_score = 0.0
            elif current_rate < 7.5:  # High rates
                base_score = -0.3
            else:  # Very high rates
                base_score = -0.6
                
            # Adjust for rate trend
            base_score += rate_trend * 0.3
            
            # Apply sector sensitivity
            sensitivity = self.sector_sensitivities.get(sector, {}).get('interest_rate', 0)
            sector_adjusted_score = base_score * (1 + sensitivity)
            
            return max(-1, min(1, sector_adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Error analyzing interest rate environment: {e}")
            return 0.0
            
    def _calculate_economic_surprise_index(self) -> float:
        """Calculate economic surprise index"""
        try:
            # Simulated economic surprises (would use real data)
            gdp_actual = self.macro_data['gdp_growth']
            gdp_expected = 6.5
            gdp_surprise = (gdp_actual - gdp_expected) / gdp_expected
            
            inflation_actual = self.macro_data['inflation_rate']
            inflation_expected = 5.5
            inflation_surprise = -(inflation_actual - inflation_expected) / inflation_expected  # Lower is better
            
            # Composite surprise index
            surprise_index = gdp_surprise * 0.6 + inflation_surprise * 0.4
            
            # Convert to score
            if surprise_index > 0.1:  # Positive surprises
                score = 0.5
            elif surprise_index > 0:
                score = 0.2
            elif surprise_index > -0.1:
                score = -0.2
            else:  # Negative surprises
                score = -0.5
                
            return max(-1, min(1, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating economic surprise index: {e}")
            return 0.0
            
    def _analyze_currency_strength_impact(self, sector: str) -> float:
        """Analyze currency strength impact"""
        try:
            # USD/INR analysis
            current_usdinr = self.macro_data['usd_inr']
            historical_avg = 80.0
            currency_deviation = (current_usdinr - historical_avg) / historical_avg
            
            # Base score from currency level
            if currency_deviation > 0.05:  # Weak INR
                base_score = -0.2  # Generally negative for imports
            elif currency_deviation < -0.05:  # Strong INR
                base_score = 0.2  # Generally positive for imports
            else:
                base_score = 0.0
                
            # Oil price impact (affects currency)
            oil_price = self.macro_data['crude_oil']
            if oil_price > 90:
                base_score -= 0.1  # High oil prices pressure INR
            elif oil_price < 70:
                base_score += 0.1  # Low oil prices support INR
                
            # Apply sector sensitivity
            sensitivity = self.sector_sensitivities.get(sector, {}).get('currency', 0)
            sector_adjusted_score = base_score * (1 + abs(sensitivity)) * np.sign(sensitivity)
            
            return max(-1, min(1, sector_adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency strength: {e}")
            return 0.0
            
    def _analyze_inflation_expectations(self, sector: str) -> float:
        """Analyze inflation expectations impact"""
        try:
            # Current inflation analysis
            current_inflation = self.macro_data['inflation_rate']
            target_inflation = 4.0  # RBI target
            
            # Inflation deviation
            inflation_gap = current_inflation - target_inflation
            
            # Base score from inflation level
            if inflation_gap < -1:  # Very low inflation
                base_score = -0.3  # Deflation concerns
            elif inflation_gap < 0:  # Below target
                base_score = 0.2  # Positive for growth
            elif inflation_gap < 2:  # Slightly above target
                base_score = -0.1
            else:  # High inflation
                base_score = -0.5
                
            # Bond yield spread as inflation expectation proxy
            yield_spread = self.macro_data['10y_yield'] - self.macro_data['repo_rate']
            if yield_spread > 1.0:  # High inflation expectations
                base_score -= 0.2
            elif yield_spread < 0.5:  # Low inflation expectations
                base_score += 0.1
                
            # Apply sector sensitivity
            sensitivity = self.sector_sensitivities.get(sector, {}).get('inflation', 0)
            sector_adjusted_score = base_score * (1 + abs(sensitivity))
            
            return max(-1, min(1, sector_adjusted_score))
            
        except Exception as e:
            self.logger.error(f"Error analyzing inflation expectations: {e}")
            return 0.0
            
    def _determine_macro_environment(self) -> str:
        """Determine overall macro economic environment"""
        gdp = self.macro_data['gdp_growth']
        inflation = self.macro_data['inflation_rate']
        rates = self.macro_data['repo_rate']
        
        if gdp > 7 and inflation < 5:
            return "GOLDILOCKS"  # High growth, low inflation
        elif gdp > 6 and inflation < 6:
            return "FAVORABLE"
        elif gdp < 5 and inflation > 6:
            return "STAGFLATION"  # Low growth, high inflation
        elif gdp < 5:
            return "SLOWDOWN"
        elif inflation > 7:
            return "OVERHEATING"
        else:
            return "NEUTRAL" 