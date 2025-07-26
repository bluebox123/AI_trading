#!/usr/bin/env python3
"""
Advanced Temporal Weighting System
=================================

This module provides advanced temporal weighting features including:
- Windowed boosting for recent news
- Custom decay profiles per sector/stock
- Multiple decay curve types (linear, exponential, power law)
- Adaptive weighting based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, date
import logging
from pathlib import Path
import json
from enum import Enum

class DecayType(Enum):
    """Types of decay curves"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    POWER_LAW = "power_law"
    LOGARITHMIC = "logarithmic"
    CUSTOM = "custom"

@dataclass
class DecayProfile:
    """Data class for decay profile configuration"""
    decay_type: DecayType
    half_life_days: float
    window_boost_multiplier: float
    window_boost_hours: int
    sector_specific: bool
    stock_specific: bool
    adaptive_parameters: Dict[str, float]

@dataclass
class WeightedSentiment:
    """Data class for weighted sentiment result"""
    original_sentiment: float
    temporal_weight: float
    window_boost: float
    final_weight: float
    weighted_sentiment: float
    decay_profile: str
    confidence_impact: float

class AdvancedTemporalWeighter:
    """Advanced temporal weighting system with windowed boosting and custom profiles"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the advanced temporal weighter"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.decay_profiles = self._load_decay_profiles()
        self.market_conditions = {}
        
        self.logger.info("Advanced Temporal Weighter initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the weighter"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for temporal weighting"""
        default_config = {
            "windowed_boosting": {
                "enabled": True,
                "default_multiplier": 1.2,
                "max_multiplier": 2.0,
                "boost_hours": 1,
                "adaptive_boosting": True
            },
            "decay_profiles": {
                "default": {
                    "type": "exponential",
                    "half_life_days": 7.0,
                    "window_boost_multiplier": 1.2,
                    "window_boost_hours": 1
                },
                "high_volatility": {
                    "type": "exponential",
                    "half_life_days": 3.0,
                    "window_boost_multiplier": 1.5,
                    "window_boost_hours": 2
                },
                "low_volatility": {
                    "type": "exponential",
                    "half_life_days": 14.0,
                    "window_boost_multiplier": 1.1,
                    "window_boost_hours": 1
                }
            },
            "sector_profiles": {
                "technology": {
                    "type": "exponential",
                    "half_life_days": 5.0,
                    "window_boost_multiplier": 1.3,
                    "window_boost_hours": 2
                },
                "finance": {
                    "type": "exponential",
                    "half_life_days": 3.0,
                    "window_boost_multiplier": 1.4,
                    "window_boost_hours": 1
                },
                "healthcare": {
                    "type": "exponential",
                    "half_life_days": 10.0,
                    "window_boost_multiplier": 1.1,
                    "window_boost_hours": 1
                }
            },
            "adaptive_weighting": {
                "enabled": True,
                "market_volatility_threshold": 0.3,
                "news_volume_threshold": 50,
                "sentiment_extremity_threshold": 0.8
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _load_decay_profiles(self) -> Dict[str, DecayProfile]:
        """Load decay profiles from configuration"""
        profiles = {}
        
        # Load default profiles
        for name, config in self.config["decay_profiles"].items():
            profiles[name] = DecayProfile(
                decay_type=DecayType(config["type"]),
                half_life_days=config["half_life_days"],
                window_boost_multiplier=config["window_boost_multiplier"],
                window_boost_hours=config["window_boost_hours"],
                sector_specific=False,
                stock_specific=False,
                adaptive_parameters={}
            )
        
        # Load sector profiles
        for sector, config in self.config["sector_profiles"].items():
            profiles[f"sector_{sector}"] = DecayProfile(
                decay_type=DecayType(config["type"]),
                half_life_days=config["half_life_days"],
                window_boost_multiplier=config["window_boost_multiplier"],
                window_boost_hours=config["window_boost_hours"],
                sector_specific=True,
                stock_specific=False,
                adaptive_parameters={}
            )
        
        return profiles
    
    def calculate_base_decay(self, days_diff: float, profile: DecayProfile) -> float:
        """Calculate base decay weight using different curve types"""
        
        if profile.decay_type == DecayType.EXPONENTIAL:
            return self._exponential_decay(days_diff, profile.half_life_days)
        elif profile.decay_type == DecayType.LINEAR:
            return self._linear_decay(days_diff, profile.half_life_days)
        elif profile.decay_type == DecayType.POWER_LAW:
            return self._power_law_decay(days_diff, profile.half_life_days)
        elif profile.decay_type == DecayType.LOGARITHMIC:
            return self._logarithmic_decay(days_diff, profile.half_life_days)
        else:
            return self._exponential_decay(days_diff, profile.half_life_days)
    
    def _exponential_decay(self, days_diff: float, half_life_days: float) -> float:
        """Exponential decay: weight = exp(-ln(2) * days_diff / half_life)"""
        decay_rate = np.log(2) / half_life_days
        weight = np.exp(-decay_rate * days_diff)
        return max(0.1, min(1.0, weight))
    
    def _linear_decay(self, days_diff: float, half_life_days: float) -> float:
        """Linear decay: weight = 1 - (days_diff / (2 * half_life))"""
        weight = 1 - (days_diff / (2 * half_life_days))
        return max(0.1, min(1.0, weight))
    
    def _power_law_decay(self, days_diff: float, half_life_days: float) -> float:
        """Power law decay: weight = (1 + days_diff / half_life) ^ (-alpha)"""
        alpha = 1.5  # Power law exponent
        weight = (1 + days_diff / half_life_days) ** (-alpha)
        return max(0.1, min(1.0, weight))
    
    def _logarithmic_decay(self, days_diff: float, half_life_days: float) -> float:
        """Logarithmic decay: weight = 1 / (1 + log(1 + days_diff / half_life))"""
        weight = 1 / (1 + np.log(1 + days_diff / half_life_days))
        return max(0.1, min(1.0, weight))
    
    def calculate_window_boost(self, hours_diff: float, profile: DecayProfile) -> float:
        """Calculate windowed boosting for recent news"""
        
        if not self.config["windowed_boosting"]["enabled"]:
            return 1.0
        
        if hours_diff <= profile.window_boost_hours:
            # Apply boosting for recent news
            boost_multiplier = profile.window_boost_multiplier
            
            # Adaptive boosting based on market conditions
            if self.config["windowed_boosting"]["adaptive_boosting"]:
                boost_multiplier = self._calculate_adaptive_boost(boost_multiplier)
            
            # Decay boost over time within window
            boost_decay = 1 - (hours_diff / profile.window_boost_hours)
            final_boost = 1 + (boost_multiplier - 1) * boost_decay
            
            return min(final_boost, self.config["windowed_boosting"]["max_multiplier"])
        
        return 1.0
    
    def _calculate_adaptive_boost(self, base_boost: float) -> float:
        """Calculate adaptive boost based on market conditions"""
        
        # Get current market conditions
        volatility = self.market_conditions.get('volatility', 0.5)
        news_volume = self.market_conditions.get('news_volume', 0.5)
        sentiment_extremity = self.market_conditions.get('sentiment_extremity', 0.5)
        
        # Adjust boost based on conditions
        volatility_factor = 1 + 0.3 * volatility  # Higher volatility = higher boost
        volume_factor = 1 + 0.2 * news_volume     # Higher volume = higher boost
        extremity_factor = 1 + 0.2 * sentiment_extremity  # Extreme sentiment = higher boost
        
        adaptive_boost = base_boost * volatility_factor * volume_factor * extremity_factor
        
        return min(adaptive_boost, self.config["windowed_boosting"]["max_multiplier"])
    
    def get_decay_profile(self, stock_symbol: str = None, sector: str = None) -> DecayProfile:
        """Get appropriate decay profile for stock/sector"""
        
        # Check for stock-specific profile
        if stock_symbol and f"stock_{stock_symbol}" in self.decay_profiles:
            return self.decay_profiles[f"stock_{stock_symbol}"]
        
        # Check for sector-specific profile
        if sector and f"sector_{sector}" in self.decay_profiles:
            return self.decay_profiles[f"sector_{sector}"]
        
        # Use default profile
        return self.decay_profiles["default"]
    
    def calculate_temporal_weight(self, 
                                article_date: datetime,
                                target_date: datetime,
                                stock_symbol: str = None,
                                sector: str = None,
                                confidence: float = 0.5) -> WeightedSentiment:
        """Calculate comprehensive temporal weight with windowed boosting"""
        
        # Get appropriate decay profile
        profile = self.get_decay_profile(stock_symbol, sector)
        
        # Calculate time differences
        days_diff = abs((article_date.date() - target_date.date()).days)
        hours_diff = abs((article_date - target_date).total_seconds() / 3600)
        
        # Calculate base decay weight
        base_weight = self.calculate_base_decay(days_diff, profile)
        
        # Calculate window boost
        window_boost = self.calculate_window_boost(hours_diff, profile)
        
        # Calculate final weight
        final_weight = base_weight * window_boost
        
        # Confidence impact (higher confidence = higher weight)
        confidence_impact = 0.5 + 0.5 * confidence
        final_weight *= confidence_impact
        
        # Ensure weight is within bounds
        final_weight = max(0.1, min(1.0, final_weight))
        
        return WeightedSentiment(
            original_sentiment=0.0,  # Will be set by caller
            temporal_weight=base_weight,
            window_boost=window_boost,
            final_weight=final_weight,
            weighted_sentiment=0.0,  # Will be set by caller
            decay_profile=profile.decay_type.value,
            confidence_impact=confidence_impact
        )
    
    def update_market_conditions(self, conditions: Dict[str, float]):
        """Update market conditions for adaptive weighting"""
        self.market_conditions.update(conditions)
        self.logger.info(f"Updated market conditions: {conditions}")
    
    def create_custom_decay_profile(self, 
                                  name: str,
                                  decay_type: DecayType,
                                  half_life_days: float,
                                  window_boost_multiplier: float = 1.2,
                                  window_boost_hours: int = 1,
                                  sector_specific: bool = False,
                                  stock_specific: bool = False) -> DecayProfile:
        """Create a custom decay profile"""
        
        profile = DecayProfile(
            decay_type=decay_type,
            half_life_days=half_life_days,
            window_boost_multiplier=window_boost_multiplier,
            window_boost_hours=window_boost_hours,
            sector_specific=sector_specific,
            stock_specific=stock_specific,
            adaptive_parameters={}
        )
        
        self.decay_profiles[name] = profile
        self.logger.info(f"Created custom decay profile: {name}")
        
        return profile
    
    def compare_decay_curves(self, days_range: int = 30) -> Dict[str, List[float]]:
        """Compare different decay curves for visualization"""
        
        days = np.arange(0, days_range + 1)
        curves = {}
        
        for name, profile in self.decay_profiles.items():
            weights = [self.calculate_base_decay(day, profile) for day in days]
            curves[name] = weights
        
        return curves
    
    def get_decay_statistics(self, profile_name: str = "default") -> Dict[str, float]:
        """Get statistics for a decay profile"""
        
        if profile_name not in self.decay_profiles:
            return {}
        
        profile = self.decay_profiles[profile_name]
        
        # Calculate weights for different time periods
        weights_1d = self.calculate_base_decay(1, profile)
        weights_7d = self.calculate_base_decay(7, profile)
        weights_14d = self.calculate_base_decay(14, profile)
        weights_30d = self.calculate_base_decay(30, profile)
        
        return {
            'profile_name': profile_name,
            'decay_type': profile.decay_type.value,
            'half_life_days': profile.half_life_days,
            'window_boost_multiplier': profile.window_boost_multiplier,
            'window_boost_hours': profile.window_boost_hours,
            'weight_1d': weights_1d,
            'weight_7d': weights_7d,
            'weight_14d': weights_14d,
            'weight_30d': weights_30d,
            'decay_rate_1d_to_7d': (weights_1d - weights_7d) / weights_1d,
            'decay_rate_7d_to_14d': (weights_7d - weights_14d) / weights_7d
        }

def main():
    """Test the advanced temporal weighting system"""
    weighter = AdvancedTemporalWeighter()
    
    # Test different decay profiles
    print("Testing Advanced Temporal Weighting System")
    print("=" * 50)
    
    # Test windowed boosting
    article_date = datetime.now() - timedelta(hours=0.5)  # 30 minutes ago
    target_date = datetime.now()
    
    print(f"\nTesting windowed boosting for article from {article_date}")
    
    # Test different profiles
    profiles_to_test = ["default", "high_volatility", "low_volatility"]
    
    for profile_name in profiles_to_test:
        if profile_name in weighter.decay_profiles:
            profile = weighter.decay_profiles[profile_name]
            result = weighter.calculate_temporal_weight(
                article_date, target_date, 
                stock_symbol="RELIANCE.NSE",
                confidence=0.8
            )
            
            print(f"\n{profile_name} profile:")
            print(f"  Base weight: {result.temporal_weight:.3f}")
            print(f"  Window boost: {result.window_boost:.3f}")
            print(f"  Final weight: {result.final_weight:.3f}")
            print(f"  Confidence impact: {result.confidence_impact:.3f}")
    
    # Compare decay curves
    print(f"\nComparing decay curves:")
    curves = weighter.compare_decay_curves(7)  # 7 days
    
    for name, weights in curves.items():
        print(f"  {name}: 1d={weights[1]:.3f}, 3d={weights[3]:.3f}, 7d={weights[7]:.3f}")
    
    # Test custom profile
    print(f"\nCreating custom profile...")
    custom_profile = weighter.create_custom_decay_profile(
        name="custom_fast",
        decay_type=DecayType.EXPONENTIAL,
        half_life_days=2.0,
        window_boost_multiplier=1.5,
        window_boost_hours=2
    )
    
    stats = weighter.get_decay_statistics("custom_fast")
    print(f"Custom profile statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 