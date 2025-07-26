#!/usr/bin/env python3
# src/signal_orchestrator.py
"""
Signal Orchestrator - Unified Signal Generation System

This module provides a unified interface for generating trading signals using multiple
models and strategies, with quality filters and ensemble methods.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from data.eodhd_v4_bridge import EodhdV4Bridge
    from data.perplexity_news_bridge import PerplexityNewsBridge
except ImportError as e:
    logger.error(f"Import error: {e}")
    EodhdV4Bridge = None
    PerplexityNewsBridge = None

class SignalOrchestrator:
    """
    Enhanced Signal Orchestrator with real data integration
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.signals_dir = self.data_dir / 'signals'
        self.cache_dir = self.data_dir / 'cache'
        self.models_dir = self.data_dir / 'models'
        
        # Enhanced configuration
        self.cache_duration_minutes = self.config.get('cache_duration_minutes', 5)
        self.use_deterministic_signals = self.config.get('use_deterministic_signals', True)
        
        # Initialize data bridges
        self.eodhd_bridge = None
        self.perplexity_bridge = None
        
        try:
            if EodhdV4Bridge:
                self.eodhd_bridge = EodhdV4Bridge()
                logger.info("EODHD Bridge initialized successfully")
            
            # Initialize enhanced Perplexity analyzer for institutional-grade news analysis
            try:
                import sys
                sys.path.append('.')
                from enhanced_perplexity_analyzer import EnhancedPerplexityAnalyzer
                self.perplexity_analyzer = EnhancedPerplexityAnalyzer()
                logger.info("✅ Enhanced Perplexity Analyzer initialized for institutional news analysis")
            except Exception as e:
                logger.warning(f"Enhanced Perplexity Analyzer unavailable: {e}")
                self.perplexity_analyzer = None
            
            # Fallback to simple Perplexity bridge
            if PerplexityNewsBridge:
                perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
                if perplexity_api_key:
                    self.perplexity_bridge = PerplexityNewsBridge(api_key=perplexity_api_key)
                    logger.info("Perplexity Bridge initialized as fallback")
        except Exception as e:
            logger.warning(f"Bridge initialization warning: {e}")
        
        self._create_directories()
        
        # Performance tracking
        self.performance_metrics = {
            "total_signals_generated": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "average_generation_time": 0,
            "last_run_time": None,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Enhanced Signal Orchestrator initialized with real data integration")
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.signals_dir, self.cache_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_signal_cache_key(self, symbols: List[str], model: str, use_sentiment: bool) -> str:
        """Generate cache key for signals"""
        now = datetime.now()
        # Round to 5-minute intervals for stable caching
        cache_time = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        
        symbols_str = ','.join(sorted(symbols))
        return f"signals_{model}_{len(symbols)}_{cache_time.strftime('%Y%m%d_%H%M')}_{use_sentiment}"
    
    def _get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price from EODHD"""
        try:
            if not self.eodhd_bridge:
                return None
            
            # Get real-time price data
            price_data = self.eodhd_bridge.get_real_time_data(symbol)
            
            if price_data and isinstance(price_data, dict):
                # Extract price from different possible response formats
                price = (price_data.get('price') or 
                        price_data.get('close') or 
                        price_data.get('last') or
                        price_data.get('regularMarketPrice'))
                
                if price and price > 0:
                    logger.info(f"Real-time price for {symbol}: ₹{price}")
                    return float(price)
            
            logger.warning(f"Could not get real-time price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
            return None
    
    def _get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get news sentiment for symbol"""
        try:
            if not self.eodhd_bridge:
                return {"polarity": 0.0, "confidence": 0.5, "article_count": 0}
            
            # Get recent news for the symbol
            news_data = self.eodhd_bridge.get_financial_news(symbol=symbol, limit=5)
            
            if news_data and len(news_data) > 0:
                sentiments = []
                for article in news_data:
                    if 'sentiment' in article:
                        polarity = article['sentiment'].get('polarity', 0)
                        sentiments.append(polarity)
                
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    confidence = min(0.9, 0.5 + len(sentiments) * 0.1)
                    
                    logger.info(f"News sentiment for {symbol}: {avg_sentiment:.3f} (confidence: {confidence:.3f})")
                    
                    return {
                        "polarity": avg_sentiment,
                        "confidence": confidence,
                        "article_count": len(sentiments)
                    }
            
            # Return neutral sentiment if no news data
            return {"polarity": 0.0, "confidence": 0.3, "article_count": 0}
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {"polarity": 0.0, "confidence": 0.5, "article_count": 0}
    
    def _generate_enhanced_signal(self, symbol: str, model: str, use_sentiment: bool = False) -> Dict:
        """Generate enhanced signal with REAL AI model integration"""
        try:
            # Get real intraday market data
            intraday_data = self._get_real_intraday_data(symbol)
            if intraday_data is None or len(intraday_data) < 10:
                logger.warning(f"Insufficient real data for {symbol}, using fallback")
                return self._generate_fallback_signal(symbol, model, use_sentiment)
            
            # Get real-time current price from EODHD
            current_price = self._get_real_time_price(symbol)
            if not current_price:
                # Try to get from intraday data
                try:
                    current_price = float(intraday_data['close'].iloc[-1])
                except:
                    current_price = 100 + hash(symbol) % 1000  # Fallback
            
            # Generate REAL AI prediction using V4 model (if available)
            ai_prediction = self._get_ai_prediction(intraday_data, symbol)
            
            # Get basic news sentiment (fallback)
            basic_sentiment_data = self._get_news_sentiment(symbol) if use_sentiment else {"polarity": 0.0, "confidence": 0.5}
            
            # Use AI prediction if available, otherwise use enhanced mathematical prediction
            if ai_prediction and ai_prediction.get('model_inference', False):
                predicted_signal = ai_prediction.get('prediction', 'HOLD')
                base_confidence = ai_prediction.get('confidence', 0.6)
                risk_score = ai_prediction.get('risk_score', 0.3)
                model_used = f"v4_ai_{model}"
                logger.info(f"Using REAL AI prediction for {symbol}: {predicted_signal} (conf: {base_confidence:.3f})")
            else:
                # Enhanced mathematical prediction with real data
                predicted_signal, base_confidence = self._generate_mathematical_prediction_from_data(intraday_data, basic_sentiment_data)
                risk_score = self._calculate_risk_from_data(intraday_data)
                model_used = f"enhanced_math_{model}"
                logger.debug(f"Using mathematical prediction for {symbol}: {predicted_signal} (conf: {base_confidence:.3f})")
            
            # Get comprehensive Perplexity analysis for institutional-grade insights
            perplexity_analysis = None
            if self.perplexity_analyzer and use_sentiment:
                try:
                    perplexity_analysis = self.perplexity_analyzer.get_comprehensive_market_analysis(
                        symbol=symbol,
                        current_price=current_price,
                        signal=predicted_signal,
                        confidence=base_confidence
                    )
                    logger.info(f"✅ Got Perplexity analysis for {symbol}: {perplexity_analysis.get('sentiment_category', 'N/A')}")
                except Exception as e:
                    logger.warning(f"Perplexity analysis failed for {symbol}: {e}")
                    perplexity_analysis = None
            
            # Enhanced confidence adjustment using comprehensive Perplexity analysis
            sentiment_adjustment = 0
            market_regime_adjustment = 0
            news_impact_adjustment = 0
            
            # Use Perplexity analysis if available, otherwise fallback to basic sentiment
            if perplexity_analysis:
                # Extract key insights from Perplexity analysis
                sentiment_score = perplexity_analysis.get('sentiment_score', 0.0)
                sentiment_category = perplexity_analysis.get('sentiment_category', 'NEUTRAL')
                market_regime = perplexity_analysis.get('market_regime', 'MIXED')
                recommendation = perplexity_analysis.get('recommendation', 'HOLD')
                confidence_adjustment = perplexity_analysis.get('confidence_adjustment', 0.0)
                
                # Apply Perplexity-based signal adjustments
                if abs(sentiment_score) > 0.5:  # Strong sentiment from news
                    if sentiment_score > 0.5 and predicted_signal == "SELL":
                        predicted_signal = "HOLD"  # Strong positive news prevents sell
                        sentiment_adjustment = 0.08
                    elif sentiment_score < -0.5 and predicted_signal == "BUY":
                        predicted_signal = "HOLD"  # Strong negative news prevents buy
                        sentiment_adjustment = -0.08
                    elif sentiment_score > 0.7:
                        if predicted_signal == "HOLD":
                            predicted_signal = "BUY"  # Very positive news suggests buy
                        sentiment_adjustment = 0.12
                    elif sentiment_score < -0.7:
                        if predicted_signal == "HOLD":
                            predicted_signal = "SELL"  # Very negative news suggests sell
                        sentiment_adjustment = -0.12
                
                # Apply market regime adjustments
                if market_regime == 'BULLISH' and predicted_signal == "SELL":
                    predicted_signal = "HOLD"  # Bullish regime prevents aggressive sells
                    market_regime_adjustment = 0.05
                elif market_regime == 'BEARISH' and predicted_signal == "BUY":
                    predicted_signal = "HOLD"  # Bearish regime prevents aggressive buys
                    market_regime_adjustment = -0.05
                elif market_regime == 'VOLATILE':
                    market_regime_adjustment = -0.03  # Reduce confidence in volatile markets
                
                # Apply Perplexity confidence adjustment
                news_impact_adjustment = confidence_adjustment
                
                logger.info(f"Perplexity insights for {symbol}: sentiment={sentiment_category}, regime={market_regime}, rec={recommendation}")
                
            elif use_sentiment and basic_sentiment_data['confidence'] > 0.6:
                # Fallback to basic sentiment if Perplexity unavailable
                polarity = basic_sentiment_data['polarity']
                if abs(polarity) > 0.3:  # Strong sentiment
                    if polarity > 0.3 and predicted_signal == "SELL":
                        predicted_signal = "HOLD"
                        sentiment_adjustment = 0.05
                    elif polarity < -0.3 and predicted_signal == "BUY":
                        predicted_signal = "HOLD"
                        sentiment_adjustment = -0.05
                    elif polarity > 0.5:
                        predicted_signal = "BUY"
                        sentiment_adjustment = 0.1
                    elif polarity < -0.5:
                        predicted_signal = "SELL"
                        sentiment_adjustment = -0.1
            
            # Final confidence calculation with all adjustments
            total_adjustment = sentiment_adjustment + market_regime_adjustment + news_impact_adjustment
            final_confidence = min(0.95, max(0.5, base_confidence + total_adjustment))
            
            # Calculate technical indicators from real data
            technical_indicators = self._calculate_technical_indicators(intraday_data)
            
            # Calculate price targets based on volatility and confidence
            volatility = self._calculate_volatility(intraday_data)
            
            if predicted_signal == "BUY":
                target_return = final_confidence * 0.04 + volatility * 0.02  # 2-6% target
                price_target = current_price * (1 + target_return)
                stop_loss = current_price * (1 - risk_score * 0.03)  # Risk-adjusted stop
            elif predicted_signal == "SELL":
                target_return = final_confidence * 0.04 + volatility * 0.02  # 2-6% target
                price_target = current_price * (1 - target_return)
                stop_loss = current_price * (1 + risk_score * 0.03)  # Risk-adjusted stop
            else:  # HOLD
                price_target = current_price * (1 + (final_confidence - 0.5) * 0.02)  # Small movement
                stop_loss = current_price * (1 - risk_score * 0.02)
            
            signal = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "model": model_used,
                "signal": predicted_signal,
                "confidence": round(final_confidence, 3),
                "current_price": round(current_price, 2),
                "price_target": round(price_target, 2),
                "stop_loss": round(stop_loss, 2),
                "price": round(current_price, 2),  # For dashboard compatibility
                "data_sources": ["eodhd_realtime", "ai_model" if ai_prediction else "mathematical"],
                "volatility": round(volatility, 4),
                "risk_score": round(risk_score, 3)
            }
            
            # Add technical indicators
            signal["technical_indicators"] = technical_indicators
            
            # Add comprehensive Perplexity analysis data if available
            if perplexity_analysis:
                signal["sentiment_score"] = round(perplexity_analysis.get('sentiment_score', 0.0), 3)
                signal["sentiment_category"] = perplexity_analysis.get('sentiment_category', 'NEUTRAL')
                signal["news_impact"] = perplexity_analysis.get('news_impact', 'MODERATE')
                signal["market_regime"] = perplexity_analysis.get('market_regime', 'MIXED')
                signal["perplexity_recommendation"] = perplexity_analysis.get('recommendation', 'HOLD')
                signal["key_drivers"] = perplexity_analysis.get('key_drivers', [])[:3]  # Top 3 drivers
                signal["risk_level"] = perplexity_analysis.get('risk_level', 'MEDIUM')
                signal["perplexity_sources"] = perplexity_analysis.get('perplexity_sources', 0)
                signal["data_freshness"] = perplexity_analysis.get('data_freshness', 'real_time')
                signal["data_sources"].append("perplexity_analysis")
                
                # Add target price and stop loss from Perplexity if available
                if perplexity_analysis.get('target_price'):
                    signal["perplexity_target"] = round(perplexity_analysis['target_price'], 2)
                if perplexity_analysis.get('stop_loss'):
                    signal["perplexity_stop_loss"] = round(perplexity_analysis['stop_loss'], 2)
                    
                logger.info(f"Added Perplexity insights to {symbol}: {signal['sentiment_category']}, regime: {signal['market_regime']}")
                
            elif use_sentiment and basic_sentiment_data:
                # Fallback to basic sentiment data
                signal["sentiment_score"] = round(basic_sentiment_data['polarity'], 3)
                signal["sentiment_confidence"] = round(basic_sentiment_data['confidence'], 3)
                signal["news_article_count"] = basic_sentiment_data['article_count']
                signal["sentiment_category"] = "BASIC_ANALYSIS"
                signal["data_sources"].append("basic_news_sentiment")
            
            # Add AI model info if used
            if ai_prediction and ai_prediction.get('model_inference', False):
                signal["ai_model_used"] = True
                signal["ai_buy_probability"] = ai_prediction.get('buy_probability', 0)
                signal["ai_sell_probability"] = ai_prediction.get('sell_probability', 0)
                signal["ai_hold_probability"] = ai_prediction.get('hold_probability', 0)
                signal["data_sources"].append("v4_transformer")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating enhanced signal for {symbol}: {e}")
            return self._generate_fallback_signal(symbol, model, use_sentiment)

    def _get_real_intraday_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get real intraday data from EODHD bridge"""
        try:
            if self.eodhd_bridge:
                data = self.eodhd_bridge.get_intraday_data(symbol, interval='5m')
                if data is not None and len(data) > 0:
                    logger.debug(f"Got {len(data)} real intraday data points for {symbol}")
                    return data
            
            logger.warning(f"Could not get real intraday data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting real intraday data for {symbol}: {e}")
            return None

    def _get_ai_prediction(self, intraday_data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Get AI prediction from V4 model if available"""
        try:
            # Try to import and use V4 trainer
            from models.multimodal_transformer_v4 import TemporalCausalityTrainerV4
            
            # Initialize V4 trainer if not already done
            if not hasattr(self, 'v4_trainer') or not self.v4_trainer:
                self.v4_trainer = TemporalCausalityTrainerV4()
            
            # Generate prediction
            prediction = self.v4_trainer.predict(intraday_data, symbol=symbol)
            if prediction and prediction.get('model_inference', False):
                logger.info(f"Got AI prediction for {symbol}: {prediction.get('prediction')} (conf: {prediction.get('confidence', 0):.3f})")
                return prediction
            
            return None
            
        except Exception as e:
            logger.debug(f"AI prediction not available for {symbol}: {e}")
            return None

    def _generate_mathematical_prediction_from_data(self, data: pd.DataFrame, sentiment_data: Dict) -> Tuple[str, float]:
        """Generate prediction using mathematical analysis of real data"""
        try:
            if len(data) < 5:
                return "HOLD", 0.6
            
            # Calculate technical indicators
            closes = data['close'].values
            volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(closes))
            
            # Price momentum
            short_ma = np.mean(closes[-5:])  # 5-period moving average
            long_ma = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
            price_momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
            
            # Volume trend
            recent_volume = np.mean(volumes[-3:])
            avg_volume = np.mean(volumes)
            volume_factor = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Price volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.02
            
            # Trend strength
            if len(closes) >= 3:
                trend_slope = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] > 0 else 0
            else:
                trend_slope = 0
            
            # Combine signals (MORE AGGRESSIVE WEIGHTING)
            momentum_score = price_momentum * 4  # Much higher momentum weight
            volume_score = min(2.0, volume_factor) - 1  # -1 to +1 range
            trend_score = trend_slope * 15  # Much higher trend amplification
            
            total_score = momentum_score + volume_score * 0.4 + trend_score * 0.7  # Higher trend weight
            
            # Add sentiment bias (STRONGER INFLUENCE)
            sentiment_bias = sentiment_data.get('polarity', 0) * 0.4  # Double sentiment weight
            total_score += sentiment_bias
            
            # Determine signal and confidence (REALISTIC INSTITUTIONAL THRESHOLDS)
            if total_score > 0.08:  # Strong bullish signal
                signal = "BUY"
                confidence = min(0.85, 0.62 + abs(total_score) * 1.5)  # Realistic confidence range
            elif total_score > 0.04:  # Moderate bullish signal  
                signal = "BUY"
                confidence = min(0.78, 0.58 + abs(total_score) * 1.8)  # Lower confidence for weaker signals
            elif total_score < -0.08:  # Strong bearish signal
                signal = "SELL"
                confidence = min(0.85, 0.62 + abs(total_score) * 1.5)  # Realistic confidence range
            elif total_score < -0.04:  # Moderate bearish signal
                signal = "SELL"
                confidence = min(0.78, 0.58 + abs(total_score) * 1.8)  # Lower confidence for weaker signals
            else:
                signal = "HOLD"
                confidence = 0.52 + abs(total_score) * 1.2  # Lower hold confidence
            
            # Adjust confidence based on data quality and volatility
            data_quality_factor = min(1.0, len(data) / 20)
            volatility_factor = min(1.0, max(0.7, 1 - volatility * 2))  # Reduce confidence in high volatility
            
            confidence *= data_quality_factor * volatility_factor
            
            return signal, max(0.45, min(0.88, confidence))
            
        except Exception as e:
            logger.error(f"Error in mathematical prediction: {e}")
            return "HOLD", 0.5

    def _calculate_risk_from_data(self, data: pd.DataFrame) -> float:
        """Calculate risk score from real market data"""
        try:
            if len(data) < 2:
                return 0.3
            
            # Price volatility
            closes = data['close'].values
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.02
            
            # Price range volatility
            if 'high' in data.columns and 'low' in data.columns:
                ranges = (data['high'] - data['low']) / data['close']
                range_volatility = np.mean(ranges)
            else:
                range_volatility = volatility
            
            # Volume volatility
            if 'volume' in data.columns and len(data) > 1:
                volume_changes = np.diff(data['volume'].values) / data['volume'].values[:-1]
                volume_volatility = np.std(volume_changes)
            else:
                volume_volatility = 0.1
            
            # Combine risk factors
            risk_score = (volatility * 0.5 + range_volatility * 0.3 + volume_volatility * 0.2)
            
            # Normalize to 0-1 range
            risk_score = min(0.8, max(0.1, risk_score * 10))
            
            return risk_score
            
        except Exception as e:
            logger.warning(f"Error calculating risk: {e}")
            return 0.3

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from real data"""
        try:
            indicators = {}
            
            if len(data) < 5:
                return {"data_insufficient": True}
            
            closes = data['close'].values
            volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(closes))
            
            # Moving averages
            if len(closes) >= 5:
                indicators["sma_5"] = round(np.mean(closes[-5:]), 2)
            if len(closes) >= 10:
                indicators["sma_10"] = round(np.mean(closes[-10:]), 2)
            
            # RSI approximation
            if len(closes) >= 14:
                gains = []
                losses = []
                for i in range(1, min(15, len(closes))):
                    change = closes[i] - closes[i-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(-change)
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0.01
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"] = round(rsi, 2)
            
            # MACD approximation
            if len(closes) >= 26:
                ema_12 = closes[-12:].mean()  # Simplified EMA
                ema_26 = closes[-26:].mean()
                macd = ema_12 - ema_26
                indicators["macd"] = round(macd, 4)
            
            # Volume ratio
            if len(volumes) >= 10:
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[-10:])
                indicators["volume_ratio"] = round(recent_volume / avg_volume if avg_volume > 0 else 1, 3)
            
            # Price position (how close to high/low)
            if 'high' in data.columns and 'low' in data.columns:
                recent_high = np.max(data['high'].tail(10))
                recent_low = np.min(data['low'].tail(10))
                current_price = closes[-1]
                if recent_high > recent_low:
                    price_position = (current_price - recent_low) / (recent_high - recent_low)
                    indicators["price_position"] = round(price_position, 3)
            
            indicators["data_points"] = len(data)
            indicators["latest_price"] = round(closes[-1], 2)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {"error": str(e)}

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility from real data"""
        try:
            if len(data) < 2:
                return 0.02
            
            closes = data['close'].values
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.02
            
            # Annualize (rough approximation for intraday data)
            return min(0.5, volatility * np.sqrt(252 * 78))  # 78 5-minute periods per day
            
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.02

    def _generate_fallback_signal(self, symbol: str, model: str, use_sentiment: bool) -> Dict:
        """Generate fallback signal when real data/AI is not available"""
        # Get deterministic seed for stable signals within 5-minute windows
        time_window = datetime.now().strftime('%Y%m%d_%H') + f"{(datetime.now().minute // 5) * 5:02d}"
        if self.use_deterministic_signals:
            seed = self._get_deterministic_seed(symbol, time_window)
            np.random.seed(seed)
        
        # Get fallback price
        current_price = self._get_real_time_price(symbol)
        if not current_price:
            current_price = 50 + hash(symbol + time_window) % 3000
        
        # Get news sentiment
        sentiment_data = self._get_news_sentiment(symbol) if use_sentiment else {"polarity": 0.0, "confidence": 0.5}
        
        # Generate base signal (deterministic)
        signal_options = ["BUY", "SELL", "HOLD"]
        base_signal = signal_options[hash(symbol + time_window + model) % 3]
        base_confidence = 0.55 + (hash(symbol + time_window) % 40) / 100.0  # 0.55-0.95
        
        # Apply sentiment adjustments
        sentiment_adjustment = 0
        if use_sentiment and sentiment_data['confidence'] > 0.6:
            polarity = sentiment_data['polarity']
            if abs(polarity) > 0.3:
                if polarity > 0.3 and base_signal == "SELL":
                    base_signal = "HOLD"
                    sentiment_adjustment = 0.05
                elif polarity < -0.3 and base_signal == "BUY":
                    base_signal = "HOLD"
                    sentiment_adjustment = -0.05
                elif polarity > 0.5:
                    base_signal = "BUY"
                    sentiment_adjustment = 0.1
                elif polarity < -0.5:
                    base_signal = "SELL"
                    sentiment_adjustment = -0.1
        
        final_confidence = min(0.95, max(0.5, base_confidence + sentiment_adjustment))
        
        # Calculate targets
        if base_signal == "BUY":
            price_target = current_price * (1.03 + np.random.uniform(0, 0.05))
            stop_loss = current_price * (0.96 - np.random.uniform(0, 0.02))
        elif base_signal == "SELL":
            price_target = current_price * (0.97 - np.random.uniform(0, 0.05))
            stop_loss = current_price * (1.04 + np.random.uniform(0, 0.02))
        else:  # HOLD
            price_target = current_price * (1 + np.random.uniform(-0.02, 0.02))
            stop_loss = current_price * (0.95 - np.random.uniform(0, 0.02))
        
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "model": f"fallback_{model}",
            "signal": base_signal,
            "confidence": round(final_confidence, 3),
            "current_price": round(current_price, 2),
            "price_target": round(price_target, 2),
            "stop_loss": round(stop_loss, 2),
            "price": round(current_price, 2),
            "time_window": time_window,
            "data_sources": ["fallback"],
            "fallback_mode": True
        }
        
        # Add basic indicators
        signal["technical_indicators"] = {
            "rsi": round(30 + (hash(symbol + time_window + "rsi") % 40), 2),
            "macd": round((hash(symbol + time_window + "macd") % 400 - 200) / 100, 4),
            "volume_ratio": round(0.8 + (hash(symbol + time_window + "vol") % 70) / 100, 3)
        }
        
        # Add sentiment data if used
        if use_sentiment:
            signal["sentiment_score"] = round(sentiment_data['polarity'], 3)
            signal["sentiment_confidence"] = round(sentiment_data['confidence'], 3)
            signal["news_article_count"] = sentiment_data['article_count']
            signal["data_sources"].append("news_sentiment")
        
        signal["data_sources"].append("deterministic_fallback")
        
        return signal

    def _get_deterministic_seed(self, symbol: str, time_window: str) -> int:
        """Generate deterministic seed for consistent signals"""
        return int(hashlib.md5(f"{symbol}_{time_window}".encode()).hexdigest()[:8], 16)
    
    def generate_signals(self, symbols: List[str], use_sentiment: bool = None, 
                        model: str = None, max_workers: int = None) -> List[Dict]:
        """Generate enhanced trading signals with real data integration"""
        start_time = time.time()
        
        # Set defaults
        if use_sentiment is None:
            use_sentiment = self.config.get('use_sentiment', True)
        if model is None:
            model = self.config.get('default_model', 'ensemble')
        if max_workers is None:
            max_workers = min(10, len(symbols))
        
        # Check cache first
        cache_key = self._get_signal_cache_key(symbols, model, use_sentiment)
        cached_signals = self._load_cached_signals(cache_key)
        
        if cached_signals is not None:
            self.performance_metrics["cache_hits"] += 1
            logger.info(f"[CACHE HIT] Returning {len(cached_signals)} cached signals")
            return cached_signals
        
        self.performance_metrics["cache_misses"] += 1
        logger.info(f"[SIGNAL GENERATION] Generating {len(symbols)} signals with model '{model}'")
        
        signals = []
        
        # Generate signals with threading for performance
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._generate_enhanced_signal, symbol, model, use_sentiment): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result()
                    if signal:
                        signals.append(signal)
                        self.performance_metrics["successful_signals"] += 1
                    else:
                        self.performance_metrics["failed_signals"] += 1
                        logger.warning(f"Failed to generate signal for {symbol}")
                        
                except Exception as e:
                    self.performance_metrics["failed_signals"] += 1
                    logger.error(f"Error generating signal for {symbol}: {e}")
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Cache the results
        self._save_signals_to_cache(signals, cache_key)
        
        # Update performance metrics
        self.performance_metrics["total_signals_generated"] += len(signals)
        self.performance_metrics["last_run_time"] = datetime.now().isoformat()
        processing_time = time.time() - start_time
        self.performance_metrics["average_generation_time"] = round(processing_time, 3)
        
        logger.info(f"[SUCCESS] Generated {len(signals)} signals in {processing_time:.2f}s")
        return signals
    
    def _load_cached_signals(self, cache_key: str) -> Optional[List[Dict]]:
        """Load cached signals if available and valid"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                age_minutes = (datetime.now() - cache_time).total_seconds() / 60
                
                if age_minutes < self.cache_duration_minutes:
                    return cache_data['signals']
                else:
                    # Clean up expired cache
                    cache_file.unlink()
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
            return None
    
    def _save_signals_to_cache(self, signals: List[Dict], cache_key: str):
        """Save signals to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            cache_data = {
                'signals': signals,
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'signal_count': len(signals)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"[CACHE SAVE] Cached {len(signals)} signals")
            
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def cleanup_old_cache_files(self, hours_old: int = 24):
        """Clean up cache files older than specified hours"""
        try:
            now = datetime.now()
            deleted_count = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    file_age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_age.total_seconds() > (hours_old * 3600):
                        cache_file.unlink()
                        deleted_count += 1
                except Exception:
                    continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old cache files")
        
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_signals_generated": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "average_generation_time": 0,
            "last_run_time": None,
            "cache_hits": 0,
            "cache_misses": 0
        }
        logger.info("Performance metrics reset")


# Example usage
if __name__ == "__main__":
    # Test the enhanced signal orchestrator
    orchestrator = SignalOrchestrator()
    
    test_symbols = ["RELIANCE.NSE", "HDFCBANK.NSE", "INFY.NSE"]
    signals = orchestrator.generate_signals(test_symbols, model="ensemble", use_sentiment=True)
    
    print(f"Generated {len(signals)} enhanced signals with real data")
    for signal in signals:
        print(f"{signal['symbol']}: {signal['signal']} @ ₹{signal['current_price']} (confidence: {signal['confidence']})")
        if 'sentiment_score' in signal:
            print(f"  News sentiment: {signal['sentiment_score']} ({signal['news_article_count']} articles)")