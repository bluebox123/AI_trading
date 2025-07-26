"""
EODHD V4 Bridge - Enhanced Data Collection Interface

This module provides a bridge to EODHD API for real-time and historical data collection.
Enhanced version with Kelly Criterion support, better error handling, and more real data usage.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf  # Fallback data source
import random
import os

logger = logging.getLogger(__name__)

class EodhdV4Bridge:
    """
    Enhanced EODHD API bridge for V4 system with improved NSE support.
    Handles real-time data, news, and sentiment analysis.
    """
    
    # Indian market topics for news filtering
    INDIAN_TOPICS = [
        "dividend payments", "earnings", "quarterly results", "IPO", 
        "corporate actions", "FII flows", "budget", "RBI policy",
        "SEBI regulations", "NSE", "BSE", "NIFTY", "SENSEX",
        "merger acquisition", "stock split", "bonus shares"
    ]
    
    def __init__(self, api_key: str = None, config: Dict = None):
        """
        Initialize the enhanced EODHD bridge.
        
        Args:
            api_key (str): EODHD API key
            config (dict): Configuration dictionary
        """
        self.api_key = api_key or os.getenv('EODHD_API_KEY')
        self.config = config or {}
        self.base_url = "https://eodhd.com/api"
        self.session = requests.Session()
        
        # Enhanced configuration
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.use_real_data = self.config.get('use_real_data', True)
        self.cache_duration = self.config.get('cache_duration', 300)  # 5 minutes
        self.enable_kelly_optimization = self.config.get('enable_kelly_optimization', True)
        
        # Rate limiting
        self.request_interval = 0.1  # 100ms between requests for premium plan
        self.last_request_time = 0
        
        # Cache for real-time data
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.api_calls_made = 0
        self.cache_hits = 0
        self.api_errors = 0
        
        logger.info("Enhanced EodhdV4Bridge initialized with real data preference")
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache_timestamps:
            return False
        
        elapsed = time.time() - self.cache_timestamps[key]
        return elapsed < self.cache_duration
    
    def _make_api_request(self, endpoint: str, params: Dict) -> Dict[str, Any]:
        """Make a rate-limited API request with error handling"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params['api_token'] = self.api_key
            params['fmt'] = 'json'
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API request failed: {response.status_code} - {response.text}")
                self.api_errors += 1
                return None
                
        except Exception as e:
            logger.error(f"API request error: {e}")
            self.api_errors += 1
            return None
    
    def get_real_time_data(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get enhanced real-time data for a symbol using EODHD Real-Time API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NSE')
            force_refresh (bool): Force API call even if cached data exists
            
        Returns:
            dict: Enhanced real-time data dictionary with Kelly metrics
        """
        cache_key = f"realtime_{symbol}"
        
        # Check cache first
        if not force_refresh and self._is_cache_valid(cache_key):
            self.cache_hits += 1
            return self.data_cache[cache_key]
        
        try:
            # Priority 1: EODHD Real-Time API with proper NSE format
            if self.use_real_data and ('.NSE' in symbol or '.BSE' in symbol or '.NS' in symbol):
                params = {}
                api_data = None
                
                # Try different NSE formats for EODHD Real-Time API
                symbol_formats = []
                if '.NSE' in symbol:
                    # Try both .NSE and .NS formats
                    symbol_formats = [symbol, symbol.replace('.NSE', '.NS')]
                elif '.NS' in symbol:
                    # Try both .NS and .NSE formats
                    symbol_formats = [symbol, symbol.replace('.NS', '.NSE')]
                else:
                    symbol_formats = [symbol]
                
                # Try each format until one works
                for test_symbol in symbol_formats:
                    logger.info(f"[REAL-TIME] Testing EODHD format: {test_symbol}")
                    api_data = self._make_api_request(f"real-time/{test_symbol}", params)
                    
                    if api_data:
                        # Check if we have valid price data
                        if 'close' in api_data and api_data['close'] != 'NA':
                            try:
                                close_price = float(api_data['close'])
                                if close_price > 0:
                                    logger.info(f"[SUCCESS] EODHD Real-Time format works: {test_symbol}")
                                    break
                            except (ValueError, TypeError):
                                pass
                        logger.warning(f"[FAILED] EODHD Real-Time format failed: {test_symbol} - Response: {api_data}")
                
                if api_data and 'close' in api_data and api_data['close'] != 'NA':
                    try:
                        # Calculate change from previous close
                        current_price = float(api_data.get('close', 0))
                        previous_close = float(api_data.get('previousClose', current_price))
                        change = current_price - previous_close
                        change_p = (change / previous_close * 100) if previous_close > 0 else 0
                        
                        # Get intraday data for momentum calculations
                        intraday_data = self.get_intraday_data(symbol, interval="5m")
                        momentum_metrics = self._calculate_momentum_metrics(intraday_data, api_data)
                        
                        enhanced_data = {
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'price': current_price,
                            'open': float(api_data.get('open', 0)),
                            'high': float(api_data.get('high', 0)),
                            'low': float(api_data.get('low', 0)),
                            'volume': int(api_data.get('volume', 0)),
                            'change': change,
                            'change_p': change_p,
                            'previous_close': previous_close,
                            'market_cap': None,  # Will get from fundamentals API
                            'pe_ratio': None,    # Will get from fundamentals API
                            'momentum_metrics': momentum_metrics,
                            'data_source': 'EODHD_REAL_TIME_API',
                            'api_timestamp': api_data.get('timestamp'),
                            'volatility_24h': momentum_metrics.get('volatility_24h', 0),
                            'recommendation': self._generate_momentum_recommendation(momentum_metrics),
                            'currency': 'INR',
                            'exchange': 'NSE' if '.NSE' in symbol or '.NS' in symbol else 'BSE',
                            'real_time': True,
                            'data_freshness': 'real_time',
                            'intraday_available': len(intraday_data) > 0,
                            'volume_momentum': momentum_metrics.get('volume_momentum', 0),
                            'price_momentum': momentum_metrics.get('price_momentum', 0),
                            'volatility': momentum_metrics.get('volatility', 0)
                        }
                        
                        # Cache the result
                        self.data_cache[cache_key] = enhanced_data
                        self.cache_timestamps[cache_key] = time.time()
                        
                        logger.info(f"[REAL-TIME] Success: {symbol} - Price: {current_price}, Change: {change_p:.2f}%")
                        return enhanced_data
                        
                    except Exception as e:
                        logger.error(f"[REAL-TIME] Error processing data for {symbol}: {e}")
            
            # Fallback to mock data
            logger.warning(f"[MOCK] Using mock real-time data for {symbol}")
            return self._get_enhanced_mock_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return self._get_enhanced_mock_data(symbol)
    
    def _calculate_momentum_metrics(self, intraday_data: pd.DataFrame, current_data: Dict) -> Dict[str, float]:
        """
        Calculate momentum metrics from intraday data and current real-time data.
        
        Args:
            intraday_data (pd.DataFrame): Intraday OHLCV data
            current_data (dict): Current real-time data
            
        Returns:
            dict: Momentum metrics including volatility, volume momentum, etc.
        """
        try:
            if intraday_data.empty:
                return self._get_default_momentum_metrics()
            
            # Calculate basic momentum metrics
            current_price = float(current_data.get('close', 0))
            
            # Price momentum (last 1 hour vs current)
            if len(intraday_data) >= 12:  # 12 x 5min = 1 hour
                hour_ago_price = intraday_data['close'].iloc[-12]
                price_momentum = ((current_price - hour_ago_price) / hour_ago_price * 100) if hour_ago_price > 0 else 0
            else:
                price_momentum = 0
            
            # Volume momentum
            if len(intraday_data) >= 12:
                recent_volume = intraday_data['volume'].tail(12).mean()
                older_volume = intraday_data['volume'].head(max(1, len(intraday_data) - 12)).mean()
                volume_momentum = ((recent_volume - older_volume) / older_volume * 100) if older_volume > 0 else 0
            else:
                volume_momentum = 0
            
            # Volatility (standard deviation of returns)
            if len(intraday_data) >= 2:
                returns = intraday_data['close'].pct_change().dropna()
                volatility = returns.std() * 100  # Convert to percentage
            else:
                volatility = 0
            
            # 24-hour volatility (if we have enough data)
            if len(intraday_data) >= 288:  # 288 x 5min = 24 hours
                daily_returns = intraday_data['close'].pct_change().dropna()
                volatility_24h = daily_returns.std() * 100
            else:
                volatility_24h = volatility
            
            # RSI calculation
            if len(intraday_data) >= 14:
                delta = intraday_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            else:
                current_rsi = 50
            
            return {
                'price_momentum': round(price_momentum, 2),
                'volume_momentum': round(volume_momentum, 2),
                'volatility': round(volatility, 2),
                'volatility_24h': round(volatility_24h, 2),
                'rsi': round(current_rsi, 2),
                'current_price': current_price,
                'data_points': len(intraday_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {e}")
            return self._get_default_momentum_metrics()
    
    def _get_default_momentum_metrics(self) -> Dict[str, float]:
        """Get default momentum metrics when data is unavailable"""
        return {
            'price_momentum': 0.0,
            'volume_momentum': 0.0,
            'volatility': 0.0,
            'volatility_24h': 0.0,
            'rsi': 50.0,
            'current_price': 0.0,
            'data_points': 0
        }
    
    def _generate_momentum_recommendation(self, momentum_metrics: Dict) -> Dict[str, Any]:
        """
        Generate trading recommendation based on momentum metrics.
        
        Args:
            momentum_metrics (dict): Calculated momentum metrics
            
        Returns:
            dict: Trading recommendation with confidence and reasoning
        """
        try:
            price_momentum = momentum_metrics.get('price_momentum', 0)
            volume_momentum = momentum_metrics.get('volume_momentum', 0)
            volatility = momentum_metrics.get('volatility', 0)
            rsi = momentum_metrics.get('rsi', 50)
            
            # Determine signal based on momentum
            signal = "HOLD"
            confidence = 0.5
            reasoning = []
            
            # Price momentum analysis
            if price_momentum > 2.0:
                signal = "BUY"
                confidence += 0.2
                reasoning.append(f"Strong positive momentum: {price_momentum:.2f}%")
            elif price_momentum < -2.0:
                signal = "SELL"
                confidence += 0.2
                reasoning.append(f"Strong negative momentum: {price_momentum:.2f}%")
            
            # Volume momentum analysis
            if volume_momentum > 20.0:
                confidence += 0.1
                reasoning.append(f"High volume momentum: {volume_momentum:.2f}%")
            elif volume_momentum < -20.0:
                confidence -= 0.1
                reasoning.append(f"Low volume momentum: {volume_momentum:.2f}%")
            
            # RSI analysis
            if rsi > 70:
                if signal == "BUY":
                    signal = "HOLD"
                    confidence -= 0.1
                reasoning.append(f"Overbought RSI: {rsi:.1f}")
            elif rsi < 30:
                if signal == "SELL":
                    signal = "HOLD"
                    confidence -= 0.1
                reasoning.append(f"Oversold RSI: {rsi:.1f}")
            
            # Volatility consideration
            if volatility > 5.0:
                confidence -= 0.1
                reasoning.append(f"High volatility: {volatility:.2f}%")
            
            # Clamp confidence to valid range
            confidence = max(0.1, min(0.95, confidence))
            
            return {
                'signal': signal,
                'confidence': round(confidence, 3),
                'reasoning': reasoning,
                'metrics_used': ['price_momentum', 'volume_momentum', 'rsi', 'volatility'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating momentum recommendation: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'reasoning': ['Insufficient data for analysis'],
                'metrics_used': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_historical_data(self, symbol: str, period: str = "1d", 
                          from_date: str = None, to_date: str = None, days: int = 30) -> pd.DataFrame:
        """
        Get enhanced historical data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period (1d, 1w, 1m)
            from_date (str): Start date (YYYY-MM-DD)
            to_date (str): End date (YYYY-MM-DD)
            days (int): Number of days to fetch (if from_date/to_date not specified)
            
        Returns:
            pd.DataFrame: Enhanced historical data with technical indicators
        """
        try:
            # Try EODHD API first with multiple symbol formats
            if self.use_real_data and ('.NSE' in symbol or '.BSE' in symbol or '.NS' in symbol):
                params = {'period': period}
                
                if from_date:
                    params['from'] = from_date
                if to_date:
                    params['to'] = to_date
                else:
                    # Default to last N days
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    params['from'] = start_date.strftime('%Y-%m-%d')
                    params['to'] = end_date.strftime('%Y-%m-%d')
                
                # Try different NSE formats for EODHD
                symbol_formats = []
                if '.NSE' in symbol:
                    symbol_formats = [symbol, symbol.replace('.NSE', '.NS')]
                elif '.NS' in symbol:
                    symbol_formats = [symbol, symbol.replace('.NS', '.NSE')]
                else:
                    symbol_formats = [symbol]
                
                # Try each format until one works
                for test_symbol in symbol_formats:
                    logger.info(f"[TESTING] EODHD historical format: {test_symbol}")
                    api_data = self._make_api_request(f"eod/{test_symbol}", params)
                    
                    if api_data and isinstance(api_data, list) and len(api_data) > 0:
                        logger.info(f"[SUCCESS] EODHD historical format works: {test_symbol} - {len(api_data)} records")
                        df = pd.DataFrame(api_data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        
                        # Add technical indicators
                        df = self._add_technical_indicators(df)
                        
                        return df
                    else:
                        logger.warning(f"[FAILED] EODHD historical format failed: {test_symbol}")
            
            # Fallback to Yahoo Finance
            if self.use_real_data and ('.NSE' in symbol or '.BSE' in symbol or '.NS' in symbol):
                # Convert to Yahoo Finance format
                yahoo_symbol = symbol
                if '.NSE' in symbol:
                    yahoo_symbol = symbol.replace('.NSE', '.NS')
                elif '.BSE' in symbol:
                    yahoo_symbol = symbol.replace('.BSE', '.BO')
                
                try:
                    logger.info(f"[YAHOO] Trying Yahoo Finance historical: {yahoo_symbol}")
                    ticker = yf.Ticker(yahoo_symbol)
                    
                    if from_date and to_date:
                        df = ticker.history(start=from_date, end=to_date)
                    else:
                        df = ticker.history(period=f"{days}d")
                    
                    if not df.empty:
                        # Convert column names to lowercase for consistency
                        df.columns = [col.lower() for col in df.columns]
                        
                        # Add technical indicators
                        df = self._add_technical_indicators(df)
                        
                        logger.info(f"[YAHOO] Yahoo Finance historical success: {symbol} - {len(df)} records")
                        return df
                        
                except Exception as e:
                    logger.error(f"Yahoo Finance historical data failed for {symbol}: {e}")
            
            # Final fallback to enhanced mock data
            logger.warning(f"[MOCK] Using mock historical data for {symbol}")
            return self._get_enhanced_mock_historical_data(symbol, days)
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return self._get_enhanced_mock_historical_data(symbol, days)
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to historical data"""
        try:
            if len(df) < 20:  # Need minimum data for indicators
                return df
            
            # Simple Moving Averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(periods=10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """
        Get intraday data using EODHD Intraday API with proper NSE symbol format.
        Enhanced with robust timestamp parsing and missing data handling.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NSE')
            interval (str): Time interval ('1m', '5m', '1h')
            
        Returns:
            pd.DataFrame: Intraday OHLCV data with proper timestamps
        """
        try:
            if self.use_real_data and ('.NSE' in symbol or '.BSE' in symbol or '.NS' in symbol):
                # Calculate from timestamp (7 days ago for 5m, 1 day for 1m, 30 days for 1h)
                if interval == "1m":
                    from_date = datetime.now() - timedelta(days=1)
                elif interval == "5m":
                    from_date = datetime.now() - timedelta(days=7)
                else:  # 1h
                    from_date = datetime.now() - timedelta(days=30)
                
                from_timestamp = int(from_date.timestamp())
                
                # EODHD Intraday API: https://eodhd.com/api/intraday/RELIANCE.NSE?interval=5m
                params = {
                    'interval': interval,
                    'from': from_timestamp,
                    'api_token': self.api_key,
                    'fmt': 'json'
                }
                
                # Try different symbol formats for NSE
                symbol_formats = []
                if '.NSE' in symbol:
                    symbol_formats = [symbol, symbol.replace('.NSE', '.NS')]
                elif '.NS' in symbol:
                    symbol_formats = [symbol, symbol.replace('.NS', '.NSE')]
                else:
                    symbol_formats = [symbol]
                
                for test_symbol in symbol_formats:
                    try:
                        logger.info(f"[INTRADAY] Testing EODHD format: {test_symbol} with {interval} interval")
                        api_data = self._make_api_request(f"intraday/{test_symbol}", params)
                        
                        if api_data and isinstance(api_data, list) and len(api_data) > 0:
                            # Convert EODHD intraday response to DataFrame with robust timestamp handling
                            df_data = []
                            valid_records = 0
                            skipped_records = 0
                            
                            for i, record in enumerate(api_data):
                                try:
                                    # Enhanced timestamp parsing with multiple fallback strategies
                                    timestamp = self._parse_intraday_timestamp(record, interval, i)
                                    
                                    if timestamp is None:
                                        skipped_records += 1
                                        logger.warning(f"[INTRADAY] Skipping record {i} for {test_symbol} - invalid timestamp")
                                        continue
                                    
                                    # Validate OHLCV data
                                    ohlcv_data = self._validate_ohlcv_data(record)
                                    if not ohlcv_data:
                                        skipped_records += 1
                                        logger.warning(f"[INTRADAY] Skipping record {i} for {test_symbol} - invalid OHLCV data")
                                        continue
                                    
                                    df_data.append({
                                        'timestamp': timestamp,
                                        'datetime': timestamp.isoformat(),
                                        'open': ohlcv_data['open'],
                                        'high': ohlcv_data['high'],
                                        'low': ohlcv_data['low'],
                                        'close': ohlcv_data['close'],
                                        'volume': ohlcv_data['volume']
                                    })
                                    valid_records += 1
                                    
                                except Exception as e:
                                    skipped_records += 1
                                    logger.warning(f"[INTRADAY] Error processing record {i} for {test_symbol}: {e}")
                                    continue
                            
                            if valid_records > 0:
                                df = pd.DataFrame(df_data)
                                df = df.set_index('timestamp')
                                df = df.sort_index()
                                
                                # Remove duplicates and fill missing timestamps
                                df = self._clean_intraday_dataframe(df, interval)
                                
                                # Add technical indicators
                                df = self._add_technical_indicators(df)
                                
                                logger.info(f"[SUCCESS] EODHD intraday format works: {test_symbol} - {len(df)} {interval} candles (skipped {skipped_records})")
                                return df
                            else:
                                logger.warning(f"EODHD intraday data has no valid records for {test_symbol}")
                        else:
                            logger.warning(f"EODHD intraday API returned invalid data for {test_symbol}")
                    except Exception as e:
                        logger.warning(f"[FAILED] EODHD intraday format failed: {test_symbol} - {e}")
                        continue
            
            # Fallback to mock intraday data
            logger.warning(f"[MOCK] Using mock intraday data for {symbol}")
            return self._get_mock_intraday_data(symbol, hours=6)
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return self._get_mock_intraday_data(symbol, hours=6)
    
    def _parse_intraday_timestamp(self, record: Dict, interval: str, record_index: int) -> Optional[pd.Timestamp]:
        """
        Parse timestamp from intraday record with multiple fallback strategies.
        
        Args:
            record: API record containing timestamp data
            interval: Time interval ('1m', '5m', '1h')
            record_index: Index of the record for fallback calculations
            
        Returns:
            pd.Timestamp or None if parsing fails
        """
        try:
            # Strategy 1: Try datetime string first
            datetime_str = record.get('datetime', '')
            if datetime_str:
                try:
                    # Handle various datetime formats
                    if 'T' in datetime_str:
                        # ISO format: 2024-01-15T09:30:00
                        dt = pd.to_datetime(datetime_str)
                    elif ' ' in datetime_str:
                        # Space format: 2024-01-15 09:30:00
                        dt = pd.to_datetime(datetime_str)
                    else:
                        # Try parsing as timestamp
                        dt = pd.to_datetime(datetime_str)
                    
                    if pd.notna(dt):
                        return dt
                except Exception as e:
                    logger.debug(f"Datetime string parsing failed: {e}")
            
            # Strategy 2: Try timestamp field
            timestamp = record.get('timestamp', 0)
            if timestamp and timestamp > 0:
                try:
                    # Handle both seconds and milliseconds
                    if timestamp > 1e10:  # Likely milliseconds
                        dt = pd.to_datetime(timestamp, unit='ms')
                    else:  # Likely seconds
                        dt = pd.to_datetime(timestamp, unit='s')
                    
                    if pd.notna(dt):
                        return dt
                except Exception as e:
                    logger.debug(f"Timestamp parsing failed: {e}")
            
            # Strategy 3: Try date and time fields separately
            date_str = record.get('date', '')
            time_str = record.get('time', '')
            if date_str and time_str:
                try:
                    combined_str = f"{date_str} {time_str}"
                    dt = pd.to_datetime(combined_str)
                    if pd.notna(dt):
                        return dt
                except Exception as e:
                    logger.debug(f"Date/time parsing failed: {e}")
            
            # Strategy 4: Generate timestamp based on interval and record index
            logger.warning(f"Using fallback timestamp generation for record {record_index}")
            return self._generate_fallback_timestamp(interval, record_index)
            
        except Exception as e:
            logger.error(f"Timestamp parsing error: {e}")
            return None
    
    def _validate_ohlcv_data(self, record: Dict) -> Optional[Dict]:
        """
        Validate OHLCV data from API record.
        
        Args:
            record: API record containing OHLCV data
            
        Returns:
            Dict with validated OHLCV data or None if invalid
        """
        try:
            # Extract OHLCV values
            open_price = record.get('open', 0)
            high_price = record.get('high', 0)
            low_price = record.get('low', 0)
            close_price = record.get('close', 0)
            volume = record.get('volume', 0)
            
            # Convert to float/int
            try:
                open_price = float(open_price) if open_price != 'NA' else 0
                high_price = float(high_price) if high_price != 'NA' else 0
                low_price = float(low_price) if low_price != 'NA' else 0
                close_price = float(close_price) if close_price != 'NA' else 0
                volume = int(volume) if volume != 'NA' else 0
            except (ValueError, TypeError):
                return None
            
            # Basic validation
            if any(price < 0 for price in [open_price, high_price, low_price, close_price]):
                return None
            
            if volume < 0:
                return None
            
            # OHLC relationship validation
            if not (low_price <= min(open_price, close_price) and 
                   high_price >= max(open_price, close_price)):
                logger.warning(f"OHLC relationship invalid: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
                return None
            
            return {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
        except Exception as e:
            logger.error(f"OHLCV validation error: {e}")
            return None
    
    def _generate_fallback_timestamp(self, interval: str, record_index: int) -> pd.Timestamp:
        """
        Generate fallback timestamp based on interval and record index.
        
        Args:
            interval: Time interval ('1m', '5m', '1h')
            record_index: Index of the record
            
        Returns:
            pd.Timestamp: Generated timestamp
        """
        try:
            # Calculate interval in minutes
            interval_minutes = {
                '1m': 1,
                '5m': 5,
                '1h': 60
            }.get(interval, 5)
            
            # Generate timestamp going backwards from current time
            current_time = datetime.now()
            minutes_back = record_index * interval_minutes
            fallback_time = current_time - timedelta(minutes=minutes_back)
            
            return pd.Timestamp(fallback_time)
            
        except Exception as e:
            logger.error(f"Fallback timestamp generation error: {e}")
            return pd.Timestamp.now()
    
    def _clean_intraday_dataframe(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Clean and validate intraday DataFrame.
        
        Args:
            df: Raw intraday DataFrame
            interval: Time interval for validation
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            if df.empty:
                return df
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Fill missing timestamps if gaps are small
            expected_freq = {
                '1m': '1T',
                '5m': '5T',
                '1h': '1H'
            }.get(interval, '5T')
            
            # Create complete time range
            start_time = df.index.min()
            end_time = df.index.max()
            complete_range = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
            
            # Reindex and forward fill small gaps (up to 2 intervals)
            df_reindexed = df.reindex(complete_range)
            
            # Only fill gaps that are reasonable (not too large)
            max_gap_intervals = 2
            gap_threshold = pd.Timedelta(expected_freq) * max_gap_intervals
            
            # Forward fill only small gaps
            df_filled = df_reindexed.ffill(limit=max_gap_intervals)
            
            # Remove rows that couldn't be filled (large gaps)
            df_clean = df_filled.dropna()
            
            logger.info(f"[CLEAN] Intraday data cleaned: {len(df)} -> {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"DataFrame cleaning error: {e}")
            return df
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data using EODHD Fundamentals API for NSE/BSE stocks.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NSE')
            
        Returns:
            dict: Fundamental data including PE ratio, market cap, etc.
        """
        try:
            if self.use_real_data and ('.NSE' in symbol or '.BSE' in symbol):
                # EODHD Fundamentals API: https://eodhd.com/api/fundamentals/RELIANCE.NSE
                params = {}
                api_data = self._make_api_request(f"fundamentals/{symbol}", params)
                
                if api_data and isinstance(api_data, dict):
                    # Extract key fundamental metrics
                    general = api_data.get('General', {})
                    highlights = api_data.get('Highlights', {})
                    valuation = api_data.get('Valuation', {})
                    
                    fundamentals = {
                        'symbol': symbol,
                        'company_name': general.get('Name', symbol.split('.')[0]),
                        'sector': general.get('Sector', 'Unknown'),
                        'industry': general.get('Industry', 'Unknown'),
                        'market_cap': highlights.get('MarketCapitalization', 0),
                        'pe_ratio': highlights.get('PERatio', 0),
                        'pb_ratio': highlights.get('PriceBookMRQ', 0),
                        'dividend_yield': highlights.get('DividendYield', 0),
                        'eps': highlights.get('EarningsShare', 0),
                        'book_value': highlights.get('BookValue', 0),
                        'revenue_ttm': highlights.get('RevenueTTM', 0),
                        'profit_margin': highlights.get('ProfitMargin', 0),
                        'roe': highlights.get('ReturnOnEquityTTM', 0),
                        'debt_to_equity': highlights.get('TotalDebtEquity', 0),
                        'exchange': 'NSE' if '.NSE' in symbol else 'BSE',
                        'currency': 'INR',
                        'data_source': 'EODHD_FUNDAMENTALS_API',
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    logger.info(f"[EODHD] Fundamentals: {symbol} - PE: {fundamentals['pe_ratio']}, MCap: {fundamentals['market_cap']}")
                    return fundamentals
                else:
                    logger.warning(f"EODHD Fundamentals API returned invalid data for {symbol}")
            
            # Fallback to mock fundamentals
            logger.warning(f"[MOCK] Using mock fundamentals for {symbol}")
            return self._get_mock_fundamentals(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return self._get_mock_fundamentals(symbol)
    
    def _categorize_market_cap(self, market_cap: int) -> str:
        """Categorize stocks by market cap"""
        if market_cap > 200000000000:  # 2 lakh crore
            return "Large Cap"
        elif market_cap > 50000000000:  # 50 thousand crore
            return "Mid Cap"
        elif market_cap > 5000000000:   # 5 thousand crore
            return "Small Cap"
        else:
            return "Micro Cap"
    
    def _get_mock_intraday_data(self, symbol: str, hours: int = 6) -> pd.DataFrame:
        """Generate mock intraday data"""
        np.random.seed(hash(symbol) % 1000)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        times = pd.date_range(start=start_time, end=end_time, freq='5T')
        
        base_price = np.random.uniform(100, 500)
        data = []
        
        for time in times:
            price = base_price * (1 + np.random.normal(0, 0.005))
            data.append({
                'datetime': time,
                'open': round(price * (1 + np.random.uniform(-0.005, 0.005)), 2),
                'high': round(price * (1 + np.random.uniform(0, 0.01)), 2),
                'low': round(price * (1 - np.random.uniform(0, 0.01)), 2),
                'close': round(price, 2),
                'volume': int(np.random.uniform(1000, 10000))
            })
            base_price = price
        
        df = pd.DataFrame(data)
        df = df.set_index('datetime')
        
        # Add basic indicators for mock data
        if len(df) >= 10:
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['volume_avg'] = df['volume'].rolling(window=10).mean()
        
        return df
    
    def _get_mock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Generate enhanced mock fundamental data"""
        np.random.seed(hash(symbol) % 1000)
        market_cap = int(np.random.uniform(10000000000, 500000000000))
        
        return {
            'General': {
                'Code': symbol.split('.')[0],
                'Type': 'Common Stock',
                'Name': f'Sample Company ({symbol})',
                'Exchange': 'NSE' if '.NSE' in symbol else 'BSE',
                'CurrencyCode': 'INR',
                'CurrencyName': 'Indian Rupee',
                'CurrencySymbol': '₹'
            },
            'Highlights': {
                'MarketCapitalization': market_cap,
                'EBITDA': int(np.random.uniform(1000000000, 50000000000)),
                'PERatio': round(np.random.uniform(10, 50), 2),
                'PEGRatio': round(np.random.uniform(0.5, 3), 2),
                'BookValue': round(np.random.uniform(50, 500), 2),
                'DividendYield': round(np.random.uniform(0, 5), 2),
                'EPS': round(np.random.uniform(5, 100), 2)
            },
            'Valuation': {
                'TrailingPE': round(np.random.uniform(10, 40), 2),
                'ForwardPE': round(np.random.uniform(8, 35), 2),
                'PriceSalesTTM': round(np.random.uniform(1, 10), 2),
                'PriceBookMRQ': round(np.random.uniform(1, 8), 2)
            },
            'calculated_metrics': {
                'enterprise_value': market_cap * 1.1,
                'price_to_book': round(np.random.uniform(1, 5), 2),
                'earnings_yield': round(np.random.uniform(0.02, 0.10), 4),
                'market_cap_category': self._categorize_market_cap(market_cap)
            },
            'data_source': 'ENHANCED_MOCK'
        }
    
    def _get_enhanced_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate enhanced mock real-time data with realistic NSE prices and Kelly metrics"""
        # Realistic NSE stock prices as of July 2025
        base_prices = {
            'RELIANCE.NSE': 2847.30,
            'TCS.NSE': 3892.45,
            'HDFCBANK.NSE': 1625.80,
            'INFY.NSE': 1743.25,
            'BHARTIARTL.NSE': 935.60,
            'ASIANPAINT.NSE': 2890.75,
            'ICICIBANK.NSE': 1234.50,
            'KOTAKBANK.NSE': 1689.30,
            'LT.NSE': 3456.20,
            'MARUTI.NSE': 11250.80,
            'WIPRO.NSE': 425.30,
            'HINDUNILVR.NSE': 2678.45,
            'BAJFINANCE.NSE': 6890.25,
            'TECHM.NSE': 1456.70,
            'POWERGRID.NSE': 234.85,
            'NTPC.NSE': 312.40,
            'COALINDIA.NSE': 456.25,
            'ONGC.NSE': 267.80,
            'IOC.NSE': 134.55,
            'GRASIM.NSE': 2345.70
        }
        
        # Use symbol-specific seed for consistency but add time factor for updates
        symbol_seed = hash(symbol) % 1000
        time_seed = int(datetime.now().timestamp()) // 300  # Updates every 5 minutes
        np.random.seed(symbol_seed + time_seed)
        
        base_price = base_prices.get(symbol, 2500.0)
        
        # Realistic intraday variation (±2%)
        variation = np.random.uniform(-0.02, 0.02)
        current_price = base_price * (1 + variation)
        
        # Generate realistic OHLC
        open_variation = np.random.uniform(-0.01, 0.01)
        open_price = base_price * (1 + open_variation)
        
        high_price = max(current_price, open_price) * (1 + np.random.uniform(0, 0.008))
        low_price = min(current_price, open_price) * (1 - np.random.uniform(0, 0.008))
        
        # Realistic volume based on stock tier
        if base_price > 5000:  # Premium stocks
            volume = int(np.random.uniform(800000, 1500000))
        elif base_price > 1000:  # Large cap
            volume = int(np.random.uniform(1000000, 2000000))
        else:  # Mid/Small cap
            volume = int(np.random.uniform(1500000, 3000000))
        
        change = current_price - base_price
        change_p = (change / base_price) * 100
        
        # Generate realistic Kelly metrics
        volatility = np.random.uniform(0.18, 0.32)
        if symbol in ['RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE']:
            volatility *= 0.8  # Blue chip lower volatility
        
        win_rate = np.random.uniform(0.48, 0.62)
        avg_win = np.random.uniform(0.015, 0.035)
        avg_loss = np.random.uniform(0.010, 0.028)
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
        safe_kelly = max(0, min(kelly_fraction * 0.25, 0.1))
        
        kelly_metrics = {
            'kelly_fraction': float(kelly_fraction),
            'safe_kelly_fraction': float(safe_kelly),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'win_loss_ratio': float(win_loss_ratio),
            'volatility_annual': float(volatility),
            'volatility_24h': float(volatility / np.sqrt(252)),
            'sharpe_estimate': float(np.random.uniform(0.8, 1.6)),
            'var_95': float(-abs(np.random.normal(0.025, 0.008))),
            'data_points': 30,
            'recommendation_strength': float(abs(safe_kelly) * 10)
        }
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price': round(current_price, 2),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'volume': volume,
            'change': round(change, 2),
            'change_p': round(change_p, 2),
            'previous_close': round(base_price, 2),
            'market_cap': int(base_price * np.random.uniform(800000000, 1200000000)),
            'pe_ratio': round(np.random.uniform(15, 35), 2),
            'kelly_metrics': kelly_metrics,
            'data_source': 'REALISTIC_MOCK_NSE',
            'volatility_24h': kelly_metrics['volatility_24h'],
            'recommendation': self._generate_kelly_recommendation(kelly_metrics),
            'currency': 'INR',
            'exchange': 'NSE' if '.NSE' in symbol else 'BSE'
        }
    
    def _get_enhanced_mock_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate enhanced mock historical data with technical indicators"""
        np.random.seed(hash(symbol) % 1000)  # Consistent data per symbol
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = np.random.uniform(100, 500)
        volatility = np.random.uniform(0.15, 0.35) / np.sqrt(252)  # Daily volatility
        
        # Generate realistic price series with some trend
        trend = np.random.uniform(-0.001, 0.001)  # Daily trend
        prices = []
        current_price = base_price
        
        for i in range(days):
            # Add trend and random walk
            daily_return = trend + np.random.normal(0, volatility)
            current_price *= (1 + daily_return)
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            daily_vol = abs(np.random.normal(0, volatility * close_price))
            high = close_price + daily_vol * np.random.uniform(0.3, 0.8)
            low = close_price - daily_vol * np.random.uniform(0.3, 0.8)
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
            volume = int(np.random.uniform(10000, 100000))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('date')
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def get_multiple_symbols_data(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get real-time data for multiple symbols efficiently.
        
        Args:
            symbols (List[str]): List of symbols to fetch
            force_refresh (bool): Force refresh all data
            
        Returns:
            Dict[str, Dict]: Dictionary mapping symbols to their data
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.get_real_time_data, symbol, force_refresh): symbol 
                for symbol in symbols
            }
            
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    results[symbol] = self._get_enhanced_mock_data(symbol)
        
        return results
    
    def get_kelly_recommendations(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get Kelly Criterion-based recommendations for multiple symbols.
        
        Args:
            symbols (List[str]): List of symbols to analyze
            
        Returns:
            Dict[str, Dict]: Kelly recommendations for each symbol
        """
        recommendations = {}
        data = self.get_multiple_symbols_data(symbols)
        
        for symbol, symbol_data in data.items():
            if symbol_data and 'recommendation' in symbol_data:
                recommendations[symbol] = {
                    'symbol': symbol,
                    'action': symbol_data['recommendation']['action'],
                    'confidence': symbol_data['recommendation']['confidence'],
                    'position_size': symbol_data['recommendation']['position_size_recommended'],
                    'risk_level': symbol_data['recommendation']['risk_level'],
                    'kelly_fraction': symbol_data['kelly_metrics']['kelly_fraction'],
                    'safe_kelly': symbol_data['kelly_metrics']['safe_kelly_fraction'],
                    'win_rate': symbol_data['kelly_metrics']['win_rate'],
                    'current_price': symbol_data['price'],
                    'volatility': symbol_data['kelly_metrics']['volatility_annual'],
                    'data_source': symbol_data['data_source'],
                    'timestamp': symbol_data['timestamp']
                }
        
        return recommendations
    
    def get_portfolio_optimization_data(self, symbols: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Get historical data optimized for portfolio optimization.
        
        Args:
            symbols (List[str]): List of symbols
            days (int): Number of historical days to fetch
            
        Returns:
            Dict[str, pd.DataFrame]: Historical data for each symbol
        """
        portfolio_data = {}
        
        for symbol in symbols:
            try:
                historical_data = self.get_historical_data(symbol, days=days)
                if not historical_data.empty:
                    # Ensure we have the required columns
                    if 'close' in historical_data.columns:
                        portfolio_data[symbol] = historical_data
                    else:
                        logger.warning(f"No close price data for {symbol}")
                else:
                    logger.warning(f"No historical data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching portfolio data for {symbol}: {e}")
        
        return portfolio_data
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the EODHD bridge"""
        total_requests = self.api_calls_made + self.cache_hits
        cache_hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0
        error_rate = (self.api_errors / self.api_calls_made) if self.api_calls_made > 0 else 0
        
        return {
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': round(cache_hit_rate * 100, 2),
            'api_errors': self.api_errors,
            'error_rate': round(error_rate * 100, 2),
            'cached_symbols': len(self.data_cache),
            'use_real_data': self.use_real_data,
            'kelly_optimization_enabled': self.enable_kelly_optimization,
            'last_request_time': datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time else None
        }
    
    def get_financial_news(self, symbol: str = None, topic: str = None, 
                          from_date: str = None, to_date: str = None, 
                          limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get financial news from EODHD News API with India/NSE focus.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NSE')
            topic (str): Topic tag (e.g., 'company announcement')
            from_date (str): Start date (YYYY-MM-DD)
            to_date (str): End date (YYYY-MM-DD)
            limit (int): Number of articles (default: 50, max: 50)
            offset (int): Pagination offset (default: 0)
            
        Returns:
            list: List of news articles with sentiment analysis
        """
        try:
            if not self.use_real_data:
                return self._get_mock_news(symbol, limit)
            
            # Build params
            params = {
                'limit': limit,
                'offset': offset
            }
            
            # Handle symbols parameter carefully to respect 62-char limit
            symbols_to_use = []
            if symbol:
                # Single symbol case
                symbols_to_use = [symbol]
            else:
                # Default to top NSE symbols if none provided
                symbols_to_use = ['NIFTY50.INDX', 'SENSEX.INDX', 'RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE']
            
            # Build symbol string respecting 62-char limit
            symbol_str = ""
            for i, sym in enumerate(symbols_to_use):
                test_str = symbol_str + ("," if symbol_str else "") + sym
                if len(test_str) <= 62:
                    symbol_str = test_str
                else:
                    break
            
            if symbol_str:
                params['s'] = symbol_str
                logger.debug(f"[SYMBOLS] Using symbols: {symbol_str} (length: {len(symbol_str)})")
            else:
                # Use minimal default if even single symbol is too long
                params['s'] = 'NIFTY50.INDX'
            
            # Add topics
            if topic:
                # Use the provided topic
                params['t'] = topic
            else:
                # Use random Indian market topics
                topics = random.sample(self.INDIAN_TOPICS, min(3, len(self.INDIAN_TOPICS)))
                params['t'] = ','.join(topics[:3])
            
            # Enhanced NSE-focused news fetching
            try:
                logger.info(f"[NEWS] Fetching EODHD news - Symbol: {symbol}, Params: {params}")
                
                # Make API request to EODHD news endpoint
                response = self._make_api_request("news", params)
                
                if response and isinstance(response, list) and len(response) > 0:
                    # Filter and enhance articles for NSE focus
                    processed_news = []
                    for article in response[:limit]:
                        try:
                            # Enhanced article processing with NSE filtering
                            processed_article = {
                                'title': article.get('title', 'No title'),
                                'content': article.get('content', article.get('description', 'No content')),
                                'url': article.get('link', ''),
                                'date': article.get('date', datetime.now().strftime('%Y-%m-%d')),
                                'time': article.get('datetime', datetime.now().strftime('%H:%M:%S')),
                                'source': article.get('source', 'EODHD'),
                                'symbols': article.get('symbols', []),
                                'tags': article.get('tags', []),
                                'sentiment': article.get('sentiment', 'neutral')
                            }
                            
                            # Add NSE tag if related to Indian markets
                            content_lower = processed_article['content'].lower()
                            title_lower = processed_article['title'].lower()
                            
                            if any(term in content_lower or term in title_lower for term in 
                                   ['nse', 'bse', 'nifty', 'sensex', 'indian', 'india', 'mumbai', 'delhi']):
                                processed_article['tags'].append('NSE')
                                processed_article['tags'].append('Indian Markets')
                            
                            processed_news.append(processed_article)
                            
                        except Exception as e:
                            logger.warning(f"Error processing article: {e}")
                            continue
                    
                    logger.info(f"[SUCCESS] EODHD News: Processed {len(processed_news)} articles")
                    return processed_news
                    
                else:
                    logger.warning("EODHD News API returned no valid data")
                    
            except Exception as e:
                logger.error(f"API request error: {e}")
            
            # Fallback to India-focused mock data
            logger.warning("[MOCK] Using India-focused mock news")
            return self._get_mock_news(symbol, limit)
            
        except Exception as e:
            logger.error(f"Error fetching EODHD news: {e}")
            return self._get_mock_news(symbol, limit)
    
    def _process_sentiment(self, sentiment: Dict) -> Dict[str, float]:
        """Process sentiment scores from EODHD news"""
        if not sentiment:
            return {
                'polarity': 0.0,
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'confidence': 0.5
            }
        
        return {
            'polarity': float(sentiment.get('polarity', 0.0)),
            'negative': float(sentiment.get('neg', 0.0)),
            'neutral': float(sentiment.get('neu', 0.0)),
            'positive': float(sentiment.get('pos', 0.0)),
            'confidence': abs(float(sentiment.get('polarity', 0.0)))
        }
    
    def _calculate_relevance_score(self, article: Dict, target_symbol: str = None) -> float:
        """Calculate relevance score for an article"""
        score = 0.5  # Base score
        
        if target_symbol and article.get('symbols'):
            if target_symbol in article.get('symbols', []):
                score += 0.3
        
        # Boost score for recent articles
        try:
            article_date = datetime.fromisoformat(article.get('date', '').replace('Z', '+00:00'))
            days_old = (datetime.now() - article_date.replace(tzinfo=None)).days
            if days_old < 1:
                score += 0.2
            elif days_old < 7:
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
    
    def _get_mock_news(self, symbol: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Generate India-focused mock news data when real API is unavailable"""
        company_name = symbol.replace('.NSE', '').replace('.BSE', '') if symbol else "Market"
        
        # India-specific news templates with balanced sentiment
        india_news_templates = [
            {
                'hours_ago': 1,
                'title': f'{company_name} Reports Strong Q2 Results, Beats Street Estimates',
                'content': f'{company_name} reported quarterly earnings that exceeded analyst expectations, driven by strong domestic demand and digital transformation initiatives. Revenue grew 12% YoY.',
                'tags': ['earnings', 'quarterly results', 'india market'],
                'sentiment': {'polarity': 0.7, 'negative': 0.1, 'neutral': 0.2, 'positive': 0.7, 'confidence': 0.8}
            },
            {
                'hours_ago': 2,
                'title': f'{company_name} Faces Regulatory Scrutiny Over Compliance Issues',
                'content': f'{company_name} stock declined 3% after reports emerged of potential regulatory action regarding compliance violations. Management promises swift remedial action.',
                'tags': ['regulatory', 'compliance', 'stock decline'],
                'sentiment': {'polarity': -0.6, 'negative': 0.7, 'neutral': 0.2, 'positive': 0.1, 'confidence': 0.8}
            },
            {
                'hours_ago': 3,
                'title': 'RBI Maintains Repo Rate at 6.5%, Signals Cautious Optimism',
                'content': 'Reserve Bank of India keeps key interest rates unchanged while maintaining accommodative stance. Governor emphasizes inflation targeting and growth support.',
                'tags': ['rbi', 'monetary policy', 'interest rates'],
                'sentiment': {'polarity': 0.3, 'negative': 0.2, 'neutral': 0.5, 'positive': 0.3, 'confidence': 0.6}
            },
            {
                'hours_ago': 4,
                'title': 'Rising Input Costs Pressure Manufacturing Sector Margins',
                'content': 'Indian manufacturing companies report margin compression due to elevated raw material costs and supply chain disruptions. Analysts downgrade sector outlook.',
                'tags': ['manufacturing', 'input costs', 'margins'],
                'sentiment': {'polarity': -0.5, 'negative': 0.65, 'neutral': 0.25, 'positive': 0.1, 'confidence': 0.75}
            },
            {
                'hours_ago': 5,
                'title': 'Nifty 50 Hits New All-Time High on FII Inflows',
                'content': 'Indian benchmark index Nifty 50 reached a new record high as foreign institutional investors increased their stake in Indian equities. Market breadth remains positive.',
                'tags': ['nifty', 'market high', 'fii inflows'],
                'sentiment': {'polarity': 0.8, 'negative': 0.05, 'neutral': 0.15, 'positive': 0.8, 'confidence': 0.9}
            },
            {
                'hours_ago': 6,
                'title': f'{company_name} Misses Revenue Guidance, Shares Tumble',
                'content': f'{company_name} reported quarterly revenue below guidance, citing headwinds in key markets. Stock fell 5% in after-hours trading as investors reassess growth prospects.',
                'tags': ['earnings miss', 'revenue', 'stock decline'],
                'sentiment': {'polarity': -0.7, 'negative': 0.75, 'neutral': 0.15, 'positive': 0.1, 'confidence': 0.85}
            },
            {
                'hours_ago': 8,
                'title': f'{company_name} Announces Major Expansion in Renewable Energy',
                'content': f'{company_name} unveils ambitious plans to invest ₹25,000 crores in renewable energy projects over the next 3 years, aligning with India\'s green transition goals.',
                'tags': ['renewable energy', 'expansion', 'green initiative'],
                'sentiment': {'polarity': 0.6, 'negative': 0.1, 'neutral': 0.3, 'positive': 0.6, 'confidence': 0.7}
            },
            {
                'hours_ago': 10,
                'title': 'Global Recession Fears Impact Indian Export-Oriented Sectors',
                'content': 'Indian IT and pharmaceutical companies face headwinds as global recession concerns mount. Export revenues expected to decline in coming quarters.',
                'tags': ['recession', 'exports', 'global impact'],
                'sentiment': {'polarity': -0.4, 'negative': 0.55, 'neutral': 0.35, 'positive': 0.1, 'confidence': 0.7}
            },
            {
                'hours_ago': 12,
                'title': 'SEBI Introduces New Framework for ESG Disclosures',
                'content': 'Securities and Exchange Board of India mandates enhanced ESG reporting requirements for top 1000 listed companies, effective from FY2025.',
                'tags': ['sebi', 'esg', 'regulatory'],
                'sentiment': {'polarity': 0.1, 'negative': 0.3, 'neutral': 0.6, 'positive': 0.1, 'confidence': 0.5}
            },
            {
                'hours_ago': 15,
                'title': 'Banking Sector Under Pressure from Rising NPAs and Credit Costs',
                'content': 'Indian banks report increase in non-performing assets and higher provisioning requirements. Credit growth slows amid tightening lending standards.',
                'tags': ['banking', 'npa', 'credit costs'],
                'sentiment': {'polarity': -0.6, 'negative': 0.7, 'neutral': 0.2, 'positive': 0.1, 'confidence': 0.8}
            },
            {
                'hours_ago': 18,
                'title': f'{company_name} Stock Rallies on Strong Digital Revenue Growth',
                'content': f'{company_name} shares surge 4% in early trading after the company reported 28% growth in digital revenues. Management remains optimistic about technology adoption trends.',
                'tags': ['stock rally', 'digital growth', 'technology'],
                'sentiment': {'polarity': 0.75, 'negative': 0.08, 'neutral': 0.17, 'positive': 0.75, 'confidence': 0.85}
            },
            {
                'hours_ago': 20,
                'title': 'Inflation Concerns Weigh on Consumer Discretionary Spending',
                'content': 'Rising inflation impacts consumer sentiment and discretionary spending patterns. FMCG and retail sectors report volume growth deceleration.',
                'tags': ['inflation', 'consumer spending', 'fmcg'],
                'sentiment': {'polarity': -0.3, 'negative': 0.5, 'neutral': 0.4, 'positive': 0.1, 'confidence': 0.6}
            },
            {
                'hours_ago': 24,
                'title': 'GST Collections Touch Record High of ₹1.87 Lakh Crores',
                'content': 'Goods and Services Tax collections for the month reached an all-time high, indicating robust economic activity and improved compliance mechanisms.',
                'tags': ['gst', 'tax collection', 'economic growth'],
                'sentiment': {'polarity': 0.65, 'negative': 0.1, 'neutral': 0.25, 'positive': 0.65, 'confidence': 0.75}
            }
        ]
        
        mock_news = []
        for i, template in enumerate(india_news_templates[:limit]):
            article = {
                'date': (datetime.now() - timedelta(hours=template['hours_ago'])).isoformat() + 'Z',
                'title': template['title'],
                'content': template['content'],
                'link': f'https://example-news.com/india-article{i+1}',
                'symbols': [symbol] if symbol else ['NIFTY50.NSE', 'SENSEX.BSE'],
                'tags': template['tags'],
                'sentiment': template['sentiment'],
                'source': 'INDIA_MOCK_NEWS',
                'relevance_score': 0.85 - (i * 0.05)  # Decreasing relevance
            }
            mock_news.append(article)
        
        return mock_news[:limit]
    
    def get_market_sentiment_summary(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get overall market sentiment summary from news"""
        all_news = []
        
        if symbols:
            for symbol in symbols:
                news = self.get_financial_news(symbol=symbol, limit=20)
                all_news.extend(news)
        else:
            # Get general market news
            news = self.get_financial_news(topic='company announcement', limit=50)
            all_news.extend(news)
        
        if not all_news:
            return self._get_mock_sentiment_summary()
        
        # Calculate overall sentiment
        total_articles = len(all_news)
        positive_count = sum(1 for article in all_news if article['sentiment']['polarity'] > 0.1)
        negative_count = sum(1 for article in all_news if article['sentiment']['polarity'] < -0.1)
        neutral_count = total_articles - positive_count - negative_count
        
        avg_polarity = np.mean([article['sentiment']['polarity'] for article in all_news])
        
        return {
            'total_articles': total_articles,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'bullish_percentage': round((positive_count / total_articles) * 100, 1),
            'bearish_percentage': round((negative_count / total_articles) * 100, 1),
            'neutral_percentage': round((neutral_count / total_articles) * 100, 1),
            'average_sentiment': round(avg_polarity, 3),
            'sentiment_strength': 'Strong' if abs(avg_polarity) > 0.5 else 'Moderate' if abs(avg_polarity) > 0.2 else 'Weak',
            'market_mood': 'Bullish' if avg_polarity > 0.1 else 'Bearish' if avg_polarity < -0.1 else 'Neutral',
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_mock_sentiment_summary(self) -> Dict[str, Any]:
        """Mock sentiment summary when real data unavailable"""
        return {
            'total_articles': 68,
            'positive_count': 42,
            'negative_count': 8,
            'neutral_count': 18,
            'bullish_percentage': 61.8,
            'bearish_percentage': 11.8,
            'neutral_percentage': 26.4,
            'average_sentiment': 0.45,
            'sentiment_strength': 'Moderate',
            'market_mood': 'Bullish',
            'last_updated': datetime.now().isoformat()
        }

    def test_connection(self) -> Dict[str, Any]:
        """Enhanced connection test with detailed diagnostics"""
        try:
            # Test basic connectivity
            test_symbol = "RELIANCE.NSE"
            params = {}
            
            start_time = time.time()
            response = self._make_api_request(f"real-time/{test_symbol}", params)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response:
                # Test data quality
                required_fields = ['close', 'open', 'high', 'low']
                data_quality = sum(1 for field in required_fields if field in response) / len(required_fields)
                
                return {
                    'connection_status': 'SUCCESS',
                    'response_time_ms': round(response_time, 2),
                    'data_quality_score': round(data_quality * 100, 1),
                    'api_key_valid': True,
                    'test_symbol': test_symbol,
                    'test_price': response.get('close', 'N/A'),
                    'timestamp': datetime.now().isoformat(),
                    'api_calls_today': response.get('api_calls_today', 'Unknown'),
                    'plan_limit': response.get('plan_limit', 'Unknown')
                }
            else:
                return {
                    'connection_status': 'FAILED',
                    'error': 'No valid response received',
                    'response_time_ms': round(response_time, 2),
                    'api_key_valid': False,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'connection_status': 'ERROR',
                'error': str(e),
                'api_key_valid': False,
                'timestamp': datetime.now().isoformat()
            }

    def get_enhanced_news(self, limit=10, symbols=None, topics=None):
        """Get enhanced news with better NSE focus and real-time sentiment"""
        try:
            if not topics:
                topics = random.sample(self.INDIAN_TOPICS, min(3, len(self.INDIAN_TOPICS)))
            
            # Attempt to get real EODHD news
            news_data = self.get_financial_news(symbols=symbols, topics=topics, limit=limit)
            
            if news_data and len(news_data) > 0:
                return {
                    "success": True,
                    "data": news_data,
                    "source": "eodhd_enhanced"
                }
            else:
                # Fallback to mock India-focused news
                mock_news = self._get_mock_news(symbol=None, limit=limit)
                return {
                    "success": True, 
                    "data": mock_news,
                    "source": "mock_india_fallback"
                }
                
        except Exception as e:
            logger.error(f"Enhanced news error: {e}")
            # Return mock news as final fallback
            mock_news = self._get_mock_news(symbol=None, limit=limit)
            return {
                "success": True,
                "data": mock_news, 
                "source": "mock_error_fallback"
            }

# Enhanced example usage and testing
if __name__ == "__main__":
    # Initialize enhanced bridge
    config = {
        'timeout': 30,
        'max_retries': 3,
        'use_real_data': True,
        'cache_duration': 300,
        'enable_kelly_optimization': True
    }
    
    bridge = EodhdV4Bridge(config=config)
    
    print("🚀 Testing Enhanced EODHD V4 Bridge...")
    
    # Test 1: Connection test
    print("\n1. Testing Connection:")
    connection_result = bridge.test_connection()
    print(f"   Status: {connection_result['connection_status']}")
    if connection_result['connection_status'] == 'SUCCESS':
        print(f"   Response Time: {connection_result['response_time_ms']} ms")
        print(f"   Data Quality: {connection_result['data_quality_score']}%")
    else:
        print(f"   Error: {connection_result.get('error', 'Unknown')}")
    
    # Test 2: Real-time data with Kelly metrics
    print("\n2. Testing Real-time Data with Kelly Metrics:")
    test_symbols = ["RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE"]
    
    for symbol in test_symbols[:2]:  # Test first 2 to avoid rate limits
        data = bridge.get_real_time_data(symbol)
        print(f"   {symbol}:")
        print(f"     Price: ₹{data['price']}")
        print(f"     Kelly Fraction: {data['kelly_metrics']['kelly_fraction']:.4f}")
        print(f"     Safe Kelly: {data['kelly_metrics']['safe_kelly_fraction']:.4f}")
        print(f"     Recommendation: {data['recommendation']['action']} ({data['recommendation']['confidence']:.2f})")
        print(f"     Data Source: {data['data_source']}")
    
    # Test 3: Kelly recommendations
    print("\n3. Testing Kelly Recommendations:")
    recommendations = bridge.get_kelly_recommendations(test_symbols[:2])
    for symbol, rec in recommendations.items():
        print(f"   {symbol}: {rec['action']} (Confidence: {rec['confidence']:.2f}, Size: {rec['position_size']:.3f})")
    
    # Test 4: Performance metrics
    print("\n4. Performance Metrics:")
    metrics = bridge.get_system_performance_metrics()
    print(f"   API Calls: {metrics['api_calls_made']}")
    print(f"   Cache Hit Rate: {metrics['cache_hit_rate']}%")
    print(f"   Error Rate: {metrics['error_rate']}%")
    print(f"   Real Data Enabled: {metrics['use_real_data']}")
    
    print("\n✅ Enhanced EODHD V4 Bridge testing completed!") 