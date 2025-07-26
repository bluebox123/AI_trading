#!/usr/bin/env python3
"""
Enhanced Perplexity News & Market Analysis System
Provides institutional-grade news sentiment and market insights for trading signals
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPerplexityAnalyzer:
    """
    Advanced Perplexity-powered market analysis for institutional trading
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.usage_file = "data/perplexity_usage.json"
        self.cache_file = "data/perplexity_cache.json"
        self.daily_limit = 20  # Increased for critical insights
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
    def _check_daily_usage(self) -> bool:
        """Check if daily usage limit is reached"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                today = datetime.now().strftime('%Y-%m-%d')
                if usage_data.get('date') == today:
                    return usage_data.get('count', 0) < self.daily_limit
            return True
        except Exception as e:
            logger.warning(f"Error checking usage: {e}")
            return True
    
    def _update_usage(self):
        """Update daily usage counter"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            usage_data = {'date': today, 'count': 1}
            
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    existing = json.load(f)
                if existing.get('date') == today:
                    usage_data['count'] = existing.get('count', 0) + 1
            
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f)
                
        except Exception as e:
            logger.warning(f"Error updating usage: {e}")
    
    def get_comprehensive_market_analysis(self, symbol: str, current_price: float, signal: str, confidence: float) -> Dict:
        """
        Get comprehensive market analysis for a specific stock signal
        This is the KEY function that provides institutional-grade insights
        """
        if not self._check_daily_usage():
            logger.warning("Daily Perplexity limit reached, using cached/fallback analysis")
            return self._get_cached_analysis(symbol, signal)
        
        try:
            # Construct comprehensive analysis prompt
            prompt = self._build_comprehensive_prompt(symbol, current_price, signal, confidence)
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",  # Valid Perplexity model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an institutional-grade financial analyst with access to real-time market data. Provide precise, actionable insights for professional traders."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1,  # Low temperature for factual analysis
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["finance"],
                "search_recency_filter": "day"  # Only recent news
            }
            
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis = self._parse_perplexity_response(result)
                
                # Update usage and cache
                self._update_usage()
                self._cache_analysis(symbol, signal, analysis)
                
                logger.info(f"✅ Perplexity analysis completed for {symbol}")
                return analysis
            else:
                logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                return self._get_fallback_analysis(symbol, signal)
                
        except Exception as e:
            logger.error(f"Error in Perplexity analysis: {e}")
            return self._get_fallback_analysis(symbol, signal)
    
    def _build_comprehensive_prompt(self, symbol: str, current_price: float, signal: str, confidence: float) -> str:
        """
        Build a comprehensive prompt for institutional-grade analysis
        """
        company_name = self._get_company_name(symbol)
        
        prompt = f"""
INSTITUTIONAL TRADING ANALYSIS REQUEST for {symbol} ({company_name})

CONTEXT:
- Current Price: ₹{current_price}
- AI Signal: {signal} 
- Model Confidence: {confidence:.1f}%
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

REQUIRED ANALYSIS (Provide ALL sections):

1. BREAKING NEWS IMPACT (Last 24 hours):
   - Any earnings announcements, management guidance, or quarterly results?
   - Corporate actions: dividends, bonuses, splits, buybacks?
   - Regulatory news: SEBI actions, compliance issues, investigations?
   - Sector-specific developments affecting {company_name}?
   - Institutional activity: FII/DII flows, block deals, insider trading?

2. MARKET SENTIMENT SCORE:
   - Overall sentiment: POSITIVE/NEGATIVE/NEUTRAL (with intensity 1-10)
   - News sentiment breakdown: % positive vs % negative stories
   - Social media/analyst sentiment if available
   - Institutional research upgrades/downgrades in last 48 hours

3. TECHNICAL MARKET CONTEXT:
   - How is {symbol} performing vs Nifty 50 today?
   - Sector performance (Banking/IT/Pharma/Auto/etc.)
   - Overall market regime: Bull/Bear/Sideways/Volatile
   - Key support/resistance levels mentioned in financial media

4. FUNDAMENTAL CATALYSTS:
   - Upcoming events: earnings dates, AGMs, result announcements
   - Industry trends affecting the stock
   - Competitive landscape changes
   - Management commentary or guidance revisions

5. RISK FACTORS:
   - Current volatility levels
   - Any red flags or warning signals in recent news
   - Regulatory or legal challenges
   - Market concentration risks

6. ACTIONABLE RECOMMENDATION:
   - Do you AGREE or DISAGREE with the {signal} signal?
   - Price targets mentioned by analysts (if any)
   - Optimal entry/exit levels
   - Stop-loss recommendations
   - Position sizing suggestions based on current volatility

Format response as JSON with clear sections. Be specific with prices, percentages, and dates.
        """
        
        return prompt
    
    def _parse_perplexity_response(self, response: dict) -> Dict:
        """Parse Perplexity response into structured analysis"""
        try:
            content = response['choices'][0]['message']['content']
            citations = response.get('citations', [])
            
            # Try to parse as JSON first, otherwise extract key information
            try:
                if content.strip().startswith('{'):
                    analysis = json.loads(content)
                else:
                    analysis = self._extract_structured_data(content)
            except:
                analysis = self._extract_structured_data(content)
            
            # Add metadata
            analysis['perplexity_sources'] = len(citations)
            analysis['analysis_timestamp'] = datetime.now().isoformat()
            analysis['citations'] = citations[:5]  # Top 5 sources
            analysis['data_freshness'] = 'real_time'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing Perplexity response: {e}")
            return self._get_fallback_analysis("UNKNOWN", "UNKNOWN")
    
    def _extract_structured_data(self, content: str) -> Dict:
        """Extract key information from unstructured Perplexity response"""
        lines = content.split('\n')
        
        analysis = {
            'sentiment_score': 0.0,
            'sentiment_category': 'NEUTRAL',
            'news_impact': 'MODERATE',
            'market_regime': 'MIXED',
            'recommendation': 'HOLD',
            'confidence_adjustment': 0.0,
            'key_drivers': [],
            'risk_level': 'MEDIUM',
            'target_price': None,
            'stop_loss': None
        }
        
        content_lower = content.lower()
        
        # Extract sentiment
        if 'very positive' in content_lower or 'strong buy' in content_lower:
            analysis['sentiment_score'] = 0.8
            analysis['sentiment_category'] = 'VERY_POSITIVE'
        elif 'positive' in content_lower or 'bullish' in content_lower:
            analysis['sentiment_score'] = 0.6
            analysis['sentiment_category'] = 'POSITIVE'
        elif 'negative' in content_lower or 'bearish' in content_lower:
            analysis['sentiment_score'] = -0.6
            analysis['sentiment_category'] = 'NEGATIVE'
        elif 'very negative' in content_lower or 'sell' in content_lower:
            analysis['sentiment_score'] = -0.8
            analysis['sentiment_category'] = 'VERY_NEGATIVE'
        
        # Extract market regime
        if 'bull' in content_lower:
            analysis['market_regime'] = 'BULLISH'
        elif 'bear' in content_lower:
            analysis['market_regime'] = 'BEARISH'
        elif 'volatile' in content_lower:
            analysis['market_regime'] = 'VOLATILE'
        elif 'sideways' in content_lower:
            analysis['market_regime'] = 'SIDEWAYS'
        
        # Extract recommendation
        if 'strong buy' in content_lower:
            analysis['recommendation'] = 'STRONG_BUY'
        elif 'buy' in content_lower:
            analysis['recommendation'] = 'BUY'
        elif 'strong sell' in content_lower:
            analysis['recommendation'] = 'STRONG_SELL'
        elif 'sell' in content_lower:
            analysis['recommendation'] = 'SELL'
        
        # Extract key information
        for line in lines:
            if any(keyword in line.lower() for keyword in ['earnings', 'result', 'dividend', 'bonus']):
                analysis['key_drivers'].append(line.strip())
        
        analysis['raw_content'] = content[:500]  # First 500 chars for reference
        
        return analysis
    
    def get_market_regime_analysis(self) -> Dict:
        """Get overall market regime analysis"""
        if not self._check_daily_usage():
            return self._get_cached_market_regime()
        
        try:
            prompt = f"""
NIFTY 50 MARKET REGIME ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

Provide current market analysis:

1. NIFTY 50 PERFORMANCE:
   - Today's performance vs yesterday
   - Weekly trend
   - Key levels (support/resistance)

2. MARKET REGIME CLASSIFICATION:
   - Current regime: BULL/BEAR/SIDEWAYS/VOLATILE
   - Regime strength (1-10)
   - Expected duration

3. SECTORAL ANALYSIS:
   - Best performing sectors today
   - Worst performing sectors today
   - Rotation patterns

4. FII/DII FLOWS:
   - Net institutional flows today
   - Impact on market sentiment

5. KEY MARKET DRIVERS:
   - Major news affecting Indian markets
   - Global market influence
   - Economic data releases

Respond in JSON format with specific data points.
            """
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.1,
                "search_domain_filter": ["finance"],
                "search_recency_filter": "day"
            }
            
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                market_analysis = self._parse_market_regime_response(result)
                self._update_usage()
                return market_analysis
            else:
                return self._get_cached_market_regime()
                
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return self._get_cached_market_regime()
    
    def _parse_market_regime_response(self, response: dict) -> Dict:
        """Parse market regime response"""
        try:
            content = response['choices'][0]['message']['content']
            
            # Default market regime
            regime_analysis = {
                'market_regime': 'MIXED',
                'regime_strength': 5,
                'nifty_performance': 0.0,
                'volatility_level': 'MEDIUM',
                'sector_leaders': [],
                'sector_laggards': [],
                'fii_flow': 'NEUTRAL',
                'key_drivers': [],
                'timestamp': datetime.now().isoformat()
            }
            
            content_lower = content.lower()
            
            # Determine market regime
            if 'bullish' in content_lower or 'bull market' in content_lower:
                regime_analysis['market_regime'] = 'BULLISH'
                regime_analysis['regime_strength'] = 7
            elif 'bearish' in content_lower or 'bear market' in content_lower:
                regime_analysis['market_regime'] = 'BEARISH'
                regime_analysis['regime_strength'] = 7
            elif 'volatile' in content_lower or 'volatility' in content_lower:
                regime_analysis['market_regime'] = 'VOLATILE'
                regime_analysis['regime_strength'] = 8
            elif 'sideways' in content_lower or 'range' in content_lower:
                regime_analysis['market_regime'] = 'SIDEWAYS'
                regime_analysis['regime_strength'] = 6
            
            # Extract performance data
            lines = content.split('\n')
            for line in lines:
                if '%' in line and ('nifty' in line.lower() or 'index' in line.lower()):
                    try:
                        # Extract percentage
                        import re
                        percentages = re.findall(r'[-+]?\d*\.?\d+%', line)
                        if percentages:
                            regime_analysis['nifty_performance'] = float(percentages[0].replace('%', ''))
                    except:
                        pass
            
            regime_analysis['raw_analysis'] = content[:300]
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error parsing market regime response: {e}")
            return self._get_cached_market_regime()
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        company_map = {
            'RELIANCE.NSE': 'Reliance Industries',
            'TCS.NSE': 'Tata Consultancy Services', 
            'HDFCBANK.NSE': 'HDFC Bank',
            'INFY.NSE': 'Infosys',
            'ICICIBANK.NSE': 'ICICI Bank',
            'NESTLEIND.NSE': 'Nestle India',
            'BAJFINANCE.NSE': 'Bajaj Finance',
            'KOTAKBANK.NSE': 'Kotak Mahindra Bank',
            'ASIANPAINT.NSE': 'Asian Paints',
            'MARUTI.NSE': 'Maruti Suzuki',
            'LALPATHLAB.NSE': 'Dr. Lal PathLabs',
            'POWERGRID.NSE': 'Power Grid Corporation',
            'TATAMOTORS.NSE': 'Tata Motors'
        }
        return company_map.get(symbol, symbol.replace('.NSE', ''))
    
    def _get_cached_analysis(self, symbol: str, signal: str) -> Dict:
        """Get cached analysis if available"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                
                cache_key = f"{symbol}_{signal}_{datetime.now().strftime('%Y-%m-%d')}"
                if cache_key in cache:
                    return cache[cache_key]
        except:
            pass
        
        return self._get_fallback_analysis(symbol, signal)
    
    def _cache_analysis(self, symbol: str, signal: str, analysis: Dict):
        """Cache analysis for reuse"""
        try:
            cache = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            
            cache_key = f"{symbol}_{signal}_{datetime.now().strftime('%Y-%m-%d')}"
            cache[cache_key] = analysis
            
            # Keep only today's cache
            today = datetime.now().strftime('%Y-%m-%d')
            cache = {k: v for k, v in cache.items() if today in k}
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f)
                
        except Exception as e:
            logger.warning(f"Error caching analysis: {e}")
    
    def _get_fallback_analysis(self, symbol: str, signal: str) -> Dict:
        """Fallback analysis when Perplexity is unavailable"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'NEUTRAL',
            'news_impact': 'UNKNOWN',
            'market_regime': 'MIXED',
            'recommendation': signal,
            'confidence_adjustment': 0.0,
            'key_drivers': ['Real-time news analysis unavailable'],
            'risk_level': 'MEDIUM',
            'data_source': 'fallback',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_cached_market_regime(self) -> Dict:
        """Get cached market regime if available"""
        return {
            'market_regime': 'MIXED',
            'regime_strength': 5,
            'nifty_performance': 0.0,
            'volatility_level': 'MEDIUM',
            'sector_leaders': ['Technology', 'Banking'],
            'sector_laggards': ['PSU', 'Metal'],
            'fii_flow': 'NEUTRAL',
            'key_drivers': ['Mixed market conditions'],
            'timestamp': datetime.now().isoformat(),
            'data_source': 'cached'
        } 