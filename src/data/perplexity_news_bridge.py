"""
Perplexity API Bridge for Indian Market News and Analysis
Provides NSE-focused news, market regime analysis, and sentiment data
"""

import requests
import json
import logging
from datetime import datetime, timedelta, time as dt_time
import os
from typing import Dict, List, Optional, Any
import time
import pytz

logger = logging.getLogger(__name__)

class PerplexityNewsBridge:
    """Bridge to Perplexity API for Indian market analysis"""
    
    DAILY_LIMIT = 5
    MARKET_HOURS_LIMIT = 3
    OFF_HOURS_LIMIT = 2
    MARKET_HOURS_GAP = timedelta(hours=3)
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Perplexity API key not provided.")
        self.base_url = "https://api.perplexity.ai"
        self.usage_file = "data/perplexity_usage.json"
        self.dev_request_made_this_session = False  # Dev flag
        self._load_usage()
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
    def _load_usage(self):
        """Load usage tracking from file, or create a new one for the day."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, 'r') as f:
                    self.usage_data = json.load(f)
            except:
                self.usage_data = {}
        else:
            self.usage_data = {}
        
        # Initialize or reset for new day
        if not self.usage_data or self.usage_data.get('date') != today_str:
            self.usage_data = {
                "date": today_str,
                "total_requests_used": 0,
                "market_hours_used": 0,
                "off_hours_used": 0,
                "requests": []
            }
            self._save_usage()
        else:
            # Ensure all required fields exist for existing data
            if 'total_requests_used' not in self.usage_data:
                self.usage_data['total_requests_used'] = 0
            if 'market_hours_used' not in self.usage_data:
                self.usage_data['market_hours_used'] = 0
            if 'off_hours_used' not in self.usage_data:
                self.usage_data['off_hours_used'] = 0
            if 'requests' not in self.usage_data:
                self.usage_data['requests'] = []
    
    def _save_usage(self):
        """Save the current usage data to the file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=4)
        except IOError as e:
            logger.error(f"Could not save Perplexity usage data: {e}")
    
    def can_make_request(self) -> (bool, str):
        """
        Check if a request can be made. 
        DEV OVERRIDE: Allows one request per server session to facilitate testing.
        """
        # DEV ONLY: To allow testing without waiting, this enables one request per server run.
        if self.dev_request_made_this_session:
            return False, "DEV MODE: Only one Perplexity request allowed per server run. Please restart the server to refresh."
        
        # We also check the daily limit even in dev mode to avoid budget overruns.
        self._load_usage()
        if self.usage_data['total_requests_used'] >= self.DAILY_LIMIT:
            logger.warning(f"Perplexity daily limit reached ({self.usage_data['total_requests_used']}/{self.DAILY_LIMIT}).")
            return False, "Daily request limit reached. Check usage file."

        return True, "Request allowed"
    
    def get_comprehensive_nse_update(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Gets a comprehensive Indian market update from Perplexity, including market regime and news.
        """
        can_make, reason = self.can_make_request()
        if not can_make:
            logger.warning(f"Perplexity request blocked: {reason}")
            return {"success": False, "error": reason, "data": None}

        symbols_str = ", ".join(symbols)
        
        prompt = f"""
As an expert financial analyst for the Indian stock market (NSE), provide a comprehensive market update as a single, clean JSON object. Do not include any text before or after the JSON.

The JSON object should have two main keys: "market_analysis" and "key_news".

1.  Under "market_analysis", provide the following:
    *   "market_regime": A string, one of "Bullish", "Bearish", "Neutral", "High Volatility".
    *   "regime_confidence": A float between 0 and 1.
    *   "key_drivers": A brief string (1-2 sentences) explaining the factors driving the current regime (e.g., FII flows, RBI policy, global cues).
    *   "sentiment_score": A float between -1 (very bearish) and 1 (very bullish) for the overall market.
    *   "top_sectors": An array of 3-5 strings of currently bullish sectors.

2.  Under "key_news", provide an array of 5 to 7 of the most impactful news articles from the last 24 hours relevant to the Indian market and these specific symbols: {symbols_str}.
    *   Prioritize news from these sources: moneycontrol.com, economictimes.indiatimes.com, business-standard.com, livemint.com, Reuters, BloombergQuint.
    *   For each article in the array, provide an object with these keys:
        *   "headline": The news headline.
        *   "summary": A 2-3 sentence summary of the key information and its market impact.
        *   "symbols": An array of relevant NSE stock symbols (e.g., ["RELIANCE.NSE", "TCS.NSE"]). If none, provide an empty array.
        *   "source": The domain of the news source (e.g., "moneycontrol.com").
        *   "sentiment": A string, one of "Positive", "Negative", "Neutral".
        *   "sentiment_score": A float from -1.0 to 1.0.

VERY IMPORTANT: Ensure the entire output is ONLY the JSON object, with no introductory text, explanations, or markdown formatting. Remove all citations.
"""
        response = self._make_request(prompt)
        
        if response.get("success"):
            # Set the dev flag to prevent more requests this session
            self.dev_request_made_this_session = True
            logger.info("Perplexity dev flag set. No more requests this session.")
            
            # Also update the persistent usage tracker
            self._update_usage()
            
            # Parse the content
            try:
                content_str = response['data']['choices'][0]['message']['content']
                # Clean potential markdown code fences
                if content_str.strip().startswith("```json"):
                    content_str = content_str.strip()[7:-4]
                
                parsed_content = json.loads(content_str)
                return {"success": True, "data": parsed_content}
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse Perplexity JSON response: {e}")
                logger.debug(f"Raw response content: {response.get('data')}")
                return {"success": False, "error": "Failed to parse Perplexity response", "raw": response.get('data')}
        
        return response

    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make request to Perplexity API"""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": "You are a financial expert providing structured JSON data for the Indian stock market."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_usage(self):
        """Update usage tracking after successful request"""
        now = datetime.now()
        self.usage_data['total_requests_used'] += 1
        self.usage_data['last_request'] = now.isoformat()
        
        if self.is_market_hours(now):
            self.usage_data['market_hours_used'] += 1
        else:
            self.usage_data['off_hours_used'] += 1
        
        self._save_usage()
    
    def get_usage_status(self) -> Dict[str, Any]:
        self._load_usage()
        can_make, reason = self.can_make_request()
        return {
            "success": True,
            "data": {
                **self.usage_data,
                "is_market_hours": self.is_market_hours(datetime.now()),
                "can_make_request": can_make,
                "reason": reason
            },
            "available": can_make,
            "timestamp": datetime.now().isoformat()
        }

    def is_market_hours(self, dt: datetime) -> bool:
        """Check if current time is during Indian market hours (9:15 AM - 3:30 PM IST)"""
        indian_tz = pytz.timezone('Asia/Kolkata')
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        
        # Make sure datetime is timezone-aware
        if dt.tzinfo is None:
            # Assume UTC if naive, then convert to India time
            dt = pytz.utc.localize(dt)
            
        local_time = dt.astimezone(indian_tz).time()

        return market_open <= local_time <= market_close
    
    def _parse_news_content(self, content: str, symbols: List[str] = None) -> List[Dict]:
        """Parse Perplexity response into structured news articles"""
        articles = []
        
        # Split content into logical sections
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:  # Minimum content length
                
                # Extract sentiment
                sentiment = 'neutral'
                if any(word in section.lower() for word in ['positive', 'gains', 'up', 'bullish', 'strong']):
                    sentiment = 'positive'
                elif any(word in section.lower() for word in ['negative', 'falls', 'down', 'bearish', 'weak']):
                    sentiment = 'negative'
                
                # Try to extract symbol mentions
                mentioned_symbols = []
                if symbols:
                    for symbol in symbols:
                        symbol_name = symbol.replace('.NSE', '')
                        if symbol_name.lower() in section.lower():
                            mentioned_symbols.append(symbol)
                
                article = {
                    'title': f"Market Update {i+1}",
                    'content': section.strip(),
                    'sentiment': sentiment,
                    'symbols': mentioned_symbols,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'source': 'Perplexity Analysis',
                    'tags': ['NSE', 'Indian Markets']
                }
                
                articles.append(article)
        
        return articles[:10]  # Limit to 10 articles
    
    def _extract_market_regime(self, content: str) -> str:
        """Extract market regime from analysis content"""
        regimes = ['BULL_MARKET', 'BEAR_MARKET', 'SIDEWAYS_CONSOLIDATION', 'HIGH_VOLATILITY', 'RECOVERY']
        
        content_upper = content.upper()
        for regime in regimes:
            if regime in content_upper:
                return regime
        
        # Fallback logic based on keywords
        if any(word in content.lower() for word in ['bull', 'bullish', 'rally', 'uptrend']):
            return 'BULL_MARKET'
        elif any(word in content.lower() for word in ['bear', 'bearish', 'correction', 'downtrend']):
            return 'BEAR_MARKET'
        elif any(word in content.lower() for word in ['volatile', 'volatility', 'uncertain']):
            return 'HIGH_VOLATILITY'
        elif any(word in content.lower() for word in ['sideways', 'range', 'consolidation']):
            return 'SIDEWAYS_CONSOLIDATION'
        else:
            return 'UNKNOWN'
    
    def get_nse_market_news(self, symbols: List[str] = None, limit: int = 10) -> Dict:
        """Get latest NSE market news and analysis"""
        can_request, reason = self.can_make_request()
        if not can_request:
            logger.warning(f"Perplexity request blocked: {reason}")
            return {
                'success': False,
                'error': f"API usage limit: {reason}",
                'usage_info': self.usage_data
            }
        
        # Build targeted NSE prompt
        if symbols:
            symbol_list = ", ".join([s.replace('.NSE', '') for s in symbols[:5]])  # Limit to 5 symbols
            prompt = f"""Get the latest financial news and developments for these NSE-listed companies: {symbol_list}. 
            Include recent price movements, earnings updates, corporate announcements, and analyst recommendations. 
            Focus only on news from today and yesterday. Provide sentiment (positive/negative/neutral) for each company.
            Remove all citations and references from your response."""
        else:
            prompt = """Get the latest Indian stock market news focusing on NSE-listed companies. 
            Include market trends, sectoral performance, top gainers/losers, and any major corporate announcements. 
            Focus on today's developments. Provide overall market sentiment.
            Remove all citations and references from your response."""
        
        try:
            response = self._make_request(prompt)
            
            if response and 'choices' in response:
                self._update_usage()
                
                content = response['choices'][0]['message']['content']
                
                # Parse the response into structured format
                articles = self._parse_news_content(content, symbols)
                
                return {
                    'success': True,
                    'articles': articles,
                    'total_articles': len(articles),
                    'data_source': 'perplexity_nse',
                    'usage_info': {
                        'requests_remaining': self.DAILY_LIMIT - self.usage_data['total_requests_used'],
                        'market_hours_remaining': self.MARKET_HOURS_LIMIT - self.usage_data['market_hours_used'],
                        'off_hours_remaining': self.OFF_HOURS_LIMIT - self.usage_data['off_hours_used']
                    },
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid response from Perplexity API',
                    'usage_info': self.usage_data
                }
                
        except Exception as e:
            logger.error(f"Error getting NSE news: {e}")
            return {
                'success': False,
                'error': str(e),
                'usage_info': self.usage_data
            }
    
    def get_market_regime_analysis(self) -> Dict:
        """Get current market regime analysis for Indian markets"""
        can_request, reason = self.can_make_request()
        if not can_request:
            return {
                'success': False,
                'error': f"API usage limit: {reason}",
                'regime': 'unknown'
            }
        
        prompt = """Analyze the current Indian stock market regime. Consider:
        1. NIFTY 50 and SENSEX recent performance and trends
        2. Market volatility levels (VIX India if available)
        3. Sectoral rotation patterns
        4. FII/DII flows
        5. Global factors affecting Indian markets
        
        Classify the market regime as one of: BULL_MARKET, BEAR_MARKET, SIDEWAYS_CONSOLIDATION, HIGH_VOLATILITY, or RECOVERY.
        Provide reasoning and key factors. Include current support and resistance levels for NIFTY.
        Remove all citations and references from your response."""
        
        try:
            response = self._make_request(prompt)
            
            if response and 'choices' in response:
                self._update_usage()
                
                content = response['choices'][0]['message']['content']
                regime = self._extract_market_regime(content)
                
                return {
                    'success': True,
                    'regime': regime,
                    'analysis': content,
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'perplexity_analysis'
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid response from Perplexity API',
                    'regime': 'unknown'
                }
                
        except Exception as e:
            logger.error(f"Error getting market regime: {e}")
            return {
                'success': False,
                'error': str(e),
                'regime': 'unknown'
            } 