# src/data/eodhd_historical_collector.py
# Historical Stock + News Data Collector for Temporal Causality Training
# UPDATED: Now uses 117 NSE stocks from sentiment data (2020-05-01 to 2025-05-31)

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import gzip
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')  # Explicitly load .env.local file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EODHDHistoricalCollector:
    """
    EODHD Historical Data Collector for Multimodal AI Training
    Collects time-aligned stock prices + news for temporal causality
    UPDATED: Now uses 117 NSE stocks from sentiment data (2020-05-01 to 2025-05-31)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Load sentiment data stocks (117 stocks)
        self.sentiment_stocks = self.load_sentiment_stocks()
        
        # Sentiment data date range
        self.start_date = "2020-05-01"  # First date in sentiment data
        self.end_date = "2025-05-31"    # Last date in sentiment data
        
        # Data directories
        self.data_dir = Path("data")
        self.raw_data_dir = self.data_dir / "raw"
        self.stock_data_dir = self.raw_data_dir / "stock_data"
        self.news_data_dir = self.raw_data_dir / "news_data"
        
        # Create directories
        self.create_directories()
        
        # API call counter
        self.api_calls_made = 0
        self.max_daily_calls = 100000

    def load_sentiment_stocks(self) -> List[str]:
        """Load the 117 stocks from sentiment data"""
        try:
            with open('sentiment_stocks.txt', 'r') as f:
                stocks = [line.strip() for line in f.readlines()]
            logger.info(f"SUCCESS: Loaded {len(stocks)} stocks from sentiment data")
            return stocks
        except FileNotFoundError:
            logger.error("ERROR: sentiment_stocks.txt not found. Please extract stocks from sentiment data first.")
            # Fallback to a few known stocks if file doesn't exist
            return [
                "RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE", "INFY.NSE", "ICICIBANK.NSE",
                "HINDUNILVR.NSE", "ITC.NSE", "SBIN.NSE", "BHARTIARTL.NSE", "ASIANPAINT.NSE"
            ]

    def create_directories(self):
        """Create necessary data directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.stock_data_dir.mkdir(exist_ok=True)
        self.news_data_dir.mkdir(exist_ok=True)
        logger.info("SUCCESS: Created data directories")

    def test_api_connection(self) -> bool:
        """Test EODHD API connection"""
        logger.info("Testing EODHD API Connection...")
        
        url = f"{self.base_url}/eod/AAPL.US"
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'period': 'd',
            'from': '2024-06-01',
            'to': '2024-06-10'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data:
                logger.info(f"SUCCESS: API Connection Success! Retrieved {len(data)} records")
                self.api_calls_made += 1
                return True
            else:
                logger.error("ERROR: API returned empty data")
                return False
                
        except Exception as e:
            logger.error(f"ERROR: API Connection Failed: {e}")
            return False

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE.NSE)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching price data for {symbol}: {start_date} to {end_date}")
        
        url = f"{self.base_url}/eod/{symbol}"
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'period': 'd',
            'from': start_date,
            'to': end_date
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.api_calls_made += 1
            
            data = response.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Add metadata
                df.attrs['symbol'] = symbol
                df.attrs['source'] = 'eodhd'
                df.attrs['collected_at'] = datetime.now().isoformat()
                
                logger.info(f"SUCCESS {symbol}: Retrieved {len(df)} price records")
                time.sleep(self.rate_limit_delay)
                return df
            else:
                logger.warning(f"WARNING {symbol}: No price data returned")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"ERROR {symbol} price fetch error: {e}")
            return pd.DataFrame()

    def save_stock_data(self, df: pd.DataFrame, symbol: str, year: int, month: int):
        """Save stock data to compressed CSV"""
        if df.empty:
            return
            
        filename = f"{symbol}_{year}-{month:02d}_prices.csv.gz"
        filepath = self.stock_data_dir / filename
        
        # Save with compression
        df.to_csv(filepath, compression='gzip')
        logger.info(f"SAVED {symbol} data: {len(df)} records -> {filename}")

    def get_historical_news(self, symbol: str, start_date: str, end_date: str, limit: int = 1000) -> List[Dict]:
        """
        Fetch historical news for a symbol
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE.NSE)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Number of articles to fetch
            
        Returns:
            List of news articles with metadata
        """
        logger.info(f"Fetching news for {symbol}: {start_date} to {end_date}")
        
        url = f"{self.base_url}/news"
        params = {
            's': symbol,  # Stock-specific news
            'api_token': self.api_key,
            'from': start_date,
            'to': end_date,
            'limit': limit,
            'fmt': 'json'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.api_calls_made += 1
            
            news_data = response.json()
            if isinstance(news_data, list):
                # Add metadata to each article
                for article in news_data:
                    article['symbol'] = symbol
                    article['news_type'] = 'stock_specific'
                    article['collected_at'] = datetime.now().isoformat()
                
                logger.info(f"SUCCESS {symbol}: Retrieved {len(news_data)} news articles")
                time.sleep(self.rate_limit_delay)
                return news_data
            else:
                logger.warning(f"⚠️ {symbol}: No news data returned")
                return []
                
        except Exception as e:
            logger.error(f"❌ {symbol} news fetch error: {e}")
            return []

    def save_news_data(self, news_data: List[Dict], symbol: str, year: int, month: int):
        """Save news data to compressed JSON"""
        if not news_data:
            return
            
        filename = f"{symbol}_{year}-{month:02d}_news.json.gz"
        filepath = self.news_data_dir / filename
        
        # Save with compression
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved {symbol} news: {len(news_data)} articles -> {filename}")

    def collect_monthly_data(self, year: int, month: int, symbols: List[str] = None) -> Dict:
        """
        Collect all data for a specific month - TEMPORAL CAUSALITY KEY
        
        Args:
            year: Year (e.g., 2021)
            month: Month (1-12)
            symbols: List of symbols to collect (default: all sentiment stocks)
            
        Returns:
            Dictionary with collected data
        """
        if symbols is None:
            symbols = self.sentiment_stocks
        
        # Calculate month boundaries
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"
        
        # Adjust end_date to last day of month
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)
        end_date = end_dt.strftime('%Y-%m-%d')
        
        logger.info(f"🚀 Collecting data for {year}-{month:02d}: {start_date} to {end_date}")
        
        monthly_data = {
            'period': f"{year}-{month:02d}",
            'start_date': start_date,
            'end_date': end_date,
            'symbols_processed': 0,
            'total_price_records': 0,
            'total_news_articles': 0,
            'errors': []
        }
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📊 Processing {symbol} ({i}/{len(symbols)})")
            
            try:
                # Collect price data
                price_df = self.get_historical_prices(symbol, start_date, end_date)
                if not price_df.empty:
                    self.save_stock_data(price_df, symbol, year, month)
                    monthly_data['total_price_records'] += len(price_df)
                
                # Collect news data
                news_data = self.get_historical_news(symbol, start_date, end_date)
                if news_data:
                    self.save_news_data(news_data, symbol, year, month)
                    monthly_data['total_news_articles'] += len(news_data)
                
                monthly_data['symbols_processed'] += 1
                
                # Progress update
                if i % 10 == 0:
                    logger.info(f"⏳ Progress: {i}/{len(symbols)} symbols ({(i/len(symbols)*100):.1f}%)")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                monthly_data['errors'].append(error_msg)
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        logger.info(f"✅ Month {year}-{month:02d} complete: {monthly_data['symbols_processed']} symbols, {monthly_data['total_price_records']} prices, {monthly_data['total_news_articles']} news")
        return monthly_data

    def run_collection(self, start_year: int = 2020, start_month: int = 5, end_year: int = 2025, end_month: int = 5):
        """
        Run the complete historical data collection
        UPDATED: For sentiment data alignment (2020-05-01 to 2025-05-31)
        """
        logger.info("Starting Historical Data Collection for Sentiment Alignment")
        logger.info(f"Date Range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        logger.info(f"Symbols: {len(self.sentiment_stocks)} stocks from sentiment data")
        
        # Test API connection
        if not self.test_api_connection():
            logger.error("ERROR: API connection failed. Exiting.")
            return
        
        collection_summary = {
            'start_time': datetime.now().isoformat(),
            'date_range': f"{start_year}-{start_month:02d} to {end_year}-{end_month:02d}",
            'total_symbols': len(self.sentiment_stocks),
            'monthly_results': [],
            'total_api_calls': 0,
            'total_price_records': 0,
            'total_news_articles': 0,
            'errors': []
        }
        
        # Generate month list
        current_year, current_month = start_year, start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            try:
                monthly_result = self.collect_monthly_data(current_year, current_month)
                collection_summary['monthly_results'].append(monthly_result)
                collection_summary['total_price_records'] += monthly_result['total_price_records']
                collection_summary['total_news_articles'] += monthly_result['total_news_articles']
                collection_summary['errors'].extend(monthly_result['errors'])
                
                # API call tracking
                collection_summary['total_api_calls'] = self.api_calls_made
                
                logger.info(f"📊 Running totals: {collection_summary['total_api_calls']} API calls, {collection_summary['total_price_records']} prices, {collection_summary['total_news_articles']} news")
                
            except Exception as e:
                logger.error(f"❌ Failed to collect data for {current_year}-{current_month:02d}: {e}")
                collection_summary['errors'].append(f"{current_year}-{current_month:02d}: {str(e)}")
            
            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        # Final summary
        collection_summary['end_time'] = datetime.now().isoformat()
        collection_summary['duration'] = str(datetime.fromisoformat(collection_summary['end_time']) - datetime.fromisoformat(collection_summary['start_time']))
        
        logger.info("🎉 DATA COLLECTION COMPLETE!")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   • Total API Calls: {collection_summary['total_api_calls']}")
        logger.info(f"   • Total Price Records: {collection_summary['total_price_records']}")
        logger.info(f"   • Total News Articles: {collection_summary['total_news_articles']}")
        logger.info(f"   • Total Errors: {len(collection_summary['errors'])}")
        logger.info(f"   • Duration: {collection_summary['duration']}")
        
        # Save collection summary
        summary_file = self.data_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(collection_summary, f, indent=2)
        
        logger.info(f"💾 Collection summary saved: {summary_file}")

def main():
    """Main function to run the collector"""
    # Get API key from environment
    api_key = os.getenv('EODHD_API_KEY')
    if not api_key:
        logger.error("ERROR: EODHD_API_KEY environment variable not set")
        return
    
    # Initialize collector
    collector = EODHDHistoricalCollector(api_key)
    
    # Run collection for sentiment data range
    collector.run_collection(
        start_year=2020, start_month=5,  # Start: 2020-05-01
        end_year=2025, end_month=5       # End: 2025-05-31
    )

if __name__ == "__main__":
    main() 