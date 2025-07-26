"""
Configuration for Comprehensive Sentiment Dataset Generator
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "sentiment_v2"

# Output directories
DAILY_DATA_DIR = DATA_DIR / "daily_data"
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

# Create directories
for dir_path in [DATA_DIR, DAILY_DATA_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset parameters
START_DATE = "2020-05-01"
END_DATE = "2025-05-31"
STOCK_SYMBOLS_FILE = BASE_DIR / "nse_stock_symbols_complete.txt"

# Google News RSS settings
GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search?q={}&hl=en-IN&gl=IN&ceid=IN:en&tbm=nws"
REQUEST_DELAY = 2  # seconds between requests
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

# Threading settings
MAX_WORKERS = 4
BATCH_SIZE = 50

# Rate limiting
RATE_LIMIT_PER_MINUTE = 30
RATE_LIMIT_PER_HOUR = 1000

# Data quality settings
MIN_ARTICLE_LENGTH = 50
MAX_ARTICLES_PER_SEARCH = 10
MAX_SEARCHES_PER_STOCK_DATE = 3

# Enhanced intraday sentiment settings
INTRADAY_SETTINGS = {
    'max_lookback_days': 30,  # Look back 30 days for intraday calculation
    'temporal_decay_half_life': 24,  # 24-hour half-life for temporal decay
    'momentum_calculation_window': 7,  # 7-day window for momentum calculation
    'confidence_trend_window': 14,  # 14-day window for confidence trends
    'bucket_types': ['hourly', '4hour', 'daily', 'weekly', 'monthly'],
    'min_articles_per_bucket': 2,  # Minimum articles needed for reliable bucket
    'temporal_weighting_enabled': True,
    'adaptive_confidence': True
}

# Temporal weighting settings
TEMPORAL_WEIGHTING = {
    'default_half_life_days': 7.0,
    'adaptive_weighting': True,
    'window_boost_enabled': True,
    'decay_profiles': {
        'breaking_news': 2.0,  # 2-day half-life for breaking news
        'earnings_reports': 7.0,  # 7-day half-life for earnings
        'general_news': 14.0,  # 14-day half-life for general news
        'historical_data': 30.0  # 30-day half-life for historical
    }
}

# Sentiment analysis settings
SENTIMENT_THRESHOLDS = {
    'positive': 0.1,
    'negative': -0.1,
    'neutral': (-0.1, 0.1)
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Progress tracking
PROGRESS_SAVE_INTERVAL = 100  # Save progress every N tasks

# User agent for requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Company name variations for better search results
COMPANY_NAME_VARIATIONS = {
    'RELIANCE.NSE': ['Reliance Industries', 'RIL', 'Reliance', 'Mukesh Ambani', 'Jio'],
    'TCS.NSE': ['Tata Consultancy Services', 'TCS', 'Tata Consultancy'],
    'HDFCBANK.NSE': ['HDFC Bank', 'Housing Development Finance Corporation', 'HDFC'],
    'INFY.NSE': ['Infosys', 'Infosys Limited', 'Narayana Murthy'],
    'ICICIBANK.NSE': ['ICICI Bank', 'ICICI'],
    'HINDUNILVR.NSE': ['Hindustan Unilever', 'HUL', 'Unilever'],
    'ITC.NSE': ['ITC Limited', 'Indian Tobacco Company', 'ITC'],
    'SBIN.NSE': ['State Bank of India', 'SBI'],
    'BHARTIARTL.NSE': ['Bharti Airtel', 'Airtel', 'Sunil Mittal'],
    'KOTAKBANK.NSE': ['Kotak Mahindra Bank', 'Kotak Bank', 'Kotak', 'Uday Kotak'],
    'LT.NSE': ['Larsen & Toubro', 'L&T', 'Larsen Toubro'],
    'ASIANPAINT.NSE': ['Asian Paints', 'Asian Paint'],
    'MARUTI.NSE': ['Maruti Suzuki', 'Maruti', 'Suzuki India'],
    'BAJFINANCE.NSE': ['Bajaj Finance', 'Bajaj Finserv'],
    'HCLTECH.NSE': ['HCL Technologies', 'HCL Tech', 'HCL'],
    'AXISBANK.NSE': ['Axis Bank', 'Axis'],
    'ULTRACEMCO.NSE': ['UltraTech Cement', 'Ultratech', 'Aditya Birla'],
    'SUNPHARMA.NSE': ['Sun Pharmaceutical', 'Sun Pharma', 'Dilip Shanghvi'],
    'TITAN.NSE': ['Titan Company', 'Titan', 'Tanishq', 'Tata Titan'],
    'TECHM.NSE': ['Tech Mahindra', 'Mahindra Tech'],
    'POWERGRID.NSE': ['Power Grid Corporation', 'PowerGrid', 'PGCIL'],
    'NTPC.NSE': ['NTPC Limited', 'NTPC'],
    'ONGC.NSE': ['Oil and Natural Gas Corporation', 'ONGC'],
    'COALINDIA.NSE': ['Coal India', 'CIL'],
    'WIPRO.NSE': ['Wipro Limited', 'Wipro', 'Azim Premji'],
    'TATAMOTORS.NSE': ['Tata Motors', 'Tata Motor', 'Jaguar Land Rover'],
    'TATASTEEL.NSE': ['Tata Steel', 'Tata Iron Steel'],
    'JSWSTEEL.NSE': ['JSW Steel', 'Jindal Steel Works'],
    'HINDALCO.NSE': ['Hindalco Industries', 'Hindalco', 'Novelis'],
    'ADANIPORTS.NSE': ['Adani Ports', 'Adani Ports and SEZ', 'Gautam Adani'],
    'ADANIENT.NSE': ['Adani Enterprises', 'Adani Group'],
    'BPCL.NSE': ['Bharat Petroleum', 'BPCL'],
    'BRITANNIA.NSE': ['Britannia Industries', 'Britannia'],
    'CIPLA.NSE': ['Cipla Limited', 'Cipla'],
    'DIVISLAB.NSE': ['Divi\'s Laboratories', 'Divis Lab'],
    'DRREDDY.NSE': ['Dr. Reddy\'s Laboratories', 'Dr Reddy', 'DRL'],
    'EICHERMOT.NSE': ['Eicher Motors', 'Royal Enfield'],
    'GRASIM.NSE': ['Grasim Industries', 'Grasim'],
    'HEROMOTOCO.NSE': ['Hero MotoCorp', 'Hero Honda'],
    'INDUSINDBK.NSE': ['IndusInd Bank', 'Indusind'],
    'NESTLEIND.NSE': ['Nestle India', 'Nestle'],
    'SHREECEM.NSE': ['Shree Cement', 'Shree'],
    'UPL.NSE': ['UPL Limited', 'United Phosphorus'],
    'APOLLOHOSP.NSE': ['Apollo Hospitals', 'Apollo Health'],
    'BAJAJ-AUTO.NSE': ['Bajaj Auto', 'Bajaj'],
    'GODREJCP.NSE': ['Godrej Consumer Products', 'Godrej'],
    'PIDILITIND.NSE': ['Pidilite Industries', 'Fevicol'],
    'TATACONSUM.NSE': ['Tata Consumer Products', 'Tata Tea'],
    'DMART.NSE': ['Avenue Supermarts', 'DMart', 'D-Mart']
}

# Sector mapping for enhanced analysis
SECTOR_MAPPING = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 
               'FEDERALBNK', 'PNB', 'BANKBARODA', 'AUBANK', 'BANDHANBNK', 'IDFCFIRSTB'],
    'Technology': ['TCS', 'INFY', 'HCLTECH', 'TECHM', 'WIPRO', 'LTIM', 'LTTS', 
                  'MINDTREE', 'MPHASIS', 'COFORGE', 'PERSISTENT'],
    'Pharmaceuticals': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'LUPIN', 'BIOCON', 
                       'AUROPHARMA', 'ALKEM', 'TORNTPHARM', 'CADILAHC', 'GLENMARK', 'IPCALAB'],
    'Automotive': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 
                  'BALKRISIND', 'EXIDEIND', 'MOTHERSON', 'CUMMINSIND'],
    'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'NTPC', 'POWERGRID', 'COALINDIA', 'GAIL'],
    'FMCG': ['HINDUNILVR', 'ITC', 'BRITANNIA', 'NESTLEIND', 'GODREJCP', 'DABUR', 
            'MARICO', 'TATACONSUM', 'COLPAL'],
    'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'NMDC', 'SAIL', 'JINDALSTEL'],
    'Cement': ['ULTRACEMCO', 'SHREECEM', 'ACC', 'AMBUJACEMENT'],
    'Telecom': ['BHARTIARTL'],
    'Healthcare': ['APOLLOHOSP', 'MAXHEALTH'],
    'Retail': ['DMART', 'TRENT'],
    'Paints': ['ASIANPAINT', 'BERGEPAINT'],
    'Chemicals': ['UPL', 'SRF', 'PIDILITIND'],
    'Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY'],
    'Capital Goods': ['LT', 'SIEMENS', 'BOSCHLTD', 'HAVELLS']
} 