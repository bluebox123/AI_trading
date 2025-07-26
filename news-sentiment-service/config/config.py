"""
Configuration settings for the NSE News Sentiment Pipeline
"""
import os
from datetime import timedelta

# ================== NEWS FETCHING SETTINGS ==================
# Google News RSS settings
GOOGLE_NEWS_BASE_URL = "https://news.google.com/rss/search"
POLLING_INTERVAL_MINUTES = 10  # How often to check for new news
MAX_ENTRIES_PER_TICKER = 64  # Google RSS limit
REQUEST_TIMEOUT = 30  # Timeout for RSS requests in seconds
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed requests
RETRY_DELAY = 5  # Delay between retries in seconds

# ================== NSE TICKERS ==================
# Top NSE tickers to monitor (can be expanded)
NSE_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
    'HDFCBANK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'ASIANPAINT.NS',
    'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
    'NESTLEIND.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'POWERGRID.NS', 'NTPC.NS',
    'COALINDIA.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'TECHM.NS', 'ONGC.NS',
    'TITAN.NS', 'DIVISLAB.NS', 'TATASTEEL.NS', 'ADANIPORTS.NS', 'GRASIM.NS'
]

# Alternative ticker names for news search (some tickers work better with company names)
TICKER_SEARCH_NAMES = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services TCS',
    'INFY.NS': 'Infosys',
    'HINDUNILVR.NS': 'Hindustan Unilever HUL',
    'ICICIBANK.NS': 'ICICI Bank',
    'HDFCBANK.NS': 'HDFC Bank',
    'ITC.NS': 'ITC Limited',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen Toubro',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'WIPRO.NS': 'Wipro',
    'NESTLEIND.NS': 'Nestle India',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'HCLTECH.NS': 'HCL Technologies',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Limited',
    'COALINDIA.NS': 'Coal India',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'TECHM.NS': 'Tech Mahindra',
    'ONGC.NS': 'Oil Natural Gas Corporation ONGC',
    'TITAN.NS': 'Titan Company',
    'DIVISLAB.NS': 'Divi\'s Laboratories',
    'TATASTEEL.NS': 'Tata Steel',
    'ADANIPORTS.NS': 'Adani Ports',
    'GRASIM.NS': 'Grasim Industries'
}

# ================== SENTIMENT ANALYSIS SETTINGS ==================
# FinBERT model settings
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence score for sentiment classification
BATCH_SIZE = 16  # Batch size for sentiment analysis
MAX_TEXT_LENGTH = 512  # Maximum text length for tokenization

# ================== DATA PERSISTENCE SETTINGS ==================
# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Output files
NEWS_SENTIMENT_CSV = os.path.join(DATA_DIR, 'nse_news_sentiment.csv')
SEEN_LINKS_FILE = os.path.join(DATA_DIR, 'seen_links.txt')
ERROR_LOG_FILE = os.path.join(LOGS_DIR, 'error.log')
INFO_LOG_FILE = os.path.join(LOGS_DIR, 'info.log')

# Database settings (if using SQLite instead of CSV)
DATABASE_URL = os.path.join(DATA_DIR, 'news_sentiment.db')
USE_DATABASE = False  # Set to True to use SQLite instead of CSV

# ================== LOGGING SETTINGS ==================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ================== SCHEDULER SETTINGS ==================
TIMEZONE = 'Asia/Kolkata'
SCHEDULER_COALESCE = True  # Coalesce missed jobs
SCHEDULER_MAX_INSTANCES = 1  # Maximum instances of the job running simultaneously

# ================== RATE LIMITING SETTINGS ==================
# To respect Google's rate limits
MIN_REQUEST_INTERVAL = 2  # Minimum seconds between requests to avoid rate limiting
BATCH_TICKER_SIZE = 5  # Process tickers in batches to avoid overwhelming the service
BATCH_DELAY = 10  # Delay between batches in seconds

# ================== PERFORMANCE SETTINGS ==================
# Memory management
SEEN_LINKS_MAX_SIZE = 10000  # Maximum number of seen links to keep in memory
CLEANUP_INTERVAL_HOURS = 24  # How often to clean up old seen links

# ================== ERROR HANDLING ==================
MAX_CONSECUTIVE_FAILURES = 5  # Stop service after this many consecutive failures
FAILURE_COOLDOWN_MINUTES = 30  # Wait time after max failures before restarting

# ================== DEVELOPMENT SETTINGS ==================
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
DRY_RUN = os.getenv('DRY_RUN', 'False').lower() == 'true'  # Don't save data in dry run mode
VERBOSE_LOGGING = os.getenv('VERBOSE', 'False').lower() == 'true'

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True) 