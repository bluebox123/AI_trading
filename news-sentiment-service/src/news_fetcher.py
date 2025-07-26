"""
News fetcher module for Google News RSS feeds
Handles fetching, parsing, and deduplication of news articles
"""
import feedparser
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set
from urllib.parse import quote_plus
import os
import json

from config.config import (
    GOOGLE_NEWS_BASE_URL, NSE_TICKERS, TICKER_SEARCH_NAMES,
    REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY,
    MIN_REQUEST_INTERVAL, SEEN_LINKS_FILE, MAX_ENTRIES_PER_TICKER,
    SEEN_LINKS_MAX_SIZE
)

class NewsFetcher:
    """
    Handles fetching and parsing of Google News RSS feeds for NSE tickers
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seen_links: Set[str] = set()
        self.last_request_time = 0
        self.load_seen_links()
        
    def load_seen_links(self):
        """Load previously seen links from file to avoid duplicates"""
        try:
            if os.path.exists(SEEN_LINKS_FILE):
                with open(SEEN_LINKS_FILE, 'r', encoding='utf-8') as f:
                    self.seen_links = set(line.strip() for line in f if line.strip())
                self.logger.info(f"Loaded {len(self.seen_links)} seen links from file")
            else:
                self.logger.info("No existing seen links file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading seen links: {e}")
            self.seen_links = set()
    
    def save_seen_links(self):
        """Save seen links to file for persistence across restarts"""
        try:
            # Limit the size to prevent file from growing too large
            if len(self.seen_links) > SEEN_LINKS_MAX_SIZE:
                # Keep only the most recent links (this is a simple approach)
                self.seen_links = set(list(self.seen_links)[-SEEN_LINKS_MAX_SIZE:])
                
            with open(SEEN_LINKS_FILE, 'w', encoding='utf-8') as f:
                for link in self.seen_links:
                    f.write(f"{link}\n")
            self.logger.debug(f"Saved {len(self.seen_links)} seen links to file")
        except Exception as e:
            self.logger.error(f"Error saving seen links: {e}")
    
    def rate_limit_check(self):
        """Ensure we don't make requests too frequently"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def build_search_url(self, ticker: str, language: str = 'en') -> str:
        """Build Google News RSS search URL for a given ticker"""
        # Use company name if available, otherwise use ticker
        search_term = TICKER_SEARCH_NAMES.get(ticker, ticker.replace('.NS', ''))
        
        # Add India and NSE context to improve relevance
        search_query = f'"{search_term}" India NSE stock market'
        encoded_query = quote_plus(search_query)
        
        url = f"{GOOGLE_NEWS_BASE_URL}?q={encoded_query}&hl={language}&gl=IN&ceid=IN:en"
        return url
    
    def fetch_ticker_news(self, ticker: str, max_retries: int = RETRY_ATTEMPTS) -> List[Dict]:
        """
        Fetch news articles for a specific ticker with retry logic
        """
        articles = []
        
        for attempt in range(max_retries):
            try:
                self.rate_limit_check()
                
                url = self.build_search_url(ticker)
                self.logger.debug(f"Fetching news for {ticker} from: {url}")
                
                # Parse RSS feed
                feed = feedparser.parse(url)
                
                if hasattr(feed, 'status') and feed.status != 200:
                    self.logger.warning(f"RSS feed returned status {feed.status} for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                
                if not hasattr(feed, 'entries') or not feed.entries:
                    self.logger.warning(f"No entries found in RSS feed for {ticker}")
                    break
                
                # Process entries
                for entry in feed.entries[:MAX_ENTRIES_PER_TICKER]:
                    try:
                        # Extract article information
                        article = {
                            'ticker': ticker,
                            'headline': entry.title,
                            'link': entry.link,
                            'pub_date': entry.published if hasattr(entry, 'published') else '',
                            'description': entry.summary if hasattr(entry, 'summary') else '',
                            'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News',
                            'fetch_timestamp': datetime.utcnow().isoformat()
                        }
                        
                        # Skip if we've seen this link before
                        if article['link'] in self.seen_links:
                            self.logger.debug(f"Skipping duplicate article: {article['headline'][:50]}...")
                            continue
                        
                        # Add to seen links
                        self.seen_links.add(article['link'])
                        articles.append(article)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing entry for {ticker}: {e}")
                        continue
                
                self.logger.info(f"Fetched {len(articles)} new articles for {ticker}")
                break  # Success, no need to retry
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"All retry attempts failed for {ticker}")
        
        return articles
    
    def fetch_all_news(self, tickers: List[str] = None) -> List[Dict]:
        """
        Fetch news for all specified tickers
        """
        if tickers is None:
            tickers = NSE_TICKERS
        
        all_articles = []
        start_time = time.time()
        
        self.logger.info(f"Starting news fetch for {len(tickers)} tickers")
        
        for i, ticker in enumerate(tickers):
            try:
                self.logger.debug(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
                articles = self.fetch_ticker_news(ticker)
                all_articles.extend(articles)
                
                # Small delay between tickers to be respectful
                if i < len(tickers) - 1:  # Don't sleep after the last ticker
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch news for {ticker}: {e}")
                continue
        
        # Save seen links after processing all tickers
        self.save_seen_links()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Completed news fetch: {len(all_articles)} new articles in {elapsed_time:.2f} seconds")
        
        return all_articles
    
    def get_stats(self) -> Dict:
        """Get statistics about the news fetcher"""
        return {
            'seen_links_count': len(self.seen_links),
            'last_request_time': datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time else None,
            'monitored_tickers': len(NSE_TICKERS)
        }
    
    def cleanup_old_links(self, days_old: int = 7):
        """
        Clean up old seen links to prevent memory issues
        This is a simple implementation - in production you might want to use timestamps
        """
        if len(self.seen_links) > SEEN_LINKS_MAX_SIZE:
            # Simple cleanup: remove oldest links (keep latest SEEN_LINKS_MAX_SIZE)
            self.seen_links = set(list(self.seen_links)[-SEEN_LINKS_MAX_SIZE:])
            self.save_seen_links()
            self.logger.info(f"Cleaned up seen links, now tracking {len(self.seen_links)} links")


if __name__ == "__main__":
    # Test the news fetcher
    logging.basicConfig(level=logging.INFO)
    fetcher = NewsFetcher()
    
    # Test with a few tickers
    test_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    articles = fetcher.fetch_all_news(test_tickers)
    
    print(f"Fetched {len(articles)} articles")
    for article in articles[:3]:  # Show first 3
        print(f"- {article['ticker']}: {article['headline']}") 