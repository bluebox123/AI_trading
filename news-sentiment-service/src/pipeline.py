"""
Main news sentiment pipeline orchestrator
Coordinates news fetching, sentiment analysis, and data persistence
"""
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
import traceback

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from data_persistence import DataPersistence
from config.config import (
    NSE_TICKERS, BATCH_TICKER_SIZE, BATCH_DELAY,
    MAX_CONSECUTIVE_FAILURES, FAILURE_COOLDOWN_MINUTES,
    DRY_RUN, DEBUG_MODE
)

class NewsSentimentPipeline:
    """
    Main pipeline that orchestrates the entire news sentiment analysis process
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.news_fetcher = None
        self.sentiment_analyzer = None
        self.data_persistence = None
        self.consecutive_failures = 0
        self.last_run_time = None
        self.total_articles_processed = 0
        self.pipeline_stats = {
            'runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_articles': 0,
            'total_new_articles': 0,
            'start_time': datetime.utcnow().isoformat()
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            self.logger.info("Initializing news sentiment pipeline components...")
            
            # Initialize news fetcher
            self.logger.info("Initializing news fetcher...")
            self.news_fetcher = NewsFetcher()
            
            # Initialize sentiment analyzer
            self.logger.info("Initializing sentiment analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer()
            
            # Initialize data persistence
            self.logger.info("Initializing data persistence...")
            self.data_persistence = DataPersistence()
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def run_single_iteration(self, tickers: Optional[List[str]] = None) -> Dict:
        """
        Run a single iteration of the news sentiment pipeline
        """
        iteration_start = time.time()
        iteration_stats = {
            'start_time': datetime.utcnow().isoformat(),
            'success': False,
            'articles_fetched': 0,
            'articles_analyzed': 0,
            'articles_saved': 0,
            'error': None,
            'duration_seconds': 0
        }
        
        try:
            self.logger.info("Starting news sentiment pipeline iteration")
            
            # Use default tickers if none provided
            if tickers is None:
                tickers = NSE_TICKERS
            
            # Step 1: Fetch news articles
            self.logger.info(f"Fetching news for {len(tickers)} tickers...")
            articles = self.news_fetcher.fetch_all_news(tickers)
            iteration_stats['articles_fetched'] = len(articles)
            
            if not articles:
                self.logger.info("No new articles found in this iteration")
                iteration_stats['success'] = True
                return iteration_stats
            
            self.logger.info(f"Fetched {len(articles)} new articles")
            
            # Step 2: Analyze sentiment
            self.logger.info("Analyzing sentiment for fetched articles...")
            analyzed_articles = self.sentiment_analyzer.analyze_articles(articles)
            iteration_stats['articles_analyzed'] = len(analyzed_articles)
            
            if not analyzed_articles:
                self.logger.warning("No articles were successfully analyzed")
                iteration_stats['error'] = "Sentiment analysis failed for all articles"
                return iteration_stats
            
            # Step 3: Save results
            if not DRY_RUN:
                self.logger.info("Saving analyzed articles to storage...")
                save_success = self.data_persistence.save_records(analyzed_articles)
                
                if save_success:
                    iteration_stats['articles_saved'] = len(analyzed_articles)
                    self.logger.info(f"Successfully saved {len(analyzed_articles)} articles")
                else:
                    self.logger.error("Failed to save analyzed articles")
                    iteration_stats['error'] = "Failed to save articles to storage"
                    return iteration_stats
            else:
                self.logger.info(f"DRY RUN: Would save {len(analyzed_articles)} articles")
                iteration_stats['articles_saved'] = len(analyzed_articles)
            
            # Update statistics
            self.total_articles_processed += len(analyzed_articles)
            self.consecutive_failures = 0  # Reset failure counter on success
            iteration_stats['success'] = True
            
            # Log sentiment summary
            self._log_sentiment_summary(analyzed_articles)
            
        except Exception as e:
            self.logger.error(f"Pipeline iteration failed: {e}")
            if DEBUG_MODE:
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            iteration_stats['error'] = str(e)
            self.consecutive_failures += 1
            
        finally:
            iteration_stats['duration_seconds'] = time.time() - iteration_start
            self.last_run_time = datetime.utcnow()
            
            # Update pipeline statistics
            self.pipeline_stats['runs'] += 1
            if iteration_stats['success']:
                self.pipeline_stats['successful_runs'] += 1
            else:
                self.pipeline_stats['failed_runs'] += 1
            
            self.pipeline_stats['total_articles'] += iteration_stats['articles_fetched']
            self.pipeline_stats['total_new_articles'] += iteration_stats['articles_saved']
        
        return iteration_stats
    
    def _log_sentiment_summary(self, articles: List[Dict]):
        """Log a summary of sentiment analysis results"""
        if not articles:
            return
        
        sentiment_counts = {}
        total_confidence = 0
        confident_count = 0
        
        for article in articles:
            sentiment = article.get('sentiment', 'unknown')
            confidence = article.get('sentiment_confidence', 0)
            is_confident = article.get('sentiment_is_confident', False)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += confidence
            if is_confident:
                confident_count += 1
        
        avg_confidence = total_confidence / len(articles) if articles else 0
        
        self.logger.info(f"Sentiment Summary - Total: {len(articles)}, "
                        f"Positive: {sentiment_counts.get('positive', 0)}, "
                        f"Neutral: {sentiment_counts.get('neutral', 0)}, "
                        f"Negative: {sentiment_counts.get('negative', 0)}, "
                        f"Avg Confidence: {avg_confidence:.3f}, "
                        f"High Confidence: {confident_count}/{len(articles)}")
    
    def run_batch_processing(self, tickers: Optional[List[str]] = None) -> Dict:
        """
        Run processing in batches to respect rate limits
        """
        if tickers is None:
            tickers = NSE_TICKERS
        
        batch_stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'total_articles': 0,
            'start_time': datetime.utcnow().isoformat(),
            'batch_results': []
        }
        
        # Process tickers in batches
        for i in range(0, len(tickers), BATCH_TICKER_SIZE):
            batch_tickers = tickers[i:i + BATCH_TICKER_SIZE]
            batch_num = i // BATCH_TICKER_SIZE + 1
            
            self.logger.info(f"Processing batch {batch_num} with {len(batch_tickers)} tickers")
            
            batch_result = self.run_single_iteration(batch_tickers)
            batch_stats['batch_results'].append(batch_result)
            batch_stats['total_batches'] += 1
            
            if batch_result['success']:
                batch_stats['successful_batches'] += 1
            
            batch_stats['total_articles'] += batch_result['articles_fetched']
            
            # Delay between batches (except for the last one)
            if i + BATCH_TICKER_SIZE < len(tickers):
                self.logger.debug(f"Waiting {BATCH_DELAY} seconds before next batch...")
                time.sleep(BATCH_DELAY)
        
        batch_stats['end_time'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Batch processing completed: {batch_stats['successful_batches']}/{batch_stats['total_batches']} successful batches, "
                        f"{batch_stats['total_articles']} total articles")
        
        return batch_stats
    
    def health_check(self) -> Dict:
        """
        Perform a health check of all pipeline components
        """
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {},
            'consecutive_failures': self.consecutive_failures,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'pipeline_stats': self.pipeline_stats.copy()
        }
        
        try:
            # Check news fetcher
            if self.news_fetcher:
                fetcher_stats = self.news_fetcher.get_stats()
                health_status['components']['news_fetcher'] = {
                    'status': 'healthy',
                    'stats': fetcher_stats
                }
            else:
                health_status['components']['news_fetcher'] = {'status': 'not_initialized'}
                health_status['overall_status'] = 'unhealthy'
            
            # Check sentiment analyzer
            if self.sentiment_analyzer:
                analyzer_info = self.sentiment_analyzer.get_model_info()
                health_status['components']['sentiment_analyzer'] = {
                    'status': 'healthy' if analyzer_info['model_loaded'] else 'unhealthy',
                    'info': analyzer_info
                }
                if not analyzer_info['model_loaded']:
                    health_status['overall_status'] = 'unhealthy'
            else:
                health_status['components']['sentiment_analyzer'] = {'status': 'not_initialized'}
                health_status['overall_status'] = 'unhealthy'
            
            # Check data persistence
            if self.data_persistence:
                health_status['components']['data_persistence'] = {'status': 'healthy'}
            else:
                health_status['components']['data_persistence'] = {'status': 'not_initialized'}
                health_status['overall_status'] = 'unhealthy'
            
            # Check for too many consecutive failures
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                health_status['overall_status'] = 'unhealthy'
                health_status['alert'] = f"Too many consecutive failures: {self.consecutive_failures}"
            
        except Exception as e:
            health_status['overall_status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_recent_summary(self, hours: int = 24) -> Dict:
        """
        Get a summary of recent activity and results
        """
        try:
            summary = {
                'period_hours': hours,
                'timestamp': datetime.utcnow().isoformat(),
                'ticker_summaries': {},
                'overall_stats': {
                    'total_articles': 0,
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
                }
            }
            
            # Get summaries for each ticker
            for ticker in NSE_TICKERS:
                ticker_summary = self.data_persistence.get_ticker_summary(ticker, hours)
                summary['ticker_summaries'][ticker] = ticker_summary
                
                # Add to overall stats
                summary['overall_stats']['total_articles'] += ticker_summary['total_articles']
                for sentiment, count in ticker_summary['sentiment_distribution'].items():
                    if sentiment in summary['overall_stats']['sentiment_distribution']:
                        summary['overall_stats']['sentiment_distribution'][sentiment] += count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate recent summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days: int = 30) -> Dict:
        """
        Clean up old data to prevent storage from growing too large
        """
        try:
            self.logger.info(f"Cleaning up data older than {days} days...")
            
            # Clean up stored records
            deleted_records = self.data_persistence.cleanup_old_records(days)
            
            # Clean up seen links
            if self.news_fetcher:
                self.news_fetcher.cleanup_old_links()
            
            cleanup_stats = {
                'deleted_records': deleted_records,
                'cleanup_date': datetime.utcnow().isoformat(),
                'days_threshold': days
            }
            
            self.logger.info(f"Cleanup completed: {deleted_records} records deleted")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    pipeline = NewsSentimentPipeline()
    
    # Test with a few tickers
    test_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    result = pipeline.run_single_iteration(test_tickers)
    
    print("Pipeline iteration result:")
    print(f"Success: {result['success']}")
    print(f"Articles fetched: {result['articles_fetched']}")
    print(f"Articles analyzed: {result['articles_analyzed']}")
    print(f"Articles saved: {result['articles_saved']}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    # Test health check
    health = pipeline.health_check()
    print(f"\nPipeline health: {health['overall_status']}") 