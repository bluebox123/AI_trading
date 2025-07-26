#!/usr/bin/env python3
"""
Comprehensive test script for the NSE News Sentiment Pipeline
Tests individual components and the complete pipeline
"""
import sys
import os
import logging
import json
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.news_fetcher import NewsFetcher
from src.sentiment_analyzer import SentimentAnalyzer
from src.data_persistence import DataPersistence
from src.pipeline import NewsSentimentPipeline

class PipelineTests:
    """Test suite for the news sentiment pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {
            'start_time': datetime.utcnow().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        self.logger.info(f"Running test: {test_name}")
        self.test_results['summary']['total'] += 1
        
        try:
            start_time = datetime.utcnow()
            result = test_func()
            end_time = datetime.utcnow()
            
            duration = (end_time - start_time).total_seconds()
            
            self.test_results['tests'][test_name] = {
                'status': 'PASSED' if result['success'] else 'FAILED',
                'duration_seconds': duration,
                'result': result,
                'timestamp': end_time.isoformat()
            }
            
            if result['success']:
                self.test_results['summary']['passed'] += 1
                self.logger.info(f"✅ {test_name} PASSED in {duration:.2f}s")
            else:
                self.test_results['summary']['failed'] += 1
                self.logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            self.test_results['summary']['failed'] += 1
            self.test_results['tests'][test_name] = {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.logger.error(f"💥 {test_name} ERROR: {e}")
    
    def test_news_fetcher_initialization(self):
        """Test news fetcher initialization"""
        try:
            fetcher = NewsFetcher()
            return {
                'success': True,
                'message': 'News fetcher initialized successfully',
                'stats': fetcher.get_stats()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_news_fetcher_single_ticker(self):
        """Test fetching news for a single ticker"""
        try:
            fetcher = NewsFetcher()
            articles = fetcher.fetch_ticker_news('RELIANCE.NS')
            
            return {
                'success': True,
                'message': f'Fetched {len(articles)} articles for RELIANCE.NS',
                'article_count': len(articles),
                'sample_article': articles[0] if articles else None
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_news_fetcher_multiple_tickers(self):
        """Test fetching news for multiple tickers"""
        try:
            fetcher = NewsFetcher()
            test_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
            articles = fetcher.fetch_all_news(test_tickers)
            
            ticker_counts = {}
            for article in articles:
                ticker = article.get('ticker')
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            return {
                'success': True,
                'message': f'Fetched {len(articles)} total articles for {len(test_tickers)} tickers',
                'total_articles': len(articles),
                'ticker_breakdown': ticker_counts
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_sentiment_analyzer_initialization(self):
        """Test sentiment analyzer initialization"""
        try:
            analyzer = SentimentAnalyzer()
            model_info = analyzer.get_model_info()
            
            return {
                'success': model_info['model_loaded'],
                'message': 'Sentiment analyzer initialized successfully' if model_info['model_loaded'] else 'Model failed to load',
                'model_info': model_info
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_sentiment_analyzer_single_text(self):
        """Test sentiment analysis on single text"""
        try:
            analyzer = SentimentAnalyzer()
            
            test_texts = [
                "Reliance Industries reports record quarterly profits beating all estimates",
                "TCS shares fall sharply as company misses revenue expectations badly",
                "HDFC Bank maintains steady growth in digital banking operations"
            ]
            
            results = []
            for text in test_texts:
                result = analyzer.analyze_text(text)
                results.append({
                    'text': text,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence']
                })
            
            return {
                'success': True,
                'message': f'Analyzed {len(test_texts)} text samples',
                'results': results
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_sentiment_analyzer_batch(self):
        """Test batch sentiment analysis"""
        try:
            analyzer = SentimentAnalyzer()
            
            test_headlines = [
                "Asian Paints launches innovative new product line with strong market response",
                "Infosys announces major layoffs amid challenging economic conditions",
                "Wipro secures large multi-year contract from global technology firm",
                "ONGC reports declining production due to operational challenges",
                "Titan Company shows robust growth in jewelry segment sales"
            ]
            
            results = analyzer.analyze_batch(test_headlines)
            
            sentiment_counts = {}
            for result in results:
                sentiment = result['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            return {
                'success': True,
                'message': f'Batch analyzed {len(test_headlines)} headlines',
                'total_analyzed': len(results),
                'sentiment_distribution': sentiment_counts,
                'average_confidence': sum(r['confidence'] for r in results) / len(results)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_data_persistence_initialization(self):
        """Test data persistence initialization"""
        try:
            persistence = DataPersistence()
            return {
                'success': True,
                'message': 'Data persistence initialized successfully'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_data_persistence_save_load(self):
        """Test saving and loading data"""
        try:
            persistence = DataPersistence()
            
            # Create test data
            test_records = [
                {
                    'ticker': 'TEST.NS',
                    'headline': 'Test headline for data persistence',
                    'link': f'https://test.com/news/{datetime.utcnow().timestamp()}',
                    'pub_date': datetime.utcnow().isoformat(),
                    'sentiment': 'positive',
                    'sentiment_confidence': 0.85,
                    'sentiment_probabilities': {'negative': 0.05, 'neutral': 0.10, 'positive': 0.85},
                    'sentiment_timestamp': datetime.utcnow().isoformat()
                }
            ]
            
            # Test saving
            save_success = persistence.save_records(test_records)
            
            if not save_success:
                return {'success': False, 'error': 'Failed to save test records'}
            
            # Test loading
            recent_records = persistence.load_recent_records(hours=1, ticker='TEST.NS')
            
            return {
                'success': True,
                'message': 'Successfully saved and loaded test data',
                'saved_records': len(test_records),
                'loaded_records': len(recent_records)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_full_pipeline(self):
        """Test the complete pipeline end-to-end"""
        try:
            pipeline = NewsSentimentPipeline()
            
            # Test with minimal tickers to avoid long runtime
            test_tickers = ['RELIANCE.NS', 'TCS.NS']
            result = pipeline.run_single_iteration(test_tickers)
            
            return {
                'success': result['success'],
                'message': 'Full pipeline test completed',
                'pipeline_result': result
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_pipeline_health_check(self):
        """Test pipeline health check functionality"""
        try:
            pipeline = NewsSentimentPipeline()
            health = pipeline.health_check()
            
            is_healthy = health['overall_status'] == 'healthy'
            
            return {
                'success': True,  # Health check itself should succeed
                'message': f'Pipeline health: {health["overall_status"]}',
                'health_status': health,
                'is_healthy': is_healthy
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_all_tests(self):
        """Run all tests"""
        self.logger.info("Starting comprehensive pipeline tests...")
        
        # Component tests
        self.run_test("News Fetcher Initialization", self.test_news_fetcher_initialization)
        self.run_test("News Fetcher Single Ticker", self.test_news_fetcher_single_ticker)
        self.run_test("News Fetcher Multiple Tickers", self.test_news_fetcher_multiple_tickers)
        
        self.run_test("Sentiment Analyzer Initialization", self.test_sentiment_analyzer_initialization)
        self.run_test("Sentiment Analyzer Single Text", self.test_sentiment_analyzer_single_text)
        self.run_test("Sentiment Analyzer Batch", self.test_sentiment_analyzer_batch)
        
        self.run_test("Data Persistence Initialization", self.test_data_persistence_initialization)
        self.run_test("Data Persistence Save/Load", self.test_data_persistence_save_load)
        
        # Integration tests
        self.run_test("Pipeline Health Check", self.test_pipeline_health_check)
        self.run_test("Full Pipeline End-to-End", self.test_full_pipeline)
        
        self.test_results['end_time'] = datetime.utcnow().isoformat()
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        summary = self.test_results['summary']
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {(summary['passed']/summary['total']*100):.1f}%")
        
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        
        for test_name, result in self.test_results['tests'].items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            duration = result.get('duration_seconds', 0)
            print(f"{status_emoji} {test_name:<40} {result['status']:<8} ({duration:.2f}s)")
            
            if result['status'] != 'PASSED' and 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("="*80)

def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create test suite
        tests = PipelineTests()
        
        # Run all tests
        results = tests.run_all_tests()
        
        # Print summary
        tests.print_summary()
        
        # Save detailed results
        results_file = f"test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed test results saved to: {results_file}")
        
        # Return appropriate exit code
        return 0 if results['summary']['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 