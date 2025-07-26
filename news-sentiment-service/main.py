#!/usr/bin/env python3
"""
Main entry point for the NSE News Sentiment Pipeline
Supports multiple run modes: scheduled, single run, and test mode
"""
import argparse
import logging
import os
import sys
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scheduler import NewsSentimentScheduler
from src.pipeline import NewsSentimentPipeline
from config.config import (
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, INFO_LOG_FILE, ERROR_LOG_FILE,
    DEBUG_MODE, DRY_RUN, NSE_TICKERS
)

def setup_logging(log_level: str = LOG_LEVEL, verbose: bool = False):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(INFO_LOG_FILE), exist_ok=True)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    if verbose:
        level = logging.DEBUG
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for info logs
    info_handler = logging.FileHandler(INFO_LOG_FILE)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    root_logger.addHandler(info_handler)
    
    # File handler for error logs
    error_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

def run_scheduled_mode():
    """Run the pipeline in scheduled mode (continuous operation)"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting News Sentiment Pipeline in scheduled mode")
        logger.info(f"Debug mode: {DEBUG_MODE}, Dry run: {DRY_RUN}")
        
        scheduler = NewsSentimentScheduler()
        scheduler.run_forever()
        
    except Exception as e:
        logger.error(f"Failed to run in scheduled mode: {e}")
        sys.exit(1)

def run_single_mode(tickers: list = None, output_file: str = None):
    """Run the pipeline once and exit"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting News Sentiment Pipeline in single run mode")
        
        if tickers:
            logger.info(f"Processing specific tickers: {tickers}")
        else:
            logger.info(f"Processing all configured tickers ({len(NSE_TICKERS)} total)")
        
        pipeline = NewsSentimentPipeline()
        result = pipeline.run_single_iteration(tickers)
        
        # Print results
        print("\n" + "="*60)
        print("PIPELINE EXECUTION RESULTS")
        print("="*60)
        print(f"Success: {'YES' if result['success'] else 'NO'}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        print(f"Articles fetched: {result['articles_fetched']}")
        print(f"Articles analyzed: {result['articles_analyzed']}")
        print(f"Articles saved: {result['articles_saved']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to: {output_file}")
        
        print("="*60)
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        logger.error(f"Failed to run in single mode: {e}")
        return 1

def run_test_mode():
    """Run in test mode with a small subset of tickers"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting News Sentiment Pipeline in test mode")
        
        # Use only 3 tickers for testing
        test_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        pipeline = NewsSentimentPipeline()
        
        # Test pipeline health first
        health = pipeline.health_check()
        print("\n" + "="*60)
        print("HEALTH CHECK RESULTS")
        print("="*60)
        print(f"Overall Status: {health['overall_status']}")
        
        for component, status in health.get('components', {}).items():
            print(f"{component}: {status.get('status', 'unknown')}")
        
        if health['overall_status'] != 'healthy':
            print("⚠️  Pipeline is not healthy, test results may be affected")
        
        # Run test iteration
        print("\n" + "="*60)
        print("TEST EXECUTION RESULTS")
        print("="*60)
        
        result = pipeline.run_single_iteration(test_tickers)
        
        print(f"Success: {'YES' if result['success'] else 'NO'}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        print(f"Articles fetched: {result['articles_fetched']}")
        print(f"Articles analyzed: {result['articles_analyzed']}")
        print(f"Articles saved: {result['articles_saved']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        print("="*60)
        
        return 0 if result['success'] else 1
        
    except Exception as e:
        logger.error(f"Failed to run in test mode: {e}")
        return 1

def run_status_mode():
    """Show current status and recent summary"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Checking pipeline status")
        
        pipeline = NewsSentimentPipeline()
        
        # Get health status
        health = pipeline.health_check()
        
        # Get recent summary
        summary = pipeline.get_recent_summary(hours=24)
        
        print("\n" + "="*60)
        print("PIPELINE STATUS")
        print("="*60)
        print(f"Overall Health: {health['overall_status']}")
        print(f"Last Run: {health.get('last_run_time', 'Never')}")
        print(f"Consecutive Failures: {health.get('consecutive_failures', 0)}")
        
        print("\nComponent Status:")
        for component, status in health.get('components', {}).items():
            print(f"  {component}: {status.get('status', 'unknown')}")
        
        print("\n" + "="*60)
        print("RECENT ACTIVITY (24 HOURS)")
        print("="*60)
        
        if 'error' not in summary:
            total_articles = summary['overall_stats']['total_articles']
            sentiment_dist = summary['overall_stats']['sentiment_distribution']
            
            print(f"Total Articles: {total_articles}")
            print(f"Positive Sentiment: {sentiment_dist.get('positive', 0)}")
            print(f"Neutral Sentiment: {sentiment_dist.get('neutral', 0)}")
            print(f"Negative Sentiment: {sentiment_dist.get('negative', 0)}")
            
            # Show top active tickers
            ticker_counts = [(ticker, data['total_articles']) 
                           for ticker, data in summary['ticker_summaries'].items() 
                           if data['total_articles'] > 0]
            ticker_counts.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nMost Active Tickers:")
            for ticker, count in ticker_counts[:10]:
                print(f"  {ticker}: {count} articles")
        else:
            print(f"Error getting summary: {summary['error']}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return 1

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="NSE News Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run in scheduled mode
  python main.py --mode single                # Run once and exit
  python main.py --mode test                  # Run test with 3 tickers
  python main.py --mode single --tickers RELIANCE.NS TCS.NS  # Specific tickers
  python main.py --mode status                # Show current status
  python main.py --mode single --output results.json  # Save results to file
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['scheduled', 'single', 'test', 'status'],
        default='scheduled',
        help='Run mode (default: scheduled)'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to process (only for single mode)'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for results (only for single mode)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=LOG_LEVEL,
        help=f'Set log level (default: {LOG_LEVEL})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving data'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    
    if args.debug:
        os.environ['DEBUG'] = 'true'
    
    # Setup logging
    setup_logging(args.log_level, args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting NSE News Sentiment Pipeline - Mode: {args.mode}")
    
    # Run based on mode
    if args.mode == 'scheduled':
        return run_scheduled_mode()
    elif args.mode == 'single':
        return run_single_mode(args.tickers, args.output)
    elif args.mode == 'test':
        return run_test_mode()
    elif args.mode == 'status':
        return run_status_mode()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 