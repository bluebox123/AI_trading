#!/usr/bin/env python3
"""
Backfill Manager for Sentiment Pipeline
======================================

Handles robust historical backfill of sentiment data from RSS feeds.
- Iterates over all dates and stocks
- Logs missing data
- CLI support for triggering backfill
"""

import os
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict

class BackfillManager:
    def __init__(self, fetch_func, stock_symbols: List[str], log_file: str = "backfill.log"):
        self.fetch_func = fetch_func  # Function to fetch news for a stock/date
        self.stock_symbols = stock_symbols
        self.logger = self._setup_logging(log_file)

    def _setup_logging(self, log_file: str):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("backfill")

    def backfill(self, start_date: date, end_date: date, output_dir: str = "backfill_data"):
        Path(output_dir).mkdir(exist_ok=True)
        missing_data = []
        for stock in self.stock_symbols:
            current_date = start_date
            while current_date <= end_date:
                try:
                    articles = self.fetch_func(stock, current_date, current_date)
                    if not articles:
                        self.logger.warning(f"No data for {stock} on {current_date}")
                        missing_data.append({"stock": stock, "date": str(current_date)})
                    else:
                        # Save articles
                        fname = f"{stock}_{current_date}.json"
                        with open(Path(output_dir) / fname, "w") as f:
                            json.dump(articles, f, indent=2)
                        self.logger.info(f"Fetched {len(articles)} for {stock} on {current_date}")
                except Exception as e:
                    self.logger.error(f"Error fetching {stock} on {current_date}: {e}")
                    missing_data.append({"stock": stock, "date": str(current_date), "error": str(e)})
                current_date += timedelta(days=1)
        # Save missing data log
        with open(Path(output_dir) / "missing_data_log.json", "w") as f:
            json.dump(missing_data, f, indent=2)
        self.logger.info(f"Backfill complete. Missing: {len(missing_data)} entries.")
        return missing_data
    
    def identify_missing_data(self, start_date: date, end_date: date, stocks: List[str] = None) -> List[Dict]:
        """Identify missing data points for the given date range and stocks"""
        if stocks is None:
            stocks = self.stock_symbols
        
        missing_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for stock in stocks:
                # Check if data exists for this stock/date combination
                # This is a simplified check - in practice, you'd check actual data files
                data_exists = False
                
                # For testing purposes, we'll simulate missing data
                # In a real implementation, you'd check if the data file exists
                if current_date.weekday() == 5 or current_date.weekday() == 6:  # Weekend
                    data_exists = False
                else:
                    data_exists = True  # Assume data exists on weekdays for testing
                
                if not data_exists:
                    missing_data.append({
                        "stock": stock,
                        "date": str(current_date),
                        "reason": "weekend" if current_date.weekday() in [5, 6] else "no_data"
                    })
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"Identified {len(missing_data)} missing data points")
        return missing_data

if __name__ == "__main__":
    # Example CLI usage
    import sys
    from comprehensive_sentiment_dataset_generator import ComprehensiveSentimentGenerator
    from datetime import datetime
    
    if len(sys.argv) < 4:
        print("Usage: python backfill_manager.py START_DATE END_DATE [STOCK1,STOCK2,...]")
        print("Example: python backfill_manager.py 2024-01-01 2024-01-31 RELIANCE.NSE,TCS.NSE")
        sys.exit(1)
    start_date = date.fromisoformat(sys.argv[1])
    end_date = date.fromisoformat(sys.argv[2])
    stocks = sys.argv[3].split(",")
    gen = ComprehensiveSentimentGenerator()
    mgr = BackfillManager(gen.fetch_news_for_stock, stocks)
    mgr.backfill(start_date, end_date) 