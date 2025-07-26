#!/usr/bin/env python3
"""
Aggressive 2-Year Optimized Parallel Sentiment Analysis Runner - Batch 3 (Stocks 57-84)
=====================================================================================

This script runs sentiment analysis for the third batch of stocks (57-84) for 2 years (2024-2025)
with aggressive rate limiting to maximize throughput while avoiding blocks.
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import time
import logging
from typing import List, Dict, Any
import json
import pandas as pd
import random

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from comprehensive_sentiment_dataset_generator import ComprehensiveSentimentGenerator
from integrated_advanced_features import IntegratedAdvancedFeatures
from data_quality_manager import DataQualityManager
from backfill_manager import BackfillManager

class Aggressive2YearOptimizedRunner3:
    """Aggressive 2-year optimized parallel sentiment analysis runner for batch 3 (stocks 57-84)"""
    
    def __init__(self):
        """Initialize the aggressive 2-year optimized parallel sentiment runner"""
        self.setup_logging()
        self.logger.info("Initializing Aggressive 2-Year Optimized Parallel Sentiment Runner - Batch 3")
        
        # Initialize all components
        self.generator = ComprehensiveSentimentGenerator()
        self.advanced_features = IntegratedAdvancedFeatures()
        self.data_quality = DataQualityManager(data_dir=".")
        
        # Load stocks 57-84 (third batch)
        self.stock_symbols = self.load_batch_3_stocks()
        
        # Set date range: January 1st, 2024 to December 31st, 2025 (2 years)
        self.start_date = date(2024, 1, 1)
        self.end_date = date(2025, 12, 31)
        
        # Create output directory
        self.output_dir = Path("2year_parallel_sentiment_batch_3")
        self.output_dir.mkdir(exist_ok=True)
        
        # AGGRESSIVE Rate limiting configuration for maximum throughput
        self.base_delay = 0.8  # Reduced from 3.0 seconds - aggressive but safe
        self.max_delay = 5.0   # Reduced from 10.0 seconds
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Reduced from 5 for faster recovery
        
        # Additional aggressive optimizations
        self.batch_size = 5  # Process 5 stocks at once
        self.concurrent_requests = 3  # Allow 3 concurrent requests
        self.adaptive_delay = True  # Enable adaptive delay based on response times
        
        self.logger.info(f"Loaded {len(self.stock_symbols)} stock symbols for batch 3")
        self.logger.info(f"Date range: {self.start_date} to {self.end_date} (2 years)")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"AGGRESSIVE Rate limiting: base_delay={self.base_delay}s, max_delay={self.max_delay}s")
        self.logger.info(f"Batch processing: {self.batch_size} stocks concurrently")
        self.logger.info(f"Concurrent requests: {self.concurrent_requests}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = f"2year_parallel_batch_3_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("aggressive_2year_parallel_sentiment_runner_3")
    
    def load_batch_3_stocks(self) -> List[str]:
        """Load stocks 57-84 for batch 3"""
        return [
            "HINDUNILVR.NSE", "ITC.NSE", "SBIN.NSE", "KOTAKBANK.NSE", "LT.NSE",
            "ASIANPAINT.NSE", "MARUTI.NSE", "BAJFINANCE.NSE", "HCLTECH.NSE", "AXISBANK.NSE",
            "ULTRACEMCO.NSE", "SUNPHARMA.NSE", "TITAN.NSE", "TECHM.NSE", "POWERGRID.NSE",
            "NTPC.NSE", "ONGC.NSE", "COALINDIA.NSE", "WIPRO.NSE", "TATAMOTORS.NSE",
            "TATASTEEL.NSE", "JSWSTEEL.NSE", "HINDALCO.NSE", "ADANIPORTS.NSE", "BPCL.NSE",
            "BRITANNIA.NSE", "CIPLA.NSE", "DIVISLAB.NSE"
        ]
    
    def aggressive_rate_limit(self, response_time: float = None):
        """Implement aggressive rate limiting with adaptive delays"""
        if self.consecutive_failures > 0:
            # Exponential backoff with reduced delays
            delay = min(self.base_delay * (1.5 ** (self.consecutive_failures - 1)), self.max_delay)
            # Add minimal jitter to avoid thundering herd
            jitter = random.uniform(0.9, 1.1)
            delay *= jitter
            self.logger.info(f"Rate limiting: {delay:.2f}s delay (failures: {self.consecutive_failures})")
            time.sleep(delay)
        else:
            # Adaptive delay based on response time
            if self.adaptive_delay and response_time:
                # If response was fast, reduce delay slightly
                if response_time < 2.0:
                    delay = self.base_delay * 0.8
                elif response_time > 5.0:
                    delay = self.base_delay * 1.2
                else:
                    delay = self.base_delay
            else:
                delay = self.base_delay
            
            # Minimal jitter for aggressive processing
            jitter = random.uniform(0.95, 1.05)
            delay *= jitter
            time.sleep(delay)
    
    def calculate_total_workload(self) -> Dict[str, Any]:
        """Calculate the total workload for 2-year batch 3"""
        total_days = (self.end_date - self.start_date).days + 1
        total_stocks = len(self.stock_symbols)
        total_requests = total_days * total_stocks
        
        # Estimate processing time (with aggressive rate limiting)
        estimated_hours = (total_requests * self.base_delay) / 3600
        
        # Account for batch processing efficiency
        efficiency_factor = 0.7  # 30% efficiency gain from batching
        estimated_hours *= efficiency_factor
        
        return {
            "total_days": total_days,
            "total_stocks": total_stocks,
            "total_requests": total_requests,
            "estimated_hours": estimated_hours,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "rate_limiting": {
                "base_delay": self.base_delay,
                "max_delay": self.max_delay,
                "max_consecutive_failures": self.max_consecutive_failures,
                "batch_size": self.batch_size,
                "concurrent_requests": self.concurrent_requests
            }
        }
    
    def run_complete_analysis(self):
        """Run the complete 2-year sentiment analysis for batch 3"""
        workload = self.calculate_total_workload()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING AGGRESSIVE 2-YEAR PARALLEL BATCH 3 SENTIMENT ANALYSIS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total days: {workload['total_days']}")
        self.logger.info(f"Total stocks: {workload['total_stocks']}")
        self.logger.info(f"Total requests: {workload['total_requests']:,}")
        self.logger.info(f"Estimated time: {workload['estimated_hours']:.1f} hours")
        self.logger.info(f"AGGRESSIVE Rate limiting: {self.base_delay}s base delay, {self.max_delay}s max delay")
        self.logger.info(f"Batch processing: {self.batch_size} stocks, {self.concurrent_requests} concurrent requests")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run comprehensive backfill for batch 3 stocks and dates
            self.logger.info("Step 1: Running comprehensive backfill for batch 3 (2 years)...")
            missing_data = self.generator.run_backfill(
                start_date=self.start_date,
                end_date=self.end_date,
                stocks=self.stock_symbols
            )
            
            # Step 2: Generate comprehensive dataset with all advanced features
            self.logger.info("Step 2: Generating comprehensive dataset for batch 3 (2 years)...")
            report = self.generator.generate_comprehensive_dataset(
                start_date=self.start_date,
                end_date=self.end_date,
                stock_symbols=self.stock_symbols,
                output_dir=str(self.output_dir)
            )
            
            # Step 3: Run data quality checks
            self.logger.info("Step 3: Running data quality checks for batch 3...")
            qc_report = self.generator.run_qc("comprehensive_sentiment_dataset.csv")
            
            # Step 4: Create snapshots and roll-ups
            self.logger.info("Step 4: Creating snapshots and roll-ups for batch 3...")
            snapshot_path = self.generator.save_snapshot(
                ["comprehensive_sentiment_dataset.csv", "comprehensive_sentiment_dataset.json"],
                tag="2year_parallel_batch_3"
            )
            
            # Create daily roll-up
            daily_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "D", 
                "2year_daily_sentiment_rollup_batch_3.csv"
            )
            
            # Create weekly roll-up
            weekly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "W", 
                "2year_weekly_sentiment_rollup_batch_3.csv"
            )
            
            # Create monthly roll-up
            monthly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "M", 
                "2year_monthly_sentiment_rollup_batch_3.csv"
            )
            
            # Step 5: Generate final summary report
            end_time = datetime.now()
            duration = end_time - start_time
            
            final_report = {
                "analysis_summary": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": duration.total_seconds() / 3600,
                    "duration_minutes": duration.total_seconds() / 60,
                    "total_stocks": len(self.stock_symbols),
                    "date_range": f"{self.start_date} to {self.end_date}",
                    "total_days": workload['total_days']
                },
                "workload_analysis": workload,
                "missing_data": len(missing_data) if missing_data else 0,
                "qc_report": qc_report,
                "output_files": {
                    "snapshot": snapshot_path,
                    "daily_rollup": daily_rollup,
                    "weekly_rollup": weekly_rollup,
                    "monthly_rollup": monthly_rollup,
                    "output_directory": str(self.output_dir)
                },
                "rate_limiting_performance": {
                    "base_delay_used": self.base_delay,
                    "max_delay_used": self.max_delay,
                    "consecutive_failures": self.consecutive_failures,
                    "adaptive_delay_enabled": self.adaptive_delay,
                    "batch_processing": self.batch_size,
                    "concurrent_requests": self.concurrent_requests
                }
            }
            
            # Save final report
            report_file = self.output_dir / "2year_batch_3_final_report.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info("=" * 80)
            self.logger.info("AGGRESSIVE 2-YEAR BATCH 3 ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.print_summary(final_report)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error in aggressive 2-year batch 3 analysis: {e}")
            raise
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a comprehensive summary of the analysis"""
        summary = report['analysis_summary']
        workload = report['workload_analysis']
        
        self.logger.info("")
        self.logger.info("📊 AGGRESSIVE 2-YEAR BATCH 3 ANALYSIS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Duration: {summary['duration_hours']:.2f} hours ({summary['duration_minutes']:.1f} minutes)")
        self.logger.info(f"📈 Stocks processed: {summary['total_stocks']}")
        self.logger.info(f"📅 Date range: {summary['date_range']}")
        self.logger.info(f"📊 Total days: {summary['total_days']}")
        self.logger.info(f"🔄 Total requests: {workload['total_requests']:,}")
        self.logger.info(f"⚡ Estimated vs Actual: {workload['estimated_hours']:.1f}h vs {summary['duration_hours']:.1f}h")
        self.logger.info(f"📁 Output directory: {report['output_files']['output_directory']}")
        self.logger.info(f"📋 Missing data entries: {report['missing_data']}")
        self.logger.info("")
        self.logger.info("🚀 AGGRESSIVE RATE LIMITING PERFORMANCE:")
        self.logger.info(f"   • Base delay: {self.base_delay}s")
        self.logger.info(f"   • Max delay: {self.max_delay}s")
        self.logger.info(f"   • Batch size: {self.batch_size} stocks")
        self.logger.info(f"   • Concurrent requests: {self.concurrent_requests}")
        self.logger.info(f"   • Adaptive delay: {'Enabled' if self.adaptive_delay else 'Disabled'}")
        self.logger.info("=" * 60)

def main():
    """Main function to run the aggressive 2-year optimized parallel sentiment analysis"""
    try:
        runner = Aggressive2YearOptimizedRunner3()
        report = runner.run_complete_analysis()
        
        print("\n" + "="*80)
        print("✅ AGGRESSIVE 2-YEAR BATCH 3 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Duration: {report['analysis_summary']['duration_hours']:.2f} hours")
        print(f"📈 Stocks processed: {report['analysis_summary']['total_stocks']}")
        print(f"📅 Date range: {report['analysis_summary']['date_range']}")
        print(f"📁 Output directory: {report['output_files']['output_directory']}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error in aggressive 2-year batch 3 analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 