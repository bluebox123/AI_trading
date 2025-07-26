#!/usr/bin/env python3
"""
Complete 18-Month Sentiment Analysis Runner - BATCH 1
====================================================

Parallel processing script for BATCH 1 (40 stocks):
- Stocks 1-40: RELIANCE.NSE to BERGEPAINT.NSE
- 18 months: January 2nd, 2024 to July 7th, 2025
- Intraday sentiment data (every single day)
- All advanced features enabled
- Independent parallel processing

This is BATCH 1 of 3 parallel scripts for faster processing.
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

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from comprehensive_sentiment_dataset_generator import ComprehensiveSentimentGenerator
from integrated_advanced_features import IntegratedAdvancedFeatures
from data_quality_manager import DataQualityManager
from backfill_manager import BackfillManager

class CompleteSentimentRunnerBatch1:
    """Complete 18-month sentiment analysis runner - BATCH 1"""
    
    def __init__(self):
        """Initialize the complete sentiment runner for batch 1"""
        self.batch_id = 1
        self.setup_logging()
        self.logger.info("Initializing Complete Sentiment Runner - BATCH 1")
        
        # Initialize all components
        self.generator = ComprehensiveSentimentGenerator()
        self.advanced_features = IntegratedAdvancedFeatures()
        self.data_quality = DataQualityManager(data_dir=".")
        
        # Load BATCH 1 stock symbols (stocks 1-40)
        self.stock_symbols = self.load_batch1_stock_symbols()
        
        # Set date range: January 2nd, 2024 to July 7th, 2025 (18 months)
        self.start_date = date(2024, 1, 2)
        self.end_date = date(2025, 7, 7)
        
        # Create batch-specific output directory
        self.output_dir = Path("complete_18month_sentiment_dataset_batch1")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Loaded {len(self.stock_symbols)} stock symbols for BATCH 1")
        self.logger.info(f"Date range: {self.start_date} to {self.end_date}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging for batch 1"""
        log_file = f"complete_18month_sentiment_batch1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("complete_sentiment_runner_batch1")
    
    def load_batch1_stock_symbols(self) -> List[str]:
        """Load BATCH 1 stock symbols (stocks 1-40)"""
        # BATCH 1: Stocks 1-40
        batch1_stocks = [
            "RELIANCE.NSE", "ABCAPITAL.NSE", "ABFRL.NSE", "ACC.NSE", "ADANIENT.NSE",
            "ADANIPORTS.NSE", "ALKEM.NSE", "AMBER.NSE", "APOLLOHOSP.NSE", "ASIANPAINT.NSE",
            "ASTRAL.NSE", "AUBANK.NSE", "AUROPHARMA.NSE", "AXISBANK.NSE", "BAJAJ-AUTO.NSE",
            "BAJFINANCE.NSE", "BALKRISIND.NSE", "BANDHANBNK.NSE", "BANKBARODA.NSE", "BATAINDIA.NSE",
            "BERGEPAINT.NSE", "BHARTIARTL.NSE", "BIOCON.NSE", "BOSCHLTD.NSE", "BPCL.NSE",
            "BRITANNIA.NSE", "CADILAHC.NSE", "CANFINHOME.NSE", "CIPLA.NSE", "COALINDIA.NSE",
            "COFORGE.NSE", "COLPAL.NSE", "CROMPTON.NSE", "CUMMINSIND.NSE", "DABUR.NSE",
            "DIXON.NSE", "DIVISLAB.NSE", "DLF.NSE", "DMART.NSE", "DRREDDY.NSE"
        ]
        
        self.logger.info(f"Loaded {len(batch1_stocks)} stock symbols for BATCH 1")
        return batch1_stocks
    
    def calculate_total_workload(self) -> Dict[str, Any]:
        """Calculate the total workload for batch 1"""
        total_days = (self.end_date - self.start_date).days + 1
        total_stocks = len(self.stock_symbols)
        total_requests = total_days * total_stocks
        
        # Estimate processing time (with rate limiting)
        estimated_hours = (total_requests * 1.5) / 3600  # 1.5 seconds per request
        
        return {
            "batch_id": self.batch_id,
            "total_days": total_days,
            "total_stocks": total_stocks,
            "total_requests": total_requests,
            "estimated_hours": estimated_hours,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat()
        }
    
    def run_complete_analysis(self):
        """Run the complete 18-month sentiment analysis for batch 1"""
        workload = self.calculate_total_workload()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE 18-MONTH SENTIMENT ANALYSIS - BATCH 1")
        self.logger.info("=" * 80)
        self.logger.info(f"Batch ID: {workload['batch_id']}")
        self.logger.info(f"Total days: {workload['total_days']}")
        self.logger.info(f"Total stocks: {workload['total_stocks']}")
        self.logger.info(f"Total requests: {workload['total_requests']:,}")
        self.logger.info(f"Estimated time: {workload['estimated_hours']:.1f} hours")
        self.logger.info(f"Stock range: {self.stock_symbols[0]} to {self.stock_symbols[-1]}")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run comprehensive backfill for batch 1 stocks
            self.logger.info("Step 1: Running comprehensive backfill for BATCH 1...")
            missing_data = self.generator.run_backfill(
                start_date=self.start_date,
                end_date=self.end_date,
                stocks=self.stock_symbols
            )
            
            # Step 2: Generate comprehensive dataset with all advanced features
            self.logger.info("Step 2: Generating comprehensive dataset for BATCH 1...")
            report = self.generator.generate_comprehensive_dataset(
                start_date=self.start_date,
                end_date=self.end_date,
                stock_symbols=self.stock_symbols,
                output_dir=str(self.output_dir)
            )
            
            # Step 3: Run data quality checks
            self.logger.info("Step 3: Running data quality checks for BATCH 1...")
            qc_report = self.generator.run_qc("comprehensive_sentiment_dataset.csv")
            
            # Step 4: Create snapshots and roll-ups
            self.logger.info("Step 4: Creating snapshots and roll-ups for BATCH 1...")
            snapshot_path = self.generator.save_snapshot(
                ["comprehensive_sentiment_dataset.csv", "comprehensive_sentiment_dataset.json"],
                tag="complete_18month_batch1"
            )
            
            # Create daily roll-up
            daily_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "D", 
                "daily_sentiment_rollup_batch1.csv"
            )
            
            # Create weekly roll-up
            weekly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "W", 
                "weekly_sentiment_rollup_batch1.csv"
            )
            
            # Create monthly roll-up
            monthly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "M", 
                "monthly_sentiment_rollup_batch1.csv"
            )
            
            # Step 5: Generate final summary report
            end_time = datetime.now()
            duration = end_time - start_time
            
            final_report = {
                "analysis_summary": {
                    "batch_id": self.batch_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": duration.total_seconds() / 3600,
                    "workload": workload,
                    "stock_list": self.stock_symbols
                },
                "data_summary": report,
                "quality_control": qc_report,
                "files_created": {
                    "main_dataset": "comprehensive_sentiment_dataset.csv",
                    "json_dataset": "comprehensive_sentiment_dataset.json",
                    "daily_rollup": daily_rollup,
                    "weekly_rollup": weekly_rollup,
                    "monthly_rollup": monthly_rollup,
                    "snapshot": snapshot_path
                },
                "missing_data_summary": {
                    "total_missing_entries": len(missing_data),
                    "missing_data_log": "missing_data_log_batch1.json"
                }
            }
            
            # Save final report
            with open(self.output_dir / "complete_18month_analysis_report_batch1.json", 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info("=" * 80)
            self.logger.info("COMPLETE 18-MONTH SENTIMENT ANALYSIS FINISHED - BATCH 1")
            self.logger.info("=" * 80)
            self.logger.info(f"Duration: {duration.total_seconds() / 3600:.2f} hours")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Files created: {list(self.output_dir.glob('*'))}")
            self.logger.info("=" * 80)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error during BATCH 1 analysis: {e}")
            raise
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the batch 1 analysis results"""
        print("\n" + "=" * 80)
        print("COMPLETE 18-MONTH SENTIMENT ANALYSIS SUMMARY - BATCH 1")
        print("=" * 80)
        
        workload = report["analysis_summary"]["workload"]
        duration = report["analysis_summary"]["duration_hours"]
        
        print(f"📊 Analysis Coverage (Batch 1):")
        print(f"   • Date Range: {workload['start_date']} to {workload['end_date']}")
        print(f"   • Total Days: {workload['total_days']:,}")
        print(f"   • Total Stocks: {workload['total_stocks']} (Batch 1)")
        print(f"   • Total Requests: {workload['total_requests']:,}")
        print(f"   • Stock Range: {report['analysis_summary']['stock_list'][0]} to {report['analysis_summary']['stock_list'][-1]}")
        
        print(f"\n⏱️ Performance (Batch 1):")
        print(f"   • Duration: {duration:.2f} hours")
        print(f"   • Estimated vs Actual: {workload['estimated_hours']:.1f}h vs {duration:.1f}h")
        
        if "data_summary" in report:
            data_summary = report["data_summary"]
            print(f"\n📈 Data Generated (Batch 1):")
            print(f"   • Total Articles: {data_summary.get('generation_summary', {}).get('total_articles', 'N/A')}")
            print(f"   • Total Processed: {data_summary.get('generation_summary', {}).get('total_processed', 'N/A')}")
            print(f"   • Mean Sentiment: {data_summary.get('sentiment_statistics', {}).get('mean_sentiment', 'N/A'):.3f}")
        
        print(f"\n📁 Files Created (Batch 1):")
        files = report.get("files_created", {})
        for file_type, file_path in files.items():
            print(f"   • {file_type}: {file_path}")
        
        print(f"\n🔍 Quality Control (Batch 1):")
        qc = report.get("quality_control", {})
        if "integrity" in qc:
            integrity = qc["integrity"]
            print(f"   • Total Rows: {integrity.get('total_rows', 'N/A')}")
            print(f"   • Duplicates: {integrity.get('duplicates', 'N/A')}")
            print(f"   • Anomalies: {len(qc.get('anomalies', []))}")
        
        print(f"\n❌ Missing Data (Batch 1):")
        missing = report.get("missing_data_summary", {})
        print(f"   • Missing Entries: {missing.get('total_missing_entries', 'N/A')}")
        
        print("=" * 80)

def main():
    """Main function to run the complete 18-month sentiment analysis - BATCH 1"""
    print("🚀 Starting Complete 18-Month Sentiment Analysis - BATCH 1")
    print("This will process 40 stocks (RELIANCE.NSE to DRREDDY.NSE) for 18 months")
    print("Expected duration: ~10 hours")
    print("Output: Comprehensive sentiment dataset for BATCH 1")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ BATCH 1 analysis cancelled by user")
        return
    
    # Run the complete analysis for batch 1
    runner = CompleteSentimentRunnerBatch1()
    report = runner.run_complete_analysis()
    
    # Print summary
    runner.print_summary(report)
    
    print("\n✅ Complete 18-month sentiment analysis BATCH 1 finished successfully!")
    print(f"📁 Check the output directory: {runner.output_dir}")

if __name__ == "__main__":
    main() 