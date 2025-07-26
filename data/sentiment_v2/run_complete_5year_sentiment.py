#!/usr/bin/env python3
"""
Complete 18-Month Sentiment Analysis Runner
==========================================

One-click script to run comprehensive sentiment analysis for:
- All 119 NSE stocks
- 18 months: January 2nd, 2024 to July 7th, 2025
- Intraday sentiment data (every single day)
- All advanced features enabled
- Comprehensive dataset generation

This will create a comprehensive sentiment dataset with all advanced features.
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

class CompleteSentimentRunner:
    """Complete 18-month sentiment analysis runner with all features"""
    
    def __init__(self):
        """Initialize the complete sentiment runner"""
        self.setup_logging()
        self.logger.info("Initializing Complete Sentiment Runner")
        
        # Initialize all components
        self.generator = ComprehensiveSentimentGenerator()
        self.advanced_features = IntegratedAdvancedFeatures()
        self.data_quality = DataQualityManager(data_dir=".")
        
        # Load all 117 stock symbols
        self.stock_symbols = self.load_all_stock_symbols()
        
        # Set date range: January 2nd, 2024 to July 7th, 2025 (18 months)
        self.start_date = date(2024, 1, 2)
        self.end_date = date(2025, 7, 7)
        
        # Create output directory
        self.output_dir = Path("complete_18month_sentiment_dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Loaded {len(self.stock_symbols)} stock symbols")
        self.logger.info(f"Date range: {self.start_date} to {self.end_date}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = f"complete_18month_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("complete_sentiment_runner")
    
    def load_all_stock_symbols(self) -> List[str]:
        """Load all 119 NSE stock symbols, skipping headers, dividers, and duplicates"""
        symbols_file = Path("../../nse_stock_symbols_complete_reliance_first.txt")
        if symbols_file.exists():
            seen = set()
            symbols = []
            with open(symbols_file, 'r') as f:
                for line in f:
                    symbol = line.strip()
                    if symbol.endswith('.NSE') and symbol not in seen:
                        symbols.append(symbol)
                        seen.add(symbol)
            self.logger.info(f"Loaded {len(symbols)} stock symbols from file")
            return symbols
        else:
            # Fallback to a comprehensive list of major NSE stocks
            self.logger.warning("Stock symbols file not found, using default list")
            return [
                "RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE", "INFY.NSE", "ICICIBANK.NSE",
                "HINDUNILVR.NSE", "ITC.NSE", "SBIN.NSE", "BHARTIARTL.NSE", "KOTAKBANK.NSE",
                "AXISBANK.NSE", "ASIANPAINT.NSE", "MARUTI.NSE", "HCLTECH.NSE", "SUNPHARMA.NSE",
                "ULTRACEMCO.NSE", "TITAN.NSE", "BAJFINANCE.NSE", "WIPRO.NSE", "NESTLEIND.NSE",
                "POWERGRID.NSE", "TECHM.NSE", "BAJAJFINSV.NSE", "NTPC.NSE", "HINDALCO.NSE",
                "ONGC.NSE", "JSWSTEEL.NSE", "TATAMOTORS.NSE", "ADANIENT.NSE", "COALINDIA.NSE",
                "DRREDDY.NSE", "SHREECEM.NSE", "CIPLA.NSE", "DIVISLAB.NSE", "EICHERMOT.NSE",
                "HEROMOTOCO.NSE", "BRITANNIA.NSE", "INDUSINDBK.NSE", "GRASIM.NSE", "TATACONSUM.NSE",
                "ADANIPORTS.NSE", "TATASTEEL.NSE", "BPCL.NSE", "VEDL.NSE", "HDFC.NSE",
                "SBILIFE.NSE", "UPL.NSE", "BAJAJ-AUTO.NSE", "TATAPOWER.NSE", "M&M.NSE",
                "HINDCOPPER.NSE", "ZEEL.NSE", "ICICIGI.NSE", "DLF.NSE", "VBL.NSE",
                "GODREJCP.NSE", "DABUR.NSE", "COLPAL.NSE", "BERGEPAINT.NSE", "HAVELLS.NSE",
                "MARICO.NSE", "UBL.NSE", "DEEPAKNTR.NSE", "ASHOKLEY.NSE", "PEL.NSE",
                "BIOCON.NSE", "MCDOWELL-N.NSE", "CADILAHC.NSE", "TORNTPHARM.NSE", "PFC.NSE",
                "RECLTD.NSE", "SIEMENS.NSE", "ABBOTINDIA.NSE", "ACC.NSE", "AMBUJACEM.NSE",
                "BANDHANBNK.NSE", "BANKBARODA.NSE", "BHARATFORG.NSE", "BOSCHLTD.NSE", "CANBK.NSE",
                "CENTURYTEX.NSE", "CHOLAFIN.NSE", "CUB.NSE", "DALMIABHA.NSE", "FEDERALBNK.NSE",
                "GAIL.NSE", "GICRE.NSE", "HAL.NSE", "HDFCAMC.NSE", "HINDCOPPER.NSE",
                "HINDPETRO.NSE", "ICICIPRULI.NSE", "IDEA.NSE", "IDFCFIRSTB.NSE", "INDIGO.NSE",
                "IOC.NSE", "JINDALSTEL.NSE", "JSWSTEEL.NSE", "KOTAKBANK.NSE", "LUPIN.NSE",
                "MUTHOOTFIN.NSE", "NATIONALUM.NSE", "NMDC.NSE", "PAGEIND.NSE", "PETRONET.NSE",
                "PIDILITIND.NSE", "PNB.NSE", "POWERGRID.NSE", "RECLTD.NSE", "RELIANCE.NSE",
                "SAIL.NSE", "SBIN.NSE", "SHREECEM.NSE", "SUNTV.NSE", "TATACOMM.NSE",
                "TATAMOTORS.NSE", "TATAPOWER.NSE", "TATASTEEL.NSE", "TECHM.NSE", "TITAN.NSE",
                "TORNTPHARM.NSE", "UPL.NSE", "VEDL.NSE", "WIPRO.NSE", "ZEEL.NSE"
            ]
    
    def calculate_total_workload(self) -> Dict[str, Any]:
        """Calculate the total workload for the 5-year analysis"""
        total_days = (self.end_date - self.start_date).days + 1
        total_stocks = len(self.stock_symbols)
        total_requests = total_days * total_stocks
        
        # Estimate processing time (with rate limiting)
        estimated_hours = (total_requests * 1.5) / 3600  # 1.5 seconds per request
        
        return {
            "total_days": total_days,
            "total_stocks": total_stocks,
            "total_requests": total_requests,
            "estimated_hours": estimated_hours,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat()
        }
    
    def run_complete_analysis(self):
        """Run the complete 18-month sentiment analysis"""
        workload = self.calculate_total_workload()
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE 18-MONTH SENTIMENT ANALYSIS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total days: {workload['total_days']}")
        self.logger.info(f"Total stocks: {workload['total_stocks']}")
        self.logger.info(f"Total requests: {workload['total_requests']:,}")
        self.logger.info(f"Estimated time: {workload['estimated_hours']:.1f} hours")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run comprehensive backfill for all stocks and dates
            self.logger.info("Step 1: Running comprehensive backfill...")
            missing_data = self.generator.run_backfill(
                start_date=self.start_date,
                end_date=self.end_date,
                stocks=self.stock_symbols
            )
            
            # Step 2: Generate comprehensive dataset with all advanced features
            self.logger.info("Step 2: Generating comprehensive dataset...")
            report = self.generator.generate_comprehensive_dataset(
                start_date=self.start_date,
                end_date=self.end_date,
                stock_symbols=self.stock_symbols,
                output_dir=str(self.output_dir)
            )
            
            # Step 3: Run data quality checks
            self.logger.info("Step 3: Running data quality checks...")
            qc_report = self.generator.run_qc("comprehensive_sentiment_dataset.csv")
            
            # Step 4: Create snapshots and roll-ups
            self.logger.info("Step 4: Creating snapshots and roll-ups...")
            snapshot_path = self.generator.save_snapshot(
                ["comprehensive_sentiment_dataset.csv", "comprehensive_sentiment_dataset.json"],
                tag="complete_18month"
            )
            
            # Create daily roll-up
            daily_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "D", 
                "daily_sentiment_rollup.csv"
            )
            
            # Create weekly roll-up
            weekly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "W", 
                "weekly_sentiment_rollup.csv"
            )
            
            # Create monthly roll-up
            monthly_rollup = self.generator.rollup_aggregate(
                "comprehensive_sentiment_dataset.csv", 
                "M", 
                "monthly_sentiment_rollup.csv"
            )
            
            # Step 5: Generate final summary report
            end_time = datetime.now()
            duration = end_time - start_time
            
            final_report = {
                "analysis_summary": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": duration.total_seconds() / 3600,
                    "workload": workload
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
                    "missing_data_log": "missing_data_log.json"
                }
            }
            
            # Save final report
            with open(self.output_dir / "complete_18month_analysis_report.json", 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info("=" * 80)
            self.logger.info("COMPLETE 18-MONTH SENTIMENT ANALYSIS FINISHED")
            self.logger.info("=" * 80)
            self.logger.info(f"Duration: {duration.total_seconds() / 3600:.2f} hours")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Files created: {list(self.output_dir.glob('*'))}")
            self.logger.info("=" * 80)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error during complete analysis: {e}")
            raise
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the analysis results"""
        print("\n" + "=" * 80)
        print("COMPLETE 18-MONTH SENTIMENT ANALYSIS SUMMARY")
        print("=" * 80)
        
        workload = report["analysis_summary"]["workload"]
        duration = report["analysis_summary"]["duration_hours"]
        
        print(f"📊 Analysis Coverage:")
        print(f"   • Date Range: {workload['start_date']} to {workload['end_date']}")
        print(f"   • Total Days: {workload['total_days']:,}")
        print(f"   • Total Stocks: {workload['total_stocks']}")
        print(f"   • Total Requests: {workload['total_requests']:,}")
        
        print(f"\n⏱️ Performance:")
        print(f"   • Duration: {duration:.2f} hours")
        print(f"   • Estimated vs Actual: {workload['estimated_hours']:.1f}h vs {duration:.1f}h")
        
        if "data_summary" in report:
            data_summary = report["data_summary"]
            print(f"\n📈 Data Generated:")
            print(f"   • Total Articles: {data_summary.get('generation_summary', {}).get('total_articles', 'N/A')}")
            print(f"   • Total Processed: {data_summary.get('generation_summary', {}).get('total_processed', 'N/A')}")
            print(f"   • Mean Sentiment: {data_summary.get('sentiment_statistics', {}).get('mean_sentiment', 'N/A'):.3f}")
        
        print(f"\n📁 Files Created:")
        files = report.get("files_created", {})
        for file_type, file_path in files.items():
            print(f"   • {file_type}: {file_path}")
        
        print(f"\n🔍 Quality Control:")
        qc = report.get("quality_control", {})
        if "integrity" in qc:
            integrity = qc["integrity"]
            print(f"   • Total Rows: {integrity.get('total_rows', 'N/A')}")
            print(f"   • Duplicates: {integrity.get('duplicates', 'N/A')}")
            print(f"   • Anomalies: {len(qc.get('anomalies', []))}")
        
        print(f"\n❌ Missing Data:")
        missing = report.get("missing_data_summary", {})
        print(f"   • Missing Entries: {missing.get('total_missing_entries', 'N/A')}")
        
        print("=" * 80)

def main():
    """Main function to run the complete 18-month sentiment analysis"""
    print("🚀 Starting Complete 18-Month Sentiment Analysis")
    print("This will process all 119 stocks for 18 months (January 2024 - July 2025)")
    print("Expected duration: ~30 hours")
    print("Output: Massive comprehensive sentiment dataset with all advanced features")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Analysis cancelled by user")
        return
    
    # Run the complete analysis
    runner = CompleteSentimentRunner()
    report = runner.run_complete_analysis()
    
    # Print summary
    runner.print_summary(report)
    
    print("\n✅ Complete 18-month sentiment analysis finished successfully!")
    print(f"📁 Check the output directory: {runner.output_dir}")

if __name__ == "__main__":
    main() 