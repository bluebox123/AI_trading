#!/usr/bin/env python3
"""
Parallel Sentiment Analysis Runner
=================================

This script helps you run 3 parallel sentiment analysis batches simultaneously.

The 119 stocks have been divided into 3 groups:
- BATCH 1: 40 stocks (RELIANCE.NSE to DRREDDY.NSE)
- BATCH 2: 40 stocks (EICHERMOT.NSE to NESTLEIND.NSE) 
- BATCH 3: 39 stocks (NMDC.NSE to ZOMATO.NSE)

Each batch processes 18 months of data (Jan 2, 2024 - Jul 7, 2025)
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime
import json

def print_batch_info():
    """Print information about the 3 parallel batches"""
    
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
    
    batch2_stocks = [
        "EICHERMOT.NSE", "EXIDEIND.NSE", "FEDERALBNK.NSE", "GAIL.NSE", "GLENMARK.NSE",
        "GODREJCP.NSE", "GODREJPROP.NSE", "GRASIM.NSE", "HAVELLS.NSE", "HCLTECH.NSE",
        "HDFCBANK.NSE", "HEROMOTOCO.NSE", "HINDALCO.NSE", "HINDUNILVR.NSE", "HONAUT.NSE",
        "ICICIBANK.NSE", "IDFCFIRSTB.NSE", "INDUSINDBK.NSE", "INFY.NSE", "INOXLEISUR.NSE",
        "IPCALAB.NSE", "IRCTC.NSE", "ITC.NSE", "JINDALSTEL.NSE", "JSWSTEEL.NSE",
        "KOTAKBANK.NSE", "LICHSGFIN.NSE", "LT.NSE", "LTIM.NSE", "LTTS.NSE",
        "LUPIN.NSE", "MARICO.NSE", "MARUTI.NSE", "MAXHEALTH.NSE", "MCDOWELL-N.NSE",
        "MINDAIND.NSE", "MINDTREE.NSE", "MOTHERSON.NSE", "MPHASIS.NSE", "NESTLEIND.NSE"
    ]
    
    batch3_stocks = [
        "NMDC.NSE", "NTPC.NSE", "NYKAA.NSE", "OBEROIRLTY.NSE", "ONGC.NSE",
        "PAGEIND.NSE", "PAYTM.NSE", "PERSISTENT.NSE", "PIDILITIND.NSE", "PNB.NSE",
        "POLICYBZR.NSE", "POLYCAB.NSE", "POWERGRID.NSE", "PVR.NSE", "RELAXO.NSE",
        "SAIL.NSE", "SBIN.NSE", "SHREECEM.NSE", "SIEMENS.NSE", "SRF.NSE",
        "STAR.NSE", "SUNPHARMA.NSE", "TATACONSUM.NSE", "TATACHEM.NSE", "TATAMOTORS.NSE",
        "TATASTEEL.NSE", "TCS.NSE", "TECHM.NSE", "TITAN.NSE", "TORNTPHARM.NSE",
        "TRENT.NSE", "ULTRACEMCO.NSE", "UPL.NSE", "VEDL.NSE", "VIPIND.NSE",
        "VOLTAS.NSE", "WHIRLPOOL.NSE", "WIPRO.NSE", "ZOMATO.NSE"
    ]
    
    print("=" * 80)
    print("🚀 PARALLEL SENTIMENT ANALYSIS SETUP")
    print("=" * 80)
    print(f"📊 Analysis Configuration:")
    print(f"   • Date Range: January 2, 2024 to July 7, 2025 (18 months)")
    print(f"   • Total Days: 556")
    print(f"   • Total Stocks: 119 (divided into 3 batches)")
    print(f"   • Total Requests: 66,164")
    print(f"   • Estimated Total Time: ~30 hours")
    print(f"   • Estimated Parallel Time: ~10 hours (3x speedup)")
    print()
    
    print("📋 Batch Distribution:")
    print(f"   • BATCH 1: {len(batch1_stocks)} stocks ({batch1_stocks[0]} to {batch1_stocks[-1]})")
    print(f"   • BATCH 2: {len(batch2_stocks)} stocks ({batch2_stocks[0]} to {batch2_stocks[-1]})")
    print(f"   • BATCH 3: {len(batch3_stocks)} stocks ({batch3_stocks[0]} to {batch3_stocks[-1]})")
    print()
    
    print("📁 Output Directories:")
    print("   • BATCH 1: complete_18month_sentiment_dataset_batch1/")
    print("   • BATCH 2: complete_18month_sentiment_dataset_batch2/")
    print("   • BATCH 3: complete_18month_sentiment_dataset_batch3/")
    print()
    
    print("📝 Log Files:")
    print("   • BATCH 1: complete_18month_sentiment_batch1_YYYYMMDD_HHMMSS.log")
    print("   • BATCH 2: complete_18month_sentiment_batch2_YYYYMMDD_HHMMSS.log")
    print("   • BATCH 3: complete_18month_sentiment_batch3_YYYYMMDD_HHMMSS.log")
    print("=" * 80)

def print_instructions():
    """Print instructions for running parallel sentiment analysis"""
    print("\n🔧 HOW TO RUN PARALLEL SENTIMENT ANALYSIS")
    print("=" * 80)
    print("1. Open 3 separate terminal windows/tabs")
    print()
    print("2. In each terminal, navigate to the sentiment_v2 directory:")
    print("   cd data/sentiment_v2")
    print()
    print("3. Run each batch in a separate terminal:")
    print()
    print("   Terminal 1 (BATCH 1):")
    print("   python run_complete_5year_sentiment_batch1.py")
    print()
    print("   Terminal 2 (BATCH 2):")
    print("   python run_complete_5year_sentiment_batch2.py")
    print()
    print("   Terminal 3 (BATCH 3):")
    print("   python run_complete_5year_sentiment_batch3.py")
    print()
    print("4. Monitor progress in each terminal")
    print("   • Each batch will show its own progress logs")
    print("   • Estimated time per batch: ~10 hours")
    print("   • Total parallel time: ~10 hours (vs 30 hours sequential)")
    print()
    print("5. After completion, you'll have 3 output directories:")
    print("   • complete_18month_sentiment_dataset_batch1/")
    print("   • complete_18month_sentiment_dataset_batch2/")
    print("   • complete_18month_sentiment_dataset_batch3/")
    print()
    print("6. Optional: Merge results using the merge script (create separately)")
    print("=" * 80)

def print_monitoring_tips():
    """Print tips for monitoring the parallel processes"""
    print("\n📊 MONITORING TIPS")
    print("=" * 80)
    print("• Check CPU and memory usage:")
    print("  - Each batch will use significant CPU/memory")
    print("  - Monitor system resources to ensure smooth operation")
    print()
    print("• Track progress in real-time:")
    print("  - Each terminal shows live progress")
    print("  - Log files contain detailed progress information")
    print()
    print("• Estimated completion times:")
    print("  - BATCH 1: ~10 hours (40 stocks)")
    print("  - BATCH 2: ~10 hours (40 stocks)")
    print("  - BATCH 3: ~9.8 hours (39 stocks)")
    print()
    print("• In case of errors:")
    print("  - Each batch is independent")
    print("  - If one fails, others continue running")
    print("  - Check log files for detailed error information")
    print()
    print("• System requirements:")
    print("  - Sufficient disk space (each batch ~1-2GB output)")
    print("  - Stable internet connection (for news data)")
    print("  - 8GB+ RAM recommended for smooth operation")
    print("=" * 80)

def create_batch_runner():
    """Create a simple batch runner script"""
    runner_script = '''#!/bin/bash
# Parallel Sentiment Analysis Batch Runner
# Run this script to start all 3 batches simultaneously

echo "🚀 Starting Parallel Sentiment Analysis"
echo "Opening 3 terminals for parallel processing..."

# For Windows (Git Bash)
if [[ "$OSTYPE" == "msys" ]]; then
    start cmd /k "cd /d data/sentiment_v2 && python run_complete_5year_sentiment_batch1.py"
    start cmd /k "cd /d data/sentiment_v2 && python run_complete_5year_sentiment_batch2.py"
    start cmd /k "cd /d data/sentiment_v2 && python run_complete_5year_sentiment_batch3.py"
fi

# For Linux/Mac
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    gnome-terminal -- bash -c "cd data/sentiment_v2 && python run_complete_5year_sentiment_batch1.py; exec bash"
    gnome-terminal -- bash -c "cd data/sentiment_v2 && python run_complete_5year_sentiment_batch2.py; exec bash"
    gnome-terminal -- bash -c "cd data/sentiment_v2 && python run_complete_5year_sentiment_batch3.py; exec bash"
fi

echo "✅ All 3 batches started in separate terminals"
echo "Monitor progress in each terminal window"
'''
    
    with open("start_parallel_analysis.sh", 'w') as f:
        f.write(runner_script)
    
    print("\n📄 Created batch runner script: start_parallel_analysis.sh")
    print("   (You can run this to automatically start all 3 batches)")

def main():
    """Main function to display parallel analysis information"""
    print_batch_info()
    print_instructions()
    print_monitoring_tips()
    create_batch_runner()
    
    print("\n✅ READY FOR PARALLEL SENTIMENT ANALYSIS")
    print("Follow the instructions above to run all 3 batches simultaneously!")
    print("\n🎯 Expected Results:")
    print("   • 3x faster processing (10 hours instead of 30)")
    print("   • Complete sentiment dataset for all 119 stocks")
    print("   • 18 months of intraday sentiment data")
    print("   • All advanced features included")

if __name__ == "__main__":
    main() 