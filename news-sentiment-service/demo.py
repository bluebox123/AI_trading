#!/usr/bin/env python3
"""
Demo Script for Simple Intraday Sentiment Analysis
=================================================

Quick demo to run sentiment analysis on all 117 NSE stocks.
"""

from simple_intraday_sentiment import SimpleIntradaySentiment

def main():
    print("=" * 60)
    print("DEMO: One-Time Sentiment Analysis for 117 NSE Stocks")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SimpleIntradaySentiment()
    
    # Run analysis
    print(f"\n📊 Analyzing sentiment for {len(analyzer.stocks)} stocks...")
    results = analyzer.analyze_all_stocks()
    
    # Display results
    analyzer.display_results(results)
    
    # Save to CSV
    filename = analyzer.save_results(results)
    
    print(f"\n🎉 Demo complete! Check {filename} for detailed results.")

if __name__ == "__main__":
    main() 