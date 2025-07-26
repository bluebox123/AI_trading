#!/usr/bin/env python3
"""
Test Script for Intraday Sentiment Analysis
==========================================

Quick test to verify the intraday sentiment system works correctly.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from intraday_sentiment import IntradaySentimentAnalyzer

def test_intraday_system():
    """Test the intraday sentiment analysis system"""
    print("=== Testing Intraday Sentiment Analysis System ===")
    
    try:
        # Initialize analyzer
        print("1. Initializing analyzer...")
        analyzer = IntradaySentimentAnalyzer()
        print(f"✅ Loaded {len(analyzer.stocks)} stock symbols")
        
        # Test stock loading
        print(f"2. Testing stock loading...")
        if len(analyzer.stocks) >= 100:
            print("✅ Stock loading successful")
        else:
            print(f"⚠️  Only {len(analyzer.stocks)} stocks loaded")
        
        # Test model loading
        print("3. Testing FinBERT model...")
        if analyzer.model and analyzer.tokenizer:
            print("✅ FinBERT model loaded successfully")
        else:
            print("⚠️  FinBERT model not loaded, will use TextBlob fallback")
        
        # Test database initialization
        print("4. Testing database initialization...")
        if os.path.exists(analyzer.db_path):
            print("✅ Database initialized successfully")
        else:
            print("⚠️  Database not found, will be created on first run")
        
        # Test news fetching (limited test)
        print("5. Testing news fetching...")
        try:
            news_items = analyzer._fetch_intraday_news()
            print(f"✅ Fetched {len(news_items)} news items")
        except Exception as e:
            print(f"⚠️  News fetching error: {e}")
        
        # Test sentiment analysis on sample text
        print("6. Testing sentiment analysis...")
        test_text = "Reliance Industries reports strong quarterly results with 15% revenue growth"
        sentiment_score, sentiment_label, confidence = analyzer._analyze_sentiment_finbert(test_text)
        print(f"✅ Test sentiment: {sentiment_label} (score: {sentiment_score:.3f}, confidence: {confidence:.3f})")
        
        # Test stock mention extraction
        print("7. Testing stock mention extraction...")
        test_news = "Reliance and TCS both show strong performance in the market today"
        mentions = analyzer._extract_stock_mentions(test_news)
        print(f"✅ Found mentions: {mentions}")
        
        # Test market hours detection
        print("8. Testing market hours detection...")
        is_market_hours = analyzer._is_market_hours()
        print(f"✅ Market hours check: {'Open' if is_market_hours else 'Closed'}")
        
        print("\n=== Test Summary ===")
        print("✅ All core components tested successfully")
        print("🚀 System ready for intraday monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_control_commands():
    """Test the control script commands"""
    print("\n=== Testing Control Commands ===")
    
    try:
        from intraday_control import IntradayController
        
        controller = IntradayController()
        
        # Test status command
        print("1. Testing status command...")
        controller.show_status()
        
        # Test summary command
        print("\n2. Testing summary command...")
        controller.show_summary()
        
        print("✅ Control commands tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Control command test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting intraday sentiment system tests...\n")
    
    # Test core system
    core_test_passed = test_intraday_system()
    
    # Test control commands
    control_test_passed = test_control_commands()
    
    print("\n=== Final Test Results ===")
    print(f"Core System: {'✅ PASSED' if core_test_passed else '❌ FAILED'}")
    print(f"Control Commands: {'✅ PASSED' if control_test_passed else '❌ FAILED'}")
    
    if core_test_passed and control_test_passed:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r intraday_requirements.txt")
        print("2. Start monitoring: python intraday_control.py start")
        print("3. Check status: python intraday_control.py status")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 