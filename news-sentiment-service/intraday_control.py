#!/usr/bin/env python3
"""
Intraday Sentiment Control Script
=================================

Control script for managing the intraday sentiment analysis system.
Provides commands for starting, stopping, monitoring, and analyzing sentiment data.
"""

import argparse
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from intraday_sentiment import IntradaySentimentAnalyzer

class IntradayController:
    """Controller for intraday sentiment analysis system"""
    
    def __init__(self):
        self.analyzer = None
        self.db_path = "intraday_sentiment.db"
    
    def start_monitoring(self):
        """Start the intraday sentiment monitoring"""
        print("Starting intraday sentiment monitoring...")
        self.analyzer = IntradaySentimentAnalyzer()
        self.analyzer.start_intraday_monitoring()
    
    def show_status(self):
        """Show current system status and recent data"""
        print("=== Intraday Sentiment System Status ===")
        
        # Check if database exists
        if not os.path.exists(self.db_path):
            print("❌ Database not found. Run the system first.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent sentiment data
            cursor.execute('''
                SELECT COUNT(*) FROM intraday_sentiment 
                WHERE timestamp > datetime('now', '-1 hour')
            ''')
            recent_count = cursor.fetchone()[0]
            
            # Get total records
            cursor.execute('SELECT COUNT(*) FROM intraday_sentiment')
            total_count = cursor.fetchone()[0]
            
            # Get recent alerts
            cursor.execute('''
                SELECT COUNT(*) FROM sentiment_alerts 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            recent_alerts = cursor.fetchone()[0]
            
            # Get sentiment distribution
            cursor.execute('''
                SELECT sentiment_label, COUNT(*) 
                FROM intraday_sentiment 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY sentiment_label
            ''')
            sentiment_dist = dict(cursor.fetchall())
            
            # Get most active stocks
            cursor.execute('''
                SELECT symbol, COUNT(*) as news_count
                FROM intraday_sentiment 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY symbol
                ORDER BY news_count DESC
                LIMIT 10
            ''')
            active_stocks = cursor.fetchall()
            
            conn.close()
            
            print(f"📊 Recent Activity (Last Hour): {recent_count} sentiment records")
            print(f"📈 Total Records: {total_count}")
            print(f"🚨 Recent Alerts (24h): {recent_alerts}")
            
            print("\n📋 Sentiment Distribution (Last Hour):")
            for label, count in sentiment_dist.items():
                print(f"  {label.capitalize()}: {count}")
            
            print("\n🔥 Most Active Stocks (Last Hour):")
            for symbol, count in active_stocks:
                print(f"  {symbol}: {count} mentions")
            
        except Exception as e:
            print(f"❌ Error reading status: {e}")
    
    def show_alerts(self, hours: int = 24):
        """Show recent sentiment alerts"""
        print(f"=== Recent Sentiment Alerts (Last {hours} hours) ===")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, symbol, alert_type, sentiment_score, message
                FROM sentiment_alerts
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (cutoff_time,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            if not alerts:
                print("No alerts found in the specified time period.")
                return
            
            for alert in alerts:
                timestamp, symbol, alert_type, score, message = alert
                emoji = "🟢" if alert_type == "strong_positive" else "🔴"
                print(f"{emoji} {timestamp} | {symbol} | {alert_type} | Score: {score:.3f}")
                print(f"   {message}")
                print()
            
        except Exception as e:
            print(f"❌ Error reading alerts: {e}")
    
    def show_stock_sentiment(self, symbol: str, hours: int = 24):
        """Show sentiment history for a specific stock"""
        print(f"=== Sentiment History for {symbol} (Last {hours} hours) ===")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, sentiment_score, sentiment_label, news_count, confidence
                FROM intraday_sentiment
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (symbol, cutoff_time))
            
            records = cursor.fetchall()
            conn.close()
            
            if not records:
                print(f"No sentiment data found for {symbol} in the specified time period.")
                return
            
            print(f"{'Timestamp':<20} {'Score':<8} {'Label':<10} {'News':<5} {'Confidence':<10}")
            print("-" * 60)
            
            for record in records:
                timestamp, score, label, news_count, confidence = record
                emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                print(f"{timestamp[:19]:<20} {score:<8.3f} {label:<10} {news_count:<5} {confidence:<10.3f} {emoji}")
            
        except Exception as e:
            print(f"❌ Error reading stock sentiment: {e}")
    
    def export_data(self, output_file: str = None):
        """Export sentiment data to CSV"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"intraday_sentiment_export_{timestamp}.csv"
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM intraday_sentiment 
                ORDER BY timestamp DESC
            ''', conn)
            conn.close()
            
            df.to_csv(output_file, index=False)
            print(f"✅ Exported {len(df)} records to {output_file}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def show_summary(self):
        """Show comprehensive sentiment summary"""
        print("=== Intraday Sentiment Summary ===")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute('SELECT COUNT(*) FROM intraday_sentiment')
            total_records = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM intraday_sentiment')
            unique_stocks = cursor.fetchone()[0]
            
            # Get sentiment distribution
            cursor.execute('''
                SELECT sentiment_label, COUNT(*) 
                FROM intraday_sentiment 
                GROUP BY sentiment_label
            ''')
            sentiment_dist = dict(cursor.fetchall())
            
            # Get average sentiment scores
            cursor.execute('''
                SELECT AVG(sentiment_score) FROM intraday_sentiment
            ''')
            avg_score = cursor.fetchone()[0]
            
            # Get top positive and negative stocks
            cursor.execute('''
                SELECT symbol, AVG(sentiment_score) as avg_score
                FROM intraday_sentiment
                GROUP BY symbol
                ORDER BY avg_score DESC
                LIMIT 5
            ''')
            top_positive = cursor.fetchall()
            
            cursor.execute('''
                SELECT symbol, AVG(sentiment_score) as avg_score
                FROM intraday_sentiment
                GROUP BY symbol
                ORDER BY avg_score ASC
                LIMIT 5
            ''')
            top_negative = cursor.fetchall()
            
            conn.close()
            
            print(f"📊 Total Records: {total_records}")
            print(f"📈 Unique Stocks: {unique_stocks}")
            print(f"📊 Average Sentiment Score: {avg_score:.3f}")
            
            print("\n📋 Overall Sentiment Distribution:")
            for label, count in sentiment_dist.items():
                percentage = (count / total_records) * 100
                print(f"  {label.capitalize()}: {count} ({percentage:.1f}%)")
            
            print("\n🟢 Top Positive Stocks:")
            for symbol, score in top_positive:
                print(f"  {symbol}: {score:.3f}")
            
            print("\n🔴 Top Negative Stocks:")
            for symbol, score in top_negative:
                print(f"  {symbol}: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")

def main():
    """Main function for the control script"""
    parser = argparse.ArgumentParser(description="Intraday Sentiment Analysis Control")
    parser.add_argument('command', choices=[
        'start', 'status', 'alerts', 'stock', 'export', 'summary'
    ], help='Command to execute')
    
    parser.add_argument('--symbol', '-s', help='Stock symbol for stock-specific commands')
    parser.add_argument('--hours', '-h', type=int, default=24, help='Hours to look back')
    parser.add_argument('--output', '-o', help='Output file for export')
    
    args = parser.parse_args()
    
    controller = IntradayController()
    
    if args.command == 'start':
        controller.start_monitoring()
    elif args.command == 'status':
        controller.show_status()
    elif args.command == 'alerts':
        controller.show_alerts(args.hours)
    elif args.command == 'stock':
        if not args.symbol:
            print("❌ Please provide a stock symbol with --symbol")
            return
        controller.show_stock_sentiment(args.symbol, args.hours)
    elif args.command == 'export':
        controller.export_data(args.output)
    elif args.command == 'summary':
        controller.show_summary()

if __name__ == "__main__":
    main() 