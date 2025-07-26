#!/usr/bin/env python3
"""
Management Script for Comprehensive Sentiment Dataset Generation
==============================================================

This script provides utilities to:
1. Start/stop/resume dataset generation
2. Monitor progress and statistics
3. Validate collected data
4. Generate reports and summaries
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import argparse
import sys
import time
from typing import Dict, List, Tuple, Optional

from config import DATA_DIR, DAILY_DATA_DIR, LOGS_DIR, CACHE_DIR
from comprehensive_sentiment_dataset_generator import ComprehensiveSentimentDatasetGenerator

class DatasetGenerationManager:
    """Manager for sentiment dataset generation process"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.progress_file = self.data_dir / "progress.json"
        self.summary_file = self.data_dir / "dataset_summary.json"
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def start_generation(self, max_workers: int = 4, resume: bool = True):
        """Start or resume dataset generation"""
        self.logger.info("🚀 Starting comprehensive sentiment dataset generation...")
        
        generator = ComprehensiveSentimentDatasetGenerator()
        
        if resume and self.progress_file.exists():
            self.logger.info("📈 Resuming from previous progress...")
        
        try:
            generator.generate_comprehensive_dataset(max_workers=max_workers)
            self.logger.info("✅ Dataset generation completed successfully!")
        except KeyboardInterrupt:
            self.logger.info("⏸️ Generation interrupted by user. Progress saved.")
        except Exception as e:
            self.logger.error(f"❌ Generation failed: {e}")
            raise
    
    def get_progress(self) -> Dict:
        """Get current progress information"""
        if not self.progress_file.exists():
            return {
                'status': 'not_started',
                'completed_tasks': 0,
                'total_tasks': 0,
                'progress_percentage': 0.0
            }
        
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        
        # Calculate additional metrics
        if progress.get('total_tasks', 0) > 0:
            progress['progress_percentage'] = (progress['completed_tasks'] / progress['total_tasks']) * 100
            progress['remaining_tasks'] = progress['total_tasks'] - progress['completed_tasks']
        
        return progress
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the dataset"""
        stats = {
            'data_collection': self._get_collection_stats(),
            'data_quality': self._get_quality_stats(),
            'coverage': self._get_coverage_stats(),
            'sentiment_distribution': self._get_sentiment_stats()
        }
        
        return stats
    
    def _get_collection_stats(self) -> Dict:
        """Get data collection statistics"""
        if not self.progress_file.exists():
            return {'status': 'not_started'}
        
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        
        return {
            'total_requests': progress.get('total_requests', 0),
            'successful_requests': progress.get('successful_requests', 0),
            'failed_requests': progress.get('failed_requests', 0),
            'success_rate': (progress.get('successful_requests', 0) / max(progress.get('total_requests', 1), 1)) * 100,
            'completed_tasks': progress.get('completed_tasks', 0),
            'total_tasks': progress.get('total_tasks', 0),
            'last_updated': progress.get('last_updated', 'unknown')
        }
    
    def _get_quality_stats(self) -> Dict:
        """Get data quality statistics"""
        daily_files = list(DAILY_DATA_DIR.rglob("*.json"))
        
        if not daily_files:
            return {'status': 'no_data'}
        
        total_articles = 0
        total_sentiment_scores = []
        confidence_scores = []
        
        for file_path in daily_files[:100]:  # Sample first 100 files for performance
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    total_articles += data.get('article_count', 0)
                    total_sentiment_scores.append(data.get('avg_sentiment_score', 0))
                    confidence_scores.append(data.get('avg_confidence', 0))
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
        
        return {
            'total_files': len(daily_files),
            'total_articles': total_articles,
            'avg_sentiment_score': np.mean(total_sentiment_scores) if total_sentiment_scores else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'sentiment_std': np.std(total_sentiment_scores) if total_sentiment_scores else 0
        }
    
    def _get_coverage_stats(self) -> Dict:
        """Get data coverage statistics"""
        daily_files = list(DAILY_DATA_DIR.rglob("*.json"))
        
        if not daily_files:
            return {'status': 'no_data'}
        
        # Extract stock symbols and dates
        stocks = set()
        dates = set()
        
        for file_path in daily_files:
            try:
                filename = file_path.stem
                if '_' in filename:
                    stock_symbol = filename.split('_')[0]
                    date_str = filename.split('_')[1]
                    stocks.add(stock_symbol)
                    dates.add(date_str)
            except Exception as e:
                self.logger.warning(f"Error parsing filename {file_path}: {e}")
        
        return {
            'unique_stocks': len(stocks),
            'unique_dates': len(dates),
            'total_files': len(daily_files),
            'avg_files_per_stock': len(daily_files) / max(len(stocks), 1),
            'avg_files_per_date': len(daily_files) / max(len(dates), 1)
        }
    
    def _get_sentiment_stats(self) -> Dict:
        """Get sentiment distribution statistics"""
        daily_files = list(DAILY_DATA_DIR.rglob("*.json"))
        
        if not daily_files:
            return {'status': 'no_data'}
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for file_path in daily_files[:100]:  # Sample for performance
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    positive_count += data.get('positive_count', 0)
                    negative_count += data.get('negative_count', 0)
                    neutral_count += data.get('neutral_count', 0)
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
        
        total_articles = positive_count + negative_count + neutral_count
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': total_articles,
            'positive_percentage': (positive_count / max(total_articles, 1)) * 100,
            'negative_percentage': (negative_count / max(total_articles, 1)) * 100,
            'neutral_percentage': (neutral_count / max(total_articles, 1)) * 100
        }
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive report"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"dataset_report_{timestamp}.txt"
        
        progress = self.get_progress()
        stats = self.get_statistics()
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE SENTIMENT DATASET GENERATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Progress Section
            f.write("📊 PROGRESS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Status: {progress.get('status', 'unknown')}\n")
            f.write(f"Completed Tasks: {progress.get('completed_tasks', 0):,}\n")
            f.write(f"Total Tasks: {progress.get('total_tasks', 0):,}\n")
            f.write(f"Progress: {progress.get('progress_percentage', 0):.2f}%\n")
            f.write(f"Remaining Tasks: {progress.get('remaining_tasks', 0):,}\n\n")
            
            # Collection Statistics
            f.write("📈 DATA COLLECTION STATISTICS\n")
            f.write("-" * 40 + "\n")
            collection_stats = stats['data_collection']
            if collection_stats.get('status') != 'not_started':
                f.write(f"Total Requests: {collection_stats.get('total_requests', 0):,}\n")
                f.write(f"Successful Requests: {collection_stats.get('successful_requests', 0):,}\n")
                f.write(f"Failed Requests: {collection_stats.get('failed_requests', 0):,}\n")
                f.write(f"Success Rate: {collection_stats.get('success_rate', 0):.2f}%\n")
                f.write(f"Last Updated: {collection_stats.get('last_updated', 'unknown')}\n")
            else:
                f.write("No data collection started yet.\n")
            f.write("\n")
            
            # Quality Statistics
            f.write("🔍 DATA QUALITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            quality_stats = stats['data_quality']
            if quality_stats.get('status') != 'no_data':
                f.write(f"Total Files: {quality_stats.get('total_files', 0):,}\n")
                f.write(f"Total Articles: {quality_stats.get('total_articles', 0):,}\n")
                f.write(f"Average Sentiment Score: {quality_stats.get('avg_sentiment_score', 0):.4f}\n")
                f.write(f"Average Confidence: {quality_stats.get('avg_confidence', 0):.4f}\n")
                f.write(f"Sentiment Standard Deviation: {quality_stats.get('sentiment_std', 0):.4f}\n")
            else:
                f.write("No data files found.\n")
            f.write("\n")
            
            # Coverage Statistics
            f.write("📋 DATA COVERAGE STATISTICS\n")
            f.write("-" * 40 + "\n")
            coverage_stats = stats['coverage']
            if coverage_stats.get('status') != 'no_data':
                f.write(f"Unique Stocks: {coverage_stats.get('unique_stocks', 0)}\n")
                f.write(f"Unique Dates: {coverage_stats.get('unique_dates', 0)}\n")
                f.write(f"Total Files: {coverage_stats.get('total_files', 0):,}\n")
                f.write(f"Average Files per Stock: {coverage_stats.get('avg_files_per_stock', 0):.2f}\n")
                f.write(f"Average Files per Date: {coverage_stats.get('avg_files_per_date', 0):.2f}\n")
            else:
                f.write("No data files found.\n")
            f.write("\n")
            
            # Sentiment Distribution
            f.write("😊 SENTIMENT DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            sentiment_stats = stats['sentiment_distribution']
            if sentiment_stats.get('status') != 'no_data':
                f.write(f"Positive Articles: {sentiment_stats.get('positive_count', 0):,} ({sentiment_stats.get('positive_percentage', 0):.1f}%)\n")
                f.write(f"Negative Articles: {sentiment_stats.get('negative_count', 0):,} ({sentiment_stats.get('negative_percentage', 0):.1f}%)\n")
                f.write(f"Neutral Articles: {sentiment_stats.get('neutral_count', 0):,} ({sentiment_stats.get('neutral_percentage', 0):.1f}%)\n")
                f.write(f"Total Articles: {sentiment_stats.get('total_articles', 0):,}\n")
            else:
                f.write("No sentiment data found.\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generation complete.\n")
        
        self.logger.info(f"📄 Report generated: {output_file}")
        return output_file
    
    def monitor_live(self, interval: int = 60):
        """Monitor generation progress in real-time"""
        self.logger.info(f"🔍 Starting live monitoring (refresh every {interval} seconds)...")
        self.logger.info("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                progress = self.get_progress()
                stats = self.get_statistics()
                
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                print("=" * 80)
                print("LIVE SENTIMENT DATASET GENERATION MONITOR")
                print("=" * 80)
                print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # Progress Bar
                if progress.get('total_tasks', 0) > 0:
                    percentage = progress.get('progress_percentage', 0)
                    bar_length = 50
                    filled_length = int(bar_length * percentage / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"Progress: [{bar}] {percentage:.2f}%")
                    print(f"Completed: {progress.get('completed_tasks', 0):,} / {progress.get('total_tasks', 0):,}")
                    print(f"Remaining: {progress.get('remaining_tasks', 0):,}")
                else:
                    print("Progress: Not started")
                print()
                
                # Collection Stats
                collection_stats = stats['data_collection']
                if collection_stats.get('status') != 'not_started':
                    print(f"Requests: {collection_stats.get('successful_requests', 0):,} successful, {collection_stats.get('failed_requests', 0):,} failed")
                    print(f"Success Rate: {collection_stats.get('success_rate', 0):.2f}%")
                print()
                
                # Quality Stats
                quality_stats = stats['data_quality']
                if quality_stats.get('status') != 'no_data':
                    print(f"Files Collected: {quality_stats.get('total_files', 0):,}")
                    print(f"Articles Analyzed: {quality_stats.get('total_articles', 0):,}")
                    print(f"Avg Sentiment: {quality_stats.get('avg_sentiment_score', 0):.4f}")
                print()
                
                # Coverage Stats
                coverage_stats = stats['coverage']
                if coverage_stats.get('status') != 'no_data':
                    print(f"Stocks Covered: {coverage_stats.get('unique_stocks', 0)}")
                    print(f"Dates Covered: {coverage_stats.get('unique_dates', 0)}")
                print()
                
                print("=" * 80)
                print(f"Next update in {interval} seconds... (Ctrl+C to stop)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def validate_data(self) -> Dict:
        """Validate collected data for quality and completeness"""
        self.logger.info("🔍 Validating collected data...")
        
        validation_results = {
            'overall_status': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        # Check if data directory exists
        if not DAILY_DATA_DIR.exists():
            validation_results['overall_status'] = 'no_data'
            validation_results['issues'].append("No data directory found")
            return validation_results
        
        # Check for data files
        daily_files = list(DAILY_DATA_DIR.rglob("*.json"))
        if not daily_files:
            validation_results['overall_status'] = 'no_files'
            validation_results['issues'].append("No data files found")
            return validation_results
        
        # Validate file structure
        valid_files = 0
        invalid_files = 0
        
        for file_path in daily_files[:100]:  # Sample for performance
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check required fields
                required_fields = ['stock_symbol', 'date', 'article_count', 'avg_sentiment_score']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    invalid_files += 1
                    validation_results['issues'].append(f"Missing fields in {file_path.name}: {missing_fields}")
                else:
                    valid_files += 1
                    
            except Exception as e:
                invalid_files += 1
                validation_results['issues'].append(f"Invalid JSON in {file_path.name}: {e}")
        
        # Calculate validation metrics
        total_checked = valid_files + invalid_files
        if total_checked > 0:
            validity_rate = (valid_files / total_checked) * 100
            validation_results['validity_rate'] = validity_rate
            
            if validity_rate >= 95:
                validation_results['overall_status'] = 'excellent'
            elif validity_rate >= 80:
                validation_results['overall_status'] = 'good'
            elif validity_rate >= 60:
                validation_results['overall_status'] = 'fair'
            else:
                validation_results['overall_status'] = 'poor'
        
        # Recommendations
        if validation_results['overall_status'] in ['fair', 'poor']:
            validation_results['recommendations'].append("Consider re-running data collection for failed files")
        
        if len(daily_files) < 1000:  # Arbitrary threshold
            validation_results['recommendations'].append("Data collection may be incomplete - consider continuing")
        
        self.logger.info(f"✅ Validation complete: {validation_results['overall_status']}")
        return validation_results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Manage comprehensive sentiment dataset generation")
    parser.add_argument('action', choices=['start', 'monitor', 'report', 'validate', 'status'],
                       help='Action to perform')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--output', type=str,
                       help='Output file for reports')
    
    args = parser.parse_args()
    
    manager = DatasetGenerationManager()
    
    if args.action == 'start':
        manager.start_generation(max_workers=args.workers)
    
    elif args.action == 'monitor':
        manager.monitor_live(interval=args.interval)
    
    elif args.action == 'report':
        output_file = args.output or None
        report_file = manager.generate_report(output_file)
        print(f"📄 Report generated: {report_file}")
    
    elif args.action == 'validate':
        results = manager.validate_data()
        print(f"🔍 Validation Status: {results['overall_status']}")
        if results.get('validity_rate'):
            print(f"📊 Validity Rate: {results['validity_rate']:.1f}%")
        if results['issues']:
            print("❌ Issues found:")
            for issue in results['issues'][:5]:  # Show first 5 issues
                print(f"  - {issue}")
        if results['recommendations']:
            print("💡 Recommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
    
    elif args.action == 'status':
        progress = manager.get_progress()
        stats = manager.get_statistics()
        
        print("📊 DATASET GENERATION STATUS")
        print("=" * 50)
        print(f"Progress: {progress.get('progress_percentage', 0):.2f}%")
        print(f"Completed: {progress.get('completed_tasks', 0):,} / {progress.get('total_tasks', 0):,}")
        
        collection_stats = stats['data_collection']
        if collection_stats.get('status') != 'not_started':
            print(f"Requests: {collection_stats.get('successful_requests', 0):,} successful")
            print(f"Success Rate: {collection_stats.get('success_rate', 0):.2f}%")
        
        quality_stats = stats['data_quality']
        if quality_stats.get('status') != 'no_data':
            print(f"Files: {quality_stats.get('total_files', 0):,}")
            print(f"Articles: {quality_stats.get('total_articles', 0):,}")

if __name__ == "__main__":
    main() 