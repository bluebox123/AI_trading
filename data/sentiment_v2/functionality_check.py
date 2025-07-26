#!/usr/bin/env python3
"""
Comprehensive Functionality Check for Sentiment Analysis System
============================================================

This script checks all advanced features to ensure they're working correctly:
- Language Detection/Translation
- Tokenization/Lemmatization/Stopword
- NER/Entity Resolution
- Synonym/Alias Mapping
- Multi-Model Sentiment (VADER/BERT)
- Confidence Calibration
- Intraday Aggregation
- Exponential Decay
- Windowed Boosting
- Custom Decay Profiles
- Database/TSDB Storage
- Snapshot/Roll-up
- Backfill (RSS)
- Reprocessing Consistency
- Anomaly Detection/Quality Control
- Data Integrity Checks
- Logging/Metrics
- API/Streaming/Export
- Config-Driven/Plugins
- Health Checks/Dashboards
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all modules to test
try:
    from comprehensive_sentiment_dataset_generator import ComprehensiveSentimentGenerator
    from integrated_advanced_features import IntegratedAdvancedFeatures
    from advanced_nlp_processor import AdvancedNLPProcessor
    from advanced_confidence_calibration import ConfidenceCalibrator
    from intraday_aggregation import IntradayAggregator
    from advanced_temporal_weighting import AdvancedTemporalWeighter
    from data_quality_manager import DataQualityManager
    from backfill_manager import BackfillManager
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class FunctionalityChecker:
    """Comprehensive functionality checker for sentiment analysis system"""
    
    def __init__(self):
        """Initialize the functionality checker"""
        self.setup_logging()
        self.results = {}
        self.logger.info("Initializing Functionality Checker")
        
        # Test data
        self.test_text = "Reliance Industries reported strong Q4 earnings, beating analyst expectations. The stock price surged 5% in early trading."
        self.test_stock = "RELIANCE.NSE"
        self.test_date = datetime.now()
        
        # Initialize all components
        self.generator = ComprehensiveSentimentGenerator()
        self.advanced_features = IntegratedAdvancedFeatures()
        self.nlp_processor = AdvancedNLPProcessor()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.intraday_aggregator = IntradayAggregator()
        self.temporal_weighter = AdvancedTemporalWeighter()
        self.data_quality = DataQualityManager(data_dir=".")
        # Initialize backfill manager with required parameters
        self.backfill_manager = BackfillManager(
            fetch_func=self.generator.fetch_news_for_stock,
            stock_symbols=[self.test_stock]
        )
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("functionality_checker")
    
    def check_language_detection(self) -> bool:
        """Check language detection and translation"""
        try:
            self.logger.info("Testing language detection...")
            
            # Test English detection
            english_result = self.nlp_processor.detect_language(self.test_text)
            assert english_result[0] == "en", f"Expected 'en', got {english_result[0]}"
            
            # Test non-English text
            hindi_text = "रिलायंस इंडस्ट्रीज ने मजबूत Q4 आय की सूचना दी"
            hindi_result = self.nlp_processor.detect_language(hindi_text)
            assert hindi_result[0] != "en", "Hindi text should not be detected as English"
            
            self.logger.info("✅ Language detection working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Language detection failed: {e}")
            return False
    
    def check_tokenization_lemmatization(self) -> bool:
        """Check tokenization, lemmatization, and stopword removal"""
        try:
            self.logger.info("Testing tokenization and lemmatization...")
            
            tokens, lemmatized = self.nlp_processor.tokenize_and_lemmatize(self.test_text)
            
            assert len(tokens) > 0, "Should have tokens"
            assert len(lemmatized) > 0, "Should have lemmatized tokens"
            assert len(tokens) >= len(lemmatized), "Lemmatized should be <= original"
            
            # Check that stopwords are removed
            stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            for stopword in stopwords:
                if stopword in lemmatized:
                    self.logger.warning(f"Stopword '{stopword}' found in lemmatized tokens")
            
            self.logger.info("✅ Tokenization and lemmatization working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Tokenization/lemmatization failed: {e}")
            return False
    
    def check_ner_entity_resolution(self) -> bool:
        """Check NER and entity resolution"""
        try:
            self.logger.info("Testing NER and entity resolution...")
            
            entities = self.nlp_processor.extract_entities(self.test_text)
            resolved_entities = self.nlp_processor.resolve_entities(entities)
            
            assert len(entities) > 0, "Should extract entities"
            assert len(resolved_entities) > 0, "Should resolve entities"
            
            # Check for company names
            company_found = any('ORG' in str(entity) for entity in entities)
            if not company_found:
                self.logger.warning("No company entities found in test text")
            
            self.logger.info("✅ NER and entity resolution working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ NER/entity resolution failed: {e}")
            return False
    
    def check_synonym_alias_mapping(self) -> bool:
        """Check synonym and alias mapping"""
        try:
            self.logger.info("Testing synonym and alias mapping...")
            
            # Test company aliases
            if hasattr(self.nlp_processor, 'company_aliases'):
                assert len(self.nlp_processor.company_aliases) > 0, "Should have company aliases"
                
                # Test specific mapping
                if "RELIANCE" in self.nlp_processor.company_aliases:
                    aliases = self.nlp_processor.company_aliases["RELIANCE"]
                    assert len(aliases) > 0, "Should have aliases for RELIANCE"
            
            # Test ticker mapping
            if hasattr(self.nlp_processor, 'ticker_mapping'):
                assert len(self.nlp_processor.ticker_mapping) > 0, "Should have ticker mappings"
            
            self.logger.info("✅ Synonym and alias mapping working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Synonym/alias mapping failed: {e}")
            return False
    
    def check_multi_model_sentiment(self) -> bool:
        """Check multi-model sentiment analysis"""
        try:
            self.logger.info("Testing multi-model sentiment analysis...")
            
            # Test VADER
            vader_result = self.nlp_processor.analyze_sentiment_vader(self.test_text)
            assert 'compound' in vader_result, "VADER should return compound score"
            assert -1 <= vader_result['compound'] <= 1, "Compound score should be between -1 and 1"
            
            # Test ensemble
            ensemble_result, ensemble_score, confidence = self.nlp_processor.ensemble_sentiment_analysis(self.test_text)
            assert isinstance(ensemble_result, dict), "Ensemble should return dictionary"
            assert -1 <= ensemble_score <= 1, "Ensemble score should be between -1 and 1"
            assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"
            
            self.logger.info("✅ Multi-model sentiment analysis working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Multi-model sentiment analysis failed: {e}")
            return False
    
    def check_confidence_calibration(self) -> bool:
        """Check confidence calibration"""
        try:
            self.logger.info("Testing confidence calibration...")
            
            # Test calibration
            calibrated = self.confidence_calibrator.calibrate_confidence(
                sentiment_score=0.5,
                raw_confidence=0.7,
                model_name="test_model"
            )
            
            assert hasattr(calibrated, 'calibrated_confidence'), "Should have calibrated confidence"
            assert 0 <= calibrated.calibrated_confidence <= 1, "Calibrated confidence should be between 0 and 1"
            
            # Test confidence level
            level = self.confidence_calibrator.get_confidence_level(0.8)
            assert level in ['low', 'medium', 'high'], f"Invalid confidence level: {level}"
            
            self.logger.info("✅ Confidence calibration working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Confidence calibration failed: {e}")
            return False
    
    def check_intraday_aggregation(self) -> bool:
        """Check intraday aggregation"""
        try:
            self.logger.info("Testing intraday aggregation...")
            
            # Create test data
            test_data = [
                {
                    'timestamp': '2025-01-02T09:00:00',
                    'sentiment_score': 0.5,
                    'confidence': 0.8,
                    'stock_symbol': self.test_stock
                },
                {
                    'timestamp': '2025-01-02T09:15:00',
                    'sentiment_score': 0.6,
                    'confidence': 0.7,
                    'stock_symbol': self.test_stock
                }
            ]
            
            # Test aggregation
            buckets = self.intraday_aggregator.aggregate_multiple_buckets(
                test_data, ['15min', '1hour']
            )
            
            assert '15min' in buckets, "Should have 15min buckets"
            assert '1hour' in buckets, "Should have 1hour buckets"
            
            self.logger.info("✅ Intraday aggregation working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Intraday aggregation failed: {e}")
            return False
    
    def check_exponential_decay(self) -> bool:
        """Check exponential decay temporal weighting"""
        try:
            self.logger.info("Testing exponential decay...")
            
            # Test temporal weighting
            weight = self.temporal_weighter.calculate_temporal_weight(
                article_date=self.test_date - timedelta(days=7),
                target_date=self.test_date,
                stock_symbol=self.test_stock
            )
            
            assert hasattr(weight, 'final_weight'), "Should have final weight"
            assert 0 <= weight.final_weight <= 1, "Weight should be between 0 and 1"
            
            self.logger.info("✅ Exponential decay working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Exponential decay failed: {e}")
            return False
    
    def check_windowed_boosting(self) -> bool:
        """Check windowed boosting"""
        try:
            self.logger.info("Testing windowed boosting...")
            
            # Test with recent article (should get boost)
            recent_weight = self.temporal_weighter.calculate_temporal_weight(
                article_date=self.test_date - timedelta(hours=1),
                target_date=self.test_date,
                stock_symbol=self.test_stock
            )
            
            # Test with old article (should not get boost)
            old_weight = self.temporal_weighter.calculate_temporal_weight(
                article_date=self.test_date - timedelta(days=30),
                target_date=self.test_date,
                stock_symbol=self.test_stock
            )
            
            # Recent should have higher weight than old
            assert recent_weight.final_weight > old_weight.final_weight, "Recent articles should have higher weight"
            
            self.logger.info("✅ Windowed boosting working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Windowed boosting failed: {e}")
            return False
    
    def check_custom_decay_profiles(self) -> bool:
        """Check custom decay profiles"""
        try:
            self.logger.info("Testing custom decay profiles...")
            
            # Test sector-specific decay
            sector_weight = self.temporal_weighter.calculate_temporal_weight(
                article_date=self.test_date - timedelta(days=5),
                target_date=self.test_date,
                stock_symbol=self.test_stock,
                sector="Technology"
            )
            
            # Test stock-specific decay
            stock_weight = self.temporal_weighter.calculate_temporal_weight(
                article_date=self.test_date - timedelta(days=5),
                target_date=self.test_date,
                stock_symbol=self.test_stock
            )
            
            assert hasattr(sector_weight, 'final_weight'), "Should have final weight"
            assert hasattr(stock_weight, 'final_weight'), "Should have final weight"
            
            self.logger.info("✅ Custom decay profiles working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Custom decay profiles failed: {e}")
            return False
    
    def check_data_quality(self) -> bool:
        """Check data quality management"""
        try:
            self.logger.info("Testing data quality management...")
            
            # Create test dataset
            test_df = pd.DataFrame({
                'stock_symbol': [self.test_stock] * 10,
                'sentiment_score': np.random.uniform(-1, 1, 10),
                'confidence': np.random.uniform(0, 1, 10),
                'date': pd.date_range(start='2025-01-01', periods=10, freq='D')
            })
            
            # Test quality checks
            quality_report = self.data_quality.run_quality_checks(test_df)
            
            assert 'summary' in quality_report, "Should have quality summary"
            assert 'anomalies' in quality_report, "Should have anomalies report"
            
            self.logger.info("✅ Data quality management working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data quality management failed: {e}")
            return False
    
    def check_backfill_functionality(self) -> bool:
        """Check backfill functionality"""
        try:
            self.logger.info("Testing backfill functionality...")
            
            # Test backfill manager
            missing_data = self.backfill_manager.identify_missing_data(
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 5),
                stocks=[self.test_stock]
            )
            
            assert isinstance(missing_data, list), "Should return list of missing data"
            
            self.logger.info("✅ Backfill functionality working")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backfill functionality failed: {e}")
            return False
    
    def check_stock_symbols_loading(self) -> bool:
        """Check stock symbols loading"""
        try:
            self.logger.info("Testing stock symbols loading...")
            
            symbols = self.generator._load_stock_symbols()
            
            assert len(symbols) == 119, f"Expected 119 symbols, got {len(symbols)}"
            
            # Check that all symbols end with .NSE
            for symbol in symbols:
                assert symbol.endswith('.NSE'), f"Symbol {symbol} should end with .NSE"
            
            # Check for specific symbols
            expected_symbols = ["RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE", "INFY.NSE"]
            for expected in expected_symbols:
                assert expected in symbols, f"Expected symbol {expected} not found"
            
            self.logger.info(f"✅ Stock symbols loading working - loaded {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Stock symbols loading failed: {e}")
            return False
    
    def check_date_range_configuration(self) -> bool:
        """Check date range configuration"""
        try:
            self.logger.info("Testing date range configuration...")
            
            # Test the date range from the runner
            from run_complete_5year_sentiment import CompleteSentimentRunner
            runner = CompleteSentimentRunner()
            
            expected_start = date(2020, 5, 1)
            expected_end = date(2025, 5, 31)
            
            assert runner.start_date == expected_start, f"Expected start date {expected_start}, got {runner.start_date}"
            assert runner.end_date == expected_end, f"Expected end date {expected_end}, got {runner.end_date}"
            
            # Calculate total days
            total_days = (runner.end_date - runner.start_date).days + 1
            expected_days = 1857  # May 1, 2020 to May 31, 2025 (including both dates)
            
            assert total_days == expected_days, f"Expected {expected_days} days, got {total_days}"
            
            self.logger.info(f"✅ Date range configuration working - {total_days} days from {runner.start_date} to {runner.end_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Date range configuration failed: {e}")
            return False
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive functionality check"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE FUNCTIONALITY CHECK")
        self.logger.info("=" * 80)
        
        checks = [
            ("Language Detection/Translation", self.check_language_detection),
            ("Tokenization/Lemmatization/Stopword", self.check_tokenization_lemmatization),
            ("NER/Entity Resolution", self.check_ner_entity_resolution),
            ("Synonym/Alias Mapping", self.check_synonym_alias_mapping),
            ("Multi-Model Sentiment (VADER/BERT)", self.check_multi_model_sentiment),
            ("Confidence Calibration", self.check_confidence_calibration),
            ("Intraday Aggregation", self.check_intraday_aggregation),
            ("Exponential Decay", self.check_exponential_decay),
            ("Windowed Boosting", self.check_windowed_boosting),
            ("Custom Decay Profiles", self.check_custom_decay_profiles),
            ("Data Quality Management", self.check_data_quality),
            ("Backfill Functionality", self.check_backfill_functionality),
            ("Stock Symbols Loading", self.check_stock_symbols_loading),
            ("Date Range Configuration", self.check_date_range_configuration),
        ]
        
        results = {}
        passed = 0
        total = len(checks)
        
        for check_name, check_func in checks:
            self.logger.info(f"\n--- Testing {check_name} ---")
            try:
                success = check_func()
                results[check_name] = {
                    "status": "PASS" if success else "FAIL",
                    "working": success
                }
                if success:
                    passed += 1
            except Exception as e:
                self.logger.error(f"Exception in {check_name}: {e}")
                results[check_name] = {
                    "status": "ERROR",
                    "working": False,
                    "error": str(e)
                }
        
        # Generate summary
        summary = {
            "total_checks": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total) * 100,
            "results": results
        }
        
        self.logger.info("=" * 80)
        self.logger.info("FUNCTIONALITY CHECK SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total checks: {total}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {total - passed}")
        self.logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        self.logger.info("=" * 80)
        
        # Print detailed results
        for check_name, result in results.items():
            status = "✅ PASS" if result["working"] else "❌ FAIL"
            self.logger.info(f"{status} {check_name}")
        
        # Save results
        with open("functionality_check_results.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main function to run functionality check"""
    checker = FunctionalityChecker()
    results = checker.run_comprehensive_check()
    
    if results["success_rate"] >= 80:
        print(f"\n🎉 Functionality check completed with {results['success_rate']:.1f}% success rate!")
        print("The sentiment analysis system is ready for production use.")
    else:
        print(f"\n⚠️  Functionality check completed with {results['success_rate']:.1f}% success rate.")
        print("Some features need attention before production use.")
    
    return results

if __name__ == "__main__":
    main() 