#!/usr/bin/env python3
"""
Advanced Confidence Calibration System
=====================================

This module provides comprehensive confidence calibration for sentiment analysis,
including temperature scaling, Platt scaling, and ensemble confidence methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json
import pickle

@dataclass
class CalibratedConfidence:
    """Data class for calibrated confidence scores"""
    raw_confidence: float
    calibrated_confidence: float
    calibration_method: str
    reliability_score: float
    uncertainty_estimate: float
    confidence_interval: Tuple[float, float]

class ConfidenceCalibrator:
    """Advanced confidence calibration system for sentiment analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the confidence calibrator"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.calibration_data = []
        self.calibrators = {}
        self.reliability_metrics = {}
        
        # Load existing calibration data if available
        self._load_calibration_data()
        
        self.logger.info("Confidence Calibrator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the calibrator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for confidence calibration"""
        default_config = {
            "calibration_methods": {
                "temperature_scaling": True,
                "platt_scaling": True,
                "isotonic_regression": True,
                "ensemble_calibration": True
            },
            "reliability_thresholds": {
                "high_confidence": 0.8,
                "medium_confidence": 0.6,
                "low_confidence": 0.4
            },
            "uncertainty_estimation": {
                "bootstrap_samples": 100,
                "confidence_level": 0.95
            },
            "calibration_window": {
                "days": 30,
                "min_samples": 100
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _load_calibration_data(self):
        """Load existing calibration data"""
        calibration_file = Path("calibration_data.json")
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                self.logger.info(f"Loaded {len(self.calibration_data)} calibration samples")
            except Exception as e:
                self.logger.warning(f"Error loading calibration data: {e}")
    
    def add_calibration_sample(self, 
                              sentiment_score: float, 
                              raw_confidence: float, 
                              actual_sentiment: Optional[str] = None,
                              model_name: str = "ensemble",
                              metadata: Dict = None):
        """Add a sample for confidence calibration"""
        sample = {
            'sentiment_score': sentiment_score,
            'raw_confidence': raw_confidence,
            'actual_sentiment': actual_sentiment,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.calibration_data.append(sample)
        
        # Save calibration data periodically
        if len(self.calibration_data) % 100 == 0:
            self._save_calibration_data()
    
    def _save_calibration_data(self):
        """Save calibration data to file"""
        try:
            with open("calibration_data.json", 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving calibration data: {e}")
    
    def temperature_scaling(self, confidence_scores: List[float], 
                          actual_labels: List[str]) -> float:
        """Apply temperature scaling to calibrate confidence scores"""
        try:
            # Convert sentiment scores to probabilities
            probs = np.array(confidence_scores)
            
            # Find optimal temperature parameter
            temperatures = np.logspace(-3, 3, 100)
            best_temp = 1.0
            best_score = -np.inf
            
            for temp in temperatures:
                scaled_probs = probs ** (1/temp)
                scaled_probs = np.clip(scaled_probs, 0, 1)
                
                # Calculate calibration score (negative log likelihood)
                score = -np.mean(np.log(scaled_probs + 1e-10))
                
                if score > best_score:
                    best_score = score
                    best_temp = temp
            
            return best_temp
            
        except Exception as e:
            self.logger.error(f"Temperature scaling failed: {e}")
            return 1.0
    
    def platt_scaling(self, confidence_scores: List[float], 
                     actual_labels: List[str]) -> Tuple[float, float]:
        """Apply Platt scaling to calibrate confidence scores"""
        try:
            # Convert to binary labels for Platt scaling
            binary_labels = [1 if label == 'positive' else 0 for label in actual_labels]
            
            # Fit Platt scaling
            lr = LogisticRegression()
            X = np.array(confidence_scores).reshape(-1, 1)
            lr.fit(X, binary_labels)
            
            return lr.coef_[0][0], lr.intercept_[0]
            
        except Exception as e:
            self.logger.error(f"Platt scaling failed: {e}")
            return 1.0, 0.0
    
    def calculate_reliability_score(self, confidence_scores: List[float], 
                                  actual_labels: List[str]) -> float:
        """Calculate reliability score based on calibration accuracy"""
        try:
            if len(confidence_scores) < 10:
                return 0.5
            
            # Group by confidence bins
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(confidence_scores, bins) - 1
            
            reliability_scores = []
            
            for bin_idx in range(len(bins) - 1):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_confidences = np.array(confidence_scores)[mask]
                    bin_labels = np.array(actual_labels)[mask]
                    
                    # Calculate accuracy in this confidence bin
                    if len(bin_labels) > 0:
                        expected_accuracy = np.mean(bin_confidences)
                        actual_accuracy = np.mean([1 if label == 'positive' else 0 
                                                 for label in bin_labels])
                        
                        # Reliability is how well confidence matches accuracy
                        reliability = 1 - abs(expected_accuracy - actual_accuracy)
                        reliability_scores.append(reliability)
            
            return np.mean(reliability_scores) if reliability_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Reliability calculation failed: {e}")
            return 0.5
    
    def estimate_uncertainty(self, confidence_scores: List[float], 
                           n_bootstrap: int = 100) -> Tuple[float, Tuple[float, float]]:
        """Estimate uncertainty using bootstrap sampling"""
        try:
            if len(confidence_scores) < 10:
                return 0.1, (0.0, 0.2)
            
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                sample = np.random.choice(confidence_scores, size=len(confidence_scores), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            # Calculate uncertainty metrics
            mean_confidence = np.mean(bootstrap_means)
            std_confidence = np.std(bootstrap_means)
            
            # Confidence interval
            confidence_level = self.config["uncertainty_estimation"]["confidence_level"]
            z_score = 1.96  # 95% confidence interval
            margin_of_error = z_score * std_confidence
            
            confidence_interval = (max(0, mean_confidence - margin_of_error),
                                min(1, mean_confidence + margin_of_error))
            
            return std_confidence, confidence_interval
            
        except Exception as e:
            self.logger.error(f"Uncertainty estimation failed: {e}")
            return 0.1, (0.0, 0.2)
    
    def calibrate_confidence(self, 
                           sentiment_score: float,
                           raw_confidence: float,
                           model_name: str = "ensemble",
                           recent_samples: int = 1000) -> CalibratedConfidence:
        """Calibrate confidence score using multiple methods"""
        
        # Get recent calibration data
        recent_data = self.calibration_data[-recent_samples:] if self.calibration_data else []
        
        if len(recent_data) < 10:
            # Not enough data for calibration, return raw confidence
            return CalibratedConfidence(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_method="raw",
                reliability_score=0.5,
                uncertainty_estimate=0.1,
                confidence_interval=(max(0, raw_confidence - 0.1), min(1, raw_confidence + 0.1))
            )
        
        # Extract relevant data
        model_data = [d for d in recent_data if d['model_name'] == model_name]
        
        if len(model_data) < 10:
            model_data = recent_data  # Use all data if not enough model-specific data
        
        confidence_scores = [d['raw_confidence'] for d in model_data]
        actual_labels = [d['actual_sentiment'] for d in model_data if d['actual_sentiment']]
        
        # Apply different calibration methods
        calibrated_confidences = []
        
        # Temperature scaling
        if self.config["calibration_methods"]["temperature_scaling"] and actual_labels:
            try:
                temp = self.temperature_scaling(confidence_scores, actual_labels)
                temp_calibrated = raw_confidence ** (1/temp)
                calibrated_confidences.append(('temperature', temp_calibrated))
            except Exception as e:
                self.logger.warning(f"Temperature scaling failed: {e}")
        
        # Platt scaling
        if self.config["calibration_methods"]["platt_scaling"] and actual_labels:
            try:
                a, b = self.platt_scaling(confidence_scores, actual_labels)
                platt_calibrated = 1 / (1 + np.exp(-(a * raw_confidence + b)))
                calibrated_confidences.append(('platt', platt_calibrated))
            except Exception as e:
                self.logger.warning(f"Platt scaling failed: {e}")
        
        # Ensemble calibration (average of methods)
        if calibrated_confidences:
            ensemble_calibrated = np.mean([conf for _, conf in calibrated_confidences])
            calibration_method = "ensemble"
        else:
            ensemble_calibrated = raw_confidence
            calibration_method = "raw"
        
        # Calculate reliability score
        reliability_score = self.calculate_reliability_score(confidence_scores, actual_labels)
        
        # Estimate uncertainty
        uncertainty_estimate, confidence_interval = self.estimate_uncertainty(confidence_scores)
        
        return CalibratedConfidence(
            raw_confidence=raw_confidence,
            calibrated_confidence=ensemble_calibrated,
            calibration_method=calibration_method,
            reliability_score=reliability_score,
            uncertainty_estimate=uncertainty_estimate,
            confidence_interval=confidence_interval
        )
    
    def get_confidence_level(self, calibrated_confidence: float) -> str:
        """Get confidence level based on calibrated confidence"""
        thresholds = self.config["reliability_thresholds"]
        
        if calibrated_confidence >= thresholds["high_confidence"]:
            return "high"
        elif calibrated_confidence >= thresholds["medium_confidence"]:
            return "medium"
        elif calibrated_confidence >= thresholds["low_confidence"]:
            return "low"
        else:
            return "very_low"
    
    def update_calibration_with_feedback(self, 
                                       sentiment_score: float,
                                       raw_confidence: float,
                                       actual_sentiment: str,
                                       model_name: str = "ensemble"):
        """Update calibration with actual sentiment feedback"""
        self.add_calibration_sample(
            sentiment_score=sentiment_score,
            raw_confidence=raw_confidence,
            actual_sentiment=actual_sentiment,
            model_name=model_name
        )
        
        # Recalibrate if we have enough new data
        if len(self.calibration_data) % 100 == 0:
            self.logger.info(f"Calibration updated with {len(self.calibration_data)} samples")

def main():
    """Test the confidence calibration system"""
    calibrator = ConfidenceCalibrator()
    
    # Simulate some calibration data
    test_data = [
        (0.8, 0.9, "positive"),
        (0.6, 0.7, "positive"),
        (-0.3, 0.8, "negative"),
        (0.1, 0.5, "neutral"),
        (0.9, 0.95, "positive"),
        (-0.7, 0.6, "negative"),
        (0.2, 0.3, "neutral"),
        (0.5, 0.8, "positive")
    ]
    
    print("Testing Confidence Calibration System")
    print("=" * 50)
    
    # Add calibration data
    for sentiment_score, confidence, actual in test_data:
        calibrator.add_calibration_sample(
            sentiment_score=sentiment_score,
            raw_confidence=confidence,
            actual_sentiment=actual
        )
    
    # Test calibration
    test_cases = [
        (0.8, 0.9, "high confidence positive"),
        (-0.5, 0.6, "medium confidence negative"),
        (0.1, 0.3, "low confidence neutral")
    ]
    
    for sentiment_score, raw_confidence, description in test_cases:
        calibrated = calibrator.calibrate_confidence(sentiment_score, raw_confidence)
        
        print(f"\n{description}:")
        print(f"  Raw confidence: {calibrated.raw_confidence:.3f}")
        print(f"  Calibrated confidence: {calibrated.calibrated_confidence:.3f}")
        print(f"  Method: {calibrated.calibration_method}")
        print(f"  Reliability: {calibrated.reliability_score:.3f}")
        print(f"  Uncertainty: {calibrated.uncertainty_estimate:.3f}")
        print(f"  Confidence level: {calibrator.get_confidence_level(calibrated.calibrated_confidence)}")

if __name__ == "__main__":
    main() 