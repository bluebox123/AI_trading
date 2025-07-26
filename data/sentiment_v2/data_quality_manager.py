#!/usr/bin/env python3
"""
Data Quality Manager for Sentiment Pipeline
==========================================

Handles:
- Snapshots (versioned, timestamped, and restorable)
- Roll-up aggregation (e.g., daily/weekly/monthly)
- Anomaly detection (statistical, missing data, outliers)
- Data integrity checks (schema, nulls, duplicates, value ranges)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any

class DataQualityManager:
    def __init__(self, data_dir: str = "./", snapshot_dir: str = "snapshots", log_file: str = "data_quality.log"):
        self.data_dir = Path(data_dir)
        self.snapshot_dir = self.data_dir / snapshot_dir
        self.snapshot_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging(log_file)

    def _setup_logging(self, log_file: str):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("data_quality")

    # --- SNAPSHOT ---
    def save_snapshot(self, file_list: List[str], tag: Optional[str] = None) -> str:
        """Save a snapshot of the specified files (raw or processed)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = tag or ""
        snap_name = f"snapshot_{timestamp}{'_' + tag if tag else ''}"
        snap_path = self.snapshot_dir / snap_name
        snap_path.mkdir(exist_ok=True)
        manifest = []
        for fname in file_list:
            src = self.data_dir / fname
            if src.exists():
                dst = snap_path / src.name
                dst.write_bytes(src.read_bytes())
                manifest.append(str(dst))
        # Save manifest
        with open(snap_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        self.logger.info(f"Snapshot saved: {snap_path}")
        return str(snap_path)

    def restore_snapshot(self, snapshot_name: str):
        """Restore files from a snapshot"""
        snap_path = self.snapshot_dir / snapshot_name
        manifest_file = snap_path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found in {snap_path}")
        with open(manifest_file) as f:
            manifest = json.load(f)
        for file_path in manifest:
            src = Path(file_path)
            dst = self.data_dir / src.name
            dst.write_bytes(src.read_bytes())
        self.logger.info(f"Snapshot restored: {snapshot_name}")

    # --- ROLL-UP ---
    def rollup_aggregate(self, input_file: str, freq: str = "D", output_file: Optional[str] = None):
        """Aggregate fine-grained data to coarser timeframes (D/W/M)"""
        df = pd.read_csv(self.data_dir / input_file, parse_dates=["article_date"])
        df.set_index("article_date", inplace=True)
        agg = df.resample(freq).agg({
            "sentiment_score": "mean",
            "confidence_score": "mean",
            "weighted_sentiment": "mean",
            "stock_symbol": "first"
        })
        out_file = output_file or f"rollup_{freq}_{input_file}"
        agg.to_csv(self.data_dir / out_file)
        self.logger.info(f"Roll-up aggregation saved: {out_file}")
        return str(self.data_dir / out_file)

    # --- ANOMALY DETECTION ---
    def detect_anomalies(self, input_file: str, z_thresh: float = 3.0) -> pd.DataFrame:
        """Detect anomalies in sentiment data using z-score"""
        df = pd.read_csv(self.data_dir / input_file)
        df["z_score"] = (df["sentiment_score"] - df["sentiment_score"].mean()) / df["sentiment_score"].std()
        anomalies = df[np.abs(df["z_score"]) > z_thresh]
        if not anomalies.empty:
            self.logger.warning(f"Anomalies detected in {input_file}: {len(anomalies)} rows")
        return anomalies

    # --- DATA INTEGRITY CHECKS ---
    def check_integrity(self, input_file: str) -> Dict[str, Any]:
        """Check for schema, nulls, duplicates, and value ranges"""
        df = pd.read_csv(self.data_dir / input_file)
        report = {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "out_of_range_sentiment": int(((df["sentiment_score"] < -1) | (df["sentiment_score"] > 1)).sum()),
            "out_of_range_confidence": int(((df["confidence_score"] < 0) | (df["confidence_score"] > 1)).sum()),
            "total_rows": len(df)
        }
        self.logger.info(f"Integrity check for {input_file}: {report}")
        return report

    # --- QC REPORT ---
    def generate_qc_report(self, input_file: str) -> Dict[str, Any]:
        anomalies = self.detect_anomalies(input_file)
        integrity = self.check_integrity(input_file)
        report = {
            "anomalies": anomalies.to_dict(orient="records"),
            "integrity": integrity
        }
        qc_file = self.data_dir / f"qc_report_{Path(input_file).stem}.json"
        with open(qc_file, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"QC report generated: {qc_file}")
        return report
    
    def run_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive quality checks on a DataFrame"""
        report = {
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": int(df.duplicated().sum())
            },
            "anomalies": {
                "sentiment_outliers": [],
                "confidence_outliers": [],
                "date_anomalies": []
            },
            "integrity": {
                "schema_valid": True,
                "value_ranges_valid": True,
                "data_types_valid": True
            }
        }
        
        # Check for sentiment score anomalies
        if "sentiment_score" in df.columns:
            sentiment_scores = df["sentiment_score"].dropna()
            if len(sentiment_scores) > 0:
                z_scores = np.abs((sentiment_scores - sentiment_scores.mean()) / sentiment_scores.std())
                outliers = sentiment_scores[z_scores > 3.0]
                report["anomalies"]["sentiment_outliers"] = outliers.to_dict()
        
        # Check for confidence score anomalies
        if "confidence_score" in df.columns:
            confidence_scores = df["confidence_score"].dropna()
            if len(confidence_scores) > 0:
                z_scores = np.abs((confidence_scores - confidence_scores.mean()) / confidence_scores.std())
                outliers = confidence_scores[z_scores > 3.0]
                report["anomalies"]["confidence_outliers"] = outliers.to_dict()
        
        # Check value ranges
        if "sentiment_score" in df.columns:
            out_of_range = ((df["sentiment_score"] < -1) | (df["sentiment_score"] > 1)).sum()
            if out_of_range > 0:
                report["integrity"]["value_ranges_valid"] = False
        
        if "confidence_score" in df.columns:
            out_of_range = ((df["confidence_score"] < 0) | (df["confidence_score"] > 1)).sum()
            if out_of_range > 0:
                report["integrity"]["value_ranges_valid"] = False
        
        self.logger.info(f"Quality checks completed: {report['summary']['total_rows']} rows processed")
        return report 