#!/usr/bin/env python3
"""
V5 Temporal Dataset Creator
==========================

Creates stock-specific temporal datasets for v5 model training using:
- 5 years of sentiment data (2020-2025)
- Corresponding stock price data
- Technical indicators and features
- Temporal sequences with lookback windows

This script references the tiered_temporal_data_processor.py architecture
but focuses on creating individual stock-specific datasets for v5 models.
"""

import os
import pandas as pd
import numpy as np
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V5TemporalDatasetCreator:
    """
    Creates stock-specific temporal datasets for v5 model training.
    
    Key Features:
    - Combines 5 years of sentiment and price data
    - Creates temporal sequences with configurable lookback windows
    - Adds technical indicators and sentiment features
    - Saves individual stock datasets for targeted training
    """
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.sentiment_dir = self.data_dir / "sentiment"
        self.v5_datasets_dir = self.processed_data_dir / "v5_temporal_datasets"
        
        # Create output directories
        self.v5_datasets_dir.mkdir(exist_ok=True)
        (self.v5_datasets_dir / "stock_specific").mkdir(exist_ok=True)
        (self.v5_datasets_dir / "market_sector").mkdir(exist_ok=True)
        (self.v5_datasets_dir / "aggregated").mkdir(exist_ok=True)
        
        # V5 Configuration - optimized for transformer models
        self.config = {
            'lookback_window': 60,  # 60 days of historical data
            'prediction_horizon': 5,  # Predict 5 days ahead
            'min_sequence_length': 100,  # Minimum data points for valid sequences
            'sentiment_features': [
                'Sentiment_Score', 'Confidence_Score', 'News_Volume', 
                'Social_Media_Mentions', 'Analyst_Coverage'
            ],
            'price_features': [
                'open', 'high', 'low', 'close', 'adjusted_close', 'volume'
            ],
            'technical_indicators': [
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'volatility_10', 'volatility_20',
                'momentum_5', 'momentum_10',
                'volume_ratio_10', 'volume_ratio_20',
                'price_spread', 'high_low_spread'
            ],
            'sentiment_engineered': [
                'sentiment_momentum_5', 'sentiment_momentum_10',
                'sentiment_volatility_10', 'sentiment_extremes',
                'confidence_weighted_sentiment', 'news_volume_normalized'
            ]
        }
        
        logger.info("🏗️ V5 Temporal Dataset Creator initialized")
        logger.info(f"📊 Lookback window: {self.config['lookback_window']} days")
        logger.info(f"🎯 Prediction horizon: {self.config['prediction_horizon']} days")
    
    def load_sentiment_data(self) -> pd.DataFrame:
        """Load and combine all 5 years of sentiment data."""
        logger.info("📰 Loading sentiment data from all years...")
        
        sentiment_files = [
            "stock_sentiment_dataset_2020-2021.csv",
            "stock_sentiment_dataset_2021-2022.csv", 
            "stock_sentiment_dataset_2022-2023.csv",
            "stock_sentiment_dataset_2023-2024.csv",
            "stock-sentiment-dataset_2024-2025.csv"
        ]
        
        all_sentiment_data = []
        
        for file in sentiment_files:
            file_path = self.sentiment_dir / file
            if file_path.exists():
                try:
                    # Read in chunks to handle large files
                    chunk_list = []
                    for chunk in pd.read_csv(file_path, chunksize=10000, parse_dates=['Date']):
                        chunk_list.append(chunk)
                    df = pd.concat(chunk_list, ignore_index=True)
                    
                    all_sentiment_data.append(df)
                    logger.info(f"✅ Loaded {file}: {len(df)} records")
                except Exception as e:
                    logger.error(f"❌ Error loading {file}: {e}")
        
        if not all_sentiment_data:
            raise ValueError("No sentiment data loaded!")
        
        # Combine all sentiment data
        combined_sentiment = pd.concat(all_sentiment_data, ignore_index=True)
        combined_sentiment['Date'] = pd.to_datetime(combined_sentiment['Date'])
        combined_sentiment = combined_sentiment.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        logger.info(f"📊 Combined sentiment data: {len(combined_sentiment)} total records")
        logger.info(f"📅 Date range: {combined_sentiment['Date'].min()} to {combined_sentiment['Date'].max()}")
        logger.info(f"🏢 Unique symbols: {combined_sentiment['Symbol'].nunique()}")
        
        return combined_sentiment
    
    def load_stock_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load all stock price data for a specific symbol."""
        stock_data_dir = self.raw_data_dir / "stock_data"
        stock_data_files = list(stock_data_dir.glob(f"{symbol}_*_prices.csv.gz"))
        
        if not stock_data_files:
            logger.warning(f"⚠️ No stock price files found for {symbol}")
            return pd.DataFrame()
        
        all_price_data = []
        
        for file_path in stock_data_files:
            try:
                df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
                all_price_data.append(df)
            except Exception as e:
                logger.warning(f"⚠️ Error loading {file_path}: {e}")
        
        if not all_price_data:
            return pd.DataFrame()
        
        # Combine and sort all price data
        combined_prices = pd.concat(all_price_data, ignore_index=False)
        combined_prices = combined_prices.sort_index().drop_duplicates()
        combined_prices.reset_index(inplace=True)
        combined_prices.rename(columns={'date': 'Date'}, inplace=True)
        combined_prices['Date'] = pd.to_datetime(combined_prices['Date'])
        
        return combined_prices
    
    def align_sentiment_and_prices(self, sentiment_df: pd.DataFrame, prices_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Align sentiment and price data by date for a specific symbol."""
        if sentiment_df.empty or prices_df.empty:
            return pd.DataFrame()
        
        # Filter sentiment data for this symbol
        symbol_sentiment = sentiment_df[sentiment_df['Symbol'] == symbol].copy()
        
        if symbol_sentiment.empty:
            return pd.DataFrame()
        
        # Merge on date
        aligned_data = pd.merge(
            symbol_sentiment,
            prices_df,
            on='Date',
            how='inner'
        )
        
        # Sort by date
        aligned_data = aligned_data.sort_values('Date').reset_index(drop=True)
        
        return aligned_data
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Volatility measures
            df['volatility_10'] = df['close'].rolling(10).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            
            # Momentum indicators
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Volume indicators
            df['volume_ratio_10'] = df['volume'] / df['volume'].rolling(10).mean()
            df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Price spreads
            df['price_spread'] = (df['high'] - df['low']) / df['close']
            df['high_low_spread'] = (df['high'] - df['low']) / df['low']
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating technical indicators: {e}")
        
        return df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered sentiment features."""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            # Sentiment momentum
            df['sentiment_momentum_5'] = df['Sentiment_Score'] - df['Sentiment_Score'].shift(5)
            df['sentiment_momentum_10'] = df['Sentiment_Score'] - df['Sentiment_Score'].shift(10)
            
            # Sentiment volatility
            df['sentiment_volatility_10'] = df['Sentiment_Score'].rolling(10).std()
            
            # Sentiment extremes (binary features)
            df['sentiment_extremes'] = ((df['Sentiment_Score'] > 0.5) | (df['Sentiment_Score'] < -0.5)).astype(int)
            
            # Confidence-weighted sentiment
            df['confidence_weighted_sentiment'] = df['Sentiment_Score'] * df['Confidence_Score']
            
            # Normalized news volume
            df['news_volume_normalized'] = df['News_Volume'] / df['News_Volume'].rolling(20).mean()
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating sentiment features: {e}")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/cyclical features."""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            # Day of week (cyclical encoding)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
            
            # Month (cyclical encoding)
            df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            
            # Quarter
            df['quarter'] = df['Date'].dt.quarter
            
            # Market timing features
            df['is_month_start'] = (df['Date'].dt.day <= 5).astype(int)
            df['is_month_end'] = (df['Date'].dt.day >= 25).astype(int)
            df['is_quarter_end'] = df['Date'].dt.month.isin([3, 6, 9, 12]).astype(int)
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating temporal features: {e}")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create temporal sequences for model training."""
        if len(df) < self.config['lookback_window'] + self.config['prediction_horizon']:
            logger.warning(f"⚠️ {symbol}: Insufficient data for sequence creation")
            return np.array([]), np.array([]), []
        
        # Select feature columns (excluding non-feature columns)
        exclude_cols = ['Date', 'Symbol', 'Company_Name', 'Sector', 'Day_of_Week', 
                       'Month', 'Quarter', 'Sentiment_Category', 'Primary_Market_Factor',
                       'Market_Cap_Category', 'Market_Phase']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_data = df[feature_cols].copy()
        
        # Handle NaN values
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        feature_data = feature_data.fillna(0)  # Final fallback
        
        # Create target variable (future returns)
        target_col = 'close'
        if target_col not in feature_data.columns:
            logger.error(f"❌ {symbol}: Target column '{target_col}' not found")
            return np.array([]), np.array([]), []
        
        sequences = []
        targets = []
        
        lookback = self.config['lookback_window']
        horizon = self.config['prediction_horizon']
        
        for i in range(lookback, len(feature_data) - horizon):
            # Input sequence (lookback window)
            sequence = feature_data.iloc[i-lookback:i].values
            
            # Target (future return)
            current_price = feature_data.iloc[i][target_col]
            future_price = feature_data.iloc[i+horizon][target_col]
            future_return = (future_price - current_price) / current_price
            
            sequences.append(sequence)
            targets.append(future_return)
        
        if not sequences:
            logger.warning(f"⚠️ {symbol}: No valid sequences created")
            return np.array([]), np.array([]), []
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"✅ {symbol}: Created {len(sequences)} sequences with {len(feature_cols)} features")
        
        return sequences, targets, feature_cols
    
    def save_dataset(self, sequences: np.ndarray, targets: np.ndarray, 
                    feature_names: List[str], symbol: str, metadata: Dict) -> None:
        """Save the temporal dataset for a symbol."""
        if sequences.size == 0:
            logger.warning(f"⚠️ {symbol}: No data to save")
            return
        
        output_dir = self.v5_datasets_dir / "stock_specific"
        output_file = output_dir / f"{symbol}_v5_temporal_dataset.npz"
        metadata_file = output_dir / f"{symbol}_v5_metadata.json"
        
        # Save sequences and targets
        np.savez_compressed(
            output_file,
            sequences=sequences,
            targets=targets,
            feature_names=feature_names
        )
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {symbol} dataset: {sequences.shape} sequences, {len(feature_names)} features")
    
    def process_symbol(self, symbol: str, sentiment_data: pd.DataFrame) -> bool:
        """Process a single symbol and create its temporal dataset."""
        logger.info(f"🔄 Processing {symbol}...")
        
        try:
            # Load stock price data
            prices_df = self.load_stock_data_for_symbol(symbol)
            if prices_df.empty:
                logger.warning(f"⚠️ {symbol}: No price data found")
                return False
            
            # Align sentiment and price data
            aligned_df = self.align_sentiment_and_prices(sentiment_data, prices_df, symbol)
            if aligned_df.empty:
                logger.warning(f"⚠️ {symbol}: No aligned data")
                return False
            
            # Add technical indicators
            aligned_df = self.create_technical_indicators(aligned_df)
            
            # Add sentiment features
            aligned_df = self.create_sentiment_features(aligned_df)
            
            # Add temporal features
            aligned_df = self.create_temporal_features(aligned_df)
            
            # Create sequences
            sequences, targets, feature_names = self.create_sequences(aligned_df, symbol)
            
            if sequences.size == 0:
                logger.warning(f"⚠️ {symbol}: No sequences created")
                return False
            
            # Create metadata
            metadata = {
                'symbol': symbol,
                'creation_date': datetime.now().isoformat(),
                'data_range': {
                    'start_date': aligned_df['Date'].min().isoformat(),
                    'end_date': aligned_df['Date'].max().isoformat(),
                    'total_days': len(aligned_df)
                },
                'sequences': {
                    'count': len(sequences),
                    'lookback_window': self.config['lookback_window'],
                    'prediction_horizon': self.config['prediction_horizon'],
                    'feature_count': len(feature_names)
                },
                'statistics': {
                    'mean_return': float(np.mean(targets)),
                    'std_return': float(np.std(targets)),
                    'min_return': float(np.min(targets)),
                    'max_return': float(np.max(targets))
                },
                'features': feature_names,
                'config': self.config
            }
            
            # Save dataset
            self.save_dataset(sequences, targets, feature_names, symbol, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            return False
    
    def create_all_datasets(self) -> None:
        """Create temporal datasets for all symbols."""
        logger.info("🚀 Starting V5 temporal dataset creation...")
        
        # Load sentiment data
        sentiment_data = self.load_sentiment_data()
        
        # Get unique symbols
        symbols = sentiment_data['Symbol'].unique()
        logger.info(f"📊 Found {len(symbols)} unique symbols")
        
        # Process each symbol
        success_count = 0
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"📈 Processing {i+1}/{len(symbols)}: {symbol}")
            
            if self.process_symbol(symbol, sentiment_data):
                success_count += 1
            else:
                failed_symbols.append(symbol)
        
        # Create summary report
        summary = {
            'creation_date': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'successful_datasets': success_count,
            'failed_symbols': failed_symbols,
            'success_rate': success_count / len(symbols) * 100,
            'config': self.config
        }
        
        summary_file = self.v5_datasets_dir / "creation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("🎉 V5 Temporal Dataset Creation Complete!")
        logger.info(f"✅ Successfully created {success_count}/{len(symbols)} datasets")
        logger.info(f"📊 Success rate: {summary['success_rate']:.1f}%")
        
        if failed_symbols:
            logger.warning(f"⚠️ Failed symbols ({len(failed_symbols)}): {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")

def main():
    """Main execution function."""
    creator = V5TemporalDatasetCreator()
    creator.create_all_datasets()

if __name__ == "__main__":
    main() 