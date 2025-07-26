#!/usr/bin/env python3
"""
V5 Temporal Dataset Creator - Comprehensive Version (Fixed)
Creates temporal datasets by merging sentiment and stock price data for 117 stocks
Time range: May 1, 2020 to May 31, 2025
"""

import pandas as pd
import numpy as np
import os
import glob
import gzip
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('temporal_dataset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TemporalDatasetCreator:
    def __init__(self):
        self.base_dir = Path(".")
        self.sentiment_dir = self.base_dir / "data" / "sentiment"
        self.stock_data_dir = self.base_dir / "data" / "raw" / "stock_data"
        self.output_dir = self.base_dir / "data" / "processed" / "v5_temporal_datasets"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stock_specific").mkdir(exist_ok=True)
        
        # Date range
        self.start_date = datetime(2020, 5, 1)
        self.end_date = datetime(2025, 5, 31)
        
        logger.info(f"Initialized TemporalDatasetCreator")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
    def normalize_symbol(self, symbol):
        """Normalize symbol to .NSE format"""
        if symbol.endswith('.NS'):
            return symbol.replace('.NS', '.NSE')
        elif not symbol.endswith('.NSE'):
            return symbol + '.NSE'
        return symbol
        
    def load_sentiment_data(self):
        """Load all sentiment data from the 5 CSV files"""
        logger.info("Loading sentiment data...")
        
        sentiment_files = [
            "stock_sentiment_dataset_2020-2021.csv",
            "stock_sentiment_dataset_2021-2022.csv", 
            "stock_sentiment_dataset_2022-2023.csv",
            "stock_sentiment_dataset_2023-2024.csv",
            "stock-sentiment-dataset_2024-2025.csv"
        ]
        
        sentiment_dfs = []
        
        for file in sentiment_files:
            file_path = self.sentiment_dir / file
            if file_path.exists():
                logger.info(f"Loading {file}...")
                df = pd.read_csv(file_path)
                sentiment_dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
            else:
                logger.warning(f"Sentiment file not found: {file}")
        
        if not sentiment_dfs:
            raise ValueError("No sentiment data files found!")
            
        # Combine all sentiment data
        combined_sentiment = pd.concat(sentiment_dfs, ignore_index=True)
        
        # Normalize symbols to .NSE format
        combined_sentiment['Symbol'] = combined_sentiment['Symbol'].apply(self.normalize_symbol)
        
        # Convert Date column to datetime
        combined_sentiment['Date'] = pd.to_datetime(combined_sentiment['Date'])
        
        # Filter by date range
        mask = (combined_sentiment['Date'] >= self.start_date) & (combined_sentiment['Date'] <= self.end_date)
        combined_sentiment = combined_sentiment[mask]
        
        logger.info(f"Total sentiment records after filtering: {len(combined_sentiment)}")
        logger.info(f"Unique symbols in sentiment data: {combined_sentiment['Symbol'].nunique()}")
        
        return combined_sentiment

    def load_stock_data_for_symbol(self, symbol):
        """Load stock price data for a specific symbol"""
        logger.info(f"Loading stock data for {symbol}...")
        
        # Find all stock data files for this symbol
        pattern = f"{symbol}_*_prices.csv.gz"
        stock_files = glob.glob(str(self.stock_data_dir / pattern))
        
        if not stock_files:
            logger.warning(f"No stock data files found for {symbol}")
            return None
            
        stock_dfs = []
        
        for file_path in stock_files:
            try:
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
                    if not df.empty:
                        stock_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        if not stock_dfs:
            logger.warning(f"No valid stock data found for {symbol}")
            return None
            
        # Combine all stock data
        combined_stock = pd.concat(stock_dfs, ignore_index=True)
        
        # Convert date column to datetime
        combined_stock['date'] = pd.to_datetime(combined_stock['date'])
        
        # Filter by date range
        mask = (combined_stock['date'] >= self.start_date) & (combined_stock['date'] <= self.end_date)
        combined_stock = combined_stock[mask]
        
        # Sort by date
        combined_stock = combined_stock.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates (keep last if multiple entries for same date)
        combined_stock = combined_stock.drop_duplicates(subset=['date'], keep='last')
        
        logger.info(f"Loaded {len(combined_stock)} stock price records for {symbol}")
        
        return combined_stock
    
    def create_technical_indicators(self, df):
        """Create technical indicators from stock price data"""
        if len(df) < 20:  # Need at least 20 days for indicators
            return df
            
        df = df.copy()  # Make a copy to avoid SettingWithCopyWarning
        
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_10d'] = df['close'].pct_change(periods=10)
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Volatility
        df['volatility_10d'] = df['price_change'].rolling(window=10).std()
        df['volatility_20d'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def merge_sentiment_and_stock_data(self, sentiment_df, stock_df, symbol):
        """Merge sentiment and stock data for a specific symbol"""
        # Filter sentiment data for this symbol
        symbol_sentiment = sentiment_df[sentiment_df['Symbol'] == symbol].copy()
        
        if symbol_sentiment.empty:
            logger.warning(f"No sentiment data found for {symbol}")
            return None
            
        if stock_df is None or stock_df.empty:
            logger.warning(f"No stock data found for {symbol}")
            return None
            
        # Rename date columns for consistency
        symbol_sentiment = symbol_sentiment.rename(columns={'Date': 'date'})
        
        # Merge on date
        merged_df = pd.merge(
            stock_df, 
            symbol_sentiment, 
            on='date', 
            how='outer'
        )
        
        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Forward fill sentiment data
        sentiment_columns = [
            'Sentiment_Score', 'Sentiment_Category', 'Confidence_Score',
            'Primary_Market_Factor', 'News_Volume', 'Social_Media_Mentions',
            'Analyst_Coverage', 'Market_Volatility_Index', 'Sector_Performance',
            'Market_Phase', 'Company_Name', 'Sector', 'Market_Cap_Category'
        ]
        
        for col in sentiment_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(method='ffill')
        
        # Fill remaining NaN values with defaults
        if 'Sentiment_Score' in merged_df.columns:
            merged_df['Sentiment_Score'] = merged_df['Sentiment_Score'].fillna(0.0)
        if 'Sentiment_Category' in merged_df.columns:
            merged_df['Sentiment_Category'] = merged_df['Sentiment_Category'].fillna('Neutral')
        if 'Confidence_Score' in merged_df.columns:
            merged_df['Confidence_Score'] = merged_df['Confidence_Score'].fillna(0.5)
        
        # Add derived features
        if 'Sentiment_Score' in merged_df.columns:
            merged_df['sentiment_momentum'] = merged_df['Sentiment_Score'].diff()
            merged_df['sentiment_sma_5'] = merged_df['Sentiment_Score'].rolling(window=5).mean()
            merged_df['sentiment_volatility'] = merged_df['Sentiment_Score'].rolling(window=10).std()
        
        # Add time-based features
        merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['quarter'] = merged_df['date'].dt.quarter
        merged_df['year'] = merged_df['date'].dt.year
        merged_df['is_month_end'] = merged_df['date'].dt.is_month_end.astype(int)
        merged_df['is_quarter_end'] = merged_df['date'].dt.is_quarter_end.astype(int)
        
        # Add lag features for prediction (reduced to save memory)
        for lag in [1, 2, 3, 5]:
            merged_df[f'close_lag_{lag}'] = merged_df['close'].shift(lag)
            if 'Sentiment_Score' in merged_df.columns:
                merged_df[f'sentiment_lag_{lag}'] = merged_df['Sentiment_Score'].shift(lag)
        
        # Add future targets (reduced to save memory)
        for horizon in [1, 2, 3, 5]:
            merged_df[f'close_future_{horizon}'] = merged_df['close'].shift(-horizon)
            merged_df[f'return_future_{horizon}'] = (merged_df[f'close_future_{horizon}'] / merged_df['close'] - 1)
        
        logger.info(f"Merged dataset for {symbol}: {len(merged_df)} records, {len(merged_df.columns)} features")
        
        return merged_df
    
    def create_dataset_for_symbol(self, symbol, sentiment_df):
        """Create complete temporal dataset for a single symbol"""
        logger.info(f"Creating dataset for {symbol}...")
        
        # Load stock data
        stock_df = self.load_stock_data_for_symbol(symbol)
        if stock_df is None:
            return None
            
        # Add technical indicators
        stock_df = self.create_technical_indicators(stock_df)
        
        # Merge with sentiment data
        merged_df = self.merge_sentiment_and_stock_data(sentiment_df, stock_df, symbol)
        if merged_df is None:
            return None
            
        # Add symbol identifier
        merged_df['symbol'] = symbol
        
        # Create dataset metadata
        metadata = {
            'symbol': symbol,
            'date_range': {
                'start': merged_df['date'].min().isoformat(),
                'end': merged_df['date'].max().isoformat()
            },
            'total_records': len(merged_df),
            'features': list(merged_df.columns),
            'feature_count': len(merged_df.columns),
            'missing_data_percentage': round((merged_df.isnull().sum().sum() / (len(merged_df) * len(merged_df.columns))) * 100, 2),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        if not merged_df.empty and 'Company_Name' in merged_df.columns:
            metadata['company_name'] = merged_df['Company_Name'].iloc[0] if pd.notna(merged_df['Company_Name'].iloc[0]) else symbol
        if not merged_df.empty and 'Sector' in merged_df.columns:
            metadata['sector'] = merged_df['Sector'].iloc[0] if pd.notna(merged_df['Sector'].iloc[0]) else 'Unknown'
        if not merged_df.empty and 'Market_Cap_Category' in merged_df.columns:
            metadata['market_cap_category'] = merged_df['Market_Cap_Category'].iloc[0] if pd.notna(merged_df['Market_Cap_Category'].iloc[0]) else 'Unknown'
        
        return merged_df, metadata
    
    def save_dataset(self, df, metadata, symbol):
        """Save dataset and metadata to files with memory optimization"""
        try:
            # Save main dataset with chunking for large datasets
            dataset_file = self.output_dir / "stock_specific" / f"{symbol}_temporal_dataset_v5.csv"
            
            # Convert all object columns to string to avoid issues
            for col in df.select_dtypes(include=['object']).columns:
                if col != 'date':  # Don't convert date column
                    df[col] = df[col].astype(str)
            
            # Save in chunks if dataset is large
            if len(df) > 1000:
                df.to_csv(dataset_file, index=False, chunksize=1000)
            else:
                df.to_csv(dataset_file, index=False)
            
            # Save metadata
            metadata_file = self.output_dir / "stock_specific" / f"{symbol}_metadata_v5.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved dataset for {symbol}: {len(df)} records")
            
            # Force garbage collection
            del df
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving dataset for {symbol}: {e}")
            return False
    
    def run(self):
        """Main execution function"""
        logger.info("Starting V5 Temporal Dataset Creation...")
        
        # Load sentiment data
        sentiment_df = self.load_sentiment_data()
        
        # Get unique symbols (after normalization)
        unique_symbols = sorted(sentiment_df['Symbol'].unique())
        logger.info(f"Processing {len(unique_symbols)} unique symbols")
        
        successful_datasets = 0
        failed_datasets = 0
        
        # Process each symbol
        for i, symbol in enumerate(unique_symbols, 1):
            logger.info(f"Processing {i}/{len(unique_symbols)}: {symbol}")
            
            try:
                result = self.create_dataset_for_symbol(symbol, sentiment_df)
                if result is not None:
                    df, metadata = result
                    if self.save_dataset(df, metadata, symbol):
                        successful_datasets += 1
                    else:
                        failed_datasets += 1
                else:
                    failed_datasets += 1
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_datasets += 1
            
            # Force garbage collection every 10 symbols
            if i % 10 == 0:
                gc.collect()
                logger.info(f"Processed {i}/{len(unique_symbols)} symbols. Memory cleanup performed.")
        
        # Create summary report
        summary = {
            'total_symbols_processed': len(unique_symbols),
            'successful_datasets': successful_datasets,
            'failed_datasets': failed_datasets,
            'success_rate': round((successful_datasets / len(unique_symbols)) * 100, 1),
            'date_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat()
            },
            'output_directory': str(self.output_dir),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "dataset_creation_summary_v5.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("DATASET CREATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total symbols processed: {len(unique_symbols)}")
        logger.info(f"Successful datasets: {successful_datasets}")
        logger.info(f"Failed datasets: {failed_datasets}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)

if __name__ == "__main__":
    creator = TemporalDatasetCreator()
    creator.run()
