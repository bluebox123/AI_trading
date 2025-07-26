"""
Data persistence module for news sentiment pipeline
Handles saving and loading of sentiment analysis results
"""
import csv
import os
import logging
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

from config.config import (
    NEWS_SENTIMENT_CSV, DATABASE_URL, USE_DATABASE, DRY_RUN
)

class DataPersistence:
    """
    Handles data persistence for news sentiment analysis results
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.csv_file = NEWS_SENTIMENT_CSV
        self.db_url = DATABASE_URL
        self.use_database = USE_DATABASE
        
        if self.use_database:
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_url) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_sentiment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        headline TEXT NOT NULL,
                        link TEXT UNIQUE NOT NULL,
                        pub_date TEXT,
                        description TEXT,
                        source TEXT,
                        fetch_timestamp TEXT,
                        sentiment TEXT NOT NULL,
                        sentiment_confidence REAL,
                        sentiment_probabilities TEXT,
                        sentiment_is_confident BOOLEAN,
                        sentiment_model TEXT,
                        sentiment_timestamp TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index on ticker and timestamp for efficient queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
                    ON news_sentiment(ticker, sentiment_timestamp)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_link 
                    ON news_sentiment(link)
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_records(self, records: List[Dict]) -> bool:
        """
        Save sentiment analysis records to storage
        """
        if not records:
            self.logger.info("No records to save")
            return True
        
        if DRY_RUN:
            self.logger.info(f"DRY RUN: Would save {len(records)} records")
            return True
        
        try:
            if self.use_database:
                return self._save_to_database(records)
            else:
                return self._save_to_csv(records)
        except Exception as e:
            self.logger.error(f"Failed to save records: {e}")
            return False
    
    def _save_to_csv(self, records: List[Dict]) -> bool:
        """Save records to CSV file"""
        try:
            # Check if file exists to determine if we need headers
            file_exists = os.path.isfile(self.csv_file)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                if records:
                    # Use the first record to determine fieldnames
                    fieldnames = self._get_csv_fieldnames(records[0])
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # Write header if file is new
                    if not file_exists:
                        writer.writeheader()
                    
                    # Convert complex fields to JSON strings for CSV
                    csv_records = []
                    for record in records:
                        csv_record = record.copy()
                        if 'sentiment_probabilities' in csv_record:
                            csv_record['sentiment_probabilities'] = json.dumps(
                                csv_record['sentiment_probabilities']
                            )
                        csv_records.append(csv_record)
                    
                    writer.writerows(csv_records)
            
            self.logger.info(f"Saved {len(records)} records to CSV: {self.csv_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            return False
    
    def _save_to_database(self, records: List[Dict]) -> bool:
        """Save records to SQLite database"""
        try:
            with sqlite3.connect(self.db_url) as conn:
                cursor = conn.cursor()
                
                for record in records:
                    # Convert probabilities dict to JSON string
                    probabilities_json = json.dumps(
                        record.get('sentiment_probabilities', {})
                    )
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO news_sentiment (
                            ticker, headline, link, pub_date, description, source,
                            fetch_timestamp, sentiment, sentiment_confidence,
                            sentiment_probabilities, sentiment_is_confident,
                            sentiment_model, sentiment_timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.get('ticker'),
                        record.get('headline'),
                        record.get('link'),
                        record.get('pub_date'),
                        record.get('description'),
                        record.get('source'),
                        record.get('fetch_timestamp'),
                        record.get('sentiment'),
                        record.get('sentiment_confidence'),
                        probabilities_json,
                        record.get('sentiment_is_confident'),
                        record.get('sentiment_model'),
                        record.get('sentiment_timestamp')
                    ))
                
                conn.commit()
            
            self.logger.info(f"Saved {len(records)} records to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
    
    def _get_csv_fieldnames(self, sample_record: Dict) -> List[str]:
        """Get consistent fieldnames for CSV from a sample record"""
        base_fields = [
            'ticker', 'headline', 'link', 'pub_date', 'description', 'source',
            'fetch_timestamp', 'sentiment', 'sentiment_confidence',
            'sentiment_probabilities', 'sentiment_is_confident',
            'sentiment_model', 'sentiment_timestamp'
        ]
        
        # Add any additional fields that might be present
        additional_fields = [k for k in sample_record.keys() if k not in base_fields]
        
        return base_fields + additional_fields
    
    def load_recent_records(self, hours: int = 24, ticker: Optional[str] = None) -> List[Dict]:
        """
        Load recent records from storage
        """
        try:
            if self.use_database:
                return self._load_from_database(hours, ticker)
            else:
                return self._load_from_csv(hours, ticker)
        except Exception as e:
            self.logger.error(f"Failed to load records: {e}")
            return []
    
    def _load_from_csv(self, hours: int, ticker: Optional[str] = None) -> List[Dict]:
        """Load records from CSV file"""
        if not os.path.exists(self.csv_file):
            return []
        
        try:
            df = pd.read_csv(self.csv_file)
            
            # Filter by ticker if specified
            if ticker:
                df = df[df['ticker'] == ticker]
            
            # Filter by time if possible
            if 'sentiment_timestamp' in df.columns:
                cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
                df['timestamp_parsed'] = pd.to_datetime(df['sentiment_timestamp'])
                df = df[df['timestamp_parsed'] >= pd.to_datetime(cutoff_time, unit='s')]
            
            # Convert back to list of dicts
            records = df.to_dict('records')
            
            # Parse JSON fields back to dicts
            for record in records:
                if 'sentiment_probabilities' in record and isinstance(record['sentiment_probabilities'], str):
                    try:
                        record['sentiment_probabilities'] = json.loads(record['sentiment_probabilities'])
                    except:
                        record['sentiment_probabilities'] = {}
            
            self.logger.info(f"Loaded {len(records)} records from CSV")
            return records
            
        except Exception as e:
            self.logger.error(f"Error loading from CSV: {e}")
            return []
    
    def _load_from_database(self, hours: int, ticker: Optional[str] = None) -> List[Dict]:
        """Load records from SQLite database"""
        try:
            with sqlite3.connect(self.db_url) as conn:
                query = '''
                    SELECT * FROM news_sentiment 
                    WHERE datetime(sentiment_timestamp) >= datetime('now', '-{} hours')
                '''.format(hours)
                
                params = []
                if ticker:
                    query += ' AND ticker = ?'
                    params.append(ticker)
                
                query += ' ORDER BY sentiment_timestamp DESC'
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    
                    # Parse JSON probabilities
                    if record.get('sentiment_probabilities'):
                        try:
                            record['sentiment_probabilities'] = json.loads(
                                record['sentiment_probabilities']
                            )
                        except:
                            record['sentiment_probabilities'] = {}
                    
                    records.append(record)
            
            self.logger.info(f"Loaded {len(records)} records from database")
            return records
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}")
            return []
    
    def get_ticker_summary(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get sentiment summary for a specific ticker
        """
        records = self.load_recent_records(hours, ticker)
        
        if not records:
            return {
                'ticker': ticker,
                'total_articles': 0,
                'sentiment_distribution': {},
                'average_confidence': 0.0,
                'period_hours': hours
            }
        
        # Calculate statistics
        sentiments = [r.get('sentiment') for r in records if r.get('sentiment')]
        confidences = [r.get('sentiment_confidence', 0) for r in records]
        
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            'ticker': ticker,
            'total_articles': len(records),
            'sentiment_distribution': sentiment_counts,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'period_hours': hours,
            'latest_timestamp': max([r.get('sentiment_timestamp', '') for r in records]) if records else None
        }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Clean up old records to prevent storage from growing too large
        """
        try:
            if self.use_database:
                return self._cleanup_database(days)
            else:
                return self._cleanup_csv(days)
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            return 0
    
    def _cleanup_database(self, days: int) -> int:
        """Clean up old records from database"""
        try:
            with sqlite3.connect(self.db_url) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM news_sentiment 
                    WHERE datetime(sentiment_timestamp) < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old records from database")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up database: {e}")
            return 0
    
    def _cleanup_csv(self, days: int) -> int:
        """Clean up old records from CSV file"""
        if not os.path.exists(self.csv_file):
            return 0
        
        try:
            df = pd.read_csv(self.csv_file)
            original_count = len(df)
            
            # Filter out old records
            if 'sentiment_timestamp' in df.columns:
                cutoff_time = datetime.utcnow().timestamp() - (days * 24 * 3600)
                df['timestamp_parsed'] = pd.to_datetime(df['sentiment_timestamp'])
                df_filtered = df[df['timestamp_parsed'] >= pd.to_datetime(cutoff_time, unit='s')]
                
                # Save filtered data back
                df_filtered.drop('timestamp_parsed', axis=1).to_csv(self.csv_file, index=False)
                
                deleted_count = original_count - len(df_filtered)
                self.logger.info(f"Cleaned up {deleted_count} old records from CSV")
                return deleted_count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error cleaning up CSV: {e}")
            return 0


if __name__ == "__main__":
    # Test the data persistence module
    logging.basicConfig(level=logging.INFO)
    
    persistence = DataPersistence()
    
    # Test data
    test_records = [
        {
            'ticker': 'RELIANCE.NS',
            'headline': 'Reliance Industries reports strong Q3 results',
            'link': 'https://example.com/news1',
            'pub_date': '2024-01-15T10:30:00Z',
            'sentiment': 'positive',
            'sentiment_confidence': 0.85,
            'sentiment_probabilities': {'negative': 0.05, 'neutral': 0.10, 'positive': 0.85},
            'sentiment_timestamp': datetime.utcnow().isoformat()
        }
    ]
    
    # Test saving
    success = persistence.save_records(test_records)
    print(f"Save successful: {success}")
    
    # Test loading
    recent_records = persistence.load_recent_records(hours=1)
    print(f"Loaded {len(recent_records)} recent records") 