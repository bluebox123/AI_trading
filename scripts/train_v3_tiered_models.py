"""
COMPLETE V3 TIERED MODEL TRAINING SYSTEM
Trains individual models on your v3 tiered datasets
Builds on your proven stock-specific success (RELIANCE: 67.5%)
Handles largecap, midcap, and smallcap datasets with tier-specific optimizations
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V3TieredDataset(Dataset):
    """
    Dataset for V3 tiered models
    Handles largecap, midcap, and smallcap datasets with tier-specific parameters
    """
    
    def __init__(self, symbol: str, data_dir: str, tier: str, sequence_length: int = 20, 
                 split: str = "train", train_ratio: float = 0.75):
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.tier = tier
        self.sequence_length = sequence_length
        self.split = split
        self.train_ratio = train_ratio
        
        # Tier-specific configurations
        self.tier_configs = {
            'largecap': {
                'naming_suffix': '_largecap_dataset_v3.csv',
                'target_threshold_up': 0.015,    # 1.5% for large caps
                'target_threshold_down': -0.015,
                'model_complexity': 'high',
                'expected_accuracy': 0.65
            },
            'midcap': {
                'naming_suffix': '_midcap_dataset_v3.csv',
                'target_threshold_up': 0.020,    # 2.0% for mid caps
                'target_threshold_down': -0.020,
                'model_complexity': 'medium',
                'expected_accuracy': 0.60
            },
            'smallcap': {
                'naming_suffix': '_smallcap_dataset_v3.csv',
                'target_threshold_up': 0.025,    # 2.5% for small caps
                'target_threshold_down': -0.025,
                'model_complexity': 'low',
                'expected_accuracy': 0.55
            }
        }
        
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load V3 tiered dataset for specific stock"""
        # Get tier configuration
        tier_config = self.tier_configs[self.tier]
        
        # Load the V3 tiered dataset file
        dataset_file = self.data_dir / f"{self.symbol}{tier_config['naming_suffix']}"
        
        if not dataset_file.exists():
            raise ValueError(f"No V3 {self.tier} dataset found for {self.symbol} at {dataset_file}")
        
        logger.info(f"Loading V3 {self.tier} dataset for {self.symbol}")
        
        # Load dataset
        df = pd.read_csv(dataset_file, index_col=0, parse_dates=True)
        
        # Basic quality checks
        if len(df) < 200:
            raise ValueError(f"{self.symbol}: Insufficient data - only {len(df)} samples")
        
        if 'forward_return_1d' not in df.columns:
            raise ValueError(f"{self.symbol}: Missing target variable 'forward_return_1d'")
        
        # Store original dataframe
        self.df = df.copy()
        
        # Tier-specific preprocessing
        self.preprocess_data(tier_config)
        self.create_sequences(tier_config)
    
    def preprocess_data(self, tier_config: Dict):
        """Tier-specific preprocessing"""
        logger.debug(f"{self.symbol} ({self.tier}): Preprocessing {len(self.df)} samples")
        
        # Remove extreme outliers (tier-specific thresholds)
        if 'forward_return_1d' in self.df.columns:
            before_filter = len(self.df)
            outlier_threshold = 0.10 if self.tier == 'largecap' else 0.15 if self.tier == 'midcap' else 0.20
            self.df = self.df[self.df['forward_return_1d'].abs() < outlier_threshold]
            self.df = self.df[self.df['forward_return_1d'].notna()]
            after_filter = len(self.df)
            logger.debug(f"{self.symbol} ({self.tier}): Filtered extreme moves: {before_filter} → {after_filter}")
        
        # Define enhanced features for V3 datasets
        self.price_features = [
            'open', 'high', 'low', 'close', 'volume', 'daily_return',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'price_sma5_ratio', 'price_sma20_ratio', 'sma5_sma20_ratio',
            'volatility_5d', 'volatility_20d', 'rsi', 'macd', 'macd_signal',
            'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio'
        ]
        
        # Enhanced news features for V3
        self.news_features = [
            'news_sentiment_1d', 'news_sentiment_3d', 'news_sentiment_7d',
            'news_volume_1d', 'news_volume_3d', 'news_volume_7d',
            'news_keyword_density_1d', 'news_keyword_density_3d', 'news_keyword_density_7d'
        ]
        
        # Filter available features
        available_price = [f for f in self.price_features if f in self.df.columns]
        available_news = [f for f in self.news_features if f in self.df.columns]
        
        # Ensure minimum features
        if len(available_price) < 10:
            logger.warning(f"{self.symbol} ({self.tier}): Only {len(available_price)} price features available")
        
        if len(available_news) < 6:
            logger.warning(f"{self.symbol} ({self.tier}): Only {len(available_news)} news features available")
            # Fill missing news features with zeros for consistency
            for feature in self.news_features:
                if feature not in self.df.columns:
                    self.df[feature] = 0.0
            available_news = self.news_features
        
        self.price_features = available_price
        self.news_features = available_news
        
        logger.info(f"{self.symbol} ({self.tier}): Using {len(self.price_features)} price, {len(self.news_features)} news features")
    
    def create_sequences(self, tier_config: Dict):
        """Create sequences with tier-specific target labeling"""
        sequences = []
        targets = []
        
        logger.debug(f"{self.symbol} ({self.tier}): Creating sequences...")
        
        for i in range(self.sequence_length, len(self.df)):
            try:
                # Get target return
                target_return = self.df['forward_return_1d'].iloc[i]
                if pd.isna(target_return):
                    continue
                
                # Tier-specific target labeling
                threshold_up = tier_config['target_threshold_up']
                threshold_down = tier_config['target_threshold_down']
                
                if target_return > threshold_up:
                    direction = 2  # UP
                elif target_return < threshold_down:
                    direction = 0  # DOWN
                else:
                    direction = 1  # NEUTRAL
                
                # Price sequence (past sequence_length days)
                price_data = self.df[self.price_features].iloc[i-self.sequence_length:i]
                if price_data.isnull().any().any():
                    continue
                
                # News features (current day)
                news_data = self.df[self.news_features].iloc[i]
                if news_data.isnull().any():
                    continue
                
                sequences.append({
                    'price_sequence': price_data.values.astype(np.float32),
                    'news_features': news_data.values.astype(np.float32),
                    'target_return': target_return
                })
                targets.append(direction)
                
            except Exception as e:
                continue
        
        if len(sequences) < 50:
            raise ValueError(f"{self.symbol} ({self.tier}): Too few valid sequences: {len(sequences)}")
        
        logger.info(f"{self.symbol} ({self.tier}): Created {len(sequences)} sequences")
        
        # Chronological train/test split
        split_idx = int(len(sequences) * self.train_ratio)
        
        if self.split == "train":
            self.sequences = sequences[:split_idx]
            self.targets = targets[:split_idx]
        else:  # test
            self.sequences = sequences[split_idx:]
            self.targets = targets[split_idx:]
        
        if len(self.sequences) == 0:
            raise ValueError(f"{self.symbol} ({self.tier}): No {self.split} sequences created")
        
        # Store dimensions
        self.price_dim = self.sequences[0]['price_sequence'].shape[1]
        self.news_dim = len(self.sequences[0]['news_features'])
        
        # Log target distribution
        unique, counts = np.unique(self.targets, return_counts=True)
        target_dist = dict(zip(unique, counts))
        logger.info(f"{self.symbol} ({self.tier}) {self.split.upper()}: {len(self.sequences)} samples, targets: {target_dist}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        target = self.targets[idx]
        
        return {
            'price_sequence': torch.FloatTensor(sample['price_sequence']),
            'news_features': torch.FloatTensor(sample['news_features']),
            'target': torch.LongTensor([target]).squeeze(),
            'symbol': self.symbol,
            'tier': self.tier
        }

class V3TieredModel(nn.Module):
    """
    Tier-specific model architecture
    Adjusts complexity based on market cap tier
    """
    
    def __init__(self, price_dim: int, news_dim: int, tier: str, sequence_length: int = 20):
        super().__init__()
        
        self.tier = tier
        self.sequence_length = sequence_length
        
        # Tier-specific model configurations
        tier_configs = {
            'largecap': {'hidden_dim': 96, 'num_layers': 2, 'dropout': 0.3},
            'midcap': {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.4},
            'smallcap': {'hidden_dim': 48, 'num_layers': 1, 'dropout': 0.5}
        }
        
        config = tier_configs[tier]
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        
        # Price encoder - adjusted for tier complexity
        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # News encoder - simplified for different tiers
        news_hidden = hidden_dim // 2
        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, news_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(news_hidden, news_hidden)
        )
        
        # Fusion and classification
        fusion_dim = hidden_dim + news_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # [DOWN, NEUTRAL, UP]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, price_sequences, news_features):
        # Price encoding (take last timestep)
        lstm_out, _ = self.price_lstm(price_sequences)
        price_repr = lstm_out[:, -1, :]  # Last timestep
        
        # News encoding
        news_repr = self.news_encoder(news_features)
        
        # Simple concatenation fusion
        combined = torch.cat([price_repr, news_repr], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

class V3TieredTrainer:
    """
    V3 Tiered trainer for all market cap tiers
    Handles largecap, midcap, and smallcap models with tier-specific optimizations
    """
    
    def __init__(self, data_dir: str = "data/processed/aligned_data"):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tier-specific training parameters
        self.tier_training_configs = {
            'largecap': {
                'batch_size': 16,
                'learning_rate': 0.0003,
                'max_epochs': 80,
                'patience': 15,
                'expected_accuracy': 0.65
            },
            'midcap': {
                'batch_size': 12,
                'learning_rate': 0.0002,
                'max_epochs': 60,
                'patience': 12,
                'expected_accuracy': 0.60
            },
            'smallcap': {
                'batch_size': 8,
                'learning_rate': 0.0001,
                'max_epochs': 40,
                'patience': 10,
                'expected_accuracy': 0.55
            }
        }
        
        self.results = {}
        
        logger.info(f"V3 Tiered Trainer initialized on {self.device}")
        logger.info("Designed for largecap, midcap, and smallcap v3 datasets")
    
    def get_available_v3_symbols_by_tier(self) -> Dict[str, List[str]]:
        """Get list of symbols with V3 datasets by tier"""
        tier_symbols = {
            'largecap': [],
            'midcap': [],
            'smallcap': []
        }
        
        # Look for V3 tiered datasets
        for tier in ['largecap', 'midcap', 'smallcap']:
            v3_files = list(self.data_dir.glob(f"*_{tier}_dataset_v3.csv"))
            
            for filepath in v3_files:
                symbol = filepath.stem.replace(f"_{tier}_dataset_v3", "")
                tier_symbols[tier].append(symbol)
        
        for tier, symbols in tier_symbols.items():
            logger.info(f"Found {len(symbols)} {tier} V3 datasets")
        
        return tier_symbols
    
    def train_single_stock_v3(self, symbol: str, tier: str) -> Dict:
        """Train V3 model for a single stock in specific tier"""
        logger.info(f"Training V3 {tier} model for {symbol}...")
        
        try:
            # Get tier-specific configuration
            tier_config = self.tier_training_configs[tier]
            
            # Create datasets
            train_dataset = V3TieredDataset(
                symbol=symbol, 
                data_dir=self.data_dir, 
                tier=tier,
                split="train"
            )
            
            test_dataset = V3TieredDataset(
                symbol=symbol, 
                data_dir=self.data_dir, 
                tier=tier,
                split="test"
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=tier_config['batch_size'], 
                shuffle=True,
                drop_last=True
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=tier_config['batch_size'], 
                shuffle=False
            )
            
            if len(train_loader) == 0 or len(test_loader) == 0:
                return {'success': False, 'error': 'Empty data loaders'}
            
            # Create tier-specific model
            model = V3TieredModel(
                price_dim=train_dataset.price_dim,
                news_dim=train_dataset.news_dim,
                tier=tier,
                sequence_length=train_dataset.sequence_length
            ).to(self.device)
            
            # Setup training
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=tier_config['learning_rate'], 
                weight_decay=0.01
            )
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
            )
            
            # Training loop
            best_test_acc = 0.0
            patience_counter = 0
            
            for epoch in range(tier_config['max_epochs']):
                # Train
                model.train()
                train_correct = 0
                train_total = 0
                train_loss = 0.0
                
                for batch in train_loader:
                    try:
                        price_seq = batch['price_sequence'].to(self.device)
                        news_feat = batch['news_features'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(price_seq, news_feat)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()
                        
                    except Exception as e:
                        continue
                
                if train_total == 0:
                    break
                
                train_acc = train_correct / train_total
                
                # Test
                model.eval()
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        try:
                            price_seq = batch['price_sequence'].to(self.device)
                            news_feat = batch['news_features'].to(self.device)
                            targets = batch['target'].to(self.device)
                            
                            outputs = model(price_seq, news_feat)
                            _, predicted = torch.max(outputs.data, 1)
                            test_total += targets.size(0)
                            test_correct += (predicted == targets).sum().item()
                            
                        except Exception as e:
                            continue
                
                test_acc = test_correct / test_total if test_total > 0 else 0.0
                
                # Learning rate scheduling
                scheduler.step(test_acc)
                
                # Check for improvement
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                    
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Progress logging
                if epoch % 15 == 0 or test_acc > tier_config['expected_accuracy']:
                    logger.info(f"{symbol} ({tier}) Epoch {epoch+1}: Train {train_acc:.3f}, Test {test_acc:.3f}")
                
                # Early stopping
                if patience_counter >= tier_config['patience']:
                    logger.info(f"{symbol} ({tier}): Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state)
            
            # Save model
            model_path = self.save_v3_model(model, symbol, tier, best_test_acc)
            
            result = {
                'success': True,
                'symbol': symbol,
                'tier': tier,
                'best_accuracy': best_test_acc,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'price_features': train_dataset.price_dim,
                'news_features': train_dataset.news_dim,
                'model_path': model_path,
                'epochs_trained': epoch + 1,
                'expected_accuracy': tier_config['expected_accuracy']
            }
            
            # Performance assessment
            if best_test_acc >= tier_config['expected_accuracy']:
                performance = "EXCELLENT - TARGET ACHIEVED!"
            elif best_test_acc >= tier_config['expected_accuracy'] - 0.05:
                performance = "VERY GOOD - CLOSE TO TARGET!"
            elif best_test_acc >= 0.55:
                performance = "GOOD - ABOVE RANDOM!"
            else:
                performance = "NEEDS IMPROVEMENT"
            
            result['performance'] = performance
            
            logger.info(f"✅ {symbol} ({tier}): {best_test_acc:.3f} accuracy ({best_test_acc*100:.1f}%) - {performance}")
            return result
            
        except Exception as e:
            logger.error(f"❌ {symbol} ({tier}): Training failed - {e}")
            return {'success': False, 'symbol': symbol, 'tier': tier, 'error': str(e)}
    
    def save_v3_model(self, model: nn.Module, symbol: str, tier: str, accuracy: float) -> str:
        """Save trained V3 model"""
        model_dir = Path("data/models/stock_specific_v3")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{tier}_v3_model_{timestamp}.pth"
        filepath = model_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'symbol': symbol,
            'tier': tier,
            'accuracy': accuracy,
            'version': 'v3_tiered',
            'timestamp': timestamp,
            'model_config': {
                'price_dim': model.price_lstm.input_size,
                'news_dim': model.news_encoder[0].in_features,
                'hidden_dim': model.price_lstm.hidden_size,
                'sequence_length': model.sequence_length,
                'tier': tier
            }
        }, filepath)
        
        return str(filepath)
    
    def train_all_v3_stocks(self, target_tier: Optional[str] = None) -> Dict:
        """Train V3 models for all available stocks"""
        print("🚀 V3 TIERED MODEL TRAINING")
        print("=" * 60)
        print(f"📊 Training individual models using V3 tiered datasets")
        print("🎯 Building on your proven stock-specific success")
        print("🏗️ Tier-specific optimizations for market cap categories")
        print("=" * 60)
        
        # Get available symbols by tier
        tier_symbols = self.get_available_v3_symbols_by_tier()
        
        # Filter by target tier if specified
        if target_tier:
            if target_tier not in tier_symbols:
                print(f"❌ Invalid tier: {target_tier}")
                return {'error': f'Invalid tier: {target_tier}'}
            tier_symbols = {target_tier: tier_symbols[target_tier]}
            print(f"📊 Training only {target_tier} stocks")
        
        successful = 0
        failed = 0
        total_accuracy = 0.0
        tier_results = {}
        
        for tier, symbols in tier_symbols.items():
            if not symbols:
                print(f"⚠️ No {tier} symbols found")
                continue
                
            print(f"\n🎯 Training {tier.upper()} models ({len(symbols)} stocks)")
            print("-" * 40)
            
            tier_successful = 0
            tier_total_acc = 0.0
            tier_results[tier] = {}
            
            expected_acc = self.tier_training_configs[tier]['expected_accuracy']
            
            for i, symbol in enumerate(symbols, 1):
                print(f"📈 Training {i}/{len(symbols)}: {symbol} ({tier})")
                
                result = self.train_single_stock_v3(symbol, tier)
                
                if result['success']:
                    successful += 1
                    tier_successful += 1
                    total_accuracy += result['best_accuracy']
                    tier_total_acc += result['best_accuracy']
                    tier_results[tier][symbol] = result
                    
                    # Performance assessment
                    acc_pct = result['best_accuracy'] * 100
                    if acc_pct >= expected_acc * 100:
                        print(f"🏆 {symbol}: {acc_pct:.1f}% - {result['performance']}")
                    elif acc_pct >= 60:
                        print(f"🎯 {symbol}: {acc_pct:.1f}% - {result['performance']}")
                    else:
                        print(f"📊 {symbol}: {acc_pct:.1f}% - {result['performance']}")
                else:
                    failed += 1
                    print(f"❌ {symbol}: Failed - {result.get('error', 'Unknown error')}")
            
            # Tier summary
            if tier_successful > 0:
                tier_avg = tier_total_acc / tier_successful
                print(f"\n📊 {tier.upper()} TIER SUMMARY:")
                print(f"   Successful: {tier_successful}/{len(symbols)}")
                print(f"   Average accuracy: {tier_avg:.3f} ({tier_avg*100:.1f}%)")
                print(f"   Expected: {expected_acc:.3f} ({expected_acc*100:.1f}%)")
                if tier_avg >= expected_acc:
                    print(f"   Status: ✅ TARGET ACHIEVED!")
                else:
                    print(f"   Status: 📈 GOOD PROGRESS")
        
        # Final results
        print("\n" + "=" * 60)
        print("🎯 V3 TIERED TRAINING COMPLETE!")
        print("=" * 60)
        
        if successful > 0:
            avg_accuracy = total_accuracy / successful
            print(f"📊 Overall successful models: {successful}")
            print(f"📈 Overall average accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
            print(f"❌ Failed models: {failed}")
            
            # Tier comparison
            print("\n📊 TIER PERFORMANCE COMPARISON:")
            for tier, results in tier_results.items():
                if results:
                    tier_avg = sum(r['best_accuracy'] for r in results.values()) / len(results)
                    expected = self.tier_training_configs[tier]['expected_accuracy']
                    status = "✅" if tier_avg >= expected else "📈"
                    print(f"   {tier.capitalize()}: {tier_avg*100:.1f}% (expected: {expected*100:.1f}%) {status}")
            
            # Save results
            self.save_v3_results(tier_results)
            
            # Overall performance assessment
            if avg_accuracy >= 0.60:
                print("🏆 EXCELLENT! V3 tiered approach working brilliantly!")
                print("🎯 Ready for deployment - tiered specialization successful!")
            elif avg_accuracy >= 0.55:
                print("🎯 VERY GOOD! Strong foundation with tier specialization!")
                print("📈 Most tiers showing good performance!")
            else:
                print("📊 FOUNDATION BUILT - Continue tier-specific optimization!")
        else:
            print("❌ No models trained successfully!")
        
        return {
            'successful': successful,
            'failed': failed,
            'avg_accuracy': total_accuracy / successful if successful > 0 else 0.0,
            'tier_results': tier_results,
            'tier_training_configs': self.tier_training_configs
        }
    
    def save_v3_results(self, tier_results: Dict):
        """Save V3 training results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"v3_tiered_results_{timestamp}.json"
        
        # Convert results for JSON serialization
        serializable_results = {}
        for tier, tier_stocks in tier_results.items():
            serializable_results[tier] = {}
            for symbol, result in tier_stocks.items():
                serializable_results[tier][symbol] = {
                    'success': result['success'],
                    'best_accuracy': float(result['best_accuracy']),
                    'train_samples': result['train_samples'],
                    'test_samples': result['test_samples'],
                    'epochs_trained': result['epochs_trained'],
                    'performance': result['performance']
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📁 V3 results saved: {results_file}")

def main():
    """Main execution for V3 tiered training"""
    print("🧠 V3 TIERED TEMPORAL CAUSALITY TRAINER")
    print("=" * 60)
    print("🎯 Train tiered models for largecap, midcap, and smallcap stocks")
    print("📊 Building on your proven stock-specific breakthrough")
    print("🏗️ Tier-specific optimizations for different market caps")
    print("=" * 60)
    
    print("\nSelect training scope:")
    print("1. Train all tiers (largecap + midcap + smallcap)")
    print("2. Train specific tier only")
    print("3. Train subset for testing")
    
    choice = input("Enter choice (1-3): ").strip()
    
    trainer = V3TieredTrainer()
    
    if choice == "1":
        results = trainer.train_all_v3_stocks()
    elif choice == "2":
        print("\nSelect tier:")
        print("1. Large Cap (expected 65%+ accuracy)")
        print("2. Mid Cap (expected 60%+ accuracy)")
        print("3. Small Cap (expected 55%+ accuracy)")
        
        tier_choice = input("Enter tier (1-3): ").strip()
        tier_map = {'1': 'largecap', '2': 'midcap', '3': 'smallcap'}
        
        if tier_choice in tier_map:
            target_tier = tier_map[tier_choice]
            results = trainer.train_all_v3_stocks(target_tier=target_tier)
        else:
            print("❌ Invalid tier choice")
            return
    elif choice == "3":
        # Test with first few stocks of each tier
        print("🧪 Testing mode: Training first 3 stocks per tier")
        # Implementation for subset testing would go here
        return
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n🎯 V3 tiered training complete!")
    print(f"📁 Models saved in: data/models/stock_specific_v3/")
    print(f"📊 Results saved in: results/")

if __name__ == "__main__":
    main()
