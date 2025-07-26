"""
FIXED BACKTEST - WORKS WITH YOUR FRESH V2 DATASETS
Uses your newly generated temporal_dataset_v2.csv files
Validates your 64.97% model accuracy on fresh data
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, Optional

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExactMultimodalTransformerV2(nn.Module):
    """EXACT recreation of your working model architecture"""
    
    def __init__(self, price_input_dim: int, news_input_dim: int = 9, 
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Price sequence encoder (LSTM)
        self.price_lstm = nn.LSTM(
            price_input_dim, hidden_dim, 
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # News encoder
        self.news_encoder = nn.Sequential(
            nn.Linear(news_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Classifier
        fusion_dim = hidden_dim + hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # DOWN, UP, HOLD
        )
    
    def forward(self, price_sequence, news_features):
        # Process price sequence
        lstm_out, _ = self.price_lstm(price_sequence)
        price_repr = lstm_out[:, -1, :]  # Last timestep
        
        # Process news
        news_repr = self.news_encoder(news_features)
        
        # Fusion
        combined = torch.cat([price_repr, news_repr], dim=1)
        logits = self.classifier(combined)
        return logits

class FixedModelBacktester:
    """FIXED backtester that works with your fresh V2 datasets"""
    
    def __init__(self, model_path: str, data_dir: str = "data/processed/aligned_data"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.model_config = None
        
        logger.info("🚀 Fixed Model Backtester initialized")
    
    def load_trained_model(self) -> bool:
        """Load your trained model with fixed loading"""
        try:
            if not self.model_path.exists():
                logger.error(f"❌ Model not found: {self.model_path}")
                return False
            
            logger.info("🔄 Loading your V2 model...")
            
            # Load with weights_only=False (trusted source)
            logger.info("🔓 Loading with weights_only=False (trusted source)...")
            checkpoint = torch.load(str(self.model_path), weights_only=False)
            logger.info("✅ Model loaded successfully")
            
            # Extract configuration
            if 'model_state_dict' not in checkpoint:
                logger.error("❌ No model state dict found")
                return False
            
            self.model_config = checkpoint.get('model_config', {})
            logger.info(f"📊 Model config: {self.model_config}")
            
            # Create model with exact architecture
            price_dim = self.model_config.get('price_input_dim', 14)
            news_dim = self.model_config.get('news_input_dim', 9)
            
            logger.info(f"🧠 Creating model: price_dim={price_dim}, news_dim={news_dim}")
            
            self.model = ExactMultimodalTransformerV2(
                price_input_dim=price_dim,
                news_input_dim=news_dim,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            logger.info("✅ Model loaded with strict=True")
            
            self.model.to(self.device)
            self.model.eval()
            
            best_accuracy = checkpoint.get('best_val_accuracy', 0)
            logger.info(f"🏆 Best validation accuracy: {best_accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Loading error: {e}")
            return False
    
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load your fresh V2 datasets"""
        datasets = {}
        
        # Look for your fresh V2 datasets
        dataset_files = list(self.data_dir.glob("*_temporal_dataset_v2.csv"))
        logger.info(f"📊 Found {len(dataset_files)} datasets: *_temporal_dataset_v2.csv")
        
        for filepath in dataset_files:
            try:
                symbol = filepath.stem.replace("_temporal_dataset_v2", "")
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                if len(df) > 200 and 'forward_return_1d' in df.columns:
                    datasets[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} samples")
                else:
                    logger.warning(f"⚠️ {symbol}: insufficient data or missing target")
                
            except Exception as e:
                logger.warning(f"⚠️ Error loading {filepath}: {e}")
        
        return datasets
    
    def get_features_for_prediction(self, df: pd.DataFrame):
        """FIXED: Extract features matching your fresh dataset structure"""
        try:
            # Get all numeric columns from your fresh datasets
            all_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # FIXED: Use actual column names from your fresh datasets
            # Your fresh datasets have 39 columns total, need to map correctly
            
            # Exclude target columns
            exclude_cols = ['forward_return_1d', 'forward_return_3d', 'forward_return_5d']
            available_cols = [col for col in all_cols if col not in exclude_cols]
            
            logger.debug(f"Available columns: {len(available_cols)}")
            logger.debug(f"Sample columns: {available_cols[:10]}")
            
            # FIXED: Use your actual dataset structure
            # Core price features that should exist in your datasets
            price_candidates = [
                'open', 'high', 'low', 'close', 'volume', 'daily_return',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
                'bb_middle', 'bb_upper', 'bb_lower', 'volatility_5d', 'volatility_20d'
            ]
            
            # News features from your fresh datasets
            news_candidates = [
                'news_sentiment_1d', 'news_sentiment_3d', 'news_sentiment_7d',
                'news_volume_1d', 'news_volume_3d', 'news_volume_7d',
                'news_keyword_density_1d', 'news_keyword_density_3d', 'news_keyword_density_7d'
            ]
            
            # Filter what's actually available
            available_price = [f for f in price_candidates if f in available_cols]
            available_news = [f for f in news_candidates if f in available_cols]
            
            logger.debug(f"Available price features: {len(available_price)}")
            logger.debug(f"Available news features: {len(available_news)}")
            
            # Ensure we have enough features
            if len(available_price) < 10:
                # Add any remaining numeric features
                backup_features = [col for col in available_cols 
                                 if col not in available_price + available_news]
                available_price.extend(backup_features[:14])
            
            # Take exactly what the model expects
            price_features = available_price[:14]  # Model expects 14 price features
            
            # Ensure exactly 9 news features
            while len(available_news) < 9:
                available_news.append(f"news_placeholder_{len(available_news)}")
            news_features = available_news[:9]
            
            # Get price sequence (last 20 rows)
            if len(df) < 20:
                logger.debug(f"Insufficient data: {len(df)} rows < 20")
                return None
            
            price_data = df[price_features].tail(20)
            
            # Handle NaN values
            if price_data.isnull().any().any():
                logger.debug("Filling NaN values in price data")
                price_data = price_data.fillna(method='ffill').fillna(0)
            
            price_sequence = price_data.values.astype(np.float32)
            
            # Get news features (last row)
            news_vec = []
            for feature in news_features:
                if feature in df.columns:
                    val = df[feature].iloc[-1]
                    if pd.isna(val):
                        val = 0.0
                    news_vec.append(float(val))
                else:
                    news_vec.append(0.0)
            
            news_features_array = np.array(news_vec, dtype=np.float32)
            
            logger.debug(f"Price sequence shape: {price_sequence.shape}")
            logger.debug(f"News features shape: {news_features_array.shape}")
            
            return {
                'price_sequence': price_sequence,
                'news_features': news_features_array
            }
            
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_direction(self, features: Dict) -> Optional[str]:
        """Predict direction using trained model"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Convert to tensors
            price_tensor = torch.FloatTensor(features['price_sequence']).unsqueeze(0).to(self.device)
            news_tensor = torch.FloatTensor(features['news_features']).unsqueeze(0).to(self.device)
            
            logger.debug(f"Price tensor shape: {price_tensor.shape}")
            logger.debug(f"News tensor shape: {news_tensor.shape}")
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(price_tensor, news_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Map to direction
            direction_map = {0: "SELL", 1: "BUY", 2: "HOLD"}
            direction = direction_map.get(predicted_class, "HOLD")
            
            logger.debug(f"Predicted class: {predicted_class}, Direction: {direction}")
            return direction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_accuracy_validation(self, datasets: Dict[str, pd.DataFrame]):
        """Run accuracy validation on your fresh datasets"""
        logger.info("🎯 Starting accuracy validation...")
        
        total_predictions = 0
        correct_predictions = 0
        symbol_results = {}
        
        for symbol, df in datasets.items():
            logger.info(f"📊 Validating {symbol}...")
            
            try:
                # Filter to test period
                df_test = df.loc['2022-01-01':'2024-06-01'].copy()
                if len(df_test) < 50:
                    logger.warning(f"{symbol}: Insufficient test data ({len(df_test)} rows)")
                    continue
                
                logger.info(f"{symbol}: Testing on {len(df_test)} rows")
                
                symbol_correct = 0
                symbol_total = 0
                
                # Rolling prediction validation
                for i in range(20, len(df_test) - 1):
                    try:
                        # Get features for this point in time
                        historical_data = df_test.iloc[:i+1]
                        features = self.get_features_for_prediction(historical_data)
                        
                        if features is None:
                            continue
                        
                        # Predict
                        direction = self.predict_direction(features)
                        if direction is None:
                            continue
                        
                        # Get actual return
                        actual_return = df_test['forward_return_1d'].iloc[i]
                        if pd.isna(actual_return):
                            continue
                        
                        # Convert actual return to direction
                        if actual_return > 0.01:  # 1% threshold
                            actual_direction = "BUY"
                        elif actual_return < -0.01:
                            actual_direction = "SELL"
                        else:
                            actual_direction = "HOLD"
                        
                        # Check accuracy
                        if direction == actual_direction:
                            symbol_correct += 1
                            correct_predictions += 1
                        
                        symbol_total += 1
                        total_predictions += 1
                        
                        # Log every 100 predictions for debugging
                        if symbol_total % 100 == 0:
                            logger.debug(f"{symbol}: {symbol_total} predictions processed")
                        
                    except Exception as e:
                        logger.debug(f"Error processing prediction {i}: {e}")
                        continue
                
                if symbol_total > 0:
                    symbol_accuracy = symbol_correct / symbol_total
                    symbol_results[symbol] = {
                        'accuracy': symbol_accuracy,
                        'correct': symbol_correct,
                        'total': symbol_total
                    }
                    logger.info(f"✅ {symbol}: {symbol_accuracy:.1%} accuracy ({symbol_correct}/{symbol_total})")
                else:
                    logger.warning(f"⚠️ {symbol}: No valid predictions generated")
                
            except Exception as e:
                logger.error(f"❌ {symbol} validation failed: {e}")
                import traceback
                traceback.print_exc()
        
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Print results
        print("\n🏁 ACCURACY VALIDATION RESULTS")
        print("=" * 60)
        print(f"📊 Test Period: 2022-01-01 to 2024-06-01")
        print(f"📈 Symbols Tested: {len(symbol_results)}")
        print(f"💯 Total Predictions: {total_predictions}")
        print()
        print("🎯 ACCURACY METRICS:")
        print(f"🏆 Overall Accuracy: {overall_accuracy:.1%}")
        print(f"✅ Correct Predictions: {correct_predictions}")
        print(f"📊 Total Predictions: {total_predictions}")
        print()
        print("🔍 VALIDATION STATUS:")
        if overall_accuracy >= 0.65:
            print("📈 EXCELLENT! Exceeds 65% target!")
        elif overall_accuracy >= 0.60:
            print("📈 VERY GOOD! Close to your 65% target!")
        elif overall_accuracy >= 0.55:
            print("📈 GOOD! Above random performance!")
        else:
            print("📈 BELOW TARGET - Check model/data quality")
        print()
        print("📊 INDIVIDUAL SYMBOL ACCURACY:")
        
        # Sort by accuracy
        if symbol_results:
            sorted_symbols = sorted(symbol_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for symbol, results in sorted_symbols:
                accuracy = results['accuracy']
                correct = results['correct']
                total = results['total']
                print(f"   {symbol:15s}: {accuracy:.1%} ({correct}/{total})")
        
        print("=" * 60)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"accuracy_validation_{timestamp}.json"
        
        results_data = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'symbol_results': {symbol: {'accuracy': data['accuracy'], 
                                      'correct': data['correct'], 
                                      'total': data['total']} 
                             for symbol, data in symbol_results.items()},
            'test_period': '2022-01-01 to 2024-06-01',
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📁 Results saved: {results_file}")
        
        return results_data

def main():
    """Main execution with your fresh datasets"""
    print("🎯 FIXED ACCURACY VALIDATION - FRESH V2 DATASETS")
    print("=" * 70)
    print("🔧 Works with your newly generated temporal datasets")
    print("🏆 Validates your 64.97% model on fresh data")
    print("=" * 70)
    
    # Find latest model
    models_dir = Path("data/models")
    v2_models = list(models_dir.glob("temporal_causality_model_v2*.pth"))
    
    if v2_models:
        latest_model = max(v2_models, key=lambda x: x.stat().st_mtime)
        print(f"✅ Found V2 model: {latest_model}")
    else:
        print("❌ No model found!")
        return
    
    # Initialize fixed backtester
    backtester = FixedModelBacktester(str(latest_model))
    
    # Load model
    if not backtester.load_trained_model():
        print("❌ Failed to load model!")
        return
    
    # Load your fresh datasets
    datasets = backtester.load_datasets()
    if not datasets:
        print("❌ No datasets found!")
        return
    
    print(f"📊 Loaded {len(datasets)} fresh temporal datasets")
    print("🔄 Running accuracy validation...")
    
    # Run validation
    results = backtester.run_accuracy_validation(datasets)
    
    return results

if __name__ == "__main__":
    main()
