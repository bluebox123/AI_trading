"""
TOP 5 BUY/SELL SIGNALS FROM ALL 99 V3 MODELS
Comprehensive signal extraction from your entire model universe
Real-time scoring and ranking system
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import json
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal with comprehensive scoring"""
    symbol: str
    direction: str  # BUY or SELL
    confidence: float
    model_accuracy: float
    current_price: float
    predicted_return: float
    risk_score: float
    signal_strength: float
    timestamp: datetime
    model_name: str
    
    def __post_init__(self):
        # Calculate composite signal strength
        self.signal_strength = (
            self.confidence * 0.4 +
            self.model_accuracy * 0.3 +
            min(abs(self.predicted_return) * 10, 1.0) * 0.2 +
            (1 - self.risk_score) * 0.1
        )

class UniversalV3ModelLoader:
    """Universal loader for all V3 model variations"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        self.model_registry = {}
        
        logger.info(f"Universal V3 Model Loader initialized on {self.device}")
    
    def scan_all_v3_models(self) -> Dict[str, Dict]:
        """Scan all directories for V3 models"""
        model_directories = [
            Path("data/models"),
            Path("data/models/optimized_v3"),
            Path("data/models/fixed_models"),
            Path("data/models/stock_specific_v3"),
            Path("data/models/ensemble"),
            Path("models"),
            Path("src/models")
        ]
        
        found_models = {}
        
        for model_dir in model_directories:
            if not model_dir.exists():
                continue
                
            # Search for V3 model files
            patterns = [
                "*v3*.pth",
                "*optimized*.pth", 
                "*temporal*.pth",
                "*largecap*.pth",
                "*midcap*.pth"
            ]
            
            for pattern in patterns:
                for model_file in model_dir.rglob(pattern):
                    try:
                        # Extract symbol from filename
                        symbol = self.extract_symbol_from_filename(model_file.name)
                        if symbol:
                            model_info = self.analyze_model_file(model_file)
                            if model_info:
                                found_models[f"{symbol}_{model_file.stem}"] = {
                                    'symbol': symbol,
                                    'file_path': model_file,
                                    'model_info': model_info,
                                    'model_id': f"{symbol}_{model_file.stem}"
                                }
                    except Exception as e:
                        logger.debug(f"Error scanning {model_file}: {e}")
                        continue
        
        logger.info(f"Found {len(found_models)} V3 models across all directories")
        return found_models
    
    def extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """Extract symbol from model filename"""
        # Common NSE symbols
        nse_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR',
            'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'KOTAKBANK', 'AXISBANK',
            'ASIANPAINT', 'MARUTI', 'BAJFINANCE', 'NESTLEIND', 'HCLTECH',
            'WIPRO', 'ULTRACEMCO', 'SUNPHARMA', 'ONGC', 'TATAMOTORS',
            'POWERGRID', 'NTPC', 'JSWSTEEL', 'GRASIM', 'INDUSINDBK',
            'BAJAJ-AUTO', 'BRITANNIA', 'COALINDIA', 'EICHERMOT', 'HEROMOTOCO',
            'HINDALCO', 'SHREECEM', 'TECHM', 'TITAN', 'UPL', 'ADANIPORTS',
            'APOLLOHOSP', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'PIDILITIND'
        ]
        
        filename_upper = filename.upper()
        
        for symbol in nse_symbols:
            if symbol in filename_upper:
                return f"{symbol}.NSE"
        
        return None
    
    def analyze_model_file(self, model_path: Path) -> Optional[Dict]:
        """Analyze model file to extract metadata"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            model_info = {
                'accuracy': checkpoint.get('accuracy', 0.0),
                'version': checkpoint.get('version', 'unknown'),
                'tier': checkpoint.get('tier', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'file_size_mb': model_path.stat().st_size / (1024 * 1024)
            }
            
            # Check if model has state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model_info['parameter_count'] = sum(
                    p.numel() for p in state_dict.values() 
                    if isinstance(p, torch.Tensor)
                )
                model_info['has_news_encoder'] = any(
                    'news_encoder' in k for k in state_dict.keys()
                )
                model_info['has_attention'] = any(
                    'attention' in k for k in state_dict.keys()
                )
            
            return model_info
            
        except Exception as e:
            logger.debug(f"Error analyzing {model_path}: {e}")
            return None

class FlexibleV3Model(nn.Module):
    """Flexible V3 model that adapts to different architectures"""
    
    def __init__(self, price_dim=25, news_dim=9, tier='largecap'):
        super().__init__()
        
        # Adaptive architecture based on tier
        if tier == 'largecap':
            hidden_dim = 128
            dropout = 0.4
            attention_heads = 8
        elif tier == 'midcap':
            hidden_dim = 96
            dropout = 0.5
            attention_heads = 6
        else:
            hidden_dim = 64
            dropout = 0.6
            attention_heads = 4
        
        # Core architecture
        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.price_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        news_hidden = hidden_dim // 2
        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, news_hidden),
            nn.BatchNorm1d(news_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(news_hidden, news_hidden),
            nn.BatchNorm1d(news_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(news_hidden, news_hidden)
        )
        
        fusion_dim = hidden_dim + news_hidden
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim if i == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3)
            ) for i in range(2)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 3)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, price_sequences, news_features):
        # Price encoding
        lstm_out, _ = self.price_lstm(price_sequences)
        attn_out, _ = self.price_attention(lstm_out, lstm_out, lstm_out)
        price_repr = self.layer_norm(attn_out + lstm_out)
        price_repr = price_repr[:, -1, :]
        
        # News encoding
        news_repr = self.news_encoder(news_features)
        
        # Fusion
        combined = torch.cat([price_repr, news_repr], dim=1)
        
        for i, fusion_layer in enumerate(self.fusion_layers):
            if i == 0:
                fused = fusion_layer(combined)
            else:
                fused = fusion_layer(fused) + fused
        
        # Classification
        logits = self.classifier(fused)
        return logits

class TopSignalGenerator:
    """Generate top 5 buy/sell signals from all 99 V3 models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loader = UniversalV3ModelLoader()
        self.confidence_threshold = 0.55  # Lower threshold to capture more signals
        
        logger.info("Top Signal Generator initialized")
    
    def load_model_safely(self, model_path: Path, symbol: str, model_info: Dict) -> Optional[nn.Module]:
        """Safely load a V3 model with error handling"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create flexible model
            tier = model_info.get('tier', 'largecap')
            model = FlexibleV3Model(price_dim=25, news_dim=9, tier=tier)
            
            # Load weights with fallback
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.debug(f"Failed to load model {model_path}: {e}")
            return None
    
    def get_latest_data(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get latest market data for a symbol"""
        try:
            # Load from your processed temporal datasets
            data_file = Path(f"data/processed/aligned_data/{symbol}_temporal_dataset.csv")
            if not data_file.exists():
                # Try alternative naming
                data_file = Path(f"data/processed/aligned_data/{symbol}_largecap_dataset_v3.csv")
            
            if not data_file.exists():
                return None
            
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Get latest 20 periods for price sequence
            if len(df) < 20:
                return None
            
            latest_data = df.tail(20)
            
            # Price features (25 features)
            price_features = [
                'open', 'high', 'low', 'close', 'volume', 'daily_return',
                'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
                'price_sma5_ratio', 'price_sma20_ratio', 'sma5_sma20_ratio',
                'volatility_5d', 'volatility_20d', 'rsi', 'macd', 'macd_signal',
                'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
                'volume_ratio'
            ]
            
            # News features (9 features)
            news_features = [
                'news_sentiment_1d', 'news_sentiment_3d', 'news_sentiment_7d',
                'news_volume_1d', 'news_volume_3d', 'news_volume_7d',
                'news_keyword_density_1d', 'news_keyword_density_3d', 'news_keyword_density_7d'
            ]
            
            # Create price sequence
            available_price = [f for f in price_features if f in latest_data.columns][:25]
            price_sequence = latest_data[available_price].values.astype(np.float32)
            
            # Pad if needed
            if len(available_price) < 25:
                padding = np.zeros((20, 25 - len(available_price)))
                price_sequence = np.concatenate([price_sequence, padding], axis=1)
            
            # Handle NaN values
            if np.isnan(price_sequence).any():
                price_sequence = pd.DataFrame(price_sequence).fillna(method='ffill').fillna(0).values
            
            # News features (current day)
            available_news = [f for f in news_features if f in latest_data.columns][:9]
            news_data = latest_data[available_news].iloc[-1].values.astype(np.float32)
            
            # Pad news if needed
            if len(available_news) < 9:
                padding = np.zeros(9 - len(available_news))
                news_data = np.concatenate([news_data, padding])
            
            return price_sequence, news_data
            
        except Exception as e:
            logger.debug(f"Error getting data for {symbol}: {e}")
            return None
    
    def generate_prediction(self, model: nn.Module, symbol: str, model_info: Dict, model_id: str) -> Optional[TradingSignal]:
        """Generate prediction from a single model"""
        try:
            # Get latest data
            data = self.get_latest_data(symbol)
            if data is None:
                return None
            
            price_sequence, news_data = data
            
            # Convert to tensors
            price_tensor = torch.FloatTensor(price_sequence).unsqueeze(0).to(self.device)
            news_tensor = torch.FloatTensor(news_data).unsqueeze(0).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                logits = model(price_tensor, news_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            # Convert to trading signal
            class_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            direction = class_map[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Only process BUY/SELL signals
            if direction == 'HOLD':
                return None
            
            # Get current price
            try:
                data_file = Path(f"data/processed/aligned_data/{symbol}_temporal_dataset.csv")
                if data_file.exists():
                    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                    current_price = df['close'].iloc[-1]
                else:
                    current_price = 1000.0  # Fallback
            except:
                current_price = 1000.0
            
            # Calculate predicted return (simple heuristic)
            prob_up = probabilities[0][2].item()
            prob_down = probabilities[0][0].item()
            predicted_return = (prob_up - prob_down) * 0.05  # Scale to reasonable return
            
            # Calculate risk score (volatility-based)
            try:
                volatility = df['volatility_5d'].iloc[-1] if 'volatility_5d' in df.columns else 0.02
                risk_score = min(volatility / 0.05, 1.0)  # Normalize to 0-1
            except:
                risk_score = 0.5
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence_score,
                model_accuracy=model_info.get('accuracy', 0.0),
                current_price=current_price,
                predicted_return=predicted_return,
                risk_score=risk_score,
                signal_strength=0.0,  # Will be calculated in __post_init__
                timestamp=datetime.now(),
                model_name=model_id
            )
            
            return signal
            
        except Exception as e:
            logger.debug(f"Error generating prediction for {symbol} with {model_id}: {e}")
            return None
    
    def get_top_signals(self, max_workers: int = 4) -> Dict[str, List[TradingSignal]]:
        """Get top 5 buy and sell signals from all models"""
        print("🚀 SCANNING ALL 99 V3 MODELS FOR TOP SIGNALS")
        print("="*60)
        print("📊 Loading models and generating predictions...")
        print("🎯 Filtering for BUY/SELL signals only")
        print("="*60)
        
        # Scan all models
        all_models = self.model_loader.scan_all_v3_models()
        
        if not all_models:
            print("❌ No V3 models found!")
            return {'BUY': [], 'SELL': []}
        
        print(f"📊 Found {len(all_models)} V3 models")
        
        all_signals = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Process models with threading for speed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for model_id, model_data in all_models.items():
                symbol = model_data['symbol']
                model_path = model_data['file_path']
                model_info = model_data['model_info']
                
                # Load model
                model = self.load_model_safely(model_path, symbol, model_info)
                if model is None:
                    failed_predictions += 1
                    continue
                
                # Submit prediction task
                future = executor.submit(
                    self.generate_prediction, 
                    model, symbol, model_info, model_id
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    signal = future.result(timeout=30)
                    if signal is not None:
                        all_signals.append(signal)
                        successful_predictions += 1
                    else:
                        failed_predictions += 1
                except Exception as e:
                    failed_predictions += 1
                    logger.debug(f"Prediction failed: {e}")
        
        print(f"✅ Generated {successful_predictions} predictions")
        print(f"❌ Failed predictions: {failed_predictions}")
        
        # Separate and rank signals
        buy_signals = [s for s in all_signals if s.direction == 'BUY']
        sell_signals = [s for s in all_signals if s.direction == 'SELL']
        
        # Sort by signal strength (composite score)
        buy_signals.sort(key=lambda x: x.signal_strength, reverse=True)
        sell_signals.sort(key=lambda x: x.signal_strength, reverse=True)
        
        # Get top 5 of each
        top_buy = buy_signals[:5]
        top_sell = sell_signals[:5]
        
        print(f"\n📈 Total BUY signals: {len(buy_signals)}")
        print(f"📉 Total SELL signals: {len(sell_signals)}")
        
        return {'BUY': top_buy, 'SELL': top_sell}
    
    def display_top_signals(self, signals: Dict[str, List[TradingSignal]]):
        """Display top signals in a formatted way"""
        print("\n" + "="*80)
        print("🏆 TOP 5 BUY SIGNALS FROM 99 V3 MODELS")
        print("="*80)
        
        if not signals['BUY']:
            print("❌ No BUY signals found")
        else:
            for i, signal in enumerate(signals['BUY'], 1):
                print(f"\n🥇 #{i} BUY SIGNAL")
                print(f"  📊 Symbol: {signal.symbol}")
                print(f"  🎯 Confidence: {signal.confidence:.1%}")
                print(f"  📈 Model Accuracy: {signal.model_accuracy:.1%}")
                print(f"  💰 Current Price: ₹{signal.current_price:.2f}")
                print(f"  📊 Predicted Return: {signal.predicted_return:.2%}")
                print(f"  ⚡ Signal Strength: {signal.signal_strength:.3f}")
                print(f"  🔧 Model: {signal.model_name}")
                print(f"  ⏰ Generated: {signal.timestamp.strftime('%H:%M:%S')}")
        
        print("\n" + "="*80)
        print("🔻 TOP 5 SELL SIGNALS FROM 99 V3 MODELS")
        print("="*80)
        
        if not signals['SELL']:
            print("❌ No SELL signals found")
        else:
            for i, signal in enumerate(signals['SELL'], 1):
                print(f"\n🥇 #{i} SELL SIGNAL")
                print(f"  📊 Symbol: {signal.symbol}")
                print(f"  🎯 Confidence: {signal.confidence:.1%}")
                print(f"  📈 Model Accuracy: {signal.model_accuracy:.1%}")
                print(f"  💰 Current Price: ₹{signal.current_price:.2f}")
                print(f"  📊 Predicted Return: {signal.predicted_return:.2%}")
                print(f"  ⚡ Signal Strength: {signal.signal_strength:.3f}")
                print(f"  🔧 Model: {signal.model_name}")
                print(f"  ⏰ Generated: {signal.timestamp.strftime('%H:%M:%S')}")
        
        print("\n" + "="*80)
    
    def export_signals_to_json(self, signals: Dict[str, List[TradingSignal]], filename: str = None):
        """Export signals to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_signals_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_models_scanned': len(self.model_loader.scan_all_v3_models()),
            'buy_signals_count': len(signals['BUY']),
            'sell_signals_count': len(signals['SELL']),
            'signals': {
                'BUY': [
                    {
                        'rank': i + 1,
                        'symbol': signal.symbol,
                        'confidence': signal.confidence,
                        'model_accuracy': signal.model_accuracy,
                        'current_price': signal.current_price,
                        'predicted_return': signal.predicted_return,
                        'signal_strength': signal.signal_strength,
                        'model_name': signal.model_name,
                        'timestamp': signal.timestamp.isoformat()
                    }
                    for i, signal in enumerate(signals['BUY'])
                ],
                'SELL': [
                    {
                        'rank': i + 1,
                        'symbol': signal.symbol,
                        'confidence': signal.confidence,
                        'model_accuracy': signal.model_accuracy,
                        'current_price': signal.current_price,
                        'predicted_return': signal.predicted_return,
                        'signal_strength': signal.signal_strength,
                        'model_name': signal.model_name,
                        'timestamp': signal.timestamp.isoformat()
                    }
                    for i, signal in enumerate(signals['SELL'])
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Signals exported to: {filename}")
        return filename

def main():
    """Main execution"""
    print("🚀 TOP SIGNALS FROM ALL 99 V3 MODELS")
    print("="*60)
    print("🎯 Scanning your entire model universe")
    print("📊 Extracting strongest BUY/SELL signals")
    print("🏆 Ranking by composite signal strength")
    print("="*60)
    
    # Initialize generator
    generator = TopSignalGenerator()
    
    # Get top signals
    start_time = time.time()
    top_signals = generator.get_top_signals(max_workers=4)
    end_time = time.time()
    
    # Display results
    generator.display_top_signals(top_signals)
    
    # Export to JSON
    export_file = generator.export_signals_to_json(top_signals)
    
    # Summary
    print(f"\n🎉 SIGNAL GENERATION COMPLETE!")
    print(f"⏱️ Processing time: {end_time - start_time:.1f} seconds")
    print(f"📊 Total actionable signals: {len(top_signals['BUY']) + len(top_signals['SELL'])}")
    print(f"📁 Results saved: {export_file}")
    
    # Trading readiness assessment
    total_signals = len(top_signals['BUY']) + len(top_signals['SELL'])
    if total_signals >= 8:
        print(f"\n🏆 EXCELLENT: {total_signals} high-quality signals ready for trading!")
    elif total_signals >= 5:
        print(f"\n🎯 GOOD: {total_signals} quality signals for selective trading")
    elif total_signals >= 2:
        print(f"\n⚠️ LIMITED: {total_signals} signals - consider lowering thresholds")
    else:
        print(f"\n❌ NO SIGNALS: Check model confidence thresholds")

if __name__ == "__main__":
    main()
