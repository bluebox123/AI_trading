"""
PAPER TRADING SYSTEM FOR HIGH-CONFIDENCE V3 MODELS
Deploys your proven 70%+ accuracy models in realistic trading simulation
Real-time data integration with your optimized V3 models
"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import asyncio
import time
import yfinance as yf
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your V3 model architecture (from your optimization code)
class OptimizedV3TieredModel(nn.Module):
    """Your proven V3 model architecture"""
    
    def __init__(self, price_dim: int, news_dim: int, tier: str, sequence_length: int = 20):
        super().__init__()
        
        self.tier = tier
        self.sequence_length = sequence_length
        
        # Your proven tier-specific configurations
        tier_configs = {
            'largecap': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.4},
            'midcap': {'hidden_dim': 96, 'num_layers': 2, 'dropout': 0.5}
        }
        
        config = tier_configs.get(tier, tier_configs['largecap'])
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        
        # Your proven architecture
        self.price_lstm = nn.LSTM(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.price_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8 if tier == 'largecap' else 6,
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
        
        # Fusion layers
        fusion_dim = hidden_dim + news_hidden
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim if i == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.3)
            ) for i in range(2)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 3)  # [DOWN, NEUTRAL, UP]
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, price_sequences, news_features):
        batch_size = price_sequences.size(0)
        
        # Price encoding with attention
        lstm_out, _ = self.price_lstm(price_sequences)
        attn_out, _ = self.price_attention(lstm_out, lstm_out, lstm_out)
        price_repr = self.layer_norm(attn_out + lstm_out)
        price_repr = price_repr[:, -1, :]
        
        # News encoding
        news_repr = self.news_encoder(news_features)
        
        # Multi-layer fusion
        combined = torch.cat([price_repr, news_repr], dim=1)
        
        for i, fusion_layer in enumerate(self.fusion_layers):
            if i == 0:
                fused = fusion_layer(combined)
            else:
                fused = fusion_layer(fused) + fused
        
        # Classification
        logits = self.classifier(fused)
        return logits

@dataclass
class Trade:
    """Trade record for tracking"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: int
    direction: str  # BUY/SELL
    confidence: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN/CLOSED

@dataclass
class Portfolio:
    """Portfolio tracking"""
    starting_capital: float = 1000000  # 10 Lakh starting capital
    current_capital: float = 1000000
    positions: Dict[str, Trade] = None
    closed_trades: List[Trade] = None
    total_pnl: float = 0.0
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.closed_trades is None:
            self.closed_trades = []

class V3ModelLoader:
    """Load and manage your high-confidence V3 models"""
    
    def __init__(self, models_dir: str = "data/models/optimized_v3"):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        
        # Your top performing models
        self.elite_models = {
            'ASIANPAINT.NSE': {'tier': 'largecap', 'accuracy': 0.730},
            'ICICIBANK.NSE': {'tier': 'largecap', 'accuracy': 0.726},
            'HDFCBANK.NSE': {'tier': 'largecap', 'accuracy': 0.711},
            'TCS.NSE': {'tier': 'largecap', 'accuracy': 0.709}
        }
        
        logger.info(f"V3 Model Loader initialized on {self.device}")
        logger.info(f"Elite models available: {list(self.elite_models.keys())}")
    
    def load_model(self, symbol: str) -> Optional[Dict]:
        """Load specific V3 model"""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]
        
        if symbol not in self.elite_models:
            logger.warning(f"Symbol {symbol} not in elite models list")
            return None
        
        # Look for the model file
        model_files = list(self.models_dir.glob(f"{symbol}*optimized_v3*.pth"))
        if not model_files:
            # Try alternative naming
            model_files = list(self.models_dir.glob(f"{symbol}*largecap*v3*.pth"))
        
        if not model_files:
            logger.error(f"No model file found for {symbol}")
            return None
        
        model_file = model_files[0]  # Take the first match
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            
            # Get model config
            tier = self.elite_models[symbol]['tier']
            
            # Create model
            model = OptimizedV3TieredModel(
                price_dim=25,  # Your V3 price features
                news_dim=9,   # Your V3 news features
                tier=tier,
                sequence_length=20
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            model_info = {
                'model': model,
                'symbol': symbol,
                'tier': tier,
                'accuracy': self.elite_models[symbol]['accuracy'],
                'path': model_file
            }
            
            self.loaded_models[symbol] = model_info
            logger.info(f"✅ Loaded {symbol} model - {model_info['accuracy']:.1%} accuracy")
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to load {symbol}: {e}")
            return None
    
    def load_all_elite_models(self) -> Dict:
        """Load all elite models"""
        logger.info("🚀 Loading all elite V3 models...")
        
        for symbol in self.elite_models.keys():
            self.load_model(symbol)
        
        logger.info(f"📊 Loaded {len(self.loaded_models)} elite models")
        return self.loaded_models

class RealTimeDataProvider:
    """Real-time data provider for paper trading"""
    
    def __init__(self):
        self.symbols = ['ASIANPAINT.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'TCS.NS']
        self.data_cache = {}
        
        # Price features that your models expect
        self.price_features = [
            'open', 'high', 'low', 'close', 'volume', 'daily_return',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'price_sma5_ratio', 'price_sma20_ratio', 'sma5_sma20_ratio',
            'volatility_5d', 'volatility_20d', 'rsi', 'macd', 'macd_signal',
            'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio'
        ]
        
        # News features (simplified for demo)
        self.news_features = [
            'news_sentiment_1d', 'news_sentiment_3d', 'news_sentiment_7d',
            'news_volume_1d', 'news_volume_3d', 'news_volume_7d',
            'news_keyword_density_1d', 'news_keyword_density_3d', 'news_keyword_density_7d'
        ]
    
    def get_live_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get live market data for symbol"""
        try:
            # Convert to Yahoo Finance format
            yf_symbol = symbol.replace('.NSE', '.NS')
            
            # Get recent data
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period='3mo')  # Need history for technical indicators
            
            if data.empty:
                logger.warning(f"No data for {symbol}")
                return None
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Add mock news features for demo
            data = self.add_mock_news_features(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = data.copy()
        
        # Basic calculations
        df['daily_return'] = df['Close'].pct_change()
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Price ratios
        df['price_sma5_ratio'] = df['Close'] / df['sma_5']
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['sma5_sma20_ratio'] = df['sma_5'] / df['sma_20']
        
        # Volatility
        df['volatility_5d'] = df['daily_return'].rolling(5).std()
        df['volatility_20d'] = df['daily_return'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Rename columns to match your model's expectations
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        return df
    
    def add_mock_news_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add mock news features for demo (replace with real news data)"""
        df = data.copy()
        
        # Add realistic mock news features
        np.random.seed(42)  # For consistency
        
        for feature in self.news_features:
            if 'sentiment' in feature:
                # Sentiment between -1 and 1
                df[feature] = np.random.normal(0.02, 0.15, len(df))  # Slight positive bias
            elif 'volume' in feature:
                # News volume (number of articles)
                df[feature] = np.random.poisson(5, len(df))  # Average 5 articles
            elif 'density' in feature:
                # Keyword density
                df[feature] = np.random.uniform(0.1, 0.8, len(df))
        
        return df

class PaperTradingEngine:
    """Main paper trading engine"""
    
    def __init__(self, starting_capital: float = 1000000):
        self.portfolio = Portfolio(starting_capital=starting_capital)
        self.model_loader = V3ModelLoader()
        self.data_provider = RealTimeDataProvider()
        self.models = {}
        
        # Trading parameters
        self.max_position_size = 0.10  # 10% max per position
        self.confidence_threshold = 0.65  # Only trade with 65%+ confidence
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        logger.info(f"📊 Paper Trading Engine initialized")
        logger.info(f"💰 Starting capital: ₹{starting_capital:,.0f}")
    
    def initialize_models(self):
        """Load all elite models"""
        logger.info("🚀 Initializing elite V3 models...")
        self.models = self.model_loader.load_all_elite_models()
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
        
        logger.info(f"✅ {len(self.models)} elite models ready for trading")
    
    def prepare_model_input(self, data: pd.DataFrame, symbol: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare data for model input"""
        try:
            if len(data) < 20:
                return None
            
            # Get latest 20 days for sequence
            recent_data = data.tail(20)
            
            # Price features
            price_features = []
            for feature in self.data_provider.price_features:
                if feature in recent_data.columns:
                    price_features.append(feature)
            
            if len(price_features) < 15:  # Need minimum features
                logger.warning(f"{symbol}: Insufficient price features")
                return None
            
            # Create price sequence
            price_sequence = recent_data[price_features].values.astype(np.float32)
            
            # Check for NaN
            if np.isnan(price_sequence).any():
                # Fill NaN with previous values
                price_sequence = pd.DataFrame(price_sequence).fillna(method='ffill').fillna(0).values
            
            # News features (current day)
            news_data = recent_data[self.data_provider.news_features].iloc[-1].values.astype(np.float32)
            
            # Convert to tensors
            price_tensor = torch.FloatTensor(price_sequence).unsqueeze(0)  # Add batch dimension
            news_tensor = torch.FloatTensor(news_data).unsqueeze(0)
            
            return price_tensor, news_tensor
            
        except Exception as e:
            logger.error(f"Error preparing input for {symbol}: {e}")
            return None
    
    def generate_prediction(self, symbol: str) -> Optional[Dict]:
        """Generate prediction for symbol"""
        if symbol not in self.models:
            return None
        
        # Get live data
        data = self.data_provider.get_live_data(symbol)
        if data is None:
            return None
        
        # Prepare model input
        model_input = self.prepare_model_input(data, symbol)
        if model_input is None:
            return None
        
        price_tensor, news_tensor = model_input
        
        try:
            model_info = self.models[symbol]
            model = model_info['model']
            
            # Move to device
            price_tensor = price_tensor.to(self.model_loader.device)
            news_tensor = news_tensor.to(self.model_loader.device)
            
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
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            prediction = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence_score,
                'current_price': current_price,
                'model_accuracy': model_info['accuracy'],
                'timestamp': datetime.now(),
                'probabilities': {
                    'sell': probabilities[0][0].item(),
                    'hold': probabilities[0][1].item(),
                    'buy': probabilities[0][2].item()
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None
    
    def should_enter_trade(self, prediction: Dict) -> bool:
        """Determine if we should enter a trade"""
        symbol = prediction['symbol']
        direction = prediction['direction']
        confidence = prediction['confidence']
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Don't trade if already have position
        if symbol in self.portfolio.positions:
            return False
        
        # Only trade BUY/SELL signals
        if direction == 'HOLD':
            return False
        
        return True
    
    def calculate_position_size(self, prediction: Dict) -> int:
        """Calculate position size based on confidence and capital"""
        confidence = prediction['confidence']
        current_price = prediction['current_price']
        
        # Kelly Criterion inspired sizing
        base_allocation = self.portfolio.current_capital * self.max_position_size
        confidence_multiplier = min(confidence / 0.65, 1.5)  # Scale with confidence
        
        position_value = base_allocation * confidence_multiplier
        quantity = int(position_value / current_price)
        
        return max(1, quantity)  # At least 1 share
    
    def enter_trade(self, prediction: Dict):
        """Enter a new trade"""
        symbol = prediction['symbol']
        direction = prediction['direction']
        current_price = prediction['current_price']
        confidence = prediction['confidence']
        
        quantity = self.calculate_position_size(prediction)
        
        trade = Trade(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=current_price,
            quantity=quantity,
            direction=direction,
            confidence=confidence
        )
        
        self.portfolio.positions[symbol] = trade
        
        logger.info(f"🎯 ENTERED {direction} {symbol}: "
                   f"₹{current_price:.2f} x {quantity} shares "
                   f"(Confidence: {confidence:.1%})")
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> bool:
        """Check if we should exit a position"""
        if symbol not in self.portfolio.positions:
            return False
        
        trade = self.portfolio.positions[symbol]
        
        # Calculate current PnL
        if trade.direction == 'BUY':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:  # SELL
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            logger.info(f"🛑 Stop loss triggered for {symbol}: {pnl_pct:.2%}")
            return True
        
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            logger.info(f"🎯 Take profit triggered for {symbol}: {pnl_pct:.2%}")
            return True
        
        # Time-based exit (hold for max 3 days for demo)
        hold_time = datetime.now() - trade.entry_time
        if hold_time.days >= 3:
            logger.info(f"⏰ Time exit for {symbol} after {hold_time.days} days")
            return True
        
        return False
    
    def exit_trade(self, symbol: str, current_price: float):
        """Exit a trade"""
        if symbol not in self.portfolio.positions:
            return
        
        trade = self.portfolio.positions[symbol]
        trade.exit_time = datetime.now()
        trade.exit_price = current_price
        trade.status = "CLOSED"
        
        # Calculate PnL
        if trade.direction == 'BUY':
            trade.pnl = (current_price - trade.entry_price) * trade.quantity
        else:  # SELL
            trade.pnl = (trade.entry_price - current_price) * trade.quantity
        
        # Update portfolio
        self.portfolio.total_pnl += trade.pnl
        self.portfolio.current_capital += trade.pnl
        self.portfolio.closed_trades.append(trade)
        
        del self.portfolio.positions[symbol]
        
        logger.info(f"🔚 EXITED {symbol}: PnL ₹{trade.pnl:,.0f} "
                   f"({(trade.pnl/(trade.entry_price * trade.quantity)):.2%})")
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        print("\n" + "="*60)
        print("📊 PORTFOLIO STATUS")
        print("="*60)
        print(f"💰 Current Capital: ₹{self.portfolio.current_capital:,.0f}")
        print(f"📈 Total PnL: ₹{self.portfolio.total_pnl:,.0f}")
        print(f"📊 Return: {(self.portfolio.total_pnl/self.portfolio.starting_capital)*100:.2f}%")
        
        print(f"\n🎯 Active Positions: {len(self.portfolio.positions)}")
        for symbol, trade in self.portfolio.positions.items():
            # Get current price for unrealized PnL
            data = self.data_provider.get_live_data(symbol)
            if data is not None:
                current_price = data['close'].iloc[-1]
                if trade.direction == 'BUY':
                    unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                print(f"  {symbol}: {trade.direction} ₹{trade.entry_price:.2f} → ₹{current_price:.2f} "
                     f"(PnL: ₹{unrealized_pnl:,.0f})")
        
        print(f"\n📝 Closed Trades: {len(self.portfolio.closed_trades)}")
        if self.portfolio.closed_trades:
            total_wins = sum(1 for t in self.portfolio.closed_trades if t.pnl > 0)
            win_rate = total_wins / len(self.portfolio.closed_trades)
            print(f"  Win Rate: {win_rate:.1%}")
        
        print("="*60)
    
    async def run_paper_trading(self, duration_minutes: int = 60):
        """Run paper trading simulation"""
        print("🚀 STARTING PAPER TRADING WITH ELITE V3 MODELS")
        print("="*60)
        print(f"⏰ Duration: {duration_minutes} minutes")
        print(f"🎯 Models: {list(self.models.keys())}")
        print(f"💰 Starting Capital: ₹{self.portfolio.starting_capital:,.0f}")
        print("="*60)
        
        # Initialize models
        self.initialize_models()
        
        start_time = datetime.now()
        iteration = 0
        
        try:
            while (datetime.now() - start_time).seconds < duration_minutes * 60:
                iteration += 1
                print(f"\n🔄 Trading Iteration {iteration}")
                
                # Process each elite model
                for symbol in self.models.keys():
                    try:
                        # Generate prediction
                        prediction = self.generate_prediction(symbol)
                        if prediction is None:
                            continue
                        
                        # Display prediction
                        print(f"📊 {symbol}: {prediction['direction']} "
                             f"(Confidence: {prediction['confidence']:.1%}, "
                             f"Price: ₹{prediction['current_price']:.2f})")
                        
                        # Check entry conditions
                        if self.should_enter_trade(prediction):
                            self.enter_trade(prediction)
                        
                        # Check exit conditions for existing positions
                        elif symbol in self.portfolio.positions:
                            if self.check_exit_conditions(symbol, prediction['current_price']):
                                self.exit_trade(symbol, prediction['current_price'])
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Print status every 5 iterations
                if iteration % 5 == 0:
                    self.print_portfolio_status()
                
                # Sleep between iterations
                await asyncio.sleep(30)  # 30 seconds between checks
                
        except KeyboardInterrupt:
            print("\n⏹️ Paper trading stopped by user")
        
        # Final portfolio status
        print("\n🏁 PAPER TRADING COMPLETE!")
        self.print_portfolio_status()
        
        return self.portfolio

async def main():
    """Main execution"""
    print("🚀 V3 ELITE MODEL PAPER TRADING SYSTEM")
    print("="*60)
    print("📊 Deploying your 70%+ accuracy models")
    print("🎯 Realistic paper trading simulation")
    print("💰 Starting with ₹10,00,000 capital")
    print("="*60)
    
    # Create trading engine
    engine = PaperTradingEngine(starting_capital=1000000)
    
    # Run paper trading
    print("\nSelect trading duration:")
    print("1. Quick test (15 minutes)")
    print("2. Standard session (60 minutes)")
    print("3. Extended session (180 minutes)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    duration_map = {'1': 15, '2': 60, '3': 180}
    duration = duration_map.get(choice, 60)
    
    # Start trading
    portfolio = await engine.run_paper_trading(duration_minutes=duration)
    
    print(f"\n🎉 Paper trading complete!")
    print(f"📈 Final return: {(portfolio.total_pnl/portfolio.starting_capital)*100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
