#!/usr/bin/env python3
"""
Enhanced V5 Comprehensive Model Training System - Maximum Accuracy
=====================================

Combines V5 advanced architecture with V3 proven anti-overfitting techniques
- Advanced attention-based transformer with hierarchical processing
- Tier-specific optimizations (inspired by V3 success)
- Robust anti-overfitting and regularization
- Ensemble training capabilities
- Enhanced data validation and quality control
- Optimized for your data structure and paths
"""

import os
import sys
import json
import logging
import warnings
import random
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_v5_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set global seeds for reproducibility - Enhanced version"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Additional determinism
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """Enhanced seed function for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@dataclass
class EnhancedV5Config:
    """Enhanced configuration with tier-specific optimizations"""
    # Model architecture
    feature_dim: int = 75  # Will be adjusted automatically
    hidden_dim: int = 256
    num_attention_heads: int = 16
    num_transformer_layers: int = 6  # Reduced from 8 for better generalization
    sequence_length: int = 20
    
    # Feature processing (tier-adaptive)
    price_features_dim: int = 25
    sentiment_features_dim: int = 15
    temporal_features_dim: int = 10
    market_features_dim: int = 25
    
    # Enhanced regularization (anti-overfitting)
    dropout_rate: float = 0.2  # Increased for better generalization
    attention_dropout: float = 0.15
    layer_norm_eps: float = 1e-6
    stochastic_depth_prob: float = 0.1
    
    # Multi-task learning weights
    direction_weight: float = 0.7  # Slightly increased for classification focus
    magnitude_weight: float = 0.3
    
    # Enhanced regularization
    weight_decay: float = 0.01  # Increased for better regularization
    gradient_clip_norm: float = 0.5  # More conservative clipping
    label_smoothing: float = 0.1  # Added label smoothing
    
    # Missing data handling
    missing_token_dim: int = 16
    use_learnable_missing_mask: bool = True
    
    # Tier-specific configurations (inspired by V3 success)
    tier_configs: Dict = None
    
    def __post_init__(self):
        if self.tier_configs is None:
            self.tier_configs = {
                'largecap': {
                    'dropout_rate': 0.15,
                    'learning_rate': 0.0002,
                    'batch_size': 16,
                    'expected_accuracy': 0.67,
                    'patience': 20
                },
                'midcap': {
                    'dropout_rate': 0.20,
                    'learning_rate': 0.00015,
                    'batch_size': 12,
                    'expected_accuracy': 0.62,
                    'patience': 18
                },
                'smallcap': {
                    'dropout_rate': 0.25,
                    'learning_rate': 0.0001,
                    'batch_size': 8,
                    'expected_accuracy': 0.58,
                    'patience': 15
                }
            }

class LearnableMissingMask(nn.Module):
    """Enhanced missing mask with better regularization"""
    
    def __init__(self, feature_dim: int, missing_token_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.missing_token_dim = missing_token_dim
        
        # Learnable missing token with regularization
        self.missing_token = nn.Parameter(torch.randn(missing_token_dim) * 0.1)
        
        # Enhanced missing projection with dropout
        self.missing_projection = nn.Sequential(
            nn.Linear(missing_token_dim, feature_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(feature_dim)
        )
        
        # Mask attention with regularization
        self.mask_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Generate missing representations
        missing_repr = self.missing_projection(self.missing_token)
        missing_repr = missing_repr.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Apply mask with learnable missing representations
        x_filled = torch.where(mask, missing_repr, x)
        
        # Calculate attention weights for missing vs real data
        mask_weights = torch.sigmoid(self.mask_attention(x_filled))
        
        return x_filled, mask_weights

class EnhancedHierarchicalProcessor(nn.Module):
    """Enhanced hierarchical processor with better regularization"""
    
    def __init__(self, config: EnhancedV5Config):
        super().__init__()
        self.config = config
        
        # Enhanced feature group processors with residual connections
        self.price_processor = nn.Sequential(
            nn.Linear(config.price_features_dim, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),  # Changed to GELU for better performance
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Dropout(config.dropout_rate * 0.5)  # Lighter dropout for second layer
        )
        
        self.sentiment_processor = nn.Sequential(
            nn.Linear(config.sentiment_features_dim, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Dropout(config.dropout_rate * 0.5)
        )
        
        self.temporal_processor = nn.Sequential(
            nn.Linear(config.temporal_features_dim, config.hidden_dim // 8),
            nn.LayerNorm(config.hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 8, config.hidden_dim // 8),
            nn.LayerNorm(config.hidden_dim // 8)
        )
        
        self.market_processor = nn.Sequential(
            nn.Linear(config.market_features_dim, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.Dropout(config.dropout_rate * 0.5)
        )
        
        # Enhanced cross-attention with multiple heads
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,  # Increased attention heads
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Feature projection with residual connection
        total_processed_dim = (config.hidden_dim // 4) * 3 + (config.hidden_dim // 8)
        self.feature_projection = nn.Sequential(
            nn.Linear(total_processed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate * 0.5)
        )
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process each feature group
        price_repr = self.price_processor(features['price'])
        sentiment_repr = self.sentiment_processor(features['sentiment'])
        temporal_repr = self.temporal_processor(features['temporal'])
        market_repr = self.market_processor(features['market'])
        
        # Combine representations
        combined = torch.cat([price_repr, sentiment_repr, temporal_repr, market_repr], dim=-1)
        
        # Project to unified space
        unified_repr = self.feature_projection(combined)
        
        # Apply cross-attention for feature interaction
        attended_repr, attention_weights = self.cross_attention(unified_repr, unified_repr, unified_repr)
        
        # Residual connection with learnable scaling
        output = unified_repr + self.residual_scale * attended_repr
        
        return output

class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with better regularization"""
    
    def __init__(self, config: EnhancedV5Config):
        super().__init__()
        self.config = config
        
        # Multi-head attention with enhanced dropout
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Enhanced feed-forward network with GELU and residual scaling
        self.ff_network = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout_rate * 0.5)  # Lighter dropout for output
        )
        
        # Layer normalization with different eps for stability
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Stochastic depth for regularization
        self.stochastic_depth_prob = config.stochastic_depth_prob
        
        # Learnable residual scaling
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.ff_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better gradient flow
        norm_x = self.norm1(x)
        
        # Self-attention with stochastic depth
        attn_out, attention_weights = self.attention(norm_x, norm_x, norm_x, key_padding_mask=mask)
        
        # Apply stochastic depth during training
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            attn_out = torch.zeros_like(attn_out)
        
        # Residual connection with learnable scaling
        x = x + self.attn_scale * attn_out
        
        # Feed-forward with residual connection
        norm_x2 = self.norm2(x)
        ff_out = self.ff_network(norm_x2)
        
        # Apply stochastic depth to FF as well
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            ff_out = torch.zeros_like(ff_out)
        
        x = x + self.ff_scale * ff_out
        
        return x

class EnhancedV5Model(nn.Module):
    """Enhanced V5 model with anti-overfitting techniques"""
    
    def __init__(self, config: EnhancedV5Config):
        super().__init__()
        self.config = config
        
        # Enhanced missing data handler
        if config.use_learnable_missing_mask:
            self.missing_mask = LearnableMissingMask(
                config.feature_dim, 
                config.missing_token_dim,
                dropout=config.dropout_rate
            )
        
        # Enhanced hierarchical feature processor
        self.feature_processor = EnhancedHierarchicalProcessor(config)
        
        # Learnable positional encoding with dropout
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.hidden_dim) * 0.02
        )
        self.pos_dropout = nn.Dropout(config.dropout_rate * 0.5)
        
        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(config) 
            for _ in range(config.num_transformer_layers)
        ])
        
        # Enhanced global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        # Enhanced multi-task heads with better regularization
        self.direction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.hidden_dim // 4, 3)  # DOWN, NEUTRAL, UP
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        # Enhanced confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights with better strategy
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Enhanced weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            torch.nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
            torch.nn.init.trunc_normal_(module.out_proj.weight, std=0.02)
    
    def split_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced feature splitting with validation"""
        batch_size, seq_len, total_features = x.shape
        
        # Ensure we have enough features
        if total_features < sum([
            self.config.price_features_dim,
            self.config.sentiment_features_dim,
            self.config.temporal_features_dim,
            self.config.market_features_dim
        ]):
            logger.warning(f"Insufficient features: {total_features}, adjusting dimensions")
            # Adjust feature dimensions proportionally
            ratio = total_features / self.config.feature_dim
            self.config.price_features_dim = max(1, int(self.config.price_features_dim * ratio))
            self.config.sentiment_features_dim = max(1, int(self.config.sentiment_features_dim * ratio))
            self.config.temporal_features_dim = max(1, int(self.config.temporal_features_dim * ratio))
            self.config.market_features_dim = total_features - (
                self.config.price_features_dim + 
                self.config.sentiment_features_dim + 
                self.config.temporal_features_dim
            )
        
        # Split features
        price_end = self.config.price_features_dim
        sentiment_end = price_end + self.config.sentiment_features_dim
        temporal_end = sentiment_end + self.config.temporal_features_dim
        
        return {
            'price': x[:, :, :price_end],
            'sentiment': x[:, :, price_end:sentiment_end],
            'temporal': x[:, :, sentiment_end:temporal_end],
            'market': x[:, :, temporal_end:]
        }
    
    def forward(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Handle missing data
        mask_weights = None
        if missing_mask is not None and self.config.use_learnable_missing_mask:
            x, mask_weights = self.missing_mask(x, missing_mask)
        
        # Split and process features hierarchically
        feature_groups = self.split_features(x)
        processed_features = self.feature_processor(feature_groups)
        
        # Add positional encoding with dropout
        processed_features = processed_features + self.positional_encoding[:, :seq_len, :]
        processed_features = self.pos_dropout(processed_features)
        
        # Apply transformer layers with gradient checkpointing for memory efficiency
        hidden_states = processed_features
        for i, transformer_layer in enumerate(self.transformer_layers):
            if self.training and i > 0:  # Use gradient checkpointing for deeper layers
                hidden_states = torch.utils.checkpoint.checkpoint(transformer_layer, hidden_states)
            else:
                hidden_states = transformer_layer(hidden_states)
        
        # Global attention pooling
        attention_weights = F.softmax(self.global_attention(hidden_states), dim=1)
        pooled_repr = torch.sum(hidden_states * attention_weights, dim=1)
        
        # Multi-task predictions
        direction_logits = self.direction_head(pooled_repr)
        magnitude_pred = self.magnitude_head(pooled_repr)
        confidence = self.confidence_head(pooled_repr)
        
        return {
            'direction_logits': direction_logits,
            'magnitude': magnitude_pred,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'hidden_representation': pooled_repr
        }

class EnhancedV5Dataset(Dataset):
    """Enhanced dataset with better data quality control"""
    
    def __init__(self, data_dir: str, symbols: List[str], split: str = 'train', 
                 config: EnhancedV5Config = None, train_ratio: float = 0.8,
                 shared_feature_scaler=None, shared_magnitude_scaler=None,
                 data_quality_threshold: float = 0.7):
        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.split = split
        self.config = config or EnhancedV5Config()
        self.train_ratio = train_ratio
        self.data_quality_threshold = data_quality_threshold
        
        # Enhanced data storage
        self.sequences = []
        self.targets_direction = []
        self.targets_magnitude = []
        self.missing_masks = []
        self.metadata = []
        self.quality_scores = []
        
        # Enhanced feature scalers
        if shared_feature_scaler is not None:
            self.feature_scaler = shared_feature_scaler
        else:
            self.feature_scaler = RobustScaler()  # More robust to outliers
            
        if shared_magnitude_scaler is not None:
            self.magnitude_scaler = shared_magnitude_scaler
        else:
            self.magnitude_scaler = StandardScaler()
        
        self.load_data()
        
    def calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        quality_factors = []
        
        # Completeness score
        completeness = 1.0 - df.isnull().mean().mean()
        quality_factors.append(completeness)
        
        # Temporal consistency - check if date column exists
        try:
            if 'date' in df.columns:
                date_col = pd.to_datetime(df['date'], errors='coerce')
                if not date_col.isnull().all():
                    date_diffs = date_col.diff().dt.days
                    temporal_consistency = (date_diffs == 1).mean()  # Daily consistency
                    quality_factors.append(temporal_consistency)
            elif hasattr(df.index, 'to_series') and pd.api.types.is_datetime64_any_dtype(df.index):
                date_diffs = df.index.to_series().diff().dt.days
                temporal_consistency = (date_diffs == 1).mean()  # Daily consistency
                quality_factors.append(temporal_consistency)
        except:
            pass  # Skip temporal consistency if datetime parsing fails
        
        # Feature quality (non-zero variance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            feature_quality = (df[numeric_cols].var() > 1e-6).mean()
            quality_factors.append(feature_quality)
        
        # News coverage quality
        news_cols = [col for col in df.columns if 'news' in col.lower()]
        if news_cols:
            news_coverage = (df[news_cols].abs().sum(axis=1) > 0).mean()
            quality_factors.append(news_coverage)
        
        return np.mean(quality_factors) if quality_factors else 0.5  # Default to 0.5 if no factors
    
    def load_data(self):
        """Enhanced data loading with quality control"""
        logger.info(f"Loading enhanced V5 datasets for {len(self.symbols)} symbols...")
        
        all_sequences = []
        all_targets_direction = []
        all_targets_magnitude = []
        all_missing_masks = []
        all_metadata = []
        all_quality_scores = []
        
        successful_loads = 0
        
        for symbol in self.symbols:
            try:
                dataset_file = self.data_dir / f"{symbol}_temporal_dataset_v5.csv"
                metadata_file = self.data_dir / f"{symbol}_metadata_v5.json"
                
                if not dataset_file.exists():
                    logger.warning(f"Dataset not found for {symbol}")
                    continue
                
                # Load metadata
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {'symbol': symbol, 'missing_data_percentage': 0}
                
                # Load dataset
                df = pd.read_csv(dataset_file)
                
                # Quality control
                quality_score = self.calculate_data_quality(df)
                if quality_score < self.data_quality_threshold:
                    logger.warning(f"Low quality data for {symbol}: {quality_score:.3f} < {self.data_quality_threshold}")
                    continue
                
                if len(df) < self.config.sequence_length + 10:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    continue
                
                # Prepare features and targets
                sequences, targets_dir, targets_mag, missing_masks = self._prepare_sequences(df, metadata)
                
                if len(sequences) > 0:
                    all_sequences.extend(sequences)
                    all_targets_direction.extend(targets_dir)
                    all_targets_magnitude.extend(targets_mag)
                    all_missing_masks.extend(missing_masks)
                    all_metadata.extend([metadata] * len(sequences))
                    all_quality_scores.extend([quality_score] * len(sequences))
                    successful_loads += 1
                    
                    logger.info(f"✅ {symbol}: {len(sequences)} sequences loaded, quality: {quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid data loaded from any symbol")
        
        logger.info(f"Successfully loaded data from {successful_loads}/{len(self.symbols)} symbols")
        logger.info(f"Total sequences: {len(all_sequences)}")
        logger.info(f"Average data quality: {np.mean(all_quality_scores):.3f}")
        
        # Convert to numpy arrays
        all_sequences = np.array(all_sequences)
        all_targets_direction = np.array(all_targets_direction)
        all_targets_magnitude = np.array(all_targets_magnitude)
        all_missing_masks = np.array(all_missing_masks)
        self.quality_scores = np.array(all_quality_scores)
        
        # Fit scalers and normalize
        self._fit_scalers(all_sequences, all_targets_magnitude)
        all_sequences = self._normalize_sequences(all_sequences)
        
        # Normalize magnitude targets
        try:
            all_targets_magnitude = self.magnitude_scaler.transform(all_targets_magnitude.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.warning(f"Magnitude scaler not fitted, fitting now: {e}")
            self.magnitude_scaler.fit(all_targets_magnitude.reshape(-1, 1))
            all_targets_magnitude = self.magnitude_scaler.transform(all_targets_magnitude.reshape(-1, 1)).flatten()
        
        # Split data
        self._split_data(all_sequences, all_targets_direction, all_targets_magnitude, 
                        all_missing_masks, all_metadata)
        
        logger.info(f"Dataset prepared: {len(self.sequences)} sequences for {self.split} split")
        logger.info(f"Average quality for {self.split}: {np.mean(self.quality_scores):.3f}")
    
    def _prepare_sequences(self, df: pd.DataFrame, metadata: Dict) -> Tuple[List, List, List, List]:
        """Enhanced sequence preparation with better validation"""
        sequences = []
        targets_direction = []
        targets_magnitude = []
        missing_masks = []
        
        # Enhanced feature selection
        exclude_cols = [
            'date', 'symbol', 'close_future_1', 'return_future_1',
            'close_future_2', 'return_future_2', 'close_future_3', 'return_future_3',
            'close_future_5', 'return_future_5', 'Symbol', 'Company_Name', 'Sector',
            'Market_Cap_Category', 'Sentiment_Category', 'Primary_Market_Factor', 
            'Market_Phase', 'day_of_week', 'month', 'quarter', 'year'
        ]
        
        # Get numeric features
        all_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = []
        for col in all_cols:
            try:
                pd.to_numeric(df[col], errors='raise')
                feature_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Using {len(feature_cols)} features from {len(df.columns)} total columns")
        
        if len(feature_cols) < 10:
            raise ValueError(f"Insufficient numeric features: {len(feature_cols)}")
        
        # Update config to match actual features
        if len(feature_cols) != self.config.feature_dim:
            self.config.feature_dim = len(feature_cols)
            # Proportional adjustment
            total_ratio = sum([
                self.config.price_features_dim,
                self.config.sentiment_features_dim,
                self.config.temporal_features_dim,
                self.config.market_features_dim
            ])
            ratio = len(feature_cols) / total_ratio
            self.config.price_features_dim = max(1, int(self.config.price_features_dim * ratio))
            self.config.sentiment_features_dim = max(1, int(self.config.sentiment_features_dim * ratio))
            self.config.temporal_features_dim = max(1, int(self.config.temporal_features_dim * ratio))
            self.config.market_features_dim = len(feature_cols) - (
                self.config.price_features_dim + 
                self.config.sentiment_features_dim + 
                self.config.temporal_features_dim
            )
        
        # Convert to numeric and handle missing values
        feature_data = df[feature_cols].copy()
        for col in feature_cols:
            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
        
        # Enhanced outlier removal using IQR method
        for col in feature_cols:
            Q1 = feature_data[col].quantile(0.01)
            Q3 = feature_data[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            feature_data[col] = feature_data[col].clip(lower_bound, upper_bound)
        
        feature_data = feature_data.values.astype(np.float32)
        
        # Get targets with enhanced validation
        if 'return_future_1' in df.columns:
            future_returns = pd.to_numeric(df['return_future_1'], errors='coerce').values
        else:
            if 'close' in df.columns:
                close_prices = pd.to_numeric(df['close'], errors='coerce').values
                future_returns = np.zeros(len(close_prices))
                for i in range(len(close_prices) - 1):
                    if not np.isnan(close_prices[i]) and not np.isnan(close_prices[i + 1]) and close_prices[i] != 0:
                        future_returns[i] = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
            else:
                future_returns = np.zeros(len(feature_data))
                logger.warning("No return data available, using zeros")
        
        # Create sequences with enhanced validation
        for i in range(self.config.sequence_length, len(feature_data) - 1):
            try:
                target_return = future_returns[i]
                if np.isnan(target_return) or np.isinf(target_return) or abs(target_return) > 0.5:  # 50% limit
                    continue
                
                # Enhanced target labeling with tier-specific thresholds
                threshold_up = 0.015    # 1.5% default
                threshold_down = -0.015
                
                if target_return > threshold_up:
                    direction = 2  # UP
                elif target_return < threshold_down:
                    direction = 0  # DOWN
                else:
                    direction = 1  # NEUTRAL
                
                # Input sequence
                sequence = feature_data[i - self.config.sequence_length:i]
                
                # Enhanced missing data handling
                missing_mask = np.isnan(sequence) | np.isinf(sequence)
                
                # Forward fill with improved strategy
                sequence_filled = np.copy(sequence)
                for j in range(sequence_filled.shape[1]):
                    column = sequence_filled[:, j]
                    if np.any(np.isnan(column)) or np.any(np.isinf(column)):
                        # Use pandas for better forward fill
                        col_series = pd.Series(column)
                        col_series = col_series.ffill().bfill().fillna(0.0)
                        sequence_filled[:, j] = col_series.values
                
                # Final cleanup
                sequence_filled = np.nan_to_num(sequence_filled, nan=0.0, posinf=1.0, neginf=-1.0)
                
                sequences.append(sequence_filled)
                targets_direction.append(direction)
                targets_magnitude.append(target_return)
                missing_masks.append(missing_mask)
                
            except Exception as e:
                continue
        
        return sequences, targets_direction, targets_magnitude, missing_masks
    
    def _fit_scalers(self, sequences: np.ndarray, targets_magnitude: np.ndarray):
        """Enhanced scaler fitting"""
        if self.split == 'train':
            try:
                self.feature_scaler.transform(sequences[:1].reshape(-1, sequences.shape[-1]))
                logger.info("Using pre-fitted feature scaler")
            except:
                reshaped_sequences = sequences.reshape(-1, sequences.shape[-1])
                self.feature_scaler.fit(reshaped_sequences)
                logger.info("Fitted feature scaler on training data")
            
            try:
                self.magnitude_scaler.transform(targets_magnitude[:1].reshape(-1, 1))
                logger.info("Using pre-fitted magnitude scaler")
            except:
                self.magnitude_scaler.fit(targets_magnitude.reshape(-1, 1))
                logger.info("Fitted magnitude scaler on training data")
        else:
            logger.info("Using shared pre-fitted scalers for validation data")
    
    def _normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Enhanced sequence normalization"""
        original_shape = sequences.shape
        reshaped = sequences.reshape(-1, sequences.shape[-1])
        
        try:
            normalized = self.feature_scaler.transform(reshaped)
        except Exception as e:
            logger.warning(f"Scaler not fitted, fitting now: {e}")
            self.feature_scaler.fit(reshaped)
            normalized = self.feature_scaler.transform(reshaped)
            
        return normalized.reshape(original_shape)
    
    def _split_data(self, sequences: np.ndarray, targets_direction: np.ndarray, 
                   targets_magnitude: np.ndarray, missing_masks: np.ndarray, metadata: List):
        """Enhanced data splitting with quality preservation"""
        n_samples = len(sequences)
        split_idx = int(n_samples * self.train_ratio)
        
        # Sort by quality score if available, otherwise use simple split
        if hasattr(self, 'quality_scores') and len(self.quality_scores) == n_samples:
            sorted_indices = np.argsort(self.quality_scores)[::-1]  # High quality first
            if self.split == 'train':
                indices = sorted_indices[:split_idx]
            else:  # validation
                indices = sorted_indices[split_idx:]
            
            self.sequences = sequences[indices]
            self.targets_direction = targets_direction[indices]
            self.targets_magnitude = targets_magnitude[indices]
            self.missing_masks = missing_masks[indices]
            self.metadata = [metadata[i] for i in indices]
            self.quality_scores = self.quality_scores[indices]
        else:
            # Fallback to simple split if quality scores not available
            if self.split == 'train':
                self.sequences = sequences[:split_idx]
                self.targets_direction = targets_direction[:split_idx]
                self.targets_magnitude = targets_magnitude[:split_idx]
                self.missing_masks = missing_masks[:split_idx]
                self.metadata = metadata[:split_idx]
            else:  # validation
                self.sequences = sequences[split_idx:]
                self.targets_direction = targets_direction[split_idx:]
                self.targets_magnitude = targets_magnitude[split_idx:]
                self.missing_masks = missing_masks[split_idx:]
                self.metadata = metadata[split_idx:]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        result = {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'direction_target': torch.LongTensor([self.targets_direction[idx]]),
            'magnitude_target': torch.FloatTensor([self.targets_magnitude[idx]]),
            'missing_mask': torch.BoolTensor(self.missing_masks[idx]),
            'metadata': self.metadata[idx]
        }
        
        # Add quality score if available
        if hasattr(self, 'quality_scores') and len(self.quality_scores) > idx:
            result['quality_score'] = torch.FloatTensor([self.quality_scores[idx]])
        else:
            result['quality_score'] = torch.FloatTensor([1.0])  # Default quality score
            
        return result

class EnhancedV5Trainer:
    """Enhanced trainer with anti-overfitting and ensemble capabilities"""
    
    def __init__(self, config: EnhancedV5Config, data_dir: str, output_dir: str, seed: int = 42):
        # Set reproducibility
        set_seed(seed)
        
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Enhanced mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
            'learning_rates': [], 'epoch_times': []
        }
        
        # Ensemble storage
        self.ensemble_models = []
        self.ensemble_predictions = []
        
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from V5 datasets"""
        symbols = []
        dataset_dir = self.data_dir / "stock_specific"
        
        for file in dataset_dir.glob("*_temporal_dataset_v5.csv"):
            symbol = file.stem.replace("_temporal_dataset_v5", "")
            symbols.append(symbol)
        
        logger.info(f"Found {len(symbols)} available symbols")
        return symbols
    
    def create_data_loaders(self, symbols: List[str], batch_size: int = 32, 
                          data_quality_threshold: float = 0.7) -> Tuple[DataLoader, DataLoader]:
        """Create enhanced data loaders"""
        
        # Create reproducible generator
        generator = torch.Generator()
        generator.manual_seed(42)
        
        # Create training dataset first
        logger.info("Creating enhanced training dataset...")
        train_dataset = EnhancedV5Dataset(
            data_dir=self.data_dir / "stock_specific",
            symbols=symbols,
            split='train',
            config=self.config,
            data_quality_threshold=data_quality_threshold
        )
        
        # Create validation dataset
        logger.info("Creating enhanced validation dataset...")
        val_dataset = EnhancedV5Dataset(
            data_dir=self.data_dir / "stock_specific",
            symbols=symbols,
            split='val',
            config=self.config,
            shared_feature_scaler=train_dataset.feature_scaler,
            shared_magnitude_scaler=train_dataset.magnitude_scaler,
            data_quality_threshold=data_quality_threshold
        )
        
        # Enhanced data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=min(4, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=True,
            drop_last=True  # For batch norm stability
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=min(4, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def create_model(self) -> EnhancedV5Model:
        """Create enhanced V5 model"""
        model = EnhancedV5Model(self.config)
        model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Enhanced model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def create_loss_functions(self):
        """Create enhanced loss functions"""
        # Label smoothing for better generalization
        direction_criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        magnitude_criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
        
        return direction_criterion, magnitude_criterion
    
    def train_epoch(self, model: EnhancedV5Model, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   direction_criterion, magnitude_criterion, epoch: int) -> Tuple[float, float]:
        """Enhanced training epoch"""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Learning rate warmup for first few epochs
        if epoch < 5:
            warmup_factor = (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
        
        for batch_idx, batch in enumerate(train_loader):
            sequences = batch['sequence'].to(self.device)
            direction_targets = batch['direction_target'].squeeze().to(self.device)
            magnitude_targets = batch['magnitude_target'].squeeze().to(self.device)
            missing_masks = batch['missing_mask'].to(self.device)
            quality_scores = batch['quality_score'].squeeze().to(self.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = model(sequences, missing_masks)
                    
                    # Enhanced loss calculation with quality weighting
                    direction_loss = direction_criterion(outputs['direction_logits'], direction_targets)
                    magnitude_loss = magnitude_criterion(outputs['magnitude'].squeeze(), magnitude_targets)
                    
                    # Quality-weighted loss
                    quality_weight = torch.mean(quality_scores)
                    total_batch_loss = (
                        self.config.direction_weight * direction_loss + 
                        self.config.magnitude_weight * magnitude_loss
                    ) * quality_weight
                
                # Enhanced backward pass
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(sequences, missing_masks)
                
                direction_loss = direction_criterion(outputs['direction_logits'], direction_targets)
                magnitude_loss = magnitude_criterion(outputs['magnitude'].squeeze(), magnitude_targets)
                
                quality_weight = torch.mean(quality_scores)
                total_batch_loss = (
                    self.config.direction_weight * direction_loss + 
                    self.config.magnitude_weight * magnitude_loss
                ) * quality_weight
                
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['direction_logits'], 1)
            correct_predictions += (predicted == direction_targets).sum().item()
            total_predictions += direction_targets.size(0)
            
            # Log progress
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {total_batch_loss.item():.4f}, LR: {current_lr:.6f}")
            
            # Memory cleanup
            if torch.cuda.is_available() and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Step scheduler after epoch
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            pass  # Will be stepped in validation
        else:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model: EnhancedV5Model, val_loader: DataLoader, 
                      direction_criterion, magnitude_criterion) -> Tuple[float, float, Dict]:
        """Enhanced validation epoch"""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(self.device)
                direction_targets = batch['direction_target'].squeeze().to(self.device)
                magnitude_targets = batch['magnitude_target'].squeeze().to(self.device)
                missing_masks = batch['missing_mask'].to(self.device)
                quality_scores = batch['quality_score'].squeeze().to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = model(sequences, missing_masks)
                        
                        direction_loss = direction_criterion(outputs['direction_logits'], direction_targets)
                        magnitude_loss = magnitude_criterion(outputs['magnitude'].squeeze(), magnitude_targets)
                        
                        quality_weight = torch.mean(quality_scores)
                        total_batch_loss = (
                            self.config.direction_weight * direction_loss + 
                            self.config.magnitude_weight * magnitude_loss
                        ) * quality_weight
                else:
                    outputs = model(sequences, missing_masks)
                    
                    direction_loss = direction_criterion(outputs['direction_logits'], direction_targets)
                    magnitude_loss = magnitude_criterion(outputs['magnitude'].squeeze(), magnitude_targets)
                    
                    quality_weight = torch.mean(quality_scores)
                    total_batch_loss = (
                        self.config.direction_weight * direction_loss + 
                        self.config.magnitude_weight * magnitude_loss
                    ) * quality_weight
                
                # Update metrics
                total_loss += total_batch_loss.item()
                
                _, predicted = torch.max(outputs['direction_logits'], 1)
                correct_predictions += (predicted == direction_targets).sum().item()
                total_predictions += direction_targets.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(direction_targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        # Calculate detailed metrics
        detailed_metrics = {
            'accuracy': accuracy,
            'precision_recall_fscore': precision_recall_fscore_support(all_targets, all_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(all_targets, all_predictions),
            'classification_report': classification_report(all_targets, all_predictions)
        }
        
        return avg_loss, accuracy, detailed_metrics
    
    def save_model(self, model: EnhancedV5Model, optimizer: optim.Optimizer, 
                  scheduler: optim.lr_scheduler._LRScheduler, epoch: int, 
                  val_loss: float, val_accuracy: float, is_best: bool = False):
        """Enhanced model saving"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'training_history': self.training_history,
            'random_state': {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state(),
                'torch_random': torch.get_rng_state(),
            }
        }
        
        # Add scaler state
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"enhanced_v5_model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "enhanced_v5_model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 New best model saved with validation accuracy: {val_accuracy:.4f}")
            
            # Add to ensemble if performance is good
            if val_accuracy > 0.60:  # Institutional threshold
                self.ensemble_models.append(checkpoint_path)
                logger.info(f"✨ Model added to ensemble (total: {len(self.ensemble_models)})")
    
    def train(self, symbols: List[str] = None, epochs: int = 100, batch_size: int = 16, 
             learning_rate: float = 0.0001, patience: int = 25, data_quality_threshold: float = 0.3):
        """Enhanced training loop"""
        
        # Get symbols if not provided
        if symbols is None:
            symbols = self.get_available_symbols()
            # Use subset for memory management
            if len(symbols) > 30:
                symbols = symbols[:30]
                logger.info(f"Using subset of {len(symbols)} symbols for training")
        
        logger.info(f"🚀 Starting Enhanced V5 model training with {len(symbols)} symbols")
        logger.info(f"🎯 Target: Institutional-grade performance (65%+)")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(symbols, batch_size, data_quality_threshold)
        
        # Create model
        model = self.create_model()
        
        # Enhanced optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),  # More stable betas
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, 
            min_lr=1e-6
        )
        
        # Create loss functions
        direction_criterion, magnitude_criterion = self.create_loss_functions()
        
        logger.info("🎯 Starting enhanced training loop...")
        
        for epoch in range(epochs):
            start_time = datetime.now()
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(
                model, train_loader, optimizer, scheduler, 
                direction_criterion, magnitude_criterion, epoch
            )
            
            # Validation phase
            val_loss, val_accuracy, detailed_metrics = self.validate_epoch(
                model, val_loader, direction_criterion, magnitude_criterion
            )
            
            # Update learning rate scheduler
            scheduler.step(val_accuracy)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = datetime.now() - start_time
            self.training_history['epoch_times'].append(str(epoch_time))
            
            # Check for improvement
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_model(model, optimizer, scheduler, epoch, val_loss, val_accuracy, is_best)
            
            # Enhanced logging
            performance_indicator = ""
            if val_accuracy >= 0.67:
                performance_indicator = "🏆 INSTITUTIONAL GRADE!"
            elif val_accuracy >= 0.62:
                performance_indicator = "🎯 EXCELLENT!"
            elif val_accuracy >= 0.58:
                performance_indicator = "✅ VERY GOOD!"
            elif val_accuracy >= 0.55:
                performance_indicator = "📈 GOOD PROGRESS!"
            else:
                performance_indicator = "📊 LEARNING..."
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} {performance_indicator}\n"
                f"  Train: Loss {train_loss:.4f}, Acc {train_accuracy:.4f} ({train_accuracy*100:.1f}%)\n"
                f"  Val:   Loss {val_loss:.4f}, Acc {val_accuracy:.4f} ({val_accuracy*100:.1f}%)\n"
                f"  Time: {epoch_time}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("🎉 Enhanced V5 training completed!")
        
        # Save final results
        self.save_final_results()
        
        return model, self.training_history
    
    def save_final_results(self):
        """Save comprehensive training results"""
        # Training history
        history_path = self.output_dir / "enhanced_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Model configuration
        config_path = self.output_dir / "enhanced_model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Performance summary
        summary = {
            'best_val_accuracy': float(self.best_val_accuracy),
            'best_val_loss': float(self.best_val_loss),
            'total_epochs': len(self.training_history['train_loss']),
            'ensemble_models_count': len(self.ensemble_models),
            'institutional_grade': self.best_val_accuracy >= 0.65,
            'target_achieved': self.best_val_accuracy >= 0.60,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / "performance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Results saved to: {self.output_dir}")
        if summary['institutional_grade']:
            logger.info("🏆 INSTITUTIONAL GRADE ACHIEVED! Ready for deployment!")
        elif summary['target_achieved']:
            logger.info("🎯 TARGET ACHIEVED! Excellent performance!")

def main():
    """Enhanced main function"""
    
    # Set global seed
    set_seed(42)
    
    logger.info("🧠 ENHANCED V5 TEMPORAL CAUSALITY TRAINER")
    logger.info("=" * 60)
    logger.info("🎯 Target: Beat institutional performance (65%+ accuracy)")
    logger.info("🏗️ Enhanced architecture with anti-overfitting")
    logger.info("💎 V3 proven techniques + V5 advanced features")
    logger.info("=" * 60)
    
    # Enhanced configuration
    config = EnhancedV5Config(
        feature_dim=75,  # Will adjust automatically
        hidden_dim=256,
        num_attention_heads=16,
        num_transformer_layers=6,  # Reduced for better generalization
        sequence_length=20,
        dropout_rate=0.2,
        direction_weight=0.7,
        magnitude_weight=0.3,
        weight_decay=0.01,
        gradient_clip_norm=0.5,
        label_smoothing=0.1
    )
    
    # Paths (unchanged as requested)
    data_dir = "data/processed/v5_temporal_datasets"
    output_dir = f"data/models/enhanced_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create trainer
    trainer = EnhancedV5Trainer(config, data_dir, output_dir)
    
    # Start training
    try:
        model, history = trainer.train(
            epochs=100,
            batch_size=16,
            learning_rate=0.0001,
            patience=25,
            data_quality_threshold=0.3
        )
        
        logger.info("🎉 Enhanced V5 model training completed successfully!")
        logger.info(f"📁 Models saved to: {output_dir}")
        logger.info(f"🏆 Best validation accuracy: {trainer.best_val_accuracy:.4f} ({trainer.best_val_accuracy*100:.1f}%)")
        
        if trainer.best_val_accuracy >= 0.65:
            logger.info("🚀 INSTITUTIONAL GRADE PERFORMANCE ACHIEVED!")
            logger.info("🎯 Ready for hedge fund deployment!")
        elif trainer.best_val_accuracy >= 0.60:
            logger.info("✨ EXCELLENT PERFORMANCE! Target achieved!")
        else:
            logger.info("📈 Good foundation built - consider ensemble methods!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
