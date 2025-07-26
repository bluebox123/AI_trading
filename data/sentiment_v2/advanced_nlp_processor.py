#!/usr/bin/env python3
"""
Advanced NLP Processor for Sentiment Analysis
============================================

This module provides comprehensive NLP preprocessing including:
- Language detection and translation
- Tokenization, lemmatization, and stopword removal
- Named Entity Recognition (NER)
- Entity resolution and synonym mapping
- Multi-model sentiment analysis (VADER + BERT)

Requirements:
pip install langdetect spacy nltk vaderSentiment transformers torch pandas numpy
python -m spacy download en_core_web_sm
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
import langdetect
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set seed for consistent language detection
langdetect.DetectorFactory.seed = 0

@dataclass
class ProcessedText:
    """Data class for processed text with NLP features"""
    original_text: str
    language: str
    is_english: bool
    translated_text: Optional[str]
    tokens: List[str]
    lemmatized_tokens: List[str]
    entities: List[Dict]
    resolved_entities: List[Dict]
    cleaned_text: str
    sentiment_scores: Dict[str, float]
    confidence: float
    processing_metadata: Dict[str, Any]

class AdvancedNLPProcessor:
    """Advanced NLP processor for sentiment analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the NLP processor with configuration"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize NLP components
        self._load_spacy_model()
        self._load_nltk_components()
        self._load_sentiment_models()
        self._load_entity_mappings()
        
        self.logger.info("Advanced NLP Processor initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the NLP processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for the NLP processor"""
        default_config = {
            "language_detection": {
                "supported_languages": ["en"],
                "translate_to_english": True,
                "min_confidence": 0.8
            },
            "text_preprocessing": {
                "remove_urls": True,
                "remove_emails": True,
                "remove_numbers": False,
                "remove_punctuation": False,
                "lowercase": True,
                "min_token_length": 2
            },
            "entity_resolution": {
                "company_aliases_file": "company_aliases.json",
                "ticker_mapping_file": "ticker_mapping.json",
                "confidence_threshold": 0.7
            },
            "sentiment_analysis": {
                "models": ["vader", "finbert"],
                "ensemble_weights": {"vader": 0.3, "finbert": 0.7},
                "confidence_threshold": 0.6
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _load_spacy_model(self):
        """Load spaCy model for NER and text processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy model loaded successfully")
        except OSError:
            self.logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def _load_nltk_components(self):
        """Load NLTK components for text preprocessing"""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.logger.info("NLTK components loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading NLTK components: {e}")
            raise
    
    def _load_sentiment_models(self):
        """Load sentiment analysis models"""
        self.sentiment_models = {}
        
        # Load VADER
        try:
            self.sentiment_models['vader'] = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer loaded")
        except Exception as e:
            self.logger.error(f"Error loading VADER: {e}")
        
        # Load FinBERT (if available)
        try:
            self.sentiment_models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("FinBERT sentiment analyzer loaded")
        except Exception as e:
            self.logger.warning(f"FinBERT not available: {e}")
    
    def _load_entity_mappings(self):
        """Load entity mappings for company names and tickers"""
        self.company_aliases = {}
        self.ticker_mapping = {}
        
        # Load company aliases
        aliases_file = Path(self.config["entity_resolution"]["company_aliases_file"])
        if aliases_file.exists():
            with open(aliases_file, 'r') as f:
                self.company_aliases = json.load(f)
        
        # Load ticker mapping
        ticker_file = Path(self.config["entity_resolution"]["ticker_mapping_file"])
        if ticker_file.exists():
            with open(ticker_file, 'r') as f:
                self.ticker_mapping = json.load(f)
        
        # Create default mappings if files don't exist
        if not self.company_aliases:
            self._create_default_company_aliases()
        
        if not self.ticker_mapping:
            self._create_default_ticker_mapping()
    
    def _create_default_company_aliases(self):
        """Create default company aliases for Indian stocks"""
        self.company_aliases = {
            "RELIANCE": ["Reliance Industries", "Reliance Industries Ltd", "RIL", "Reliance"],
            "TCS": ["Tata Consultancy Services", "TCS Ltd", "Tata CS"],
            "HDFCBANK": ["HDFC Bank", "HDFC Bank Ltd", "HDFC"],
            "INFY": ["Infosys", "Infosys Ltd", "Infosys Limited"],
            "ICICIBANK": ["ICICI Bank", "ICICI Bank Ltd", "ICICI"],
            "HINDUNILVR": ["Hindustan Unilever", "HUL", "Hindustan Unilever Ltd"],
            "ITC": ["ITC Ltd", "ITC Limited", "ITC"],
            "SBIN": ["State Bank of India", "SBI", "State Bank"],
            "BHARTIARTL": ["Bharti Airtel", "Airtel", "Bharti"],
            "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank", "Kotak"],
            "AXISBANK": ["Axis Bank", "Axis Bank Ltd"],
            "ASIANPAINT": ["Asian Paints", "Asian Paints Ltd"],
            "MARUTI": ["Maruti Suzuki", "Maruti Suzuki India", "Maruti"],
            "SUNPHARMA": ["Sun Pharmaceutical", "Sun Pharma", "Sun Pharmaceutical Industries"],
            "TATAMOTORS": ["Tata Motors", "Tata Motors Ltd"],
            "WIPRO": ["Wipro Ltd", "Wipro Limited", "Wipro"],
            "ULTRACEMCO": ["UltraTech Cement", "UltraTech", "UltraTech Cement Ltd"],
            "TITAN": ["Titan Company", "Titan", "Titan Ltd"],
            "BAJFINANCE": ["Bajaj Finance", "Bajaj Finance Ltd"],
            "NESTLEIND": ["Nestle India", "Nestle", "Nestle India Ltd"]
        }
        
        # Save to file
        with open(self.config["entity_resolution"]["company_aliases_file"], 'w') as f:
            json.dump(self.company_aliases, f, indent=2)
    
    def _create_default_ticker_mapping(self):
        """Create default ticker mapping for Indian stocks"""
        self.ticker_mapping = {
            "RELIANCE": "RELIANCE.NSE",
            "TCS": "TCS.NSE",
            "HDFCBANK": "HDFCBANK.NSE",
            "INFY": "INFY.NSE",
            "ICICIBANK": "ICICIBANK.NSE",
            "HINDUNILVR": "HINDUNILVR.NSE",
            "ITC": "ITC.NSE",
            "SBIN": "SBIN.NSE",
            "BHARTIARTL": "BHARTIARTL.NSE",
            "KOTAKBANK": "KOTAKBANK.NSE",
            "AXISBANK": "AXISBANK.NSE",
            "ASIANPAINT": "ASIANPAINT.NSE",
            "MARUTI": "MARUTI.NSE",
            "SUNPHARMA": "SUNPHARMA.NSE",
            "TATAMOTORS": "TATAMOTORS.NSE",
            "WIPRO": "WIPRO.NSE",
            "ULTRACEMCO": "ULTRACEMCO.NSE",
            "TITAN": "TITAN.NSE",
            "BAJFINANCE": "BAJFINANCE.NSE",
            "NESTLEIND": "NESTLEIND.NSE"
        }
        
        # Save to file
        with open(self.config["entity_resolution"]["ticker_mapping_file"], 'w') as f:
            json.dump(self.ticker_mapping, f, indent=2)
    
    def detect_language(self, text: str) -> Tuple[str, bool]:
        """Detect language and determine if it's English"""
        try:
            language = detect(text)
            is_english = language in self.config["language_detection"]["supported_languages"]
            return language, is_english
        except LangDetectException:
            return "unknown", False
    
    def translate_text(self, text: str, target_language: str = "en") -> Optional[str]:
        """Translate text to target language (placeholder for now)"""
        # For now, return None (no translation)
        # In production, you'd integrate with Google Translate API or similar
        return None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        if self.config["text_preprocessing"]["remove_urls"]:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        if self.config["text_preprocessing"]["remove_emails"]:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        if self.config["text_preprocessing"]["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation (optional)
        if self.config["text_preprocessing"]["remove_punctuation"]:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        if self.config["text_preprocessing"]["lowercase"]:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> Tuple[List[str], List[str]]:
        """Tokenize and lemmatize text with stopword removal"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens by length and remove stopwords
        min_length = self.config["text_preprocessing"]["min_token_length"]
        filtered_tokens = []
        for token in tokens:
            # Check length
            if len(token) >= min_length:
                # Check if it's a stopword
                if token.lower() not in self.stop_words:
                    filtered_tokens.append(token)
        
        # Lemmatize
        lemmatized_tokens = []
        for token in filtered_tokens:
            # Get POS tag for better lemmatization
            pos_tagged = pos_tag([token])
            pos = pos_tagged[0][1]
            
            # Map POS tags to WordNet POS tags
            if pos.startswith('J'):
                pos = 'a'  # adjective
            elif pos.startswith('V'):
                pos = 'v'  # verb
            elif pos.startswith('N'):
                pos = 'n'  # noun
            elif pos.startswith('R'):
                pos = 'r'  # adverb
            else:
                pos = 'n'  # default to noun
            
            lemmatized = self.lemmatizer.lemmatize(token, pos)
            lemmatized_tokens.append(lemmatized)
        
        return filtered_tokens, lemmatized_tokens
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def resolve_entities(self, entities: List[Dict]) -> List[Dict]:
        """Resolve entities to company names and tickers"""
        resolved_entities = []
        
        for entity in entities:
            if entity['label'] in ['ORG', 'PERSON']:  # Organization or Person
                company_name = entity['text'].upper()
                
                # Check if it matches any company alias
                matched_ticker = None
                confidence = 0.0
                
                for ticker, aliases in self.company_aliases.items():
                    if company_name in aliases or any(alias.upper() in company_name for alias in aliases):
                        matched_ticker = ticker
                        confidence = 0.8
                        break
                
                if matched_ticker:
                    resolved_entities.append({
                        'original_text': entity['text'],
                        'entity_type': entity['label'],
                        'company_name': matched_ticker,
                        'ticker': self.ticker_mapping.get(matched_ticker, f"{matched_ticker}.NSE"),
                        'confidence': confidence
                    })
        
        return resolved_entities
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if 'vader' not in self.sentiment_models:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        
        scores = self.sentiment_models['vader'].polarity_scores(text)
        return scores
    
    def analyze_sentiment_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        if 'finbert' not in self.sentiment_models:
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.0}
        
        try:
            result = self.sentiment_models['finbert'](text[:512])  # Limit length
            return {
                'label': result[0]['label'],
                'score': result[0]['score'],
                'confidence': result[0]['score']
            }
        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {e}")
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.0}
    
    def ensemble_sentiment_analysis(self, text: str) -> Tuple[Dict[str, float], float]:
        """Perform ensemble sentiment analysis"""
        results = {}
        weights = self.config["sentiment_analysis"]["ensemble_weights"]
        total_weight = 0
        weighted_score = 0
        
        # VADER analysis
        if 'vader' in self.sentiment_models:
            vader_scores = self.analyze_sentiment_vader(text)
            results['vader'] = vader_scores
            weight = weights.get('vader', 0.3)
            weighted_score += vader_scores['compound'] * weight
            total_weight += weight
        
        # FinBERT analysis
        if 'finbert' in self.sentiment_models:
            finbert_result = self.analyze_sentiment_finbert(text)
            results['finbert'] = finbert_result
            weight = weights.get('finbert', 0.7)
            
            # Convert FinBERT label to score
            if finbert_result['label'] == 'positive':
                finbert_score = finbert_result['score']
            elif finbert_result['label'] == 'negative':
                finbert_score = -finbert_result['score']
            else:
                finbert_score = 0.0
            
            weighted_score += finbert_score * weight
            total_weight += weight
        
        # Calculate ensemble score
        if total_weight > 0:
            ensemble_score = weighted_score / total_weight
        else:
            ensemble_score = 0.0
        
        # Calculate confidence
        confidences = []
        for model, result in results.items():
            if model == 'vader':
                confidences.append(abs(result['compound']))
            elif model == 'finbert':
                confidences.append(result['confidence'])
        
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        return results, ensemble_score, avg_confidence
    
    def process_text(self, text: str) -> ProcessedText:
        """Complete text processing pipeline"""
        self.logger.debug(f"Processing text: {text[:100]}...")
        
        # Language detection
        language, is_english = self.detect_language(text)
        
        # Translation (if needed)
        translated_text = None
        if not is_english and self.config["language_detection"]["translate_to_english"]:
            translated_text = self.translate_text(text)
            if translated_text:
                text = translated_text
        
        # Text preprocessing
        cleaned_text = self.preprocess_text(text)
        
        # Tokenization and lemmatization
        tokens, lemmatized_tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        # Entity extraction
        entities = self.extract_entities(text)
        
        # Entity resolution
        resolved_entities = self.resolve_entities(entities)
        
        # Sentiment analysis
        sentiment_results, ensemble_score, confidence = self.ensemble_sentiment_analysis(cleaned_text)
        
        # Create sentiment scores dict
        sentiment_scores = {
            'ensemble_score': ensemble_score,
            'vader_compound': sentiment_results.get('vader', {}).get('compound', 0.0),
            'finbert_score': sentiment_results.get('finbert', {}).get('score', 0.0),
            'finbert_label': sentiment_results.get('finbert', {}).get('label', 'neutral')
        }
        
        # Processing metadata
        metadata = {
            'language_detected': language,
            'is_english': is_english,
            'translated': translated_text is not None,
            'token_count': len(tokens),
            'entity_count': len(entities),
            'resolved_entity_count': len(resolved_entities),
            'sentiment_models_used': list(sentiment_results.keys())
        }
        
        return ProcessedText(
            original_text=text,
            language=language,
            is_english=is_english,
            translated_text=translated_text,
            tokens=tokens,
            lemmatized_tokens=lemmatized_tokens,
            entities=entities,
            resolved_entities=resolved_entities,
            cleaned_text=cleaned_text,
            sentiment_scores=sentiment_scores,
            confidence=confidence,
            processing_metadata=metadata
        )

def main():
    """Test the advanced NLP processor"""
    processor = AdvancedNLPProcessor()
    
    # Test texts
    test_texts = [
        "Reliance Industries reported strong Q4 earnings today, beating analyst expectations.",
        "TCS stock price surged 5% after positive quarterly results.",
        "HDFC Bank announced new digital banking initiatives.",
        "Infosys shares fell 3% due to weak guidance."
    ]
    
    print("Testing Advanced NLP Processor")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 40)
        
        result = processor.process_text(text)
        
        print(f"Language: {result.language} (English: {result.is_english})")
        print(f"Cleaned text: {result.cleaned_text}")
        print(f"Tokens: {result.tokens[:10]}...")
        print(f"Entities found: {len(result.entities)}")
        print(f"Resolved entities: {len(result.resolved_entities)}")
        
        for entity in result.resolved_entities:
            print(f"  - {entity['original_text']} -> {entity['ticker']} (confidence: {entity['confidence']:.2f})")
        
        print(f"Sentiment scores:")
        print(f"  - Ensemble: {result.sentiment_scores['ensemble_score']:.3f}")
        print(f"  - VADER: {result.sentiment_scores['vader_compound']:.3f}")
        print(f"  - FinBERT: {result.sentiment_scores['finbert_score']:.3f} ({result.sentiment_scores['finbert_label']})")
        print(f"Confidence: {result.confidence:.3f}")

if __name__ == "__main__":
    main() 