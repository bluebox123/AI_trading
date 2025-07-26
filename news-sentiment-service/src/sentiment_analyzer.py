"""
Sentiment analyzer module using FinBERT for financial news sentiment analysis
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import numpy as np

from config.config import (
    FINBERT_MODEL_NAME, SENTIMENT_LABELS, MIN_CONFIDENCE_THRESHOLD,
    BATCH_SIZE, MAX_TEXT_LENGTH
)

class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news headlines
    """
    
    def __init__(self, model_name: str = FINBERT_MODEL_NAME):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.labels = SENTIMENT_LABELS
        self._load_model()
        
    def _get_device(self) -> str:
        """Determine the best device for inference"""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            self.logger.info("Using Apple Silicon MPS")
        else:
            device = "cpu"
            self.logger.info("Using CPU for inference")
        return device
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer"""
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Test the model with a sample input
            self._test_model()
            
        except Exception as e:
            self.logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def _test_model(self):
        """Test the model with a sample input to ensure it's working"""
        try:
            test_text = "The company reported strong quarterly earnings with revenue growth."
            result = self.analyze_text(test_text)
            self.logger.info(f"Model test successful. Sample result: {result['sentiment']} ({result['confidence']:.3f})")
        except Exception as e:
            self.logger.error(f"Model test failed: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better sentiment analysis
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (tokenizer will handle this too, but let's be explicit)
        if len(text) > MAX_TEXT_LENGTH * 4:  # Rough character estimate
            text = text[:MAX_TEXT_LENGTH * 4]
            self.logger.debug("Text truncated due to length")
        
        return text
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {label: 0.0 for label in self.labels},
                'error': 'Empty text'
            }
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=MAX_TEXT_LENGTH
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy()
            
            # Get predicted class
            predicted_idx = int(torch.argmax(logits, dim=-1).cpu())
            predicted_sentiment = self.labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Create probability dictionary
            prob_dict = {
                label: float(prob) for label, prob in zip(self.labels, probabilities)
            }
            
            # Check confidence threshold
            is_confident = confidence >= MIN_CONFIDENCE_THRESHOLD
            
            result = {
                'sentiment': predicted_sentiment,
                'confidence': confidence,
                'probabilities': prob_dict,
                'is_confident': is_confident,
                'model_name': self.model_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.debug(f"Analyzed text: '{text[:50]}...' -> {predicted_sentiment} ({confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {label: 0.0 for label in self.labels},
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts (more efficient for multiple texts)
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            self.logger.debug(f"Processing batch {i//BATCH_SIZE + 1}, size: {len(batch_texts)}")
            
            try:
                # Preprocess all texts in batch
                processed_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # Filter out empty texts but keep track of indices
                valid_texts = []
                valid_indices = []
                for idx, text in enumerate(processed_texts):
                    if text and text.strip():
                        valid_texts.append(text)
                        valid_indices.append(idx)
                
                if not valid_texts:
                    # All texts in this batch are empty
                    batch_results = [self._empty_result() for _ in batch_texts]
                    results.extend(batch_results)
                    continue
                
                # Tokenize batch
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=MAX_TEXT_LENGTH
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                # Get probabilities for all samples in batch
                probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                predicted_indices = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # Process results for valid texts
                valid_results = []
                for idx, (probs, pred_idx) in enumerate(zip(probabilities, predicted_indices)):
                    sentiment = self.labels[pred_idx]
                    confidence = float(probs[pred_idx])
                    
                    prob_dict = {
                        label: float(prob) for label, prob in zip(self.labels, probs)
                    }
                    
                    result = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'probabilities': prob_dict,
                        'is_confident': confidence >= MIN_CONFIDENCE_THRESHOLD,
                        'model_name': self.model_name,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    valid_results.append(result)
                
                # Reconstruct full batch results (including empty texts)
                batch_results = [self._empty_result() for _ in batch_texts]
                for valid_idx, result in zip(valid_indices, valid_results):
                    batch_results[valid_idx] = result
                
                results.extend(batch_results)
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                # Return error results for the entire batch
                batch_results = [self._error_result(str(e)) for _ in batch_texts]
                results.extend(batch_results)
        
        self.logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def _empty_result(self) -> Dict:
        """Return result for empty text"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in self.labels},
            'is_confident': False,
            'error': 'Empty text',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _error_result(self, error_msg: str) -> Dict:
        """Return result for error case"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in self.labels},
            'is_confident': False,
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of news articles
        """
        if not articles:
            return []
        
        self.logger.info(f"Analyzing sentiment for {len(articles)} articles")
        start_time = time.time()
        
        # Extract headlines for batch processing
        headlines = [article.get('headline', '') for article in articles]
        
        # Perform batch sentiment analysis
        sentiment_results = self.analyze_batch(headlines)
        
        # Combine articles with sentiment results
        enriched_articles = []
        for article, sentiment in zip(articles, sentiment_results):
            enriched_article = article.copy()
            enriched_article.update({
                'sentiment': sentiment['sentiment'],
                'sentiment_confidence': sentiment['confidence'],
                'sentiment_probabilities': sentiment['probabilities'],
                'sentiment_is_confident': sentiment.get('is_confident', False),
                'sentiment_model': sentiment.get('model_name', self.model_name),
                'sentiment_timestamp': sentiment['timestamp']
            })
            
            if 'error' in sentiment:
                enriched_article['sentiment_error'] = sentiment['error']
            
            enriched_articles.append(enriched_article)
        
        analysis_time = time.time() - start_time
        self.logger.info(f"Sentiment analysis completed in {analysis_time:.2f} seconds")
        
        return enriched_articles
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'labels': self.labels,
            'min_confidence_threshold': MIN_CONFIDENCE_THRESHOLD,
            'batch_size': BATCH_SIZE,
            'max_text_length': MAX_TEXT_LENGTH,
            'model_loaded': self.model is not None
        }


if __name__ == "__main__":
    # Test the sentiment analyzer
    logging.basicConfig(level=logging.INFO)
    
    analyzer = SentimentAnalyzer()
    
    # Test with sample headlines
    test_headlines = [
        "Reliance Industries reports record quarterly profits beating estimates",
        "TCS shares fall as company misses revenue expectations",
        "HDFC Bank maintains steady growth in digital banking segment",
        "Infosys announces major layoffs amid economic uncertainty",
        "Asian Paints launches new product line in premium segment"
    ]
    
    print("Testing individual analysis:")
    for headline in test_headlines:
        result = analyzer.analyze_text(headline)
        print(f"'{headline}' -> {result['sentiment']} ({result['confidence']:.3f})")
    
    print("\nTesting batch analysis:")
    batch_results = analyzer.analyze_batch(test_headlines)
    for headline, result in zip(test_headlines, batch_results):
        print(f"'{headline}' -> {result['sentiment']} ({result['confidence']:.3f})") 