#!/usr/bin/env python3
"""
Simple Intraday Sentiment Analysis for 117 NSE Stocks
=====================================================

One-time sentiment analysis that fetches current news and provides
sentiment scores for all 117 NSE stocks immediately.
"""

import re
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import feedparser
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob

class SimpleIntradaySentiment:
    """Simple one-time sentiment analysis for NSE stocks"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.stocks = self._load_stocks()
        self.model = None
        self.tokenizer = None
        self._load_models()
        
        # News sources
        self.news_sources = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/businessnews.xml',
            'economictimes': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'livemint': 'https://www.livemint.com/rss/markets',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'ndtv_business': 'https://feeds.feedburner.com/ndtvbusiness-latest'
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup simple logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_stocks(self) -> List[str]:
        """Load the 117 NSE stock symbols"""
        stocks = []
        try:
            with open('../../nse_stock_symbols_complete.txt', 'r') as f:
                content = f.read()
                pattern = r'([A-Z][A-Z0-9\-]*\.NSE)'
                stocks = re.findall(pattern, content)
                self.logger.info(f"Loaded {len(stocks)} stock symbols")
        except FileNotFoundError:
            self.logger.error("Stock symbols file not found")
            # Fallback list
            stocks = [
                'RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE', 'INFY.NSE', 'ICICIBANK.NSE',
                'HINDUNILVR.NSE', 'ITC.NSE', 'SBIN.NSE', 'BHARTIARTL.NSE', 'KOTAKBANK.NSE'
            ]
        
        return stocks
    
    def _load_models(self):
        """Load FinBERT model for sentiment analysis"""
        try:
            self.logger.info("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.warning(f"FinBERT model loading failed: {e}. Using TextBlob fallback.")
            self.model = None
            self.tokenizer = None
    
    def _analyze_sentiment_finbert(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using FinBERT"""
        if not self.model or not self.tokenizer:
            return self._analyze_sentiment_textblob(text)
        
        try:
            # Clean and limit text
            text = text[:512]
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: negative, neutral, positive
            labels = ['negative', 'neutral', 'positive']
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert to sentiment score (-1 to 1)
            if predicted_class == 0:  # negative
                sentiment_score = -confidence
            elif predicted_class == 2:  # positive
                sentiment_score = confidence
            else:  # neutral
                sentiment_score = 0.0
            
            return sentiment_score, labels[predicted_class], confidence
            
        except Exception as e:
            self.logger.warning(f"FinBERT analysis error: {e}. Using TextBlob fallback.")
            return self._analyze_sentiment_textblob(text)
    
    def _analyze_sentiment_textblob(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment using TextBlob as fallback"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            confidence = abs(blob.sentiment.subjectivity)
            
            if sentiment_score > 0.1:
                label = "positive"
            elif sentiment_score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return sentiment_score, label, confidence
            
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {e}")
            return 0.0, "neutral", 0.5
    
    def _fetch_news(self) -> List[Dict]:
        """Fetch current news from all sources"""
        all_news = []
        
        for source_name, rss_url in self.news_sources.items():
            try:
                self.logger.info(f"Fetching news from {source_name}...")
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:20]:  # Get more entries for better coverage
                    news_item = {
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', entry.get('description', '')),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_name,
                        'full_text': f"{entry.get('title', '')} {entry.get('summary', entry.get('description', ''))}"
                    }
                    all_news.append(news_item)
                
                self.logger.info(f"Fetched {len(feed.entries)} articles from {source_name}")
                
            except Exception as e:
                self.logger.error(f"Error fetching news from {source_name}: {e}")
        
        self.logger.info(f"Total news articles fetched: {len(all_news)}")
        return all_news
    
    def _extract_stock_mentions(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        text_lower = text.lower()
        mentions = []
        
        for stock in self.stocks:
            # Remove .NSE suffix for matching
            stock_name = stock.replace('.NSE', '').lower()
            
            # Check for exact stock name matches
            if stock_name in text_lower:
                mentions.append(stock)
                continue
            
            # Check for common variations
            variations = [
                stock_name.replace('-', ' '),  # BAJAJ-AUTO -> bajaj auto
                stock_name.replace('_', ' '),  # Any underscores
                stock_name.replace('LTD', ''),  # Remove LTD
            ]
            
            for variation in variations:
                if variation and variation in text_lower:
                    mentions.append(stock)
                    break
        
        return list(set(mentions))  # Remove duplicates
    
    def _get_stock_sentiment(self, stock: str, news_articles: List[Dict]) -> Dict:
        """Get sentiment for a specific stock"""
        # Find articles mentioning this stock
        relevant_articles = []
        stock_name = stock.replace('.NSE', '').lower()
        
        for article in news_articles:
            text = article['full_text'].lower()
            if stock_name in text:
                relevant_articles.append(article)
        
        if not relevant_articles:
            return {
                'symbol': stock,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.5,
                'news_count': 0,
                'sample_headlines': []
            }
        
        # Combine all relevant text
        combined_text = " ".join([article['full_text'] for article in relevant_articles])
        
        # Analyze sentiment
        sentiment_score, sentiment_label, confidence = self._analyze_sentiment_finbert(combined_text)
        
        # Get sample headlines
        sample_headlines = [article['title'] for article in relevant_articles[:3]]
        
        return {
            'symbol': stock,
            'sentiment_score': round(sentiment_score, 4),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 4),
            'news_count': len(relevant_articles),
            'sample_headlines': sample_headlines
        }
    
    def analyze_all_stocks(self) -> List[Dict]:
        """Analyze sentiment for all 117 stocks"""
        self.logger.info("Starting sentiment analysis for all stocks...")
        
        # Fetch current news
        news_articles = self._fetch_news()
        
        if not news_articles:
            self.logger.warning("No news articles found. Results may be limited.")
        
        # Analyze each stock
        results = []
        for i, stock in enumerate(self.stocks, 1):
            self.logger.info(f"Analyzing {stock} ({i}/{len(self.stocks)})")
            sentiment_data = self._get_stock_sentiment(stock, news_articles)
            results.append(sentiment_data)
        
        return results
    
    def display_results(self, results: List[Dict]):
        """Display sentiment results in a formatted way"""
        print(f"\n{'='*80}")
        print(f"INTRADAY SENTIMENT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Analyzed {len(results)} stocks")
        
        # Summary statistics
        positive_count = sum(1 for r in results if r['sentiment_score'] > 0.1)
        negative_count = sum(1 for r in results if r['sentiment_score'] < -0.1)
        neutral_count = len(results) - positive_count - negative_count
        news_total = sum(r['news_count'] for r in results)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Positive: {positive_count} stocks")
        print(f"   Negative: {negative_count} stocks")
        print(f"   Neutral:  {neutral_count} stocks")
        print(f"   Total news analyzed: {news_total} articles")
        
        # Sort by sentiment score
        results_sorted = sorted(results, key=lambda x: x['sentiment_score'], reverse=True)
        
        print(f"\n🔥 TOP 10 MOST POSITIVE:")
        print(f"{'Stock':<15} {'Score':<8} {'Label':<10} {'News':<5} {'Confidence':<10}")
        print("-" * 55)
        for result in results_sorted[:10]:
            if result['sentiment_score'] > 0:
                print(f"{result['symbol']:<15} {result['sentiment_score']:<8} {result['sentiment_label']:<10} {result['news_count']:<5} {result['confidence']:<10}")
        
        print(f"\n💥 TOP 10 MOST NEGATIVE:")
        print(f"{'Stock':<15} {'Score':<8} {'Label':<10} {'News':<5} {'Confidence':<10}")
        print("-" * 55)
        for result in results_sorted[-10:]:
            if result['sentiment_score'] < 0:
                print(f"{result['symbol']:<15} {result['sentiment_score']:<8} {result['sentiment_label']:<10} {result['news_count']:<5} {result['confidence']:<10}")
        
        print(f"\n📈 STOCKS WITH MOST NEWS COVERAGE:")
        most_news = sorted(results, key=lambda x: x['news_count'], reverse=True)[:10]
        print(f"{'Stock':<15} {'News Count':<10} {'Score':<8} {'Label':<10}")
        print("-" * 50)
        for result in most_news:
            if result['news_count'] > 0:
                print(f"{result['symbol']:<15} {result['news_count']:<10} {result['sentiment_score']:<8} {result['sentiment_label']:<10}")
        
        # Show strong sentiments
        strong_positive = [r for r in results if r['sentiment_score'] > 0.5]
        strong_negative = [r for r in results if r['sentiment_score'] < -0.5]
        
        if strong_positive:
            print(f"\n🚀 STRONG POSITIVE SENTIMENT (>0.5):")
            for result in strong_positive:
                print(f"   {result['symbol']}: {result['sentiment_score']} ({result['news_count']} articles)")
                if result['sample_headlines']:
                    print(f"      Sample: {result['sample_headlines'][0][:60]}...")
        
        if strong_negative:
            print(f"\n📉 STRONG NEGATIVE SENTIMENT (<-0.5):")
            for result in strong_negative:
                print(f"   {result['symbol']}: {result['sentiment_score']} ({result['news_count']} articles)")
                if result['sample_headlines']:
                    print(f"      Sample: {result['sample_headlines'][0][:60]}...")
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intraday_sentiment_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main function"""
    print("🚀 Starting Simple Intraday Sentiment Analysis...")
    
    analyzer = SimpleIntradaySentiment()
    
    # Analyze all stocks
    results = analyzer.analyze_all_stocks()
    
    # Display results
    analyzer.display_results(results)
    
    # Save results
    analyzer.save_results(results)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 