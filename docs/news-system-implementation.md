# TradingSignals News System Implementation Guide

## Overview

The TradingSignals News System is a comprehensive market intelligence platform that integrates EODHD and Perplexity APIs to provide real-time NSE market news analysis with advanced sentiment scoring and impact assessment.

## Architecture

### Data Flow
```
EODHD API (Hourly) ──┐
                     ├──> News Aggregator ──> Sentiment Analyzer ──> UI Components
Perplexity API (3hr) ──┘
```

### Key Components

#### 1. API Layer (`/api/news/`)
- **Main Route** (`route.ts`): Orchestrates data from both APIs
- **Sentiment Route** (`sentiment/route.ts`): Provides aggregated sentiment analysis
- **Caching Strategy**: In-memory caching with Redis ready structure
- **Rate Limiting**: Built-in quota management for Perplexity API

#### 2. Data Models (`/lib/types.ts`)
- **NewsItem**: Unified data structure for all news sources
- **SentimentTrend**: Time-series sentiment data
- **TopMover**: Stocks with significant sentiment changes
- **SectorSentiment**: Sector-wide sentiment analysis

#### 3. UI Components
- **NewsCard**: Rich news article display with sentiment badges
- **NewsFilters**: Advanced filtering with search, date range, source, sentiment, sector, and ticker filters
- **SentimentVisualization**: Interactive charts and trend analysis
- **AppLayout Integration**: Seamless integration with existing layout

## Features Implemented

### ✅ Core Features
- [x] Unified news aggregation from multiple sources
- [x] Advanced sentiment analysis and scoring
- [x] Multi-dimensional filtering (date, source, sentiment, sector, ticker)
- [x] Real-time search with debouncing
- [x] Infinite scroll with pagination
- [x] Responsive design with dark mode support
- [x] Offline support with caching
- [x] Error handling and graceful degradation

### ✅ Sentiment Analysis
- [x] Sentiment trend sparklines
- [x] Top sentiment movers identification
- [x] Sector-wise sentiment heatmap
- [x] Market regime detection
- [x] Confidence scoring

### ✅ User Experience
- [x] Interactive news cards with expandable content
- [x] One-click ticker/sector filtering
- [x] Social sharing functionality
- [x] Bookmark capability
- [x] Loading states and skeleton screens
- [x] Toast notifications for errors

## API Integration TODOs

### 🔧 EODHD API Integration

```typescript
// File: /api/news/route.ts
async function fetchEODHDNews(tickers: string[], fromDate?: string, toDate?: string): Promise<NewsItem[]> {
  const results: NewsItem[] = [];
  
  for (const ticker of tickers) {
    const url = `https://eodhd.com/api/news?s=${ticker}&from=${fromDate}&to=${toDate}&limit=50&api_token=${EODHD_API_KEY}&fmt=json`;
    
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'TradingSignals/1.0' }
      });
      
      if (!response.ok) {
        throw new Error(`EODHD API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform EODHD data to NewsItem format
      const transformedNews = data.map(item => normalizeEODHDNewsItem(item, ticker));
      results.push(...transformedNews);
      
      // Rate limiting: 20 requests per minute
      await new Promise(resolve => setTimeout(resolve, 3000));
      
    } catch (error) {
      console.error(`Failed to fetch EODHD news for ${ticker}:`, error);
      // Continue with other tickers
    }
  }
  
  return results;
}

function normalizeEODHDNewsItem(item: any, ticker: string): NewsItem {
  return {
    id: `eodhd-${item.title.slice(0, 20).replace(/\s+/g, '-')}-${Date.now()}`,
    title: item.title,
    summary: item.content.slice(0, 200) + '...',
    content: item.content,
    url: item.link,
    publishedAt: new Date(item.date),
    source: 'EODHD',
    sourceUrl: extractDomain(item.link),
    sentiment: {
      label: item.sentiment?.polarity > 0.1 ? 'Positive' : 
             item.sentiment?.polarity < -0.1 ? 'Negative' : 'Neutral',
      score: item.sentiment?.polarity || 0,
      confidence: item.sentiment ? 0.8 : 0.5
    },
    impact: {
      affectedTickers: item.symbols || [ticker],
      sectors: extractSectorsFromTags(item.tags),
      tags: item.tags || []
    }
  };
}
```

### 🔧 Perplexity API Integration

```typescript
// File: /api/news/route.ts
async function fetchPerplexityAnalysis(tickers: string[]): Promise<NewsItem[]> {
  if (!canMakePerplexityRequest()) {
    console.log('Perplexity quota exceeded, skipping analysis');
    return [];
  }
  
  const prompt = `
As a financial analyst, provide comprehensive market analysis for NSE stocks: ${tickers.join(', ')}.

Return a JSON array of news analysis objects with these fields:
- headline: Analysis title
- summary: 2-3 sentence market impact summary
- symbols: Array of relevant NSE tickers
- sentiment: "Positive", "Negative", or "Neutral"
- sentiment_score: Number from -1 to 1
- sectors: Array of affected sectors
- citations: Array of source URLs

Focus on:
1. Recent earnings and financial performance
2. Regulatory changes affecting these stocks
3. Sector trends and comparative analysis
4. Market sentiment and institutional activity

Return ONLY the JSON array, no additional text.
`;

  try {
    const response = await fetch('https://api.perplexity.ai/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${PERPLEXITY_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'llama-3.1-sonar-small-128k-online',
        messages: [
          {
            role: 'system',
            content: 'You are a financial analyst providing structured market analysis in JSON format.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.1,
        max_tokens: 2000
      })
    });

    if (!response.ok) {
      throw new Error(`Perplexity API error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices[0].message.content;
    
    // Parse JSON response
    const analysisItems = JSON.parse(content);
    
    // Transform to NewsItem format
    return analysisItems.map((item: any, index: number) => 
      normalizePerplexityNewsItem(item, index)
    );
    
  } catch (error) {
    console.error('Perplexity API error:', error);
    return [];
  }
}

function normalizePerplexityNewsItem(item: any, index: number): NewsItem {
  return {
    id: `perplexity-${Date.now()}-${index}`,
    title: item.headline,
    summary: item.summary,
    publishedAt: new Date(),
    source: 'Perplexity',
    sentiment: {
      label: item.sentiment,
      score: item.sentiment_score,
      confidence: 0.9
    },
    impact: {
      affectedTickers: item.symbols,
      sectors: item.sectors,
      tags: ['ai-analysis', 'market-intelligence']
    },
    citations: item.citations
  };
}
```

### 🔧 Sentiment Enhancement

```typescript
// File: /api/news/sentiment/route.ts
async function fetchEODHDSentiment(tickers: string[]): Promise<SentimentTrend[]> {
  const trends: SentimentTrend[] = [];
  
  for (const ticker of tickers) {
    const url = `https://eodhd.com/api/sentiments?s=${ticker}&from=${getDateDaysAgo(7)}&fmt=json&api_token=${EODHD_API_KEY}`;
    
    try {
      const response = await fetch(url);
      const data = await response.json();
      
      // Transform EODHD sentiment data
      Object.entries(data[ticker] || {}).forEach(([date, sentiment]: [string, any]) => {
        trends.push({
          date: new Date(date).toISOString(),
          sentiment: sentiment.polarity,
          articleCount: Math.round((sentiment.pos + sentiment.neg + sentiment.neu) * 100),
          ticker
        });
      });
      
    } catch (error) {
      console.error(`Failed to fetch sentiment for ${ticker}:`, error);
    }
  }
  
  return trends.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}
```

## Caching Strategy

### 🔧 Redis Implementation

```typescript
// File: /lib/cache.ts
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

export class NewsCache {
  private static readonly EODHD_KEY = 'news:eodhd';
  private static readonly PERPLEXITY_KEY = 'news:perplexity';
  private static readonly SENTIMENT_KEY = 'sentiment';
  
  static async getEODHDNews(): Promise<NewsItem[] | null> {
    const cached = await redis.get(this.EODHD_KEY);
    return cached ? JSON.parse(cached) : null;
  }
  
  static async setEODHDNews(news: NewsItem[]): Promise<void> {
    await redis.setex(this.EODHD_KEY, 3600, JSON.stringify(news)); // 1 hour TTL
  }
  
  static async getPerplexityNews(): Promise<NewsItem[] | null> {
    const cached = await redis.get(this.PERPLEXITY_KEY);
    return cached ? JSON.parse(cached) : null;
  }
  
  static async setPerplexityNews(news: NewsItem[]): Promise<void> {
    await redis.setex(this.PERPLEXITY_KEY, 10800, JSON.stringify(news)); // 3 hours TTL
  }
  
  static async getSentimentTrends(key: string): Promise<SentimentTrend[] | null> {
    const cached = await redis.get(`${this.SENTIMENT_KEY}:${key}`);
    return cached ? JSON.parse(cached) : null;
  }
  
  static async setSentimentTrends(key: string, trends: SentimentTrend[]): Promise<void> {
    await redis.setex(`${this.SENTIMENT_KEY}:${key}`, 1800, JSON.stringify(trends)); // 30 min TTL
  }
}
```

## Performance Optimizations

### 🔧 Virtual Scrolling (Optional)
For handling large news feeds:

```bash
npm install @tanstack/react-virtual
```

```typescript
// File: /components/VirtualizedNewsList.tsx
import { useVirtualizer } from '@tanstack/react-virtual';

export function VirtualizedNewsList({ news }: { news: NewsItem[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: news.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200, // Estimated height of NewsCard
    overscan: 5
  });

  return (
    <div ref={parentRef} className="h-screen overflow-auto">
      <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: virtualItem.size,
              transform: `translateY(${virtualItem.start}px)`
            }}
          >
            <NewsCard article={news[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Deployment Checklist

### 🔧 Environment Variables
```bash
# EODHD Configuration
EODHD_API_KEY=your_api_key_here
EODHD_BASE_URL=https://eodhd.com/api
EODHD_RATE_LIMIT=20

# Perplexity Configuration
PERPLEXITY_API_KEY=your_api_key_here
PERPLEXITY_DAILY_LIMIT=5

# Redis Configuration
REDIS_URL=redis://localhost:6379

# System Configuration
NEWS_CACHE_TTL=1800
NEWS_UPDATE_INTERVAL=3600000
```

### 🔧 Monitoring & Logging
```typescript
// File: /lib/monitoring.ts
export function logAPIUsage(source: 'EODHD' | 'Perplexity', success: boolean, responseTime: number) {
  console.log(`[${source}] ${success ? 'SUCCESS' : 'FAILED'} - ${responseTime}ms`);
  
  // Send to monitoring service (e.g., DataDog, New Relic)
  if (process.env.NODE_ENV === 'production') {
    // await monitoringService.track({
    //   event: 'api_call',
    //   source,
    //   success,
    //   responseTime
    // });
  }
}
```

## Future Enhancements

### 🚀 Phase 2 Features
- [ ] **Real-time WebSocket Updates**: Live news streaming
- [ ] **AI-Powered Summaries**: Automated article summarization
- [ ] **Personalized Recommendations**: User-specific news curation
- [ ] **Advanced Analytics**: Detailed sentiment correlation analysis
- [ ] **Mobile App**: React Native implementation
- [ ] **Export Features**: PDF reports and CSV data export

### 🚀 Phase 3 Features
- [ ] **Multi-language Support**: Hindi and regional languages
- [ ] **Voice Summaries**: AI-generated audio news briefs
- [ ] **Predictive Analytics**: ML-based market movement predictions
- [ ] **Social Sentiment**: Twitter/Reddit sentiment integration
- [ ] **Custom Alerts**: Webhook notifications for specific events

## Support & Maintenance

### 🔧 Regular Tasks
1. **API Key Rotation**: Monthly rotation for security
2. **Cache Optimization**: Monitor Redis usage and optimize TTL values
3. **Sentiment Model Updates**: Quarterly model retraining
4. **Performance Monitoring**: Track API response times and error rates

### 🔧 Troubleshooting
- **API Rate Limits**: Implement exponential backoff
- **Cache Invalidation**: Clear cache during deployment
- **Sentiment Accuracy**: Monitor sentiment vs. market movement correlation

## Contributing

When contributing to the news system:
1. Follow the established data models in `/lib/types.ts`
2. Add proper error handling and logging
3. Write unit tests for new API integrations
4. Update this documentation for new features

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintainer**: TradingSignals Team 