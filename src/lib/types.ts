// News API Types
export interface EODHDNewsItem {
  date: string;
  title: string;
  content: string;
  link: string;
  symbols: string[];
  tags?: string[];
  sentiment?: {
    polarity: number;
    neg: number;
    neu: number;
    pos: number;
  };
}

export interface PerplexityNewsItem {
  headline: string;
  summary: string;
  symbols: string[];
  source: string;
  sentiment: 'Positive' | 'Negative' | 'Neutral';
  sentiment_score: number;
  citations?: string[];
}

export interface PerplexityMarketAnalysis {
  market_regime: 'Bullish' | 'Bearish' | 'Neutral' | 'High Volatility';
  regime_confidence: number;
  key_drivers: string;
  sentiment_score: number;
  top_sectors: string[];
}

// Unified News Item Model
export interface NewsItem {
  id: string;
  title: string;
  summary?: string;
  content?: string;
  url?: string;
  publishedAt: Date | string; // Can be Date object or ISO string from API
  source: 'EODHD' | 'Perplexity' | 'Combined';
  sourceUrl?: string;
  
  // Sentiment Analysis
  sentiment: {
    label: 'Positive' | 'Negative' | 'Neutral';
    score: number; // -1 to +1
    confidence: number; // 0 to 1
  };
  
  // Impact Analysis
  impact: {
    affectedTickers: string[]; // NSE symbols
    sectors: string[];
    tags: string[];
  };
  
  // Meta Information
  articleCount?: number; // For combined/aggregated items
  citations?: string[];
  isDuplicate?: boolean;
  duplicateOf?: string;
}

// Market Data Types
export interface SentimentTrend {
  date: string;
  sentiment: number;
  articleCount: number;
  ticker?: string;
  sector?: string;
}

export interface TopMover {
  ticker: string;
  name: string;
  sentimentChange: number;
  timeframe: '1h' | '4h' | '1d';
  articleCount: number;
  currentSentiment: number;
}

export interface SectorSentiment {
  sector: string;
  sentiment: number;
  confidence: number;
  articleCount: number;
  topTickers: string[];
  trend: 'increasing' | 'decreasing' | 'stable';
}

// Filter Types
export interface NewsFilters {
  dateRange: 'intraday' | '3days' | '1week' | '1month';
  sources: ('EODHD' | 'Perplexity' | 'Combined')[];
  sentiments: ('Positive' | 'Negative' | 'Neutral')[];
  sectors: string[];
  tickers: string[];
  searchQuery: string;
}

// API Response Types
export interface NewsAPIResponse {
  success: boolean;
  data: NewsItem[];
  totalCount: number;
  hasMore: boolean;
  lastUpdated: string;
  error?: string;
}

export interface SentimentAPIResponse {
  success: boolean;
  data: {
    trends: SentimentTrend[];
    topMovers: TopMover[];
    sectorSentiments: SectorSentiment[];
    marketOverview: {
      overallSentiment: number;
      confidence: number;
      regime: string;
    };
  };
  error?: string;
}

// Cache Types
export interface CacheMetadata {
  lastEODHDUpdate: Date | string;
  lastPerplexityUpdate: Date | string;
  nextScheduledUpdate: Date | string;
  eodhdHealth: 'healthy' | 'degraded' | 'down';
  perplexityQuotaUsed: number;
  perplexityQuotaRemaining: number;
}

// UI Component Props
export interface NewsCardProps {
  article: NewsItem;
  onTickerClick?: (ticker: string) => void;
  onSectorClick?: (sector: string) => void;
  showFullContent?: boolean;
}

export interface NewsFiltersProps {
  filters: NewsFilters;
  onFiltersChange: (filters: Partial<NewsFilters>) => void;
  availableSectors: string[];
  availableTickers: string[];
  isLoading?: boolean;
}

export interface SentimentVisualizationProps {
  trends: SentimentTrend[];
  topMovers: TopMover[];
  sectorSentiments: SectorSentiment[];
  selectedTicker?: string;
  selectedSector?: string;
  timeframe: '1h' | '4h' | '1d' | '1w';
}

// Constants
export const NSE_SECTORS = [
  'Banking',
  'IT',
  'Pharma',
  'Auto',
  'FMCG',
  'Energy',
  'Infrastructure',
  'Metals',
  'Media',
  'Telecom',
  'Textiles',
  'Chemicals'
] as const;

export const NSE_LARGE_MID_CAP_TICKERS = [
  'RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE', 'ICICIBANK.NSE', 'BHARTIARTL.NSE',
  'INFY.NSE', 'SBIN.NSE', 'LT.NSE', 'ITC.NSE', 'KOTAKBANK.NSE',
  'HINDUNILVR.NSE', 'ASIANPAINT.NSE', 'MARUTI.NSE', 'AXISBANK.NSE', 'WIPRO.NSE',
  'ONGC.NSE', 'NTPC.NSE', 'POWERGRID.NSE', 'TITAN.NSE', 'NESTLEIND.NSE',
  // Add remaining 80 tickers...
] as const;

export type NSESector = typeof NSE_SECTORS[number];
export type NSETicker = typeof NSE_LARGE_MID_CAP_TICKERS[number]; 