"use client"

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { TrendingUp, TrendingDown, Minus, RefreshCw, Filter, AlertTriangle, Activity, BarChart3, Clock, Zap, Brain, Target } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Signal {
  symbol: string
  company_name?: string
  signal: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  current_price: number
  price_target: number
  stop_loss: number
  model: string
  v5_score?: number
  v4_score?: number
  intraday_sentiment?: number
  sentiment_score?: number
  sentiment_category?: string
  sentiment_momentum?: number
  market_regime?: string
  risk_score: number
  indicators?: {
    risk_score?: number
    volatility?: number
    technical_indicators?: {
      rsi?: number
      macd?: number
      sma_20?: number
      sma_50?: number
      volume_ratio?: number
    }
  }
  technical_indicators?: {
    rsi?: number
    macd?: number
    sma_20?: number
    sma_50?: number
    volume_ratio?: number
  }
  timestamp: string
  data_sources: string[]
  key_drivers?: string[]
  csv_sentiment_data?: {
    sentiment_score: number
    sentiment_category: string
    confidence_score: number
    news_volume: number
    social_media_mentions: number
    price_change_percent: number
    volume_change_percent: number
    market_volatility_index: number
    sector_performance: number
    primary_market_factor: string
  }
}

interface MarketStats {
  total_signals: number
  buy_signals: number
  sell_signals: number
  hold_signals: number
  avg_confidence: number
  market_sentiment: number
  active_models: string[]
}

export default function SignalsPage() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [filteredSignals, setFilteredSignals] = useState<Signal[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [signalFilter, setSignalFilter] = useState<'all' | 'buy' | 'sell' | 'hold'>('all')
  const [confidenceFilter, setConfidenceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [sortBy, setSortBy] = useState<'confidence' | 'momentum' | 'risk'>('confidence')
  const [marketStats, setMarketStats] = useState<MarketStats | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchSignals = useCallback(async () => {
    try {
      setRefreshing(true)
      // Determine backend API base URL (default to localhost:8002 for dev)
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8002'

      // Use absolute URL to avoid Next.js handling the request itself
      const response = await fetch(`${API_BASE_URL}/api/enhanced-signals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: [],
          max_symbols: 117
        })
      })

      if (!response.ok) throw new Error('Failed to fetch signals')
      
      const data = await response.json()
      
      if (data.success && data.data) {
        const signalsData = data.data.signals || []
        setSignals(signalsData)
        
        // Create market stats from the signals
        const buySignals = signalsData.filter((s: Signal) => s.signal === 'BUY').length
        const sellSignals = signalsData.filter((s: Signal) => s.signal === 'SELL').length
        const holdSignals = signalsData.filter((s: Signal) => s.signal === 'HOLD').length
        const avgConfidence = signalsData.length > 0 ? signalsData.reduce((sum: number, s: Signal) => sum + s.confidence, 0) / signalsData.length : 0
        
        // Use backend market_sentiment if available, otherwise calculate from signals
        const marketSentiment = data.data.market_sentiment !== undefined 
          ? data.data.market_sentiment 
          : signalsData.length > 0 
            ? signalsData.reduce((sum: number, s: Signal) => sum + (s.intraday_sentiment || 0), 0) / signalsData.length 
            : 0
        
        setMarketStats({
          total_signals: signalsData.length,
          buy_signals: buySignals,
          sell_signals: sellSignals,
          hold_signals: holdSignals,
          avg_confidence: avgConfidence,
          market_sentiment: marketSentiment,
          active_models: [...new Set(signalsData.map((s: Signal) => s.model))] as string[]
        })
      } else {
        setSignals([])
        setMarketStats(null)
      }
      
      setLastUpdate(new Date())
    } catch (error) {
      console.error('Error fetching signals:', error)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchSignals()
    
    const interval = autoRefresh ? setInterval(fetchSignals, 30000) : null
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [fetchSignals, autoRefresh])

  useEffect(() => {
    let filtered = [...signals]
    // Patch: Ensure risk_score is properly extracted from indicators
    filtered = filtered.map(s => ({
      ...s,
      risk_score: getRiskScore(s)
    }))
    if (searchTerm) {
      filtered = filtered.filter(s => 
        s.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.company_name?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }
    if (signalFilter !== 'all') {
      filtered = filtered.filter(s => s.signal.toLowerCase() === signalFilter)
    }
    if (confidenceFilter !== 'all') {
      filtered = filtered.filter(s => {
        if (confidenceFilter === 'high') return s.confidence >= 0.75
        if (confidenceFilter === 'medium') return s.confidence >= 0.6 && s.confidence < 0.75
        return s.confidence < 0.6
      })
    }
    filtered.sort((a, b) => {
      if (sortBy === 'confidence') return b.confidence - a.confidence
      if (sortBy === 'momentum') return (b.sentiment_momentum || 0) - (a.sentiment_momentum || 0)
      return getRiskScore(a) - getRiskScore(b)
    })
    setFilteredSignals(filtered)
  }, [signals, searchTerm, signalFilter, confidenceFilter, sortBy])

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="h-5 w-5" />
      case 'SELL': return <TrendingDown className="h-5 w-5" />
      default: return <Minus className="h-5 w-5" />
    }
  }

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-600 bg-green-50'
      case 'SELL': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getRiskBadgeColor = (risk: number) => {
    if (risk < 0.3) return 'bg-green-100 text-green-800'
    if (risk < 0.6) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  // Helper function to extract risk score from signal
  const getRiskScore = (signal: Signal): number => {
    // Try to get risk score from indicators first, then fallback to direct risk_score
    if (signal.indicators?.risk_score !== undefined) {
      return signal.indicators.risk_score
    }
    if (signal.risk_score !== undefined) {
      return signal.risk_score
    }
    return 0
  }

  // Helper function to format V5 score properly
  const formatV5Score = (score: number): string => {
    // Convert decimal to percentage and ensure it's reasonable
    const percentage = score * 100
    if (percentage > 100) return '100.0%'
    if (percentage < -100) return '-100.0%'
    return `${percentage.toFixed(1)}%`
  }

  // Helper function to get technical indicators
  const getTechnicalIndicators = (signal: Signal) => {
    return signal.indicators?.technical_indicators || signal.technical_indicators
  }

  // Helper function to enhance key drivers
  const getEnhancedKeyDrivers = (signal: Signal): string[] => {
    const drivers = signal.key_drivers || []
    const enhanced = [...drivers]
    
    // Add technical indicators as drivers if they're significant
    const tech = getTechnicalIndicators(signal)
    if (tech?.rsi) {
      if (tech.rsi > 70) enhanced.push('RSI Overbought')
      else if (tech.rsi < 30) enhanced.push('RSI Oversold')
    }
    if (tech?.volume_ratio && tech.volume_ratio > 2) {
      enhanced.push('High Volume')
    }
    
    // Add sentiment-based drivers
    if (signal.intraday_sentiment && Math.abs(signal.intraday_sentiment) > 0.1) {
      if (signal.intraday_sentiment > 0.1) enhanced.push('Positive Sentiment')
      else enhanced.push('Negative Sentiment')
    }
    
    return enhanced.length > 0 ? enhanced : ['Technical Analysis']
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading live signals...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Zap className="h-8 w-8 text-yellow-500" />
            Live Trading Signals
          </h1>
          <p className="text-muted-foreground mt-1">
            AI-powered signals from v5 & v4 models with real-time sentiment analysis
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant="outline" className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {lastUpdate ? `Updated ${formatTime(lastUpdate.toISOString())}` : 'Loading...'}
          </Badge>
          <Button
            onClick={fetchSignals}
            disabled={refreshing}
            size="sm"
            className="gap-2"
          >
            <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} />
            Refresh
          </Button>
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
          >
            Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
          </Button>
        </div>
      </div>

      {marketStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Total Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{marketStats.total_signals}</div>
              <div className="flex items-center gap-2 mt-2">
                <Badge variant="secondary" className="text-xs">
                  {marketStats.active_models.length} Models Active
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Signal Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-green-500 rounded-full" />
                  <span className="text-sm">{marketStats.buy_signals}</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-red-500 rounded-full" />
                  <span className="text-sm">{marketStats.sell_signals}</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-gray-500 rounded-full" />
                  <span className="text-sm">{marketStats.hold_signals}</span>
                </div>
              </div>
              <Progress 
                value={(marketStats.buy_signals / marketStats.total_signals) * 100} 
                className="mt-2 h-2"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Average Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(marketStats.avg_confidence * 100).toFixed(1)}%</div>
              <Progress value={marketStats.avg_confidence * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Market Sentiment</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-500" />
                <span className="text-2xl font-bold">
                  {marketStats.market_sentiment > 0 ? '+' : ''}{(marketStats.market_sentiment * 100).toFixed(1)}%
                </span>
              </div>
              <Badge 
                variant="outline"
                className={cn(
                  "mt-2",
                  marketStats.market_sentiment > 0.2 ? "text-green-600" :
                  marketStats.market_sentiment < -0.2 ? "text-red-600" : "text-gray-600"
                )}
              >
                {marketStats.market_sentiment > 0.2 ? 'Bullish' :
                 marketStats.market_sentiment < -0.2 ? 'Bearish' : 'Neutral'}
              </Badge>
            </CardContent>
          </Card>
        </div>
      )}

      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder="Search symbols..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            
            <Select value={signalFilter} onValueChange={(v: 'all' | 'buy' | 'sell' | 'hold') => setSignalFilter(v)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Signal type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Signals</SelectItem>
                <SelectItem value="buy">Buy Only</SelectItem>
                <SelectItem value="sell">Sell Only</SelectItem>
                <SelectItem value="hold">Hold Only</SelectItem>
              </SelectContent>
            </Select>

            <Select value={confidenceFilter} onValueChange={(v: 'all' | 'high' | 'medium' | 'low') => setConfidenceFilter(v)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Confidence" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Levels</SelectItem>
                <SelectItem value="high">High (≥75%)</SelectItem>
                <SelectItem value="medium">Medium (60-75%)</SelectItem>
                <SelectItem value="low">Low (&lt;60%)</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={(v: 'confidence' | 'momentum' | 'risk') => setSortBy(v)}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="confidence">Confidence</SelectItem>
                <SelectItem value="momentum">Momentum</SelectItem>
                <SelectItem value="risk">Risk Score</SelectItem>
              </SelectContent>
            </Select>

            <Badge variant="outline" className="gap-1">
              <Filter className="h-3 w-3" />
              {filteredSignals.length} results
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredSignals.map((signal) => (
          <Card key={signal.symbol} className="overflow-hidden hover:shadow-lg transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">{signal.symbol}</CardTitle>
                  {signal.company_name && (
                    <CardDescription className="text-xs mt-1">
                      {signal.company_name}
                    </CardDescription>
                  )}
                </div>
                <Badge className={cn("gap-1", getSignalColor(signal.signal))}>
                  {getSignalIcon(signal.signal)}
                  {signal.signal}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div>
                  <p className="text-muted-foreground">Current</p>
                  <p className="font-semibold">₹{signal.current_price.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Target</p>
                  <p className="font-semibold text-green-600">₹{signal.price_target.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Stop Loss</p>
                  <p className="font-semibold text-red-600">₹{signal.stop_loss.toFixed(2)}</p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Confidence</span>
                  <div className="flex items-center gap-2">
                    <Progress value={signal.confidence * 100} className="w-24 h-2" />
                    <span className="text-sm font-semibold">{Math.max(signal.confidence * 100, 0.1).toFixed(1)}%</span>
                  </div>
                </div>

                {signal.v5_score !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">v5 Model</span>
                    <Badge variant="outline" className="text-xs">
                      <Brain className="h-3 w-3 mr-1" />
                      {formatV5Score(signal.v5_score)}
                    </Badge>
                  </div>
                )}

                {signal.v4_score !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">v4 Model</span>
                    <Badge variant="outline" className="text-xs">
                      {(signal.v4_score * 100).toFixed(1)}%
                    </Badge>
                  </div>
                )}

                {signal.intraday_sentiment !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">EODHD Sentiment</span>
                    <Badge 
                      variant="outline" 
                      className={cn(
                        "text-xs",
                        signal.intraday_sentiment > 0.2 ? "text-green-600" :
                        signal.intraday_sentiment < -0.2 ? "text-red-600" : "text-gray-600"
                      )}
                    >
                      {(signal.intraday_sentiment * 100).toFixed(1)}%
                    </Badge>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                {signal.sentiment_category && (
                  <Badge 
                    variant="secondary"
                    className={cn(
                      "text-xs",
                      signal.sentiment_category === 'BULLISH' ? "bg-green-100 text-green-800" :
                      signal.sentiment_category === 'BEARISH' ? "bg-red-100 text-red-800" :
                      "bg-gray-100 text-gray-800"
                    )}
                  >
                    {signal.sentiment_category}
                  </Badge>
                )}

                {signal.sentiment_momentum !== undefined && (
                  <Badge variant="outline" className="text-xs">
                    Momentum: {signal.sentiment_momentum > 0 ? '+' : ''}{(signal.sentiment_momentum * 100).toFixed(1)}%
                  </Badge>
                )}

                {signal.market_regime && (
                  <Badge variant="outline" className="text-xs">
                    {signal.market_regime}
                  </Badge>
                )}

                <Badge 
                  variant="secondary"
                  className={cn("text-xs", getRiskBadgeColor(getRiskScore(signal)))}
                >
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Risk: {(getRiskScore(signal) * 100).toFixed(0)}%
                </Badge>
              </div>

              {getTechnicalIndicators(signal) && (
                <div className="pt-2 border-t">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {getTechnicalIndicators(signal)?.rsi !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">RSI</span>
                        <span className={cn(
                          "font-medium",
                          getTechnicalIndicators(signal)!.rsi! > 70 ? "text-red-600" :
                          getTechnicalIndicators(signal)!.rsi! < 30 ? "text-green-600" : ""
                        )}>
                          {getTechnicalIndicators(signal)!.rsi!.toFixed(1)}
                        </span>
                      </div>
                    )}
                    {getTechnicalIndicators(signal)?.volume_ratio !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Vol Ratio</span>
                        <span className="font-medium">
                          {getTechnicalIndicators(signal)!.volume_ratio!.toFixed(2)}x
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* CSV Sentiment Data */}
              {signal.csv_sentiment_data && (
                <div className="pt-2 border-t">
                  <p className="text-xs text-muted-foreground mb-1">CSV Sentiment Data:</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">News Volume</span>
                      <span className="font-medium">
                        {signal.csv_sentiment_data.news_volume}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Social Mentions</span>
                      <span className="font-medium">
                        {signal.csv_sentiment_data.social_media_mentions}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Price Change</span>
                      <span className={cn(
                        "font-medium",
                        signal.csv_sentiment_data.price_change_percent > 0 ? "text-green-600" : "text-red-600"
                      )}>
                        {signal.csv_sentiment_data.price_change_percent.toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Market Factor</span>
                      <span className="font-medium text-xs">
                        {signal.csv_sentiment_data.primary_market_factor}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {signal.key_drivers && signal.key_drivers.length > 0 && (
                <div className="pt-2 border-t">
                  <p className="text-xs text-muted-foreground mb-1">Key Drivers:</p>
                  <div className="flex flex-wrap gap-1">
                    {getEnhancedKeyDrivers(signal).map((driver, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {driver}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-2 border-t text-xs text-muted-foreground">
                <span>{formatTime(signal.timestamp)}</span>
                <div className="flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  {signal.model}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredSignals.length === 0 && (
        <Card className="mt-8">
          <CardContent className="text-center py-12">
            <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No signals found</h3>
            <p className="text-muted-foreground">
              Try adjusting your filters or refresh to get the latest signals
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}