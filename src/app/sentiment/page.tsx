'use client'

import { useState, useEffect, useMemo, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Slider } from '@/components/ui/slider'
import { AppLayout } from '@/components/AppLayout'
import { SentimentTreemap } from '@/components/SentimentTreemap'
import { SectorHeatmap } from '@/components/SectorHeatmap'
import { 
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Filter,
  Search,
  Gauge,
  Target,
  Zap,
  Minus,
  RefreshCw,
  AlertCircle,
  Database
} from 'lucide-react'

interface SentimentData {
  Date: string
  Day_of_Week?: string
  Month?: string
  Quarter?: string
  Symbol: string
  Company_Name: string
  Sector: string
  Market_Cap_Category: string
  Sentiment_Score: number
  Sentiment_Category: string
  Confidence_Score: number
  Primary_Market_Factor: string
  News_Volume: number
  Social_Media_Mentions: number
  Analyst_Coverage: number
  Price_Change_Percent: number
  Volume_Change_Percent: number
  Market_Volatility_Index: number
  Sector_Performance: number
}

interface StockDetailProps {
  stock: SentimentData;
  onBack: () => void;
}

function StockDetailView({ stock, onBack }: StockDetailProps) {
  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.3) return 'text-green-400';
    if (sentiment < -0.3) return 'text-red-400';
    return 'text-yellow-400';
  };

  return (
    <Card className="bg-black/50 border-gray-800 h-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>{stock.Company_Name} ({stock.Symbol})</CardTitle>
          <Button onClick={onBack} variant="ghost" size="sm">Back to list</Button>
        </div>
        <div className="text-sm text-gray-400">{stock.Sector} • {stock.Market_Cap_Category}</div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold mb-2">Sentiment Snapshot</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-gray-900/50 rounded-lg">
              <div className="text-xs text-gray-400">Sentiment Score</div>
              <div className={`text-2xl font-bold ${getSentimentColor(stock.Sentiment_Score)}`}>
                {(stock.Sentiment_Score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-3 bg-gray-900/50 rounded-lg">
              <div className="text-xs text-gray-400">Confidence</div>
              <div className="text-2xl font-bold text-blue-400">
                {stock.Confidence_Score.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
        <div>
          <h3 className="text-lg font-semibold mb-2">Market Data</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span>Price Change:</span> <span className={stock.Price_Change_Percent >= 0 ? 'text-green-400' : 'text-red-400'}>{stock.Price_Change_Percent.toFixed(2)}%</span></div>
            <div className="flex justify-between"><span>Volume Change:</span> <span>{stock.Volume_Change_Percent.toFixed(2)}%</span></div>
            <div className="flex justify-between"><span>News Volume:</span> <span>{stock.News_Volume}</span></div>
            <div className="flex justify-between"><span>Social Mentions:</span> <span>{stock.Social_Media_Mentions}</span></div>
          </div>
        </div>
        <div>
          <h3 className="text-lg font-semibold mb-2">Context</h3>
          <div className="space-y-2 text-sm">
             <div className="flex justify-between"><span>Primary Market Factor:</span> <span>{stock.Primary_Market_Factor}</span></div>
             <div className="flex justify-between"><span>Market Volatility Index:</span> <span>{stock.Market_Volatility_Index.toFixed(2)}</span></div>
             <div className="flex justify-between"><span>Sector Performance:</span> <span>{stock.Sector_Performance.toFixed(2)}%</span></div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface MarketSentimentGauge {
  overall: number
  confidence: number
  regime: 'Bullish' | 'Bearish' | 'Neutral'
  totalStocks: number
  positiveCount: number
  negativeCount: number
  neutralCount: number
  marketCap: string
}

interface SectorMetrics {
  sector: string
  avgSentiment: number
  stockCount: number
  topMovers: string[]
  avgConfidence: number
  totalNewsVolume: number
  avgPriceChange: number
}

interface TopMover {
  symbol: string
  company: string
  sentiment: number
  confidence: number
  sector: string
  newsVolume: number
  priceChange: number
  marketCap: string
}

interface DataStats {
  totalRecords: number
  uniqueStocks: number
  dateRange: { start: string; end: string } | null
}

// Helper function to aggregate sentiment data
/*
const aggregateSentimentData = (data: SentimentData[]): SentimentData[] => {
  if (data.length === 0) return []

  // Group by symbol and get latest data for each stock
  const latestBySymbol = data.reduce((acc: Record<string, SentimentData>, curr) => {
    const existing = acc[curr.Symbol]
    if (!existing || new Date(curr.Date) > new Date(existing.Date)) {
      acc[curr.Symbol] = curr
    }
    return acc
  }, {})

  // Get all unique stocks and their market caps
  const stocksWithData = Object.values(latestBySymbol)
  const allStocks = stocksWithData.map(stock => ({
    ...stock,
    Symbol: stock.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE',
    Sentiment_Category: getSentimentCategory(stock.Sentiment_Score)
  }))

  // Sort by market cap and take top 117
  const sortedStocks = allStocks.sort((a, b) => {
    const capOrder = { 'Large Cap': 3, 'Mid Cap': 2, 'Small Cap': 1 }
    return (capOrder[b.Market_Cap_Category as keyof typeof capOrder] || 0) - 
           (capOrder[a.Market_Cap_Category as keyof typeof capOrder] || 0)
  }).slice(0, 117)

  return sortedStocks
}
*/

const getSentimentColor = (sentiment: number): string => {
  if (sentiment > 0.5) return 'rgb(34, 197, 94)' // Strong positive - green
  if (sentiment > 0.2) return 'rgb(132, 204, 22)' // Positive - light green
  if (sentiment > -0.2) return 'rgb(234, 179, 8)' // Neutral - yellow
  if (sentiment > -0.5) return 'rgb(234, 88, 12)' // Negative - orange
  return 'rgb(239, 68, 68)' // Strong negative - red
}

/*
const getSentimentCategory = (score: number): string => {
  if (score > 0.5) return 'Very Bullish'
  if (score > 0.2) return 'Bullish'
  if (score > -0.2) return 'Neutral'
  if (score > -0.5) return 'Bearish'
  return 'Very Bearish'
}
*/

const DURATION_BUTTONS = [
  { label: 'Intraday', value: 'today'},
  { label: 'Week', value: 'week' },
  { label: 'Month', value: 'month' },
  { label: '1 Year', value: '1-year' },
  { label: '2 Years', value: '2-years' },
  { label: '5 Years', value: 'all' }
] as const

type DurationButton = typeof DURATION_BUTTONS[number]

export default function SentimentPage() {
  // Data states
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rawData, setRawData] = useState<SentimentData[]>([])
  const [selectedTimeRange, setSelectedTimeRange] = useState('2-years')
  const [focusedElement, setFocusedElement] = useState<string | null>('header')
  const [animateCards, setAnimateCards] = useState(false)
  const [dataStats, setDataStats] = useState<DataStats>({
    totalRecords: 0,
    uniqueStocks: 117,
    dateRange: null
  })
  const [selectedStock, setSelectedStock] = useState<SentimentData | null>(null)
  const [filteredData, setFilteredData] = useState<SentimentData[]>([])
  const [selectedSectors, setSelectedSectors] = useState<string[]>([])
  const [selectedMarketCaps, setSelectedMarketCaps] = useState<string[]>([])

  // Filter states
  const [sentimentRange, setSentimentRange] = useState([-1, 1])
  const [confidenceRange, setConfidenceRange] = useState([0, 100])
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState('sentiment_desc')
  const [selectedFactors, setSelectedFactors] = useState<string[]>([])
  const [minNewsVolume, setMinNewsVolume] = useState(0)

  // Analytics states
  const [marketGauge, setMarketGauge] = useState<MarketSentimentGauge | null>(null)
  const [sectorMetrics, setSectorMetrics] = useState<SectorMetrics[]>([])
  const [topMovers, setTopMovers] = useState<TopMover[]>([])

  const filterStocks = useCallback((data: SentimentData[]): SentimentData[] => {
    return data.filter(stock => {
      const matchesSector = selectedSectors.length === 0 || selectedSectors.includes(stock.Sector)
      const matchesMarketCap = selectedMarketCaps.length === 0 || selectedMarketCaps.includes(stock.Market_Cap_Category)
      const matchesSentiment = stock.Sentiment_Score >= sentimentRange[0] && stock.Sentiment_Score <= sentimentRange[1]
      const matchesConfidence = stock.Confidence_Score >= confidenceRange[0] && stock.Confidence_Score <= confidenceRange[1]
      return matchesSector && matchesMarketCap && matchesSentiment && matchesConfidence
    })
  }, [selectedSectors, selectedMarketCaps, sentimentRange, confidenceRange])

  const calculateMarketStats = useCallback((data: SentimentData[]) => {
    if (!data || data.length === 0) return {
      gauge: {
        overall: 0,
        confidence: 65.7,
        regime: 'Neutral' as const,
        totalStocks: 117,
        positiveCount: 0,
        negativeCount: 0,
        neutralCount: 117,
        marketCap: 'All'
      },
      sectorMetrics: [] as SectorMetrics[],
      topMovers: [] as TopMover[]
    }

    // Calculate market sentiment gauge
    const positiveStocks = data.filter(d => d.Sentiment_Score > 0.2).length
    const negativeStocks = data.filter(d => d.Sentiment_Score < -0.2).length
    const neutralStocks = data.length - positiveStocks - negativeStocks

    const avgSentiment = data.reduce((sum, d) => sum + d.Sentiment_Score, 0) / data.length
    const avgConfidence = data.reduce((sum, d) => sum + d.Confidence_Score, 0) / data.length

    const regime = avgSentiment > 0.2 ? 'Bullish' as const : 
                  avgSentiment < -0.2 ? 'Bearish' as const : 
                  'Neutral' as const

    const gauge: MarketSentimentGauge = {
      overall: avgSentiment * 100,
      confidence: avgConfidence,
      regime,
      totalStocks: 117,
      positiveCount: positiveStocks,
      negativeCount: negativeStocks,
      neutralCount: neutralStocks,
      marketCap: 'All'
    }

    // Calculate sector metrics
    const sectorMetrics = Object.values(data.reduce((acc: Record<string, SectorMetrics>, stock) => {
      if (!acc[stock.Sector]) {
        acc[stock.Sector] = {
          sector: stock.Sector,
          avgSentiment: 0,
          stockCount: 0,
          topMovers: [],
          avgConfidence: 0,
          totalNewsVolume: 0,
          avgPriceChange: 0
        }
      }
      
      acc[stock.Sector].avgSentiment += stock.Sentiment_Score
      acc[stock.Sector].stockCount++
      acc[stock.Sector].avgConfidence += stock.Confidence_Score
      acc[stock.Sector].totalNewsVolume += stock.News_Volume
      acc[stock.Sector].avgPriceChange += stock.Price_Change_Percent
      
      return acc
    }, {})).map(sector => ({
      ...sector,
      avgSentiment: sector.avgSentiment / sector.stockCount,
      avgConfidence: sector.avgConfidence / sector.stockCount,
      avgPriceChange: sector.avgPriceChange / sector.stockCount
    }))

    // Get top movers
    const topMovers = data
      .sort((a, b) => Math.abs(b.Sentiment_Score) - Math.abs(a.Sentiment_Score))
      .filter(stock => Math.abs(stock.Sentiment_Score) > 0.2)
      .slice(0, 10)
      .map(stock => ({
        symbol: stock.Symbol,
        company: stock.Company_Name,
        sentiment: stock.Sentiment_Score,
        confidence: stock.Confidence_Score,
        sector: stock.Sector,
        newsVolume: stock.News_Volume,
        priceChange: stock.Price_Change_Percent,
        marketCap: stock.Market_Cap_Category
      }))

    return { gauge, sectorMetrics, topMovers }
  }, [])

  const loadSentimentData = useCallback(async (timeRange: string) => {
    setLoading(true)
    setError(null)
    setAnimateCards(false)
    
    try {
      const response = await fetch(`/api/sentiment/data?timeRange=${timeRange}`)
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.message || 'Failed to load sentiment data.')
      }
      const data: SentimentData[] = await response.json()
      
      // Get unique stocks and normalize symbols
      const uniqueStocks = [...new Set(data.map(item => item.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE'))]
        .sort((a, b) => {
          const aData = data.find(d => d.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE' === a)
          const bData = data.find(d => d.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE' === b)
          // Sort by market cap category (Large > Mid > Small)
          const capOrder = {
            'Large Cap': 0,
            'Mid Cap': 1,
            'Small Cap': 2
          } as const
          const aCategory = aData?.Market_Cap_Category as keyof typeof capOrder
          const bCategory = bData?.Market_Cap_Category as keyof typeof capOrder
          return (capOrder[aCategory] ?? 3) - (capOrder[bCategory] ?? 3)
        })
        .slice(0, 117) // Take exactly 117 stocks

      // Filter and normalize data for selected stocks
      const normalizedData = data
        .filter(item => uniqueStocks.includes(item.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE'))
        .map(item => ({
          ...item,
          Symbol: item.Symbol.replace(/\.(NS|NSE)$/, '') + '.NSE'
        }))

      // Group by symbol and get latest data for each stock
      const latestBySymbol = normalizedData.reduce((acc: Record<string, SentimentData>, curr) => {
        const existing = acc[curr.Symbol]
        if (!existing || new Date(curr.Date) > new Date(existing.Date)) {
          acc[curr.Symbol] = curr
        }
        return acc
      }, {})

      const finalData = Object.values(latestBySymbol)
      
      console.log(`Final data loaded for ${timeRange}:`, {
        totalRecords: finalData.length,
        uniqueStocks: finalData.length,
        dateRange: finalData.length > 0 ? {
          start: new Date(Math.min(...finalData.map(d => new Date(d.Date).getTime()))).toISOString().split('T')[0],
          end: new Date(Math.max(...finalData.map(d => new Date(d.Date).getTime()))).toISOString().split('T')[0]
        } : null
      })

      setRawData(finalData)
      setDataStats({
        totalRecords: finalData.length,
        uniqueStocks: 117,
        dateRange: finalData.length > 0 ? {
          start: new Date(Math.min(...finalData.map(d => new Date(d.Date).getTime()))).toISOString().split('T')[0],
          end: new Date(Math.max(...finalData.map(d => new Date(d.Date).getTime()))).toISOString().split('T')[0]
        } : null
      })

      setAnimateCards(true)
      setTimeout(() => setAnimateCards(false), 1000)
    } catch (err) {
      console.error('Error loading sentiment data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load sentiment data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadSentimentData(selectedTimeRange)
  }, [selectedTimeRange, loadSentimentData])

  // All data processing is now primarily handled by the `loadSentimentData` function.
  // The `periodData` memo is simplified to just pass through the `rawData`
  // which is already correctly fetched, filtered, and aggregated for the selected time range.
  const periodData = useMemo(() => {
    return rawData
  }, [rawData]);

  // Step 2: Apply user-driven filters (sliders, search, etc.) on top of the period data
  const computedFilteredData = useMemo(() => {
    let filtered = [...periodData]

    // Apply all other filters
    if (selectedSectors.length > 0) {
      filtered = filtered.filter(item => selectedSectors.includes(item.Sector))
    }

    if (selectedMarketCaps.length > 0) {
      filtered = filtered.filter(item => selectedMarketCaps.includes(item.Market_Cap_Category))
    }

    filtered = filtered.filter(item => 
      item.Sentiment_Score >= sentimentRange[0] && 
      item.Sentiment_Score <= sentimentRange[1]
    )

    filtered = filtered.filter(item => 
      item.Confidence_Score >= confidenceRange[0] && 
      item.Confidence_Score <= confidenceRange[1]
    )

    if (selectedFactors.length > 0) {
      filtered = filtered.filter(item => selectedFactors.includes(item.Primary_Market_Factor))
    }

    if (minNewsVolume > 0) {
      filtered = filtered.filter(item => item.News_Volume >= minNewsVolume)
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(item => 
        item.Company_Name.toLowerCase().includes(query) ||
        item.Symbol.toLowerCase().includes(query) ||
        item.Sector.toLowerCase().includes(query)
      )
    }

    // Sort data
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'sentiment_desc':
          return b.Sentiment_Score - a.Sentiment_Score
        case 'sentiment_asc':
          return a.Sentiment_Score - b.Sentiment_Score
        case 'confidence_desc':
          return b.Confidence_Score - a.Confidence_Score
        case 'news_volume_desc':
          return b.News_Volume - a.News_Volume
        case 'price_change_desc':
          return b.Price_Change_Percent - a.Price_Change_Percent
        case 'date_desc':
          return new Date(b.Date).getTime() - new Date(a.Date).getTime()
        case 'alphabetical':
          return a.Company_Name.localeCompare(b.Company_Name)
        default:
          return 0
      }
    })

    return filtered
  }, [periodData, selectedSectors, selectedMarketCaps, sentimentRange, confidenceRange, selectedFactors, minNewsVolume, searchQuery, sortBy])

  // Update filtered data
  useEffect(() => {
    setFilteredData(computedFilteredData)
  }, [computedFilteredData])

  // Analytics computation for the OVERVIEW panel (based on period data, before user filters)
  useEffect(() => {
    const dataForGauge = periodData;
    if (dataForGauge.length === 0) {
      setMarketGauge(null)
      return
    }

    // Market gauge
    const sentiments = dataForGauge.map(d => d.Sentiment_Score)
    const confidences = dataForGauge.map(d => d.Confidence_Score)
    
    const avgSentiment = sentiments.length > 0 ? sentiments.reduce((s, v) => s + v, 0) / sentiments.length : 0
    const avgConfidence = confidences.length > 0 ? confidences.reduce((s, v) => s + v, 0) / confidences.length : 0
    
    const positiveCount = dataForGauge.filter(d => d.Sentiment_Score > 0.3).length
    const negativeCount = dataForGauge.filter(d => d.Sentiment_Score < -0.3).length
    const neutralCount = dataForGauge.length - positiveCount - negativeCount
    
    const regime = avgSentiment > 0.3 ? 'Bullish' : avgSentiment < -0.3 ? 'Bearish' : 'Neutral'

    const marketCaps = [...new Set(dataForGauge.map(d => d.Market_Cap_Category))];
    const marketCapDisplay = marketCaps.length > 1 ? 'Multiple' : marketCaps[0] || 'N/A';

    const newMarketGauge: MarketSentimentGauge = {
      overall: avgSentiment,
      confidence: avgConfidence,
      regime,
      totalStocks: dataForGauge.length,
      positiveCount,
      negativeCount,
      neutralCount,
      marketCap: marketCapDisplay
    }
    setMarketGauge(newMarketGauge)
  }, [periodData])

  // Analytics for TABS (based on the final filteredData)
  const tabsAnalytics = useMemo(() => {
    if (filteredData.length === 0) return null

    // For aggregated data, we don't need to filter by the latest date again.
    // The data is already aggregated for the selected period.
    const latestData = filteredData

    console.log(`Analytics computed from ${latestData.length} stocks`)

    // Sector metrics
    const sectorGroups: { [key: string]: SentimentData[] } = {}
    latestData.forEach(item => {
      if (!sectorGroups[item.Sector]) sectorGroups[item.Sector] = []
      sectorGroups[item.Sector].push(item)
    })

    const sectorMetrics: SectorMetrics[] = Object.entries(sectorGroups)
      .map(([sector, items]) => {
        const avgSentiment = items.reduce((s, i) => s + i.Sentiment_Score, 0) / items.length
        const avgConfidence = items.reduce((s, i) => s + i.Confidence_Score, 0) / items.length
        const totalNewsVolume = items.reduce((s, i) => s + i.News_Volume, 0)
        const avgPriceChange = items.reduce((s, i) => s + i.Price_Change_Percent, 0) / items.length

        const topMovers = items
          .sort((a, b) => Math.abs(b.Sentiment_Score) - Math.abs(a.Sentiment_Score))
          .slice(0, 3)
          .map(i => i.Symbol)

        return {
          sector,
          avgSentiment,
          stockCount: items.length,
          topMovers,
          avgConfidence,
          totalNewsVolume,
          avgPriceChange
        }
      })
      .sort((a, b) => b.avgSentiment - a.avgSentiment)

    // Top movers
    const topMovers: TopMover[] = latestData
      .map(item => ({
        symbol: item.Symbol,
        company: item.Company_Name,
        sentiment: item.Sentiment_Score,
        confidence: item.Confidence_Score,
        sector: item.Sector,
        newsVolume: item.News_Volume,
        priceChange: item.Price_Change_Percent,
        marketCap: item.Market_Cap_Category
      }))
      .sort((a, b) => Math.abs(b.sentiment) - Math.abs(a.sentiment))
      .slice(0, 20)

    return {
      sectorMetrics,
      topMovers
    }
  }, [filteredData])

  // Update analytics states for tabs
  useEffect(() => {
    if (tabsAnalytics) {
      setSectorMetrics(tabsAnalytics.sectorMetrics)
      setTopMovers(tabsAnalytics.topMovers)
    } else {
      setSectorMetrics([])
      setTopMovers([])
    }
  }, [tabsAnalytics])

  const handleStockClick = (stock: SentimentData) => {
    // The `periodData` contains the correctly aggregated (or latest) data for each stock.
    // We find the matching stock here to ensure the detail view is consistent.
    const stockData = periodData.find(d => d.Symbol === stock.Symbol);
    setSelectedStock(stockData || stock);
  };

  // Helper functions
  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.1) return <TrendingUp className="w-4 h-4" />
    if (sentiment < -0.1) return <TrendingDown className="w-4 h-4" />
    return <Minus className="w-4 h-4" />
  }

  const getAvailableOptions = useMemo(() => {
    const sectors = [...new Set(rawData.map(item => item.Sector))].sort()
    const marketCaps = [...new Set(rawData.map(item => item.Market_Cap_Category))].sort()
    const factors = [...new Set(rawData.map(item => item.Primary_Market_Factor))].sort()
    return { sectors, marketCaps, factors }
  }, [rawData])

  const clearAllFilters = () => {
    setSelectedSectors([])
    setSelectedMarketCaps([])
    setSentimentRange([-1, 1])
    setConfidenceRange([0, 100])
    setSelectedFactors([])
    setMinNewsVolume(0)
    setSearchQuery('')
    setSortBy('sentiment_desc')
  }

  const exportData = () => {
    const csvContent = [
      // Header
      ['Date', 'Symbol', 'Company_Name', 'Sector', 'Market_Cap_Category', 'Sentiment_Score', 'Sentiment_Category', 'Confidence_Score', 'News_Volume', 'Price_Change_Percent'].join(','),
      // Data rows
      ...filteredData.map(item => [
        item.Date,
        item.Symbol,
        `"${item.Company_Name}"`,
        item.Sector,
        item.Market_Cap_Category,
        (item.Sentiment_Score * 100).toFixed(2),
        item.Sentiment_Category,
        item.Confidence_Score.toFixed(2),
        item.News_Volume,
        item.Price_Change_Percent.toFixed(2)
      ].join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `sentiment_data_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  useEffect(() => {
    if (!rawData || rawData.length === 0) return

    const filteredStocks = filterStocks(rawData)
    setFilteredData(filteredStocks)

    // Calculate market stats
    const stats = calculateMarketStats(filteredStocks)
    setMarketGauge(stats.gauge)
    setSectorMetrics(stats.sectorMetrics)
    setTopMovers(stats.topMovers)

  }, [rawData, filterStocks, calculateMarketStats])

  if (loading) {
    return (
      <AppLayout>
        <div className="min-h-screen bg-black/95 text-white flex items-center justify-center relative overflow-hidden">
          {/* Static background */}
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/3 via-blue-500/3 to-purple-500/3 opacity-50"></div>
          
          {/* Loading content */}
          <div className="relative z-10 text-center space-y-8">
            {/* Main loading spinner */}
            <div className="relative">
              <div className="w-24 h-24 border-4 border-emerald-500/20 rounded-full animate-spin">
                <div className="absolute top-0 left-0 w-24 h-24 border-t-4 border-emerald-400 rounded-full animate-spin"></div>
              </div>
              <div className="absolute inset-0 flex items-center justify-center">
                <Database className="w-8 h-8 text-emerald-400 animate-pulse" />
              </div>
            </div>
            
            {/* Loading text */}
            <div className="space-y-3">
                             <h2 className="text-2xl font-black bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent tracking-wide">
                 Loading Market Sentiment Data
               </h2>
              <p className="text-gray-400 max-w-md mx-auto">
                Analyzing real-time market sentiment across multiple timeframes and sectors...
              </p>
            </div>
            
            {/* Loading progress indicators */}
            <div className="space-y-3">
              <div className="flex justify-center space-x-2">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 bg-emerald-400 rounded-full animate-bounce`}
                    style={{ animationDelay: `${i * 0.2}s` }}
                  ></div>
                ))}
              </div>
              <div className="text-sm text-gray-500">
                Fetching data for {selectedTimeRange}...
              </div>
            </div>
          </div>
        </div>
      </AppLayout>
    )
  }

  if (error) {
    return (
      <AppLayout>
        <div className="p-6">
          <div className="text-center py-12">
            <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <div className="text-red-500 text-lg mb-4">Error: {error}</div>
            <Button onClick={() => loadSentimentData(selectedTimeRange)}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout>
      <div className={`
        min-h-screen bg-black/95 text-white transition-all duration-300
        ${searchQuery ? 'pb-4' : 'pb-8'}
      `}>
        {/* Enhanced Header with Dynamic Blur and Animations */}
        <div className={`
          border-b border-gray-800/50 bg-black/30 backdrop-blur-2xl 
          transition-all duration-700 ease-out relative overflow-hidden
          ${focusedElement === 'header' ? 'blur-sm scale-95 opacity-70' : 'blur-0 scale-100 opacity-100'}
          ${animateCards ? 'animate-pulse' : ''}
        `}>
          {/* Static background gradient - no animation */}
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/3 via-transparent to-blue-500/2 opacity-30"></div>
          
          <div className={`
            relative transition-all duration-500
            ${focusedElement === 'header' ? 'scale-105 p-8' : 'p-6'}
            ${searchQuery ? 'p-4' : ''}
          `}>
            <div className="flex items-center justify-between">
              <div className={`
                transition-all duration-500 ease-out
                ${animateCards ? 'translate-y-2 opacity-80' : 'translate-y-0 opacity-100'}
              `}>
                <h1 className={`
                  font-black mb-3 bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 
                  bg-clip-text text-transparent transition-all duration-500 tracking-tight
                  ${focusedElement === 'header' ? 'text-5xl' : 'text-4xl'}
                  hover:from-emerald-300 hover:via-teal-300 hover:to-cyan-300
                  font-mono
                `} style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
                  Advanced Sentiment Analysis
                </h1>
                <p className={`
                  text-gray-300 transition-all duration-500
                  ${focusedElement === 'header' ? 'text-lg' : 'text-base'}
                `}>
                  Analyzing <span className="text-emerald-400 font-bold">117</span> stocks 
                  across <span className="text-blue-400 font-bold">{getAvailableOptions.sectors.length}</span> sectors
                  {dataStats.dateRange && (
                    <span className="ml-2 text-gray-400">
                      • Data from {dataStats.dateRange.start} to {dataStats.dateRange.end}
                    </span>
                  )}
                </p>
                <div className={`
                  flex items-center mt-4 space-x-4 text-sm transition-all duration-500
                  ${focusedElement === 'header' ? 'space-x-6' : 'space-x-4'}
                `}>
                  <Badge variant="outline" className={`
                    bg-blue-500/10 text-blue-400 border-blue-500/30 backdrop-blur-sm
                    transition-all duration-300 hover:scale-110 hover:bg-blue-500/20
                    ${animateCards ? 'animate-bounce' : ''}
                  `}>
                    <Database className="w-3 h-3 mr-1" />
                    {dataStats.totalRecords.toLocaleString()} Total Records
                  </Badge>
                  <Badge variant="outline" className={`
                    bg-green-500/10 text-green-400 border-green-500/30 backdrop-blur-sm
                    transition-all duration-300 hover:scale-110 hover:bg-green-500/20
                    ${animateCards ? 'animate-bounce delay-100' : ''}
                  `}>
                    All {dataStats.uniqueStocks} Stocks Loaded
                  </Badge>
                  {loading && (
                    <Badge variant="outline" className="bg-yellow-500/10 text-yellow-400 border-yellow-500/30 animate-pulse">
                      <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                      Loading Data...
                    </Badge>
                  )}
                </div>
              </div>
              <div className={`
                flex items-center space-x-3 transition-all duration-500
                ${animateCards ? 'translate-x-2 opacity-80' : 'translate-x-0 opacity-100'}
                ${focusedElement === 'header' ? 'space-x-4 scale-110' : 'space-x-3 scale-100'}
              `}>
                 <div className={`
                   relative w-full max-w-sm transition-all duration-300
                   ${searchQuery ? 'scale-110 ring-2 ring-emerald-500/50 shadow-lg shadow-emerald-500/20' : 'scale-100'}
                 `}>
                    <Search className={`
                      absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 transition-colors duration-300
                      ${searchQuery ? 'text-emerald-400' : 'text-gray-500'}
                    `} />
                    <Input
                      placeholder="Search stocks (Symbol, Company, Sector)..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onFocus={() => setFocusedElement('search')}
                      onBlur={() => setFocusedElement(null)}
                      className={`
                        pl-10 bg-gray-900/50 border-gray-700/50 backdrop-blur-sm
                        transition-all duration-300 hover:bg-gray-800/60 focus:ring-2 focus:ring-emerald-500/30
                        ${searchQuery ? 'bg-gray-800/80 border-emerald-500/70 text-white font-medium' : ''}
                      `}
                    />
                  </div>
                <Button
                  onClick={exportData}
                  size="sm"
                  variant="outline"
                  className={`
                    border-emerald-500/30 text-emerald-400 bg-emerald-500/5 backdrop-blur-sm
                    hover:bg-emerald-500/20 hover:scale-105 hover:border-emerald-400/50
                    transition-all duration-300 active:scale-95
                    ${animateCards ? 'animate-pulse delay-200' : ''}
                  `}
                >
                  <Database className="w-4 h-4 mr-2" />
                  Export CSV
                </Button>
                <Button
                  onClick={() => loadSentimentData(selectedTimeRange)}
                  size="sm"
                  disabled={loading}
                  className={`
                    bg-emerald-600/80 hover:bg-emerald-700/90 backdrop-blur-sm
                    hover:scale-105 transition-all duration-300 active:scale-95
                    disabled:opacity-50 disabled:cursor-not-allowed
                    ${loading ? 'animate-pulse' : ''}
                  `}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                  {loading ? 'Loading...' : 'Refresh Data'}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Market Overview with Dynamic Focus */}
        {marketGauge && (
          <div className={`
            border-b border-gray-800/30 transition-all duration-700
            ${focusedElement === 'overview' ? 'blur-sm scale-95 opacity-60' : 'blur-0 scale-100 opacity-100'}
            ${searchQuery ? 'p-4' : 'p-8'}
          `}>
            <Card className={`
              bg-gradient-to-br from-black/40 via-gray-900/30 to-black/50 
              border-gray-700/50 backdrop-blur-xl relative overflow-hidden
              transition-all duration-500 hover:scale-[1.02] hover:border-gray-600/50
              ${focusedElement === 'overview' ? 'scale-105 border-emerald-500/30 shadow-2xl shadow-emerald-500/10' : ''}
              ${animateCards ? 'animate-pulse' : ''}
            `}>
              {/* Static background overlay - no animation */}
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/2 via-transparent to-blue-500/2 opacity-50"></div>
              
              <CardHeader className="relative">
                <CardTitle 
                  className="flex items-center justify-between cursor-pointer group"
                  onClick={() => setFocusedElement(focusedElement === 'overview' ? null : 'overview')}
                >
                  <div className={`
                    flex items-center transition-all duration-500 group-hover:scale-105
                    ${focusedElement === 'overview' ? 'text-2xl' : 'text-xl'}
                  `}>
                    <Gauge className={`
                      mr-3 text-emerald-400 transition-all duration-500
                      ${focusedElement === 'overview' ? 'w-8 h-8 animate-spin' : 'w-6 h-6'}
                      ${animateCards ? 'animate-pulse' : ''}
                    `} />
                                         <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent font-bold tracking-wide">
                       Market Sentiment Overview
                     </span>
                    <span className="ml-2 text-gray-400 font-normal">
                      • {marketGauge.totalStocks} Stocks Analyzed
                    </span>
                  </div>
                  <Badge className={`
                    transition-all duration-300 hover:scale-110 backdrop-blur-sm
                    ${focusedElement === 'overview' ? 'scale-125 shadow-lg' : 'scale-100'}
                    ${marketGauge.regime === 'Bullish' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                      marketGauge.regime === 'Bearish' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                      'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'}
                    ${animateCards ? 'animate-bounce' : ''}
                  `}>
                    {marketGauge.regime} Market
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="relative">
                <div className={`
                  grid gap-6 transition-all duration-500
                  ${focusedElement === 'overview' ? 'grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-8' : 'grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4'}
                `}>
                  {[
                    {
                      value: `${(marketGauge.overall * 100).toFixed(1)}%`,
                      label: 'Overall Sentiment',
                      color: getSentimentColor(marketGauge.overall).replace('bg-', 'text-').split(' ')[0],
                      icon: marketGauge.overall > 0 ? TrendingUp : marketGauge.overall < 0 ? TrendingDown : Minus,
                      delay: 'delay-0'
                    },
                    {
                      value: `${(marketGauge.confidence * 100).toFixed(1)}%`,
                      label: 'Avg Confidence',
                      color: 'text-blue-400',
                      icon: Target,
                      delay: 'delay-75'
                    },
                    {
                      value: marketGauge.positiveCount.toString(),
                      label: 'Bullish Stocks',
                      color: 'text-green-400',
                      icon: TrendingUp,
                      delay: 'delay-150'
                    },
                    {
                      value: marketGauge.negativeCount.toString(),
                      label: 'Bearish Stocks',
                      color: 'text-red-400',
                      icon: TrendingDown,
                      delay: 'delay-225'
                    },
                    {
                      value: marketGauge.neutralCount.toString(),
                      label: 'Neutral Stocks',
                      color: 'text-yellow-400',
                      icon: Minus,
                      delay: 'delay-300'
                    },
                    {
                      value: marketGauge.totalStocks.toString(),
                      label: 'Total Analyzed',
                      color: 'text-emerald-400',
                      icon: Activity,
                      delay: 'delay-375'
                    },
                    {
                      value: marketGauge.totalStocks > 0 ? `${((marketGauge.positiveCount / marketGauge.totalStocks) * 100).toFixed(0)}%` : '0%',
                      label: 'Bullish Ratio',
                      color: 'text-purple-400',
                      icon: Gauge,
                      delay: 'delay-450'
                    }
                  ].map((metric, index) => {
                    const IconComponent = metric.icon;
                    return (
                      <div 
                        key={index}
                        className={`
                          text-center p-4 rounded-xl bg-gray-900/30 backdrop-blur-sm border border-gray-700/30
                          hover:bg-gray-800/40 hover:border-gray-600/50 hover:scale-105
                          transition-all duration-300 cursor-pointer group relative overflow-hidden
                          ${animateCards ? `animate-bounce ${metric.delay}` : ''}
                          ${focusedElement === 'overview' ? 'p-6' : 'p-4'}
                        `}
                        onClick={() => setFocusedElement(`metric-${index}`)}
                      >
                        {/* Glow effect on hover */}
                        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        
                        <div className="relative">
                          <div className="flex items-center justify-center mb-2">
                            <IconComponent className={`
                              w-4 h-4 mr-2 transition-all duration-300
                              ${metric.color.replace('text-', 'text-')}
                              ${focusedElement === 'overview' ? 'w-6 h-6' : 'w-4 h-4'}
                              group-hover:scale-110
                            `} />
                          </div>
                          <div className={`
                            font-bold mb-2 transition-all duration-300
                            ${metric.color}
                            ${focusedElement === 'overview' ? 'text-3xl' : 'text-2xl'}
                            group-hover:scale-110
                          `}>
                            {metric.value}
                          </div>
                          <div className={`
                            text-gray-400 transition-all duration-300
                            ${focusedElement === 'overview' ? 'text-sm' : 'text-xs'}
                            group-hover:text-gray-300
                          `}>
                            {metric.label}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Enhanced Filters with Dynamic Focus */}
        <div className={`
          border-b border-gray-800/30 transition-all duration-700
          ${focusedElement === 'filters' ? 'blur-sm scale-95 opacity-60' : 'blur-0 scale-100 opacity-100'}
          ${searchQuery ? 'p-4' : 'p-8'}
        `}>
          <Card className={`
            bg-gradient-to-br from-black/40 via-gray-900/30 to-black/50 
            border-gray-700/50 backdrop-blur-xl relative overflow-hidden
            transition-all duration-500 hover:border-gray-600/50
            ${focusedElement === 'filters' ? 'scale-105 border-emerald-500/30 shadow-2xl' : ''}
            ${animateCards ? 'animate-pulse' : ''}
          `}>
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/2 via-transparent to-blue-500/2 opacity-20"></div>
            
            <CardHeader className="relative">
              <CardTitle 
                className="flex items-center justify-between cursor-pointer group"
                onClick={() => setFocusedElement(focusedElement === 'filters' ? null : 'filters')}
              >
                <div className={`
                  flex items-center transition-all duration-500 group-hover:scale-105
                  ${focusedElement === 'filters' ? 'text-2xl' : 'text-xl'}
                `}>
                  <Filter className={`
                    mr-3 text-emerald-400 transition-all duration-500
                    ${focusedElement === 'filters' ? 'w-8 h-8 animate-spin' : 'w-6 h-6'}
                    ${animateCards ? 'animate-pulse' : ''}
                  `} />
                                     <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent font-bold tracking-wide">
                     Filter by duration
                   </span>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={clearAllFilters}
                  className={`
                    text-gray-400 border-gray-600/50 bg-gray-900/20 backdrop-blur-sm
                    hover:bg-gray-800/40 hover:border-gray-500/50 hover:scale-105
                    transition-all duration-300 active:scale-95
                    ${animateCards ? 'animate-bounce delay-100' : ''}
                  `}
                >
                  Clear All
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className={`
              space-y-8 relative transition-all duration-500
              ${focusedElement === 'filters' ? 'p-8' : 'p-6'}
            `}>
               <div className={`
                 flex items-center space-x-3 transition-all duration-500
                 ${animateCards ? 'animate-pulse' : ''}
               `}>
                  {DURATION_BUTTONS.map((duration: DurationButton, index: number) => (
                      <Button
                          key={duration.value}
                          variant={selectedTimeRange === duration.value ? "default" : "outline"}
                          onClick={() => {
                            setSelectedTimeRange(duration.value)
                            setFocusedElement('duration')
                            setTimeout(() => setFocusedElement(null), 1000)
                          }}
                          disabled={loading}
                          className={`
                              relative overflow-hidden backdrop-blur-sm transition-all duration-300
                              hover:scale-105 active:scale-95 disabled:opacity-50
                              ${selectedTimeRange === duration.value 
                                  ? 'bg-emerald-600/80 hover:bg-emerald-700/90 text-white shadow-lg shadow-emerald-500/20 border-emerald-500/50' 
                                  : 'border-gray-600/50 text-gray-300 hover:bg-gray-800/60 hover:border-gray-500/50 hover:text-gray-200'
                              }
                              ${animateCards ? `animate-bounce delay-${index * 50}` : ''}
                              ${focusedElement === 'duration' && selectedTimeRange === duration.value ? 'scale-110 shadow-xl shadow-emerald-500/30' : ''}
                          `}
                      >
                          <span className="relative z-10">{duration.label}</span>
                          {selectedTimeRange === duration.value && (
                            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/20 to-blue-500/20 animate-gradient-x"></div>
                          )}
                      </Button>
                  ))}
              </div>
              {/* Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Sentiment Range: {(sentimentRange[0] * 100).toFixed(0)}% to {(sentimentRange[1] * 100).toFixed(0)}%
                  </label>
                  <Slider
                    value={sentimentRange}
                    onValueChange={setSentimentRange}
                    min={-1}
                    max={1}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Confidence Range: {confidenceRange[0]}% to {confidenceRange[1]}%
                  </label>
                  <Slider
                    value={confidenceRange}
                    onValueChange={setConfidenceRange}
                    min={0}
                    max={100}
                    step={5}
                    className="w-full"
                  />
                </div>
              </div>

              {/* Multi-select filters */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Sectors ({selectedSectors.length} selected)
                  </label>
                  <div className="max-h-32 overflow-y-auto bg-gray-900 rounded border border-gray-700 p-2">
                    {getAvailableOptions.sectors.map(sector => (
                      <label key={sector} className="flex items-center space-x-2 mb-1">
                        <input
                          type="checkbox"
                          checked={selectedSectors.includes(sector)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedSectors([...selectedSectors, sector])
                            } else {
                              setSelectedSectors(selectedSectors.filter(s => s !== sector))
                            }
                          }}
                          className="rounded"
                        />
                        <span className="text-sm">{sector}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Market Cap ({selectedMarketCaps.length} selected)
                  </label>
                  <div className="max-h-32 overflow-y-auto bg-gray-900 rounded border border-gray-700 p-2">
                    {getAvailableOptions.marketCaps.map(cap => (
                      <label key={cap} className="flex items-center space-x-2 mb-1">
                        <input
                          type="checkbox"
                          checked={selectedMarketCaps.includes(cap)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedMarketCaps([...selectedMarketCaps, cap])
                            } else {
                              setSelectedMarketCaps(selectedMarketCaps.filter(c => c !== cap))
                            }
                          }}
                          className="rounded"
                        />
                        <span className="text-sm">{cap}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">
                    Market Factors ({selectedFactors.length} selected)
                  </label>
                  <div className="max-h-32 overflow-y-auto bg-gray-900 rounded border border-gray-700 p-2">
                    {getAvailableOptions.factors.map(factor => (
                      <label key={factor} className="flex items-center space-x-2 mb-1">
                        <input
                          type="checkbox"
                          checked={selectedFactors.includes(factor)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedFactors([...selectedFactors, factor])
                            } else {
                              setSelectedFactors(selectedFactors.filter(f => f !== factor))
                            }
                          }}
                          className="rounded"
                        />
                        <span className="text-xs">{factor}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
         <div className={`
           transition-all duration-700
           ${focusedElement === 'stocks' ? 'blur-sm opacity-60' : 'blur-0 opacity-100'}
           ${searchQuery ? 'p-4' : 'p-8'}
         `}>
          <ResizablePanelGroup direction="horizontal" className="min-h-[800px]">
            <ResizablePanel defaultSize={selectedStock ? 40 : 100}>
                <Tabs defaultValue="stocks" className="space-y-8 h-full flex flex-col">
                    <TabsList className={`
                      bg-gray-900/50 border border-gray-700/50 backdrop-blur-xl p-2 rounded-xl
                      transition-all duration-500 hover:border-gray-600/70
                      ${animateCards ? 'animate-pulse' : ''}
                    `}>
                      <TabsTrigger 
                        value="stocks"
                        onClick={() => setFocusedElement('tab-stocks')}
                        className={`
                          transition-all duration-300 hover:scale-105 data-[state=active]:bg-emerald-600/80
                          data-[state=active]:text-white data-[state=active]:shadow-lg
                          ${focusedElement === 'tab-stocks' ? 'scale-110' : 'scale-100'}
                        `}
                      >
                        <Activity className="w-4 h-4 mr-2" />
                        All Stocks (117)
                      </TabsTrigger>
                      <TabsTrigger 
                        value="treemap"
                        onClick={() => setFocusedElement('tab-treemap')}
                        className={`
                          transition-all duration-300 hover:scale-105 data-[state=active]:bg-emerald-600/80
                          data-[state=active]:text-white data-[state=active]:shadow-lg
                          ${focusedElement === 'tab-treemap' ? 'scale-110' : 'scale-100'}
                        `}
                      >
                        <Target className="w-4 h-4 mr-2" />
                        Sentiment Treemap
                      </TabsTrigger>
                      <TabsTrigger 
                        value="sectors"
                        onClick={() => setFocusedElement('tab-sectors')}
                        className={`
                          transition-all duration-300 hover:scale-105 data-[state=active]:bg-emerald-600/80
                          data-[state=active]:text-white data-[state=active]:shadow-lg
                          ${focusedElement === 'tab-sectors' ? 'scale-110' : 'scale-100'}
                        `}
                      >
                        <Target className="w-4 h-4 mr-2" />
                        Sectors
                      </TabsTrigger>
                       <TabsTrigger 
                         value="movers"
                         onClick={() => setFocusedElement('tab-movers')}
                         className={`
                           transition-all duration-300 hover:scale-105 data-[state=active]:bg-emerald-600/80
                           data-[state=active]:text-white data-[state=active]:shadow-lg
                           ${focusedElement === 'tab-movers' ? 'scale-110' : 'scale-100'}
                         `}
                       >
                        <Zap className="w-4 h-4 mr-2" />
                        Top Movers
                      </TabsTrigger>
                    </TabsList>

                    {/* Enhanced All Stocks Tab */}
                    <TabsContent value="stocks" className="flex-grow">
                      <Card className={`
                        bg-gradient-to-br from-black/40 via-gray-900/30 to-black/50 
                        border-gray-700/50 backdrop-blur-xl h-full relative overflow-hidden
                        transition-all duration-500 hover:border-gray-600/50
                        ${focusedElement === 'stocks' ? 'scale-105 border-emerald-500/30 shadow-2xl' : ''}
                      `}>
                        <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/2 via-transparent to-blue-500/2 opacity-30"></div>
                        
                        <CardHeader className="relative">
                          <CardTitle className={`
                            flex items-center justify-between cursor-pointer group
                            transition-all duration-300 hover:scale-105
                            ${focusedElement === 'stocks' ? 'text-2xl' : 'text-xl'}
                          `}>
                                                                                     <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent font-bold tracking-wide">
                              Stock Analysis {searchQuery && (
                                <span className="text-xs text-gray-400 font-normal">
                                  • Search: &ldquo;{searchQuery}&rdquo;
                                </span>
                              )}
                            </span>
                            <div className="flex items-center space-x-2">
                              <Badge className={`
                                bg-emerald-500/10 text-emerald-400 border-emerald-500/30
                                transition-all duration-300 hover:scale-110
                                ${animateCards ? 'animate-bounce' : ''}
                              `}>
                                {filteredData.length} records
                              </Badge>
                              {searchQuery && (
                                <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/30 text-xs">
                                  Filtered
                                </Badge>
                              )}
                            </div>
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="relative">
                          <div className={`
                            space-y-3 max-h-[700px] overflow-y-auto pr-2
                            scrollbar-thin scrollbar-thumb-emerald-500/20 scrollbar-track-gray-800/20
                            ${animateCards ? 'animate-pulse' : ''}
                          `}>
                            {searchQuery && filteredData.length === 0 && (
                              <div className="text-center py-12">
                                <div className="text-gray-400 mb-2">No stocks found matching &ldquo;{searchQuery}&rdquo;</div>
                                <div className="text-xs text-gray-500">Try adjusting your search terms or filters</div>
                              </div>
                            )}
                            {filteredData.slice(0, 300).map((stock, index) => (
                              <div 
                                key={`${stock.Symbol}-${stock.Date}-${index}`} 
                                className={`
                                  group relative flex items-center justify-between p-4 rounded-xl 
                                  border border-gray-700/50 bg-gray-900/20 backdrop-blur-sm
                                  hover:bg-gray-800/40 hover:border-gray-600/50 hover:scale-[1.02]
                                  transition-all duration-300 cursor-pointer overflow-hidden
                                  ${focusedElement === `stock-${index}` ? 'scale-105 border-emerald-500/50 shadow-lg shadow-emerald-500/10' : ''}
                                  ${animateCards ? `animate-fade-in-up delay-${Math.min(index * 10, 1000)}` : ''}
                                `}
                                onClick={() => {
                                  handleStockClick(stock)
                                  setFocusedElement(`stock-${index}`)
                                }}
                                onMouseEnter={() => setFocusedElement(`stock-${index}`)}
                                onMouseLeave={() => setFocusedElement(null)}
                              >
                                {/* Glow effect */}
                                <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                
                                <div className="relative flex items-center space-x-4">
                                  <div className={`
                                    p-2 rounded-lg bg-gray-800/50 transition-all duration-300
                                    group-hover:bg-gray-700/50 group-hover:scale-110
                                  `}>
                                    {getSentimentIcon(stock.Sentiment_Score)}
                                  </div>
                                  <div className="flex-grow">
                                    <div className="font-medium flex items-center space-x-3 mb-1">
                                      <span className={`
                                        text-lg transition-all duration-300
                                        ${focusedElement === `stock-${index}` ? 'text-emerald-400' : 'text-white'}
                                      `}>
                                        {stock.Symbol}
                                      </span>
                                      <span className="text-xs text-gray-500 bg-gray-800/50 px-2 py-1 rounded">
                                        {stock.Date}
                                      </span>
                                    </div>
                                    <div className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
                                      {stock.Company_Name}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                      {stock.Sector} • {stock.News_Volume} news • {stock.Confidence_Score.toFixed(0)}% confidence
                                    </div>
                                  </div>
                                </div>
                                <div className="relative text-right">
                                  <Badge className={`
                                    ${getSentimentColor(stock.Sentiment_Score)} 
                                    transition-all duration-300 hover:scale-110
                                    ${focusedElement === `stock-${index}` ? 'scale-110 shadow-lg' : ''}
                                  `}>
                                    {(stock.Sentiment_Score * 100).toFixed(1)}%
                                  </Badge>
                                  <div className="text-xs text-gray-500 mt-1">
                                    {stock.Price_Change_Percent >= 0 ? '+' : ''}{stock.Price_Change_Percent.toFixed(2)}%
                                  </div>
                                </div>
                              </div>
                            ))}
                            {filteredData.length > 300 && (
                              <div className={`
                                text-center py-6 text-gray-400 bg-gray-900/20 rounded-xl border border-gray-700/30
                                backdrop-blur-sm transition-all duration-300 hover:bg-gray-800/30
                              `}>
                                <div className="text-sm">
                                  Showing first 300 results. Use filters to narrow down.
                                </div>
                                <div className="text-xs text-gray-500 mt-1">
                                  Total: {filteredData.length} stocks found
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    </TabsContent>
                    
                    {/* Treemap Tab */}
                    <TabsContent value="treemap" className="flex-grow">
                      <Card className="bg-black/50 border-gray-800 h-full">
                        <CardHeader>
                          <CardTitle>Sentiment Treemap</CardTitle>
                        </CardHeader>
                        <CardContent className="h-[700px]">
                          {filteredData.length > 0 ? (
                            <SentimentTreemap 
                              data={filteredData}
                              onStockClick={(symbol) => {
                                const stockData = filteredData.find(s => s.Symbol === symbol);
                                if (stockData) handleStockClick(stockData);
                              }}
                              showAllStocks={true}
                            />
                          ) : (
                            <div className="flex items-center justify-center h-full">
                              <div className="text-gray-400">No data to display</div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </TabsContent>
                     {/* Sectors Tab */}
                    <TabsContent value="sectors">
                      {sectorMetrics.length > 0 ? (
                        <SectorHeatmap data={sectorMetrics} />
                      ) : (
                        <Card className="bg-black/50 border-gray-800">
                          <CardContent className="p-12 text-center text-gray-500">
                            No sector data available for the selected filters.
                          </CardContent>
                        </Card>
                      )}
                    </TabsContent>
                     {/* Top Movers Tab */}
                    <TabsContent value="movers">
                      <Card className="bg-black/50 border-gray-800">
                        <CardHeader>
                          <CardTitle>Top Sentiment Performers</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {topMovers.map((mover, index) => (
                              <div key={mover.symbol} className="flex items-center justify-between p-4 rounded-lg border border-gray-700 bg-gray-900/30 hover:bg-gray-800/50 transition-colors">
                                <div className="flex items-center space-x-4">
                                  <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold text-sm">
                                    {index + 1}
                                  </div>
                                  <div>
                                    <div className="font-medium flex items-center space-x-2">
                                      <span>{mover.symbol}</span>
                                      {getSentimentIcon(mover.sentiment)}
                                    </div>
                                    <div className="text-sm text-gray-400">{mover.company}</div>
                                    <div className="text-xs text-gray-500">{mover.sector} • {mover.marketCap}</div>
                                  </div>
                                </div>
                                <div className="text-right space-y-1">
                                  <div className="flex items-center space-x-3">
                                    <Badge className={getSentimentColor(mover.sentiment)}>
                                      {(mover.sentiment * 100).toFixed(1)}%
                                    </Badge>
                                    <div className="text-sm text-gray-400">
                                      {mover.confidence.toFixed(0)}% conf
                                    </div>
                                  </div>
                                  <div className="text-xs text-gray-500 space-x-2">
                                    <span>{mover.newsVolume} news</span>
                                    <span className={mover.priceChange >= 0 ? 'text-green-400' : 'text-red-400'}>
                                      {mover.priceChange.toFixed(2)}% price
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </TabsContent>
                </Tabs>
            </ResizablePanel>
            {selectedStock && (
                <>
                    <ResizableHandle withHandle />
                    <ResizablePanel defaultSize={60}>
                        <StockDetailView stock={selectedStock} onBack={() => setSelectedStock(null)} />
                    </ResizablePanel>
                </>
            )}
          </ResizablePanelGroup>
        </div>
      </div>
    </AppLayout>
  )
}
