'use client'

import { useState, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  BarChart3,
  Maximize2
} from 'lucide-react'
import { ResponsiveTreeMap } from '@nivo/treemap'

interface SentimentData {
  Symbol: string
  Company_Name: string
  Sector: string
  Market_Cap_Category: string
  Sentiment_Score: number
  Sentiment_Category: string
  Confidence_Score: number
  News_Volume: number
  Social_Media_Mentions: number
}

interface TreemapNode {
  id: string
  symbol: string
  name: string
  value: number
  sentiment: number
  confidence: number
  sector: string
  marketCap: string
  color: string
  textColor: string
}

interface SentimentTreemapProps {
  data: SentimentData[]
  onStockClick?: (symbol: string) => void
  className?: string
  showAllStocks?: boolean
}

const getSentimentColor = (sentiment: number): string => {
  if (sentiment > 0.5) return 'rgb(34, 197, 94)' // Strong positive - green
  if (sentiment > 0.2) return 'rgb(132, 204, 22)' // Positive - light green
  if (sentiment > -0.2) return 'rgb(234, 179, 8)' // Neutral - yellow
  if (sentiment > -0.5) return 'rgb(234, 88, 12)' // Negative - orange
  return 'rgb(239, 68, 68)' // Strong negative - red
}

const getSentimentCategory = (score: number): string => {
  if (score > 0.5) return 'Very Bullish'
  if (score > 0.2) return 'Bullish'
  if (score > -0.2) return 'Neutral'
  if (score > -0.5) return 'Bearish'
  return 'Very Bearish'
}

export function SentimentTreemap({ data, onStockClick, className, showAllStocks }: SentimentTreemapProps) {
  const [selectedSector, setSelectedSector] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<'marketCap' | 'sentiment' | 'newsVolume'>('marketCap')

  // Process data for treemap
  const treemapData = useMemo(() => {
    let filteredData = selectedSector 
      ? data.filter(item => item.Sector === selectedSector)
      : data

    // Remove duplicates by symbol, keeping the first occurrence
    const uniqueData = filteredData.filter((item, index, self) => 
      self.findIndex(d => d.Symbol === item.Symbol) === index
    )

    // Calculate relative sizes based on sort criteria
    const nodes: TreemapNode[] = uniqueData.map(item => {
      const marketCapWeight = item.Market_Cap_Category === 'Large Cap' ? 100 : 
                              item.Market_Cap_Category === 'Mid Cap' ? 50 : 25
      
      let value: number
      switch (sortBy) {
        case 'sentiment':
          value = Math.abs(item.Sentiment_Score) * 100
          break
        case 'newsVolume':
          value = item.News_Volume
          break
        default:
          value = marketCapWeight
      }

      // Color based on sentiment
      const sentiment = item.Sentiment_Score
      let color: string
      let textColor: string

      if (sentiment > 0.3) {
        const intensity = Math.min(sentiment, 1) * 0.8 + 0.2
        color = `rgba(34, 197, 94, ${intensity})` // Green
        textColor = sentiment > 0.6 ? 'white' : 'black'
      } else if (sentiment < -0.3) {
        const intensity = Math.min(Math.abs(sentiment), 1) * 0.8 + 0.2
        color = `rgba(239, 68, 68, ${intensity})` // Red
        textColor = Math.abs(sentiment) > 0.6 ? 'white' : 'black'
      } else {
        const intensity = 0.3
        color = `rgba(251, 191, 36, ${intensity})` // Yellow
        textColor = 'black'
      }

      return {
        id: item.Symbol,
        symbol: item.Symbol,
        name: item.Company_Name,
        value,
        sentiment: item.Sentiment_Score,
        confidence: item.Confidence_Score,
        sector: item.Sector,
        marketCap: item.Market_Cap_Category,
        color,
        textColor
      }
    })

    return nodes.sort((a, b) => b.value - a.value)
  }, [data, selectedSector, sortBy])

  // Simple treemap layout calculation
  const layoutNodes = useMemo(() => {
    const totalValue = treemapData.reduce((sum, node) => sum + node.value, 0)
    const containerWidth = 800
    const containerHeight = 600
    const totalArea = containerWidth * containerHeight

    // When showing all, don't limit the nodes. Otherwise, show top 20.
    const nodesToShow = showAllStocks ? treemapData : treemapData.slice(0, 20)

    let currentX = 0
    let currentY = 0
    let currentRowHeight = 0
    const rowMaxWidth = containerWidth

    // A simple row-based layout algorithm
    return nodesToShow.map(node => {
      const area = (node.value / totalValue) * totalArea
      
      // Make aspect ratio and minimum sizes dependent on the number of nodes
      const aspectRatio = nodesToShow.length > 50 ? 1 : 1.5
      let width = Math.sqrt(area * aspectRatio)
      let height = area / width
      
      const minWidth = nodesToShow.length > 50 ? 50 : 80
      const minHeight = nodesToShow.length > 50 ? 40 : 60
      
      width = Math.max(width, minWidth)
      height = Math.max(height, minHeight)

      if (currentX + width > rowMaxWidth) {
        currentX = 0
        currentY += currentRowHeight
        currentRowHeight = 0
      }

      const layout = { ...node, x: currentX, y: currentY, width, height }

      currentX += width
      currentRowHeight = Math.max(currentRowHeight, height)

      return layout
    })
  }, [treemapData, showAllStocks])

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.1) return <TrendingUp className="w-3 h-3" />
    if (sentiment < -0.1) return <TrendingDown className="w-3 h-3" />
    return <Minus className="w-3 h-3" />
  }

  const sectors = useMemo(() => {
    return [...new Set(data.map(item => item.Sector))].sort()
  }, [data])

  const transformedData = useMemo(() => {
    const stocksByMarketCap = data.reduce((acc: any, stock) => {
      const category = stock.Market_Cap_Category || 'Unknown'
      if (!acc[category]) {
        acc[category] = {
          name: category,
          children: []
        }
      }
      acc[category].children.push({
        name: stock.Symbol,
        company: stock.Company_Name,
        sector: stock.Sector,
        sentiment: stock.Sentiment_Score,
        category: getSentimentCategory(stock.Sentiment_Score),
        confidence: stock.Confidence_Score,
        size: 1
      })
      return acc
    }, {})

    return {
      name: 'Market Sentiment',
      children: Object.values(stocksByMarketCap)
    }
  }, [data])

  return (
    <Card className={`bg-black/50 border-gray-800 ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-emerald-400" />
            Sentiment Treemap
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortBy('marketCap')}
              className={sortBy === 'marketCap' ? 'bg-emerald-500/20 text-emerald-400' : ''}
            >
              Market Cap
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortBy('sentiment')}
              className={sortBy === 'sentiment' ? 'bg-emerald-500/20 text-emerald-400' : ''}
            >
              Sentiment
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortBy('newsVolume')}
              className={sortBy === 'newsVolume' ? 'bg-emerald-500/20 text-emerald-400' : ''}
            >
              News Vol
            </Button>
          </div>
        </div>
        
        {/* Sector Filter */}
        <div className="flex flex-wrap gap-2 mt-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSelectedSector(null)}
            className={!selectedSector ? 'bg-emerald-500/20 text-emerald-400' : ''}
          >
            All Sectors
          </Button>
          {sectors.slice(0, 8).map(sector => (
            <Button
              key={sector}
              variant="outline"
              size="sm"
              onClick={() => setSelectedSector(sector)}
              className={selectedSector === sector ? 'bg-emerald-500/20 text-emerald-400' : ''}
            >
              {sector}
            </Button>
          ))}
        </div>
      </CardHeader>

      <CardContent>
        {/* Legend */}
        <div className="mb-6 p-4 bg-gray-900/50 rounded-lg">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <span className="text-gray-400">Color Scale:</span>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-red-400">Bearish</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                <span className="text-yellow-400">Neutral</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-green-400">Bullish</span>
              </div>
            </div>
            <div className="text-gray-400">
              Size: {sortBy === 'marketCap' ? 'Market Cap' : sortBy === 'sentiment' ? 'Sentiment Strength' : 'News Volume'}
            </div>
          </div>
        </div>

        {/* Treemap Visualization */}
        <div className="h-[600px] w-full">
          <ResponsiveTreeMap
            data={transformedData}
            identity="name"
            value="size"
            valueFormat=".02s"
            margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
            labelSkipSize={12}
            labelTextColor={{
              from: 'color',
              modifiers: [['darker', 3]]
            }}
            parentLabelPosition="left"
            parentLabelTextColor={{
              from: 'color',
              modifiers: [['darker', 2]]
            }}
            colors={(node: any) => getSentimentColor(node.data.sentiment)}
            borderColor={{
              from: 'color',
              modifiers: [['darker', 0.1]]
            }}
            onClick={(node: any) => {
              if (node.data.company) {
                onStockClick?.(node.data.name)
              }
            }}
            animate={true}
            motionConfig="gentle"
          />
        </div>

        {/* Summary Statistics */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-gray-900/50 rounded-lg">
            <div className="text-lg font-bold text-emerald-400">
              {layoutNodes.length}
            </div>
            <div className="text-xs text-gray-400">
              {showAllStocks ? `Showing ${layoutNodes.length} of ${treemapData.length}` : 'Stocks Shown'}
            </div>
          </div>
          <div className="text-center p-3 bg-gray-900/50 rounded-lg">
            <div className="text-lg font-bold text-green-400">
              {treemapData.filter(n => n.sentiment > 0.3).length}
            </div>
            <div className="text-xs text-gray-400">Bullish</div>
          </div>
          <div className="text-center p-3 bg-gray-900/50 rounded-lg">
            <div className="text-lg font-bold text-red-400">
              {treemapData.filter(n => n.sentiment < -0.3).length}
            </div>
            <div className="text-xs text-gray-400">Bearish</div>
          </div>
          <div className="text-center p-3 bg-gray-900/50 rounded-lg">
            <div className="text-lg font-bold text-blue-400">
              {treemapData.length > 0 ? (treemapData.reduce((sum, n) => sum + n.sentiment, 0) / treemapData.length * 100).toFixed(0) : '0'}%
            </div>
            <div className="text-xs text-gray-400">Avg Sentiment</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 