'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface SectorMetric {
  sector: string
  avgSentiment: number
  stockCount: number
}

interface SectorHeatmapProps {
  data: SectorMetric[]
}

// Helper to get a color from green to red based on sentiment
const getHeatmapColor = (sentiment: number) => {
  if (sentiment > 0.6) return 'bg-green-700/80 hover:bg-green-600'
  if (sentiment > 0.3) return 'bg-green-600/70 hover:bg-green-500'
  if (sentiment > 0.1) return 'bg-green-500/60 hover:bg-green-400'
  if (sentiment < -0.6) return 'bg-red-700/80 hover:bg-red-600'
  if (sentiment < -0.3) return 'bg-red-600/70 hover:bg-red-500'
  if (sentiment < -0.1) return 'bg-red-500/60 hover:bg-red-400'
  return 'bg-gray-600/50 hover:bg-gray-500'
}

export function SectorHeatmap({ data }: SectorHeatmapProps) {
  const sortedSectors = useMemo(() => {
    return [...data].sort((a, b) => b.stockCount - a.stockCount)
  }, [data])

  return (
    <Card className="bg-black/50 border-gray-800">
      <CardHeader>
        <CardTitle>Sector Sentiment Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
          {sortedSectors.map(sector => (
            <div
              key={sector.sector}
              className={`p-4 rounded-lg transition-all duration-200 text-white ${getHeatmapColor(sector.avgSentiment)}`}
            >
              <div className="font-bold text-sm truncate">{sector.sector}</div>
              <div className="text-2xl font-light">
                {(sector.avgSentiment * 100).toFixed(1)}%
              </div>
              <div className="text-xs opacity-80">{sector.stockCount} stocks</div>
            </div>
          ))}
        </div>
        <div className="flex items-center justify-end space-x-4 text-xs mt-4 text-gray-400">
          <span>Color Scale:</span>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-sm bg-red-600"></div>
            <span>Bearish</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-sm bg-gray-600"></div>
            <span>Neutral</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 rounded-sm bg-green-600"></div>
            <span>Bullish</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 