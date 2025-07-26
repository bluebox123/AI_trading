'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown, Activity, ChevronUp, ChevronDown } from 'lucide-react'
import MiniChart from './MiniChart'

interface MarketIndex {
  name: string
  code: string
  value: number
  change: number
  changePercent: number
  high: number
  low: number
  volume: string
  lastUpdate: string
  chartData: number[]
  status: 'OPEN' | 'CLOSED' | 'PRE_OPEN'
}

export default function MarketIndicesWidget() {
  const [indices, setIndices] = useState<MarketIndex[]>([])
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  useEffect(() => {
    // Simulate API call with mock data
    const timer = setTimeout(() => {
      setIndices(generateMockIndices())
      setLoading(false)
    }, 800)

    // Update indices every 30 seconds to simulate live data
    const updateInterval = setInterval(() => {
      setIndices(prev => prev.map(index => ({
        ...index,
        value: index.value + (Math.random() - 0.5) * 50,
        change: index.change + (Math.random() - 0.5) * 10,
        changePercent: index.changePercent + (Math.random() - 0.5) * 0.5,
        chartData: [...index.chartData.slice(1), index.value + (Math.random() - 0.5) * 30],
        lastUpdate: new Date().toLocaleTimeString()
      })))
      setLastUpdate(new Date())
    }, 30000)

    return () => {
      clearTimeout(timer)
      clearInterval(updateInterval)
    }
  }, [])

  function generateMockIndices(): MarketIndex[] {
    const baseIndices = [
      {
        name: 'NIFTY 50',
        code: 'NIFTY',
        baseValue: 22000,
        volatility: 200
      },
      {
        name: 'BANK NIFTY',
        code: 'BANKNIFTY',
        baseValue: 45000,
        volatility: 500
      },
      {
        name: 'INDIA VIX',
        code: 'VIX',
        baseValue: 15,
        volatility: 2
      }
    ]

    return baseIndices.map(index => {
      const change = (Math.random() - 0.4) * index.volatility
      const value = index.baseValue + change
      const changePercent = (change / index.baseValue) * 100

      // Generate chart data for the last 30 points
      const chartData = Array.from({ length: 30 }, (_, i) => {
        const trend = Math.sin(i * 0.2) * index.volatility * 0.3
        const noise = (Math.random() - 0.5) * index.volatility * 0.5
        return index.baseValue + trend + noise
      })

      return {
        name: index.name,
        code: index.code,
        value: Number(value.toFixed(2)),
        change: Number(change.toFixed(2)),
        changePercent: Number(changePercent.toFixed(2)),
        high: Number((value + Math.random() * index.volatility * 0.5).toFixed(2)),
        low: Number((value - Math.random() * index.volatility * 0.5).toFixed(2)),
        volume: `${(Math.random() * 500 + 100).toFixed(0)}K Cr`,
        lastUpdate: new Date().toLocaleTimeString(),
        chartData,
        status: new Date().getHours() >= 9 && new Date().getHours() < 16 ? 'OPEN' : 'CLOSED'
      }
    })
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'OPEN':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
      case 'CLOSED':
        return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'PRE_OPEN':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
  }

  const getChartColor = (change: number, code: string): 'green' | 'red' | 'yellow' => {
    if (code === 'VIX') return 'yellow'
    return change >= 0 ? 'green' : 'red'
  }

  if (loading) {
    return (
      <Card className="bg-black/80 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <span>Market Indices</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="w-20 h-4 bg-gray-700 rounded mb-2"></div>
                  <div className="w-24 h-6 bg-gray-700 rounded mb-2"></div>
                  <div className="w-16 h-4 bg-gray-700 rounded mb-2"></div>
                  <div className="w-full h-8 bg-gray-700 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-black/80 border-gray-700 hover:border-gray-600 transition-colors">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <span>Market Indices</span>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className={getStatusBadge(indices[0]?.status || 'CLOSED')}>
              {indices[0]?.status || 'CLOSED'}
            </Badge>
            <span className="text-xs text-gray-400">
              Updated: {lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {indices.map((index) => (
            <div 
              key={index.code}
              className="p-4 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-all duration-200 hover:scale-[1.02] cursor-pointer group"
            >
              {/* Index Header */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="font-bold text-white text-sm group-hover:text-emerald-400 transition-colors">
                    {index.name}
                  </h3>
                  <p className="text-xs text-gray-400">{index.code}</p>
                </div>
                {index.code === 'VIX' && (
                  <Badge variant="outline" className="text-xs border-yellow-500/30 text-yellow-400">
                    Volatility
                  </Badge>
                )}
              </div>

              {/* Current Value */}
              <div className="mb-3">
                <div className="text-2xl font-bold text-white">
                  {index.code === 'VIX' ? index.value.toFixed(2) : index.value.toLocaleString()}
                </div>
                <div className={`flex items-center text-sm ${
                  index.change >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {index.change >= 0 ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                  <span className="mr-2">
                    {Math.abs(index.change).toFixed(2)}
                  </span>
                  <span>({Math.abs(index.changePercent).toFixed(2)}%)</span>
                </div>
              </div>

              {/* High/Low */}
              <div className="flex justify-between text-xs text-gray-400 mb-3">
                <span>H: <span className="text-emerald-400">{index.high.toLocaleString()}</span></span>
                <span>L: <span className="text-red-400">{index.low.toLocaleString()}</span></span>
              </div>

              {/* Volume */}
              <div className="text-xs text-gray-400 mb-3">
                Volume: <span className="text-white">{index.volume}</span>
              </div>

              {/* Mini Chart */}
              <div className="h-12 mb-2">
                <MiniChart 
                  data={index.chartData}
                  color={getChartColor(index.change, index.code)}
                  height={48}
                />
              </div>

              {/* Last Update */}
              <div className="text-xs text-gray-500 text-center">
                Last: {index.lastUpdate}
              </div>
            </div>
          ))}
        </div>

        {/* Market Status Summary */}
        <div className="mt-6 pt-4 border-t border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-sm text-gray-400">Market Status</div>
              <div className={`font-medium ${
                indices[0]?.status === 'OPEN' ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {indices[0]?.status || 'CLOSED'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Advancing</div>
              <div className="font-medium text-emerald-400">1,247</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Declining</div>
              <div className="font-medium text-red-400">863</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Unchanged</div>
              <div className="font-medium text-gray-400">124</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 