'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown, ChevronUp, ChevronDown, Activity } from 'lucide-react'
import MiniChart from './MiniChart'

interface StockSignal {
  symbol: string
  signal: 'BUY' | 'SELL' | 'HOLD'
  final_score: number
  confidence: number
  current_price: number
  price_target?: number
  price?: number
  change?: number
  change_percent?: number
  volume?: number
  sector?: string
  company?: string
  chartData?: number[]
}

interface TopSignalsTableProps {
  type: 'buy' | 'sell'
  title?: string
}

export default function TopSignalsTable({ type, title }: TopSignalsTableProps) {
  const [signals, setSignals] = useState<StockSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchSignals() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch('/api/signals')
        const data = await res.json()
        
        if (data.error) {
          throw new Error(data.error)
        }
        
        const allSignals = data.signals || []
        
        // Filter signals by type and sort by final_score
        const filteredSignals = allSignals.filter((signal: StockSignal) => {
          if (type === 'buy') {
            return signal.signal === 'BUY'
          } else {
            return signal.signal === 'SELL'
          }
        })
        
        // Sort by final_score (descending for buy, ascending for sell)
        const sortedSignals = filteredSignals.sort((a: StockSignal, b: StockSignal) => {
          if (type === 'buy') {
            return b.final_score - a.final_score
          } else {
            return a.final_score - b.final_score
          }
        })
        
        // Take top 10
        const topSignals = sortedSignals.slice(0, 10)
        
        // Add mock chart data for visualization
        const signalsWithChartData = topSignals.map(signal => ({
          ...signal,
          chartData: Array.from({ length: 20 }, (_, i) => 
            (signal.current_price || signal.price || 1000) + Math.sin(i * 0.5) * 20 + (Math.random() - 0.5) * 30
          )
        }))
        
        setSignals(signalsWithChartData)
      } catch (e: any) {
        setError(e.message || 'Failed to fetch signals')
        console.error('Error fetching signals:', e)
      } finally {
        setLoading(false)
      }
    }

    fetchSignals()
  }, [type])

  const getSignalColor = (score: number) => {
    const absScore = Math.abs(score)
    if (absScore >= 0.8) return type === 'buy' ? 'text-emerald-400' : 'text-red-400'
    if (absScore >= 0.6) return type === 'buy' ? 'text-green-400' : 'text-orange-400'
    if (absScore >= 0.4) return type === 'buy' ? 'text-yellow-400' : 'text-yellow-400'
    return type === 'buy' ? 'text-gray-400' : 'text-gray-400'
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.9) return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
    if (confidence >= 0.8) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    if (confidence >= 0.7) return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
    return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }

  const getRating = (signal: StockSignal) => {
    const score = Math.abs(signal.final_score)
    if (signal.signal === 'BUY') {
      if (score >= 0.8) return 'Strong Buy'
      if (score >= 0.6) return 'Buy'
      if (score >= 0.4) return 'Weak Buy'
      return 'Hold'
    } else if (signal.signal === 'SELL') {
      if (score >= 0.8) return 'Strong Sell'
      if (score >= 0.6) return 'Sell'
      if (score >= 0.4) return 'Weak Sell'
      return 'Hold'
    }
    return 'Hold'
  }

  if (loading) {
    return (
      <Card className="bg-black/80 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {type === 'buy' ? (
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
            <span>{title || `Top 10 ${type === 'buy' ? 'Buy' : 'Sell'} Signals`}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="flex items-center space-x-4">
                  <div className="w-8 h-8 bg-gray-700 rounded"></div>
                  <div className="flex-1">
                    <div className="w-24 h-4 bg-gray-700 rounded mb-1"></div>
                    <div className="w-32 h-3 bg-gray-700 rounded"></div>
                  </div>
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                  <div className="w-20 h-4 bg-gray-700 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="bg-black/80 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {type === 'buy' ? (
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
            <span>{title || `Top 10 ${type === 'buy' ? 'Buy' : 'Sell'} Signals`}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="text-red-400 mb-2">Error loading signals</div>
            <div className="text-sm text-gray-400">{error}</div>
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
            {type === 'buy' ? (
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
            <span>{title || `Top 10 ${type === 'buy' ? 'Buy' : 'Sell'} Signals`}</span>
          </div>
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Live</span>
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {signals.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-gray-400 mb-2">No {type} signals available</div>
              <div className="text-sm text-gray-500">Check back later for updates</div>
            </div>
          ) : (
            signals.map((signal, index) => (
              <div 
                key={signal.symbol} 
                className="flex items-center space-x-4 p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors cursor-pointer group"
              >
                {/* Rank */}
                <div className="flex items-center justify-center w-8 h-8 bg-gray-700 rounded-lg font-bold text-sm">
                  {index + 1}
                </div>

                {/* Stock Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-bold text-white group-hover:text-emerald-400 transition-colors">
                      {signal.symbol}
                    </h3>
                    <Badge variant="outline" className="text-xs border-gray-600 text-gray-400">
                      {signal.sector || 'N/A'}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-400 truncate">{signal.company || 'N/A'}</p>
                  <p className="text-xs text-gray-500">{getRating(signal)}</p>
                </div>

                {/* Signal Score */}
                <div className="text-center">
                  <div className={`text-lg font-bold ${getSignalColor(signal.final_score)}`}>
                    {(signal.final_score * 100).toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-400">Score</div>
                </div>

                {/* Confidence */}
                <div className="text-center">
                  <Badge className={`${getConfidenceBadge(signal.confidence)} text-xs`}>
                    {(signal.confidence * 100).toFixed(0)}%
                  </Badge>
                  <div className="text-xs text-gray-400 mt-1">Confidence</div>
                </div>

                {/* Price & Change */}
                <div className="text-right">
                  <div className="font-bold text-white">
                    ₹{(signal.current_price || signal.price || 0).toLocaleString()}
                  </div>
                  {signal.price_target && (
                    <div className={`flex items-center text-xs ${
                      signal.price_target >= (signal.current_price || signal.price || 0) 
                        ? 'text-emerald-400' 
                        : 'text-red-400'
                    }`}>
                      {signal.price_target >= (signal.current_price || signal.price || 0) ? (
                        <ChevronUp className="w-3 h-3" />
                      ) : (
                        <ChevronDown className="w-3 h-3" />
                      )}
                      <span>Target: ₹{signal.price_target.toFixed(2)}</span>
                    </div>
                  )}
                  {signal.change_percent && (
                    <div className={`text-xs ${
                      signal.change_percent >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {signal.change_percent >= 0 ? '+' : ''}{signal.change_percent.toFixed(2)}%
                    </div>
                  )}
                </div>

                {/* Mini Chart */}
                <div className="w-20 h-10">
                  <MiniChart 
                    data={signal.chartData || []} 
                    color={type === 'buy' ? 'green' : 'red'}
                    height={40}
                  />
                </div>
              </div>
            ))
          )}
        </div>

        {/* View All Button */}
        <div className="mt-6 pt-4 border-t border-gray-700">
          <button className="w-full py-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors font-medium">
            View All {type === 'buy' ? 'Buy' : 'Sell'} Signals →
          </button>
        </div>
      </CardContent>
    </Card>
  )
} 