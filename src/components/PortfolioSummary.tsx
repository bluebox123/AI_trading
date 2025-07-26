'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown, DollarSign, PieChart, Activity, Target, AlertTriangle } from 'lucide-react'

interface PortfolioData {
  totalValue: number
  totalChange: number
  totalChangePercent: number
  dayPL: number
  dayPLPercent: number
  totalPL: number
  totalPLPercent: number
  positions: number
  buyingPower: number
  allocation: {
    equity: number
    cash: number
    others: number
  }
  riskMetrics: {
    beta: number
    sharpe: number
    maxDrawdown: number
    volatility: number
  }
}

export default function PortfolioSummary() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate API call with mock data
    const timer = setTimeout(() => {
      setPortfolio(generateMockPortfolio())
      setLoading(false)
    }, 1200)

    return () => clearTimeout(timer)
  }, [])

  function generateMockPortfolio(): PortfolioData {
    const totalValue = 2500000 + Math.random() * 1000000 // 25L - 35L
    const dayPL = (Math.random() - 0.3) * 50000 // -15K to +35K
    const totalPL = (Math.random() - 0.2) * 500000 // -100K to +400K
    
    return {
      totalValue,
      totalChange: totalValue * 0.02, // Assume 2% of total value
      totalChangePercent: 2.0,
      dayPL,
      dayPLPercent: (dayPL / totalValue) * 100,
      totalPL,
      totalPLPercent: (totalPL / (totalValue - totalPL)) * 100,
      positions: Math.floor(Math.random() * 15) + 8, // 8-22 positions
      buyingPower: 500000 + Math.random() * 300000, // 5L - 8L
      allocation: {
        equity: 75 + Math.random() * 15, // 75-90%
        cash: 5 + Math.random() * 15, // 5-20%
        others: 0 + Math.random() * 10 // 0-10%
      },
      riskMetrics: {
        beta: 0.8 + Math.random() * 0.6, // 0.8-1.4
        sharpe: 1.2 + Math.random() * 0.8, // 1.2-2.0
        maxDrawdown: -(Math.random() * 15 + 5), // -5% to -20%
        volatility: 15 + Math.random() * 10 // 15-25%
      }
    }
  }

  if (loading) {
    return (
      <Card className="bg-black/80 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <PieChart className="w-5 h-5 text-emerald-400" />
            <span>Portfolio Summary</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="animate-pulse">
              <div className="w-32 h-8 bg-gray-700 rounded mb-2"></div>
              <div className="w-24 h-6 bg-gray-700 rounded mb-4"></div>
              <div className="grid grid-cols-2 gap-4">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="w-full h-12 bg-gray-700 rounded"></div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!portfolio) return null

  return (
    <Card className="bg-black/80 border-gray-700 hover:border-gray-600 transition-colors">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <PieChart className="w-5 h-5 text-emerald-400" />
            <span>Portfolio Summary</span>
          </div>
          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
            {portfolio.positions} Positions
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Total Portfolio Value */}
        <div className="mb-6">
          <div className="text-3xl font-bold text-white mb-1">
            ₹{(portfolio.totalValue / 100000).toFixed(1)}L
          </div>
          <div className={`flex items-center text-sm ${
            portfolio.dayPL >= 0 ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {portfolio.dayPL >= 0 ? (
              <TrendingUp className="w-4 h-4 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 mr-1" />
            )}
            <span>
              ₹{Math.abs(portfolio.dayPL).toLocaleString()} ({Math.abs(portfolio.dayPLPercent).toFixed(2)}%) Today
            </span>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="p-3 bg-gray-800/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-1">
              <DollarSign className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Day P&L</span>
            </div>
            <div className={`text-lg font-bold ${
              portfolio.dayPL >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              ₹{portfolio.dayPL.toLocaleString()}
            </div>
            <div className={`text-xs ${
              portfolio.dayPL >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {portfolio.dayPLPercent > 0 ? '+' : ''}{portfolio.dayPLPercent.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 bg-gray-800/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-1">
              <Target className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Total P&L</span>
            </div>
            <div className={`text-lg font-bold ${
              portfolio.totalPL >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              ₹{portfolio.totalPL.toLocaleString()}
            </div>
            <div className={`text-xs ${
              portfolio.totalPL >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {portfolio.totalPLPercent > 0 ? '+' : ''}{portfolio.totalPLPercent.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 bg-gray-800/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-1">
              <Activity className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Buying Power</span>
            </div>
            <div className="text-lg font-bold text-white">
              ₹{(portfolio.buyingPower / 100000).toFixed(1)}L
            </div>
            <div className="text-xs text-gray-400">Available</div>
          </div>

          <div className="p-3 bg-gray-800/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-1">
              <AlertTriangle className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-400">Max Drawdown</span>
            </div>
            <div className="text-lg font-bold text-red-400">
              {portfolio.riskMetrics.maxDrawdown.toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Risk Metric</div>
          </div>
        </div>

        {/* Allocation Breakdown */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Asset Allocation</h4>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Equity</span>
              <span className="text-sm font-medium text-emerald-400">
                {portfolio.allocation.equity.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div 
                className="bg-emerald-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${portfolio.allocation.equity}%` }}
              ></div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-300">Cash</span>
              <span className="text-sm font-medium text-cyan-400">
                {portfolio.allocation.cash.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div 
                className="bg-cyan-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${portfolio.allocation.cash}%` }}
              ></div>
            </div>

            {portfolio.allocation.others > 0 && (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Others</span>
                  <span className="text-sm font-medium text-yellow-400">
                    {portfolio.allocation.others.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div 
                    className="bg-yellow-400 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${portfolio.allocation.others}%` }}
                  ></div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="pt-4 border-t border-gray-700">
          <h4 className="text-sm font-medium text-gray-400 mb-3">Risk Analytics</h4>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Beta:</span>
              <span className="text-white font-medium">{portfolio.riskMetrics.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Sharpe:</span>
              <span className="text-emerald-400 font-medium">{portfolio.riskMetrics.sharpe.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility:</span>
              <span className="text-yellow-400 font-medium">{portfolio.riskMetrics.volatility.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Positions:</span>
              <span className="text-white font-medium">{portfolio.positions}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 