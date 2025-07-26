import { Suspense, useEffect, useState } from 'react'
import { AppLayout } from '@/components/Sidebar'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RatingBadge, type RatingType } from '@/components/RatingBadge'
import { IndexTicker } from '@/components/MiniChart'
import { 
  TrendingUp, 
  TrendingDown, 
  Eye, 
  Refresh,
  Clock,
  Activity
} from 'lucide-react'

// Mock data - in production this would come from your API
const topBuySignals = [
  { symbol: 'RELIANCE.NSE', name: 'Reliance Industries', price: 2847.30, change: 42.15, changePercent: 1.5, rating: 'Strong Buy' as RatingType, confidence: 92 },
  { symbol: 'TCS.NSE', name: 'Tata Consultancy Services', price: 4156.80, change: 78.25, changePercent: 1.9, rating: 'Strong Buy' as RatingType, confidence: 89 },
  { symbol: 'INFY.NSE', name: 'Infosys Limited', price: 1789.45, change: 23.60, changePercent: 1.3, rating: 'Buy' as RatingType, confidence: 85 },
  { symbol: 'HDFCBANK.NSE', name: 'HDFC Bank', price: 1654.30, change: 28.90, changePercent: 1.8, rating: 'Strong Buy' as RatingType, confidence: 91 },
  { symbol: 'ICICIBANK.NSE', name: 'ICICI Bank', price: 1234.65, change: 18.45, changePercent: 1.5, rating: 'Buy' as RatingType, confidence: 87 },
]

const topSellSignals = [
  { symbol: 'ITC.NSE', name: 'ITC Limited', price: 456.75, change: -12.35, changePercent: -2.6, rating: 'Sell' as RatingType, confidence: 84 },
  { symbol: 'BHARTIARTL.NSE', name: 'Bharti Airtel', price: 1087.20, change: -24.80, changePercent: -2.2, rating: 'Strong Sell' as RatingType, confidence: 88 },
  { symbol: 'KOTAKBANK.NSE', name: 'Kotak Mahindra Bank', price: 1789.30, change: -31.45, changePercent: -1.7, rating: 'Sell' as RatingType, confidence: 82 },
  { symbol: 'LT.NSE', name: 'Larsen & Toubro', price: 3567.85, change: -45.20, changePercent: -1.3, rating: 'Sell' as RatingType, confidence: 86 },
  { symbol: 'MARUTI.NSE', name: 'Maruti Suzuki India', price: 11234.50, change: -156.75, changePercent: -1.4, rating: 'Strong Sell' as RatingType, confidence: 90 },
]

const indexData = [
  {
    symbol: 'NIFTY 50',
    name: 'Nifty 50 Index',
    price: 24563.75,
    change: 125.40,
    changePercent: 0.51,
    data: Array.from({ length: 20 }, (_, i) => ({
      value: 24500 + Math.random() * 200 - 100,
      timestamp: new Date(Date.now() - (19 - i) * 300000).toISOString()
    }))
  },
  {
    symbol: 'BANK NIFTY',
    name: 'Bank Nifty Index',
    price: 52847.30,
    change: -234.60,
    changePercent: -0.44,
    data: Array.from({ length: 20 }, (_, i) => ({
      value: 52800 + Math.random() * 300 - 150,
      timestamp: new Date(Date.now() - (19 - i) * 300000).toISOString()
    }))
  },
  {
    symbol: 'NIFTY IT',
    name: 'Nifty IT Index',
    price: 40156.85,
    change: 456.20,
    changePercent: 1.15,
    data: Array.from({ length: 20 }, (_, i) => ({
      value: 40100 + Math.random() * 200 - 100,
      timestamp: new Date(Date.now() - (19 - i) * 300000).toISOString()
    }))
  }
]

function getRating(signal: any): RatingType {
  // Map signal and score to rating
  if (signal.signal === 'BUY') {
    if (signal.final_score > 0.5) return 'Strong Buy'
    if (signal.final_score > 0.2) return 'Buy'
    return 'Hold'
  } else if (signal.signal === 'SELL') {
    if (signal.final_score < -0.5) return 'Strong Sell'
    if (signal.final_score < -0.2) return 'Sell'
    return 'Hold'
  }
  return 'Hold'
}

function SignalCard({ signal, type }: { signal: any, type: 'buy' | 'sell' }) {
  const isPositive = (signal.price_target ?? signal.price ?? 0) >= (signal.price ?? 0)
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex-1">
            <h4 className="font-semibold text-sm">{signal.symbol}</h4>
            {/* Optionally show more info here */}
          </div>
          <RatingBadge rating={getRating(signal)} size="sm" />
        </div>
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-lg">₹{(signal.current_price ?? signal.price ?? 0).toLocaleString()}</span>
          <span className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>{isPositive ? '+' : ''}{((signal.price_target ?? 0) - (signal.price ?? 0)).toFixed(2)}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Confidence: {Math.round((signal.confidence ?? 0) * 100)}%</span>
          <Button variant="ghost" size="sm" className="h-6 px-2">
            <Eye className="w-3 h-3 mr-1" />
            View
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function DashboardHeader() {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Trading Dashboard</h1>
          <p className="text-muted-foreground">
            AI-powered signals for NSE stocks • Last updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="secondary" className="flex items-center">
            <Activity className="w-3 h-3 mr-1" />
            Engine Active
          </Badge>
          <Button variant="outline" size="sm">
            <Refresh className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Market Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Market Status</p>
                <p className="text-lg font-semibold text-green-600">Open</p>
              </div>
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Signals Generated</p>
                <p className="text-lg font-semibold">1,247</p>
              </div>
              <TrendingUp className="w-5 h-5 text-blue-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Accuracy Rate</p>
                <p className="text-lg font-semibold">92.4%</p>
              </div>
              <Badge variant="secondary">+2.1%</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const [signals, setSignals] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchSignals() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch('/api/signals')
        const data = await res.json()
        setSignals(data.signals || [])
      } catch (e: any) {
        setError(e.message || 'Failed to fetch signals')
      } finally {
        setLoading(false)
      }
    }
    fetchSignals()
  }, [])

  // Compute top 10 buy and sell signals by final_score
  const buySignals = signals.filter(s => s.signal === 'BUY').sort((a, b) => b.final_score - a.final_score).slice(0, 10)
  const sellSignals = signals.filter(s => s.signal === 'SELL').sort((a, b) => a.final_score - b.final_score).slice(0, 10)

  return (
    <AppLayout>
      <div className="p-6">
        <DashboardHeader />
        {loading ? (
          <div className="text-center py-10">Loading signals...</div>
        ) : error ? (
          <div className="text-center text-red-600 py-10">{error}</div>
        ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Top Buy Signals */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center text-green-600">
                    <TrendingUp className="w-5 h-5 mr-2" />
                    Top 10 Buy Signals
                  </CardTitle>
                  <CardDescription>Highest scoring buy recommendations</CardDescription>
                </div>
                <Badge variant="secondary">{buySignals.length}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {buySignals.map((signal, index) => (
                <SignalCard key={signal.symbol} signal={signal} type="buy" />
              ))}
            </CardContent>
          </Card>

          {/* Top Sell Signals */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center text-red-600">
                    <TrendingDown className="w-5 h-5 mr-2" />
                    Top 10 Sell Signals
                  </CardTitle>
                  <CardDescription>Highest scoring sell recommendations</CardDescription>
                </div>
                <Badge variant="secondary">{sellSignals.length}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {sellSignals.map((signal, index) => (
                <SignalCard key={signal.symbol} signal={signal} type="sell" />
              ))}
            </CardContent>
          </Card>

          {/* Index Tickers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Clock className="w-5 h-5 mr-2" />
                Market Indices
              </CardTitle>
              <CardDescription>Real-time index movements with sparklines</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {indexData.map((index) => (
                <IndexTicker
                  key={index.symbol}
                  symbol={index.symbol}
                  name={index.name}
                  price={index.price}
                  change={index.change}
                  changePercent={index.changePercent}
                  data={index.data}
                />
              ))}
            </CardContent>
          </Card>
        </div>
        )}
        {/* TODO: Add more sections */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg border-2 border-dashed">
          <h3 className="font-semibold mb-2">TODO: Additional Dashboard Features</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Portfolio performance charts</li>
            <li>• Recent news affecting top signals</li>
            <li>• Risk assessment summary</li>
            <li>• Quick action buttons for order placement</li>
            <li>• WebSocket integration for real-time updates</li>
          </ul>
        </div>
      </div>
    </AppLayout>
  )
} 