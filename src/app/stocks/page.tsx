import Link from 'next/link'
import { AppLayout } from '@/components/Sidebar'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RatingBadge, type RatingType } from '@/components/RatingBadge'
import { StockSparkline } from '@/components/MiniChart'
import { 
  Search, 
  Eye, 
  ArrowUpDown,
  Filter
} from 'lucide-react'

// Mock stock data for 100 stocks
const generateMockStocks = () => {
  const stockSymbols = [
    'RELIANCE.NSE', 'TCS.NSE', 'HDFCBANK.NSE', 'INFY.NSE', 'ICICIBANK.NSE',
    'KOTAKBANK.NSE', 'HINDUNILVR.NSE', 'ITC.NSE', 'LT.NSE', 'SBIN.NSE',
    'BHARTIARTL.NSE', 'ASIANPAINT.NSE', 'MARUTI.NSE', 'AXISBANK.NSE', 'BAJFINANCE.NSE'
    // ... would have 100 stocks in real implementation
  ]
  
  const ratings: RatingType[] = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
  
  return stockSymbols.map((symbol, index) => {
    const price = 1000 + Math.random() * 4000
    const change = (Math.random() - 0.5) * 200
    const changePercent = (change / price) * 100
    
    return {
      symbol,
      name: symbol.replace('.NSE', ' Limited'),
      price,
      change,
      changePercent,
      volume: Math.floor(Math.random() * 1000000),
      marketCap: Math.floor(Math.random() * 500000),
      rating: ratings[Math.floor(Math.random() * ratings.length)],
      confidence: 70 + Math.floor(Math.random() * 30),
      data: Array.from({ length: 20 }, (_, i) => ({
        value: price + (Math.random() - 0.5) * 100,
        timestamp: new Date(Date.now() - (19 - i) * 300000).toISOString()
      }))
    }
  })
}

const mockStocks = generateMockStocks()

function StockRow({ stock }: { stock: typeof mockStocks[0] }) {
  const isPositive = stock.change >= 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-7 gap-4 p-4 border-b hover:bg-muted/50 transition-colors">
      {/* Symbol & Name */}
      <div className="md:col-span-2">
        <div className="flex items-center space-x-3">
          <div>
            <h4 className="font-semibold">{stock.symbol}</h4>
            <p className="text-sm text-muted-foreground truncate">{stock.name}</p>
          </div>
        </div>
      </div>

      {/* Price */}
      <div className="text-right md:text-left">
        <p className="font-mono font-semibold">₹{stock.price.toFixed(2)}</p>
        <p className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {isPositive ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent.toFixed(2)}%)
        </p>
      </div>

      {/* Rating */}
      <div className="flex justify-center">
        <RatingBadge rating={stock.rating} size="sm" />
      </div>

      {/* Confidence */}
      <div className="text-center">
        <p className="font-medium">{stock.confidence}%</p>
      </div>

      {/* Sparkline */}
      <div className="flex items-center justify-center">
        <div className="w-20 h-8">
          <StockSparkline data={stock.data} change={stock.change} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end">
        <Link href={`/stocks/${stock.symbol}`}>
          <Button variant="ghost" size="sm">
            <Eye className="w-4 h-4 mr-2" />
            View
          </Button>
        </Link>
      </div>
    </div>
  )
}

export default function StocksPage() {
  return (
    <AppLayout>
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Stock Analysis</h1>
          <p className="text-muted-foreground">
            Complete watch-list of 100 NSE largecap and midcap stocks with AI-powered ratings
          </p>
        </div>

        {/* Filters & Search */}
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex flex-col lg:flex-row gap-4">
              {/* Search */}
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input 
                    placeholder="Search stocks by symbol or name..." 
                    className="pl-10"
                  />
                </div>
              </div>

              {/* Filters */}
              <div className="flex gap-2">
                <Button variant="outline" size="sm">Strong Buy</Button>
                <Button variant="outline" size="sm">Buy</Button>
                <Button variant="outline" size="sm">Hold</Button>
                <Button variant="outline" size="sm">Sell</Button>
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-2" />
                  More Filters
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stock Table */}
        <Card>
          {/* Table Header */}
          <div className="grid grid-cols-1 md:grid-cols-7 gap-4 p-4 border-b bg-muted/50 font-semibold text-sm">
            <div className="md:col-span-2 flex items-center">
              <span>Stock</span>
              <ArrowUpDown className="w-3 h-3 ml-1" />
            </div>
            <div className="text-right md:text-left flex items-center">
              <span>Price</span>
              <ArrowUpDown className="w-3 h-3 ml-1" />
            </div>
            <div className="text-center">Rating</div>
            <div className="text-center">Confidence</div>
            <div className="text-center">Trend</div>
            <div className="text-right">Actions</div>
          </div>

          {/* Stock Rows - In a real implementation, this would be virtualized */}
          <div className="max-h-[600px] overflow-y-auto">
            {mockStocks.map((stock) => (
              <StockRow key={stock.symbol} stock={stock} />
            ))}
          </div>
        </Card>

        {/* TODO Section */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg border-2 border-dashed">
          <h3 className="font-semibold mb-2">TODO: Stocks Page Implementation</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Implement react-window for virtualized table performance</li>
            <li>• Add sorting by price, rating, confidence, change</li>
            <li>• Implement advanced filtering by sector, market cap, etc.</li>
            <li>• Add real-time price updates via WebSocket</li>
            <li>• Connect to EODHD API for live market data</li>
            <li>• Add watchlist functionality</li>
            <li>• Implement bulk actions for multiple stocks</li>
            <li>• Add export functionality (CSV, PDF)</li>
          </ul>
        </div>
      </div>
    </AppLayout>
  )
} 