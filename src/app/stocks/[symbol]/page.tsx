import { AppLayout } from '@/components/Sidebar'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { RatingDisplay, type RatingType } from '@/components/RatingBadge'
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Newspaper,
  Users,
  AlertCircle,
  ArrowLeft
} from 'lucide-react'

// Mock data for a specific stock
const mockStockData = {
  symbol: 'RELIANCE.NSE',
  name: 'Reliance Industries Limited',
  price: 2847.30,
  change: 42.15,
  changePercent: 1.5,
  volume: 1234567,
  marketCap: 1925000,
  rating: 'Strong Buy' as RatingType,
  confidence: 92,
  lastUpdated: new Date().toISOString(),
  
  technicalIndicators: {
    rsi: 68.5,
    macd: {
      macd: 12.45,
      signal: 8.32,
      histogram: 4.13
    },
    sma50: 2756.20,
    sma200: 2650.80,
    atr: 89.45
  },
  
  marketRegime: {
    regime: 'Bull' as const,
    confidence: 85,
    description: 'Strong bullish momentum with high volume support'
  },
  
  newsSentiment: {
    score: 0.73,
    label: 'Positive',
    articlesCount: 12,
    lastUpdate: '2 hours ago'
  },
  
  fundamentals: {
    pe: 24.5,
    pb: 2.1,
    dividendYield: 0.65,
    eps: 116.2,
    bookValue: 1354.7
  }
}

function TechnicalIndicatorCard({ title, value, description, status }: {
  title: string
  value: string | number
  description: string
  status: 'bullish' | 'bearish' | 'neutral'
}) {
  const statusConfig = {
    bullish: { color: 'text-green-600', bg: 'bg-green-100' },
    bearish: { color: 'text-red-600', bg: 'bg-red-100' },
    neutral: { color: 'text-yellow-600', bg: 'bg-yellow-100' }
  }
  
  const config = statusConfig[status]

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-semibold text-sm">{title}</h4>
          <Badge variant="secondary" className={`${config.color} ${config.bg} text-xs`}>
            {status}
          </Badge>
        </div>
        <div className="font-mono text-lg font-bold mb-1">{value}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  )
}

export default function StockDetailPage({ params }: { params: { symbol: string } }) {
  const isPositive = mockStockData.change >= 0

  return (
    <AppLayout>
      <div className="p-6">
        {/* Back Navigation */}
        <div className="mb-6">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Stocks
          </Button>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-1">{mockStockData.symbol}</h1>
              <p className="text-muted-foreground">{mockStockData.name}</p>
            </div>
            
            <div className="text-right">
              <div className="font-mono text-2xl font-bold">₹{mockStockData.price.toLocaleString()}</div>
              <div className={`text-lg font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {isPositive ? '+' : ''}{mockStockData.change.toFixed(2)} ({mockStockData.changePercent.toFixed(2)}%)
              </div>
            </div>
          </div>
        </div>

        {/* Rating Banner */}
        <div className="mb-6">
          <RatingDisplay
            rating={mockStockData.rating}
            confidence={mockStockData.confidence}
            lastUpdated={mockStockData.lastUpdated}
          />
        </div>

        {/* Information Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Technical Indicators */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Technical Indicators
              </CardTitle>
              <CardDescription>Key technical analysis metrics</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <TechnicalIndicatorCard
                  title="RSI (14)"
                  value={mockStockData.technicalIndicators.rsi}
                  description="Relative Strength Index"
                  status="neutral"
                />
                <TechnicalIndicatorCard
                  title="MACD"
                  value={mockStockData.technicalIndicators.macd.macd.toFixed(2)}
                  description="Moving Average Convergence"
                  status="bullish"
                />
                <TechnicalIndicatorCard
                  title="SMA 50"
                  value={`₹${mockStockData.technicalIndicators.sma50.toFixed(2)}`}
                  description="50-day Simple Moving Average"
                  status="bullish"
                />
                <TechnicalIndicatorCard
                  title="ATR"
                  value={mockStockData.technicalIndicators.atr.toFixed(2)}
                  description="Average True Range"
                  status="neutral"
                />
              </div>
            </CardContent>
          </Card>

          {/* Market Regime & Sentiment */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Newspaper className="w-5 h-5 mr-2" />
                Market Context
              </CardTitle>
              <CardDescription>Regime detection and sentiment analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Market Regime */}
              <div className="p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">Market Regime</h4>
                  <Badge variant="secondary" className="text-green-600 bg-green-100">
                    {mockStockData.marketRegime.regime}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mb-2">
                  {mockStockData.marketRegime.description}
                </p>
                <p className="text-xs text-muted-foreground">
                  Confidence: {mockStockData.marketRegime.confidence}%
                </p>
              </div>

              {/* News Sentiment */}
              <div className="p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">News Sentiment (7d)</h4>
                  <Badge variant="secondary" className="text-green-600 bg-green-100">
                    {mockStockData.newsSentiment.label}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mb-2">
                  Score: {mockStockData.newsSentiment.score} ({mockStockData.newsSentiment.articlesCount} articles)
                </p>
                <p className="text-xs text-muted-foreground">
                  Last updated: {mockStockData.newsSentiment.lastUpdate}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Expandable Sections */}
        <Card>
          <CardHeader>
            <CardTitle>Detailed Analysis</CardTitle>
            <CardDescription>Comprehensive stock analysis and data</CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="multiple" className="w-full">
              <AccordionItem value="fundamentals">
                <AccordionTrigger>Fundamental Analysis</AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="font-bold text-lg">{mockStockData.fundamentals.pe}</div>
                      <div className="text-sm text-muted-foreground">P/E Ratio</div>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="font-bold text-lg">{mockStockData.fundamentals.pb}</div>
                      <div className="text-sm text-muted-foreground">P/B Ratio</div>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="font-bold text-lg">{mockStockData.fundamentals.eps}</div>
                      <div className="text-sm text-muted-foreground">EPS</div>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="font-bold text-lg">{mockStockData.fundamentals.dividendYield}%</div>
                      <div className="text-sm text-muted-foreground">Dividend Yield</div>
                    </div>
                    <div className="text-center p-3 bg-muted rounded-lg">
                      <div className="font-bold text-lg">₹{mockStockData.fundamentals.bookValue}</div>
                      <div className="text-sm text-muted-foreground">Book Value</div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="technical-details">
                <AccordionTrigger>Technical Indicator Series</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-semibold mb-2">MACD Details</h4>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>MACD: {mockStockData.technicalIndicators.macd.macd}</div>
                        <div>Signal: {mockStockData.technicalIndicators.macd.signal}</div>
                        <div>Histogram: {mockStockData.technicalIndicators.macd.histogram}</div>
                      </div>
                    </div>
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-semibold mb-2">Moving Averages</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>SMA 50: ₹{mockStockData.technicalIndicators.sma50}</div>
                        <div>SMA 200: ₹{mockStockData.technicalIndicators.sma200}</div>
                      </div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="insider-trades">
                <AccordionTrigger>Recent Insider Trades</AccordionTrigger>
                <AccordionContent>
                  <div className="text-center py-8 text-muted-foreground">
                    <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Insider trading data will be displayed here</p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>

        {/* TODO Section */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg border-2 border-dashed">
          <h3 className="font-semibold mb-2">TODO: Stock Detail Implementation</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Connect to real-time stock data APIs</li>
            <li>• Implement interactive price charts (TradingView)</li>
            <li>• Add model prediction confidence intervals</li>
            <li>• Implement real-time technical indicator calculations</li>
            <li>• Add historical performance comparison</li>
            <li>• Integrate with news API for stock-specific articles</li>
            <li>• Add order placement functionality</li>
            <li>• Implement alerts and notifications</li>
          </ul>
        </div>
      </div>
    </AppLayout>
  )
} 