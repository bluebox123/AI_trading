import { AppLayout } from '@/components/Sidebar'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Search, 
  Clock, 
  ExternalLink,
  Filter
} from 'lucide-react'

// Mock news data
const mockNews = [
  {
    id: 1,
    headline: "RBI Keeps Repo Rate Unchanged at 6.5%, Maintains Hawkish Stance",
    source: "Economic Times",
    publishedAt: "2 hours ago",
    sentiment: "neutral" as const,
    affectedSymbols: ["HDFCBANK.NSE", "ICICIBANK.NSE", "KOTAKBANK.NSE"],
    sectors: ["Banking", "Finance"],
    url: "#"
  },
  {
    id: 2,
    headline: "TCS Reports Strong Q3 Results, Revenue Up 16.8% YoY",
    source: "Business Standard",
    publishedAt: "4 hours ago",
    sentiment: "positive" as const,
    affectedSymbols: ["TCS.NSE", "INFY.NSE", "WIPRO.NSE"],
    sectors: ["IT", "Technology"],
    url: "#"
  },
  {
    id: 3,
    headline: "Oil Prices Surge on Middle East Tensions, Reliance Gains",
    source: "Reuters",
    publishedAt: "6 hours ago",
    sentiment: "positive" as const,
    affectedSymbols: ["RELIANCE.NSE", "ONGC.NSE"],
    sectors: ["Energy", "Oil & Gas"],
    url: "#"
  }
]

const sentimentConfig = {
  positive: { color: 'text-green-600', bg: 'bg-green-100', label: 'Positive' },
  negative: { color: 'text-red-600', bg: 'bg-red-100', label: 'Negative' },
  neutral: { color: 'text-yellow-600', bg: 'bg-yellow-100', label: 'Neutral' }
}

function NewsCard({ article }: { article: typeof mockNews[0] }) {
  const sentiment = sentimentConfig[article.sentiment]

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-lg font-semibold leading-tight flex-1 mr-4">
            {article.headline}
          </h3>
          <Button variant="ghost" size="sm">
            <ExternalLink className="w-4 h-4" />
          </Button>
        </div>
        
        <div className="flex items-center text-sm text-muted-foreground mb-3">
          <span>{article.source}</span>
          <span className="mx-2">•</span>
          <Clock className="w-3 h-3 mr-1" />
          <span>{article.publishedAt}</span>
          <span className="mx-2">•</span>
          <Badge 
            variant="secondary" 
            className={`${sentiment.color} ${sentiment.bg}`}
          >
            {sentiment.label}
          </Badge>
        </div>

        <div className="flex flex-wrap gap-2 mb-3">
          {article.sectors.map((sector) => (
            <Badge key={sector} variant="outline" className="text-xs">
              {sector}
            </Badge>
          ))}
        </div>

        <div className="text-sm">
          <span className="text-muted-foreground">Affected stocks: </span>
          {article.affectedSymbols.map((symbol, index) => (
            <span key={symbol} className="font-medium">
              {symbol}
              {index < article.affectedSymbols.length - 1 && ', '}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default function NewsPage() {
  return (
    <AppLayout>
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Market News</h1>
          <p className="text-muted-foreground">
            Real-time news analysis with sentiment scoring and stock impact assessment
          </p>
        </div>

        {/* Filters */}
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex flex-col lg:flex-row gap-4">
              {/* Search */}
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input 
                    placeholder="Search news articles..." 
                    className="pl-10"
                  />
                </div>
              </div>

              {/* Quick Filter Pills */}
              <div className="flex gap-2">
                <Button variant="outline" size="sm">1 Month</Button>
                <Button variant="outline" size="sm">1 Week</Button>
                <Button variant="outline" size="sm">3 Days</Button>
                <Button variant="default" size="sm">Intraday</Button>
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-2" />
                  More Filters
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* News Articles */}
        <div className="space-y-4">
          {mockNews.map((article) => (
            <NewsCard key={article.id} article={article} />
          ))}
        </div>

        {/* TODO Section */}
        <div className="mt-8 p-4 bg-muted/50 rounded-lg border-2 border-dashed">
          <h3 className="font-semibold mb-2">TODO: News Page Implementation</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Connect to NewsFilter.io API for real-time news</li>
            <li>• Implement sentiment analysis with Arya Financial Sentiment API</li>
            <li>• Add advanced filtering by sector, sentiment, and date range</li>
            <li>• Implement infinite scroll or pagination</li>
            <li>• Add news impact scoring and alerts</li>
            <li>• Integrate with stock symbols for quick navigation</li>
            <li>• Add export functionality for news analysis</li>
          </ul>
        </div>
      </div>
    </AppLayout>
  )
} 