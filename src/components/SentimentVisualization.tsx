'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { SentimentVisualizationProps } from '@/lib/types';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  ArrowUp,
  ArrowDown,
  Activity,
  Clock,
  BarChart3
} from 'lucide-react';

export function SentimentVisualization({ 
  trends, 
  topMovers, 
  sectorSentiments, 
  selectedTicker,
  selectedSector,
  timeframe 
}: SentimentVisualizationProps) {
  const [activeTab, setActiveTab] = useState<'trends' | 'movers' | 'sectors'>('trends');

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.3) return 'text-green-600 bg-green-100';
    if (sentiment < -0.3) return 'text-red-600 bg-red-100';
    return 'text-yellow-600 bg-yellow-100';
  };

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.1) return <TrendingUp className="w-4 h-4" />;
    if (sentiment < -0.1) return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getChangeIcon = (change: number) => {
    return change > 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />;
  };

  const getTrendArrow = (trend: 'increasing' | 'decreasing' | 'stable') => {
    if (trend === 'increasing') return <ArrowUp className="w-3 h-3 text-green-500" />;
    if (trend === 'decreasing') return <ArrowDown className="w-3 h-3 text-red-500" />;
    return <Minus className="w-3 h-3 text-gray-500" />;
  };

  // Simple sparkline component
  const Sparkline = ({ data, color = 'blue' }: { data: number[], color?: string }) => {
    if (data.length === 0) return null;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="w-16 h-8" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline
          fill="none"
          stroke={color === 'green' ? '#10b981' : color === 'red' ? '#ef4444' : '#3b82f6'}
          strokeWidth="2"
          points={points}
        />
        <circle
          cx={((data.length - 1) / (data.length - 1)) * 100}
          cy={100 - ((data[data.length - 1] - min) / range) * 100}
          r="2"
          fill={color === 'green' ? '#10b981' : color === 'red' ? '#ef4444' : '#3b82f6'}
        />
      </svg>
    );
  };

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
        <Button
          variant={activeTab === 'trends' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('trends')}
          className="flex-1"
        >
          <Activity className="w-4 h-4 mr-2" />
          Sentiment Trends
        </Button>
        <Button
          variant={activeTab === 'movers' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('movers')}
          className="flex-1"
        >
          <TrendingUp className="w-4 h-4 mr-2" />
          Top Movers
        </Button>
        <Button
          variant={activeTab === 'sectors' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setActiveTab('sectors')}
          className="flex-1"
        >
          <BarChart3 className="w-4 h-4 mr-2" />
          Sector Analysis
        </Button>
      </div>

      {/* Sentiment Trends */}
      {activeTab === 'trends' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Sentiment Trends
              {(selectedTicker || selectedSector) && (
                <Badge variant="outline" className="ml-2">
                  {selectedTicker || selectedSector}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trends.length > 0 ? (
              <div className="space-y-4">
                {/* Overall trend line */}
                <div className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      {selectedTicker ? `${selectedTicker} Sentiment` : 
                       selectedSector ? `${selectedSector} Sector` : 
                       'Overall Market Sentiment'}
                    </span>
                    <Badge className={getSentimentColor(trends[trends.length - 1]?.sentiment || 0)}>
                      {((trends[trends.length - 1]?.sentiment || 0) * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  
                  {/* Sparkline visualization */}
                  <div className="flex items-center space-x-4">
                    <Sparkline 
                      data={trends.map(t => t.sentiment)} 
                      color={trends[trends.length - 1]?.sentiment > 0 ? 'green' : 
                             trends[trends.length - 1]?.sentiment < 0 ? 'red' : 'blue'}
                    />
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {trends.length} data points ({timeframe})
                    </div>
                  </div>
                </div>

                {/* Recent data points */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Recent Activity
                  </h4>
                  {trends.slice(-5).reverse().map((trend, index) => (
                    <div key={index} className="flex items-center justify-between p-2 rounded border">
                      <div className="flex items-center space-x-2">
                        {getSentimentIcon(trend.sentiment)}
                        <span className="text-sm">
                          {new Date(trend.date).toLocaleString()}
                        </span>
                        <Badge variant="outline" className="text-xs">
                          {trend.articleCount} articles
                        </Badge>
                      </div>
                      <Badge className={getSentimentColor(trend.sentiment)}>
                        {(trend.sentiment * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No sentiment data available</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Top Movers */}
      {activeTab === 'movers' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              Top Sentiment Movers
            </CardTitle>
          </CardHeader>
          <CardContent>
            {topMovers.length > 0 ? (
              <div className="space-y-3">
                {topMovers.map((mover, index) => (
                  <div 
                    key={mover.ticker} 
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-6 h-6 bg-gray-100 dark:bg-gray-800 rounded text-xs font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium font-mono text-sm">
                          {mover.ticker.replace('.NSE', '')}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {mover.name}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <div className="text-right">
                        <div className="flex items-center space-x-1">
                          {getChangeIcon(mover.sentimentChange)}
                          <span className={`text-sm font-medium ${
                            mover.sentimentChange > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {Math.abs(mover.sentimentChange * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {mover.timeframe} • {mover.articleCount} articles
                        </div>
                      </div>
                      
                      <Badge className={getSentimentColor(mover.currentSentiment)}>
                        {(mover.currentSentiment * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No sentiment movers available</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Sector Analysis */}
      {activeTab === 'sectors' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Sector Sentiment Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            {sectorSentiments.length > 0 ? (
              <div className="space-y-3">
                {sectorSentiments.map((sector) => (
                  <div 
                    key={sector.sector}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        {getSentimentIcon(sector.sentiment)}
                        <div>
                          <div className="font-medium">{sector.sector}</div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            {sector.articleCount} articles • {(sector.confidence * 100).toFixed(0)}% confidence
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-1">
                        {getTrendArrow(sector.trend)}
                        <span className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                          {sector.trend}
                        </span>
                      </div>
                      
                      <Badge className={getSentimentColor(sector.sentiment)}>
                        {(sector.sentiment * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                ))}

                {/* Top Tickers per Sector */}
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                    Key Stocks by Sector
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {sectorSentiments.slice(0, 6).map((sector) => (
                      <div key={sector.sector} className="p-3 bg-gray-50 dark:bg-gray-800/30 rounded-lg">
                        <div className="text-sm font-medium mb-2 flex items-center justify-between">
                          {sector.sector}
                          <Badge variant="outline" className="text-xs">
                            {sector.topTickers.length}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {sector.topTickers.slice(0, 4).map((ticker) => (
                            <Badge 
                              key={ticker}
                              variant="secondary" 
                              className="text-xs font-mono"
                            >
                              {ticker.replace('.NSE', '')}
                            </Badge>
                          ))}
                          {sector.topTickers.length > 4 && (
                            <Badge variant="outline" className="text-xs">
                              +{sector.topTickers.length - 4}
                            </Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No sector sentiment data available</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
} 