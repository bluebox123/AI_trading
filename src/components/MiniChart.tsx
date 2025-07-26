'use client'

import { LineChart, Line, ResponsiveContainer } from 'recharts'
import { cn } from '@/lib/utils'

interface MiniChartProps {
  data: Array<{ value: number; timestamp: string }>
  color?: string
  className?: string
  height?: number
}

export function MiniChart({ 
  data, 
  color = "#10b981", 
  className,
  height = 40 
}: MiniChartProps) {
  // Determine if the trend is positive or negative
  const firstValue = data[0]?.value || 0
  const lastValue = data[data.length - 1]?.value || 0
  const isPositive = lastValue >= firstValue

  const chartColor = color === "auto" ? (isPositive ? "#10b981" : "#ef4444") : color

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={chartColor}
            strokeWidth={1.5}
            dot={false}
            activeDot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Index ticker component that includes the mini chart
interface IndexTickerProps {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  data: Array<{ value: number; timestamp: string }>
}

export function IndexTicker({ 
  symbol, 
  name, 
  price, 
  change, 
  changePercent, 
  data 
}: IndexTickerProps) {
  const isPositive = change >= 0

  return (
    <div className="p-4 bg-card rounded-lg border">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="font-semibold text-sm">{symbol}</h3>
          <p className="text-xs text-muted-foreground">{name}</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-sm">₹{price.toLocaleString()}</p>
          <p className={cn(
            "text-xs font-medium",
            isPositive ? "text-green-600" : "text-red-600"
          )}>
            {isPositive ? "+" : ""}{change.toFixed(2)} ({changePercent.toFixed(2)}%)
          </p>
        </div>
      </div>
      <MiniChart 
        data={data} 
        color="auto"
        height={32}
      />
    </div>
  )
}

// Stock sparkline for the stocks list page
interface StockSparklineProps {
  data: Array<{ value: number; timestamp: string }>
  change: number
  className?: string
}

export function StockSparkline({ data, change, className }: StockSparklineProps) {
  const isPositive = change >= 0

  return (
    <MiniChart 
      data={data}
      color={isPositive ? "#10b981" : "#ef4444"}
      className={className}
      height={24}
    />
  )
} 