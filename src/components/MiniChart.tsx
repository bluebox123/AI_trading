'use client'

import { useEffect, useState } from 'react'

interface MiniChartProps {
  data?: number[]
  className?: string
  color?: 'green' | 'red' | 'blue' | 'yellow'
  height?: number
}

export default function MiniChart({ 
  data = [], 
  className = '', 
  color = 'green',
  height = 40 
}: MiniChartProps) {
  const [animatedData, setAnimatedData] = useState<number[]>([])

  // Generate mock data if none provided
  const chartData = data.length > 0 ? data : generateMockData()

  useEffect(() => {
    // Animate the chart on mount
    const timer = setTimeout(() => {
      setAnimatedData(chartData)
    }, 100)

    return () => clearTimeout(timer)
  }, [chartData])

  function generateMockData(): number[] {
    const points = 20 + Math.floor(Math.random() * 10)
    const baseValue = 100 + Math.random() * 50
    const volatility = 5 + Math.random() * 10
    
    return Array.from({ length: points }, (_, i) => {
      const trend = Math.sin(i * 0.3) * 2
      const noise = (Math.random() - 0.5) * volatility
      return baseValue + trend + noise
    })
  }

  const colorMap = {
    green: 'stroke-emerald-400',
    red: 'stroke-red-400',
    blue: 'stroke-cyan-400',
    yellow: 'stroke-yellow-400'
  }

  const gradientMap = {
    green: 'fill-emerald-400/20',
    red: 'fill-red-400/20',
    blue: 'fill-cyan-400/20',
    yellow: 'fill-yellow-400/20'
  }

  if (animatedData.length === 0) {
    return (
      <div className={`w-full animate-pulse ${className}`} style={{ height }}>
        <div className="w-full h-full bg-gray-700/30 rounded"></div>
      </div>
    )
  }

  const min = Math.min(...animatedData)
  const max = Math.max(...animatedData)
  const range = max - min || 1

  const points = animatedData.map((value, index) => {
    const x = (index / (animatedData.length - 1)) * 100
    const y = ((max - value) / range) * (height - 8) + 4
    return `${x},${y}`
  }).join(' ')

  const pathData = animatedData.map((value, index) => {
    const x = (index / (animatedData.length - 1)) * 100
    const y = ((max - value) / range) * (height - 8) + 4
    return index === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
  }).join(' ')

  const areaPath = `${pathData} L 100 ${height} L 0 ${height} Z`

  return (
    <div className={`w-full ${className}`} style={{ height }}>
      <svg 
        width="100%" 
        height="100%" 
        viewBox={`0 0 100 ${height}`} 
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        {/* Gradient definition */}
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" className={gradientMap[color]} />
            <stop offset="100%" style={{ stopColor: 'transparent' }} />
          </linearGradient>
        </defs>
        
        {/* Area fill */}
        <path
          d={areaPath}
          fill={`url(#gradient-${color})`}
          className="opacity-60"
        />
        
        {/* Line */}
        <path
          d={pathData}
          fill="none"
          strokeWidth="1.5"
          className={`${colorMap[color]} transition-all duration-500`}
          style={{
            filter: `drop-shadow(0 0 4px ${color === 'green' ? '#10b981' : color === 'red' ? '#ef4444' : color === 'blue' ? '#06b6d4' : '#eab308'})`
          }}
        />
        
        {/* Animated dots on line */}
        {animatedData.slice(-3).map((value, index) => {
          const x = ((animatedData.length - 3 + index) / (animatedData.length - 1)) * 100
          const y = ((max - value) / range) * (height - 8) + 4
          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r="1"
              className={colorMap[color]}
              fill="currentColor"
              opacity={0.4 + index * 0.3}
            >
              <animate
                attributeName="opacity"
                values="0.4;1;0.4"
                dur="2s"
                repeatCount="indefinite"
                begin={`${index * 0.5}s`}
              />
            </circle>
          )
        })}
      </svg>
    </div>
  )
} 