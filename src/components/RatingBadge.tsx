'use client'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export type RatingType = 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell'

interface RatingBadgeProps {
  rating: RatingType
  size?: 'sm' | 'md' | 'lg'
  showIcon?: boolean
  className?: string
}

const ratingConfig = {
  'Strong Buy': {
    color: 'bg-green-600 text-white hover:bg-green-700',
    icon: TrendingUp,
    textColor: 'text-green-600'
  },
  'Buy': {
    color: 'bg-green-500 text-white hover:bg-green-600',
    icon: TrendingUp,
    textColor: 'text-green-500'
  },
  'Hold': {
    color: 'bg-yellow-500 text-white hover:bg-yellow-600',
    icon: Minus,
    textColor: 'text-yellow-600'
  },
  'Sell': {
    color: 'bg-red-500 text-white hover:bg-red-600',
    icon: TrendingDown,
    textColor: 'text-red-500'
  },
  'Strong Sell': {
    color: 'bg-red-600 text-white hover:bg-red-700',
    icon: TrendingDown,
    textColor: 'text-red-600'
  }
}

export function RatingBadge({ 
  rating, 
  size = 'md', 
  showIcon = false, 
  className 
}: RatingBadgeProps) {
  const config = ratingConfig[rating]
  const Icon = config.icon

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2 font-semibold'
  }

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  }

  return (
    <Badge
      className={cn(
        config.color,
        sizeClasses[size],
        'font-medium',
        className
      )}
    >
      {showIcon && (
        <Icon className={cn(iconSizes[size], 'mr-1')} />
      )}
      {rating}
    </Badge>
  )
}

// Large rating display for stock detail pages
interface RatingDisplayProps {
  rating: RatingType
  confidence?: number
  lastUpdated?: string
  className?: string
}

export function RatingDisplay({ 
  rating, 
  confidence, 
  lastUpdated, 
  className 
}: RatingDisplayProps) {
  const config = ratingConfig[rating]
  const Icon = config.icon

  return (
    <div className={cn("text-center p-6 bg-card rounded-lg border", className)}>
      <div className="flex items-center justify-center mb-2">
        <Icon className={cn("w-8 h-8 mr-2", config.textColor)} />
        <h1 className={cn("text-3xl font-bold", config.textColor)}>
          {rating}
        </h1>
      </div>
      
      {confidence !== undefined && (
        <p className="text-sm text-muted-foreground mb-1">
          Confidence: {confidence}%
        </p>
      )}
      
      {lastUpdated && (
        <p className="text-xs text-muted-foreground">
          Last updated: {new Date(lastUpdated).toLocaleString()}
        </p>
      )}
    </div>
  )
}

// Rating trend indicator
interface RatingTrendProps {
  currentRating: RatingType
  previousRating?: RatingType
  className?: string
}

export function RatingTrend({ 
  currentRating, 
  previousRating, 
  className 
}: RatingTrendProps) {
  if (!previousRating) {
    return <RatingBadge rating={currentRating} className={className} />
  }

  const ratings = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
  const currentIndex = ratings.indexOf(currentRating)
  const previousIndex = ratings.indexOf(previousRating)
  
  const isUpgrade = currentIndex > previousIndex
  const isDowngrade = currentIndex < previousIndex
  
  return (
    <div className={cn("flex items-center space-x-2", className)}>
      <RatingBadge rating={currentRating} />
      {isUpgrade && (
        <div className="flex items-center text-green-600 text-xs">
          <TrendingUp className="w-3 h-3 mr-1" />
          Upgraded
        </div>
      )}
      {isDowngrade && (
        <div className="flex items-center text-red-600 text-xs">
          <TrendingDown className="w-3 h-3 mr-1" />
          Downgraded
        </div>
      )}
    </div>
  )
} 