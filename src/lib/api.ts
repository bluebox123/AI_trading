import { supabase } from './supabase-client'

// Base API configuration
const API_CONFIG = {
  EODHD_BASE_URL: process.env.EODHD_BASE_URL || 'https://eodhd.com/api',
  EODHD_API_KEY: process.env.EODHD_API_KEY,
  NEWS_API_KEY: process.env.NEWS_API_KEY,
  ALPHA_VANTAGE_API_KEY: process.env.ALPHA_VANTAGE_API_KEY,
  PERPLEXITY_API_KEY: process.env.PERPLEXITY_API_KEY,
}

// API Response Types
export interface StockQuote {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap: number
}

export interface TechnicalIndicators {
  rsi: number
  macd: {
    macd: number
    signal: number
    histogram: number
  }
  sma50: number
  sma200: number
  atr: number
}

export interface ModelScore {
  symbol: string
  score: number
  rating: 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell'
  confidence: number
  lastUpdated: string
}

export interface MarketRegime {
  regime: 'Bull' | 'Bear' | 'Sideways'
  confidence: number
  description: string
}

// Internal API functions (Next.js API routes)
export const fetchRankings = async (type: 'buy' | 'sell'): Promise<any[]> => {
  const response = await fetch(`/api/rankings?type=${type}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${type} rankings`)
  }
  return response.json()
}

export const fetchStockDetail = async (symbol: string): Promise<{
  quote: StockQuote
  indicators: TechnicalIndicators
  modelScore: ModelScore
  regime: MarketRegime
}> => {
  const response = await fetch(`/api/stocks/${symbol}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch stock detail for ${symbol}`)
  }
  return response.json()
}

export const fetchNews = async (params: {
  dateRange?: string
  search?: string
  limit?: number
}): Promise<any[]> => {
  const searchParams = new URLSearchParams(params as any)
  const response = await fetch(`/api/news?${searchParams}`)
  if (!response.ok) {
    throw new Error('Failed to fetch news')
  }
  return response.json()
}

export const fetchEngineStatus = async (): Promise<{
  status: 'active' | 'idle' | 'error'
  lastUpdate: string
  modelsLoaded: number
  signalsGenerated: number
}> => {
  const response = await fetch('/api/engine/status')
  if (!response.ok) {
    throw new Error('Failed to fetch engine status')
  }
  return response.json()
}

// OTP API functions
export const sendOTP = async (phone: string): Promise<{ success: boolean; message: string }> => {
  const response = await fetch('/api/otp/send', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone }),
  })
  
  if (!response.ok) {
    throw new Error('Failed to send OTP')
  }
  
  return response.json()
}

export const verifyOTP = async (phone: string, code: string): Promise<{ verified: boolean }> => {
  const response = await fetch('/api/otp/verify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone, code }),
  })
  
  if (!response.ok) {
    throw new Error('Failed to verify OTP')
  }
  
  return response.json()
}

// Profile management
export const updateProfile = async (profileData: Partial<{
  full_name: string
  phone: string
  avatar_url: string
}>): Promise<void> => {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) throw new Error('Not authenticated')

  const { error } = await supabase
    .from('profiles')
    .update(profileData)
    .eq('id', user.id)

  if (error) throw error
}

// Error handling utility
export const handleApiError = (error: any): string => {
  if (error.message) return error.message
  if (typeof error === 'string') return error
  return 'An unexpected error occurred'
} 