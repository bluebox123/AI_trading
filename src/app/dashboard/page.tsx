'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { getCurrentUser, signOut, resendEmailVerification, checkEmailVerified } from '@/lib/supabase'
import type { User as SupabaseUser } from '@supabase/supabase-js'
import { EmailVerificationAlert } from '@/components/EmailVerificationAlert'
import { toast } from 'sonner'

// Import our new dashboard components
import NavigationSidebar from '@/components/NavigationSidebar'
import MobileBottomNav from '@/components/MobileBottomNav'
import TopSignalsTable from '@/components/TopSignalsTable'
import MarketIndicesWidget from '@/components/MarketIndicesWidget'
import PortfolioSummary from '@/components/PortfolioSummary'
import QuickShortcuts from '@/components/QuickShortcuts'

interface Signal {
  symbol: string
  signal: 'BUY' | 'SELL' | 'HOLD'
  final_score: number
  confidence: number
  current_price: number
  price_target?: number
  price?: number
  change?: number
  change_percent?: number
  volume?: number
  sector?: string
  company?: string
}

export default function DashboardPage() {
  const [user, setUser] = useState<SupabaseUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [showTopAlert, setShowTopAlert] = useState(false)
  const [signals, setSignals] = useState<Signal[]>([])
  const [signalsLoading, setSignalsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const currentUser = await getCurrentUser()
        setUser(currentUser)
        
        // Check if email is verified and show appropriate alert
        if (currentUser && !checkEmailVerified(currentUser)) {
          // Show top alert initially, then full overlay after 3 seconds
          setShowTopAlert(true)
          setTimeout(() => {
            setShowTopAlert(false)
          }, 10000) // Show top alert for 10 seconds
        }
      } catch (error) {
        console.error('Error fetching user:', error)
        router.push('/auth/sign-in')
      } finally {
        setLoading(false)
      }
    }

    fetchUser()
  }, [router])

  useEffect(() => {
    async function fetchSignals() {
      setSignalsLoading(true)
      try {
        const res = await fetch('/api/signals')
        const data = await res.json()
        setSignals(data.signals || [])
      } catch (error) {
        console.error('Error fetching signals:', error)
      } finally {
        setSignalsLoading(false)
      }
    }

    fetchSignals()
  }, [])

  const handleSignOut = async () => {
    try {
      await signOut()
      toast.success('Signed out successfully')
      router.push('/')
    } catch {
      toast.error('Error signing out')
    }
  }

  const handleResendVerification = async () => {
    try {
      await resendEmailVerification()
      toast.success('Verification email sent! Check your inbox.')
    } catch (error) {
      toast.error('Failed to send verification email')
    }
  }

  const isEmailVerified = user ? checkEmailVerified(user) : false

  // Calculate signal statistics
  const buySignals = signals.filter(s => s.signal === 'BUY')
  const sellSignals = signals.filter(s => s.signal === 'SELL')
  const holdSignals = signals.filter(s => s.signal === 'HOLD')
  const totalSignals = signals.length

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-2xl flex items-center justify-center mx-auto mb-6 animate-pulse">
            <div className="w-8 h-8 bg-black rounded-lg"></div>
          </div>
          <div className="space-y-3">
            <div className="text-2xl font-bold neon-text">Loading Dashboard</div>
            <div className="text-gray-400">Initializing trading environment...</div>
            <div className="flex items-center justify-center space-x-1">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Email Verification Alerts */}
      {user && (
        <EmailVerificationAlert
          isEmailVerified={isEmailVerified}
          userEmail={user.email || ''}
          onResendVerification={handleResendVerification}
          onDismiss={() => setShowTopAlert(false)}
          showTopAlert={showTopAlert}
        />
      )}

      {/* Navigation Sidebar - Desktop */}
      <NavigationSidebar user={user} onSignOut={handleSignOut} />

      {/* Mobile Bottom Navigation */}
      <MobileBottomNav />

      {/* Main Dashboard Content */}
      <main className={`lg:pl-72 min-h-screen ${showTopAlert ? 'pt-16' : ''}`}>
        <div className={`${!isEmailVerified && !showTopAlert ? 'blur-sm pointer-events-none' : ''}`}>
          {/* Header Section */}
          <div className="bg-black/90 backdrop-blur-xl border-b border-gray-800 sticky top-0 z-40">
            <div className="px-6 py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-black mb-2">
                    Trading <span className="neon-text">Dashboard</span>
                  </h1>
                  <p className="text-gray-400">
                    Welcome back, {user?.user_metadata?.full_name || 'Professional Trader'}
                  </p>
                </div>
                
                {/* Real-time Status Indicator */}
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
                    <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
                    <span className="text-emerald-400 font-medium text-sm">Live Market Data</span>
                  </div>
                  <div className="text-sm text-gray-400">
                    Last update: {new Date().toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Dashboard Grid */}
          <div className="p-6 space-y-8 pb-24 lg:pb-8">
            {/* Market Overview Section */}
            <section>
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <span className="w-1 h-6 bg-emerald-400 rounded-full mr-3"></span>
                Market Overview
              </h2>
              <MarketIndicesWidget />
            </section>

            {/* Portfolio & Quick Actions */}
            <section>
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="xl:col-span-1">
                  <h2 className="text-xl font-bold mb-4 flex items-center">
                    <span className="w-1 h-6 bg-cyan-400 rounded-full mr-3"></span>
                    Portfolio
                  </h2>
                  <PortfolioSummary />
                </div>
                <div className="xl:col-span-2">
                  <h2 className="text-xl font-bold mb-4 flex items-center">
                    <span className="w-1 h-6 bg-purple-400 rounded-full mr-3"></span>
                    Quick Actions
                  </h2>
                  <QuickShortcuts />
                </div>
              </div>
            </section>

            {/* Trading Signals Section */}
            <section>
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <span className="w-1 h-6 bg-yellow-400 rounded-full mr-3"></span>
                Live Trading Signals
              </h2>
              
              {/* Signal Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-400">Total Signals</p>
                      <p className="text-2xl font-bold text-white">{totalSignals}</p>
                    </div>
                    <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-blue-400 rounded-full"></div>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-400">Buy Signals</p>
                      <p className="text-2xl font-bold text-emerald-400">{buySignals.length}</p>
                    </div>
                    <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-emerald-400 rounded-full"></div>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-400">Sell Signals</p>
                      <p className="text-2xl font-bold text-red-400">{sellSignals.length}</p>
                    </div>
                    <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-red-400 rounded-full"></div>
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-400">Hold Signals</p>
                      <p className="text-2xl font-bold text-yellow-400">{holdSignals.length}</p>
                    </div>
                    <div className="w-8 h-8 bg-yellow-500/20 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-yellow-400 rounded-full"></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <TopSignalsTable type="buy" />
                <TopSignalsTable type="sell" />
              </div>
            </section>

            {/* System Performance Metrics */}
            <section>
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <span className="w-1 h-6 bg-pink-400 rounded-full mr-3"></span>
                System Performance
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="p-6 bg-gray-800/30 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-all">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-12 h-12 bg-emerald-500/20 rounded-xl flex items-center justify-center">
                      <div className="w-6 h-6 bg-emerald-400 rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-2xl font-bold text-emerald-400">95.7%</span>
                  </div>
                  <h3 className="font-bold text-white mb-1">Model Accuracy</h3>
                  <p className="text-sm text-gray-400">Last 30 days performance</p>
                  <div className="mt-3 text-xs text-emerald-400">+2.3% vs last month</div>
                </div>

                <div className="p-6 bg-gray-800/30 rounded-xl border border-gray-700 hover:border-cyan-500/50 transition-all">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-12 h-12 bg-cyan-500/20 rounded-xl flex items-center justify-center">
                      <div className="w-6 h-6 bg-cyan-400 rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-2xl font-bold text-cyan-400">{totalSignals}</span>
                  </div>
                  <h3 className="font-bold text-white mb-1">Active Signals</h3>
                  <p className="text-sm text-gray-400">Across 100+ NSE stocks</p>
                  <div className="mt-3 text-xs text-cyan-400">Updated 2m ago</div>
                </div>

                <div className="p-6 bg-gray-800/30 rounded-xl border border-gray-700 hover:border-yellow-500/50 transition-all">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-12 h-12 bg-yellow-500/20 rounded-xl flex items-center justify-center">
                      <div className="w-6 h-6 bg-yellow-400 rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-2xl font-bold text-yellow-400">12.4s</span>
                  </div>
                  <h3 className="font-bold text-white mb-1">Avg Response</h3>
                  <p className="text-sm text-gray-400">Signal generation time</p>
                  <div className="mt-3 text-xs text-yellow-400">-0.8s improvement</div>
                </div>

                <div className="p-6 bg-gray-800/30 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center">
                      <div className="w-6 h-6 bg-purple-400 rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-2xl font-bold text-purple-400">99.9%</span>
                  </div>
                  <h3 className="font-bold text-white mb-1">Uptime</h3>
                  <p className="text-sm text-gray-400">System availability</p>
                  <div className="mt-3 text-xs text-purple-400">24/7 monitoring</div>
                </div>
              </div>
            </section>

            {/* Success Message for New Users */}
            {isEmailVerified && (
              <section className="text-center py-8">
                <div className="inline-flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/30 rounded-2xl">
                  <div className="w-12 h-12 bg-emerald-500/20 rounded-xl flex items-center justify-center">
                    <div className="w-6 h-6 bg-emerald-400 rounded-full"></div>
                  </div>
                  <div className="text-left">
                    <div className="text-emerald-400 font-bold text-lg">
                      🎉 Welcome to TradingSignals Pro!
                    </div>
                    <div className="text-gray-300 text-sm">
                      Your account is fully verified and ready for institutional-grade trading
                    </div>
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
      </main>
    </div>
  )
} 