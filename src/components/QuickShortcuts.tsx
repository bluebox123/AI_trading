'use client'

import Link from 'next/link'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  TrendingUp, 
  Newspaper, 
  Zap, 
  Settings, 
  BarChart3, 
  Shield,
  Target,
  BookOpen,
  Bell,
  Download,
  ArrowRight
} from 'lucide-react'

interface Shortcut {
  title: string
  description: string
  href: string
  icon: React.ComponentType<{ className?: string }>
  color: string
  bgColor: string
  borderColor: string
  isExternal?: boolean
}

export default function QuickShortcuts() {
  const shortcuts: Shortcut[] = [
    {
      title: 'Live Signals',
      description: 'Browse 100+ stock analysis with real-time buy/sell signals',
      href: '/stocks',
      icon: TrendingUp,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/30'
    },
    {
      title: 'Market News',
      description: 'AI-curated news and sentiment analysis affecting your portfolio',
      href: '/news',
      icon: Newspaper,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10',
      borderColor: 'border-cyan-500/30'
    },
    {
      title: 'Engine Console',
      description: 'Monitor system status and model performance in real-time',
      href: '/engine',
      icon: Zap,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/10',
      borderColor: 'border-yellow-500/30'
    },
    {
      title: 'Risk Manager',
      description: 'Configure portfolio risk parameters and position sizing',
      href: '/risk',
      icon: Shield,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/30'
    },
    {
      title: 'Watchlist',
      description: 'Manage your custom stock watchlists and alerts',
      href: '/watchlist',
      icon: Target,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/30'
    },
    {
      title: 'Analytics',
      description: 'Deep-dive portfolio performance and strategy backtesting',
      href: '/analytics',
      icon: BarChart3,
      color: 'text-indigo-400',
      bgColor: 'bg-indigo-500/10',
      borderColor: 'border-indigo-500/30'
    },
    {
      title: 'API Documentation',
      description: 'Access trading APIs and integration documentation',
      href: '/docs/api',
      icon: BookOpen,
      color: 'text-gray-400',
      bgColor: 'bg-gray-500/10',
      borderColor: 'border-gray-500/30'
    },
    {
      title: 'Alert Center',
      description: 'Manage notifications and trading alerts preferences',
      href: '/alerts',
      icon: Bell,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/10',
      borderColor: 'border-orange-500/30'
    },
    {
      title: 'Export Reports',
      description: 'Download portfolio reports and trading statements',
      href: '/reports',
      icon: Download,
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/10',
      borderColor: 'border-pink-500/30'
    }
  ]

  return (
    <Card className="bg-black/80 border-gray-700 hover:border-gray-600 transition-colors">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-emerald-400" />
          <span>Quick Actions</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {shortcuts.map((shortcut) => {
            const Icon = shortcut.icon
            
            return (
              <Link
                key={shortcut.title}
                href={shortcut.href}
                className="group block"
              >
                <div className={`p-4 rounded-lg border transition-all duration-200 hover:scale-[1.02] cursor-pointer ${shortcut.bgColor} ${shortcut.borderColor} hover:border-opacity-60`}>
                  <div className="flex items-start space-x-3">
                    <div className={`p-2 rounded-lg ${shortcut.bgColor} border ${shortcut.borderColor}`}>
                      <Icon className={`w-5 h-5 ${shortcut.color}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className={`font-semibold text-sm group-hover:${shortcut.color} transition-colors text-white`}>
                        {shortcut.title}
                      </h3>
                      <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                        {shortcut.description}
                      </p>
                    </div>
                    <ArrowRight className={`w-4 h-4 text-gray-500 group-hover:${shortcut.color} group-hover:translate-x-1 transition-all`} />
                  </div>
                </div>
              </Link>
            )
          })}
        </div>

        {/* Quick Stats Row */}
        <div className="mt-6 pt-6 border-t border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-emerald-400">247</div>
              <div className="text-xs text-gray-400">Active Signals</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-cyan-400">95.7%</div>
              <div className="text-xs text-gray-400">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">100+</div>
              <div className="text-xs text-gray-400">Stocks Tracked</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">24/7</div>
              <div className="text-xs text-gray-400">Monitoring</div>
            </div>
          </div>
        </div>

        {/* Pro Features Banner */}
        <div className="mt-6 p-4 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border border-emerald-500/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-emerald-400 text-sm">Pro Plan Active</h4>
              <p className="text-xs text-gray-400 mt-1">
                Access to all premium features and unlimited signals
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/20"
            >
              Manage Plan
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 