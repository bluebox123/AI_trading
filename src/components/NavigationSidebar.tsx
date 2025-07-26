'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { 
  LayoutDashboard, 
  TrendingUp, 
  Zap, 
  User,
  Activity,
  Settings,
  LogOut,
  Shield
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'

interface NavigationSidebarProps {
  user?: {
    email?: string
    user_metadata?: {
      full_name?: string
      avatar_url?: string
      risk_profile?: string
    }
  } | null
  onSignOut?: () => void
}

export default function NavigationSidebar({ user, onSignOut }: NavigationSidebarProps) {
  const pathname = usePathname()

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
      description: 'Overview & Signals'
    },
    {
      name: 'Live Signals',
      href: '/signals',
      icon: TrendingUp,
      description: '100+ Stock Analysis'
    },
    {
      name: 'Sentiment Analysis',
      href: '/sentiment',
      icon: Activity,
      description: '5-Year Sentiment Data'
    },
    {
      name: 'Engine Console',
      href: '/engine',
      icon: Zap,
      description: 'System Status'
    },
    {
      name: 'Profile',
      href: '/profile',
      icon: User,
      description: 'Account Settings'
    }
  ]

  const isActive = (href: string) => pathname === href

  return (
    <div className="hidden lg:flex lg:w-72 lg:flex-col lg:fixed lg:inset-y-0 lg:z-50">
      <div className="flex flex-col flex-grow bg-black/90 backdrop-blur-xl border-r border-gray-800">
        {/* Logo Section */}
        <div className="flex items-center px-6 py-6 border-b border-gray-800">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl flex items-center justify-center mr-3">
            <TrendingUp className="w-6 h-6 text-black font-bold" />
          </div>
          <div>
            <h1 className="text-xl font-bold neon-text">TradingSignals</h1>
            <p className="text-xs text-gray-400">Institutional Platform</p>
          </div>
        </div>

        {/* User Profile Section */}
        {user && (
          <div className="px-6 py-4 border-b border-gray-800">
            <div className="flex items-center space-x-3">
              <Avatar className="w-12 h-12 border border-emerald-500/30">
                <AvatarImage 
                  src={user.user_metadata?.avatar_url} 
                  alt={user.user_metadata?.full_name || 'User'} 
                />
                <AvatarFallback className="bg-emerald-500/20 text-emerald-400 font-bold">
                  {user.user_metadata?.full_name?.charAt(0) || user.email?.charAt(0) || 'U'}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {user.user_metadata?.full_name || 'Professional Trader'}
                </p>
                <p className="text-xs text-gray-400 truncate">
                  {user.email}
                </p>
                <div className="flex items-center mt-1">
                  <Shield className="w-3 h-3 text-emerald-400 mr-1" />
                  <span className="text-xs text-emerald-400 font-medium">Pro Plan</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {user.user_metadata?.risk_profile || 'Moderate'} Risk
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-4 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`group flex items-center px-3 py-3 text-sm font-medium rounded-xl transition-all duration-200 ${
                  active
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'text-gray-300 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                <Icon className={`mr-3 h-5 w-5 flex-shrink-0 ${
                  active ? 'text-emerald-400' : 'text-gray-400 group-hover:text-white'
                }`} />
                <div className="flex-1">
                  <div className={`font-medium ${active ? 'text-emerald-400' : ''}`}>
                    {item.name}
                  </div>
                  <div className="text-xs text-gray-500 group-hover:text-gray-400">
                    {item.description}
                  </div>
                </div>
                {active && (
                  <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                )}
              </Link>
            )
          })}
        </nav>

        {/* System Status */}
        <div className="px-6 py-4 border-t border-gray-800">
          <div className="flex items-center space-x-3 mb-3">
            <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-sm text-emerald-400 font-medium">System Online</span>
          </div>
          <div className="space-y-1 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>Active Signals:</span>
              <span className="text-emerald-400 font-medium">247</span>
            </div>
            <div className="flex justify-between">
              <span>Model Accuracy:</span>
              <span className="text-emerald-400 font-medium">95.7%</span>
            </div>
            <div className="flex justify-between">
              <span>Last Update:</span>
              <span className="text-emerald-400 font-medium">2m ago</span>
            </div>
          </div>
        </div>

        {/* Bottom Actions */}
        <div className="px-4 py-4 border-t border-gray-800 space-y-2">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-gray-400 hover:text-white hover:bg-gray-800/50"
          >
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
          
          {onSignOut && (
            <Button
              onClick={onSignOut}
              variant="ghost"
              size="sm"
              className="w-full justify-start text-gray-400 hover:text-red-400 hover:bg-red-500/10"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          )}
        </div>
      </div>
    </div>
  )
} 