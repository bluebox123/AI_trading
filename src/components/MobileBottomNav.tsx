'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { 
  LayoutDashboard, 
  Newspaper, 
  TrendingUp, 
  Zap, 
  User
} from 'lucide-react'

export default function MobileBottomNav() {
  const pathname = usePathname()

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      name: 'Signals',
      href: '/stocks',
      icon: TrendingUp,
    },
    {
      name: 'News',
      href: '/news',
      icon: Newspaper,
    },
    {
      name: 'Engine',
      href: '/engine',
      icon: Zap,
    },
    {
      name: 'Profile',
      href: '/profile',
      icon: User,
    }
  ]

  const isActive = (href: string) => pathname === href

  return (
    <div className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-black/95 backdrop-blur-xl border-t border-gray-800">
      <div className="grid grid-cols-5 h-16">
        {navigationItems.map((item) => {
          const Icon = item.icon
          const active = isActive(item.href)
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex flex-col items-center justify-center space-y-1 transition-all duration-200 ${
                active 
                  ? 'text-emerald-400' 
                  : 'text-gray-400 hover:text-white active:scale-95'
              }`}
            >
              <div className={`relative ${active ? 'animate-pulse' : ''}`}>
                <Icon className={`h-5 w-5 ${
                  active ? 'text-emerald-400' : 'text-gray-400'
                }`} />
                {active && (
                  <div className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                )}
              </div>
              <span className={`text-xs font-medium ${
                active ? 'text-emerald-400' : 'text-gray-400'
              }`}>
                {item.name}
              </span>
              {active && (
                <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-8 h-0.5 bg-emerald-400 rounded-full" />
              )}
            </Link>
          )
        })}
      </div>
    </div>
  )
} 