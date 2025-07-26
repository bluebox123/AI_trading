'use client'

import { ReactNode } from 'react'
import NavigationSidebar from './NavigationSidebar'

interface AppLayoutProps {
  children: ReactNode
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

export function AppLayout({ children, user, onSignOut }: AppLayoutProps) {
  return (
    <div className="flex h-screen bg-black">
      <NavigationSidebar user={user} onSignOut={onSignOut} />
      <main className="flex-1 lg:ml-72 overflow-y-auto">
        {children}
      </main>
    </div>
  )
} 