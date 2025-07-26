import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { Toaster } from 'sonner'
import { Providers } from '@/components/providers'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'Trading Signals - Institutional Grade Stock Analysis',
  description: 'AI-powered buy/sell signals for NSE largecap and midcap stocks using advanced ML models.',
  keywords: 'trading signals, stock analysis, NSE, AI trading, machine learning, buy sell signals',
  authors: [{ name: 'Trading Signals Team' }],
  openGraph: {
    title: 'Trading Signals - Institutional Grade Stock Analysis',
    description: 'AI-powered buy/sell signals for NSE stocks',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-background font-sans`}
      >
        <Providers>
          {children}
          <Toaster 
            position="top-right" 
            theme="system"
            richColors
          />
        </Providers>
      </body>
    </html>
  )
} 