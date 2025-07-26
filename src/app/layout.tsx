import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from 'sonner'

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TradingSignals - Institutional-Grade AI Trading Signals",
  description: "Advanced ML-powered trading signals for professional traders and institutions. Real-time analysis of 100+ NSE stocks with 95%+ accuracy.",
  keywords: ["trading signals", "AI trading", "machine learning", "NSE stocks", "algorithmic trading", "institutional trading"],
  authors: [{ name: "TradingSignals" }],
  robots: "index, follow",
  openGraph: {
    title: "TradingSignals - Institutional-Grade AI Trading Signals",
    description: "Advanced ML-powered trading signals for professional traders and institutions.",
    type: "website",
    url: "https://tradingsignals.ai",
  },
  twitter: {
    card: "summary_large_image",
    title: "TradingSignals - Institutional-Grade AI Trading Signals",
    description: "Advanced ML-powered trading signals for professional traders and institutions.",
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {children}
        <Toaster 
          theme="dark"
          position="top-right"
          expand={false}
          richColors
          toastOptions={{
            style: {
              background: "rgb(0 0 0 / 0.9)",
              border: "1px solid rgb(16 185 129 / 0.3)",
              color: "white",
            },
          }}
        />
      </body>
    </html>
  );
}
