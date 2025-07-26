'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { 
  Activity, 
  Circle, 
  Pause, 
  Play, 
  Trash2,
  Download
} from 'lucide-react'

interface LogEntry {
  id: string
  timestamp: Date
  level: 'info' | 'warning' | 'error' | 'success'
  message: string
  source?: string
}

interface EngineStatus {
  status: 'active' | 'idle' | 'error'
  lastUpdate: Date
  modelsLoaded: number
  signalsGenerated: number
}

const levelConfig = {
  info: { color: 'text-blue-400', bg: 'bg-blue-500/10', badge: 'bg-blue-500' },
  warning: { color: 'text-yellow-400', bg: 'bg-yellow-500/10', badge: 'bg-yellow-500' },
  error: { color: 'text-red-400', bg: 'bg-red-500/10', badge: 'bg-red-500' },
  success: { color: 'text-green-400', bg: 'bg-green-500/10', badge: 'bg-green-500' }
}

const statusConfig = {
  active: { color: 'text-green-400', bg: 'bg-green-500', label: 'Active' },
  idle: { color: 'text-yellow-400', bg: 'bg-yellow-500', label: 'Idle' },
  error: { color: 'text-red-400', bg: 'bg-red-500', label: 'Error' }
}

export function EngineConsole() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [status, setStatus] = useState<EngineStatus>({
    status: 'active',
    lastUpdate: new Date(),
    modelsLoaded: 99,
    signalsGenerated: 1247
  })
  const [isPaused, setIsPaused] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const consoleRef = useRef<HTMLDivElement>(null)

  // Mock log entries for demonstration
  const mockLogs: Omit<LogEntry, 'id' | 'timestamp'>[] = [
    { level: 'info', message: '> v3 & v4 ensemble loaded', source: 'ModelLoader' },
    { level: 'success', message: '> Sentiment analyser initialised', source: 'SentimentEngine' },
    { level: 'info', message: '> News fetched – 326 articles (1 m)', source: 'NewsCollector' },
    { level: 'info', message: '> Processing NIFTY 50 components...', source: 'DataProcessor' },
    { level: 'success', message: '> Generated 47 buy signals, 23 sell signals', source: 'SignalGenerator' },
    { level: 'warning', message: '> Rate limit approaching for Alpha Vantage API', source: 'APIManager' },
    { level: 'info', message: '> Risk assessment complete - portfolio within limits', source: 'RiskManager' },
    { level: 'success', message: '> Cache updated successfully', source: 'CacheManager' },
    { level: 'info', message: '> Next analysis cycle in 5 minutes', source: 'Scheduler' }
  ]

  // Simulate real-time logs
  useEffect(() => {
    if (isPaused) return

    const interval = setInterval(() => {
      const randomLog = mockLogs[Math.floor(Math.random() * mockLogs.length)]
      const newLog: LogEntry = {
        ...randomLog,
        id: Date.now().toString(),
        timestamp: new Date()
      }
      
      setLogs(prev => [...prev.slice(-99), newLog]) // Keep last 100 logs
      
      // Update status occasionally
      if (Math.random() < 0.1) {
        setStatus(prev => ({
          ...prev,
          lastUpdate: new Date(),
          signalsGenerated: prev.signalsGenerated + Math.floor(Math.random() * 5)
        }))
      }
    }, 2000 + Math.random() * 3000) // Random interval between 2-5 seconds

    return () => clearInterval(interval)
  }, [isPaused])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const clearLogs = () => {
    setLogs([])
  }

  const exportLogs = () => {
    const logText = logs.map(log => 
      `[${log.timestamp.toISOString()}] ${log.level.toUpperCase()}: ${log.message}`
    ).join('\n')
    
    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `engine-logs-${new Date().toISOString().split('T')[0]}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const statusInfo = statusConfig[status.status]

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Circle className={cn("w-4 h-4", statusInfo.color)} fill="currentColor" />
              {status.status === 'active' && (
                <div className={cn("absolute inset-0 w-4 h-4 rounded-full animate-ping", statusInfo.bg)} />
              )}
            </div>
            <h2 className="text-xl font-semibold">Trading Engine</h2>
            <Badge className={cn("text-white", statusInfo.bg)}>
              {statusInfo.label}
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              {isPaused ? 'Resume' : 'Pause'}
            </Button>
            <Button variant="outline" size="sm" onClick={clearLogs}>
              <Trash2 className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={exportLogs}>
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-2xl font-bold">{status.modelsLoaded}</p>
            <p className="text-sm text-muted-foreground">Models Loaded</p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-2xl font-bold">{status.signalsGenerated}</p>
            <p className="text-sm text-muted-foreground">Signals Generated</p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm font-mono">{status.lastUpdate.toLocaleTimeString()}</p>
            <p className="text-sm text-muted-foreground">Last Update</p>
          </div>
        </div>
      </Card>

      {/* Console */}
      <Card className="p-0 overflow-hidden">
        <div className="bg-black text-green-400 font-mono text-sm">
          {/* Console Header */}
          <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4" />
              <span>Live Engine Console</span>
            </div>
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-1 text-xs">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  className="w-3 h-3"
                />
                <span>Auto-scroll</span>
              </label>
            </div>
          </div>

          {/* Console Content */}
          <div 
            ref={scrollRef}
            className="h-96 overflow-y-auto p-4 space-y-1"
            onScroll={() => {
              if (scrollRef.current) {
                const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
                setAutoScroll(scrollTop + clientHeight >= scrollHeight - 10)
              }
            }}
          >
            <AnimatePresence>
              {logs.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className={cn(
                    "flex items-start space-x-2 p-2 rounded",
                    levelConfig[log.level].bg
                  )}
                >
                  <span className="text-gray-500 text-xs flex-shrink-0 w-20">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                  <span className={cn(
                    "w-1 h-1 rounded-full mt-2 flex-shrink-0",
                    levelConfig[log.level].badge
                  )} />
                  <span className={cn("flex-1", levelConfig[log.level].color)}>
                    {log.message}
                  </span>
                  {log.source && (
                    <span className="text-gray-500 text-xs flex-shrink-0">
                      [{log.source}]
                    </span>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
            
            {logs.length === 0 && (
              <div className="text-center text-gray-500 py-8">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>Waiting for engine logs...</p>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
} 