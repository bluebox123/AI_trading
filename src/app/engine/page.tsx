import { AppLayout } from '@/components/Sidebar'
import { EngineConsole } from '@/components/EngineConsole'

export default function EnginePage() {
  return (
    <AppLayout>
      <div className="p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Trading Engine</h1>
          <p className="text-muted-foreground">
            Live monitoring of the AI model orchestrator and signal generation system
          </p>
        </div>
        
        <EngineConsole />
      </div>
    </AppLayout>
  )
} 