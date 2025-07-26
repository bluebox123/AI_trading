import * as React from "react"

import { cn } from "@/lib/utils"

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * Current progress value (0-100)
   */
  value?: number
  /**
   * Maximum progress value. Defaults to 100.
   */
  max?: number
}

/**
 * A minimal progress bar component used across the dashboard.
 * Matches the API & styling conventions of other shadcn/ui components
 * so existing imports (`import { Progress } from '@/components/ui/progress'`) work.
 */
export const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, ...props }, ref) => {
    // Clamp value between 0 and max
    const clamped = Math.min(Math.max(value, 0), max)
    const percentage = (clamped / max) * 100

    return (
      <div
        ref={ref}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={max}
        className={cn("relative w-full h-2 overflow-hidden rounded-full bg-muted", className)}
        {...props}
      >
        <div
          className="h-full bg-primary transition-transform"
          style={{ transform: `translateX(${percentage - 100}%)` }}
        />
      </div>
    )
  }
)

Progress.displayName = "Progress" 