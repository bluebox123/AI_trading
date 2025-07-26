'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Mail, 
  AlertTriangle, 
  CheckCircle, 
  X, 
  RefreshCw,
  Shield,
  Clock
} from 'lucide-react'
import { toast } from 'sonner'

interface EmailVerificationAlertProps {
  isEmailVerified: boolean
  userEmail: string
  onResendVerification?: () => Promise<void>
  onDismiss?: () => void
  showTopAlert?: boolean
}

export function EmailVerificationAlert({ 
  isEmailVerified, 
  userEmail, 
  onResendVerification,
  onDismiss,
  showTopAlert = false
}: EmailVerificationAlertProps) {
  const [isResending, setIsResending] = useState(false)

  const handleResendVerification = async () => {
    if (!onResendVerification) return
    
    setIsResending(true)
    try {
      await onResendVerification()
      toast.success('Verification email sent! Check your inbox.')
    } catch (error) {
      toast.error('Failed to send verification email. Please try again.')
    } finally {
      setIsResending(false)
    }
  }

  // Top alert banner (less intrusive)
  if (showTopAlert && !isEmailVerified) {
    return (
      <div className="fixed top-0 left-0 right-0 z-50 bg-black/95 backdrop-blur-xl border-b border-amber-500/30">
        <Alert className="rounded-none border-0 bg-gradient-to-r from-amber-500/10 to-orange-500/10">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
          <AlertDescription className="text-amber-200 font-medium">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <span>
                  Please verify your email address ({userEmail}) to access all features
                </span>
                <Button
                  onClick={handleResendVerification}
                  disabled={isResending}
                  size="sm"
                  className="bg-amber-600 hover:bg-amber-500 text-black font-medium h-8"
                >
                  {isResending ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    <>
                      <Mail className="w-3 h-3 mr-1" />
                      Resend Email
                    </>
                  )}
                </Button>
              </div>
              {onDismiss && (
                <Button
                  onClick={onDismiss}
                  size="sm"
                  variant="ghost"
                  className="text-amber-200 hover:text-white h-8 w-8 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  // Full page overlay (more intrusive for unverified emails)
  if (!isEmailVerified && !showTopAlert) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop blur */}
        <div className="absolute inset-0 bg-black/80 backdrop-blur-md"></div>
        
        {/* Alert Content */}
        <div className="relative z-10 w-full max-w-md mx-4">
          <Card className="bg-black/90 border-2 border-amber-500/50 shadow-2xl">
            <CardContent className="p-8 text-center">
              <div className="mb-6">
                <div className="w-16 h-16 bg-amber-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Mail className="w-8 h-8 text-amber-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">
                  Email Verification Required
                </h2>
                <p className="text-gray-300 mb-4">
                  To ensure account security and access all trading features, please verify your email address.
                </p>
              </div>

              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 mb-6">
                <div className="flex items-center justify-center space-x-2 text-amber-200 text-sm">
                  <Shield className="w-4 h-4" />
                  <span className="font-medium">{userEmail}</span>
                </div>
              </div>

              <div className="space-y-4">
                <Button
                  onClick={handleResendVerification}
                  disabled={isResending}
                  className="w-full bg-amber-600 hover:bg-amber-500 text-black font-bold h-12"
                >
                  {isResending ? (
                    <>
                      <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                      Sending Verification Email...
                    </>
                  ) : (
                    <>
                      <Mail className="w-5 h-5 mr-2" />
                      Send Verification Email
                    </>
                  )}
                </Button>

                <div className="flex items-center space-x-2 text-xs text-gray-400">
                  <Clock className="w-3 h-3" />
                  <span>Check your spam folder if you don't see the email</span>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-700">
                <p className="text-xs text-gray-500">
                  Having trouble? Contact support at help@tradingsignals.com
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  // Verified state (can be used as a success indicator)
  if (isEmailVerified && showTopAlert) {
    return (
      <div className="fixed top-0 left-0 right-0 z-50 bg-black/95 backdrop-blur-xl border-b border-emerald-500/30">
        <Alert className="rounded-none border-0 bg-gradient-to-r from-emerald-500/10 to-green-500/10">
          <CheckCircle className="w-5 h-5 text-emerald-400" />
          <AlertDescription className="text-emerald-200 font-medium">
            <div className="flex items-center justify-between">
              <span>✓ Email verified - Full access enabled</span>
              {onDismiss && (
                <Button
                  onClick={onDismiss}
                  size="sm"
                  variant="ghost"
                  className="text-emerald-200 hover:text-white h-8 w-8 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return null
} 