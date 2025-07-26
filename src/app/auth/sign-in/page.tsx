'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { signIn } from '@/lib/supabase'
import { 
  TrendingUp, 
  Eye, 
  EyeOff, 
  Loader2, 
  Mail, 
  Lock,
  ArrowRight,
  Sparkles,
  Shield,
  CheckCircle,
  AlertCircle
} from 'lucide-react'
import { toast } from 'sonner'

// Validation schema
const signInSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required')
})

type SignInFormData = z.infer<typeof signInSchema>

export default function SignInPage() {
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const router = useRouter()
  const searchParams = useSearchParams()

  const {
    register,
    handleSubmit,
    formState: { errors, isValid }
  } = useForm<SignInFormData>({
    resolver: zodResolver(signInSchema),
    mode: 'onChange'
  })

  // Check for success message from sign-up
  useEffect(() => {
    const urlMessage = searchParams?.get('message')
    if (urlMessage) {
      setMessage({ type: 'success', text: urlMessage })
    }
  }, [searchParams])

  const onSubmit = async (data: SignInFormData) => {
    setIsLoading(true)
    setMessage(null)
    
    try {
      const { user } = await signIn(data.email, data.password)
      
      if (user) {
        toast.success('Welcome back!')
        router.push('/dashboard')
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Sign in failed'
      setMessage({ type: 'error', text: errorMessage })
      toast.error(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="trading-grid opacity-30"></div>
        <div className="neural-network"></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center space-x-3 mb-6 group">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl flex items-center justify-center">
                <TrendingUp className="w-7 h-7 text-black font-bold" />
              </div>
              <div className="absolute -inset-1 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl blur opacity-30 group-hover:opacity-60 transition-opacity"></div>
            </div>
            <span className="text-2xl font-bold neon-text">TradingSignals</span>
          </Link>
          
          <h1 className="text-4xl font-black mb-4">
            Welcome Back to <span className="neon-text">TradingSignals</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Access your institutional-grade trading dashboard
          </p>
        </div>

        {/* Success/Error Messages */}
        {message && (
          <div className="mb-6">
            <Alert className={`border-2 ${
              message.type === 'success' 
                ? 'border-emerald-500/50 bg-emerald-500/10' 
                : 'border-red-500/50 bg-red-500/10'
            }`}>
              <div className="flex items-center">
                {message.type === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-emerald-400 mr-2" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-red-400 mr-2" />
                )}
                <AlertDescription className={
                  message.type === 'success' ? 'text-emerald-300' : 'text-red-300'
                }>
                  {message.text}
                </AlertDescription>
              </div>
            </Alert>
          </div>
        )}

        {/* Sign In Card */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-2xl blur-xl"></div>
          
          <Card className="relative bg-black/80 border-emerald-500/30 backdrop-blur-xl">
            <CardContent className="p-8">
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Email */}
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-emerald-400 font-medium flex items-center">
                    <Mail className="w-4 h-4 mr-2" />
                    Email Address
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    {...register('email')}
                    placeholder="your.email@company.com"
                    className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12"
                    autoComplete="email"
                  />
                  {errors.email && (
                    <p className="text-red-400 text-sm flex items-center">
                      <AlertCircle className="w-4 h-4 mr-1" />
                      {errors.email.message}
                    </p>
                  )}
                </div>

                {/* Password */}
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-emerald-400 font-medium flex items-center">
                    <Lock className="w-4 h-4 mr-2" />
                    Password
                  </Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      {...register('password')}
                      placeholder="Enter your password"
                      className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 pr-12"
                      autoComplete="current-password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-emerald-400 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="text-red-400 text-sm flex items-center">
                      <AlertCircle className="w-4 h-4 mr-1" />
                      {errors.password.message}
                    </p>
                  )}
                </div>

                {/* Forgot Password Link */}
                <div className="text-right">
                  <Link 
                    href="/auth/forgot-password" 
                    className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    Forgot your password?
                  </Link>
                </div>

                {/* Submit Button */}
                <Button
                  type="submit"
                  disabled={!isValid || isLoading}
                  className="w-full cyber-btn-hero h-14 text-lg font-black disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Signing In...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5 mr-2" />
                      Access Trading Platform
                      <ArrowRight className="w-5 h-5 ml-2" />
                    </>
                  )}
                </Button>
              </form>

              {/* Divider */}
              <div className="relative my-8">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-600"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-black/80 text-gray-400">New to TradingSignals?</span>
                </div>
              </div>

              {/* Sign Up Link */}
              <div className="text-center">
                <Link 
                  href="/auth/sign-up"
                  className="cyber-btn-secondary w-full h-12 text-base font-bold inline-flex items-center justify-center rounded-xl"
                >
                  Create Your Account
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Link>
              </div>


            </CardContent>
          </Card>
        </div>

        {/* Security Notice */}
        <div className="mt-6 text-center">
          <div className="inline-flex items-center space-x-2 text-sm text-gray-400">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>Protected by enterprise-grade security and 2FA</span>
          </div>
        </div>

        {/* Features Preview */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div className="p-4 bg-black/40 rounded-xl border border-gray-700">
            <div className="text-emerald-400 font-bold text-lg">95%+</div>
            <div className="text-gray-400 text-sm">Signal Accuracy</div>
          </div>
          <div className="p-4 bg-black/40 rounded-xl border border-gray-700">
            <div className="text-emerald-400 font-bold text-lg">100+</div>
            <div className="text-gray-400 text-sm">NSE Stocks</div>
          </div>
          <div className="p-4 bg-black/40 rounded-xl border border-gray-700">
            <div className="text-emerald-400 font-bold text-lg">24/7</div>
            <div className="text-gray-400 text-sm">AI Analysis</div>
          </div>
        </div>
      </div>
    </div>
  )
} 