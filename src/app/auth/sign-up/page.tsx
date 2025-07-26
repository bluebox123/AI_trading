'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Card, CardContent } from '@/components/ui/card'
import { signUp, sendOTP, verifyOTP } from '@/lib/supabase'
import { 
  TrendingUp, 
  Eye, 
  EyeOff, 
  Loader2, 
  Check, 
  X, 
  Shield, 
  Phone, 
  Mail, 
  User, 
  Lock,
  ArrowRight,
  Sparkles
} from 'lucide-react'
import { toast } from 'sonner'

// Country codes list
const countryCodes = [
  { code: '+1', country: 'US', flag: '🇺🇸' },
  { code: '+91', country: 'IN', flag: '🇮🇳' },
  { code: '+44', country: 'UK', flag: '🇬🇧' },
  { code: '+86', country: 'CN', flag: '🇨🇳' },
  { code: '+49', country: 'DE', flag: '🇩🇪' },
  { code: '+33', country: 'FR', flag: '🇫🇷' },
  { code: '+81', country: 'JP', flag: '🇯🇵' },
  { code: '+82', country: 'KR', flag: '🇰🇷' },
  { code: '+61', country: 'AU', flag: '🇦🇺' },
  { code: '+65', country: 'SG', flag: '🇸🇬' },
]

// Validation schema
const signUpSchema = z.object({
  fullName: z.string().min(2, 'Full name must be at least 2 characters'),
  email: z.string().email('Please enter a valid email address'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
      'Password must contain uppercase, lowercase, number and special character'),
  confirmPassword: z.string(),
  countryCode: z.string().min(1, 'Please select country code'),
  phoneNumber: z.string()
    .min(8, 'Phone number must be at least 8 digits')
    .max(15, 'Phone number cannot exceed 15 digits')
    .regex(/^[\d]+$/, 'Phone number should contain only digits'),
  riskProfile: z.enum(['conservative', 'moderate', 'aggressive'], {
    required_error: 'Please select your risk profile'
  }),
  agreeToTerms: z.boolean().refine(val => val === true, {
    message: 'You must agree to the terms and conditions'
  })
}).refine(data => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"]
})

type SignUpFormData = z.infer<typeof signUpSchema>

// Password strength checker
const getPasswordStrength = (password: string) => {
  let strength = 0
  const checks = {
    length: password.length >= 8,
    lowercase: /[a-z]/.test(password),
    uppercase: /[A-Z]/.test(password),
    number: /\d/.test(password),
    special: /[@$!%*?&]/.test(password)
  }
  
  strength = Object.values(checks).filter(Boolean).length
  
  return {
    score: strength,
    checks,
    label: strength < 2 ? 'Weak' : strength < 4 ? 'Medium' : 'Strong',
    color: strength < 2 ? 'red' : strength < 4 ? 'yellow' : 'emerald'
  }
}

// OTP Modal Component
interface OTPModalProps {
  isOpen: boolean
  phone: string
  onVerify: (code: string) => void
  onCancel: () => void
  isLoading: boolean
  onResend: () => void
}

function OTPModal({ isOpen, phone, onVerify, onCancel, isLoading, onResend }: OTPModalProps) {
  const [code, setCode] = useState('')
  const [timeLeft, setTimeLeft] = useState(60)
  const [canResend, setCanResend] = useState(false)

  useEffect(() => {
    if (isOpen && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0) {
      setCanResend(true)
    }
  }, [isOpen, timeLeft])

  const handleResend = () => {
    onResend()
    setTimeLeft(60)
    setCanResend(false)
    setCode('')
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="relative max-w-md w-full">
        <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 rounded-2xl blur-xl"></div>
        
        <Card className="relative bg-black/90 border-emerald-500/30 backdrop-blur-xl">
          <CardContent className="p-8">
            <div className="text-center mb-6">
              <div className="mx-auto mb-4 w-16 h-16 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-2xl flex items-center justify-center">
                <Phone className="w-8 h-8 text-black" />
              </div>
              <h2 className="text-2xl font-bold neon-text">Verify Your Phone</h2>
              <p className="text-gray-400 mt-2">Enter the 6-digit code sent to {phone}</p>
            </div>
            
            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="otp" className="text-emerald-400 font-medium">Verification Code</Label>
                <Input
                  id="otp"
                  type="text"
                  placeholder="000000"
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  maxLength={6}
                  className="text-center text-2xl tracking-[0.5em] font-mono bg-black/50 border-emerald-500/30 focus:border-emerald-500 h-14"
                />
              </div>
              
              <div className="text-center">
                {!canResend ? (
                  <p className="text-sm text-gray-400">Resend code in {timeLeft}s</p>
                ) : (
                  <button
                    onClick={handleResend}
                    className="text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    Resend verification code
                  </button>
                )}
              </div>
              
              <div className="flex space-x-3">
                <Button
                  onClick={() => onVerify(code)}
                  disabled={code.length !== 6 || isLoading}
                  className="flex-1 cyber-btn-primary h-12 text-base font-bold"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    <>
                      <Check className="w-5 h-5 mr-2" />
                      Verify & Continue
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={onCancel}
                  disabled={isLoading}
                  className="px-6 h-12 border-gray-600 hover:border-emerald-500 hover:text-emerald-400"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function SignUpPage() {
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [showOTP, setShowOTP] = useState(false)
  const [passwordStrength, setPasswordStrength] = useState(getPasswordStrength(''))
  const [agreeToTerms, setAgreeToTerms] = useState(false)
  const router = useRouter()

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors, isValid },
    trigger
  } = useForm<SignUpFormData>({
    resolver: zodResolver(signUpSchema),
    mode: 'onChange',
    defaultValues: {
      countryCode: '+91', // Default to India
      agreeToTerms: false
    }
  })

  const password = watch('password', '')
  const countryCode = watch('countryCode', '+91')
  const phoneNumber = watch('phoneNumber', '')
  const fullName = watch('fullName', '')
  const email = watch('email', '')
  const confirmPassword = watch('confirmPassword', '')
  const riskProfile = watch('riskProfile')
  
  // Combine country code and phone number for OTP
  const fullPhone = `${countryCode}${phoneNumber}`

  // Check if form is actually valid
  const isFormValid = fullName && 
                     email && 
                     password && 
                     confirmPassword && 
                     phoneNumber && 
                     countryCode && 
                     riskProfile && 
                     agreeToTerms &&
                     !errors.fullName && 
                     !errors.email && 
                     !errors.password && 
                     !errors.confirmPassword && 
                     !errors.phoneNumber && 
                     !errors.countryCode

  useEffect(() => {
    setPasswordStrength(getPasswordStrength(password))
  }, [password])

  // Handle checkbox change
  const handleCheckboxChange = (checked: boolean) => {
    setAgreeToTerms(checked)
    setValue('agreeToTerms', checked)
    trigger('agreeToTerms')
  }

  const onSubmit = async (data: SignUpFormData) => {
    if (!isFormValid) return
    
    setIsLoading(true)
    
    try {
      const fullPhoneNumber = `${data.countryCode}${data.phoneNumber}`
      
      // Send OTP first
      await sendOTP(fullPhoneNumber)
      setShowOTP(true)
      toast.success('Verification code sent! Check terminal console for development code.')
      
      // Show development help in console
      if (process.env.NODE_ENV === 'development') {
        console.log('📱 DEVELOPMENT MODE: Check the terminal where you ran "npm run dev" for your OTP code!')
      }
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Failed to send verification code')
    } finally {
      setIsLoading(false)
    }
  }

  const handleOTPVerify = async (code: string) => {
    setIsLoading(true)
    
    try {
      const { verified } = await verifyOTP(fullPhone, code)
      
      if (!verified) {
        throw new Error('Invalid verification code')
      }

      const formData = watch()
      const { user } = await signUp(formData.email, formData.password, {
        full_name: formData.fullName,
        phone: fullPhone,
        risk_profile: formData.riskProfile,
        plan: 'pro'
      })

      if (user) {
        toast.success('Account created successfully!')
        router.push('/auth/sign-in?message=Account created successfully. Please sign in.')
      }
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Registration failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleOTPCancel = () => setShowOTP(false)

  const handleResendOTP = async () => {
    try {
      await sendOTP(fullPhone)
      toast.success('New verification code sent!')
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : 'Failed to resend code')
    }
  }

  return (
    <>
      <div className="min-h-screen bg-black text-white flex items-center justify-center p-4 relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="trading-grid opacity-30"></div>
          <div className="neural-network"></div>
        </div>

        <div className="relative z-10 w-full max-w-lg">
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
              Create Your <span className="neon-text">Account</span>
            </h1>
            <p className="text-gray-400 text-lg">
              Join thousands of professional traders using AI-powered signals
            </p>
          </div>

          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-2xl blur-xl"></div>
            
            <Card className="relative bg-black/80 border-emerald-500/30 backdrop-blur-xl">
              <CardContent className="p-8">
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="fullName" className="text-emerald-400 font-medium flex items-center">
                      <User className="w-4 h-4 mr-2" />
                      Full Name
                    </Label>
                    <Input
                      id="fullName"
                      {...register('fullName')}
                      placeholder="Enter your full name"
                      className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 text-gray-200 placeholder:text-gray-500"
                    />
                    {errors.fullName && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.fullName.message}
                      </p>
                    )}
                  </div>

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
                      className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 text-gray-200 placeholder:text-gray-500"
                    />
                    {errors.email && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.email.message}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label className="text-emerald-400 font-medium flex items-center">
                      <Phone className="w-4 h-4 mr-2" />
                      Phone Number
                    </Label>
                    <div className="flex space-x-2">
                      <Select value={countryCode} onValueChange={(value) => {
                        setValue('countryCode', value)
                        trigger('countryCode')
                      }}>
                        <SelectTrigger className="w-32 bg-black/50 border-gray-600 focus:border-emerald-500 h-12 text-gray-200">
                          <SelectValue placeholder="Code" />
                        </SelectTrigger>
                        <SelectContent className="bg-black border-gray-600">
                          {countryCodes.map((country) => (
                            <SelectItem key={country.code} value={country.code} className="text-gray-200">
                              <div className="flex items-center space-x-2">
                                <span>{country.flag}</span>
                                <span>{country.code}</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      
                      <Input
                        id="phoneNumber"
                        type="tel"
                        {...register('phoneNumber')}
                        placeholder="7066830353"
                        className="flex-1 bg-black/50 border-gray-600 focus:border-emerald-500 h-12 text-gray-200 placeholder:text-gray-500"
                      />
                    </div>
                    {(errors.countryCode || errors.phoneNumber) && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.countryCode?.message || errors.phoneNumber?.message}
                      </p>
                    )}
                  </div>

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
                        placeholder="Create a strong password"
                        className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 pr-12 text-gray-200 placeholder:text-gray-500"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-emerald-400 transition-colors"
                      >
                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                    
                    {password && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-400">Password strength:</span>
                          <span className={`font-medium ${
                            passwordStrength.color === 'red' ? 'text-red-400' :
                            passwordStrength.color === 'yellow' ? 'text-yellow-400' : 'text-emerald-400'
                          }`}>
                            {passwordStrength.label}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-300 ${
                              passwordStrength.color === 'red' ? 'bg-red-500' :
                              passwordStrength.color === 'yellow' ? 'bg-yellow-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-400">
                          Requirements: 8+ chars, uppercase, lowercase, number, special char
                        </div>
                      </div>
                    )}
                    
                    {errors.password && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.password.message}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword" className="text-emerald-400 font-medium flex items-center">
                      <Lock className="w-4 h-4 mr-2" />
                      Confirm Password
                    </Label>
                    <div className="relative">
                      <Input
                        id="confirmPassword"
                        type={showConfirmPassword ? 'text' : 'password'}
                        {...register('confirmPassword')}
                        placeholder="Confirm your password"
                        className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 pr-12 text-gray-200 placeholder:text-gray-500"
                      />
                      <button
                        type="button"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-emerald-400 transition-colors"
                      >
                        {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                    {errors.confirmPassword && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.confirmPassword.message}
                      </p>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label className="text-emerald-400 font-medium flex items-center">
                      <Shield className="w-4 h-4 mr-2" />
                      Risk Profile
                    </Label>
                    <Select onValueChange={(value) => {
                      setValue('riskProfile', value as 'conservative' | 'moderate' | 'aggressive')
                      trigger('riskProfile')
                    }}>
                      <SelectTrigger className="bg-black/50 border-gray-600 focus:border-emerald-500 h-12 text-gray-200">
                        <SelectValue placeholder="Select your risk tolerance" />
                      </SelectTrigger>
                      <SelectContent className="bg-black border-gray-600">
                        <SelectItem value="conservative" className="text-gray-200">Conservative - Low risk, steady returns</SelectItem>
                        <SelectItem value="moderate" className="text-gray-200">Moderate - Balanced risk/reward</SelectItem>
                        <SelectItem value="aggressive" className="text-gray-200">Aggressive - High risk, high potential returns</SelectItem>
                      </SelectContent>
                    </Select>
                    {errors.riskProfile && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.riskProfile.message}
                      </p>
                    )}
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-start space-x-3 p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                      <Checkbox
                        id="agreeToTerms"
                        checked={agreeToTerms}
                        onCheckedChange={handleCheckboxChange}
                        className="mt-0.5 data-[state=checked]:bg-emerald-500 data-[state=checked]:border-emerald-500"
                      />
                      <div className="flex-1">
                        <Label htmlFor="agreeToTerms" className="text-sm text-gray-300 leading-relaxed cursor-pointer">
                          I agree to the{' '}
                          <Link href="/legal/terms" className="text-emerald-400 hover:text-emerald-300 underline">
                            Terms of Service
                          </Link>{' '}
                          and{' '}
                          <Link href="/legal/privacy" className="text-emerald-400 hover:text-emerald-300 underline">
                            Privacy Policy
                          </Link>
                          . I understand the risks involved in trading and that past performance does not guarantee future results.
                        </Label>
                      </div>
                    </div>
                    {errors.agreeToTerms && (
                      <p className="text-red-400 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        {errors.agreeToTerms.message}
                      </p>
                    )}
                  </div>

                  <Button
                    type="submit"
                    disabled={!isFormValid || isLoading}
                    className="w-full cyber-btn-hero h-14 text-lg font-black disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Creating Account...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-2" />
                        Create Account
                        <ArrowRight className="w-5 h-5 ml-2" />
                      </>
                    )}
                  </Button>
                </form>

                <div className="mt-8 text-center">
                  <p className="text-gray-400">
                    Already have an account?{' '}
                    <Link href="/auth/sign-in" className="text-emerald-400 hover:text-emerald-300 font-medium transition-colors">
                      Sign in instead
                    </Link>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="mt-6 text-center">
            <div className="inline-flex items-center space-x-2 text-sm text-gray-400">
              <Shield className="w-4 h-4 text-emerald-400" />
              <span>Your data is encrypted and secured with institutional-grade security</span>
            </div>
          </div>
        </div>
      </div>

      <OTPModal
        isOpen={showOTP}
        phone={fullPhone}
        onVerify={handleOTPVerify}
        onCancel={handleOTPCancel}
        onResend={handleResendOTP}
        isLoading={isLoading}
      />
    </>
  )
} 