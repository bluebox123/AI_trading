import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  }
})

// Types for our database schema
export interface Profile {
  id: string
  full_name?: string
  phone?: string
  risk_profile?: 'conservative' | 'moderate' | 'aggressive'
  plan: 'free' | 'pro' | 'premium'
  avatar_url?: string
  email_verified?: boolean
  phone_verified?: boolean
  created_at: string
  updated_at: string
}

export interface UserMetadata {
  full_name: string
  phone: string
  risk_profile: 'conservative' | 'moderate' | 'aggressive'
  plan: 'free' | 'pro' | 'premium'
}

// Auth helpers
export const getCurrentUser = async () => {
  const { data: { user }, error } = await supabase.auth.getUser()
  if (error) throw error
  return user
}

export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  })
  if (error) throw error
  return data
}

export const signUp = async (email: string, password: string, metadata: UserMetadata) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: metadata
    }
  })
  if (error) throw error
  return data
}

export const signOut = async () => {
  const { error } = await supabase.auth.signOut()
  if (error) throw error
}

export const updateProfile = async (userId: string, profile: Partial<Profile>) => {
  const { data, error } = await supabase
    .from('profiles')
    .update(profile)
    .eq('id', userId)
    .select()
    .single()
  
  if (error) throw error
  return data
}

export const getProfile = async (userId: string) => {
  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single()
  
  if (error) throw error
  return data
}

// OTP functionality
export const sendOTP = async (phone: string) => {
  const response = await fetch('/api/auth/send-otp', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to send OTP')
  }

  return response.json()
}

export const verifyOTP = async (phone: string, code: string) => {
  const response = await fetch('/api/auth/verify-otp', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ phone, code }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to verify OTP')
  }

  return response.json()
}

// Email verification helpers
export const resendEmailVerification = async () => {
  const user = await getCurrentUser()
  if (!user?.email) throw new Error('No user email found')

  const response = await fetch('/api/auth/resend-verification', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email: user.email }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to resend verification email')
  }

  return response.json()
}

export const checkEmailVerified = (user: any) => {
  return user?.email_confirmed_at !== null || user?.email_confirmed_at !== undefined
} 