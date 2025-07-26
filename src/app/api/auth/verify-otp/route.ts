import { NextRequest, NextResponse } from 'next/server'
import { otpStore, MAX_ATTEMPTS } from '../shared-store'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { phone, code } = body

    // Debug logging
    console.log('🔍 OTP Verification Request:')
    console.log('📞 Phone:', phone)
    console.log('🔢 Code:', code)
    console.log('🗃️ Current OTP Store:', Object.fromEntries(otpStore.entries()))

    // Validate input
    if (!phone || !code || typeof phone !== 'string' || typeof code !== 'string') {
      console.log('❌ Invalid input - missing phone or code')
      return NextResponse.json(
        { success: false, message: 'Phone number and code are required' },
        { status: 400 }
      )
    }

    // Check if OTP exists
    const record = otpStore.get(phone)
    
    if (!record) {
      console.log('❌ No OTP record found for phone:', phone)
      console.log('📋 Available phones in store:', Array.from(otpStore.keys()))
      return NextResponse.json(
        { success: false, message: 'No OTP found. Please request a new one.' },
        { status: 404 }
      )
    }

    console.log('✅ OTP record found:', record)

    // Check if OTP is expired
    if (Date.now() > record.expires) {
      console.log('⏰ OTP expired')
      otpStore.delete(phone)
      return NextResponse.json(
        { success: false, message: 'OTP has expired. Please request a new one.' },
        { status: 410 }
      )
    }

    // Check max attempts
    if (record.attempts >= MAX_ATTEMPTS) {
      console.log('🚫 Max attempts reached')
      otpStore.delete(phone)
      return NextResponse.json(
        { success: false, message: 'Too many failed attempts. Please request a new OTP.' },
        { status: 429 }
      )
    }

    // Verify code
    console.log('🔍 Comparing codes:')
    console.log('📱 Stored code:', record.code)
    console.log('🔢 Submitted code:', code.toString())
    
    if (record.code !== code.toString()) {
      record.attempts++
      console.log('❌ Code mismatch, attempts:', record.attempts)
      
      const remainingAttempts = MAX_ATTEMPTS - record.attempts
      
      if (remainingAttempts === 0) {
        otpStore.delete(phone)
        return NextResponse.json(
          { success: false, message: 'Invalid code. Maximum attempts reached. Please request a new OTP.' },
          { status: 401 }
        )
      }
      
      return NextResponse.json(
        { 
          success: false, 
          message: `Invalid code. ${remainingAttempts} attempt${remainingAttempts > 1 ? 's' : ''} remaining.` 
        },
        { status: 401 }
      )
    }

    // Success - remove OTP from store
    console.log('✅ OTP verification successful!')
    otpStore.delete(phone)

    return NextResponse.json({
      success: true,
      verified: true,
      message: 'Phone number verified successfully'
    })

  } catch (error) {
    console.error('❌ Verify OTP error:', error)
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    )
  }
} 