import { NextRequest, NextResponse } from 'next/server'
import { 
  otpStore, 
  rateLimitStore, 
  RATE_LIMIT, 
  RATE_LIMIT_WINDOW, 
  OTP_EXPIRY 
} from '../shared-store'

function generateOTP(): string {
  return Math.floor(100000 + Math.random() * 900000).toString()
}

function isRateLimited(phone: string): boolean {
  const now = Date.now()
  const record = rateLimitStore.get(phone)
  
  if (!record || now > record.resetTime) {
    rateLimitStore.set(phone, { count: 1, resetTime: now + RATE_LIMIT_WINDOW })
    return false
  }
  
  if (record.count >= RATE_LIMIT) {
    return true
  }
  
  record.count++
  return false
}

function validatePhoneNumber(phone: string): boolean {
  // Basic phone validation - can be enhanced
  const phoneRegex = /^[+]?[\d\s\-()]{10,15}$/
  return phoneRegex.test(phone)
}

// Mock SMS sending function - replace with actual SMS service
async function sendSMS(phone: string, code: string): Promise<boolean> {
  try {
    // In development, log the OTP instead of sending SMS
    console.log(`🔐 OTP for ${phone}: ${code}`)
    console.log(`📱 DEVELOPMENT MODE: Check this terminal for your verification code!`)
    
    // TODO: Integrate with actual SMS service (Twilio, MSG91, etc.)
    // Example with Twilio:
    // const client = twilio(accountSid, authToken);
    // await client.messages.create({
    //   body: `Your TradingSignals verification code is: ${code}. Valid for 5 minutes.`,
    //   from: '+1234567890',
    //   to: phone
    // });
    
    return true
  } catch (error) {
    console.error('SMS sending failed:', error)
    return false
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { phone } = body

    // Debug logging
    console.log('📤 OTP Send Request:')
    console.log('📞 Phone:', phone)

    // Validate input
    if (!phone || typeof phone !== 'string') {
      console.log('❌ Invalid phone input')
      return NextResponse.json(
        { success: false, message: 'Phone number is required' },
        { status: 400 }
      )
    }

    if (!validatePhoneNumber(phone)) {
      console.log('❌ Invalid phone format')
      return NextResponse.json(
        { success: false, message: 'Invalid phone number format' },
        { status: 400 }
      )
    }

    // Check rate limiting
    if (isRateLimited(phone)) {
      console.log('⏰ Rate limited')
      return NextResponse.json(
        { success: false, message: 'Too many OTP requests. Please try again later.' },
        { status: 429 }
      )
    }

    // Generate and store OTP
    const code = generateOTP()
    const expires = Date.now() + OTP_EXPIRY
    
    console.log('🔐 Generated OTP:', code)
    console.log('⏰ Expires at:', new Date(expires))
    
    otpStore.set(phone, { code, expires, attempts: 0 })
    
    console.log('💾 OTP stored for phone:', phone)
    console.log('🗃️ Current OTP Store:', Object.fromEntries(otpStore.entries()))

    // Send OTP
    const sent = await sendSMS(phone, code)
    
    if (!sent) {
      console.log('❌ Failed to send SMS')
      return NextResponse.json(
        { success: false, message: 'Failed to send OTP. Please try again.' },
        { status: 500 }
      )
    }

    console.log('✅ OTP sent successfully')
    return NextResponse.json({
      success: true,
      message: 'OTP sent successfully',
      expiresIn: OTP_EXPIRY / 1000 // Return expiry in seconds
    })

  } catch (error) {
    console.error('❌ Send OTP error:', error)
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    )
  }
} 