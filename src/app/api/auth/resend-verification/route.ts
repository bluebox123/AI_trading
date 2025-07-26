import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email } = body

    if (!email) {
      return NextResponse.json(
        { success: false, message: 'Email is required' },
        { status: 400 }
      )
    }

    // Use service role key for admin operations
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    const { error } = await supabase.auth.resend({
      type: 'signup',
      email: email,
    })

    if (error) {
      console.error('Email verification error:', error)
      return NextResponse.json(
        { success: false, message: error.message },
        { status: 400 }
      )
    }

    return NextResponse.json({
      success: true,
      message: 'Verification email sent successfully'
    })

  } catch (error) {
    console.error('Resend verification error:', error)
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    )
  }
} 