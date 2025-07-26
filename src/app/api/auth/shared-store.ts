// Shared store for OTP data across API routes
// In production, replace with Redis or a proper database

export interface OTPRecord {
  code: string;
  expires: number;
  attempts: number;
}

export interface RateLimitRecord {
  count: number;
  resetTime: number;
}

// Shared stores
export const otpStore = new Map<string, OTPRecord>();
export const rateLimitStore = new Map<string, RateLimitRecord>();

// Configuration
export const RATE_LIMIT = 5;
export const RATE_LIMIT_WINDOW = 60 * 60 * 1000; // 1 hour
export const OTP_EXPIRY = 5 * 60 * 1000; // 5 minutes
export const MAX_ATTEMPTS = 3;

// Clean up expired OTPs
export function cleanupExpiredOTPs() {
  const now = Date.now();
  for (const [phone, record] of otpStore.entries()) {
    if (now > record.expires) {
      otpStore.delete(phone);
    }
  }
}

// Clean up expired rate limits
export function cleanupExpiredRateLimits() {
  const now = Date.now();
  for (const [phone, record] of rateLimitStore.entries()) {
    if (now > record.resetTime) {
      rateLimitStore.delete(phone);
    }
  }
}

// Run cleanup every 10 minutes
setInterval(() => {
  cleanupExpiredOTPs();
  cleanupExpiredRateLimits();
}, 10 * 60 * 1000); 