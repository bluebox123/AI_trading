#!/usr/bin/env node

/**
 * Generate a secure JWT secret for the TradingSignals application
 * Run this script with: node scripts/generate-secret.js
 */

const crypto = require('crypto');

// Generate a random 64-character hex string
const secret = crypto.randomBytes(32).toString('hex');

console.log('🔐 Generated JWT Secret:');
console.log('='.repeat(50));
console.log(secret);
console.log('='.repeat(50));
console.log('');
console.log('📝 Add this to your .env.local file:');
console.log(`JWT_SECRET=${secret}`);
console.log('');
console.log('⚠️  Keep this secret secure and never commit it to version control!'); 