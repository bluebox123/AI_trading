# Environment Setup Guide

## Quick Start

1. **Copy the template**: The `.env.local` file has been created with placeholder values
2. **Replace the placeholders**: Update the values with your actual API keys
3. **Restart the server**: Run `npm run dev` to pick up the new environment variables

## Required Environment Variables

### 🔐 Supabase Configuration (REQUIRED)
You need to create a Supabase project at https://supabase.com

```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

**How to get these:**
1. Go to https://supabase.com and create a new project
2. Go to Settings → API
3. Copy the "Project URL" and "anon public" key
4. Copy the "service_role" key (keep this secret!)

### 📊 Market Data APIs (OPTIONAL for development)

#### EODHD API (Free tier available)
```bash
EODHD_API_KEY=your_eodhd_api_key_here
```
- Sign up at https://eodhd.com
- Free tier: 1,000 requests/day

#### Alpha Vantage API (Free tier available)
```bash
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```
- Sign up at https://www.alphavantage.co
- Free tier: 5 requests/minute, 500 requests/day

### 🤖 AI/News APIs (OPTIONAL for development)

#### Perplexity API
```bash
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```
- Sign up at https://www.perplexity.ai
- Used for news sentiment analysis

#### News API
```bash
NEWS_API_KEY=your_news_api_key_here
```
- Sign up at https://newsapi.org
- Free tier: 1,000 requests/day

### 💰 Trading APIs (OPTIONAL for development)

#### Alpaca Trading API
```bash
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```
- Sign up at https://alpaca.markets
- Paper trading available for free

### ☁️ AWS Configuration (OPTIONAL)
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
```

### 🔑 JWT Secret (REQUIRED)
```bash
JWT_SECRET=your_jwt_secret_here
```
Generate a random string (32+ characters) for session management.

## Development vs Production

### Development (Local)
- Use free API tiers where possible
- Mock data is available for testing
- OTP codes are logged to console instead of SMS

### Production
- Upgrade to paid API tiers for higher limits
- Set up proper SMS service (Twilio, MSG91, etc.)
- Use environment variables in your hosting platform (Vercel, etc.)

## Testing Without API Keys

The app includes mock data and fallbacks, so you can test most features without setting up all API keys:

1. **Supabase**: Required for authentication
2. **JWT_SECRET**: Required for sessions
3. **Others**: Optional - app will use mock data

## Security Notes

⚠️ **IMPORTANT**: Never commit your `.env.local` file to version control!

- ✅ `.env.local` is already in `.gitignore`
- ✅ Use different API keys for development and production
- ✅ Rotate API keys regularly
- ✅ Use environment variables in production hosting platforms

## Troubleshooting

### "Missing Supabase environment variables" error
- Make sure `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` are set
- Restart the development server after adding environment variables

### API rate limiting
- Check your API usage limits
- Consider upgrading to paid tiers for production

### Authentication issues
- Verify your Supabase project is set up correctly
- Check that RLS (Row Level Security) policies are configured

## Next Steps

1. Set up your Supabase project
2. Add your API keys to `.env.local`
3. Run `npm run dev` to start the development server
4. Visit http://localhost:3000 to test the application

Need help? Check the main README.md for more detailed setup instructions. 