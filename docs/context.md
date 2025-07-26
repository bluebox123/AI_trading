# Trading Signals Web App – Developer Guide

> **Purpose**: This document maps the complete user flow, page-level requirements, data interactions, and security considerations for the "Trading Signals" website.  The guide is written for a full-stack web team working with **Next.js 14 (App Router)**, **TypeScript**, **Supabase**, **Tailwind CSS**, and assorted market-data APIs.

---

## 1  High-Level Architecture

| Layer | Responsibility | Key Libraries |
|-------|----------------|---------------|
| **Client (Next.js)** | UI, routing, state, charts | React 18, Next.js 14, TanStack Query, Recharts / TradingView Light-weight embeds |
| **Edge / Serverless Functions** | Protected API proxies, model inference dispatch, OTP verification bridge | Next.js API routes / Vercel Edge Functions |
| **Data** | Auth, user profiles, app tables | Supabase (PostgreSQL + Row Level Security) |
| **ML Model Serving** | Ensemble of internal v3 & v4 models exposed over REST (FastAPI) | Docker-ised micro-service cluster |
| **External Providers** | Market data, news, sentiment | EODHD, NewsFilter.io, Perplexity API |

All traffic enters through the Next.js frontend, which SSRs public pages and hydrates on the client.  Authenticated routes enforce Supabase JWT verification via middleware.

---

## 2  Tech Stack Summary

* **Runtime**   Node 20, Bun as dev runner
* **Frontend**  Next.js 14 (App Router), TypeScript, Tailwind CSS, shadcn/ui component library, Lucide icons
* **State**     TanStack Query + Zustand (local UI state)
* **Auth**      Supabase email + password **plus** mandatory SMS-OTP for new devices
* **Persistence** Supabase Postgres – schemas: `public`, `private`, `storage`
* **CI/CD**     Vercel preview branches → Production

---

## 3  Page & Route Matrix

| Route | Access | Purpose | Key Elements |
|-------|--------|---------|--------------|
| `/` | Public | Marketing landing page | Hero, About, Plan cards, ML-tech teaser, **Sign In CTA** |
| `/auth/sign-in` | Public | Email+pwd entry, *SMS OTP modal*, link to sign-up | Supabase `signInWithPassword()` |
| `/auth/sign-up` | Public | Full registration (email, phone, name, risk profile) | Supabase `signUp()` + profile insert |
| `/dashboard` | Auth | Pro landing | Top 10 Buy/Sell lists, index tickers w/ spark-lines, nav sidebar, user card |
| `/news` | Auth | Aggregated news explorer | Date-range filter pills, keyword search, sentiment badges |
| `/stocks` | Auth | 100-stock watch-list & search | Virtualised table inside scroll box |
| `/stocks/[symbol]` | Auth | Deep-dive analysis | Model score panel, indicator grid, regime tag, BIG rating badge |
| `/engine` | Auth | "Terminal" status feed | Live log stream from model orchestrator |
| `/profile` | Auth | Account settings | Avatar upload, theme switch, plan indicator, password reset |

> **Navigation**: A persistent left sidebar (desktop) / bottom tab bar (mobile) links **Dashboard → News → Stocks → Engine → Profile**.

---

## 4  Authentication & OTP Flow

1. **Sign-Up**  
   a. User submits form → `supabase.auth.signUp()` (email+pwd)  
   b. Backend triggers **SMS OTP** via serverless edge (`/api/otp/send`) using Twilio / Clerk.  
   c. User enters 6-digit code → `/api/otp/verify` → on success, Supabase session cookie set.
2. **Sign-In** – same OTP challenge if device or IP ≠ lastSeen.
3. **Middleware** – `middleware.ts` refreshes and injects Supabase session so that Server Components receive `supabaseUser`.

Row Level Security ensures users may read only their own rows.

---

## 5  Dashboard Requirements

* **Top-10 Lists** – call `/api/rankings?type=buy|sell` → pre-computed nightly job stored in `rankings` table.
* **Index Tickers** – small `<MiniChart>` component shows price spark-line (24 h) via WebSocket (Stockdio) and latest quote.
* **Layout Grid** – use CSS Grid (12-col) with responsive breakpoints; cards auto-re-order on mobile.

---

 6. Stock Sentiment Analysis Page Details
Public Access – /sentiment page serves historical sentiment data from CSV files (5 years intraday + past month predictions)

Core Visualizations – Market sentiment gauge, interactive treemap (market cap sized, sentiment colored), sector heatmap, and top sentiment movers lists

Interactive Analytics – Historical trend charts with zoom/pan, sentiment vs price correlation scatter plots, and real-time predictions dashboard

Filtering System – Time range selectors (intraday to 5 years), stock/sector multi-select, sentiment threshold sliders, and smart search with type-ahead

Data Features – Live sentiment updates every 15 minutes, confidence scores, news volume metrics, social media mentions, and ML-based next-day predictions using 19 CSV data fields per stock

## 7  Stock Analysis Page

1. **Query** `/stocks/[symbol]` SSRs: latest fundamentals + model snapshot.
2. **Rating Banner** – `Strong Buy`/`Buy`/… derived from ensemble probability thresholds. Colour-coded, large typography.
3. **Information Grid**
   * Model Score
   * Technical Indicators (RSI 14, MACD, SMA 50/200, ATR)
   * Market Regime (Bull/Bear/Sideways)
   * News Sentiment 7-day average
4. **Expandable Sections** – accordions for raw indicator series, recent insider trades, etc.

---

## 8  Engine Page (Fun "Terminal")

* `<EngineConsole>` React component streams server-sent events (`/api/engine/logs`).
* Fade-in typing animation (Framer Motion) for log lines such as:
  * `> v3 & v4 ensemble loaded`
  * `> Sentiment analyser initialised`
  * `> sentiment fetched – 117 stocks (1 m)`

---

## 9  Data Model (Supabase)

```sql
-- auth.users is managed by Supabase
create table profiles (
  id uuid references auth.users primary key,
  full_name text,
  phone varchar(15),
  plan       text default 'pro',
  avatar_url text,
  created_at timestamptz default now()
);

create table rankings (
  trade_date date,
  direction  text check (direction in ('buy','sell')),
  symbol     text,
  score      numeric,
  primary key (trade_date, direction, symbol)
);

create table news_items (
  id          bigint primary key,
  published_at timestamptz,
  title text,
  url   text,
  sentiment text,
  impact jsonb -- { sectors: [...], symbols: [...] }
);
```
RLS policies restrict `profiles` updates to owner; `rankings` and `news_items` are read-only for all authenticated users.

---

## 10  Security Considerations

* **Password + OTP** = MFA
* *httpOnly cookies* for Supabase session
* **Content Security Policy** – restrict script src, connect src to trusted APIs
* **Rate limiting** on `/api/otp/*`, `/api/news/*`
* **.env.local** never committed; use Vercel Encrypted Env Vars

---

## 11  Component Directory Sketch

```
/src
 ├─ app
 │   ├─ layout.tsx
 │   ├─ page.tsx            # Home
 │   ├─ dashboard/
 │   │   └─ page.tsx
 │   ├─ news/
 │   │   ├─ page.tsx
 │   │   └─ NewsCard.tsx
 │   ├─ stocks/
 │   │   ├─ page.tsx        # list view
 │   │   └─ [symbol]/page.tsx
 │   └─ engine/page.tsx
 ├─ components/
 │   ├─ Sidebar.tsx
 │   ├─ MiniChart.tsx
 │   ├─ RatingBadge.tsx
 │   └─ EngineConsole.tsx
 ├─ lib/
 │   ├─ supabaseClient.ts
 │   └─ api.ts              # typed fetch wrappers
 └─ styles/tailwind.css
```

---

## 12  Deployment & Environment

* **Vercel** – GitHub integration, preview deployments per PR.
* **Supabase** – free tier during dev, point to prod instance via env.
* **Cron Jobs** – Vercel Cron → `/api/cron/pull-signals` (04:15 IST daily).
* **Secrets** – `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `EODHD_API_KEY`, `NEWS_API_KEY`, `SENTIMENT_API_KEY`, `JWT_SECRET`.

---

## 13  Open Items / Roadmap

1. Tier-based feature gating (`plan` column) and Stripe billing integration.
2. Web-socket live sentiment push for intraday news.
3. Mobile PWA install prompt (Next-PWA plugin).
4. Accessibility audit – WCAG 2.1 AA.

---

### Happy Building!  
This guide should give your engineering team a 360° view of the required flows, endpoints, and implementation details for the **Trading Signals** platform.


**IMPORTANT: Environment variables should be set securely and never committed to version control.**

Example environment variables structure (DO NOT include actual keys):

```
EODHD_API_KEY=your_eodhd_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
NEWS_API_KEY=your_news_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
SUPABASE_JWT_SECRET=your_supabase_jwt_secret_here
```

**Security Note:** Always use environment variables for sensitive data and never hardcode API keys in source code.

