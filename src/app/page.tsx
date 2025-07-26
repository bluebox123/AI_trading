'use client';

import React from 'react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  Brain, 
  Shield, 
  Zap, 
  BarChart3, 
  Newspaper,
  CheckCircle,
  ArrowRight,
  Menu,
  X,
  ChevronRight,
  Star,
  Activity,
  Target,
  Users,
  Play,
  Globe
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'Advanced v3 & v4 ML models that combine price data, technical indicators, and market sentiment for precise buy/sell signals.'
  },
  {
    icon: BarChart3,
    title: '100 NSE Stocks',
    description: 'Complete coverage of largecap and midcap NSE stocks with real-time analysis and rating updates.'
  },
  {
    icon: Newspaper,
    title: 'News Sentiment',
    description: 'Real-time news analysis and sentiment scoring to factor market-moving events into trading decisions.'
  },
  {
    icon: Shield,
    title: 'Risk Management',
    description: 'Built-in portfolio risk assessment and position sizing recommendations for institutional-grade safety.'
  },
  {
    icon: Zap,
    title: 'Real-Time Signals',
    description: 'Live trading signals updated every 5 minutes during market hours with confidence scoring.'
  },
  {
    icon: TrendingUp,
    title: 'Market Regime Detection',
    description: 'Automatic detection of Bull, Bear, and Sideways market conditions to adjust strategy accordingly.'
  }
]

const plans = [
  {
    name: 'Pro',
    price: '₹2,999',
    period: '/month',
    description: 'Perfect for individual traders',
    features: [
      'Access to all 100 stock signals',
      'Real-time news sentiment analysis',
      'Technical indicator dashboard',
      'Mobile app access',
      'Email alerts',
      'Basic risk management'
    ],
    badge: null,
    cta: 'Start Free Trial'
  },
  {
    name: 'Premium',
    price: '₹4,999',
    period: '/month',
    description: 'For serious institutional traders',
    features: [
      'Everything in Pro',
      'Advanced portfolio optimization',
      'API access for algorithmic trading',
      'Custom risk parameters',
      'Priority support',
      'Advanced backtesting tools'
    ],
    badge: 'Most Popular',
    cta: 'Start Free Trial'
  }
]

export default function LandingPage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Navigation Header */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrollY > 50 ? 'glass backdrop-blur-md' : 'bg-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-2 logo-float">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold hero-gradient-text">TradingSignals</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <Link href="#features" className="text-slate-300 hover:text-emerald-400 transition-colors">Features</Link>
              <Link href="#technology" className="text-slate-300 hover:text-emerald-400 transition-colors">Technology</Link>
              <Link href="#pricing" className="text-slate-300 hover:text-emerald-400 transition-colors">Pricing</Link>
              <Link href="#about" className="text-slate-300 hover:text-emerald-400 transition-colors">About</Link>
              <Link href="/auth/sign-in" className="btn-emerald px-6 py-2 rounded-lg text-white font-medium hover:shadow-lg transition-all">
                Sign In
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden p-2 rounded-lg hover:bg-slate-800 transition-colors"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>

          {/* Mobile Menu */}
          {isMenuOpen && (
            <div className="md:hidden glass backdrop-blur-md border-t border-slate-700 py-4">
              <div className="flex flex-col space-y-3">
                <Link href="#features" className="text-slate-300 hover:text-emerald-400 transition-colors px-4 py-2">Features</Link>
                <Link href="#technology" className="text-slate-300 hover:text-emerald-400 transition-colors px-4 py-2">Technology</Link>
                <Link href="#pricing" className="text-slate-300 hover:text-emerald-400 transition-colors px-4 py-2">Pricing</Link>
                <Link href="#about" className="text-slate-300 hover:text-emerald-400 transition-colors px-4 py-2">About</Link>
                <Link href="/auth/sign-in" className="btn-emerald mx-4 px-6 py-3 rounded-lg text-white font-medium text-center">
                  Sign In
                </Link>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 gradient-bg"></div>
        <div className="absolute inset-0 textured-bg opacity-30"></div>
        <div className="absolute inset-0 chart-pattern"></div>
        
        {/* Floating Elements */}
        <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-emerald-400 rounded-full float opacity-60"></div>
        <div className="absolute top-1/3 right-1/4 w-3 h-3 bg-emerald-300 rounded-full float opacity-40" style={{animationDelay: '2s'}}></div>
        <div className="absolute bottom-1/3 left-1/3 w-1 h-1 bg-emerald-500 rounded-full float opacity-50" style={{animationDelay: '4s'}}></div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="max-w-4xl mx-auto">
            {/* Main Headline */}
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="block">Institutional-Grade</span>
              <span className="block hero-gradient-text">Trading Signals</span>
              <span className="block text-slate-300">that Actually Work</span>
            </h1>

            {/* Subheadline */}
            <p className="text-xl md:text-2xl text-slate-300 mb-4">
              For <span className="text-emerald-400 font-semibold">Day Traders</span>, 
              <span className="text-emerald-400 font-semibold"> Portfolio Managers</span> and 
              <span className="text-emerald-400 font-semibold"> Financial Institutions</span>
            </p>

            {/* Description */}
            <p className="text-lg text-slate-400 mb-12 max-w-3xl mx-auto leading-relaxed">
              Powered by advanced v3 & v4 ML models combining market data, technical indicators, 
              news sentiment, and market regime analysis to deliver actionable buy/hold/sell signals 
              for 100+ NSE stocks.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link 
                href="/auth/sign-up" 
                className="btn-emerald px-8 py-4 rounded-xl text-lg font-semibold text-white shadow-2xl hover:shadow-emerald-500/25 transition-all flex items-center group"
              >
                Get Started
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              
              <button className="px-8 py-4 rounded-xl text-lg font-semibold border-2 border-slate-600 text-slate-300 hover:border-emerald-400 hover:text-emerald-400 transition-all flex items-center group">
                <Play className="mr-2 w-5 h-5 group-hover:scale-110 transition-transform" />
                View Demo
              </button>
            </div>

            {/* Trust Badge */}
            <div className="mt-16 flex items-center justify-center space-x-2 text-slate-400">
              <Shield className="w-5 h-5 text-emerald-400" />
              <span className="text-sm">Trusted by 500+ traders & institutions</span>
            </div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <ChevronRight className="w-6 h-6 text-slate-400 rotate-90" />
        </div>
      </section>

      {/* Trust Indicators */}
      <section className="py-16 bg-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">Trusted by Industry Leaders</h2>
            <p className="text-slate-400">Join thousands of professionals who rely on our signals</p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <div className="text-center">
              <div className="text-4xl font-bold hero-gradient-text mb-2">95%+</div>
              <div className="text-slate-300 font-medium">Accuracy Rate</div>
              <div className="text-sm text-slate-400">Validated signals</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold hero-gradient-text mb-2">100+</div>
              <div className="text-slate-300 font-medium">NSE Stocks</div>
              <div className="text-sm text-slate-400">Real-time coverage</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold hero-gradient-text mb-2">24/7</div>
              <div className="text-slate-300 font-medium">Analysis</div>
              <div className="text-sm text-slate-400">Market monitoring</div>
            </div>
          </div>

          {/* Mock Client Logos */}
          <div className="flex flex-wrap justify-center items-center gap-8 opacity-50">
            <div className="flex items-center space-x-2 text-slate-400">
              <Globe className="w-6 h-6" />
              <span className="font-semibold">Global Fund</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-400">
              <Target className="w-6 h-6" />
              <span className="font-semibold">Alpha Capital</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-400">
              <Activity className="w-6 h-6" />
              <span className="font-semibold">Trade Pro</span>
            </div>
            <div className="flex items-center space-x-2 text-slate-400">
              <Users className="w-6 h-6" />
              <span className="font-semibold">Wealth Partners</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Powered by <span className="hero-gradient-text">Advanced Technology</span>
            </h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Our sophisticated ML models analyze multiple data streams to provide you with 
              the most accurate trading signals in the market.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {/* Feature Cards */}
            <div className="bg-slate-800 rounded-xl p-6 card-hover border border-slate-700">
              <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">AI-Powered Analysis</h3>
              <p className="text-slate-400">
                Advanced neural networks process market data, news sentiment, and technical indicators simultaneously.
              </p>
            </div>

            <div className="bg-slate-800 rounded-xl p-6 card-hover border border-slate-700">
              <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center mb-4">
                <Activity className="w-6 h-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Real-time Sentiment</h3>
              <p className="text-slate-400">
                Live news analysis and social sentiment tracking to capture market mood shifts instantly.
              </p>
            </div>

            <div className="bg-slate-800 rounded-xl p-6 card-hover border border-slate-700">
              <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Technical Indicators</h3>
              <p className="text-slate-400">
                50+ technical indicators combined with market regime detection for precise entry and exit points.
              </p>
            </div>

            <div className="bg-slate-800 rounded-xl p-6 card-hover border border-slate-700">
              <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Risk Management</h3>
              <p className="text-slate-400">
                Built-in portfolio optimization and risk assessment to protect your capital.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Showcase */}
      <section id="technology" className="py-20 bg-slate-800/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-4xl font-bold text-white mb-6">
                Built on <span className="hero-gradient-text">Cutting-Edge ML</span>
              </h2>
              <p className="text-xl text-slate-400 mb-8">
                Our ensemble of v3 & v4 models delivers institutional-grade accuracy 
                through advanced temporal analysis and market regime detection.
              </p>

              {/* Tech Pills */}
              <div className="flex flex-wrap gap-3 mb-8">
                {['v3 & v4 Ensemble Models', 'Market Regime Detection', 'Sentiment Analysis', 'Technical Indicators'].map((tech) => (
                  <span key={tech} className="px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-full text-sm font-medium border border-emerald-500/30">
                    {tech}
                  </span>
                ))}
              </div>

              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Multi-modal data fusion</span>
                </div>
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Real-time model orchestration</span>
                </div>
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Automated backtesting & validation</span>
                </div>
              </div>
            </div>

            {/* Terminal Preview */}
            <div className="terminal-text rounded-xl p-6 font-mono text-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                </div>
                <span className="text-slate-400">model_orchestrator.py</span>
              </div>
              
              <div className="space-y-2 text-emerald-400">
                <div>{'>'} Loading v4 ensemble models...</div>
                <div className="text-slate-400">✓ Temporal causality model loaded</div>
                <div className="text-slate-400">✓ Market regime detector initialized</div>
                <div className="text-slate-400">✓ News sentiment analyzer ready</div>
                <div>{'>'} Processing RELIANCE.NSE...</div>
                <div className="text-yellow-400">Signal: STRONG_BUY (confidence: 0.94)</div>
                <div className="text-blue-400">Risk Score: 0.23 (LOW)</div>
                <div className="text-emerald-400">Entry: ₹2,847.50 | Target: ₹3,120.00</div>
                <div>{'>'} <span className="animate-pulse">█</span></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 bg-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Choose Your Plan</h2>
            <p className="text-xl text-slate-400">Start free, scale as you grow</p>
            <div className="inline-flex items-center mt-4 px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-full text-sm">
              <Zap className="w-4 h-4 mr-2" />
              Full functionality shown - billing coming soon
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Free Plan */}
            <div className="bg-slate-800 rounded-xl p-8 border border-slate-700">
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-white mb-2">Free</h3>
                <div className="text-4xl font-bold text-white mb-2">₹0</div>
                <div className="text-slate-400">Perfect for beginners</div>
              </div>
              
              <ul className="space-y-4 mb-8">
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">5 stocks coverage</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Basic signals</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Email alerts</span>
                </li>
              </ul>
              
              <button className="w-full py-3 rounded-lg border border-slate-600 text-slate-300 hover:border-emerald-400 hover:text-emerald-400 transition-all">
                Get Started
              </button>
            </div>

            {/* Pro Plan */}
            <div className="bg-slate-800 rounded-xl p-8 border-2 border-emerald-500 relative">
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <span className="bg-emerald-500 text-white px-4 py-1 rounded-full text-sm font-medium">Most Popular</span>
              </div>
              
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-white mb-2">Pro</h3>
                <div className="text-4xl font-bold hero-gradient-text mb-2">₹2,999</div>
                <div className="text-slate-400">Per month</div>
              </div>
              
              <ul className="space-y-4 mb-8">
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">100+ NSE stocks</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Advanced v3/v4 signals</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Real-time alerts</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Risk analysis</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Portfolio optimization</span>
                </li>
              </ul>
              
              <button className="w-full btn-emerald py-3 rounded-lg text-white font-medium">
                Start Pro Trial
              </button>
            </div>

            {/* Enterprise Plan */}
            <div className="bg-slate-800 rounded-xl p-8 border border-slate-700">
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-white mb-2">Enterprise</h3>
                <div className="text-4xl font-bold text-white mb-2">Custom</div>
                <div className="text-slate-400">For institutions</div>
              </div>
              
              <ul className="space-y-4 mb-8">
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Custom models</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">API access</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">White-label solution</span>
                </li>
                <li className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-slate-300">Dedicated support</span>
                </li>
              </ul>
              
              <button className="w-full py-3 rounded-lg border border-slate-600 text-slate-300 hover:border-emerald-400 hover:text-emerald-400 transition-all">
                Contact Sales
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-800 border-t border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {/* Company Info */}
            <div className="col-span-1 md:col-span-2">
              <Link href="/" className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold hero-gradient-text">TradingSignals</span>
              </Link>
              <p className="text-slate-400 mb-4 max-w-md">
                Institutional-grade trading signals powered by advanced ML models for the modern trader.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">
                  <Globe className="w-5 h-5" />
                </a>
                <a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">
                  <Activity className="w-5 h-5" />
                </a>
                <a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">
                  <Target className="w-5 h-5" />
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h4 className="text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2">
                <li><a href="#features" className="text-slate-400 hover:text-emerald-400 transition-colors">Features</a></li>
                <li><a href="#pricing" className="text-slate-400 hover:text-emerald-400 transition-colors">Pricing</a></li>
                <li><a href="/auth/sign-in" className="text-slate-400 hover:text-emerald-400 transition-colors">Sign In</a></li>
                <li><a href="/auth/sign-up" className="text-slate-400 hover:text-emerald-400 transition-colors">Get Started</a></li>
              </ul>
            </div>

            {/* Legal */}
            <div>
              <h4 className="text-white font-semibold mb-4">Legal</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">Terms of Service</a></li>
                <li><a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">Risk Disclosure</a></li>
                <li><a href="#" className="text-slate-400 hover:text-emerald-400 transition-colors">Contact</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-slate-700 pt-8 mt-8">
            <div className="flex flex-col md:flex-row justify-between items-center">
              <p className="text-slate-400 text-sm">
                © 2024 TradingSignals. All rights reserved.
              </p>
              <p className="text-slate-400 text-sm mt-2 md:mt-0">
                Built with institutional-grade security and reliability.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
} 