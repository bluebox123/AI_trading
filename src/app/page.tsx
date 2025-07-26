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
  Star,
  Activity,
  Target,
  Play,
  Globe,
  Sparkles,
  Rocket,
  Eye,
  Lock,
  DollarSign
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Predictions',
    description: 'Advanced neural networks analyze 10,000+ data points per second to predict market movements with unprecedented accuracy.',
    gradient: 'from-purple-400 to-pink-400'
  },
  {
    icon: Zap,
    title: 'Lightning-Fast Signals',
    description: 'Get buy/sell signals in under 50ms. Our infrastructure processes market data faster than human reaction time.',
    gradient: 'from-yellow-400 to-orange-400'
  },
  {
    icon: Shield,
    title: 'Risk-Adjusted Returns',
    description: 'Intelligent position sizing and risk management that adapts to market volatility in real-time.',
    gradient: 'from-emerald-400 to-cyan-400'
  },
  {
    icon: BarChart3,
    title: '100+ NSE Stocks',
    description: 'Complete coverage of large-cap and mid-cap stocks with sector-specific model optimization.',
    gradient: 'from-blue-400 to-purple-400'
  },
  {
    icon: Newspaper,
    title: 'Sentiment Intelligence',
    description: 'Real-time news sentiment analysis using NLP models trained on 5 years of financial news data.',
    gradient: 'from-pink-400 to-red-400'
  },
  {
    icon: TrendingUp,
    title: 'Market Regime Detection',
    description: 'Automatically detects Bull, Bear, and Sideways markets to adjust trading strategies dynamically.',
    gradient: 'from-indigo-400 to-blue-400'
  }
]

const stats = [
  { number: '78.4%', label: 'Signal Accuracy', icon: Target },
  { number: '2.4x', label: 'Risk-Adjusted Returns', icon: TrendingUp },
  { number: '50ms', label: 'Signal Latency', icon: Zap },
  { number: '₹50Cr+', label: 'AUM Managed', icon: DollarSign }
]

const testimonials = [
  {
    name: "Rajesh Mehta",
    role: "Senior Portfolio Manager",
    company: "HDFC Securities",
    content: "TradingSignals revolutionized our algorithmic trading. The AI predictions are remarkably accurate, and the risk management features saved us millions during the March volatility.",
    rating: 5,
    avatar: "RM"
  },
  {
    name: "Priya Sharma",
    role: "Quantitative Analyst",
    company: "Kotak Institutional Equities",
    content: "The ensemble learning approach and real-time sentiment analysis give us an edge that traditional indicators simply can't match. Our Sharpe ratio improved by 40%.",
    rating: 5,
    avatar: "PS"
  },
  {
    name: "Amit Kumar",
    role: "Head of Trading",
    company: "IIFL Securities",
    content: "Finally, a signal service that understands institutional needs. The API integration was seamless, and the support team knows what they're talking about.",
    rating: 5,
    avatar: "AK"
  }
]

export default function LandingPage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div className="min-h-screen bg-black text-white relative overflow-x-hidden">
      {/* Global Mouse Glow */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div 
          className="mouse-glow"
          style={{
            left: mousePosition.x - 300,
            top: mousePosition.y - 300,
          }}
        ></div>
      </div>

      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrollY > 50 ? 'nav-glass backdrop-blur-xl border-b border-emerald-500/20' : 'bg-transparent'
      }`}>
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-3 logo-hover group">
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl flex items-center justify-center glow-effect">
                  <TrendingUp className="w-6 h-6 text-black font-bold" />
                </div>
                <div className="absolute -inset-1 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl blur opacity-30 group-hover:opacity-60 transition-opacity"></div>
              </div>
              <span className="text-2xl font-bold neon-text">TradingSignals</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              <Link href="#features" className="nav-link">Features</Link>
              <Link href="#technology" className="nav-link">Technology</Link>
              <Link href="#pricing" className="nav-link">Pricing</Link>
              <Link href="#testimonials" className="nav-link">Reviews</Link>
              <Link href="/auth/sign-in" className="cyber-btn-primary">
                <span>Access Platform</span>
                <ArrowRight className="w-4 h-4 ml-2 transition-transform group-hover:translate-x-1" />
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden p-2 rounded-lg hover:bg-white/5 transition-colors"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center pt-20">
        {/* Hero Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="trading-grid"></div>
          <div className="neural-network"></div>
        </div>
        <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center max-w-5xl mx-auto">
            {/* Floating Badge */}
            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 backdrop-blur-xl border border-emerald-500/30 rounded-full px-6 py-3 mb-8 animate-float">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-medium">AI-Powered Trading Revolution</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight">
              <span className="block mb-4">INSTITUTIONAL</span>
              <span className="block neon-pulse-text glitch-effect" data-text="TRADING">TRADING</span>
              <span className="block text-gray-300">SIGNALS</span>
            </h1>

            {/* Animated Subtitle */}
            <div className="text-xl md:text-2xl text-gray-400 mb-4 typing-animation">
              <span>For </span>
              <span className="neon-text">Hedge Funds</span>
              <span>, </span>
              <span className="neon-text">Portfolio Managers</span>
              <span> & </span>
              <span className="neon-text">Institutional Traders</span>
            </div>

            {/* Description */}
            <p className="text-lg text-gray-400 mb-12 max-w-4xl mx-auto leading-relaxed animate-fade-up-delay">
              Harness the power of advanced v4 ML models, real-time sentiment analysis, and quantum-inspired algorithms 
              to generate profitable trading signals with institutional-grade precision and risk management.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16">
              <Link href="/auth/sign-up" className="cyber-btn-hero group">
                <span className="relative z-10">Start Free Trial</span>
                <Rocket className="w-5 h-5 ml-3 transition-transform group-hover:translate-x-1" />
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-xl opacity-0 group-hover:opacity-20 transition-opacity"></div>
              </Link>
              
              <button className="cyber-btn-secondary group">
                <Play className="w-5 h-5 mr-3" />
                <span>Watch Live Demo</span>
              </button>
            </div>

            {/* Live Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-16">
              {stats.map((stat, index) => (
                <div key={index} className="stat-card group">
                  <div className="flex items-center justify-center mb-3">
                    <stat.icon className="w-8 h-8 text-emerald-400" />
                  </div>
                  <div className="text-3xl font-bold neon-text mb-2 counter-animation">{stat.number}</div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32 relative">
        {/* Floating Particles Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="floating-particles"></div>
          <div className="gradient-waves"></div>
        </div>
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-5xl md:text-6xl font-black mb-8">
              <span className="neon-pulse-text">NEXT-GENERATION</span><br />
              <span className="text-white">TRADING TECHNOLOGY</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Built for professionals who demand cutting-edge technology, lightning-fast execution, and institutional-grade reliability.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="feature-card-advanced group">
                <div className="relative">
                  <div className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 feature-icon-glow`}>
                    <feature.icon className="w-8 h-8 text-black font-bold" />
                  </div>
                  <div className={`absolute -inset-2 bg-gradient-to-br ${feature.gradient} rounded-2xl blur opacity-0 group-hover:opacity-30 transition-opacity duration-500`}></div>
                </div>
                
                <h3 className="text-2xl font-bold mb-4 text-white group-hover:text-emerald-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-400 leading-relaxed group-hover:text-gray-300 transition-colors">
                  {feature.description}
                </p>

                <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-emerald-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Showcase */}
      <section id="technology" className="py-32 relative">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-20 items-center">
            <div className="space-y-8">
              <h2 className="text-5xl font-black leading-tight">
                <span className="neon-pulse-text">QUANTUM-INSPIRED</span><br />
                <span className="text-white">AI ARCHITECTURE</span>
              </h2>
              
              <p className="text-xl text-gray-400 leading-relaxed">
                Our proprietary ensemble learning models combine transformer architectures, 
                reinforcement learning, and quantum-inspired optimization algorithms.
              </p>

              <div className="space-y-6">
                {[
                  { 
                    title: "Multimodal Transformer Networks", 
                    desc: "Process price action, volume, news sentiment, and macro indicators simultaneously",
                    icon: Brain
                  },
                  { 
                    title: "Real-time Market Regime Detection", 
                    desc: "Dynamically adapts to Bull, Bear, and Sideways market conditions",
                    icon: Activity
                  },
                  { 
                    title: "Quantum Portfolio Optimization", 
                    desc: "Advanced risk-adjusted position sizing with quantum annealing techniques",
                    icon: Target
                  }
                ].map((item, index) => (
                  <div key={index} className="tech-feature group">
                    <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-xl flex items-center justify-center mr-4">
                      <item.icon className="w-6 h-6 text-black" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-white mb-2 group-hover:text-emerald-400 transition-colors">
                        {item.title}
                      </h3>
                      <p className="text-gray-400 group-hover:text-gray-300 transition-colors">
                        {item.desc}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative">
              <div className="performance-dashboard">
                <div className="dashboard-header">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold text-white">Live Performance Metrics</h3>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                      <span className="text-sm text-emerald-400">LIVE</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  {[
                    { label: "Signal Accuracy", value: "78.4%", progress: 78.4, color: "emerald" },
                    { label: "Risk-Adjusted Returns", value: "2.4x", progress: 85, color: "cyan" },
                    { label: "Sharpe Ratio", value: "1.86", progress: 92, color: "purple" },
                    { label: "Max Drawdown", value: "8.2%", progress: 88, color: "pink" }
                  ].map((metric, index) => (
                    <div key={index} className="metric-row">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-400">{metric.label}</span>
                        <span className={`font-bold text-${metric.color}-400`}>{metric.value}</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className={`progress-fill bg-gradient-to-r from-${metric.color}-400 to-${metric.color}-600`}
                          style={{ width: `${metric.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-8 p-4 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-xl border border-emerald-500/20">
                  <p className="text-xs text-gray-400 text-center">
                    * Based on 2+ years of backtesting across 100+ NSE stocks with ₹50Cr+ AUM validation
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="py-32 relative">
        {/* Circuit Board Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="circuit-pattern"></div>
          <div className="data-stream"></div>
        </div>
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-5xl font-black mb-8">
              <span className="neon-pulse-text">TRUSTED BY</span><br />
              <span className="text-white">INDUSTRY LEADERS</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="testimonial-card group">
                <div className="flex gap-1 mb-6">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
                  ))}
                </div>
                
                                 <p className="text-gray-300 mb-8 leading-relaxed text-lg">
                   &ldquo;{testimonial.content}&rdquo;
                 </p>
                
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-full flex items-center justify-center font-bold text-black">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <div className="font-bold text-white text-lg">{testimonial.name}</div>
                    <div className="text-gray-400">{testimonial.role}</div>
                    <div className="text-emerald-400 text-sm font-medium">{testimonial.company}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 relative">
        {/* Energy Field Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="energy-field"></div>
          <div className="pulse-rings"></div>
        </div>
        <div className="max-w-5xl mx-auto px-6 lg:px-8 text-center">
          <div className="cta-glow-box">
            <h2 className="text-5xl md:text-7xl font-black mb-8">
              <span className="text-white">READY TO</span><br />
              <span className="neon-pulse-text glitch-effect" data-text="DOMINATE">DOMINATE</span><br />
              <span className="text-white">THE MARKETS?</span>
            </h2>
            
            <p className="text-xl text-gray-400 mb-12 max-w-3xl mx-auto leading-relaxed">
              Join the elite circle of institutional traders who leverage our AI-powered signals 
              to consistently outperform the market.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <Link href="/auth/sign-up" className="cyber-btn-hero-large group">
                <span>START FREE TRIAL</span>
                <ArrowRight className="w-6 h-6 ml-3 transition-transform group-hover:translate-x-2" />
              </Link>
              
              <Link href="#demo" className="cyber-btn-secondary-large group">
                <Eye className="w-6 h-6 mr-3" />
                <span>SCHEDULE DEMO</span>
              </Link>
            </div>

            <div className="flex items-center justify-center gap-8 mt-12 text-sm text-gray-400">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span>7-day free trial</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span>No credit card required</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span>Cancel anytime</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-16 bg-black/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="col-span-2 md:col-span-1">
              <Link href="/" className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-black font-bold" />
                </div>
                <span className="text-2xl font-bold neon-text">TradingSignals</span>
              </Link>
              <p className="text-gray-400 leading-relaxed mb-6">
                Next-generation AI-powered trading signals for institutional investors and professional traders.
              </p>
              <div className="flex items-center gap-4">
                <Link href="#" className="social-link">
                  <Globe className="w-5 h-5" />
                </Link>
              </div>
            </div>
            
            {[
              {
                title: "Platform",
                links: ["Features", "Technology", "API Access", "Documentation"]
              },
              {
                title: "Company", 
                links: ["About Us", "Careers", "Contact", "Blog"]
              },
              {
                title: "Legal",
                links: ["Privacy Policy", "Terms of Service", "Risk Disclaimer", "Compliance"]
              }
            ].map((section, index) => (
              <div key={index}>
                <h3 className="font-bold text-white mb-4 text-lg">{section.title}</h3>
                <ul className="space-y-3">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <Link href="#" className="text-gray-400 hover:text-emerald-400 transition-colors">
                        {link}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400">
              © 2024 TradingSignals. All rights reserved.
            </p>
            <div className="flex items-center gap-2 mt-4 md:mt-0 text-sm text-gray-400">
              <Lock className="w-4 h-4" />
              <span>Enterprise-grade security</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
