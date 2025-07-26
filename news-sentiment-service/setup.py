#!/usr/bin/env python3
"""
Setup script for NSE News Sentiment Pipeline
Handles installation and initial configuration
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Check if Python version is supported"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['data', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'feedparser',
        'torch',
        'transformers', 
        'apscheduler',
        'pandas',
        'numpy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("🚀 Running quick test...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test news fetcher
        from src.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        print("✅ News fetcher initialized")
        
        # Test sentiment analyzer (this will download the model)
        print("📥 Loading FinBERT model (this may take a while on first run)...")
        from src.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        print("✅ Sentiment analyzer initialized")
        
        # Test data persistence
        from src.data_persistence import DataPersistence
        persistence = DataPersistence()
        print("✅ Data persistence initialized")
        
        # Quick sentiment test
        test_result = analyzer.analyze_text("Test headline for sentiment analysis")
        print(f"✅ Test sentiment: {test_result['sentiment']} (confidence: {test_result['confidence']:.3f})")
        
        print("🎉 Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Test the pipeline:")
    print("   python demo.py")
    print("   python main.py --mode test")
    print()
    print("2. Run a single iteration:")
    print("   python main.py --mode single")
    print()
    print("3. Start continuous monitoring:")
    print("   python main.py --mode scheduled")
    print()
    print("4. Check pipeline status:")
    print("   python main.py --mode status")
    print()
    print("5. Run comprehensive tests:")
    print("   python test_pipeline.py")
    print()
    print("📖 For detailed documentation, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 NSE News Sentiment Pipeline Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        print("Try installing manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed during import testing")
        sys.exit(1)
    
    # Run quick test
    print("\n" + "="*50)
    print("🧪 RUNNING QUICK TEST")
    print("="*50)
    
    if not run_quick_test():
        print("❌ Setup failed during quick test")
        print("You can still try running manually:")
        print("  python main.py --mode test")
        sys.exit(1)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 