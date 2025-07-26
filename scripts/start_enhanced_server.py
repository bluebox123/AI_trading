#!/usr/bin/env python3
"""
Start Enhanced API Server with Live Signals Support
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def start_server():
    """Start the enhanced API server"""
    print("🚀 Starting Enhanced Trading Signals API Server...")
    print("="*60)
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    # Check if required files exist
    api_server_path = PROJECT_ROOT / 'src' / 'api_server.py'
    if not api_server_path.exists():
        print(f"❌ Error: API server not found at {api_server_path}")
        return
    
    print("✅ API server found")
    print("✅ Starting server on http://localhost:8000")
    print("✅ Live signals available at http://localhost:8000/api/signals/live")
    print("✅ API documentation at http://localhost:8000/docs")
    print("="*60)
    
    try:
        # Start the server
        subprocess.run([
            sys.executable,
            str(api_server_path)
        ], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server() 