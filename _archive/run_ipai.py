#!/usr/bin/env python3
"""
IPAI System Launcher

Simple launcher script to start the IPAI FastAPI server
with proper Python path configuration.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ.setdefault("PYTHONPATH", str(src_path))

# Import and run the application
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting IPAI System...")
    print(f"📁 Source path: {src_path}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print(f"📊 API documentation at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop\n")
    
    # Configuration
    config = {
        "app": "api.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n⏹️  IPAI System stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start IPAI System: {e}")
        sys.exit(1)