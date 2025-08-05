#!/usr/bin/env python3
"""
Runner script for Technical Document AI Streamlit app
"""
import subprocess
import sys
import os
import warnings
from pathlib import Path

def setup_environment():
    """Setup environment to suppress warnings"""
    # Suppress common warnings
    warnings.filterwarnings("ignore", message="PyPDF2 is deprecated")
    warnings.filterwarnings("ignore", message=".*applymap.*deprecated.*")
    os.environ['JUPYTER_PLATFORM_DIRS'] = '1'
    
    # Set Python warnings environment
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::FutureWarning'

def main():
    """Run the Streamlit application"""
    setup_environment()
    
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--theme.base", "light",
        "--logger.level", "error"  # Reduce Streamlit logging
    ]
    
    print("🚀 Starting Technical Document AI...")
    print(f"📍 URL: http://localhost:8501")
    print("💡 Tip: Upload a PDF and ask questions like:")
    print("   'How much ventilation is required for a 6-car carpark?'")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()