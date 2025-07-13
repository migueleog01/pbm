#!/usr/bin/env python3
"""
Voice-to-Mermaid Chrome Extension Setup Script
Helps set up and start the voice-to-mermaid system
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print the setup banner"""
    print("=" * 60)
    print("🎤 Voice-to-Mermaid Chrome Extension Setup")
    print("📊 Generate Mermaid diagrams from voice commands")
    print("=" * 60)
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    
    # Check if requirements are satisfied
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies installed")
    except ImportError:
        print("❌ Flask dependencies missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed")
    
    # Check if voice-to-mermaid system exists
    voice_script = Path("../enhanced_realtime_mermaid.py")
    if not voice_script.exists():
        print("❌ Voice-to-mermaid system not found at:", voice_script.absolute())
        return False
    
    print("✅ Voice-to-mermaid system found")
    return True

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting Voice-to-Mermaid server...")
    print("📡 Server will run on: http://localhost:5000")
    print("🎤 Ready to receive voice commands from Chrome extension!")
    print("👋 Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")

def print_extension_instructions():
    """Print Chrome extension setup instructions"""
    print("\n📝 Chrome Extension Setup Instructions:")
    print("-" * 40)
    print("1. Open Chrome and go to: chrome://extensions/")
    print("2. Enable 'Developer Mode' (toggle in top right)")
    print("3. Click 'Load unpacked'")
    print("4. Select this folder:", Path().absolute())
    print("5. Pin the extension to your toolbar")
    print("6. Click the extension icon to start listening")
    print("7. Say 'hey computer' to trigger voice-to-mermaid")
    print()

def print_usage_guide():
    """Print usage guide"""
    print("🎯 How to Use:")
    print("-" * 20)
    print("1. Click the 'Voice to Mermaid' extension icon")
    print("2. Click 'Start Listening' button")
    print("3. Say 'hey computer' to activate")
    print("4. Speak your diagram description")
    print("5. Watch as Mermaid diagrams are generated!")
    print()
    print("📊 Example voice commands:")
    print("• 'User logs in, validates credentials, redirects to dashboard'")
    print("• 'Payment flows from user to gateway to bank'")
    print("• 'API receives request, processes data, returns response'")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    if not check_dependencies():
        print("❌ Setup failed. Please resolve dependencies first.")
        return 1
    
    print_extension_instructions()
    print_usage_guide()
    
    response = input("🚀 Ready to start the server? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        start_server()
    else:
        print("👋 Setup complete! Run 'python server.py' when ready.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 