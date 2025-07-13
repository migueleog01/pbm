#!/usr/bin/env python3
"""
Voice-to-Mermaid Setup Script
Helps users set up the complete system quickly
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_status(message, emoji="🔧"):
    """Print status message with emoji"""
    print(f"{emoji} {message}")

def run_command(command, description=""):
    """Run a command and return success status"""
    print_status(f"Running: {description or command}", "⚙️")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_status(f"Error: {e.stderr}", "❌")
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...", "🐍")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor} is compatible ✅", "✅")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} is not compatible. Need Python 3.8+", "❌")
        return False

def check_system_info():
    """Display system information"""
    print_status("System Information:", "💻")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")

def install_dependencies():
    """Install Python dependencies"""
    print_status("Installing Python dependencies...", "📦")
    
    # Upgrade pip first
    success, output = run_command("pip install --upgrade pip", "Upgrading pip")
    if not success:
        return False
    
    # Install requirements
    requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
    if requirements_path.exists():
        success, output = run_command(f"pip install -r {requirements_path}", "Installing requirements")
        return success
    else:
        print_status("requirements.txt not found. Installing core dependencies...", "⚠️")
        core_deps = [
            "flask==3.0.0",
            "flask-cors==4.0.0",
            "llama-cpp-python==0.2.90",
            "huggingface-hub==0.19.0"
        ]
        
        for dep in core_deps:
            success, output = run_command(f"pip install {dep}", f"Installing {dep}")
            if not success:
                return False
        return True

def setup_model_directory():
    """Set up the model directory"""
    print_status("Setting up model directory...", "📁")
    
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "voice-to-mermaid-llm" / "models"
    
    # Create directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "llama-v3.1-8b-instruct.Q4_K_M.gguf"
    
    if model_file.exists():
        print_status("Model file already exists ✅", "✅")
        return True
    
    print_status("Model file not found. Please download it manually:", "⚠️")
    print("\nDownload options:")
    print("1. Using download script (recommended):")
    print(f"   cd {model_dir}")
    print("   python download_model.py")
    print("\n2. Manual download:")
    print("   - Go to: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
    print("   - Download: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    print("   - Rename to: llama-v3.1-8b-instruct.Q4_K_M.gguf")
    print(f"   - Place in: {model_dir}")
    print("\n3. Using huggingface-hub:")
    print("   pip install huggingface-hub")
    print("   python -c \"from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', local_dir='./voice-to-mermaid-llm/models'); os.rename('./voice-to-mermaid-llm/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', './voice-to-mermaid-llm/models/llama-v3.1-8b-instruct.Q4_K_M.gguf')\"")
    
    return False

def test_server():
    """Test if the server can start"""
    print_status("Testing server startup...", "🚀")
    
    try:
        import flask
        from flask_cors import CORS
        print_status("Flask dependencies available ✅", "✅")
        
        # Try to import LLaMA components
        sys.path.append(str(Path(__file__).parent.parent.parent))
        try:
            from llama_mermaid import LlamaMermaidConverter
            print_status("LLaMA components available ✅", "✅")
        except ImportError as e:
            print_status(f"LLaMA components not available: {e}", "⚠️")
            print_status("Server will run with fallback mode", "ℹ️")
        
        return True
    except ImportError as e:
        print_status(f"Missing dependencies: {e}", "❌")
        return False

def show_next_steps():
    """Show next steps to user"""
    print_status("Setup completed! Next steps:", "🎉")
    print("\n1. Start the server:")
    print("   cd pbm/extension")
    print("   python server.py")
    print("\n2. Install Chrome extension:")
    print("   - Open Chrome and go to chrome://extensions/")
    print("   - Enable 'Developer mode'")
    print("   - Click 'Load unpacked' and select 'pbm/extension' folder")
    print("\n3. Grant microphone permissions:")
    print("   - Click the extension icon")
    print("   - Click 'Test Server Connection'")
    print("   - Allow microphone access when prompted")
    print("\n4. Use the extension:")
    print("   - Click 'Start Listening'")
    print("   - Say 'hey computer' followed by your diagram description")
    print("   - Example: 'hey computer draw a user login flow'")

def main():
    """Main setup function"""
    print_status("Voice-to-Mermaid Setup Script", "🎤")
    print("=" * 50)
    
    # Check system info
    check_system_info()
    print()
    
    # Check Python version
    if not check_python_version():
        print_status("Please install Python 3.8+ and try again", "❌")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print_status("Failed to install dependencies", "❌")
        sys.exit(1)
    
    # Set up model directory
    model_ready = setup_model_directory()
    
    # Test server
    if test_server():
        print_status("Server test passed ✅", "✅")
    else:
        print_status("Server test failed ❌", "❌")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    if model_ready:
        print_status("Setup completed successfully! 🎉", "🎉")
    else:
        print_status("Setup mostly completed. Please download the model file manually.", "⚠️")
    
    show_next_steps()

if __name__ == "__main__":
    main() 