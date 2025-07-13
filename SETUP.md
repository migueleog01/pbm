# 🔧 Cross-Platform Setup Guide

Detailed setup instructions for **Voice-to-Mermaid with LLaMA Integration** across different platforms.

## 📋 **System Requirements**

### **Minimum Requirements**
- **RAM**: 8GB+ (6GB for LLaMA model + 2GB for system)
- **Storage**: 8GB free space
- **CPU**: Multi-core ARM64 or x64 processor
- **Python**: 3.8+
- **Git**: With submodule support

### **Recommended Requirements**
- **RAM**: 16GB+ (for smooth operation)
- **Storage**: 16GB+ free space
- **CPU**: ARM64 (M1/M2/M3, Snapdragon X Elite) for optimal performance
- **Microphone**: USB headset or high-quality built-in mic

## 🖥️ **Platform-Specific Setup**

---

## 🍎 **macOS (Apple Silicon M1/M2/M3)**

### **Prerequisites**
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake portaudio git python3
```

### **Setup Steps**
```bash
# 1. Clone repository
git clone --recurse-submodules https://github.com/yourusername/voice-to-mermaid.git
cd voice-to-mermaid

# 2. Install Python dependencies
pip3 install -r requirements.txt

# 3. Build Whisper.cpp
cd whisper.cpp
make -j8  # Use 8 threads for M1/M2
cd ..

# 4. Download Whisper model
cd whisper.cpp
./models/download-ggml-model.sh base.en
cd ..

# 5. Install LLaMA dependencies
pip3 install llama-cpp-python

# 6. Download LLaMA model
cd voice-to-mermaid-llm/models
python3 download_model.py
cd ../..

# 7. Test the system
python3 enhanced_realtime_mermaid.py
```

### **macOS Optimization**
```bash
# For M1/M2 Pro/Max with more cores
export WHISPER_THREADS=10
export LLAMA_THREADS=10

# For maximum performance
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

---

## 🪟 **Windows ARM64 (Snapdragon X Elite)**

### **Prerequisites**
```powershell
# Install Visual Studio Build Tools 2022
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Select: C++ build tools, Windows SDK, CMake

# Install Python 3.8+ from Microsoft Store or python.org
# Install Git from git-scm.com

# Install Windows Package Manager (winget) if not available
# Or use Chocolatey: https://chocolatey.org/
```

### **Setup Steps**
```powershell
# 1. Open PowerShell as Administrator
# 2. Clone repository
git clone --recurse-submodules https://github.com/yourusername/voice-to-mermaid.git
cd voice-to-mermaid

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Build Whisper.cpp
cd whisper.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..\..

# 5. Download Whisper model
cd whisper.cpp
bash models/download-ggml-model.sh base.en
cd ..

# 6. Install LLaMA dependencies for Windows ARM64
pip install llama-cpp-python

# 7. Download LLaMA model
cd voice-to-mermaid-llm/models
python download_model.py
cd ..\..

# 8. Test the system
python enhanced_realtime_mermaid.py
```

### **Windows ARM64 Optimization**
```powershell
# Set environment variables for Snapdragon X Elite
$env:WHISPER_THREADS="12"  # Adjust based on core count
$env:LLAMA_THREADS="12"
$env:OMP_NUM_THREADS="12"

# For WSL2 users (recommended for development)
wsl --install
# Then follow Linux setup inside WSL2
```

### **Alternative: WSL2 Setup**
```bash
# Inside WSL2 Ubuntu
sudo apt update
sudo apt install build-essential cmake git python3 python3-pip portaudio19-dev

# Follow Linux setup steps below
```

---

## 🐧 **Linux (Ubuntu/Debian)**

### **Prerequisites**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y build-essential cmake git python3 python3-pip
sudo apt install -y portaudio19-dev libffi-dev libssl-dev

# For ARM64 systems
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### **Setup Steps**
```bash
# 1. Clone repository
git clone --recurse-submodules https://github.com/yourusername/voice-to-mermaid.git
cd voice-to-mermaid

# 2. Install Python dependencies
pip3 install -r requirements.txt

# 3. Build Whisper.cpp
cd whisper.cpp
make -j$(nproc)
cd ..

# 4. Download Whisper model
cd whisper.cpp
./models/download-ggml-model.sh base.en
cd ..

# 5. Install LLaMA dependencies
pip3 install llama-cpp-python

# 6. Download LLaMA model
cd voice-to-mermaid-llm/models
python3 download_model.py
cd ../..

# 7. Test the system
python3 enhanced_realtime_mermaid.py
```

---

## 🔧 **Model Download Scripts**

### **Automatic LLaMA Model Download**
Create `voice-to-mermaid-llm/models/download_model.py`:
```python
#!/usr/bin/env python3
"""
Automatic LLaMA model download script
"""
import os
import sys
from pathlib import Path

def download_llama_model():
    """Download LLaMA v3.1 8B Instruct model"""
    
    # Check if already downloaded
    model_path = Path("llama-v3.1-8b-instruct.Q4_K_M.gguf")
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return
    
    try:
        # Install huggingface-hub if not present
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
        
        # Download model
        print("🔄 Downloading LLaMA v3.1 8B Instruct model (4.6GB)...")
        subprocess.run([
            "huggingface-cli", "download",
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "--local-dir", ".",
            "--local-dir-use-symlinks", "False"
        ], check=True)
        
        # Rename to expected filename
        if Path("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf").exists():
            Path("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf").rename(model_path)
            print(f"✅ Model downloaded and renamed to: {model_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("💡 Manual download:")
        print("   1. Visit: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        print("   2. Download: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        print("   3. Rename to: llama-v3.1-8b-instruct.Q4_K_M.gguf")
        print("   4. Place in: voice-to-mermaid-llm/models/")

if __name__ == "__main__":
    download_llama_model()
```

### **Alternative Models**
```bash
# For faster inference (smaller model)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf --local-dir . --local-dir-use-symlinks False

# For better quality (larger model)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

---

## 🧪 **Testing & Validation**

### **Test Audio Setup**
```python
# Create test_audio.py
import sounddevice as sd
import numpy as np

def test_audio():
    print("🎤 Testing audio input...")
    
    # List available devices
    print("\n📋 Available audio devices:")
    print(sd.query_devices())
    
    # Test recording
    print("\n🔴 Recording 3 seconds of audio...")
    duration = 3
    sample_rate = 16000
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    # Check audio levels
    rms = np.sqrt(np.mean(audio**2))
    print(f"📊 Audio RMS: {rms:.4f}")
    
    if rms > 0.005:
        print("✅ Audio input working correctly!")
    else:
        print("❌ Audio input too quiet or not detected")
        print("💡 Try adjusting microphone levels or selecting a different device")

if __name__ == "__main__":
    test_audio()
```

### **Test Whisper.cpp**
```bash
# Test whisper.cpp directly
cd whisper.cpp
echo "Testing Whisper.cpp..." > test.txt
./build/bin/whisper-cli -m models/ggml-base.en-q5_1.bin -f test.wav -t 8 --language en --no-timestamps

# Expected output: transcribed text
```

### **Test LLaMA Integration**
```bash
# Test LLaMA model
cd voice-to-mermaid-llm
python llama_mermaid.py -t "user logs in and accesses dashboard"

# Expected output: Mermaid diagram
```

---

## 🔧 **Configuration Options**

### **Performance Tuning**
Edit `enhanced_realtime_mermaid.py`:
```python
# Whisper Configuration
WHISPER_THREADS = 8        # Adjust based on CPU cores
BEAM_SIZE = 3              # 1=fastest, 5=most accurate
BEST_OF = 1               # 1=fastest, 3=most accurate

# LLaMA Configuration
LLAMA_THREADS = 8          # Adjust based on CPU cores
LLAMA_TEMPERATURE = 0.3    # 0.1=more consistent, 0.5=more creative
LLAMA_CONTEXT = 2048       # Context window size

# Audio Configuration
SAMPLE_RATE = 16000        # 16kHz for Whisper
CHUNK_DURATION = 3.0       # Process every 3 seconds
SILENCE_THRESHOLD = 0.005  # Voice activity detection
```

### **Platform-Specific Optimizations**

#### **Apple Silicon (M1/M2/M3)**
```python
# Use Metal acceleration
LLAMA_METAL = True
LLAMA_N_GPU_LAYERS = -1  # Use all GPU layers

# Optimize for unified memory
LLAMA_USE_MLOCK = True
LLAMA_F16_KV = True
```

#### **Windows ARM64 (Snapdragon X Elite)**
```python
# CPU-optimized settings
LLAMA_METAL = False
LLAMA_N_GPU_LAYERS = 0

# Adjust for Snapdragon X Elite
WHISPER_THREADS = 12      # Adjust based on core count
LLAMA_THREADS = 12
```

#### **Intel/AMD x64**
```python
# Standard CPU settings
LLAMA_THREADS = 8         # Adjust based on CPU cores
LLAMA_USE_MLOCK = False   # May not be needed on x64
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Model Not Found**
```bash
# Error: Model file not found
# Solution: Download models first
cd voice-to-mermaid-llm/models
python download_model.py

cd ../../whisper.cpp
./models/download-ggml-model.sh base.en
```

#### **2. Whisper Compilation Fails**
```bash
# macOS: Install Xcode command line tools
xcode-select --install

# Linux: Install build dependencies
sudo apt install build-essential cmake

# Windows: Install Visual Studio Build Tools
# Download from Microsoft website
```

#### **3. Audio Device Issues**
```python
# Test audio devices
import sounddevice as sd
print(sd.query_devices())

# Set default device
sd.default.device = 'device_name'
```

#### **4. High CPU Usage**
```python
# Reduce thread count
WHISPER_THREADS = 4
LLAMA_THREADS = 4

# Use smaller model
# Download Q4_K_S instead of Q4_K_M
```

#### **5. Memory Issues**
```python
# Use smaller context window
LLAMA_CONTEXT = 1024

# Or use smaller model variant
# Q4_K_S (smaller) vs Q4_K_M (current) vs Q5_K_M (larger)
```

### **Platform-Specific Issues**

#### **macOS**
```bash
# Permission issues
# System Preferences > Security & Privacy > Privacy > Microphone
# Add Terminal.app or your Python executable

# Metal acceleration warnings
# These are normal and don't affect functionality
```

#### **Windows ARM64**
```powershell
# Long path support
git config --system core.longpaths true

# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Visual Studio Build Tools
# Ensure C++ build tools and Windows SDK are installed
```

#### **Linux**
```bash
# Audio permissions
sudo usermod -a -G audio $USER
# Logout and login again

# Missing dependencies
sudo apt install portaudio19-dev libffi-dev libssl-dev
```

---

## 🚀 **Deployment Tips**

### **For Development**
```bash
# Use virtual environment
python -m venv voice-mermaid-env
source voice-mermaid-env/bin/activate  # Linux/Mac
# voice-mermaid-env\Scripts\activate   # Windows

pip install -r requirements.txt
```

### **For Production**
```bash
# Optimize for production
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Use systemd service (Linux)
sudo cp voice-mermaid.service /etc/systemd/system/
sudo systemctl enable voice-mermaid
sudo systemctl start voice-mermaid
```

### **For Hackathon Demo**
```bash
# Pre-download all models
bash setup_all_models.sh

# Test all features
python test_full_pipeline.py

# Prepare backup commands
echo "Computer, start to process to check to end" > demo_commands.txt
```

---

## 📊 **Performance Benchmarks**

### **Expected Performance**
| Platform | Whisper Speed | LLaMA Speed | Total Latency |
|----------|---------------|-------------|---------------|
| M1 Pro | 0.3s | 1.2s | 1.5s |
| M2 Max | 0.25s | 1.0s | 1.25s |
| Snapdragon X Elite | 0.4s | 2.0s | 2.4s |
| Intel i7-12700K | 0.5s | 2.5s | 3.0s |

### **Memory Usage**
- **Whisper Model**: 57MB
- **LLaMA Model**: 4.6GB
- **Python Runtime**: 500MB
- **Audio Buffers**: 50MB
- **Total**: ~6GB RAM

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Fork the repository
git clone https://github.com/yourusername/voice-to-mermaid.git
cd voice-to-mermaid

# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
python enhanced_realtime_mermaid.py

# Submit pull request
git push origin feature/new-feature
```

### **Testing**
```bash
# Run test suite
python -m pytest tests/

# Test on multiple platforms
# macOS, Windows ARM64, Linux x64
```

---

## 📞 **Support**

### **Getting Help**
1. Check this SETUP.md guide
2. Review troubleshooting section
3. Check GitHub issues
4. Create new issue with:
   - Platform information
   - Error messages
   - Steps to reproduce

### **Hackathon Support**
- **Demo Issues**: Check demo_commands.txt
- **Platform Issues**: Platform-specific sections above
- **Model Issues**: Re-download models
- **Performance Issues**: Adjust thread counts

---

**🎯 Ready to build amazing voice-to-diagram experiences!** 