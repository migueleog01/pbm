#!/usr/bin/env python3
"""
Automatic LLaMA model download script
Downloads LLaMA v3.1 8B Instruct model for voice-to-mermaid pipeline
"""
import os
import sys
import subprocess
from pathlib import Path

def download_llama_model():
    """Download LLaMA v3.1 8B Instruct model"""
    
    # Check if already downloaded
    model_path = Path("llama-v3.1-8b-instruct.Q4_K_M.gguf")
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        file_size = model_path.stat().st_size / (1024**3)  # GB
        print(f"📦 Size: {file_size:.1f}GB")
        return
    
    print("🧠 LLaMA v3.1 8B Instruct Model Download")
    print("📋 Model: Q4_K_M quantized (4.6GB)")
    print("🎯 Optimized for: Apple Silicon (M1/M2/M3), Snapdragon X Elite, x64")
    print("⏱️  Expected download time: 2-5 minutes (depends on internet speed)")
    print()
    
    try:
        # Install huggingface-hub if not present
        print("🔧 Installing huggingface-hub...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "huggingface-hub", "--quiet"
        ], check=True)
        
        # Download model
        print("🔄 Downloading LLaMA v3.1 8B Instruct model...")
        print("📡 Source: Hugging Face (bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)")
        print("📦 File: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        print()
        
        cmd = [
            "huggingface-cli", "download",
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "--local-dir", ".",
            "--local-dir-use-symlinks", "False"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Rename to expected filename
        original_path = Path("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        if original_path.exists():
            original_path.rename(model_path)
            print(f"✅ Model downloaded and renamed to: {model_path}")
            
            # Show file size
            file_size = model_path.stat().st_size / (1024**3)  # GB
            print(f"📦 Size: {file_size:.1f}GB")
            print()
            print("🎉 Model ready for use!")
            print("▶️  Next: Run 'python ../../enhanced_realtime_mermaid.py'")
        else:
            print("❌ Download completed but file not found")
            print("🔍 Checking directory contents...")
            for f in Path(".").glob("*.gguf"):
                print(f"   Found: {f}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        print()
        print("🔧 Manual download instructions:")
        print("   1. Visit: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        print("   2. Download: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        print("   3. Rename to: llama-v3.1-8b-instruct.Q4_K_M.gguf")
        print("   4. Place in: voice-to-mermaid-llm/models/")
        print()
        print("🌐 Alternative download methods:")
        print("   • Use wget/curl with direct download link")
        print("   • Use browser download and manual placement")
        print("   • Use git-lfs if available")
        
    except FileNotFoundError:
        print("❌ huggingface-cli not found after installation")
        print("🔧 Alternative installation methods:")
        print("   pip install --upgrade huggingface-hub")
        print("   conda install -c conda-forge huggingface_hub")
        print("   pip install --user huggingface-hub")

def show_model_info():
    """Show information about available model variants"""
    print("📊 Available LLaMA v3.1 8B Instruct Models:")
    print()
    
    models = [
        {
            "name": "Q4_K_M",
            "size": "4.6GB",
            "quality": "Good",
            "speed": "Fast",
            "recommended": True,
            "description": "Balanced performance, good for most use cases"
        },
        {
            "name": "Q4_K_S",
            "size": "4.4GB",
            "quality": "Fair",
            "speed": "Very Fast",
            "recommended": False,
            "description": "Smaller, faster but slightly lower quality"
        },
        {
            "name": "Q5_K_M",
            "size": "5.8GB",
            "quality": "Better",
            "speed": "Medium",
            "recommended": False,
            "description": "Higher quality but slower and more memory"
        },
        {
            "name": "Q8_0",
            "size": "8.5GB",
            "quality": "Best",
            "speed": "Slow",
            "recommended": False,
            "description": "Highest quality but requires 12GB+ RAM"
        }
    ]
    
    for model in models:
        status = "✅ RECOMMENDED" if model["recommended"] else "  Alternative"
        print(f"{status} {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   Quality: {model['quality']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Description: {model['description']}")
        print()

def download_alternative_model(model_name):
    """Download alternative model variant"""
    filename_map = {
        "Q4_K_S": "Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
        "Q5_K_M": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "Q8_0": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    }
    
    if model_name not in filename_map:
        print(f"❌ Unknown model: {model_name}")
        print("Available models: Q4_K_S, Q5_K_M, Q8_0")
        return
    
    filename = filename_map[model_name]
    local_filename = f"llama-v3.1-8b-instruct.{model_name}.gguf"
    
    print(f"🔄 Downloading {model_name} variant...")
    
    try:
        cmd = [
            "huggingface-cli", "download",
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename,
            "--local-dir", ".",
            "--local-dir-use-symlinks", "False"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Rename to local filename
        if Path(filename).exists():
            Path(filename).rename(local_filename)
            print(f"✅ {model_name} model downloaded: {local_filename}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info":
            show_model_info()
        elif sys.argv[1] == "--model":
            if len(sys.argv) > 2:
                download_alternative_model(sys.argv[2])
            else:
                print("Usage: python download_model.py --model Q4_K_S")
        else:
            print("Usage:")
            print("  python download_model.py           # Download default Q4_K_M model")
            print("  python download_model.py --info    # Show model information")
            print("  python download_model.py --model Q4_K_S  # Download alternative model")
    else:
        download_llama_model() 