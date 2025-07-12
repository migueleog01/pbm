#!/usr/bin/env python3
"""
Whisper Model Manager for M1 Pro
Download and manage quantized models for optimal performance
"""

import subprocess
import argparse
from pathlib import Path
import json
import time

# Available quantized models and their characteristics
QUANTIZED_MODELS = {
    'tiny.en-q5_1': {
        'size_mb': 32,
        'accuracy': 'Good',
        'speed': 'Fastest',
        'description': 'Tiny model with Q5_1 quantization - best for speed'
    },
    'base.en-q5_1': {
        'size_mb': 58,
        'accuracy': 'Good',
        'speed': 'Very Fast',
        'description': 'Base model with Q5_1 quantization - good balance'
    },
    'small.en-q5_1': {
        'size_mb': 188,
        'accuracy': 'Better',
        'speed': 'Fast',
        'description': 'Small model with Q5_1 quantization - still real-time'
    },
    'medium.en-q5_0': {
        'size_mb': 515,
        'accuracy': 'Very Good',
        'speed': 'Medium',
        'description': 'Medium model with Q5_0 quantization - high accuracy'
    },
    'large-v3-q5_0': {
        'size_mb': 1030,
        'accuracy': 'Excellent',
        'speed': 'Slower',
        'description': 'Large model with Q5_0 quantization - best accuracy'
    }
}

def download_model(model_name: str, whisper_dir: str = 'whisper.cpp') -> bool:
    """Download a specific quantized model."""
    whisper_path = Path(whisper_dir)
    if not whisper_path.exists():
        print(f"❌ Whisper directory not found: {whisper_path}")
        return False
    
    script_path = whisper_path / 'models' / 'download-ggml-model.sh'
    if not script_path.exists():
        print(f"❌ Download script not found: {script_path}")
        return False
    
    print(f"📥 Downloading {model_name}...")
    
    try:
        # Run download script
        result = subprocess.run([
            'bash', str(script_path), model_name
        ], cwd=str(whisper_path), capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded {model_name}")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def check_available_models(whisper_dir: str = 'whisper.cpp') -> dict:
    """Check which models are available locally."""
    models_dir = Path(whisper_dir) / 'models'
    available = {}
    
    for model_name in QUANTIZED_MODELS:
        model_file = models_dir / f'ggml-{model_name}.bin'
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            available[model_name] = {
                'path': str(model_file),
                'size_mb': size_mb,
                'info': QUANTIZED_MODELS[model_name]
            }
    
    return available

def benchmark_quantized_model(model_name: str, whisper_dir: str = 'whisper.cpp') -> dict:
    """Quick benchmark of a quantized model."""
    from benchmark_whisper import create_test_audio, benchmark_config
    
    model_path = Path(whisper_dir) / 'models' / f'ggml-{model_name}.bin'
    whisper_cli = Path(whisper_dir) / 'build' / 'bin' / 'whisper-cli'
    
    if not model_path.exists():
        return {'error': 'Model not found'}
    
    if not whisper_cli.exists():
        return {'error': 'Whisper CLI not found'}
    
    print(f"⚡ Benchmarking {model_name}...")
    
    # Create test audio
    test_audio = create_test_audio(duration=3.0)
    
    # Benchmark with optimal settings
    optimal_settings = {
        'threads': 8,
        'beam_size': 1,
        'best_of': 1
    }
    
    result = benchmark_config(
        str(model_path), test_audio, **optimal_settings
    )
    
    # Clean up
    Path(test_audio).unlink(missing_ok=True)
    
    return result

def recommend_model_for_use_case(use_case: str) -> str:
    """Recommend best model for specific use case."""
    recommendations = {
        'realtime': 'base.en-q5_1',
        'continuous': 'tiny.en-q5_1', 
        'accuracy': 'medium.en-q5_0',
        'maximum': 'large-v3-q5_0',
        'balanced': 'small.en-q5_1'
    }
    
    return recommendations.get(use_case, 'base.en-q5_1')

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Whisper Model Manager for M1 Pro")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--all', action='store_true', help='Show all models (not just downloaded)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a model')
    download_parser.add_argument('model', choices=QUANTIZED_MODELS.keys(), help='Model to download')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark a model')
    benchmark_parser.add_argument('model', help='Model to benchmark')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get model recommendations')
    recommend_parser.add_argument('use_case', 
                                 choices=['realtime', 'continuous', 'accuracy', 'maximum', 'balanced'],
                                 help='Use case for recommendation')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        print("📋 WHISPER MODELS STATUS")
        print("=" * 60)
        
        available = check_available_models()
        
        if args.all:
            print("\n🔍 All Available Models:")
            for model_name, info in QUANTIZED_MODELS.items():
                status = "✅ Downloaded" if model_name in available else "❌ Not downloaded"
                print(f"  {model_name:20} | {info['size_mb']:4d}MB | {info['speed']:10} | {status}")
                print(f"     {info['description']}")
        else:
            print("\n✅ Downloaded Models:")
            if available:
                for model_name, info in available.items():
                    print(f"  {model_name:20} | {info['size_mb']:6.1f}MB | {info['info']['description']}")
            else:
                print("  No models downloaded yet")
    
    elif args.command == 'download':
        model_name = args.model
        if model_name in QUANTIZED_MODELS:
            info = QUANTIZED_MODELS[model_name]
            print(f"📥 Downloading {model_name}")
            print(f"   Size: {info['size_mb']}MB")
            print(f"   Description: {info['description']}")
            download_model(model_name)
        else:
            print(f"❌ Unknown model: {model_name}")
    
    elif args.command == 'benchmark':
        model_name = args.model
        result = benchmark_quantized_model(model_name)
        
        if 'error' in result:
            print(f"❌ Benchmark failed: {result['error']}")
        else:
            print(f"📊 Benchmark Results for {model_name}:")
            print(f"   Inference time: {result['inference_time']:.2f}s")
            print(f"   Real-time factor: {result['inference_time']/3.0:.2f}x")
            print(f"   Status: {'✅ Real-time' if result['inference_time'] < 3.0 else '⚠️ Slower than real-time'}")
    
    elif args.command == 'recommend':
        use_case = args.use_case
        recommended = recommend_model_for_use_case(use_case)
        info = QUANTIZED_MODELS[recommended]
        
        print(f"🎯 RECOMMENDATION FOR {use_case.upper()} USE:")
        print(f"   Model: {recommended}")
        print(f"   Size: {info['size_mb']}MB")
        print(f"   Description: {info['description']}")
        print(f"   Expected speed: {info['speed']}")
        print(f"   Expected accuracy: {info['accuracy']}")
        
        # Check if it's downloaded
        available = check_available_models()
        if recommended not in available:
            print(f"\n📥 To download this model, run:")
            print(f"   python scripts/model_manager.py download {recommended}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 