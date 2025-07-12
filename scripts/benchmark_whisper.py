#!/usr/bin/env python3
"""
Whisper Performance Benchmarking Script for M1 Pro
Tests different models and settings to find optimal configuration
"""

import subprocess
import tempfile
import time
import wave
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse

# Test configurations
MODELS = {
    'tiny.en': 'whisper.cpp/models/ggml-tiny.en.bin',
    'base.en': 'whisper.cpp/models/ggml-base.en.bin', 
    'small.en': 'whisper.cpp/models/ggml-small.en.bin',
    'medium.en': 'whisper.cpp/models/ggml-medium.en.bin',
    'large-v3': 'whisper.cpp/models/ggml-large-v3.bin'
}

# M1 Pro optimal settings to test
THREAD_CONFIGS = [4, 6, 8, 10]  # M1 Pro has 8 performance cores + 2 efficiency cores
BEAM_SIZES = [1, 3, 5]  # Higher = better quality, slower
BEST_OF_VALUES = [1, 3, 5]  # Higher = better quality, slower

def create_test_audio(duration: float = 10.0, sample_rate: int = 16000) -> str:
    """Create a test audio file with speech-like characteristics."""
    # Generate pink noise (more speech-like than white noise)
    samples = int(duration * sample_rate)
    
    # Create pink noise by filtering white noise
    white_noise = np.random.randn(samples)
    
    # Simple pink noise filter (1/f spectrum)
    freqs = np.fft.fftfreq(samples, 1/sample_rate)
    pink_filter = np.where(freqs != 0, 1/np.sqrt(np.abs(freqs)), 1)
    pink_filter[0] = 1  # DC component
    
    noise_fft = np.fft.fft(white_noise)
    pink_noise = np.real(np.fft.ifft(noise_fft * pink_filter))
    
    # Add speech-like modulation
    t = np.linspace(0, duration, samples)
    speech_envelope = 0.3 * (1 + np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz modulation
    test_audio = (pink_noise * speech_envelope * 0.1).astype(np.float32)
    
    # Save as WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        with wave.open(f.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            audio_int16 = (test_audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        return f.name

def benchmark_config(model_path: str, audio_file: str, threads: int = 4, 
                    beam_size: int = 1, best_of: int = 1, use_fp16: bool = True,
                    language: str = "en") -> Dict:
    """Benchmark a specific Whisper configuration."""
    
    cmd = [
        'whisper.cpp/build/bin/whisper-cli',
        '-m', model_path,
        '-f', audio_file,
        '-t', str(threads),
        '--beam-size', str(beam_size),
        '--best-of', str(best_of),
        '--language', language,
        '--output-txt'
    ]
    
    if use_fp16:
        cmd.append('--fp16')
    
    # Benchmark inference time
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        end_time = time.time()
        
        if result.returncode == 0:
            inference_time = end_time - start_time
            # Extract text output
            output_text = result.stdout.strip()
            
            return {
                'success': True,
                'inference_time': inference_time,
                'output_text': output_text,
                'error': None
            }
        else:
            return {
                'success': False,
                'inference_time': None,
                'output_text': None,
                'error': result.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'inference_time': None,
            'output_text': None,
            'error': 'Timeout'
        }

def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across all configurations."""
    print("🚀 Starting Whisper M1 Pro Optimization Benchmark")
    print("=" * 60)
    
    # Create test audio
    print("📝 Creating test audio file...")
    test_audio = create_test_audio(duration=5.0)  # 5 seconds for speed
    
    results = []
    
    for model_name, model_path in MODELS.items():
        if not Path(model_path).exists():
            print(f"⚠️  Skipping {model_name} - model not found")
            continue
            
        print(f"\n🔍 Testing {model_name}...")
        
        for threads in THREAD_CONFIGS:
            for beam_size in BEAM_SIZES:
                for best_of in BEST_OF_VALUES:
                    config = {
                        'model': model_name,
                        'threads': threads,
                        'beam_size': beam_size,
                        'best_of': best_of,
                        'fp16': True
                    }
                    
                    print(f"  ⚡ threads={threads}, beam={beam_size}, best_of={best_of}")
                    
                    benchmark_result = benchmark_config(
                        model_path, test_audio, threads, beam_size, best_of
                    )
                    
                    result = {**config, **benchmark_result}
                    results.append(result)
                    
                    if benchmark_result['success']:
                        print(f"     ✅ {benchmark_result['inference_time']:.2f}s")
                    else:
                        print(f"     ❌ {benchmark_result['error']}")
    
    # Clean up
    Path(test_audio).unlink(missing_ok=True)
    
    # Analyze results
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    successful_results.sort(key=lambda x: x['inference_time'])
    
    print("\n🏆 TOP 10 FASTEST CONFIGURATIONS:")
    for i, result in enumerate(successful_results[:10]):
        print(f"{i+1:2d}. {result['model']:12} | "
              f"threads={result['threads']:2d} | "
              f"beam={result['beam_size']} | "
              f"best_of={result['best_of']} | "
              f"{result['inference_time']:.2f}s")
    
    print(f"\n📈 RECOMMENDATIONS FOR M1 PRO:")
    
    # Find best config for each model
    model_bests = {}
    for result in successful_results:
        model = result['model']
        if model not in model_bests:
            model_bests[model] = result
    
    print("\n🎯 Optimal settings per model:")
    for model, best_config in model_bests.items():
        print(f"  {model:12}: {best_config['threads']}t, beam={best_config['beam_size']}, "
              f"best_of={best_config['best_of']} → {best_config['inference_time']:.2f}s")
    
    # Real-time suitability analysis
    print(f"\n⚡ REAL-TIME SUITABILITY (target: <1.0s for 5s audio):")
    realtime_suitable = [r for r in successful_results if r['inference_time'] < 1.0]
    
    if realtime_suitable:
        print("  ✅ Real-time capable configurations:")
        for result in realtime_suitable[:5]:
            print(f"    {result['model']:12} | "
                  f"threads={result['threads']:2d} | "
                  f"beam={result['beam_size']} | "
                  f"best_of={result['best_of']} | "
                  f"{result['inference_time']:.2f}s")
    else:
        print("  ⚠️  No configurations achieved real-time performance")
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to benchmark_results.json")
    
    return successful_results

def quick_model_comparison():
    """Quick comparison of just the models with optimal settings."""
    print("🔥 QUICK MODEL COMPARISON - Optimal Settings Only")
    print("=" * 60)
    
    test_audio = create_test_audio(duration=3.0)
    
    # Optimal settings based on typical M1 Pro performance
    optimal_configs = {
        'tiny.en': {'threads': 4, 'beam_size': 1, 'best_of': 1},
        'base.en': {'threads': 6, 'beam_size': 1, 'best_of': 1},
        'small.en': {'threads': 8, 'beam_size': 1, 'best_of': 3},
        'medium.en': {'threads': 8, 'beam_size': 3, 'best_of': 3},
        'large-v3': {'threads': 8, 'beam_size': 5, 'best_of': 5}
    }
    
    for model_name, config in optimal_configs.items():
        model_path = MODELS.get(model_name)
        if not model_path or not Path(model_path).exists():
            continue
            
        print(f"\n🎯 {model_name} (optimal settings)")
        result = benchmark_config(model_path, test_audio, **config)
        
        if result['success']:
            print(f"  ⚡ Inference time: {result['inference_time']:.2f}s")
            print(f"  🎬 Real-time factor: {result['inference_time']/3.0:.2f}x")
            print(f"  ✅ Real-time suitable: {'Yes' if result['inference_time'] < 3.0 else 'No'}")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    Path(test_audio).unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Whisper on M1 Pro")
    parser.add_argument('--quick', action='store_true', help='Run quick comparison only')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_model_comparison()
    elif args.full:
        run_comprehensive_benchmark()
    else:
        print("Choose --quick for fast comparison or --full for comprehensive benchmark") 