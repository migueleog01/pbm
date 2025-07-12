#!/usr/bin/env python3
"""
Base.en Performance Maximizer for M1 Pro
Specialized optimization for base.en model
"""

import subprocess
import tempfile
import time
import wave
import numpy as np
import sounddevice as sd
from pathlib import Path
import argparse
import threading
import queue
import signal
import sys

# Base.en specific optimal configurations for M1 Pro
BASE_CONFIGS = {
    'base.en': {
        'model_path': 'whisper.cpp/models/ggml-base.en.bin',
        'optimal_threads': 6,  # Sweet spot for M1 Pro
        'beam_size': 1,        # Fast processing
        'best_of': 3,          # Good quality balance
        'expected_rtf': 0.25,  # Expected real-time factor
        'description': 'Full precision base model'
    },
    'base.en-q5_1': {
        'model_path': 'whisper.cpp/models/ggml-base.en-q5_1.bin',
        'optimal_threads': 6,
        'beam_size': 1,
        'best_of': 1,          # Quantized can use lower best_of
        'expected_rtf': 0.15,  # Should be faster
        'description': 'Quantized base model (Q5_1)'
    }
}

# Advanced M1 Pro specific settings to test
THREADING_TESTS = [4, 6, 8, 10]  # Test different thread counts
BEAM_TESTS = [1, 2, 3]           # Test beam sizes
BEST_OF_TESTS = [1, 2, 3, 5]     # Test best_of values

class BaseOptimizer:
    """Specialized optimizer for base.en models on M1 Pro."""
    
    def __init__(self, whisper_cli: str = 'whisper.cpp/build/bin/whisper-cli'):
        self.whisper_cli = Path(whisper_cli)
        self.test_audio = None
        
        if not self.whisper_cli.exists():
            raise FileNotFoundError(f"Whisper CLI not found: {self.whisper_cli}")
    
    def create_realistic_test_audio(self, duration: float = 5.0) -> str:
        """Create more realistic test audio for benchmarking."""
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Create speech-like audio with multiple formants
        t = np.linspace(0, duration, samples)
        
        # Fundamental frequency varies like speech
        f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)  # 120-150 Hz variation
        
        # Create formants (speech-like resonances)
        formant1 = np.sin(2 * np.pi * 800 * t)   # First formant ~800Hz
        formant2 = np.sin(2 * np.pi * 1200 * t)  # Second formant ~1200Hz
        formant3 = np.sin(2 * np.pi * 2500 * t)  # Third formant ~2500Hz
        
        # Combine with varying amplitudes
        speech_like = (
            0.4 * np.sin(2 * np.pi * f0 * t) +
            0.3 * formant1 +
            0.2 * formant2 +
            0.1 * formant3
        )
        
        # Add speech envelope (pauses and emphasis)
        envelope = np.ones_like(t)
        # Add some pauses
        envelope[int(samples*0.2):int(samples*0.3)] *= 0.1  # Pause
        envelope[int(samples*0.6):int(samples*0.7)] *= 0.1  # Another pause
        # Add emphasis
        envelope[int(samples*0.4):int(samples*0.5)] *= 1.5
        
        # Apply envelope and normalize
        audio = speech_like * envelope * 0.3
        audio = audio / np.max(np.abs(audio)) * 0.8  # Normalize to 80% max
        
        # Save as WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                audio_int16 = (audio * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            self.test_audio = f.name
            return f.name
    
    def benchmark_config(self, model_name: str, threads: int, beam_size: int, 
                        best_of: int, use_vad: bool = True) -> dict:
        """Benchmark a specific configuration."""
        config = BASE_CONFIGS.get(model_name)
        if not config or not Path(config['model_path']).exists():
            return {'error': f'Model {model_name} not available'}
        
        if not self.test_audio:
            self.create_realistic_test_audio()
        
        cmd = [
            str(self.whisper_cli),
            '-m', config['model_path'],
            '-f', self.test_audio,
            '-t', str(threads),
            '--beam-size', str(beam_size),
            '--best-of', str(best_of),
            '--language', 'en',
            '--fp16',
            '--no-timestamps',
            '--output-txt'
        ]
        
        if use_vad:
            cmd.extend(['--vad-filter', '--vad-thold', '0.6'])
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            end_time = time.time()
            
            if result.returncode == 0:
                inference_time = end_time - start_time
                return {
                    'success': True,
                    'inference_time': inference_time,
                    'rtf': inference_time / 5.0,  # 5-second audio
                    'output': result.stdout.strip()
                }
            else:
                return {'error': result.stderr}
        except subprocess.TimeoutExpired:
            return {'error': 'Timeout'}
    
    def find_optimal_settings(self, model_name: str) -> dict:
        """Find optimal settings for a specific base model."""
        print(f"🔍 Finding optimal settings for {model_name}...")
        
        best_config = None
        best_time = float('inf')
        results = []
        
        total_tests = len(THREADING_TESTS) * len(BEAM_TESTS) * len(BEST_OF_TESTS)
        current_test = 0
        
        for threads in THREADING_TESTS:
            for beam_size in BEAM_TESTS:
                for best_of in BEST_OF_TESTS:
                    current_test += 1
                    print(f"  📊 Test {current_test}/{total_tests}: t={threads}, beam={beam_size}, best_of={best_of}")
                    
                    result = self.benchmark_config(model_name, threads, beam_size, best_of)
                    
                    if result.get('success'):
                        results.append({
                            'threads': threads,
                            'beam_size': beam_size,
                            'best_of': best_of,
                            'inference_time': result['inference_time'],
                            'rtf': result['rtf']
                        })
                        
                        if result['inference_time'] < best_time:
                            best_time = result['inference_time']
                            best_config = {
                                'threads': threads,
                                'beam_size': beam_size,
                                'best_of': best_of,
                                'inference_time': result['inference_time'],
                                'rtf': result['rtf']
                            }
                        
                        print(f"    ✅ {result['inference_time']:.3f}s (RTF: {result['rtf']:.3f}x)")
                    else:
                        print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        
        return {'best_config': best_config, 'all_results': results}
    
    def compare_models(self) -> None:
        """Compare both base models with optimal settings."""
        print("🏁 BASE.EN MODEL COMPARISON")
        print("=" * 60)
        
        for model_name, config in BASE_CONFIGS.items():
            if not Path(config['model_path']).exists():
                print(f"⚠️  {model_name} not available")
                continue
            
            print(f"\n🎯 Testing {model_name}")
            optimization_result = self.find_optimal_settings(model_name)
            
            if optimization_result['best_config']:
                best = optimization_result['best_config']
                print(f"\n✨ OPTIMAL SETTINGS for {model_name}:")
                print(f"   Threads: {best['threads']}")
                print(f"   Beam size: {best['beam_size']}")
                print(f"   Best of: {best['best_of']}")
                print(f"   Inference time: {best['inference_time']:.3f}s")
                print(f"   Real-time factor: {best['rtf']:.3f}x")
                print(f"   Real-time capable: {'✅ Yes' if best['rtf'] < 1.0 else '❌ No'}")
    
    def run_optimized_realtime(self, model_name: str = 'base.en-q5_1') -> None:
        """Run real-time transcription with optimized base.en settings."""
        config = BASE_CONFIGS.get(model_name)
        if not config or not Path(config['model_path']).exists():
            print(f"❌ Model {model_name} not available")
            return
        
        print(f"🎤 Starting optimized real-time transcription with {model_name}")
        print(f"📝 {config['description']}")
        print("Press Ctrl+C to stop")
        
        # Use optimal settings
        optimal_settings = {
            'threads': config['optimal_threads'],
            'beam_size': config['beam_size'],
            'best_of': config['best_of']
        }
        
        print(f"⚙️  Using: threads={optimal_settings['threads']}, beam={optimal_settings['beam_size']}, best_of={optimal_settings['best_of']}")
        
        # Audio processing setup
        sample_rate = 16000
        chunk_duration = 3.0
        audio_queue = queue.Queue()
        running = True
        
        def signal_handler(signum, frame):
            nonlocal running
            print("\n🛑 Stopping...")
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        def audio_callback(indata, frames, time_info, status):
            if running and not audio_queue.full():
                audio_queue.put(indata.copy())
        
        def process_audio():
            buffer = []
            buffer_duration = 0.0
            
            while running:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                    buffer.append(chunk)
                    buffer_duration += len(chunk) / sample_rate
                    
                    if buffer_duration >= chunk_duration:
                        # Process accumulated audio
                        audio_data = np.concatenate(buffer).flatten()
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            with wave.open(tmp_file.name, 'wb') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(sample_rate)
                                
                                # Normalize and convert
                                if np.max(np.abs(audio_data)) > 0:
                                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
                                
                                audio_int16 = (audio_data * 32767).astype(np.int16)
                                wav_file.writeframes(audio_int16.tobytes())
                        
                        # Transcribe
                        start_time = time.time()
                        result = self.benchmark_config(
                            model_name, 
                            optimal_settings['threads'],
                            optimal_settings['beam_size'],
                            optimal_settings['best_of']
                        )
                        
                        if result.get('success') and result.get('output'):
                            elapsed = time.time() - start_time
                            print(f"⚡ {elapsed:.2f}s | {result['output']}")
                        
                        # Clean up
                        Path(tmp_file.name).unlink(missing_ok=True)
                        
                        # Reset buffer
                        buffer = []
                        buffer_duration = 0.0
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Processing error: {e}")
        
        # Start processing thread
        process_thread = threading.Thread(target=process_audio)
        process_thread.start()
        
        # Start audio stream
        print("🎧 Listening...")
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1),
            dtype=np.float32
        ):
            try:
                while running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        
        running = False
        process_thread.join()
        print("✅ Stopped")
    
    def cleanup(self):
        """Clean up test files."""
        if self.test_audio and Path(self.test_audio).exists():
            Path(self.test_audio).unlink()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Base.en Performance Maximizer")
    parser.add_argument('--compare', action='store_true', help='Compare both base models')
    parser.add_argument('--optimize', choices=['base.en', 'base.en-q5_1'], help='Optimize specific model')
    parser.add_argument('--realtime', choices=['base.en', 'base.en-q5_1'], default='base.en-q5_1',
                       help='Run optimized real-time transcription')
    
    args = parser.parse_args()
    
    optimizer = BaseOptimizer()
    
    try:
        if args.compare:
            optimizer.compare_models()
        elif args.optimize:
            result = optimizer.find_optimal_settings(args.optimize)
            if result['best_config']:
                best = result['best_config']
                print(f"\n🏆 OPTIMAL SETTINGS for {args.optimize}:")
                print(f"   --threads {best['threads']} --beam-size {best['beam_size']} --best-of {best['best_of']}")
                print(f"   Performance: {best['inference_time']:.3f}s ({best['rtf']:.3f}x RTF)")
        else:
            optimizer.run_optimized_realtime(args.realtime)
    
    finally:
        optimizer.cleanup()

if __name__ == "__main__":
    main() 