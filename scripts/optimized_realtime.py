#!/usr/bin/env python3
"""
Optimized Real-Time Whisper Transcription for M1 Pro
Uses best practices for maximum clarity and performance
"""

import argparse
import subprocess
import tempfile
import time
import wave
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import queue
import signal
import sys

# Optimized configurations for M1 Pro
OPTIMAL_CONFIGS = {
    'tiny.en': {
        'threads': 4,
        'beam_size': 1,
        'best_of': 1,
        'expected_rtf': 0.1,  # Real-time factor
        'description': 'Fastest, good for continuous transcription'
    },
    'base.en': {
        'threads': 6,
        'beam_size': 1,
        'best_of': 3,
        'expected_rtf': 0.3,
        'description': 'Good balance of speed and accuracy'
    },
    'small.en': {
        'threads': 8,
        'beam_size': 3,
        'best_of': 3,
        'expected_rtf': 0.6,
        'description': 'Better accuracy, still real-time capable'
    },
    'medium.en': {
        'threads': 8,
        'beam_size': 3,
        'best_of': 5,
        'expected_rtf': 1.2,
        'description': 'High accuracy, slower but usable'
    },
    'large-v3': {
        'threads': 8,
        'beam_size': 5,
        'best_of': 5,
        'expected_rtf': 2.5,
        'description': 'Maximum accuracy, not real-time'
    }
}

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3.0  # 3-second chunks for good accuracy/latency balance
OVERLAP_DURATION = 0.5  # 0.5s overlap to avoid word cutoffs
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
MIN_SPEECH_DURATION = 0.8  # Minimum speech duration to process

class AudioPreprocessor:
    """Audio preprocessing for optimal Whisper performance."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.prev_chunk = None
        
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing optimizations."""
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply gentle high-pass filter to remove DC bias and low rumble
        # Simple difference-based high-pass
        if len(audio_data) > 1:
            filtered = np.concatenate([[audio_data[0]], np.diff(audio_data) * 0.95 + audio_data[1:] * 0.05])
            audio_data = filtered
        
        # Apply gentle noise gate
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < SILENCE_THRESHOLD:
            audio_data = audio_data * 0.1  # Reduce but don't eliminate
        
        return audio_data.astype(np.float32)
    
    def create_overlapped_chunk(self, current_chunk: np.ndarray) -> np.ndarray:
        """Create overlapped audio chunk to avoid word cutoffs."""
        if self.prev_chunk is None:
            self.prev_chunk = current_chunk
            return current_chunk
        
        # Calculate overlap samples
        overlap_samples = int(OVERLAP_DURATION * self.sample_rate)
        
        # Take end of previous chunk and beginning of current chunk
        if len(self.prev_chunk) >= overlap_samples:
            prev_end = self.prev_chunk[-overlap_samples:]
            overlapped = np.concatenate([prev_end, current_chunk])
        else:
            overlapped = current_chunk
        
        self.prev_chunk = current_chunk
        return overlapped

class OptimizedWhisperTranscriber:
    """Optimized Whisper transcriber for M1 Pro."""
    
    def __init__(self, model_name: str = 'base.en', whisper_path: str = 'whisper.cpp/build/bin/whisper-cli'):
        self.model_name = model_name
        self.whisper_path = Path(whisper_path)
        self.model_path = Path(f'whisper.cpp/models/ggml-{model_name}.bin')
        self.config = OPTIMAL_CONFIGS.get(model_name, OPTIMAL_CONFIGS['base.en'])
        self.preprocessor = AudioPreprocessor()
        
        # Validate paths
        if not self.whisper_path.exists():
            raise FileNotFoundError(f"Whisper CLI not found: {self.whisper_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"🚀 Initialized {model_name}: {self.config['description']}")
        print(f"   Expected real-time factor: {self.config['expected_rtf']:.1f}x")
        
    def transcribe_audio(self, audio_data: np.ndarray, use_vad: bool = True) -> Optional[str]:
        """Transcribe audio chunk with optimal settings."""
        
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess_audio(audio_data)
        
        # Apply VAD (Voice Activity Detection)
        if use_vad:
            rms = np.sqrt(np.mean(processed_audio ** 2))
            if rms < SILENCE_THRESHOLD:
                return None  # Skip silent audio
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Write audio to WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                
                # Convert to int16
                audio_int16 = (processed_audio * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
        
        try:
            # Build optimized command
            cmd = [
                str(self.whisper_path),
                '-m', str(self.model_path),
                '-f', tmp_path,
                '-t', str(self.config['threads']),
                '--beam-size', str(self.config['beam_size']),
                '--best-of', str(self.config['best_of']),
                '--language', 'en',
                '--fp16',  # Use fp16 for M1 Pro
                '--no-timestamps',  # Faster processing
                '--output-txt'
            ]
            
            # Add VAD filter if requested
            if use_vad:
                cmd.extend(['--vad-filter', '--vad-thold', '0.6'])
            
            # Run whisper
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            inference_time = time.time() - start_time
            
            if result.returncode == 0:
                # Clean up output
                text = result.stdout.strip()
                if text and len(text) > 1:
                    print(f"⚡ {inference_time:.2f}s | {text}")
                    return text
                return None
            else:
                print(f"❌ Whisper error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("⏱️ Whisper timeout")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

class RealtimeTranscriptionManager:
    """Manages real-time transcription with optimized buffering."""
    
    def __init__(self, model_name: str = 'base.en'):
        self.transcriber = OptimizedWhisperTranscriber(model_name)
        self.audio_queue = queue.Queue()
        self.running = False
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Shutting down...")
        self.running = False
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue
        if self.running:
            self.audio_queue.put(indata.copy())
    
    def process_audio_worker(self):
        """Worker thread for processing audio."""
        while self.running:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=1.0)
                
                # Add to buffer
                self.audio_buffer.append(chunk)
                self.buffer_duration += len(chunk) / SAMPLE_RATE
                
                # Process when buffer is full
                if self.buffer_duration >= CHUNK_DURATION:
                    # Concatenate buffer
                    audio_data = np.concatenate(self.audio_buffer)
                    
                    # Create overlapped chunk
                    overlapped_audio = self.transcriber.preprocessor.create_overlapped_chunk(audio_data)
                    
                    # Transcribe
                    self.transcriber.transcribe_audio(overlapped_audio)
                    
                    # Reset buffer (keep some overlap)
                    overlap_samples = int(OVERLAP_DURATION * SAMPLE_RATE)
                    if len(audio_data) > overlap_samples:
                        remaining_audio = audio_data[-overlap_samples:]
                        self.audio_buffer = [remaining_audio]
                        self.buffer_duration = len(remaining_audio) / SAMPLE_RATE
                    else:
                        self.audio_buffer = []
                        self.buffer_duration = 0.0
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
    
    def start_transcription(self):
        """Start real-time transcription."""
        print("🎤 Starting real-time transcription...")
        print("Press Ctrl+C to stop")
        
        self.running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio_worker)
        process_thread.start()
        
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
            dtype=np.float32
        ):
            print("🎧 Listening... speak now!")
            
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        
        # Clean shutdown
        self.running = False
        process_thread.join()
        print("✅ Transcription stopped")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Optimized Real-Time Whisper Transcription")
    parser.add_argument('--model', choices=OPTIMAL_CONFIGS.keys(), default='base.en',
                       help='Whisper model to use')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and their characteristics')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("📋 Available Models:")
        for model_name, config in OPTIMAL_CONFIGS.items():
            print(f"  {model_name:12} | RTF: {config['expected_rtf']:4.1f}x | {config['description']}")
        return
    
    # Start transcription
    manager = RealtimeTranscriptionManager(args.model)
    manager.start_transcription()

if __name__ == "__main__":
    main() 