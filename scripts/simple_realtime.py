#!/usr/bin/env python3
"""
Simple, Reliable Real-Time Transcription with Optimal Base.en Settings
"""

import subprocess
import tempfile
import time
import wave
import numpy as np
import sounddevice as sd
from pathlib import Path
import signal
import sys

# Optimal settings for base.en-q5_1 on M1 Pro
MODEL_PATH = 'whisper.cpp/models/ggml-base.en-q5_1.bin'
WHISPER_CLI = 'whisper.cpp/build/bin/whisper-cli'
OPTIMAL_THREADS = 8
OPTIMAL_BEAM = 3
OPTIMAL_BEST_OF = 1

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0  # Process every 3 seconds
SILENCE_THRESHOLD = 0.005  # Minimum audio level to process

class SimpleRealTimeTranscriber:
    def __init__(self):
        self.running = True
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Check if model exists
        if not Path(MODEL_PATH).exists():
            print(f"❌ Model not found: {MODEL_PATH}")
            print("Run: cd whisper.cpp && bash models/download-ggml-model.sh base.en-q5_1")
            sys.exit(1)
            
        if not Path(WHISPER_CLI).exists():
            print(f"❌ Whisper CLI not found: {WHISPER_CLI}")
            sys.exit(1)
            
        print("✅ Model and CLI found!")
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Stopping...")
        self.running = False
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio chunk using optimal settings."""
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Check if audio has content
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < SILENCE_THRESHOLD:
            return None
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                
                # Convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
        
        try:
            # Run whisper with optimal settings
            cmd = [
                WHISPER_CLI,
                '-m', MODEL_PATH,
                '-f', tmp_file.name,
                '-t', str(OPTIMAL_THREADS),
                '--beam-size', str(OPTIMAL_BEAM),
                '--best-of', str(OPTIMAL_BEST_OF),
                '--language', 'en',
                '--no-timestamps'
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse whisper output - it typically contains metadata followed by text
                full_output = result.stdout.strip()
                
                # Extract the actual transcription (usually the last non-empty line)
                lines = [line.strip() for line in full_output.split('\n') if line.strip()]
                
                # Find the transcription (skip metadata lines)
                text = None
                for line in reversed(lines):
                    # Skip common metadata patterns
                    if not any(pattern in line.lower() for pattern in ['[', 'whisper', 'processing', 'model']):
                        text = line
                        break
                
                if text and len(text) > 2:
                    inference_time = end_time - start_time
                    print(f"⚡ {inference_time:.2f}s | {text}")
                    return text
            else:
                print(f"❌ Whisper error: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            print("⏱️ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Clean up temp file
            Path(tmp_file.name).unlink(missing_ok=True)
        
        return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio."""
        if not self.running:
            return
            
        if status:
            print(f"Audio status: {status}")
        
        # Add audio to buffer
        audio_chunk = indata.flatten()
        self.audio_buffer.append(audio_chunk)
        self.buffer_duration += len(audio_chunk) / SAMPLE_RATE
        
        # Process when buffer is full
        if self.buffer_duration >= CHUNK_DURATION:
            # Concatenate all audio
            full_audio = np.concatenate(self.audio_buffer)
            
            # Show audio level
            rms = np.sqrt(np.mean(full_audio ** 2))
            if rms > SILENCE_THRESHOLD:
                print(f"🔊 Processing audio (level: {rms:.4f})...")
                self.transcribe_audio(full_audio)
            else:
                print("🔇 Silent audio, skipping...")
            
            # Reset buffer
            self.audio_buffer = []
            self.buffer_duration = 0.0
    
    def start_listening(self):
        """Start real-time transcription."""
        print("🎤 Starting real-time transcription with optimal base.en-q5_1 settings")
        print(f"⚙️  Threads: {OPTIMAL_THREADS}, Beam: {OPTIMAL_BEAM}, Best-of: {OPTIMAL_BEST_OF}")
        print("🎧 Listening... speak clearly!")
        print("Press Ctrl+C to stop")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
            dtype=np.float32
        ):
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        
        print("✅ Transcription stopped")

def main():
    """Main function."""
    transcriber = SimpleRealTimeTranscriber()
    transcriber.start_listening()

if __name__ == "__main__":
    main() 