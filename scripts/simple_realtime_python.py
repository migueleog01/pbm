#!/usr/bin/env python3
"""
Simple, Reliable Real-Time Transcription with Optimal Base.en Settings
Using Python Whisper Library (Windows Compatible)
"""

import whisper
import tempfile
import time
import wave
import numpy as np
import sounddevice as sd
from pathlib import Path
import signal
import sys

# Optimal settings for base.en on Windows (matching SETTINGS_SUMMARY.md)
MODEL_NAME = 'base.en'  # 57MB quantized equivalent
OPTIMAL_THREADS = 8
OPTIMAL_BEAM = 3
OPTIMAL_BEST_OF = 1

# Audio settings (matching SETTINGS_SUMMARY.md)
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0  # Process every 3 seconds
SILENCE_THRESHOLD = 0.005  # Minimum audio level to process

class SimpleRealTimeTranscriber:
    def __init__(self):
        self.running = True
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        print("🔄 Loading Whisper model...")
        try:
            # Load the optimal model
            self.model = whisper.load_model(MODEL_NAME)
            print(f"✅ Model '{MODEL_NAME}' loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
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
        
        try:
            start_time = time.time()
            
            # Use optimal settings matching SETTINGS_SUMMARY.md
            # Pass audio data directly to whisper instead of using temp files
            result = self.model.transcribe(
                audio_data,
                language='en',
                beam_size=OPTIMAL_BEAM,
                best_of=OPTIMAL_BEST_OF,
                temperature=0.0,  # Deterministic output
                no_speech_threshold=0.1,
                condition_on_previous_text=False
            )
            
            end_time = time.time()
            
            text = result['text'].strip()
            if text and len(text) > 2:
                inference_time = end_time - start_time
                print(f"⚡ {inference_time:.2f}s | {text}")
                return text
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
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
        print("🎤 Starting real-time transcription with optimal base.en settings")
        print(f"⚙️  Model: {MODEL_NAME}, Beam: {OPTIMAL_BEAM}, Best-of: {OPTIMAL_BEST_OF}")
        print(f"🎯 Target: <0.5s processing time (your optimal config)")
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