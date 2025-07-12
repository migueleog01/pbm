#!/usr/bin/env python3
"""
Enhanced Real-Time Voice-to-Mermaid Pipeline with LLaMA Integration

Combines Whisper.cpp for speech recognition with LLaMA v3.1 8B Instruct for intelligent
Mermaid diagram generation - all running locally on Apple Silicon.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import wave
import signal

import numpy as np
import sounddevice as sd

# Import LLaMA converter
sys.path.append(str(Path(__file__).parent / "voice-to-mermaid-llm"))
try:
    from llama_mermaid import LlamaMermaidConverter
except ImportError:
    print("❌ LLaMA integration not available. Please set up the voice-to-mermaid-llm directory.")
    LlamaMermaidConverter = None

# Audio configuration (from original pipeline)
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
SILENCE_THRESHOLD = 0.005

# Enhanced pipeline configuration
WHISPER_CLI = 'whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL = 'whisper.cpp/models/ggml-base.en-q5_1.bin'
LLAMA_MODEL = 'voice-to-mermaid-llm/models/llama-v3.1-8b-instruct.Q4_K_M.gguf'

# Global state
running = True
llama_converter = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n🛑 Stopping enhanced pipeline...")
    running = False

def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))

def has_speech(audio_chunk: np.ndarray) -> bool:
    """Simple voice activity detection."""
    rms = calculate_rms(audio_chunk)
    return rms > SILENCE_THRESHOLD

def transcribe_with_whisper(audio_data: np.ndarray) -> Optional[str]:
    """Transcribe audio using Whisper.cpp (from original pipeline)."""
    # Normalize audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    
    # Check if audio has content
    rms = calculate_rms(audio_data)
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
            '-m', WHISPER_MODEL,
            '-f', tmp_file.name,
            '-t', '8',
            '--beam-size', '3',
            '--best-of', '1',
            '--language', 'en',
            '--no-timestamps'
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        end_time = time.time()
        
        if result.returncode == 0:
            # Parse whisper output
            full_output = result.stdout.strip()
            lines = [line.strip() for line in full_output.split('\n') if line.strip()]
            
            # Find the transcription
            text = None
            for line in reversed(lines):
                if not any(pattern in line.lower() for pattern in ['[', 'whisper', 'processing', 'model']):
                    text = line
                    break
            
            if text and len(text) > 2:
                inference_time = end_time - start_time
                return text, inference_time
        else:
            print(f"❌ Whisper error: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        print("⏱️ Whisper timeout")
    except Exception as e:
        print(f"❌ Whisper error: {e}")
    finally:
        # Clean up temp file
        Path(tmp_file.name).unlink(missing_ok=True)
    
    return None

def should_generate_diagram(text: str) -> bool:
    """Quick check if text might be a diagram description."""
    text_lower = text.lower()
    
    # Diagram trigger words
    diagram_keywords = [
        'diagram', 'flowchart', 'flow', 'chart', 'graph', 'sequence',
        'workflow', 'process', 'mindmap', 'connect', 'flow',
        'architecture', 'design', 'structure', 'relationship'
    ]
    
    # Action words
    action_keywords = [
        'draw', 'create', 'make', 'show', 'generate', 'build',
        'design', 'illustrate', 'visualize'
    ]
    
    # Connection words
    connection_keywords = [
        'to', 'from', 'then', 'after', 'before', 'connects',
        'leads', 'flows', 'goes', 'calls', 'sends', 'receives'
    ]
    
    has_diagram = any(keyword in text_lower for keyword in diagram_keywords)
    has_action = any(keyword in text_lower for keyword in action_keywords)
    has_connection = any(keyword in text_lower for keyword in connection_keywords)
    
    # Be more permissive - if it has connections or actions, try LLaMA
    return has_diagram or has_action or has_connection

class EnhancedRealTimeTranscriber:
    """Enhanced transcriber with LLaMA integration."""
    
    def __init__(self, use_llama: bool = True):
        self.use_llama = use_llama
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Check Whisper setup
        if not Path(WHISPER_CLI).exists():
            print(f"❌ Whisper CLI not found: {WHISPER_CLI}")
            sys.exit(1)
        
        if not Path(WHISPER_MODEL).exists():
            print(f"❌ Whisper model not found: {WHISPER_MODEL}")
            sys.exit(1)
        
        # Initialize LLaMA if available
        if self.use_llama and LlamaMermaidConverter:
            try:
                if Path(LLAMA_MODEL).exists():
                    print("🧠 Initializing LLaMA for diagram generation...")
                    global llama_converter
                    llama_converter = LlamaMermaidConverter(LLAMA_MODEL)
                    print("✅ LLaMA ready for diagram generation!")
                else:
                    print(f"⚠️  LLaMA model not found: {LLAMA_MODEL}")
                    print("📝 Falling back to regex-based diagram detection")
                    self.use_llama = False
            except Exception as e:
                print(f"❌ Failed to initialize LLaMA: {e}")
                print("📝 Falling back to regex-based diagram detection")
                self.use_llama = False
        
        # Set up signal handling
        signal.signal(signal.SIGINT, signal_handler)
        
        print("✅ Enhanced voice-to-Mermaid pipeline ready!")
    
    def process_transcript(self, text: str) -> Optional[str]:
        """Process transcript and generate Mermaid diagram if appropriate."""
        if not text or len(text.strip()) < 3:
            return None
        
        # Quick check if this might be a diagram description
        if not should_generate_diagram(text):
            return None
        
        if self.use_llama and llama_converter:
            # Use LLaMA for intelligent diagram generation
            print("🧠 Using LLaMA for diagram generation...")
            return llama_converter.generate_mermaid(text)
        else:
            # Fallback to simple regex-based detection (simplified version)
            print("📝 Using regex-based diagram detection...")
            return self.simple_diagram_detection(text)
    
    def simple_diagram_detection(self, text: str) -> Optional[str]:
        """Simple fallback diagram generation."""
        text_lower = text.lower()
        
        # Very basic diagram generation
        if 'sequence' in text_lower:
            return """sequenceDiagram
    A->>B: interaction
    B-->>A: response"""
        elif any(word in text_lower for word in ['flow', 'process', 'to', 'then']):
            return """graph TD
    A[Start] --> B[Process]
    B --> C[End]"""
        else:
            return """graph TD
    A[Input] --> B[Output]"""
    
    def audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio."""
        global running
        if not running:
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
            rms = calculate_rms(full_audio)
            if rms > SILENCE_THRESHOLD:
                print(f"🔊 Processing audio (level: {rms:.4f})...")
                
                # Transcribe with Whisper
                result = transcribe_with_whisper(full_audio)
                if result:
                    text, inference_time = result
                    print(f"🎤 Whisper ({inference_time:.2f}s): {text}")
                    
                    # Try to generate diagram
                    mermaid_code = self.process_transcript(text)
                    if mermaid_code:
                        print("\n🎯 MERMAID DIAGRAM GENERATED:")
                        print("```mermaid")
                        print(mermaid_code)
                        print("```\n")
                    else:
                        print("💬 Regular transcript (no diagram detected)")
                else:
                    print("🔇 No speech detected")
            else:
                print("🔇 Silent audio, skipping...")
            
            # Reset buffer
            self.audio_buffer = []
            self.buffer_duration = 0.0
    
    def start_listening(self):
        """Start the enhanced real-time transcription."""
        print("🎤 Starting Enhanced Voice-to-Mermaid Pipeline")
        print("🧠 Whisper.cpp for speech recognition")
        if self.use_llama:
            print("⚡ LLaMA v3.1 8B Instruct for diagram generation")
        else:
            print("📝 Regex-based diagram detection")
        print("\n💡 Example commands:")
        print("   • 'Draw a flowchart from user to database'")
        print("   • 'Create a sequence diagram showing API calls'")
        print("   • 'User authentication then dashboard access'")
        print("   • 'Data flows from sensors to analytics'")
        print("\nPress Ctrl+C to stop")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
            dtype=np.float32
        ):
            try:
                while running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        
        print("✅ Enhanced pipeline stopped")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Voice-to-Mermaid Pipeline with LLaMA Integration"
    )
    parser.add_argument(
        "--no-llama", 
        action="store_true",
        help="Disable LLaMA integration (use regex-based detection only)"
    )
    args = parser.parse_args()
    
    use_llama = not args.no_llama
    
    try:
        transcriber = EnhancedRealTimeTranscriber(use_llama=use_llama)
        transcriber.start_listening()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
