#!/usr/bin/env python3
"""
Enhanced Voice-to-Mermaid Pipeline with Wake Word Detection
"""

import argparse
import os
import sys
import time
from pathlib import Path
import subprocess
import tempfile
import wave
import signal
import re
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "RenderMermaid"))
from render_flowchart import render_mermaid_html

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
SILENCE_THRESHOLD = 0.005

# Wake word settings
WAKE_WORDS = ["hey pbm", "hey pvm", "pbm", "pvm", "ppm"]
WAKE_WORD_TIMEOUT = 10.0

# Paths
WHISPER_CLI = 'whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL = 'whisper.cpp/models/ggml-base.en-q5_1.bin'

# Global state
running = True
listening_for_command = False
wake_word_detected_time = 0

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n🛑 Stopping enhanced pipeline...")
    running = False

def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate RMS of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))

def detect_wake_word(text: str) -> bool:
    """Detect wake word in text."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    for wake_word in WAKE_WORDS:
        if wake_word in text_lower:
            return True
    return False

def extract_nodes_from_speech(text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Extract actual nodes and connections from speech."""
    if not text:
        return [], []
    
    # Clean up text
    text_lower = text.lower()
    for wake_word in WAKE_WORDS:
        text_lower = text_lower.replace(wake_word, "")
    
    # Remove command words
    for word in ["draw", "create", "make", "show", "flowchart", "diagram"]:
        text_lower = text_lower.replace(word, "")
    
    text_lower = text_lower.strip()
    
    # Extract connections
    connections = []
    nodes = set()
    
    # Pattern: "A to B"
    pattern = r'(\w+(?:\s+\w+)*?)\s+to\s+(\w+(?:\s+\w+)*)'
    matches = re.findall(pattern, text_lower)
    for match in matches:
        source = match[0].strip().title()
        target = match[1].strip().title()
        if source and target:
            connections.append((source, target))
            nodes.add(source)
            nodes.add(target)
    
    # Pattern: "A then B"
    pattern = r'(\w+(?:\s+\w+)*?)\s+then\s+(\w+(?:\s+\w+)*)'
    matches = re.findall(pattern, text_lower)
    for match in matches:
        source = match[0].strip().title()
        target = match[1].strip().title()
        if source and target:
            connections.append((source, target))
            nodes.add(source)
            nodes.add(target)
    
    return list(nodes), connections

def generate_mermaid_from_speech(text: str) -> Optional[str]:
    """Generate Mermaid diagram from actual speech content."""
    nodes, connections = extract_nodes_from_speech(text)
    
    if not connections:
        return None
    
    # Generate flowchart
    lines = ["graph TD"]
    for source, target in connections:
        source_id = source.replace(" ", "_")
        target_id = target.replace(" ", "_")
        lines.append(f"    {source_id}[{source}] --> {target_id}[{target}]")
    
    return "\n".join(lines)

def transcribe_with_whisper(audio_data: np.ndarray) -> Optional[Tuple[str, float]]:
    """Transcribe audio using Whisper."""
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    
    rms = calculate_rms(audio_data)
    if rms < SILENCE_THRESHOLD:
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
    
    try:
        cmd = [WHISPER_CLI, '-m', WHISPER_MODEL, '-f', tmp_file.name, 
               '-t', '8', '--beam-size', '3', '--best-of', '1', 
               '--language', 'en', '--no-timestamps']
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        end_time = time.time()
        
        if result.returncode == 0:
            full_output = result.stdout.strip()
            lines = [line.strip() for line in full_output.split('\n') if line.strip()]
            
            for line in reversed(lines):
                if not any(pattern in line.lower() for pattern in ['[', 'whisper', 'processing', 'model']):
                    if line and len(line) > 2:
                        return line, end_time - start_time
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")
    finally:
        Path(tmp_file.name).unlink(missing_ok=True)
    
    return None

def should_generate_diagram(text: str) -> bool:
    """Check if text contains diagram keywords."""
    text_lower = text.lower()
    keywords = ['to', 'then', 'flow', 'chart', 'diagram']
    return any(keyword in text_lower for keyword in keywords)

class EnhancedTranscriber:
    """Enhanced transcriber with wake word detection."""
    
    def __init__(self):
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Check setup
        if not Path(WHISPER_CLI).exists():
            print(f"❌ Whisper CLI not found: {WHISPER_CLI}")
            sys.exit(1)
        
        if not Path(WHISPER_MODEL).exists():
            print(f"❌ Whisper model not found: {WHISPER_MODEL}")
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        print("✅ Enhanced voice-to-Mermaid pipeline ready!")
    
    def process_transcript(self, text: str) -> Optional[str]:
        """Process transcript with wake word detection."""
        global listening_for_command, wake_word_detected_time
        
        if not text:
            return None
        
        # Check for wake word
        if detect_wake_word(text):
            listening_for_command = True
            wake_word_detected_time = time.time()
            print("👂 Wake word detected! Listening for diagram command...")
            
            # Check if command is in same utterance
            if should_generate_diagram(text):
                listening_for_command = False
                return generate_mermaid_from_speech(text)
            else:
                return "WAKE_WORD_ONLY"
        
        # If listening for command
        elif listening_for_command:
            if should_generate_diagram(text):
                listening_for_command = False
                return generate_mermaid_from_speech(text)
            else:
                print("💭 Waiting for diagram command...")
                return None
        
        return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Handle audio input."""
        global running, listening_for_command, wake_word_detected_time
        
        if not running:
            return
        
        # Check timeout
        if listening_for_command and time.time() - wake_word_detected_time > WAKE_WORD_TIMEOUT:
            listening_for_command = False
            print("⏰ Wake word timeout - back to listening for 'Hey PBM'")
        
        # Add to buffer
        audio_chunk = indata.flatten()
        self.audio_buffer.append(audio_chunk)
        self.buffer_duration += len(audio_chunk) / SAMPLE_RATE
        
        # Process when buffer full
        if self.buffer_duration >= CHUNK_DURATION:
            full_audio = np.concatenate(self.audio_buffer)
            rms = calculate_rms(full_audio)
            
            if rms > SILENCE_THRESHOLD:
                result = transcribe_with_whisper(full_audio)
                if result:
                    text, inference_time = result
                    status_icon = "👂" if listening_for_command else "🎤"
                    print(f"{status_icon} Whisper ({inference_time:.2f}s): {text}")
                    
                    mermaid_result = self.process_transcript(text)
                    if mermaid_result == "WAKE_WORD_ONLY":
                        print("🔄 Ready for diagram command...")
                    elif mermaid_result:
                        print("\n🎯 MERMAID DIAGRAM GENERATED:")
                        print("```mermaid")
                        print(mermaid_result)
                        print("```\n")
                        output_path = os.path.join(os.path.dirname(__file__), "RenderMermaid", "whisper_output.txt")
                        with open(output_path, "w") as f:
                            f.write(f"```mermaid\n{mermaid_result}\n```")
                            print("TEXT WRITTEN TO WHISPER_OUTPUT.TXT FILE!")
                        
                        render_mermaid_html()
                    elif not listening_for_command:
                        print("💬 Say 'Hey PBM' followed by a diagram command")
            
            # Reset buffer
            self.audio_buffer = []
            self.buffer_duration = 0.0
    
    def start_listening(self):
        """Start listening."""
        print("🎤 Enhanced Voice-to-Mermaid Pipeline with Wake Word Detection")
        print("🧠 Whisper.cpp for speech recognition")
        print("🎨 Smart content-aware diagram generation")
        
        print("\n🔥 NEW FEATURES:")
        print("   🎯 Wake Word Detection: Say 'Hey PBM' first!")
        print("   🎨 Smart Content Parsing: Uses your actual words!")
        
        print("\n💡 How to use:")
        print("   1. Say: 'Hey PBM'")
        print("   2. Then: 'Draw user to database'")
        print("   3. Or: 'Hey PBM, create login to dashboard'")
        
        print("\n📝 Example commands:")
        print("   • 'Hey PBM, draw user to database'")
        print("   • 'Hey PBM, create login to dashboard'")
        print("   • 'Hey PBM, show payment to confirmation'")
        print("   • 'Hey PBM, make authentication then success'")
        
        print("\nPress Ctrl+C to stop")
        print("👂 Listening for 'Hey PBM'...")
        
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
    parser = argparse.ArgumentParser(description="Enhanced Voice-to-Mermaid Pipeline")
    parser.add_argument("--no-llama", action="store_true", help="Disable LLaMA")
    args = parser.parse_args()
    
    try:
        transcriber = EnhancedTranscriber()
        transcriber.start_listening()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
