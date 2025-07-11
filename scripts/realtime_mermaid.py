#!/usr/bin/env python3
"""
Real-Time Voice-to-Mermaid Pipeline

Captures microphone audio, processes it through whisper.cpp on CPU,
and converts simple diagram commands into Mermaid code blocks.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import whispercpp as wc
except ImportError:
    print("ERROR: whispercpp not found. Please install requirements.txt")
    sys.exit(1)

# Audio configuration
SAMPLE_RATE = 16000  # 16 kHz for whisper
CHUNK_DURATION = 0.16  # 160ms chunks
BUFFER_DURATION = 1.0  # 1-second processing windows
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Global state
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_index = 0
whisper_model = None
running = True


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nShutting down gracefully...")
    running = False


def detect_diagram_command(text: str) -> Optional[tuple]:
    """
    Detect diagram commands in transcribed text.
    
    Returns:
        tuple: (source_node, target_node) if command detected, None otherwise
    """
    # Regex pattern for diagram commands
    patterns = [
        r'(?:draw|create|make)\s+diagram\s+(\w+)\s+to\s+(\w+)',
        r'(?:draw|create|make)\s+a\s+diagram\s+from\s+(\w+)\s+to\s+(\w+)',
        r'diagram\s+(\w+)\s+(?:goes\s+to|leads\s+to|connects\s+to)\s+(\w+)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1), match.group(2)
    
    return None


def generate_mermaid_code(source: str, target: str) -> str:
    """Generate Mermaid code block for a simple diagram."""
    return f"""```mermaid
graph TD
    {source} --> {target}
```"""


def process_audio_chunk(audio_chunk: np.ndarray) -> Optional[str]:
    """Process audio chunk through whisper and return transcript."""
    try:
        # Ensure audio is in the right format for whisper
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Whisper expects audio in range [-1, 1]
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        # Transcribe using whisper
        result = whisper_model.transcribe(audio_chunk)
        
        # Extract text from result
        if hasattr(result, 'text'):
            text = result.text.strip()
        elif isinstance(result, str):
            text = result.strip()
        else:
            text = str(result).strip()
        
        return text if text else None
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def audio_callback(indata, frames, time, status):
    """Callback for audio input stream."""
    global audio_buffer, buffer_index
    
    if status:
        print(f"Audio callback status: {status}")
    
    # Convert to mono if stereo
    if len(indata.shape) > 1:
        audio_data = indata[:, 0]
    else:
        audio_data = indata.flatten()
    
    # Add to circular buffer
    chunk_size = len(audio_data)
    if buffer_index + chunk_size <= BUFFER_SIZE:
        audio_buffer[buffer_index:buffer_index + chunk_size] = audio_data
        buffer_index += chunk_size
    else:
        # Wrap around buffer
        overflow = (buffer_index + chunk_size) - BUFFER_SIZE
        audio_buffer[buffer_index:BUFFER_SIZE] = audio_data[:-overflow]
        audio_buffer[0:overflow] = audio_data[-overflow:]
        buffer_index = overflow


def main():
    """Main application loop."""
    global whisper_model, running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-Time Voice-to-Mermaid Pipeline")
    parser.add_argument("--input", help="Input WAV file (instead of microphone)")
    parser.add_argument("--model", help="Path to whisper model", 
                       default="whisper.cpp/models/ggml-tiny.en.bin")
    args = parser.parse_args()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load whisper model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please download the model using:")
        print("  cd whisper.cpp && python models/download-ggml-model.py tiny.en")
        sys.exit(1)
    
    print(f"Loading whisper model: {model_path}")
    try:
        whisper_model = wc.Whisper(str(model_path))
    except Exception as e:
        print(f"ERROR: Failed to load whisper model: {e}")
        sys.exit(1)
    
    print("Model loaded successfully!")
    
    # Handle file input vs microphone
    if args.input:
        print(f"Processing file: {args.input}")
        # TODO: Implement file processing
        print("File input not yet implemented. Use microphone mode.")
        return
    
    # Set up audio stream
    print("Starting microphone capture...")
    print("Speak diagram commands like: 'draw diagram start to end'")
    print("Press Ctrl+C to exit")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=CHUNK_SIZE,
            dtype=np.float32
        ):
            last_process_time = time.time()
            
            while running:
                current_time = time.time()
                
                # Process buffer every second
                if current_time - last_process_time >= BUFFER_DURATION:
                    if buffer_index > 0:
                        # Get current buffer content
                        if buffer_index >= BUFFER_SIZE:
                            process_buffer = audio_buffer.copy()
                        else:
                            process_buffer = audio_buffer[:buffer_index].copy()
                        
                        # Process through whisper
                        transcript = process_audio_chunk(process_buffer)
                        
                        if transcript:
                            # Check for diagram commands
                            diagram_nodes = detect_diagram_command(transcript)
                            
                            if diagram_nodes:
                                source, target = diagram_nodes
                                mermaid_code = generate_mermaid_code(source, target)
                                print(f"\n🎯 DIAGRAM COMMAND DETECTED:")
                                print(mermaid_code)
                                print()
                            else:
                                print(f"💬 {transcript}")
                    
                    last_process_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print("Goodbye!")


if __name__ == "__main__":
    main() 