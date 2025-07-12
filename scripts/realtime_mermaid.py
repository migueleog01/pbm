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
import subprocess
import tempfile
import wave

# Audio configuration
SAMPLE_RATE = 16000  # 16 kHz for whisper
CHUNK_DURATION = 0.16  # 160ms chunks
BUFFER_DURATION = 2.0  # 2-second processing windows (increased for better phrases)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Voice activity detection parameters
SILENCE_THRESHOLD = 0.002  # Much lower threshold for better sensitivity
MIN_SPEECH_DURATION = 0.3  # Shorter minimum duration
SPEECH_PAUSE_THRESHOLD = 1.2  # Longer pause threshold to capture full sentences

# Global state
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_index = 0
running = True
speech_detected = False
speech_start_time = 0
last_speech_time = 0
last_level_report = 0  # For periodic audio level reporting


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nShutting down gracefully...")
    running = False


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of audio chunk."""
    return np.sqrt(np.mean(audio_chunk ** 2))


def has_speech(audio_chunk: np.ndarray) -> bool:
    """
    Simple voice activity detection based on RMS energy.
    
    Returns:
        bool: True if speech is detected, False otherwise
    """
    rms = calculate_rms(audio_chunk)
    return rms > SILENCE_THRESHOLD


def should_process_buffer(audio_chunk: np.ndarray) -> bool:
    """
    Determine if audio buffer should be processed based on speech activity.
    
    Returns:
        bool: True if buffer should be processed
    """
    # Check if there's meaningful audio content
    rms = calculate_rms(audio_chunk)
    
    # Very permissive - process almost any audio with some energy
    if rms < SILENCE_THRESHOLD:  # Much more permissive
        return False
    
    # Check for dynamic range (speech has varying amplitude)
    max_val = np.max(np.abs(audio_chunk))
    if max_val < SILENCE_THRESHOLD * 2:  # More permissive threshold
        return False
    
    # Check for audio activity throughout the buffer
    # Split buffer into segments and check if multiple segments have activity
    segment_size = len(audio_chunk) // 8  # Fewer segments for easier threshold
    active_segments = 0
    segment_energies = []
    
    for i in range(0, len(audio_chunk), segment_size):
        segment = audio_chunk[i:i + segment_size]
        if len(segment) > 0:
            segment_rms = calculate_rms(segment)
            segment_energies.append(segment_rms)
            if segment_rms > SILENCE_THRESHOLD:  # Much more permissive
                active_segments += 1
    
    # Require only 1 active segment to consider it speech
    if active_segments < 1:
        return False
    
    # Much more permissive - process almost everything
    return True


def detect_diagram_command(text: str) -> Optional[dict]:
    """
    Detect diagram commands in transcribed text with flexible node names.
    
    Returns:
        dict: {
            'type': 'flowchart'|'sequence'|'mindmap'|'etc',
            'nodes': [list of node names],
            'connections': [list of (source, target) tuples],
            'command': original command text
        } if command detected, None otherwise
    """
    text_lower = text.lower()
    
    # Diagram trigger words
    diagram_triggers = [
        'diagram', 'flowchart', 'flow chart', 'chart', 'graph', 'map',
        'sequence', 'process', 'workflow', 'mindmap', 'mind map'
    ]
    
    # Action words
    action_words = [
        'draw', 'create', 'make', 'build', 'design', 'show', 'generate'
    ]
    
    # Check if this looks like a diagram command
    has_trigger = any(trigger in text_lower for trigger in diagram_triggers)
    has_action = any(action in text_lower for action in action_words)
    
    if not (has_trigger or has_action):
        return None
    
    # Connection words and patterns
    connection_patterns = [
        # "A to B", "A goes to B", "A connects to B"
        (r'(.+?)\s+(?:to|goes\s+to|connects\s+to|leads\s+to|points\s+to)\s+(.+)', 'flowchart'),
        # "A then B", "A followed by B"
        (r'(.+?)\s+(?:then|followed\s+by|after)\s+(.+)', 'flowchart'),
        # "A calls B", "A sends to B" (for sequence diagrams)
        (r'(.+?)\s+(?:calls|sends\s+to|requests|communicates\s+with)\s+(.+)', 'sequence'),
        # "from A to B"
        (r'from\s+(.+?)\s+to\s+(.+)', 'flowchart'),
        # "A and B are connected"
        (r'(.+?)\s+and\s+(.+?)\s+(?:are\s+)?connected', 'flowchart'),
        # "A with B" (for simple associations)
        (r'(.+?)\s+with\s+(.+)', 'flowchart'),
    ]
    
    # Try to find connections
    connections = []
    diagram_type = 'flowchart'  # default
    
    for pattern, dtype in connection_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            diagram_type = dtype
            for match in matches:
                if len(match) == 2:
                    source = clean_node_name(match[0])
                    target = clean_node_name(match[1])
                    if source and target:
                        connections.append((source, target))
    
    # If no connections found, try to extract node names from context
    if not connections:
        # Look for lists of items that might be nodes
        node_patterns = [
            r'(?:diagram|flowchart|chart)\s+(?:with|of|showing|for)\s+(.+)',
            r'(?:create|make|draw)\s+(?:a\s+)?(?:diagram|flowchart|chart)\s+(.+)',
            r'(?:boxes|nodes|elements|components)\s+(?:called|named|labeled)\s+(.+)',
        ]
        
        for pattern in node_patterns:
            match = re.search(pattern, text_lower)
            if match:
                node_text = match.group(1)
                # Try to extract individual node names
                nodes = extract_node_names(node_text)
                if len(nodes) >= 2:
                    # Create connections between sequential nodes
                    for i in range(len(nodes) - 1):
                        connections.append((nodes[i], nodes[i + 1]))
                    break
    
    if connections:
        # Extract all unique nodes
        all_nodes = set()
        for source, target in connections:
            all_nodes.add(source)
            all_nodes.add(target)
        
        return {
            'type': diagram_type,
            'nodes': list(all_nodes),
            'connections': connections,
            'command': text
        }
    
    return None


def clean_node_name(name: str) -> str:
    """Clean and normalize node names."""
    # Remove common filler words and clean up
    name = name.strip()
    
    # Remove diagram-related words that might be included
    remove_words = [
        'diagram', 'flowchart', 'chart', 'box', 'node', 'element',
        'the', 'a', 'an', 'this', 'that', 'called', 'named'
    ]
    
    words = name.split()
    cleaned_words = []
    
    for word in words:
        if word.lower() not in remove_words and len(word) > 1:
            cleaned_words.append(word.title())  # Title case for node names
    
    return ' '.join(cleaned_words) if cleaned_words else name.title()


def extract_node_names(text: str) -> list:
    """Extract node names from text, handling various separators."""
    # Common separators for lists
    separators = [' and ', ' with ', ' plus ', ' also ', ', ', ' then ', ' or ']
    
    # Start with the full text
    items = [text]
    
    # Split by each separator
    for separator in separators:
        new_items = []
        for item in items:
            new_items.extend(item.split(separator))
        items = new_items
    
    # Clean up each item
    cleaned_items = []
    for item in items:
        cleaned = clean_node_name(item)
        if cleaned and len(cleaned) > 1:
            cleaned_items.append(cleaned)
    
    return cleaned_items[:10]  # Limit to 10 nodes to avoid noise


def generate_mermaid_code(diagram_info: dict) -> str:
    """Generate Mermaid code block for detected diagram."""
    diagram_type = diagram_info['type']
    connections = diagram_info['connections']
    
    if diagram_type == 'sequence':
        # Sequence diagram
        mermaid_lines = ['```mermaid', 'sequenceDiagram']
        for source, target in connections:
            mermaid_lines.append(f'    {source}->>+{target}: interaction')
        mermaid_lines.append('```')
        
    elif diagram_type == 'mindmap':
        # Mind map (simplified)
        mermaid_lines = ['```mermaid', 'mindmap']
        mermaid_lines.append('  root((Central Topic))')
        nodes = diagram_info['nodes']
        for node in nodes:
            mermaid_lines.append(f'    {node}')
        mermaid_lines.append('```')
        
    else:
        # Default to flowchart
        mermaid_lines = ['```mermaid', 'graph TD']
        
        # Add connections
        for source, target in connections:
            # Use safe node IDs (replace spaces with underscores)
            source_id = source.replace(' ', '_').replace('-', '_')
            target_id = target.replace(' ', '_').replace('-', '_')
            mermaid_lines.append(f'    {source_id}["{source}"] --> {target_id}["{target}"]')
        
        mermaid_lines.append('```')
    
    return '\n'.join(mermaid_lines)


def filter_transcript(transcript: str) -> tuple[str, bool]:
    """
    Filter and clean transcript, returning (cleaned_text, should_display).
    
    Returns:
        tuple: (cleaned_transcript, should_display_flag)
    """
    if not transcript or not transcript.strip():
        return "", False
    
    # Clean up transcript
    cleaned = transcript.strip()
    
    # Remove timestamp information that whisper-cli might add
    import re
    cleaned = re.sub(r'\[[\d:.\s\-\>]+\]', '', cleaned).strip()
    
    # Skip blank audio markers
    if '[BLANK_AUDIO]' in cleaned.upper():
        return "", False
    
    # Much more permissive - show almost everything
    if len(cleaned) < 1:  # Only skip completely empty
        return "", False
    
    # Show everything else - no filtering for filler words or repetition
    return cleaned, True


def process_audio_chunk(audio_chunk: np.ndarray, whisper_cli_path: str, model_path: str) -> Optional[str]:
    """Process audio chunk through whisper CLI and return transcript."""
    try:
        # Ensure audio is in the right format for whisper
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Whisper expects audio in range [-1, 1]
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Write audio to WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                
                # Convert float32 to int16
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
        
        # Run whisper CLI
        result = subprocess.run([
            whisper_cli_path,
            '-m', model_path,
            '-f', tmp_path,
            '--output-txt'
        ], capture_output=True, text=True)
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        if result.returncode == 0:
            # Extract text from stdout
            text = result.stdout.strip()
            # Remove file path prefix that whisper-cli adds
            if text.startswith(tmp_path):
                text = text[len(tmp_path):].strip()
            return text if text else None
        else:
            print(f"Whisper CLI error: {result.stderr}")
            return None
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def audio_callback(indata, frames, time_info, status):
    """Callback for audio input stream."""
    global audio_buffer, buffer_index, speech_detected, speech_start_time, last_speech_time, last_level_report
    
    if status:
        print(f"Audio callback status: {status}")
    
    # Convert to mono if stereo
    if len(indata.shape) > 1:
        audio_data = indata[:, 0]
    else:
        audio_data = indata.flatten()
    
    # Check for speech activity in this chunk
    current_time = time.time()
    chunk_has_speech = has_speech(audio_data)
    
    # Periodic audio level reporting (every 15 seconds)
    if current_time - last_level_report > 15.0:  # Less frequent reporting
        rms_level = calculate_rms(audio_data)
        if rms_level > 0.001:  # Only report if there's some audio
            print(f"🔊 Audio level: {rms_level:.4f} (threshold: {SILENCE_THRESHOLD:.4f})")
        last_level_report = current_time
    
    if chunk_has_speech:
        if not speech_detected:
            speech_detected = True
            speech_start_time = current_time
            print("🎤 Speech detected...")
        last_speech_time = current_time
    else:
        # Check if we should end speech detection
        if speech_detected and (current_time - last_speech_time) > SPEECH_PAUSE_THRESHOLD:
            speech_detected = False
            print("⏸️  Speech ended")
    
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
    global running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-Time Voice-to-Mermaid Pipeline")
    parser.add_argument("--input", help="Input WAV file (instead of microphone)")
    parser.add_argument("--model", help="Path to whisper model", 
                       default="whisper.cpp/models/ggml-tiny.en.bin")
    parser.add_argument("--whisper-cli", help="Path to whisper CLI executable",
                       default="whisper.cpp/build/bin/whisper-cli")
    args = parser.parse_args()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check whisper CLI
    whisper_cli_path = Path(args.whisper_cli)
    if not whisper_cli_path.exists():
        print(f"ERROR: Whisper CLI not found: {whisper_cli_path}")
        print("Please build whisper.cpp first:")
        print("  cd whisper.cpp && mkdir build && cd build")
        print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
        print("  cmake --build . --config Release")
        sys.exit(1)
    
    # Check whisper model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Please download the model using:")
        print("  cd whisper.cpp && bash models/download-ggml-model.sh tiny.en")
        sys.exit(1)
    
    print(f"Using whisper CLI: {whisper_cli_path}")
    print(f"Using whisper model: {model_path}")
    print("Setup complete!")
    
    # Handle file input vs microphone
    if args.input:
        print(f"Processing file: {args.input}")
        # TODO: Implement file processing
        print("File input not yet implemented. Use microphone mode.")
        return
    
    # Set up audio stream
    print("Starting microphone capture...")
    print("Examples of diagram commands you can say:")
    print("  • 'Draw a diagram from User Login to Database Server'")
    print("  • 'Create a flowchart showing Payment Gateway connects to API'")
    print("  • 'Make a sequence diagram where Client calls Authentication Service'")
    print("  • 'User Authentication then Database Query then Response'")
    print("  • 'Frontend and Backend are connected'")
    print("Press Ctrl+C to exit")
    print("🎧 Listening for speech...")
    
    # Initialize timing variables
    global speech_detected, speech_start_time, last_speech_time, last_level_report, buffer_index, audio_buffer
    speech_detected = False
    speech_start_time = 0
    last_speech_time = 0
    last_level_report = 0
    
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
                
                # Process buffer when speech activity is detected and has ended
                # or when buffer is full and contains speech
                should_process = False
                
                if buffer_index > 0:
                    # Get current buffer content
                    if buffer_index >= BUFFER_SIZE:
                        process_buffer = audio_buffer.copy()
                    else:
                        process_buffer = audio_buffer[:buffer_index].copy()
                    
                    # Check if we should process this buffer
                    if should_process_buffer(process_buffer):
                        # Process more frequently - after speech ends
                        if not speech_detected and last_speech_time > 0 and (current_time - last_speech_time) <= 2.0:
                            should_process = True
                        # Or process if buffer has content and speech is active
                        elif speech_detected and buffer_index >= BUFFER_SIZE * 0.4:
                            should_process = True
                        # Or process regularly (shorter intervals)
                        elif (current_time - last_process_time) >= BUFFER_DURATION * 0.8:
                            should_process = True
                        # Or process if buffer is getting full regardless
                        elif buffer_index >= BUFFER_SIZE * 0.8:
                            should_process = True
                
                if should_process:
                    print("🔄 Processing audio...")
                    
                    # Process through whisper
                    transcript = process_audio_chunk(process_buffer, str(whisper_cli_path), str(model_path))
                    
                    if transcript and transcript.strip():
                        # Clean up transcript
                        transcript, should_display = filter_transcript(transcript)
                        
                        if should_display:
                            # Check for diagram commands
                            diagram_info = detect_diagram_command(transcript)
                            
                            if diagram_info:
                                mermaid_code = generate_mermaid_code(diagram_info)
                                print(f"\n🎯 DIAGRAM COMMAND DETECTED:")
                                print(f"   Type: {diagram_info['type']}")
                                print(f"   Nodes: {', '.join(diagram_info['nodes'])}")
                                print(f"   Original: {diagram_info['command']}")
                                print(mermaid_code)
                                print()
                            else:
                                print(f"💬 {transcript}")
                        else:
                            # Show a minimal message for filtered content
                            print(f"🔇 [filtered]")
                    else:
                        print("🔇 No speech detected")
                    
                    last_process_time = current_time
                    
                    # Reset buffer after processing
                    buffer_index = 0
                    audio_buffer.fill(0)
                    
                    # Show listening indicator again
                    print("🎧 Listening for speech...")
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print("Goodbye!")


if __name__ == "__main__":
    main() 