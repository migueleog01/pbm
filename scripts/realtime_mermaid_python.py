#!/usr/bin/env python3
"""
Real-Time Voice-to-Mermaid Pipeline (Python Whisper Version)

Captures microphone audio, processes it through Python whisper,
and converts simple diagram commands into Mermaid code blocks.
Optimized with settings from SETTINGS_SUMMARY.md
"""

import whisper
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

# Optimal settings from SETTINGS_SUMMARY.md
MODEL_NAME = 'base.en'  # 57MB quantized equivalent
OPTIMAL_THREADS = 8
OPTIMAL_BEAM = 3
OPTIMAL_BEST_OF = 1

# Audio configuration (from SETTINGS_SUMMARY.md)
SAMPLE_RATE = 16000  # 16 kHz for whisper
CHUNK_DURATION = 3.0  # 3-second chunks (optimal from SETTINGS_SUMMARY.md)
SILENCE_THRESHOLD = 0.005  # Optimal threshold from SETTINGS_SUMMARY.md

# Global state
running = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nShutting down gracefully...")
    running = False

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
    """Clean and normalize node names for Mermaid."""
    # Remove articles and common filler words
    name = re.sub(r'\b(?:the|a|an|and|or|but|with|in|on|at|to|for|of|from)\b', '', name)
    
    # Clean up extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name.split())
    
    # Remove special characters except letters, numbers, and spaces
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    
    return name

def extract_node_names(text: str) -> list:
    """Extract potential node names from text."""
    # Split by common separators
    separators = [',', ' and ', ' or ', ' with ', ';']
    
    parts = [text]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    
    # Clean and filter nodes
    nodes = []
    for part in parts:
        cleaned = clean_node_name(part)
        if cleaned and len(cleaned) > 1:
            nodes.append(cleaned)
    
    return nodes[:6]  # Limit to 6 nodes for readability

def generate_mermaid_code(diagram_info: dict) -> str:
    """Generate Mermaid code from diagram information."""
    diagram_type = diagram_info['type']
    connections = diagram_info['connections']
    
    if diagram_type == 'sequence':
        # Sequence diagram
        code = "sequenceDiagram\n"
        for source, target in connections:
            code += f"    {source}->>+{target}: Request\n"
            code += f"    {target}-->>-{source}: Response\n"
    
    elif diagram_type == 'mindmap':
        # Mind map
        code = "mindmap\n"
        # For mindmap, create a hierarchical structure
        root = connections[0][0] if connections else "Root"
        code += f"  root({root})\n"
        
        added_nodes = {root}
        for source, target in connections:
            if source not in added_nodes:
                code += f"    {source}\n"
                added_nodes.add(source)
            if target not in added_nodes:
                code += f"    {target}\n"
                added_nodes.add(target)
    
    else:
        # Default flowchart
        code = "flowchart TD\n"
        for source, target in connections:
            # Create unique node IDs
            source_id = source.replace(" ", "")
            target_id = target.replace(" ", "")
            code += f"    {source_id}[{source}] --> {target_id}[{target}]\n"
    
    return code

class VoiceToMermaidTranscriber:
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
                
                # Check for diagram commands
                diagram_info = detect_diagram_command(text)
                if diagram_info:
                    print(f"🎨 Diagram detected: {diagram_info['type']}")
                    print(f"📝 Connections: {diagram_info['connections']}")
                    
                    mermaid_code = generate_mermaid_code(diagram_info)
                    print(f"\n🎯 MERMAID CODE:\n```mermaid\n{mermaid_code}```\n")
                    
                    # Save to file
                    timestamp = int(time.time())
                    filename = f"diagram_{timestamp}.md"
                    with open(filename, 'w') as f:
                        f.write(f"# Voice-Generated Diagram\n\n")
                        f.write(f"**Command:** {text}\n\n")
                        f.write(f"```mermaid\n{mermaid_code}```\n")
                    
                    print(f"💾 Saved to: {filename}")
                
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
        """Start real-time voice-to-mermaid transcription."""
        print("🎤 Starting Voice-to-Mermaid with optimal base.en settings")
        print(f"⚙️  Model: {MODEL_NAME}, Beam: {OPTIMAL_BEAM}, Best-of: {OPTIMAL_BEST_OF}")
        print(f"🎯 Target: <0.5s processing time (your optimal config)")
        print("🎨 Say diagram commands like:")
        print("   - 'Create flowchart A to B to C'")
        print("   - 'Draw diagram User calls API then Database'")
        print("   - 'Make chart Login goes to Dashboard'")
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
        
        print("✅ Voice-to-Mermaid stopped")

def main():
    """Main function."""
    print("🎨 Voice-to-Mermaid Pipeline")
    print("=" * 50)
    
    transcriber = VoiceToMermaidTranscriber()
    transcriber.start_listening()

if __name__ == "__main__":
    main() 