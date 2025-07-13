import os
import re
import sys
import time
import numpy as np
import sounddevice as sd
import subprocess
from pathlib import Path
from typing import List, Optional

# Import the HTML rendering function
sys.path.append(os.path.join(os.path.dirname(__file__), "RenderMermaid"))
from render_flowchart import render_mermaid_html

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
SILENCE_THRESHOLD = 0.005
INACTIVITY_TIMEOUT = 10.0  # 10 seconds of silence before processing
WHISPER_CLI = 'whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL = 'whisper.cpp/models/ggml-base.en-q5_1.bin'

class ContinuousTranscriber:
    def __init__(self):
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.accumulated_transcripts = []
        self.last_activity_time = time.time()
        self.is_processing = False
        
    def extract_keywords_and_phrases(self, text: str) -> List[str]:
        """Extract relevant keywords and phrases from text."""
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words and phrases
        words = text.split()
        keywords = []
        
        # Single word keywords (non-stop words)
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        # Two-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 5 and not any(stop in phrase for stop in stop_words):
                keywords.append(phrase)
        
        # Three-word phrases (important concepts)
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) > 8:
                keywords.append(phrase)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:20]  # Limit to top 20 keywords
    
    def generate_mindmap_from_keywords(self, keywords: List[str]) -> str:
        """Generate a mindmap from extracted keywords and phrases."""
        if not keywords:
            return """```mermaid\nmindmap\n    (No content to summarize)\n```"""
        
        # Use the first keyword as the root
        root = keywords[0].title()
        
        mindmap_lines = ["mindmap", f"    {root}"]
        
        # Group remaining keywords into branches
        remaining_keywords = keywords[1:]
        
        # Create branches (limit to 5 main branches for readability)
        for i, keyword in enumerate(remaining_keywords[:5]):
            mindmap_lines.append(f"        {keyword.title()}")
            
            # Add sub-branches for related concepts
            related_keywords = remaining_keywords[5:][i*3:(i+1)*3]
            for sub_keyword in related_keywords:
                mindmap_lines.append(f"            {sub_keyword.title()}")
        
        mindmap_code = '\n'.join(mindmap_lines)
        return f"""```mermaid\n{mindmap_code}\n```"""

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        return np.sqrt(np.mean(audio_chunk ** 2))

    def transcribe_with_whisper(self, audio_data: np.ndarray) -> Optional[str]:
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        rms = self.calculate_rms(audio_data)
        if rms < SILENCE_THRESHOLD:
            return None
        
        import tempfile, wave
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                full_output = result.stdout.strip()
                lines = [line.strip() for line in full_output.split('\n') if line.strip()]
                for line in reversed(lines):
                    if not any(pattern in line.lower() for pattern in ['[', 'whisper', 'processing', 'model']):
                        if line and len(line) > 2:
                            return line
        except Exception as e:
            print(f"❌ Whisper error: {e}")
        finally:
            Path(tmp_file.name).unlink(missing_ok=True)
        return None

    def process_accumulated_text(self):
        """Process accumulated transcripts after inactivity timeout."""
        if not self.accumulated_transcripts:
            return
        
        # Combine all transcripts
        full_text = " ".join(self.accumulated_transcripts)
        print(f"\n📝 Processing accumulated text ({len(self.accumulated_transcripts)} segments):")
        print(f"Full text: {full_text}")
        
        # Extract keywords and phrases
        keywords = self.extract_keywords_and_phrases(full_text)
        print(f"\n🔑 Extracted keywords: {keywords}")
        
        # Generate mindmap
        mindmap = self.generate_mindmap_from_keywords(keywords)
        
        # Write to file and render
        output_path = os.path.join(os.path.dirname(__file__), "RenderMermaid", "whisper_output.txt")
        with open(output_path, "w") as f:
            f.write(mindmap)
        
        print("\n=== GENERATED MERMAID MINDMAP ===")
        print(mindmap)
        print("=== END MINDMAP ===")
        
        render_mermaid_html()
        
        # Reset for next session
        self.accumulated_transcripts = []
        self.is_processing = False

    def audio_callback(self, indata, frames, time_info, status):
        """Handle audio input with continuous registration."""
        # Add to buffer
        audio_chunk = indata.flatten()
        self.audio_buffer.append(audio_chunk)
        self.buffer_duration += len(audio_chunk) / SAMPLE_RATE
        
        # Process when buffer full
        if self.buffer_duration >= CHUNK_DURATION:
            full_audio = np.concatenate(self.audio_buffer)
            rms = self.calculate_rms(full_audio)
            
            if rms > SILENCE_THRESHOLD:
                # Speech detected - update activity time
                self.last_activity_time = time.time()
                
                # Transcribe
                transcript = self.transcribe_with_whisper(full_audio)
                if transcript:
                    print(f"🎤 {transcript}")
                    self.accumulated_transcripts.append(transcript)
                    self.is_processing = False
            else:
                # Silence detected - check for timeout
                if (time.time() - self.last_activity_time > INACTIVITY_TIMEOUT and 
                    self.accumulated_transcripts and not self.is_processing):
                    self.is_processing = True
                    print(f"\n⏰ {INACTIVITY_TIMEOUT}s of inactivity detected. Processing accumulated text...")
                    self.process_accumulated_text()
            
            # Reset buffer
            self.audio_buffer = []
            self.buffer_duration = 0.0

    def start_listening(self):
        """Start continuous listening."""
        print("🎤 Continuous Voice-to-Mindmap Pipeline")
        print("📝 Continuously registers speech and processes after 10s of silence")
        print("🔑 Extracts keywords and phrases for mindmap generation")
        print("⏰ Inactivity timeout: 10 seconds")
        print("\nPress Ctrl+C to stop")
        print("👂 Listening continuously...")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
            dtype=np.float32
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                # Process any remaining text before exiting
                if self.accumulated_transcripts:
                    print("\n🔄 Processing remaining text before exit...")
                    self.process_accumulated_text()
                print("\n✅ Continuous pipeline stopped")

def main():
    print("🎤 Enhanced Real-time Mermaid Mindmap Generator")
    print("Workflow: Continuous speech → 10s silence → Keyword extraction → Mindmap generation")
    
    try:
        transcriber = ContinuousTranscriber()
        transcriber.start_listening()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



'''

"We need to develop a new application. 
First, we gather requirements from the client. 
Then we design the architecture and create wireframes. 
Next, we implement the features using Python and JavaScript. 
After testing, we deploy to production and monitor performance."


'''
