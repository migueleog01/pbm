#!/usr/bin/env python3
"""
Enhanced Voice-to-Mindmap Pipeline with Intelligent Relationship Detection
"""

import os
import re
import sys
import time
import numpy as np
import sounddevice as sd
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import spacy

# Import the HTML rendering function
sys.path.append(os.path.join(os.path.dirname(__file__), "RenderMermaid"))
from render_flowchart import render_mermaid_html

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
SILENCE_THRESHOLD = 0.005
INACTIVITY_TIMEOUT = 5.0  # 5 seconds of silence before processing
WHISPER_CLI = 'whisper.cpp/build/bin/whisper-cli'
WHISPER_MODEL = 'whisper.cpp/models/ggml-base.en-q5_1.bin'

class IntelligentMindmapTranscriber:
    def __init__(self):
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.accumulated_transcripts = []
        self.last_activity_time = time.time()
        self.is_processing = False
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Please run: python3 -m spacy download en_core_web_sm")
            sys.exit(1)
    
    def extract_mindmap_elements(self, text: str) -> Dict[str, List[str]]:
        """Extract elements for mindmap generation using spaCy."""
        if not text or not text.strip():
            return {"entities": [], "keywords": [], "concepts": [], "categories": [], "relationships": []}
        
        doc = self.nlp(text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                entities.append(ent.text)
        
        # Extract important keywords and concepts
        keywords = []
        concepts = []
        categories = []
        
        for token in doc:
            # Skip stop words and punctuation
            if token.is_stop or token.is_punct or token.is_space:
                continue
            
            # Extract important keywords (nouns, verbs, adjectives)
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and len(token.text) > 2:
                keywords.append(token.text)
                
                # Categorize concepts
                if token.pos_ == "NOUN":
                    concepts.append(token.text)
                elif token.pos_ == "ADJ":
                    categories.append(token.text)
        
        # Extract relationships and hierarchical patterns
        relationships = self.extract_relationships(doc)
        
        # Remove duplicates while preserving order
        def remove_duplicates(items):
            seen = set()
            unique_items = []
            for item in items:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique_items.append(item)
            return unique_items
        
        return {
            "entities": remove_duplicates(entities),
            "keywords": remove_duplicates(keywords),
            "concepts": remove_duplicates(concepts),
            "categories": remove_duplicates(categories),
            "relationships": relationships
        }
    
    def extract_relationships(self, doc) -> List[Tuple[str, str, str]]:
        """Extract relationships between concepts."""
        relationships = []
        
        # Look for dependency patterns
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "prep"] and token.head.pos_ in ["VERB", "NOUN"]:
                if token.pos_ in ["NOUN", "PROPN"] and token.head.pos_ in ["VERB", "NOUN"]:
                    relationships.append((token.text, "relates_to", token.head.text))
        
        # Look for prepositional relationships
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "VERB"]:
                for child in token.children:
                    if child.pos_ in ["NOUN", "PROPN"]:
                        relationships.append((token.head.text, token.text, child.text))
        
        return relationships
    
    def generate_intelligent_mindmap(self, nlp_data: Dict[str, List[str]]) -> str:
        """Generate intelligent mindmap following Mermaid mindmap syntax guide."""
        entities = nlp_data["entities"]
        keywords = nlp_data["keywords"]
        concepts = nlp_data["concepts"]
        categories = nlp_data["categories"]
        relationships = nlp_data["relationships"]
        
        if not keywords and not entities and not concepts:
            return """```mermaid\nmindmap\n    (No content to summarize)\n```"""
        
        # Determine root concept
        root = self.determine_root_concept(entities, keywords, concepts)
        
        # Start mindmap with proper syntax
        mindmap_lines = ["mindmap", f"    {root}"]
        
        # Create hierarchical structure following Mermaid guidelines
        hierarchy = self.create_mermaid_hierarchy(nlp_data, root)
        
        # Build mindmap with proper indentation (4 spaces per level)
        for level1, level1_items in hierarchy.items():
            mindmap_lines.append(f"        {level1}")
            
            for level2, level2_items in level1_items.items():
                mindmap_lines.append(f"            {level2}")
                
                for level3 in level2_items:
                    mindmap_lines.append(f"                {level3}")
        
        mindmap_code = '\n'.join(mindmap_lines)
        return f"""```mermaid\n{mindmap_code}\n```"""
    
    def determine_root_concept(self, entities: List[str], keywords: List[str], concepts: List[str]) -> str:
        """Determine the main topic/root concept."""
        # Priority: entities > concepts > keywords
        if entities:
            return entities[0]
        elif concepts:
            return concepts[0]
        elif keywords:
            return keywords[0]
        else:
            return "Main Topic"
    
    def create_mermaid_hierarchy(self, nlp_data: Dict[str, List[str]], root: str) -> Dict[str, Dict[str, List[str]]]:
        """Create hierarchical structure following Mermaid mindmap guidelines."""
        entities = nlp_data["entities"]
        keywords = nlp_data["keywords"]
        concepts = nlp_data["concepts"]
        categories = nlp_data["categories"]
        relationships = nlp_data["relationships"]
        
        hierarchy = {}
        
        # Create main branches based on content type (following Mermaid structure)
        if entities:
            hierarchy["Entities"] = {}
            for entity in entities[1:3]:  # Skip root entity
                hierarchy["Entities"][entity] = []
        
        if concepts:
            hierarchy["Concepts"] = {}
            for concept in concepts[:3]:
                hierarchy["Concepts"][concept] = []
        
        if categories:
            hierarchy["Categories"] = {}
            for category in categories[:2]:
                hierarchy["Categories"][category] = []
        
        # Add keywords as sub-branches (distribute evenly)
        remaining_keywords = [kw for kw in keywords if kw not in [root] + entities + concepts + categories]
        
        # Distribute keywords across branches following Mermaid indentation rules
        if hierarchy:
            branch_names = list(hierarchy.keys())
            for i, keyword in enumerate(remaining_keywords[:6]):
                branch_name = branch_names[i % len(branch_names)]
                branch_items = list(hierarchy[branch_name].keys())
                if branch_items:
                    target_branch = branch_items[i % len(branch_items)]
                    hierarchy[branch_name][target_branch].append(keyword)
        
        # If no hierarchy created, create simple structure following Mermaid syntax
        if not hierarchy:
            hierarchy["Main"] = {}
            for keyword in keywords[:3]:
                hierarchy["Main"][keyword] = []
        
        return hierarchy
    
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
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
        
        # Extract mindmap elements using spaCy
        nlp_data = self.extract_mindmap_elements(full_text)
        print(f"\n🧠 Mindmap Analysis:")
        print(f"   Entities: {nlp_data['entities']}")
        print(f"   Keywords: {nlp_data['keywords']}")
        print(f"   Concepts: {nlp_data['concepts']}")
        print(f"   Categories: {nlp_data['categories']}")
        print(f"   Relationships: {nlp_data['relationships']}")
        
        # Generate intelligent mindmap following Mermaid syntax
        mindmap = self.generate_intelligent_mindmap(nlp_data)
        
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
        print("🎤 Intelligent Voice-to-Mindmap Pipeline")
        print("📝 Continuously registers speech and processes after 5s of silence")
        print("🧠 Uses spaCy for intelligent keyword extraction and relationship detection")
        print("🌳 Creates hierarchical mindmaps following Mermaid syntax guidelines")
        print("⏰ Inactivity timeout: 5 seconds")
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
                print("\n✅ Intelligent mindmap pipeline stopped")

def main():
    print("🎤 Enhanced Real-time Intelligent Mindmap Generator")
    print("Workflow: Continuous speech → 5s silence → spaCy analysis → Mermaid mindmap generation")
    
    try:
        transcriber = IntelligentMindmapTranscriber()
        transcriber.start_listening()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 