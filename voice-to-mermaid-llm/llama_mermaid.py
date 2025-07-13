#!/usr/bin/env python3
"""
LLaMA-powered Text-to-Mermaid Converter
Converts natural language descriptions into Mermaid diagrams using LLaMA v3.1 8B Instruct
Optimized for Apple Silicon (M1/M2/M3) with Metal backend acceleration
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


# Import the HTML rendering function
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RenderMermaid"))
from render_flowchart import render_mermaid_html

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    print("Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
    sys.exit(1)

# Configuration for optimal M1/M2/M3 performance
DEFAULT_MODEL_PATH = "models/llama-v3.1-8b-instruct.Q4_K_M.gguf"
OPTIMAL_THREADS = 4  # Reduced from 8 to avoid overload
CONTEXT_SIZE = 1024  # Reduced from 2048 to avoid memory issues
MAX_TOKENS = 256     # Reduced from 512 for simpler outputs
TEMPERATURE = 0.1    # Lower temperature for more consistent output

class LlamaMermaidConverter:
    """Converts text descriptions to Mermaid diagrams using LLaMA v3.1 8B Instruct."""
    
    def __init__(self, model_path: str, verbose: bool = False):
        """Initialize the LLaMA model with optimal settings for Apple Silicon."""
        self.model_path = Path(model_path)
        self.verbose = verbose
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🧠 Loading LLaMA v3.1 8B Instruct from {model_path}")
        print("⚡ Optimized for Apple Silicon with Metal backend")
        
        try:
            # Initialize LLaMA with conservative settings to avoid decoding errors
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=1024,           # Reduced context window
                n_threads=4,          # Fewer threads
                n_gpu_layers=0,       # CPU only to avoid GPU issues
                use_mlock=False,      # Don't lock memory
                verbose=self.verbose,
                # Disable Metal to avoid decoding issues
                metal=False,           # Disable Metal backend
                f16_kv=False,         # Use f32 for key/value cache
            )
            print("✅ LLaMA model loaded successfully with CPU-only mode!")
            
        except Exception as e:
            print(f"❌ Failed to load LLaMA model: {e}")
            raise
    
    def create_prompt(self, text: str) -> str:
        """Create a well-structured prompt for LLaMA v3.1 8B Instruct."""
        
        # Simplified system prompt for better compatibility
        system_prompt = """You are a diagram assistant. Convert natural language descriptions into valid Mermaid diagrams.

Rules:
1. Output ONLY valid Mermaid syntax - no explanations or code blocks
2. Use 'graph TD' for flowcharts
3. Use clear node names
4. Keep diagrams simple
5. Feel free to use looping or bidirectional arrows when appropriate, according to the context.

Examples:
Input: "User logs in and accesses dashboard"
Output: graph TD
    User[User] --> Login[Login]
    Login --> Dashboard[Dashboard]"""

        # Use a simpler format without special tokens that cause decoding errors
        prompt = f"""{system_prompt}

Convert to Mermaid diagram: {text.strip()}

Mermaid diagram:"""
        
        return prompt
    
    def generate_mermaid(self, text: str) -> Optional[str]:
        """Generate Mermaid diagram from text description."""
        if not text.strip():
            return None
        
        print(f"🔄 Converting: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        prompt = self.create_prompt(text)
        
        try:
            start_time = time.time()
            
            # Generate response with optimal settings
            response = self.llm(
                prompt,
                max_tokens=256,      # Reduced for simpler outputs
                temperature=0.1,     # Lower temperature for consistency
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["\n\n", "```", "Description:", "Mermaid diagram:"],  # Simpler stop tokens
                echo=False  # Don't echo the prompt
            )
            
            end_time = time.time()
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean up the output
            mermaid_code = self.clean_mermaid_output(generated_text)
            
            if mermaid_code:
                print(f"⚡ Generated in {end_time - start_time:.2f}s")
                
                # Write to whisper_output.txt and render HTML
                self.write_to_whisper_output_and_render(mermaid_code)
                
                return mermaid_code
            else:
                print("❌ Failed to generate valid Mermaid code")
                return None
            
        except Exception as e:
            print(f"❌ Error generating diagram: {e}")
            return None
    
    def clean_mermaid_output(self, text: str) -> Optional[str]:
        """Clean and validate Mermaid output from LLaMA."""
        if not text:
            return None
        
        # Remove any markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split('\n')
            # Remove first line (```mermaid or ```)
            if lines[0].strip() in ["```", "```mermaid"]:
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        # Remove any extra whitespace
        text = text.strip()
        
        # Basic validation - should start with a diagram type
        diagram_types = ['graph', 'sequenceDiagram', 'mindmap', 'flowchart', 'gitGraph', 'erDiagram']
        starts_with_diagram = any(text.startswith(dtype) for dtype in diagram_types)
        
        if not starts_with_diagram:
            print(f"⚠️  Output doesn't start with valid diagram type: {text[:50]}...")
            return None
        
        return text
    
    def write_to_whisper_output_and_render(self, mermaid_code: str) -> None:
        """Write LLaMA output to whisper_output.txt and render HTML diagram."""
        if not mermaid_code:
            print("❌ No Mermaid code to write")
            return
        
        # Path to whisper_output.txt
        whisper_output_path = os.path.join(os.path.dirname(__file__), "..", "RenderMermaid", "whisper_output.txt")
        
        try:
            # Write raw Mermaid code to whisper_output.txt
            with open(whisper_output_path, "w") as f:
                f.write(mermaid_code)
            
            print(f"✅ Written LLaMA output to: {whisper_output_path}")
            print(f"📄 Content length: {len(mermaid_code)} characters")
            
            # Render HTML diagram
            print("🎨 Rendering HTML diagram...")
            render_mermaid_html()
            
            print("✅ HTML diagram updated!")
            print(f"🌐 Open: {os.path.join(os.path.dirname(__file__), '..', 'RenderMermaid', 'diagram.html')}")
            
        except Exception as e:
            print(f"❌ Error writing to whisper_output.txt: {e}")
    
    def process_file(self, input_file: str) -> None:
        """Process a file with multiple text descriptions."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"❌ Input file not found: {input_file}")
            return
        
        print(f"📄 Processing file: {input_file}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newlines (paragraphs)
            descriptions = [desc.strip() for desc in content.split('\n\n') if desc.strip()]
            
            print(f"📝 Found {len(descriptions)} descriptions to process")
            
            last_mermaid_code = None
            for i, description in enumerate(descriptions, 1):
                print(f"\n--- Description {i}/{len(descriptions)} ---")
                mermaid_code = self.generate_mermaid(description)
                
                if mermaid_code:
                    print("✅ Generated Mermaid diagram:")
                    print("```mermaid")
                    print(mermaid_code)
                    print("```")
                    last_mermaid_code = mermaid_code
                else:
                    print("❌ Failed to generate diagram")
                
                print("-" * 60)
            
            # Write the last successful diagram to whisper_output.txt
            if last_mermaid_code:
                print(f"\n📝 Writing last diagram to whisper_output.txt...")
                self.write_to_whisper_output_and_render(last_mermaid_code)
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    
    def interactive_mode(self) -> None:
        """Run in interactive mode for testing."""
        print("🎯 Interactive Mode - Enter text descriptions to convert to Mermaid diagrams")
        print("💡 Examples:")
        print("   • 'User logs in and accesses dashboard'")
        print("   • 'Client calls API then API queries database'")
        print("   • 'Data flows from sensors to processing unit to dashboard'")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                text = input("📝 Enter description: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    continue
                
                mermaid_code = self.generate_mermaid(text)
                
                if mermaid_code:
                    print("\n✅ Generated Mermaid diagram:")
                    print("```mermaid")
                    print(mermaid_code)
                    print("```\n")
                else:
                    print("❌ Failed to generate diagram\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert text to Mermaid diagrams using LLaMA v3.1 8B Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python llama_mermaid.py
  
  # Process a file
  python llama_mermaid.py -f test_inputs/sample_transcript.txt
  
  # Convert single text
  python llama_mermaid.py -t "User logs in and accesses dashboard"
  
  # Use custom model path
  python llama_mermaid.py -m models/my-model.gguf -t "Create a flowchart"
        """
    )
    
    parser.add_argument(
        "-m", "--model", 
        default=DEFAULT_MODEL_PATH,
        help=f"Path to LLaMA model file (default: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Input file with text descriptions"
    )
    
    parser.add_argument(
        "-t", "--text",
        help="Single text description to convert"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("\n💡 To download LLaMA v3.1 8B Instruct Q4_K_M:")
        print("   1. Visit: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        print("   2. Download: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        print(f"   3. Place in: {args.model}")
        print("\n   Or use a different model with -m flag")
        return
    
    try:
        # Initialize converter
        converter = LlamaMermaidConverter(args.model, verbose=args.verbose)
        
        if args.file:
            # Process file
            converter.process_file(args.file)
        elif args.text:
            # Process single text
            mermaid_code = converter.generate_mermaid(args.text)
            if mermaid_code:
                print("✅ Generated Mermaid diagram:")
                print("```mermaid")
                print(mermaid_code)
                print("```")
            else:
                print("❌ Failed to generate diagram")
        else:
            # Interactive mode
            converter.interactive_mode()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
