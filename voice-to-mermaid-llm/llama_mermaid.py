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

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    print("Install with: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
    sys.exit(1)

# Configuration for optimal M1/M2/M3 performance
DEFAULT_MODEL_PATH = "models/llama-v3.1-8b-instruct.Q4_K_M.gguf"
OPTIMAL_THREADS = 8  # Adjust based on your Mac's performance cores
CONTEXT_SIZE = 2048  # Sufficient for most diagram descriptions
MAX_TOKENS = 512     # Enough for complex Mermaid diagrams
TEMPERATURE = 0.3    # Lower temperature for more consistent diagram output

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
            # Initialize LLaMA with Metal acceleration and optimal settings
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=CONTEXT_SIZE,           # Context window
                n_threads=OPTIMAL_THREADS,    # CPU threads (adjust for your Mac)
                n_gpu_layers=-1,              # Use all GPU layers (Metal)
                use_mlock=True,               # Lock model in memory
                verbose=self.verbose,
                # Metal-specific optimizations
                metal=True,                   # Enable Metal backend
                f16_kv=True,                  # Use f16 for key/value cache
            )
            print("✅ LLaMA model loaded successfully with Metal acceleration!")
            
        except Exception as e:
            print(f"❌ Failed to load LLaMA model: {e}")
            raise
    
    def create_prompt(self, text: str) -> str:
        """Create a well-structured prompt for LLaMA v3.1 8B Instruct."""
        
        # LLaMA v3.1 Instruct format with system instruction
        system_prompt = """You are a diagram assistant specialized in converting natural language descriptions into valid Mermaid.js diagrams.

Rules:
1. Output ONLY valid Mermaid syntax - no explanations, no code blocks, no additional text
2. Choose the most appropriate diagram type (graph, sequenceDiagram, mindmap, etc.)
3. Use clear, descriptive node names
4. For flowcharts, use 'graph TD' (top-down) or 'graph LR' (left-right)
5. For sequence diagrams, use 'sequenceDiagram' format
6. For mindmaps, use 'mindmap' format
7. Keep diagrams simple and readable

Examples:
Input: "User logs in and accesses dashboard"
Output: graph TD
    User[User] --> Login[Login]
    Login --> Dashboard[Dashboard]

Input: "Client calls API then API queries database"
Output: sequenceDiagram
    Client->>API: Request
    API->>Database: Query
    Database-->>API: Response
    API-->>Client: Response"""

        # Format the prompt for LLaMA v3.1 Instruct
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert this description into a Mermaid diagram:
{text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
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
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|eot_id|>", "<|end_of_text|>"],  # Stop tokens for LLaMA v3.1
                echo=False  # Don't echo the prompt
            )
            
            end_time = time.time()
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean up the output
            mermaid_code = self.clean_mermaid_output(generated_text)
            
            if mermaid_code:
                print(f"⚡ Generated in {end_time - start_time:.2f}s")
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
            
            for i, description in enumerate(descriptions, 1):
                print(f"\n--- Description {i}/{len(descriptions)} ---")
                mermaid_code = self.generate_mermaid(description)
                
                if mermaid_code:
                    print("✅ Generated Mermaid diagram:")
                    print("```mermaid")
                    print(mermaid_code)
                    print("```")
                else:
                    print("❌ Failed to generate diagram")
                
                print("-" * 60)
            
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
