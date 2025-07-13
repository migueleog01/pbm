#!/usr/bin/env python3
"""
LLaMA-powered Text-to-Mermaid Converter
Converts natural language descriptions into Mermaid diagrams using LLaMA v3.1 8B Instruct
Cross-platform optimized for Apple Silicon (M1/M2/M3) and Windows AMD64
"""

import argparse
import os
import sys
import time
import platform
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    print("Install with: pip install llama-cpp-python")
    sys.exit(1)

# Configuration for optimal performance
DEFAULT_MODEL_PATH = "models/llama-v3.1-8b-instruct.Q4_K_M.gguf"

# Platform-specific settings
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

if IS_WINDOWS:
    OPTIMAL_THREADS = 4        # Conservative for Windows
    CONTEXT_SIZE = 1024        # Smaller context for stability
    USE_MLOCK = False          # Disable memory locking on Windows
    USE_METAL = False          # No Metal on Windows
    N_GPU_LAYERS = 0           # CPU only for stability
    print("🪟 Windows detected - using CPU-optimized settings")
elif IS_MAC:
    OPTIMAL_THREADS = 8        # Adjust based on your Mac's performance cores
    CONTEXT_SIZE = 2048        # Sufficient for most diagram descriptions
    USE_MLOCK = True           # Enable memory locking on Mac
    USE_METAL = True           # Enable Metal backend on Mac
    N_GPU_LAYERS = -1          # Use all GPU layers on Mac
    print("🍎 macOS detected - using Metal-optimized settings")
else:
    OPTIMAL_THREADS = 6        # Default for Linux
    CONTEXT_SIZE = 2048
    USE_MLOCK = False
    USE_METAL = False
    N_GPU_LAYERS = 0
    print("🐧 Linux detected - using CPU-optimized settings")

MAX_TOKENS = 512               # Enough for complex Mermaid diagrams
TEMPERATURE = 0.3              # Lower temperature for more consistent diagram output

class LlamaMermaidConverter:
    """Converts text descriptions to Mermaid diagrams using LLaMA v3.1 8B Instruct."""
    
    def __init__(self, model_path: str, verbose: bool = False):
        """Initialize the LLaMA model with platform-optimized settings."""
        self.model_path = Path(model_path)
        self.verbose = verbose
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🧠 Loading LLaMA v3.1 8B Instruct from {model_path}")
        print(f"⚡ Platform: {platform.system()} {platform.machine()}")
        
        try:
            # Initialize LLaMA with platform-optimized settings
            llama_config = {
                "model_path": str(self.model_path),
                "n_ctx": CONTEXT_SIZE,
                "n_threads": OPTIMAL_THREADS,
                "n_gpu_layers": N_GPU_LAYERS,
                "use_mlock": USE_MLOCK,
                "verbose": self.verbose,
                "f16_kv": True,  # Use f16 for key/value cache (works on all platforms)
            }
            
            # Add Metal backend only on macOS
            if IS_MAC and USE_METAL:
                llama_config["metal"] = True
                print("🚀 Metal backend enabled")
            
            self.llm = Llama(**llama_config)
            
            if IS_WINDOWS:
                print("✅ LLaMA model loaded successfully with Windows CPU optimization!")
            elif IS_MAC:
                print("✅ LLaMA model loaded successfully with Metal acceleration!")
            else:
                print("✅ LLaMA model loaded successfully with CPU optimization!")
                
        except Exception as e:
            print(f"❌ Failed to load LLaMA model: {e}")
            print(f"💡 Try reducing n_threads to {OPTIMAL_THREADS//2} or disable f16_kv")
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
        """Generate Mermaid diagram from text description with timeout protection."""
        if not text.strip():
            return None
        
        print(f"🔄 Converting: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        prompt = self.create_prompt(text)
        
        try:
            start_time = time.time()
            
            # Windows-specific timeout protection
            if IS_WINDOWS:
                import threading
                import queue
                
                result_queue = queue.Queue()
                
                def generate_with_timeout():
                    try:
                        response = self.llm(
                            prompt,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=0.9,
                            top_k=40,
                            repeat_penalty=1.1,
                            stop=["<|eot_id|>", "<|end_of_text|>"],
                            echo=False
                        )
                        result_queue.put(('success', response))
                    except Exception as e:
                        result_queue.put(('error', str(e)))
                
                # Start generation in a separate thread
                thread = threading.Thread(target=generate_with_timeout)
                thread.daemon = True
                thread.start()
                
                # Wait for result with timeout
                thread.join(timeout=30)  # 30-second timeout
                
                if thread.is_alive():
                    print("⏰ Generation timed out after 30 seconds")
                    return None
                
                if result_queue.empty():
                    print("❌ Generation failed - no response")
                    return None
                
                result_type, result_data = result_queue.get()
                
                if result_type == 'error':
                    print(f"❌ Generation error: {result_data}")
                    return None
                
                response = result_data
                
            else:
                # Direct generation for Mac/Linux
                response = self.llm(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False
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
