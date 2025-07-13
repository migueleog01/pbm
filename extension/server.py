#!/usr/bin/env python3
"""
Voice Assistant Chrome Extension Server
Integrates with voice-to-mermaid LLaMA system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
import subprocess
import threading
import time
import platform
from datetime import datetime

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# Add parent directory to path to import the voice-to-mermaid system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# Windows-specific configuration
if IS_WINDOWS:
    # Disable LLaMA by default on Windows to prevent hanging
    # Set environment variable ENABLE_LLAMA_WINDOWS=1 to enable
    ENABLE_LLAMA_ON_WINDOWS = os.environ.get('ENABLE_LLAMA_WINDOWS', '0') == '1'
    if ENABLE_LLAMA_ON_WINDOWS:
        print("🪟 Windows: LLaMA enabled via ENABLE_LLAMA_WINDOWS environment variable")
    else:
        print("🪟 Windows: LLaMA disabled by default (set ENABLE_LLAMA_WINDOWS=1 to enable)")
else:
    ENABLE_LLAMA_ON_WINDOWS = True  # Always enabled on non-Windows platforms

# Global variable to track the voice-to-mermaid process
voice_process = None
pipeline_runs = []

# Global LLaMA converter instance
llama_converter = None

def initialize_llama():
    """Initialize the LLaMA converter for diagram generation"""
    global llama_converter
    
    # Skip LLaMA initialization on Windows unless explicitly enabled
    if IS_WINDOWS and not ENABLE_LLAMA_ON_WINDOWS:
        print("🪟 Skipping LLaMA initialization on Windows (use simple fallback)")
        return None
    
    if llama_converter is not None:
        return llama_converter
    
    try:
        # Import the LLaMA system - fix the path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        llama_dir = os.path.join(parent_dir, 'voice-to-mermaid-llm')
        
        if llama_dir not in sys.path:
            sys.path.insert(0, llama_dir)
        
        print(f"🔍 Looking for LLaMA module in: {llama_dir}")
        
        from llama_mermaid import LlamaMermaidConverter
        
        # Path to the LLaMA model
        model_path = os.path.join(llama_dir, 'models', 'llama-v3.1-8b-instruct.Q4_K_M.gguf')
        
        if not os.path.exists(model_path):
            print(f"❌ LLaMA model not found at: {model_path}")
            return None
        
        print(f"🤖 Initializing LLaMA converter...")
        llama_converter = LlamaMermaidConverter(model_path, verbose=True)
        print(f"✅ LLaMA converter initialized successfully!")
        
        return llama_converter
        
    except Exception as e:
        print(f"❌ Failed to initialize LLaMA converter: {str(e)}")
        if IS_WINDOWS:
            print("💡 Windows users: This is normal. Using simple fallback instead.")
        return None

def start_voice_to_mermaid():
    """Start the voice-to-mermaid pipeline"""
    global voice_process
    
    try:
        # Path to the voice-to-mermaid script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'enhanced_realtime_mermaid.py')
        
        if not os.path.exists(script_path):
            print(f"❌ Voice-to-mermaid script not found at: {script_path}")
            return False
            
        # Start the voice-to-mermaid process
        print(f"🚀 Starting voice-to-mermaid pipeline...")
        voice_process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"✅ Voice-to-mermaid pipeline started with PID: {voice_process.pid}")
        return True
        
    except Exception as e:
        print(f"❌ Error starting voice-to-mermaid pipeline: {str(e)}")
        return False

def stop_voice_to_mermaid():
    """Stop the voice-to-mermaid pipeline"""
    global voice_process
    
    if voice_process and voice_process.poll() is None:
        try:
            voice_process.terminate()
            voice_process.wait(timeout=5)
            print("✅ Voice-to-mermaid pipeline stopped")
        except subprocess.TimeoutExpired:
            voice_process.kill()
            print("⚠️ Voice-to-mermaid pipeline forcefully killed")
        except Exception as e:
            print(f"❌ Error stopping voice-to-mermaid pipeline: {str(e)}")
    voice_process = None

@app.route('/run-pipeline', methods=['POST', 'OPTIONS'])
def run_pipeline():
    """Handle pipeline trigger requests from the Chrome extension"""
    global voice_process
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        # Get the request data
        data = request.get_json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🎤 [{timestamp}] CHROME EXTENSION TRIGGERED!")
        print(f"📝 Request data: {json.dumps(data, indent=2) if data else 'No data'}")
        
        if data and data.get('trigger'):
            # Check if voice-to-mermaid is already running
            if voice_process and voice_process.poll() is None:
                print("⚠️ Voice-to-mermaid pipeline is already running")
                response = {
                    'status': 'info',
                    'message': 'Voice-to-mermaid pipeline is already running',
                    'timestamp': timestamp,
                    'pid': voice_process.pid
                }
            else:
                # Start the voice-to-mermaid pipeline
                if start_voice_to_mermaid():
                    # Store the run info
                    run_info = {
                        'timestamp': timestamp,
                        'trigger': data.get('trigger'),
                        'status': 'success',
                        'pid': voice_process.pid if voice_process else None
                    }
                    pipeline_runs.append(run_info)
                    
                    print(f"✅ Voice-to-mermaid pipeline triggered successfully!")
                    print(f"📊 Total pipeline runs: {len(pipeline_runs)}")
                    print(f"🎯 Now listening for voice commands...")
                    
                    response = {
                        'status': 'success',
                        'message': 'Voice-to-mermaid pipeline started successfully',
                        'timestamp': timestamp,
                        'run_count': len(pipeline_runs),
                        'pid': voice_process.pid,
                        'instructions': 'Speak into your microphone to generate Mermaid diagrams'
                    }
                else:
                    response = {
                        'status': 'error',
                        'message': 'Failed to start voice-to-mermaid pipeline'
                    }
                    
            return jsonify(response), 200
            
        else:
            print("❌ Invalid trigger data")
            return jsonify({
                'status': 'error',
                'message': 'Invalid trigger data'
            }), 400
            
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/stop-pipeline', methods=['POST', 'OPTIONS'])
def stop_pipeline():
    """Stop the voice-to-mermaid pipeline"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        stop_voice_to_mermaid()
        return jsonify({
            'status': 'success',
            'message': 'Voice-to-mermaid pipeline stopped'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error stopping pipeline: {str(e)}'
        }), 500

@app.route('/generate-diagram', methods=['POST', 'OPTIONS'])
def generate_diagram():
    """Generate Mermaid diagram from text description using LLaMA"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get the request data
        data = request.get_json()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🎨 [{timestamp}] DIAGRAM GENERATION REQUEST!")
        print(f"📝 Request data: {json.dumps(data, indent=2) if data else 'No data'}")
        
        if not data or not data.get('description'):
            return jsonify({
                'status': 'error',
                'message': 'No description provided'
            }), 400
            
        description = data.get('description')
        source = data.get('source', 'unknown')
        
        print(f"🎨 Description: {description}")
        print(f"🎨 Source: {source}")
        
        # Try to use LLaMA for intelligent diagram generation
        converter = initialize_llama()
        
        if converter:
            print(f"🤖 Generating diagram with LLaMA...")
            mermaid_code = converter.generate_mermaid(description)
            
            if mermaid_code:
                print(f"✅ Diagram generated successfully!")
                print(f"📊 Mermaid code preview: {mermaid_code[:100]}...")
                
                response = {
                    'status': 'success',
                    'message': 'Diagram generated with LLaMA v3.1 8B Instruct',
                    'timestamp': timestamp,
                    'description': description,
                    'mermaid': mermaid_code,
                    'source': source,
                    'ai_generated': True
                }
                
                return jsonify(response), 200
            else:
                print(f"❌ LLaMA failed to generate diagram, falling back to simple generation")
        else:
            print(f"❌ LLaMA system not available, using simple generation")
        
        # Fallback to simple text processing
        mermaid_code = generate_simple_diagram(description)
        
        response = {
            'status': 'success',
            'message': 'Diagram generated with simple text processing (LLaMA unavailable)',
            'timestamp': timestamp,
            'description': description,
            'mermaid': mermaid_code,
            'source': source,
            'fallback': True
        }
        
        return jsonify(response), 200
            
    except Exception as e:
        print(f"❌ Error generating diagram: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

def generate_simple_diagram(description):
    """Enhanced fallback diagram generation without LLaMA"""
    description_lower = description.lower()
    
    # Enhanced keyword-based diagram generation
    if any(word in description_lower for word in ['login', 'user', 'authentication', 'password', 'signin']):
        return """graph TD
    A[User] --> B[Login Form]
    B --> C[Validate Credentials]
    C --> D[Access Granted]
    C --> E[Access Denied]
    D --> F[Dashboard]
    E --> B"""
    
    elif any(word in description_lower for word in ['payment', 'money', 'transaction', 'pay', 'checkout']):
        return """graph TD
    A[User] --> B[Shopping Cart]
    B --> C[Payment Gateway]
    C --> D[Process Payment]
    D --> E[Payment Success]
    D --> F[Payment Failed]
    E --> G[Order Confirmation]
    F --> C"""
    
    elif any(word in description_lower for word in ['api', 'request', 'response', 'server', 'endpoint']):
        return """graph TD
    A[Client] --> B[API Request]
    B --> C[Server Processing]
    C --> D[Database Query]
    D --> E[API Response]
    E --> A"""
    
    elif any(word in description_lower for word in ['database', 'data', 'query', 'sql']):
        return """graph TD
    A[Application] --> B[Database Query]
    B --> C[Execute Query]
    C --> D[Return Results]
    D --> A"""
    
    elif any(word in description_lower for word in ['upload', 'download', 'file', 'document']):
        return """graph TD
    A[User] --> B[Select File]
    B --> C[Upload File]
    C --> D[Process File]
    D --> E[Save to Storage]
    E --> F[Confirm Success]"""
    
    elif any(word in description_lower for word in ['register', 'signup', 'account', 'create']):
        return """graph TD
    A[User] --> B[Registration Form]
    B --> C[Validate Input]
    C --> D[Create Account]
    C --> E[Show Errors]
    D --> F[Send Confirmation]
    E --> B"""
    
    # Try to extract process steps from description
    elif any(word in description_lower for word in ['workflow', 'process', 'step', 'flow', 'then', 'next']):
        words = description.split()
        steps = []
        
        # Look for common process words
        process_words = ['start', 'begin', 'process', 'check', 'validate', 'send', 'receive', 'end', 'finish', 'complete']
        for word in words:
            clean_word = word.strip('.,!?;:').lower()
            if clean_word in process_words:
                steps.append(word.title())
        
        if len(steps) >= 2:
            diagram = "graph TD\n"
            for i, step in enumerate(steps[:5]):  # Limit to 5 steps
                letter = chr(65 + i)  # A, B, C, etc.
                diagram += f"    {letter}[{step}]"
                if i < len(steps) - 1:
                    next_letter = chr(65 + i + 1)
                    diagram += f" --> {next_letter}"
                diagram += "\n"
            return diagram.strip()
    
    # Generic flowchart based on description analysis
    words = [word.strip('.,!?;:') for word in description.split()]
    
    if len(words) >= 3:
        # Try to create a meaningful flow from the words
        start_word = words[0].title()
        
        # Find connecting words that suggest flow
        connecting_words = ['to', 'then', 'and', 'with', 'through', 'via', 'using']
        flow_parts = []
        current_part = [start_word]
        
        for word in words[1:]:
            if word.lower() in connecting_words:
                if current_part:
                    flow_parts.append(' '.join(current_part))
                    current_part = []
            else:
                current_part.append(word.title())
        
        if current_part:
            flow_parts.append(' '.join(current_part))
        
        if len(flow_parts) >= 2:
            diagram = "graph TD\n"
            for i, part in enumerate(flow_parts[:5]):  # Limit to 5 parts
                letter = chr(65 + i)
                diagram += f"    {letter}[{part}]"
                if i < len(flow_parts) - 1:
                    next_letter = chr(65 + i + 1)
                    diagram += f" --> {next_letter}"
                diagram += "\n"
            return diagram.strip()
    
    # Final fallback - simple generic diagram
    return """graph TD
    A[Start] --> B[Process]
    B --> C[Decision]
    C --> D[Success]
    C --> E[Retry]
    E --> B
    D --> F[End]"""

@app.route('/status', methods=['GET'])
def get_status():
    """Get server status and pipeline run history"""
    global voice_process
    
    is_running = voice_process and voice_process.poll() is None
    
    return jsonify({
        'status': 'running',
        'message': 'Voice Assistant server is running',
        'voice_pipeline_running': is_running,
        'voice_pipeline_pid': voice_process.pid if is_running else None,
        'total_runs': len(pipeline_runs),
        'recent_runs': pipeline_runs[-5:] if pipeline_runs else []
    })

@app.route('/', methods=['GET'])
def home():
    """Enhanced home page with voice-to-mermaid integration"""
    global voice_process
    
    total_runs = len(pipeline_runs)
    is_running = voice_process and voice_process.poll() is None
    status_color = "#e8f5e8" if is_running else "#f0f0f0"
    status_text = "🟢 Running" if is_running else "🔴 Stopped"
    
    return f"""
    <html>
    <head>
        <title>Voice-to-Mermaid Assistant Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #667eea; }}
            .status {{ background: {status_color}; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .endpoint {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .pipeline-status {{ background: #e8f4ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            code {{ background: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
            .button {{ background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice-to-Mermaid Assistant Server</h1>
            <div class="status">
                <strong>✅ Server Status:</strong> Running on http://localhost:5000
            </div>
            <div class="pipeline-status">
                <strong>Voice-to-Mermaid Pipeline:</strong> {status_text}
                {f'<br><strong>PID:</strong> {voice_process.pid}' if is_running else ''}
            </div>
            
            <h2>🚀 Features:</h2>
            <ul>
                <li>🎙️ <strong>Voice Recognition:</strong> Whisper.cpp for real-time speech-to-text</li>
                <li>🤖 <strong>LLaMA Integration:</strong> v3.1 8B Instruct for intelligent diagram generation</li>
                <li>📊 <strong>Mermaid Diagrams:</strong> Automatic flowchart generation from speech</li>
                <li>🌐 <strong>Chrome Extension:</strong> Voice-triggered activation</li>
                <li>⚡ <strong>Real-time Processing:</strong> ~1.5-2.5s end-to-end latency</li>
            </ul>
            
            <h2>📡 Available Endpoints:</h2>
            <div class="endpoint">
                <strong>POST /run-pipeline</strong><br>
                Starts the voice-to-mermaid pipeline (triggered by Chrome extension)
            </div>
            <div class="endpoint">
                <strong>POST /stop-pipeline</strong><br>
                Stops the voice-to-mermaid pipeline
            </div>
            <div class="endpoint">
                <strong>GET /status</strong><br>
                Returns server status and pipeline information
            </div>
            
            <p><strong>Total pipeline runs:</strong> {total_runs}</p>
            <p><strong>Ready to receive voice commands!</strong> 🎯</p>
            
            <h2>🎯 How to Use:</h2>
            <ol>
                <li>Install the Chrome extension</li>
                <li>Click the extension icon or say "hey computer"</li>
                <li>Pipeline will start automatically</li>
                <li>Speak your diagram description</li>
                <li>Watch as Mermaid diagrams are generated in real-time!</li>
            </ol>
        </div>
    </body>
    </html>
    """

def cleanup_on_exit():
    """Clean up processes when server exits"""
    stop_voice_to_mermaid()

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)
    
    print("🚀 Starting Voice-to-Mermaid Assistant Server...")
    print("📡 Server will run on: http://localhost:5000")
    print("🎤 Ready to receive voice commands from Chrome extension!")
    print("🤖 LLaMA v3.1 8B Instruct integration enabled")
    print("👋 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    app.run(
        host='localhost',
        port=5000,
        debug=True,
        threaded=True
    ) 