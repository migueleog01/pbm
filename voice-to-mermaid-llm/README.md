# 🧠 LLaMA Integration for Voice-to-Mermaid

**LLaMA v3.1 8B Instruct powered diagram generation for the Voice-to-Mermaid pipeline**

This directory contains the LLaMA integration that powers intelligent Mermaid diagram generation from natural language voice input.

## 🎯 **What This Does**

Instead of simple text parsing, this system uses **LLaMA v3.1 8B Instruct** to:
- Understand natural language context
- Generate appropriate diagram types (flowcharts, sequence diagrams, etc.)
- Create clean, professional Mermaid syntax
- Handle complex multi-node workflows intelligently

## 📋 **Files Overview**

```
voice-to-mermaid-llm/
├── llama_mermaid.py              # Main LLaMA converter class
├── models/
│   ├── download_model.py         # Automatic model download script
│   ├── DOWNLOAD_MODEL.md         # Manual download instructions
│   └── *.gguf                    # Model files (gitignored)
├── test_inputs/
│   ├── sample_transcript.txt     # Test input examples
│   └── test_commands.txt         # Voice command examples
├── requirements.txt              # LLaMA-specific dependencies
└── README.md                     # This file
```

## 🚀 **Quick Start**

### **1. Download Model**
```bash
cd models/
python download_model.py
```

### **2. Test Standalone**
```bash
# Test with single input
python llama_mermaid.py -t "user logs in and accesses dashboard"

# Test with file input
python llama_mermaid.py -f test_inputs/sample_transcript.txt

# Interactive mode
python llama_mermaid.py
```

### **3. Integration Usage**
The LLaMA system is automatically used by the main voice pipeline (`../enhanced_realtime_mermaid.py`) when:
- Model file exists
- LLaMA dependencies are installed
- System has sufficient memory (6GB+)

## 🔧 **Technical Details**

### **Model Configuration**
- **Model**: LLaMA v3.1 8B Instruct
- **Quantization**: Q4_K_M (4-bit mixed precision)
- **Size**: 4.6GB
- **Context Window**: 2048 tokens
- **Output Limit**: 512 tokens (sufficient for complex diagrams)

### **Optimization Settings**
```python
# Apple Silicon (M1/M2/M3)
OPTIMAL_THREADS = 8
CONTEXT_SIZE = 2048
TEMPERATURE = 0.3
METAL_ACCELERATION = True

# Windows ARM64 (Snapdragon X Elite)
OPTIMAL_THREADS = 12
CONTEXT_SIZE = 2048
TEMPERATURE = 0.3
METAL_ACCELERATION = False
```

### **Performance**
| Platform | Loading Time | Generation Time | Memory Usage |
|----------|--------------|-----------------|--------------|
| M1 Pro | 3-5s | 1-2s | 6GB |
| M2 Max | 2-4s | 1-2s | 6GB |
| Snapdragon X Elite | 5-7s | 2-3s | 6GB |
| Intel i7-12700K | 6-8s | 3-4s | 6GB |

## 🎨 **Input/Output Examples**

### **Simple Chain**
```
Input: "user to database"
Output:
graph TD
    User[User] --> Database[Database]
```

### **Complex Workflow**
```
Input: "user authentication with database validation and error handling"
Output:
graph TD
    User[User] --> Auth[Authentication]
    Auth --> Database[Database]
    Database --> Validate[Validation]
    Validate --> Error[Error Handling]
    Error --> User
```

### **Sequence Diagram**
```
Input: "client calls API then API queries database"
Output:
sequenceDiagram
    Client->>API: Request
    API->>Database: Query
    Database-->>API: Response
    API-->>Client: Response
```

## 🔄 **Integration Architecture**

### **Voice Pipeline Integration**
```python
# In enhanced_realtime_mermaid.py
if llama_converter:
    # Try LLaMA first
    result = llama_converter.generate_mermaid(text)
    if result:
        return result
    
# Fallback to simple processing
return simple_chain_detection(text)
```

### **Hybrid Approach**
1. **Primary**: LLaMA generates intelligent diagrams
2. **Fallback**: Simple text parsing for basic chains
3. **Error Handling**: Graceful degradation on failure

## 🛠️ **Model Management**

### **Download Script Usage**
```bash
# Download default Q4_K_M model
python download_model.py

# Show available models
python download_model.py --info

# Download alternative model
python download_model.py --model Q5_K_M
```

### **Manual Download**
If automatic download fails:
1. Visit: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
2. Download: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
3. Rename to: `llama-v3.1-8b-instruct.Q4_K_M.gguf`
4. Place in: `models/` directory

### **Model Variants**
| Model | Size | Quality | Speed | Memory |
|-------|------|---------|-------|--------|
| Q4_K_M | 4.6GB | Good | Fast | 6GB |
| Q4_K_S | 4.4GB | Fair | Very Fast | 6GB |
| Q5_K_M | 5.8GB | Better | Medium | 8GB |
| Q8_0 | 8.5GB | Best | Slow | 12GB |

## 🧪 **Testing**

### **Unit Tests**
```bash
# Test model loading
python -c "from llama_mermaid import LlamaMermaidConverter; print('✅ Import successful')"

# Test generation
python llama_mermaid.py -t "test input"
```

### **Integration Tests**
```bash
# Test full pipeline
cd ..
python enhanced_realtime_mermaid.py
# Say: "Computer, test user to database"
```

### **Performance Tests**
```bash
# Benchmark generation speed
python -c "
from llama_mermaid import LlamaMermaidConverter
import time

converter = LlamaMermaidConverter('models/llama-v3.1-8b-instruct.Q4_K_M.gguf')
start = time.time()
result = converter.generate_mermaid('user to database')
end = time.time()
print(f'Generation time: {end-start:.2f}s')
"
```

## 🔧 **Configuration Options**

### **Model Parameters**
```python
# In llama_mermaid.py
DEFAULT_MODEL_PATH = "models/llama-v3.1-8b-instruct.Q4_K_M.gguf"
OPTIMAL_THREADS = 8
CONTEXT_SIZE = 2048
MAX_TOKENS = 512
TEMPERATURE = 0.3
```

### **Platform Optimization**
```python
# Apple Silicon
metal=True
n_gpu_layers=-1
use_mlock=True
f16_kv=True

# Windows ARM64
metal=False
n_gpu_layers=0
use_mlock=False
f16_kv=False

# CPU Optimization
n_threads=8  # Adjust based on cores
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Model Loading Fails**
```bash
# Check model file
ls -la models/llama-v3.1-8b-instruct.Q4_K_M.gguf

# Re-download model
python models/download_model.py
```

#### **2. Out of Memory**
```python
# Use smaller model
python models/download_model.py --model Q4_K_S

# Or reduce context size
CONTEXT_SIZE = 1024
```

#### **3. Slow Generation**
```python
# Increase thread count
OPTIMAL_THREADS = 12

# Or use faster model
python models/download_model.py --model Q4_K_S
```

#### **4. Metal Acceleration Issues (macOS)**
```bash
# Reinstall with Metal support
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### **Debug Mode**
```python
# Enable verbose output
converter = LlamaMermaidConverter(model_path, verbose=True)
```

## 📊 **Performance Optimization**

### **Memory Management**
- **Model Loading**: ~5GB RAM
- **Generation**: ~1GB additional
- **Optimization**: Use `use_mlock=True` for consistency

### **CPU Optimization**
- **Threads**: Match CPU performance cores
- **M1/M2**: 8 threads optimal
- **Snapdragon X Elite**: 12 threads recommended

### **Speed vs Quality**
```python
# For speed
TEMPERATURE = 0.1
MAX_TOKENS = 256
BEAM_SIZE = 1

# For quality
TEMPERATURE = 0.3
MAX_TOKENS = 512
BEAM_SIZE = 3
```

## 🔄 **Development**

### **Adding New Prompt Templates**
```python
# Edit create_prompt() method in llama_mermaid.py
def create_prompt(self, text: str) -> str:
    system_prompt = """Your custom system prompt here..."""
    # Format prompt for LLaMA v3.1 Instruct
    return formatted_prompt
```

### **Custom Model Support**
```python
# Initialize with custom model
converter = LlamaMermaidConverter("/path/to/custom/model.gguf")
```

### **Output Format Customization**
```python
# Edit clean_mermaid_output() method
def clean_mermaid_output(self, text: str) -> Optional[str]:
    # Add custom cleaning logic
    return cleaned_text
```

## 📚 **Resources**

### **Documentation**
- [LLaMA v3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
- [Mermaid.js Documentation](https://mermaid.js.org/intro/)

### **Model Sources**
- [Primary Model](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [Alternative Quantizations](https://huggingface.co/models?search=llama-3.1-8b-instruct-gguf)

### **Performance Benchmarks**
- [llama.cpp Performance](https://github.com/ggerganov/llama.cpp#performance)
- [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)

## 🤝 **Contributing**

### **Code Style**
```bash
# Format code
black llama_mermaid.py

# Check style
flake8 llama_mermaid.py
```

### **Testing**
```bash
# Run tests
python -m pytest tests/

# Add new test
def test_new_feature():
    converter = LlamaMermaidConverter("models/test-model.gguf")
    result = converter.generate_mermaid("test input")
    assert result is not None
```

### **Pull Requests**
1. Fork the repository
2. Create feature branch
3. Test on your platform
4. Submit pull request with:
   - Clear description
   - Test results
   - Performance impact

---

**🎯 Powering intelligent voice-to-diagram generation with LLaMA!** 