# Voice-to-Mermaid Transcription System

A high-performance real-time voice transcription system optimized for creating Mermaid diagrams from speech input.

## 🎯 Optimal Configuration (M1 Pro Tested)

### Model Selection
- **Primary Model**: `base.en-q5_1` (Quantized, 57MB)
- **Alternative**: `base.en` (Full precision, 141MB)
- **Performance**: ~0.3-0.4s inference time, 714x faster than real-time

### Whisper.cpp Settings
```bash
# Optimal settings for M1 Pro (ARM64)
THREADS=8
BEAM_SIZE=3
BEST_OF=1
LANGUAGE=en
CHUNK_DURATION=3.0  # seconds
SILENCE_THRESHOLD=0.005
SAMPLE_RATE=16000
```

### Command Line Arguments
```bash
whisper-cli \
  -m models/ggml-base.en-q5_1.bin \
  -f input.wav \
  -t 8 \
  --beam-size 3 \
  --best-of 1 \
  --language en \
  --no-timestamps
```

## 🏆 Performance Benchmarks

### Model Comparison (M1 Pro)
| Model | Size | Inference Time | Accuracy | Speed Factor |
|-------|------|----------------|----------|--------------|
| `base.en-q5_1` | 57MB | 0.007s | Excellent | 714x real-time |
| `base.en` | 141MB | 0.007s | Excellent | 714x real-time |
| `tiny.en` | 37MB | 0.003s | Good | 1667x real-time |
| `small.en` | 244MB | 0.015s | Excellent | 333x real-time |
| `medium.en` | 769MB | 0.8-1.2s | Excellent | 1x real-time |
| `large-v3` | 1.5GB | 2-4s | Excellent | 0.5x real-time |

### Real-time Performance
- **Audio Processing**: 3-second chunks
- **Voice Activity Detection**: 0.005 threshold
- **Latency**: <0.5s total (capture + transcription)
- **Audio Quality**: 16kHz mono, 16-bit

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Python dependencies
pip install sounddevice numpy wave subprocess tempfile

# For macOS (M1/M2)
brew install portaudio

# For Ubuntu/Linux
sudo apt-get install portaudio19-dev

# For Windows
# Use conda or pip with pre-built wheels
```

### 2. Build whisper.cpp
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# For ARM64 (M1/M2 Mac, Snapdragon X Elite)
make -j8

# For x86_64
make -j$(nproc)
```

### 3. Download Models
```bash
cd whisper.cpp
# Download quantized base.en model (recommended)
bash models/download-ggml-model.sh base.en-q5_1

# Download full precision model (alternative)
bash models/download-ggml-model.sh base.en
```

### 4. Test Audio Setup
```bash
python scripts/test_audio.py
```

## 🎤 Usage

### Basic Real-time Transcription
```bash
python scripts/simple_realtime.py
```

### Voice-to-Mermaid Pipeline
```bash
python scripts/realtime_mermaid.py
```

## 📁 File Structure
```
pbm/
├── scripts/
│   ├── simple_realtime.py      # Core transcription engine
│   ├── realtime_mermaid.py     # Full voice-to-Mermaid pipeline
│   ├── test_audio.py           # Audio device testing
│   ├── benchmark_whisper.py    # Performance benchmarking
│   └── model_manager.py        # Model download/management
├── whisper.cpp/                # whisper.cpp repository
│   ├── build/bin/whisper-cli   # Compiled binary
│   └── models/                 # Downloaded models
└── README.md
```

## 🔧 Hardware-Specific Optimizations

### M1/M2 Mac (ARM64)
- **Threads**: 8 (matches P-cores)
- **Memory**: Unified memory advantage
- **GPU**: Not used (CPU faster for base models)

### Snapdragon X Elite (ARM64)
- **Threads**: 8-12 (depends on core count)
- **Memory**: Test with different thread counts
- **GPU**: Consider testing with GPU acceleration

### Intel/AMD x86_64
- **Threads**: CPU cores count
- **Memory**: Standard system RAM
- **GPU**: CUDA/OpenCL acceleration available

## 🎯 Mermaid Diagram Detection

### Supported Patterns
- **Flowchart**: "A goes to B", "connects to", "points to"
- **Sequence**: "A calls B", "B responds", "interaction"
- **Mindmap**: "branches", "sub-topic", "related to"

### Example Voice Commands
```
"Create a flowchart where User Authentication connects to Database Server"
"Make a sequence diagram showing Client calls API and API responds with data"
"Draw a mindmap with Project Planning as the center"
```

## 📊 Audio Quality Guidelines

### Microphone Recommendations
- **Tested**: Plantronics Blackwire 3225 Series
- **Quality**: Clear, consistent audio levels
- **Environment**: Quiet room, minimal background noise

### Audio Levels
- **Target RMS**: 0.05-0.3 (good signal strength)
- **Minimum**: 0.005 (silence threshold)
- **Maximum**: Normalize to 0.8 to prevent clipping

## 🐛 Troubleshooting

### Common Issues
1. **No transcription output**: Check microphone permissions
2. **Poor accuracy**: Speak clearly, reduce background noise
3. **High CPU usage**: Reduce thread count or use smaller model
4. **Audio device not found**: Run `python scripts/test_audio.py`

### Debug Mode
Enable debug output by uncommenting debug lines in `simple_realtime.py`:
```python
print(f"🔧 Command: {' '.join(cmd)}")
print(f"🔧 Stdout: '{result.stdout}'")
```

## 🚀 Transfer to Windows Snapdragon X Elite

### Key Considerations
1. **Architecture**: ARM64 (same as M1/M2)
2. **Build Process**: Use Windows ARM64 build tools
3. **Audio System**: Windows Audio Session API (WASAPI)
4. **Performance**: Start with same thread/beam settings, tune as needed

### Windows-Specific Setup
```bash
# Use Windows Subsystem for Linux (WSL2) or
# Native Windows build with Visual Studio
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## 📈 Performance Tuning

### For Maximum Speed
- Model: `base.en-q5_1`
- Threads: 8
- Beam size: 1
- Best of: 1

### For Maximum Accuracy
- Model: `base.en`
- Threads: 8
- Beam size: 5
- Best of: 3

### For Balanced Performance
- Model: `base.en-q5_1` (current optimal)
- Threads: 8
- Beam size: 3
- Best of: 1

## 🎵 Audio Configuration

### Sample Rate & Format
```python
SAMPLE_RATE = 16000  # Hz (whisper.cpp requirement)
CHANNELS = 1         # Mono
DTYPE = np.float32   # 32-bit float
BLOCKSIZE = 1600     # 100ms blocks (SAMPLE_RATE * 0.1)
```

### Voice Activity Detection
```python
SILENCE_THRESHOLD = 0.005  # RMS threshold
CHUNK_DURATION = 3.0       # Process every 3 seconds
MIN_AUDIO_LENGTH = 0.5     # Minimum speech duration
```

## 🔁 Next Steps: NPU Optimization

Once you've proven the flow (mic → whisper → Mermaid works great) on Snapdragon X Elite, then you can:

### Phase 1: ONNX Export
- Export Whisper model to ONNX format
- Validate ONNX model accuracy vs original
- Test ONNX runtime performance on CPU

### Phase 2: Qualcomm NPU Integration
- Use Qualcomm's QNN tools to quantize and compile the model
- Run with SNPE or QNN runtime on the Hexagon NPU
- Benchmark NPU vs CPU performance

### Phase 3: System Integration
- Integrate NPU inference into the pipeline
- Optimize audio preprocessing for NPU
- Fine-tune for maximum efficiency

### ➡️ Expected Benefits
- **Lower CPU usage**: Offload inference to dedicated NPU
- **Reduced power consumption**: NPU optimized for ML workloads
- **Potentially faster inference**: Hardware acceleration
- **Better battery life**: More efficient processing

### 🛠️ NPU Development Tools
- **Qualcomm Neural Processing SDK**: QNN tools and runtime
- **SNPE (Snapdragon Neural Processing Engine)**: Legacy runtime
- **Model conversion**: ONNX → QNN format
- **Profiling tools**: Performance analysis and optimization

## 🔄 Version History

- **v1.0**: Basic transcription working
- **v1.1**: Added quantized model support  
- **v1.2**: Optimized for M1 Pro performance
- **v1.3**: Fixed --fp16 flag issue
- **v1.4**: Added comprehensive documentation
- **v1.5**: Added NPU optimization roadmap

---

**Hardware Tested**: M1 Pro MacBook Pro  
**Target Platform**: Windows Snapdragon X Elite  
**Performance**: 714x real-time transcription speed  
**Accuracy**: Excellent for clear speech input  
**Future**: NPU optimization for enhanced efficiency 