# Real-Time Voice-to-Mermaid Pipeline

A minimal, fully-local pipeline that captures microphone audio, processes it through whisper.cpp on CPU, and converts simple diagram commands into Mermaid code blocks.

## Prerequisites

- **Python**: ≥3.10
- **Git**: For cloning with submodules
- **CMake**: For building whisper.cpp
- **Build Tools**: 
  - **Windows**: Visual Studio Build Tools 2019+ or Visual Studio Community
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Linux**: GCC/G++ compiler suite

## Installation & Setup

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/your-username/pbm.git
cd pbm
```

### 2. Build whisper.cpp

```bash
cd whisper.cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ../..
```

### 3. Download the Whisper Model

```bash
cd whisper.cpp
python models/download-ggml-model.py tiny.en
cd ..
```

This downloads the `ggml-tiny.en.bin` model file to `whisper.cpp/models/`.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Real-Time Voice Processing

```bash
python scripts/realtime_mermaid.py
```

The script will:
- Start capturing audio from your default microphone
- Process speech in real-time using whisper.cpp
- Listen for diagram commands like "draw diagram A to B"
- Output Mermaid code blocks when commands are detected
- Print regular transcript for other speech

### Voice Commands

Say commands like:
- "draw diagram start to end"
- "create diagram login to dashboard"
- "make diagram user to server"

These will generate Mermaid code blocks like:
```mermaid
graph TD
    start --> end
```

### Test with WAV File (Optional)

If microphone is unavailable, you can test with a WAV file:

```bash
# First, record a test file or use an existing one
python scripts/realtime_mermaid.py --input test.wav
```

### Exit

Press `Ctrl+C` to gracefully exit the application.

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── whisper.cpp/             # Git submodule (whisper.cpp repository)
├── scripts/
│   └── realtime_mermaid.py  # Main voice processing script
└── models/                  # Directory for model files
```

## Platform Support

- **Primary Target**: Snapdragon X Elite (Windows-on-ARM64)
- **Tested On**: Windows, macOS, Linux
- **Processing**: CPU-only (NPU optimization planned for Phase 2)

## Troubleshooting

### Build Issues

**Windows**: Ensure Visual Studio Build Tools are installed with C++ development tools.

**macOS**: Install Xcode Command Line Tools if CMake fails.

**Linux**: Install build essentials: `sudo apt-get install build-essential cmake`

### Audio Issues

- Check microphone permissions
- Verify default audio device in system settings
- On Linux, ensure ALSA/PulseAudio is configured

### Python Dependencies

If whispercpp installation fails, ensure whisper.cpp is built first, then retry:
```bash
pip install --force-reinstall whispercpp
```

## TODO - Future Phases

### Phase 2: Live Rendering
- [ ] Web interface for real-time Mermaid rendering
- [ ] WebSocket connection for live updates
- [ ] Multiple diagram type support (flowchart, sequence, class)
- [ ] Voice command expansion (colors, styles, layouts)

### Phase 3: NPU Optimization
- [ ] Snapdragon X Elite NPU integration
- [ ] ONNX model conversion for hardware acceleration
- [ ] Performance benchmarking CPU vs NPU
- [ ] Power consumption optimization

### Phase 4: Enhanced Features
- [ ] Multi-language support beyond English
- [ ] Complex diagram commands (conditional flows, loops)
- [ ] Diagram editing via voice ("change A to B", "remove node C")
- [ ] Export to various formats (PNG, SVG, PDF)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on your target platform
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 