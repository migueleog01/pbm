# Voice-to-Mermaid: Key Settings Summary

## 🎯 What We Discovered

### The Problem We Solved
- **Issue**: `--fp16` flag not supported by whisper-cli
- **Solution**: Remove `--fp16` flag from command arguments
- **Result**: Perfect transcription with ~0.3s inference time

### Optimal Model Choice
- **Winner**: `base.en-q5_1` (57MB quantized)
- **Performance**: 714x faster than real-time
- **Quality**: Excellent accuracy for clear speech

### Perfect Settings for M1 Pro
```python
THREADS = 8
BEAM_SIZE = 3
BEST_OF = 1
CHUNK_DURATION = 3.0
SILENCE_THRESHOLD = 0.005
```

## 🚀 For Windows Snapdragon X Elite

### Quick Start Command
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make -j8  # or cmake build on Windows
bash models/download-ggml-model.sh base.en-q5_1
```

### Copy These Files
1. `scripts/simple_realtime.py` - Core engine
2. `scripts/test_audio.py` - Audio testing
3. `README.md` - Full documentation
4. `optimal_config.md` - Detailed settings

### Key Changes for Windows
- Change `/` to `\` in file paths
- Add `.exe` to `whisper-cli` executable
- Test with 8-12 threads (X Elite has more cores)

## 📊 Performance Targets

### M1 Pro Baseline
- **Speed**: 0.32-0.41s per 3-second chunk
- **Accuracy**: Excellent for clear speech
- **CPU**: ~50% utilization with 8 threads

### Snapdragon X Elite Goals
- **Speed**: <0.5s per chunk
- **Accuracy**: Similar to M1 Pro
- **CPU**: <70% utilization
- **Memory**: <300MB

## 🎤 Audio Setup

### Hardware Tested
- **Microphone**: Plantronics Blackwire 3225 Series
- **Environment**: Quiet room
- **Levels**: 0.05-0.3 RMS optimal

### Software Config
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Processing**: 3-second chunks
- **Silence**: 0.005 RMS threshold

## 🔧 Command That Works
```bash
whisper-cli -m models/ggml-base.en-q5_1.bin -f audio.wav -t 8 --beam-size 3 --best-of 1 --language en --no-timestamps
```

## 📋 Migration Checklist

### Before Transfer
- [x] Document all settings
- [x] Test on M1 Pro
- [x] Verify optimal performance
- [x] Create setup guides

### After Transfer to Windows
- [ ] Build whisper.cpp for ARM64 Windows
- [ ] Download base.en-q5_1 model
- [ ] Test audio device detection
- [ ] Verify transcription accuracy
- [ ] Tune thread count for X Elite cores
- [ ] Benchmark performance vs M1 Pro

### NPU Optimization (Future)
- [ ] Export Whisper model to ONNX format
- [ ] Install Qualcomm Neural Processing SDK
- [ ] Convert ONNX to QNN format for Hexagon NPU
- [ ] Integrate NPU runtime into pipeline
- [ ] Benchmark NPU vs CPU performance
- [ ] Optimize for power efficiency

## 🏆 Success Criteria

### Must Work
- Transcription speed <0.5s per chunk
- Clear speech accuracy >95%
- Real-time processing without lag
- Audio levels properly detected

### Should Work
- Background noise filtering
- Technical terms captured
- Multiple speakers distinguished
- Diagram commands detected

## 🔁 NPU Future Enhancement

Once CPU version works perfectly on Snapdragon X Elite:

### ➡️ Expected NPU Benefits
- **CPU Usage**: 50% → 20% reduction
- **Power**: 30-50% less consumption  
- **Speed**: Potentially <0.2s per chunk
- **Battery**: Extended runtime
- **Temperature**: Cooler operation

### 🛠️ NPU Tools Required
- Qualcomm Neural Processing SDK
- QNN tools for model conversion
- SNPE or QNN runtime
- ONNX export capabilities

---

**Status**: ✅ Optimized and documented  
**Ready for**: Windows Snapdragon X Elite migration  
**Next**: Test on target hardware → NPU optimization  
**Future**: Hexagon NPU acceleration for enhanced efficiency 