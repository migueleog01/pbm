#!/usr/bin/env python3
"""
Simple audio input test to diagnose microphone issues
"""

import sounddevice as sd
import numpy as np
import time

def test_audio_input():
    """Test if microphone input is working."""
    print("🎤 Testing microphone input...")
    print("Speak now - you should see audio levels appear:")
    print("Press Ctrl+C to stop")
    
    sample_rate = 16000
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(indata ** 2))
        
        # Show audio level with bars
        level_bars = int(rms * 50)  # Scale for display
        bar_display = "█" * level_bars + "░" * (20 - level_bars)
        
        if rms > 0.001:  # Only show if there's actual audio
            print(f"\r🔊 {rms:.4f} |{bar_display}|", end="", flush=True)
    
    try:
        # List available audio devices
        print("\n📋 Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  {i}: {device['name']}{default_marker}")
        
        print(f"\n🎧 Using default input device: {sd.query_devices(sd.default.device[0])['name']}")
        print("🔊 Speak now...")
        
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1),  # 100ms blocks
            dtype=np.float32
        ):
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n✅ Audio test stopped")
    except Exception as e:
        print(f"\n❌ Audio error: {e}")
        print("\n🔧 Possible solutions:")
        print("  1. Check microphone permissions in System Preferences > Security & Privacy > Microphone")
        print("  2. Make sure your microphone is not being used by another app")
        print("  3. Try running: sudo python scripts/test_audio.py")

if __name__ == "__main__":
    test_audio_input() 