import os
import sys
from pydub import AudioSegment

# Simulate the logic in audio_processing.py
base_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(base_dir, "ffmpeg", "bin", "ffmpeg.exe")
ffprobe_path = os.path.join(base_dir, "ffmpeg", "bin", "ffprobe.exe")

print(f"Base Dir: {base_dir}")
print(f"Expected FFMPEG Path: {ffmpeg_path}")

if os.path.exists(ffmpeg_path):
    print("SUCCESS: FFMPEG binary found at path.")
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
    try:
        # Create a silent audio segment to test converter availability
        silence = AudioSegment.silent(duration=1000)
        print("SUCCESS: AudioSegment created (pydub loaded).")
        # Attempt export to check if converter actually works (requires write access)
        silence.export("test_silence.wav", format="wav")
        print("SUCCESS: Exported wav using ffmpeg.")
        os.remove("test_silence.wav")
    except Exception as e:
        print(f"FAILURE: Pydub operation failed: {e}")
else:
    print("FAILURE: FFMPEG binary NOT found at path.")
