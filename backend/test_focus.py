import numpy as np
import audio_processing
import librosa

def test_voice_focus():
    sr = 16000
    duration = 5
    t = np.linspace(0, duration, sr * duration)
    
    # Create a synthetic voice segment (Loud)
    # 1s - 2s: Main Speaker
    main_speaker = np.zeros_like(t)
    main_speaker[sr*1 : sr*2] = 0.5 * np.sin(2 * np.pi * 440 * t[sr*1 : sr*2])
    
    # Create a synthetic background voice segment (Quiet)
    # 3s - 4s: Background Speaker (-20dB)
    bg_speaker = np.zeros_like(t)
    bg_speaker[sr*3 : sr*4] = 0.05 * np.sin(2 * np.pi * 550 * t[sr*3 : sr*4])
    
    y = main_speaker + bg_speaker
    
    print("Testing Voice Focus Algorithm...")
    focused_y, filtered_count = audio_processing.focus_dominant_speaker(y, sr, threshold_db=15.0)
    
    print(f"Filtered segments: {filtered_count}")
    
    # Check if bg_speaker segment is attenuated
    bg_segment_after = focused_y[sr*3+500 : sr*4-500] # Stay away from fades
    max_bg_after = np.max(np.abs(bg_segment_after))
    
    print(f"Max Amplitude in background segment before: 0.05")
    print(f"Max Amplitude in background segment after: {max_bg_after}")
    
    if max_bg_after < 0.001:
        print("SUCCESS: Background voice was successfully filtered.")
    else:
        print("FAILURE: Background voice was not filtered.")

    # Check if main_speaker segment is preserved
    main_segment_after = focused_y[sr*1+500 : sr*2-500]
    max_main_after = np.max(np.abs(main_segment_after))
    print(f"Max Amplitude in main segment after: {max_main_after}")
    
    if max_main_after > 0.4:
        print("SUCCESS: Main speaker was preserved.")
    else:
        print("FAILURE: Main speaker was attenuated.")

if __name__ == "__main__":
    test_voice_focus()
