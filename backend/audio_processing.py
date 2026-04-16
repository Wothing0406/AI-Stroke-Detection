import os
import sys
import logging

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import librosa
import numpy as np
import soundfile as sf

# Constants for Scientific Research
SAMPLE_RATE = 16000  
DURATION = 5         
FRAME_MS = 25        # 25ms clinical standard
HOP_MS = 10          # 10ms hop length
FRAME_LENGTH = int(SAMPLE_RATE * FRAME_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * HOP_MS / 1000)

def reduce_noise_spectral_subtraction(y, sr):
    """
    Advanced Scientific Noise Reduction: Spectral Subtraction.
    Estimates the noise floor and subtracts it from the signal.
    """
    if y is None or len(y) < 512: return y
    
    # 1. Estimation of Noise (Use first 300ms, much safer)
    noise_sample_len = int(0.3 * sr)
    if len(y) > noise_sample_len:
        noise_est = y[:noise_sample_len]
    else:
        noise_est = y
        
    noise_mag = np.mean(np.abs(librosa.stft(noise_est)), axis=1)
    
    # 2. Extract Signal Magnitude
    stft_y = librosa.stft(y)
    mag_y, phase_y = np.abs(stft_y), np.angle(stft_y)
    
    # 3. Subtraction with soft floor (0.1 instead of 0.02)
    noise_mag_expanded = noise_mag[:, np.newaxis]
    # Reduce over-subtraction from 2.0 to 1.5 to preserve signal
    mag_denoised = np.maximum(mag_y - 1.5 * noise_mag_expanded, 0.1 * mag_y)
    
    # 4. Reconstruct
    stft_denoised = mag_denoised * np.exp(1j * phase_y)
    return librosa.istft(stft_denoised)

def pre_emphasis(y, coeff=0.97):
    """
    Apply pre-emphasis filter to accentuate high frequencies.
    Essential for spectral clarity in Jitter/Shimmer analysis.
    Formula: y(t) = x(t) - coeff * x(t-1)
    """
    if y is None or len(y) < 2: return y
    return np.append(y[0], y[1:] - coeff * y[:-1])

def apply_hamming_window(y):
    """
    Apply Hamming window to the signal to reduce spectral leakage.
    Used before STFT or LPC analysis.
    """
    if y is None or len(y) == 0: return y
    window = np.hamming(len(y))
    return y * window

def load_audio(file_path):
    """
    Load an audio file, resample to 16kHz, and convert to mono.
    Robustly handles formats using pydub/ffmpeg if librosa fails.
    """
    if not file_path:
        logger.error("DEBUG: file_path is empty!")
        return None, None
         
    abs_path = os.path.abspath(file_path)
    logger.info(f"DEBUG: Attempting to load audio from: '{abs_path}'")

    if not os.path.exists(abs_path):
        logger.error(f"DEBUG: File DOES NOT EXIST at: {abs_path}")
        return None, None
        
    if os.path.getsize(abs_path) < 100:
        logger.error(f"File too small ({os.path.getsize(abs_path)} bytes): {abs_path}")
        return None, None

    # Step 1: Try pydub/ffmpeg FIRST (most robust for WebM/Opus)
    try:
        from pydub import AudioSegment
        import io
        

        audio = AudioSegment.from_file(abs_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        y, sr = librosa.load(wav_buffer, sr=SAMPLE_RATE, mono=True)
        return y, sr
    except Exception as e:
        logger.warning(f"Pydub load failed: {e}. Falling back to librosa direct...")
        
        # Step 2: Fallback to Librosa Direct
        try:
            y, sr = librosa.load(abs_path, sr=SAMPLE_RATE, mono=True)
            return y, sr
        except Exception as e2:
            logger.error(f"Librosa direct load also failed: {e2}")
            return None, None

def pad_or_trim_audio(y, duration=DURATION, sr=SAMPLE_RATE):
    """
    Pad or trim the audio signal to a fixed length.
    """
    if y is None:
        return None
    
    target_length = int(duration * sr)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    return y



    
def estimate_pitch_confidence(y, sr):
    """
    Estimate pitch confidence to verify if the audio contains tonal speech 
    suitable for F0 analysis.
    """
    # Use a small segment to save time, or resample?
    # PyIN is slow. Let's use autocorrelation or zero-crossing consistency if we can't afford PyIN.
    # For now, let's use a very simplified harmonic check or just trust librosa.pyin on a short crop.
    
    try:
        # Crop to center 3 seconds for speed
        duration = len(y) / sr
        if duration > 3.0:
            start = int((duration - 3.0) / 2 * sr)
            y_crop = y[start : start + int(3.0 * sr)]
        else:
            y_crop = y

        # fmin=50, fmax=300 cover typical human fundamental freq range
        f0, voiced_flag, voiced_probs = librosa.pyin(y_crop, fmin=50, fmax=300, sr=sr)
        
        # Confidence is roughly the mean probability of voiced frames
        confidence = np.nanmean(voiced_probs)
        return confidence if not np.isnan(confidence) else 0.0
    except Exception:
        return 0.0

def detect_speech(y, sr):
    """
    True Speech Presence Detection using Voice Activity Detection (VAD).
    Steps: Compute RMS energy, frame length=30ms, hop=10ms, count frames above threshold.
    Returns: dict with validation result and metrics.
    """
    duration = len(y) / sr
    if duration < 2.0:
        return {
            "status": "INVALID",
            "reason": f"Thời gian quá ngắn ({duration:.1f}s). Cần tối thiểu 2 giây.",
            "speech_ratio": 0.0,
            "energy": 0.0,
            "duration": duration
        }
        
    # Calculate global mean energy
    if y is None or len(y) == 0:
        return {"status": "INVALID", "reason": "Dữ liệu trống"}
    
    # Cast to float64 for intermediate calculation to avoid overflow and satisfy linter
    y_float = y.astype(np.float64)
    rms_total = np.sqrt(np.mean(y_float**2))
    if rms_total < 0.01:
        return {
            "status": "INVALID",
            "reason": "Âm lượng quá nhỏ (gần như im lặng). Vui lòng nói rõ hơn.",
            "speech_ratio": 0.0,
            "energy": float(rms_total),
            "duration": duration
        }

    # Frame-based VAD analysis
    frame_length = int(0.030 * sr) # 30ms
    hop_length = int(0.010 * sr)   # 10ms
    
    # Extract RMS for each frame
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Improved Threshold: Use median instead of max to avoid spike-induced rejection
    energy_threshold = max(0.005, 0.5 * np.median(rms_frames) + 0.1 * np.mean(rms_frames))
    
    speech_frames = np.sum(rms_frames > energy_threshold)
    total_frames = len(rms_frames)
    speech_ratio = float(speech_frames / total_frames) if total_frames > 0 else 0.0

    # Relaxed gate: 10% for noise-reduced signals (was 15%)
    if speech_ratio < 0.10:
        return {
            "status": "INVALID",
            "reason": f"Chỉ thấy {speech_ratio*100:.0f}% là tiếng nói. Cần ít nhất 15% sau lọc nhiễu.",
            "speech_ratio": speech_ratio,
            "energy": float(rms_total),
            "duration": duration
        }
    
    # Anti-Fake / Constant Noise Detection
    # Calculate spectral centroid variance. Constant noise/tones have very low variance.
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_std = np.std(cent)
    
    # Relaxed fan noise detection from 100.0 to 30.0 for home environments
    if cent_std < 30.0:
        return {
            "status": "INVALID",
            "reason": f"Phát hiện âm thanh nền liên tục/tiếng quạt (độ biến thiên tần số thấp: {cent_std:.0f}Hz). Không ghi nhận giọng nói thật.",
            "speech_ratio": speech_ratio,
            "energy": float(rms_total),
            "duration": duration
        }
        
    return {
        "status": "VALID",
        "reason": "Âm thanh hợp lệ",
        "speech_ratio": speech_ratio,
        "energy": float(rms_total),
        "duration": duration
    }

def validate_audio(y, sr):
    """
    Validate TECHNICAL audio quality and SPEECH PRESENCE.
    Returns: (is_valid, list of reasons, metrics dictionary)
    """
    reasons = []
    metrics = {
        "duration": 0.0,
        "snr": 0.0,
        "vad_ratio": 0.0,
        "pitch_confidence": 0.0,
        "clipping_ratio": 0.0,
        "energy": 0.0,
        "tier": "INVALID"
    }

    if y is None or len(y) == 0:
        return False, ["Dữ liệu âm thanh trống"], metrics

    # 1. Scientific Noise Reduction (Adaptive)
    y = reduce_noise_spectral_subtraction(y, sr)
    
    # 2. Run strict speech detection
    speech_result = detect_speech(y, sr)
    metrics["duration"] = float(speech_result["duration"])
    metrics["vad_ratio"] = float(speech_result["speech_ratio"])
    metrics["energy"] = float(speech_result["energy"])

    if speech_result["status"] == "INVALID":
        reasons.append(str(speech_result["reason"]))
        return False, reasons, metrics

    # 2. Advanced Quality Metrics (only if speech detected)
    # SNR Calc based on VAD frames (Optimized for noise-reduced signals)
    # Using 40dB top_db for better sensitivity
    non_silent_intervals = librosa.effects.split(y, top_db=40)
    snr = 0
    if len(non_silent_intervals) > 0:
        speech_parts = np.concatenate([y[s:e] for s, e in non_silent_intervals])
        signal_power = np.mean(speech_parts**2)
        
        mask = np.ones(len(y), dtype=bool)
        for s, e in non_silent_intervals:
             mask[s:e] = False
        noise_parts = y[mask]
        
        if len(noise_parts) > 0:
            noise_power = np.mean(noise_parts**2)
            # Add tiny epsilon to noise power to prevent log errors
            noise_power = max(noise_power, 1e-10) 
            snr = 10 * np.log10(signal_power / noise_power)
            # Clamp SNR to 0-40 range for clinical display
            snr = max(0.0, min(40.0, float(snr)))
        else:
            snr = 35.0 # Extremely clean
    else:
        snr = 0.0
        
    metrics["snr"] = float(snr)
    # Extremely relaxed SNR rejection: 1.0dB after denoising
    if snr < 1.0:
        reasons.append(f"Tín hiệu quá yếu hoặc quá nhiễu (SNR={snr:.1f}dB).")

    # 3. Clipping check
    clipping_count = np.sum(np.abs(y) > 0.98)
    metrics["clipping_ratio"] = float(clipping_count / len(y))
    if metrics["clipping_ratio"] > 0.05:
         reasons.append("Âm thanh bị rè (clipping > 5%). Hạ âm lượng mic.")

    # 4. Clinical Metrics (Pitch Confidence) -> Never used to reject!
    if float(metrics["duration"]) >= 3.0: 
        metrics["pitch_confidence"] = float(estimate_pitch_confidence(y, sr))
    
    # 5. Assign Tier based on TECHNICAL quality 
    if len(reasons) == 0:
        if snr > 15 and float(metrics["vad_ratio"]) > 0.5:
             metrics["tier"] = "HIGH"
        elif snr > 8:
             metrics["tier"] = "MEDIUM"
        else:
             metrics["tier"] = "LOW"
    else:
        metrics["tier"] = "INVALID"

    return (len(reasons) == 0), reasons, metrics

def isolate_loudest_voice(y, sr, top_db=20):
    """
    Isolate the loudest parts of the audio (voice) and trim silence/background.
    top_db: The threshold (in decibels) below reference to consider as silence.
    """
    # 1. Split into non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    
    if len(intervals) == 0:
        return y # Return original if nothing found (validation will catch it)
        
    # 2. Concatenate only the loud parts
    y_loud = np.concatenate([y[start:end] for start, end in intervals])
    return y_loud


def normalize_audio(y):
    """Ensure consistent amplitude for feature extraction."""
    if y is None or len(y) == 0: return y
    abs_max = np.max(np.abs(y))
    if abs_max > 0:
        return y / abs_max
    return y

def save_temp_audio(y, sr, output_path="temp.wav"):
    """
    Save the processed audio to a temporary file.
    """
    sf.write(output_path, y, sr)
