import librosa
import numpy as np
import soundfile as sf
import os

# Constants
SAMPLE_RATE = 16000  # Standard for speech processing
DURATION = 5  # Duration in seconds to analyze (pad/trim)

def load_audio(file_path):
    """
    Load an audio file, resample to 16kHz, and convert to mono.
    Robustly handles formats using pydub/ffmpeg if librosa fails.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 1. Try Loading Directly with Librosa
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y, sr
    except Exception as e:
        logger.warning(f"Librosa direct load failed: {e}. Trying pydub conversion...")
        try:
            # 2. Fallback: Convert using Pydub
            # Import pydub and imageio_ffmpeg to find ffmpeg binary
            from pydub import AudioSegment
            import imageio_ffmpeg
            import io
            
            # Monkey patch pydub's ffmpeg path if needed, or just hope it finds it
            # Better: convert to wav using pydub, export to buffer, then load with librosa
            
            # Ensure ffmpeg uses the local binary as requested
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Found in backend/ffmpeg/bin
            ffmpeg_bin_dir = os.path.join(base_dir, "ffmpeg", "bin")
            
            # Add to PATH so pydub can find ffmpeg and ffprobe automatically
            if os.path.exists(ffmpeg_bin_dir):
                os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.path.expandvars("%PATH%")
            
            # Fallback to imageio-ffmpeg if local binary missing (logic mostly redundant if local exists)
            # if not os.path.exists(os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")):
            #    import imageio_ffmpeg
            #    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            
            audio = AudioSegment.from_file(file_path)
            
            # Resample and mono
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            
            # Export to buffer as WAV
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Load from buffer
            y, sr = librosa.load(wav_buffer, sr=SAMPLE_RATE, mono=True)
            return y, sr
            
        except Exception as e2:
            logger.error(f"Pydub conversion also failed: {e2}")
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

def save_temp_audio(y, sr, output_path="temp.wav"):
    """
    Save the processed audio to a temporary file.
    """
    sf.write(output_path, y, sr)
