import librosa
import numpy as np

def extract_features(y, sr):
    """
    Extract MFCCs and other features from the audio signal.
    Returns a 1D array of aggregated features (mean/std).
    """
    if y is None or len(y) == 0:
        return None

    # 1. MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Spectral Centroid (Brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # 4. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    
    # Combine all features into a single vector
    features = np.hstack([mfcc_mean, mfcc_std, spectral_centroid_mean, zcr_mean, rms_mean])
    
    return features
