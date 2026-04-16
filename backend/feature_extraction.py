import librosa
import numpy as np
import audio_processing

try:
    from scipy.signal import butter, filtfilt, resample
except Exception:
    butter = filtfilt = resample = None

# Total features = 13 MFCC + 13 Delta + 13 Delta-Delta + 15 Clinical = 54
FEATURE_DIM = 54

def compute_f0_stats(y, sr):
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_values = f0[~np.isnan(f0)]
    except Exception:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_values = f0[~np.isnan(f0)] if hasattr(f0, 'mask') or np.isnan(f0).any() else f0

    if len(f0_values) == 0:
        return {'mean_f0': 0.0, 'std_f0': 0.0, 'f0_range': 0.0, 'jitter': 0.0}

    mean_f0 = float(np.mean(f0_values))
    std_f0 = float(np.std(f0_values))
    f0_range = float(np.max(f0_values) - np.min(f0_values))

    jitter = 0.0
    if len(f0_values) > 1:
        periods = 1.0 / (f0_values + 1e-9)
        jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))

    return {'mean_f0': mean_f0, 'std_f0': std_f0, 'f0_range': f0_range, 'jitter': jitter}

def compute_shimmer(y, sr, hop_length=512):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-12))

def compute_hnr(y, sr):
    try:
        autocorr = librosa.autocorrelate(y)
        max_idx = np.argmax(autocorr[100:]) + 100
        periodic_energy = autocorr[max_idx]
        total_energy = autocorr[0]
        if total_energy <= periodic_energy or total_energy < 1e-10: return 0.0
        return float(10 * np.log10(periodic_energy / (total_energy - periodic_energy)))
    except: return 0.0

def extract_features(y, sr):
    """
    Extract 54 Deep Clinical features.
    Standard: 13 MFCC + 13 Delta + 13 Delta2 + 15 Physiological.
    """
    if y is None or len(y) == 0: return None, None

    try:
        import clinical_features
        
        # 1. MFCC & Temporal Dynamics (39)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)

        # 2. Physiological Research Indicators (15)
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms_feat = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms_feat))
        rms_var = float(np.var(rms_feat))

        f0_stats = compute_f0_stats(y, sr)
        shimmer = compute_shimmer(y, sr)
        hnr = compute_hnr(y, sr)

        clinical_data = clinical_features.extract_clinical_features(y, sr)
        clinical = clinical_data.get('vector', {})

        # Assemble Final Vector (Exactly 54)
        extra_list = [
            f0_stats['mean_f0'], f0_stats['std_f0'], f0_stats['jitter'], # 3
            shimmer, hnr, spec_centroid, zcr, rms_mean, # 5
            clinical.get('vot', 0.035), clinical.get('speech_rate', 0.0), # 2
            clinical.get('f1_stability', 0.95), clinical.get('avg_formant_stability', 0.95), # 2
            clinical.get('formant_f1', 500.0), clinical.get('spectral_flux', 0.01), # 2
            float(rms_var / (rms_mean + 1e-9)) # 1 (Tremor Proxy)
        ]
        
        features = np.concatenate([mfcc_mean, delta_mean, delta2_mean, extra_list])

        metrics = {
            'mfcc_mean': mfcc_mean.tolist(),
            **f0_stats,
            'shimmer': shimmer,
            'hnr': hnr,
            'vot': clinical.get('vot', 0.035),
            'f1_stability': clinical.get('f1_stability', 0.95),
            'articulation_rate': clinical.get('speech_rate', 0.0),
            'formants': [clinical.get('formant_f1', 500.0), clinical.get('formant_f2', 1500.0), clinical.get('formant_f3', 2500.0)]
        }

        return features, metrics
    except Exception as e:
        print(f"Deep Analysis Error: {e}")
        return None, None

def extract_features_from_file(file_path):
    y, sr = audio_processing.load_audio(file_path)
    if y is None: return None, None
    return extract_features(y, sr)
